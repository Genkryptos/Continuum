"""
evals/longmemeval/wiki.py
=========================
Wiki-style synthesis layer for LongMemEval — adapted from Karpathy's
"LLM Wiki" pattern (https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

The problem (proven across 7 LongMemEval runs)
-----------------------------------------------
Every single-shot / decompose run plateaus around 30 %. The failure
breakdown of the decompose run was decisive: **319 of 354 failures
(90 %) had the ground-truth session retrieved and still answered
wrong.** The model can't compose 3-5 facts at query time from raw
turns, even when long-context gives it 100 % recall (32 % ceiling
on long-context confirmed this independently).

The fix
-------
**Don't re-compose at query time. Pre-compose at ingest time.**

For each haystack session, the LLM writes once:

* a **summary** (2-3 sentences of concrete facts / decisions / events)
* an **atomic-fact list** (short self-contained user-facing facts)

Retrieval then queries this dense synthesis layer — summaries and
facts — instead of raw turns. The cross-session reasoning the
single-pass model fails *at query time* has already been compressed
into a small set of indexed sentences *at ingest time*.

This is the eval-time equivalent of Continuum's MTM (summaries) and
LTM (facts) tiers, but kept in-memory so the benchmark doesn't need
Postgres / pgvector / GLiNER. It is also exactly the pattern the
LongMemEval paper's strong systems use under the name "index
expansion" (session-summ + session-userfact).

Cost: one summary + one fact-extraction call per haystack session.
A LongMemEval-S row has ~40 sessions → ~80 cheap calls (~$0.008 on
gpt-4o-mini). Crucially these are **cached to disk** by session
content-hash, so re-runs (and shared sessions across questions) are
free after the first pass.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import hashlib
import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from continuum.core.types import (
    ContextBundle,
    MemoryItem,
    MemoryTier,
    Query,
    TokenBudget,
)
from continuum.optimizer.base import estimate_tokens_text

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts — module constants so prompt caching can hit.
# ---------------------------------------------------------------------------


#: Bump when WIKI_PAGE_PROMPT changes — the cache key incorporates this so
#: old cached pages from an earlier prompt version are automatically
#: bypassed (no manual cache wipe needed).
_PROMPT_VERSION = "v2"


#: Combined summary + facts in ONE LLM call. Halves both wall-clock
#: (vs separate calls) and per-row LLM cost, which matters a lot
#: because ingest is the dominant cost in wiki mode.
#:
#: v2 (post-smoke diagnosis) — addresses two concrete prompt failures
#: observed on the v1 smoke:
#:
#:  * detail loss: v1 dropped "Business Administration" because it
#:    treated incidental personal details as less important than the
#:    session's main topic (task-management apps). Rule (A) forces
#:    the LLM to capture every concrete personal detail.
#:
#:  * over-atomization: v1 split "redeemed a $5 coupon on coffee
#:    creamer at Target" into "shops at Target" + "redeemed a coupon",
#:    losing the connection. Rule (B) requires each fact to keep all
#:    its attributes (what/where/when/how much) together.
WIKI_PAGE_PROMPT = (
    "Compress this chat session into a wiki entry.\n\n"
    "Output format (use these EXACT markers, no preamble, no other "
    "content):\n"
    "### SUMMARY\n"
    "<2-3 sentences on the session's main topics + key user decisions "
    "or events. Skip greetings.>\n"
    "### FACTS\n"
    "<one fact per line, no numbering or bullets, 0-12 facts>\n\n"
    "Fact rules — these matter for retrieval and answering downstream:\n"
    "(A) Preserve EVERY concrete personal detail the user states or "
    "implies about themselves, even when it's incidental to the "
    "session's main topic. Include: education (degree, school), "
    "employer & job title, hometown / current city, names of family / "
    "pets / partners, brands & stores they use, exact dollar amounts, "
    "exact dates, specific numbers, medical details, hobbies, "
    "vehicles. If the user mentions in passing \"I just graduated with "
    "a Business Admin degree,\" that fact goes in regardless of what "
    "the rest of the session was about.\n"
    "(B) Each fact must contain ALL relevant attributes of one event "
    "or thing together — what, where, when, how much, with whom — in "
    "a single sentence. Example: write \"The user redeemed a $5 "
    "coupon on coffee creamer at Target last Sunday.\" NOT split into "
    "multiple facts (\"The user shops at Target\" + \"The user "
    "redeemed a $5 coupon on coffee creamer\").\n"
    "(C) Each fact must stand alone — no pronouns referring to other "
    "facts.\n"
    "(D) Skip greetings, small talk, and assistant-only content.\n\n"
    "Session:\n{session_text}\n\nWiki entry:"
)


_NUMBER_PREFIX = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")
_SUMMARY_HDR = re.compile(r"^\s*#{0,3}\s*summary\s*:?\s*$", re.IGNORECASE)
_FACTS_HDR = re.compile(r"^\s*#{0,3}\s*facts\s*:?\s*$", re.IGNORECASE)
#: Qwen3 / DeepSeek-R1 / o1-style reasoning models emit a
#: <think>…</think> block before the visible reply. Strip it so the
#: parser doesn't treat the chain-of-thought as the summary.
_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
#: Hard cap on facts per session — keeps token budget bounded.
_MAX_FACTS = 12


# ---------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------


def parse_facts(reply: str) -> list[str]:
    """
    Parse a bullet/line list of facts into a clean list.

    Strips numbering / bullets, drops blank lines, de-dupes
    case-insensitively, caps at :data:`_MAX_FACTS`. Returns an empty
    list if the reply is empty / unparseable — callers should treat
    zero facts as a valid outcome (some sessions are pure chit-chat).
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in (reply or "").splitlines():
        line = _NUMBER_PREFIX.sub("", raw).strip()
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
        if len(out) >= _MAX_FACTS:
            break
    return out


def _strip_think_blocks(reply: str) -> str:
    """Remove ``<think>…</think>`` blocks from reasoning-model replies."""
    return _THINK_BLOCK.sub("", reply or "").strip()


def parse_wiki_page(reply: str) -> tuple[str, list[str]]:
    """
    Split a combined-prompt reply into ``(summary, facts)``.

    The expected reply shape (per :data:`WIKI_PAGE_PROMPT`) is:

    .. code-block::

        ### SUMMARY
        <one or more lines>
        ### FACTS
        <0..10 lines>

    Header-matching is forgiving — case-insensitive, optional `#`
    prefixes and trailing colon, so reasonable model deviations
    still parse. If the markers are absent entirely we treat the
    whole reply as the summary and emit no facts.
    """
    reply = _strip_think_blocks(reply)
    if not reply:
        return "", []
    summary_lines: list[str] = []
    facts_buf: list[str] = []
    section = "summary"
    saw_marker = False
    for raw in reply.splitlines():
        if _SUMMARY_HDR.match(raw):
            section = "summary"
            saw_marker = True
            continue
        if _FACTS_HDR.match(raw):
            section = "facts"
            saw_marker = True
            continue
        if section == "summary":
            summary_lines.append(raw)
        else:
            facts_buf.append(raw)
    if not saw_marker:
        # No markers found — model didn't follow format. Whole reply
        # is treated as the summary; no facts extracted.
        return " ".join(line.strip() for line in summary_lines).strip(), []
    summary = " ".join(line.strip() for line in summary_lines if line.strip())
    facts = parse_facts("\n".join(facts_buf))
    return summary.strip(), facts


def _normalize_safely(raw: np.ndarray) -> np.ndarray:
    """
    L2-normalize ``raw`` (shape ``(N, D)``) into unit rows, with two
    robustness fixes for the cosine-search use case:

    * any non-finite entry coming out of the embedder is replaced with
      zero before normalization (sentence-transformers occasionally
      emits NaN/inf for very short or degenerate inputs);
    * rows whose norm is effectively zero are left as zero vectors
      rather than divided by an epsilon (which overflows on any
      non-trivial residual and poisons every later matmul).

    The output is always finite, so downstream ``A @ q.T`` cannot
    propagate NaN through the ranking.
    """
    if not np.all(np.isfinite(raw)):
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    safe = np.where(norms > 1e-9, norms, 1.0)
    vecs = raw / safe
    return np.where(norms > 1e-9, vecs, 0.0)


def format_session(messages: Iterable[dict[str, Any]]) -> str:
    """Render a session's messages into a single prompt-ready string."""
    lines: list[str] = []
    for m in messages:
        role = str(m.get("role", "user")).strip() or "user"
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WikiPage:
    """One haystack session's compiled wiki entry."""

    session_id: str
    session_date: str               # ISO date, "" if unparseable
    summary: str
    facts: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.summary and not self.facts


# ---------------------------------------------------------------------------
# Disk cache — content-hash keyed, so re-runs and shared sessions are free.
# ---------------------------------------------------------------------------


def _cache_key(session_id: str, content: str) -> str:
    """
    Cache key: prompt version + session id + first 4 KB of content.

    Including :data:`_PROMPT_VERSION` means bumping the prompt
    automatically invalidates every old cache entry — re-runs after
    a prompt change recompute from scratch instead of silently
    serving stale compressions.
    """
    payload = f"{_PROMPT_VERSION}\n{session_id}\n{content[:4096]}"
    return hashlib.sha1(payload.encode()).hexdigest()


class WikiDiskCache:
    """Tiny content-addressed cache for wiki pages."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict[str, Any] | None:
        p = self.root / f"{key}.json"
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            return None
        return None

    def put(self, key: str, value: dict[str, Any]) -> None:
        p = self.root / f"{key}.json"
        with contextlib.suppress(OSError):
            p.write_text(json.dumps(value, ensure_ascii=False))


# ---------------------------------------------------------------------------
# WikiBuilder — one LLM call pair per session, cached, bounded-concurrency.
# ---------------------------------------------------------------------------


class WikiBuilder:
    """
    Convert one session's messages into a :class:`WikiPage`.

    Two LLM calls per session: summary + facts. Both are bounded by
    a shared :class:`asyncio.Semaphore` so a single eval row's ~40
    sessions don't all hit the provider at once and trip rate limits.

    Disk-cached by content hash — second invocation on the same
    session (or the same session reused across questions) is free.
    """

    def __init__(
        self,
        *,
        llm: Any,
        page_max_tokens: int = 1024,
        cache: WikiDiskCache | None = None,
        concurrency: int = 6,
    ) -> None:
        self.llm = llm
        self.page_max_tokens = page_max_tokens
        self.cache = cache
        self._sem = asyncio.Semaphore(concurrency)

    async def build_page(
        self,
        *,
        session_id: str,
        session_date: str,
        session_text: str,
    ) -> WikiPage:
        """Return a :class:`WikiPage` for one session. Cached on content."""
        if not session_text.strip():
            return WikiPage(
                session_id=session_id, session_date=session_date,
                summary="", facts=[],
            )
        key = _cache_key(session_id, session_text)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return WikiPage(
                    session_id=session_id,
                    session_date=session_date,
                    summary=str(hit.get("summary", "")),
                    facts=[str(f) for f in hit.get("facts", [])][:_MAX_FACTS],
                )

        async with self._sem:
            reply = await self._safe_complete(
                WIKI_PAGE_PROMPT.format(session_text=session_text),
                self.page_max_tokens,
            )

        summary, facts = parse_wiki_page(reply)
        # NEVER cache an empty compression. An empty (summary AND facts
        # both blank) page almost always means the LLM call failed
        # (connection error, thinking-budget exhaustion, timeout, …) —
        # not that the session was genuinely empty. Persisting it
        # would poison every future re-run; better to retry next time.
        if self.cache is not None and (summary or facts):
            self.cache.put(key, {"summary": summary, "facts": facts})
        return WikiPage(
            session_id=session_id, session_date=session_date,
            summary=summary, facts=facts,
        )

    async def _safe_complete(self, prompt: str, max_tokens: int) -> str:
        try:
            return await self.llm.complete(
                prompt=prompt, max_tokens=max_tokens,
            )
        except Exception:
            log.exception("wiki LLM call failed — returning empty")
            return ""


# ---------------------------------------------------------------------------
# WikiStore — in-memory index of wiki entries (summary + per-fact rows).
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    """One indexable row — either a session summary or a single fact."""

    text: str
    kind: str              # "summary" | "fact"
    session_id: str
    session_date: str


class WikiStore:
    """
    In-memory store of wiki entries, indexed for cosine retrieval.

    Each :class:`WikiPage` contributes one summary row plus N fact
    rows. Embeddings are L2-normalized so retrieval is a single
    matrix-vector dot product.
    """

    def __init__(self, embedder: Any) -> None:
        self.embedder = embedder
        self.pages: list[WikiPage] = []
        self._entries: list[_Entry] = []
        self._embeddings: np.ndarray | None = None

    @property
    def n_entries(self) -> int:
        return len(self._entries)

    def add_pages(self, pages: Iterable[WikiPage]) -> None:
        """Append wiki pages and (re-)embed their entries."""
        new_pages = [p for p in pages if not p.is_empty()]
        if not new_pages:
            return
        self.pages.extend(new_pages)
        new_entries: list[_Entry] = []
        for p in new_pages:
            if p.summary:
                new_entries.append(_Entry(
                    text=p.summary, kind="summary",
                    session_id=p.session_id, session_date=p.session_date,
                ))
            for fact in p.facts:
                new_entries.append(_Entry(
                    text=fact, kind="fact",
                    session_id=p.session_id, session_date=p.session_date,
                ))
        if not new_entries:
            return
        texts = [e.text for e in new_entries]
        vecs = _normalize_safely(
            np.asarray(self.embedder.encode(texts), dtype=np.float32)
        )
        self._entries.extend(new_entries)
        self._embeddings = (
            vecs if self._embeddings is None
            else np.vstack([self._embeddings, vecs])
        )

    def search(self, query: str, *, top_k: int) -> list[_Entry]:
        """Cosine-rank entries against *query*; return top-k highest."""
        if self._embeddings is None or not self._entries:
            return []
        q = _normalize_safely(
            np.asarray(self.embedder.encode([query]), dtype=np.float32)
        )
        # The post-matmul np.nan_to_num below already guards the ranking
        # against any NaN/inf — silence the matmul-site RuntimeWarnings
        # so they don't clutter the run log on every query.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = (self._embeddings @ q.T).ravel()
        # Defensive: any residual NaN/inf gets ranked last, not first.
        if not np.all(np.isfinite(sims)):
            sims = np.nan_to_num(sims, nan=-1.0, posinf=-1.0, neginf=-1.0)
        k = min(top_k, len(self._entries))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [self._entries[int(i)] for i in idx]


# ---------------------------------------------------------------------------
# WikiRetriever — drop-in for STMSemanticRetriever, queries the wiki layer.
# ---------------------------------------------------------------------------


class WikiRetriever:
    """
    Same ``async retrieve(query, budget) -> ContextBundle`` surface as
    :class:`STMSemanticRetriever`, but pulls from the wiki synthesis
    layer instead of raw turns.

    Returned :class:`MemoryItem`s are tagged with tier ``MTM`` for
    summaries and ``LTM`` for facts, so the adapter's prompt
    formatter sections them naturally ("Project summary:" /
    "Long-term knowledge:").
    """

    def __init__(
        self,
        *,
        store: WikiStore,
        top_k: int = 12,
        session_id: str = "default",
    ) -> None:
        self.store = store
        self.top_k = top_k
        self.session_id = session_id

    async def retrieve(
        self, query: Query, budget: TokenBudget
    ) -> ContextBundle:
        hits = self.store.search(query.text, top_k=self.top_k)
        items: list[MemoryItem] = []
        messages: list[dict[str, str]] = []
        for h in hits:
            tier = MemoryTier.MTM if h.kind == "summary" else MemoryTier.LTM
            md: dict[str, Any] = {
                "role": "user",
                "session_id": h.session_id,
                "kind": h.kind,
            }
            it = MemoryItem(content=h.text, tier=tier, metadata=md)
            if h.session_date:
                with contextlib.suppress(ValueError):
                    it.created_at = dt.datetime.fromisoformat(h.session_date)
            items.append(it)
            messages.append({"role": "user", "content": h.text})

        tokens_used = sum(estimate_tokens_text(it.content) for it in items)
        breakdown = {"stm": 0, "mtm": 0, "ltm": 0}
        for it in items:
            breakdown[it.tier.value] = (
                breakdown.get(it.tier.value, 0)
                + estimate_tokens_text(it.content)
            )
        return ContextBundle(
            items=items,
            messages=messages,
            tokens_used=tokens_used,
            budget=budget,
            tier_breakdown=breakdown,
            debug_info={
                "retrieval_mode": "wiki",
                "n_pages": len(self.store.pages),
                "n_entries": self.store.n_entries,
                "n_hits": len(items),
                "hit_kinds": [h.kind for h in hits],
            },
        )


__all__ = [
    "WIKI_PAGE_PROMPT",
    "WikiBuilder",
    "WikiDiskCache",
    "WikiPage",
    "WikiRetriever",
    "WikiStore",
    "format_session",
    "parse_facts",
    "parse_wiki_page",
]
