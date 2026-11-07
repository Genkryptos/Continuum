"""
evals/longmemeval/ingest_cache.py
=================================
Content-addressed cache for the expensive half of ``--use-ltm`` ingest.

Why
---
``ContinuumLTMHaystackStore.finalize()`` runs one LLM fact-extraction call per
*session block* — ~53 per LongMemEval row — then one promoter pass over the
resulting facts. Measured: ~511 s and ~53 calls for a single row. The A5
experiment needs 78 knowledge-update rows × 2 arms, which is ~8 300 calls and
roughly 22 hours of wall time. That is not a thing anyone iterates on.

The saving observation is that **the two arms share everything up to the last
step**. Extraction is identical, the promoter's decisions are identical, and
only ``_execute_decisions`` differs (``invalidate`` vs closing ``valid_to``).
So the ingest can be computed once per row and replayed:

    extract facts ─┐
                   ├─► CACHED (deterministic given haystack + config)
    decide ops ────┘
                   └─► execute ──► differs per arm, cheap, no LLM

First pass costs the same as today; every re-run — the second arm, a repeat, an
ablation that only changes execution — is free.

Correctness
-----------
The key covers the haystack content *and* the model config that produced the
facts, so changing the extractor model, token budget, or promoter model misses
the cache rather than silently replaying stale facts. A cached run is expected
to be identical to an uncached one; ``--no-ingest-cache`` forces recomputation.

Decisions are stored by **fact index**, not by ``target_id``: the LTM assigns
fresh UUIDs on every run, so indices are the only stable reference. The eval
builds a fresh store per row, hence every decision target is one of this row's
own facts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: Bump when the payload shape **or the semantics that produced it** change —
#: old entries then miss instead of replaying decisions that would no longer be
#: made.
#:
#: **This is easy to forget and the failure is silent**: the fingerprint covers
#: the model and token budget, NOT the code that turns facts into decisions. A
#: logic change with an unbumped version replays pre-fix decisions and the run
#: looks like it simply reproduced the old result. That has already happened
#: once (the cosine fix below), costing a full debugging cycle. If you change
#: anything upstream of ``decisions``, bump this.
#:
#: * v2 — decide-then-upsert per session, so candidates no longer see their own
#:   just-inserted row as a neighbour.
#: * v3 — neighbour ``relevance`` rescored from RRF to true cosine, so
#:   candidates stop short-circuiting to ADD below ``add_threshold``.
CACHE_VERSION = 3

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".ingest_cache"


def config_fingerprint(
    *,
    fact_model: str,
    fact_max_tokens: int,
    fact_temperature: float,
    promoter_model: str,
    promoter_path: str,
) -> str:
    """Stable digest of every setting that can change the cached output."""
    blob = json.dumps(
        {
            "v": CACHE_VERSION,
            "fact_model": fact_model,
            "fact_max_tokens": fact_max_tokens,
            "fact_temperature": fact_temperature,
            "promoter_model": promoter_model,
            "promoter_path": promoter_path,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def cache_key(session_texts: dict[str, str], fingerprint: str) -> str:
    """
    Content address for one row's ingest.

    *session_texts* maps session id → the block text handed to the extractor.
    Both the ids and the text participate, so a haystack that differs only in
    session ordering still keys differently (order affects fact order, which
    affects decision indices).
    """
    h = hashlib.sha256()
    h.update(fingerprint.encode())
    for sid, text in session_texts.items():
        h.update(b"\x00")
        h.update(sid.encode())
        h.update(b"\x01")
        h.update(text.encode())
    return h.hexdigest()


class IngestCache:
    """One JSON file per row keyed by content hash. No eviction — entries are
    small (facts + decisions) and keyed by content, so staleness is impossible."""

    def __init__(self, cache_dir: Path | str | None = None, *, enabled: bool = True) -> None:
        self.dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def _path(self, key: str) -> Path:
        return self.dir / f"{key[:2]}" / f"{key}.json"

    def load(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        p = self._path(key)
        if not p.is_file():
            self.misses += 1
            return None
        try:
            payload = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            log.warning("ingest cache: unreadable entry %s — treating as miss", p)
            self.misses += 1
            return None
        if payload.get("version") != CACHE_VERSION:
            self.misses += 1
            return None
        self.hits += 1
        return payload

    def save(self, key: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        p = self._path(key)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename so a crash mid-write cannot leave a torn entry
            # that a later run would read as valid.
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps({**payload, "version": CACHE_VERSION}, default=str))
            tmp.replace(p)
            self.writes += 1
        except OSError as exc:
            log.warning("ingest cache: could not write %s (%s) — continuing", p, exc)

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
        }


# ── (de)serialisation ──────────────────────────────────────────────────────


def facts_to_payload(facts_with_session: list[tuple[str, Any]]) -> list[dict[str, Any]]:
    """Serialise ``(session_id, Fact)`` pairs, preserving order."""
    out = []
    for sid, f in facts_with_session:
        out.append(
            {
                "session_id": sid,
                "text": f.text,
                "confidence": float(f.confidence),
                "entities_mentioned": list(f.entities_mentioned or []),
                "source_block_id": str(f.source_block_id) if f.source_block_id else None,
                "category": f.category,
                "metadata": dict(f.metadata or {}),
            }
        )
    return out


def payload_to_facts(rows: list[dict[str, Any]], fact_cls: type) -> list[tuple[str, Any]]:
    """Rebuild ``(session_id, Fact)`` pairs in cached order."""
    out = []
    for r in rows:
        sbid = r.get("source_block_id")
        out.append(
            (
                r["session_id"],
                fact_cls(
                    text=r["text"],
                    confidence=r["confidence"],
                    entities_mentioned=list(r.get("entities_mentioned") or []),
                    source_block_id=uuid.UUID(sbid) if sbid else uuid.uuid4(),
                    category=r.get("category"),
                    metadata=dict(r.get("metadata") or {}),
                ),
            )
        )
    return out


def decisions_to_payload(
    decisions: list[Any], id_to_index: dict[str, int]
) -> list[dict[str, Any]] | None:
    """
    Serialise decisions, rewriting ``target_id`` UUIDs to fact indices.

    Returns ``None`` — meaning *do not cache this row* — when any UPDATE or
    DELETE targets a row that is not one of this ingest's own facts. Such a
    target has no stable identity across runs (the LTM mints fresh UUIDs), so
    it cannot be replayed faithfully.

    Refusing to cache is deliberate: the alternative is degrading the decision
    to NOOP on replay, which would silently turn a supersession into a
    no-op and make the cached run disagree with the uncached one. A cache that
    changes results is worse than no cache. The cost is only that such rows
    stay expensive.
    """
    out = []
    for d in decisions:
        tid = getattr(d, "target_id", None)
        op = str(getattr(d, "op", "NOOP")).upper()
        idx = id_to_index.get(str(tid)) if tid is not None else None
        if op in ("UPDATE", "DELETE") and idx is None:
            log.debug("ingest cache: unresolvable %s target %s — row not cached", op, tid)
            return None
        out.append(
            {
                "op": op,
                "target_index": idx,
                "merged_text": getattr(d, "merged_text", None),
                "rationale": getattr(d, "rationale", None),
            }
        )
    return out


def payload_to_decisions(
    rows: list[dict[str, Any]], index_to_id: list[Any], decision_cls: type
) -> list[Any]:
    """Rebuild decisions, mapping fact indices back to this run's UUIDs."""
    out = []
    for r in rows:
        idx = r.get("target_index")
        target = index_to_id[idx] if idx is not None and 0 <= idx < len(index_to_id) else None
        op = r["op"]
        if op in ("UPDATE", "DELETE") and target is None:
            # Unreachable: decisions_to_payload refuses to cache these. Guard
            # anyway so a hand-edited or format-skewed entry cannot silently
            # retarget a supersession at the wrong row.
            raise ValueError(
                f"ingest cache: {op} decision has no resolvable target — entry is corrupt"
            )
        out.append(
            decision_cls(
                op=op,
                target_id=target,
                rationale=r.get("rationale") or "replayed from ingest cache",
                merged_text=r.get("merged_text"),
            )
        )
    return out
