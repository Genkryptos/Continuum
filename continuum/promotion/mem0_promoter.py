"""
continuum/promotion/mem0_promoter.py
====================================
``Mem0Promoter`` — the Mem0-style four-operation promotion engine.

For each candidate :class:`~continuum.extraction.fact_extractor.Fact` and its
nearest existing LTM neighbours, decide one of:

* **ADD**    — genuinely new fact → write it.
* **UPDATE** — augments an existing fact → merge into ``target_id``.
* **DELETE** — contradicts an existing fact → retire ``target_id``.
* **NOOP**   — redundant / no change needed.

The decision is an LLM **function call** (``MEMORY_OP_SCHEMA``), with two
deterministic short-circuits that skip the LLM for the obvious cases:

* best-neighbour cosine ``< add_threshold`` (0.5)  → **ADD**  (definitely new)
* best-neighbour cosine ``> noop_threshold`` (0.97) → **NOOP** (near-duplicate)

Cost control
------------
* Up to ``config.batch_size`` (20) candidates per LLM call, using parallel
  function calling — the model emits one ``memory_operation`` tool-call per
  candidate, aligned by position.
* Small fast model (``config.llm_model`` = ``gpt-4o-mini``); ``temperature``
  0; ``max_tokens`` capped; per-attempt ``timeout``; tenacity retry on
  transient/rate-limit errors.

Audit
-----
Every decision (short-circuited or LLM) is written to ``memory_promotions``
(op, candidate_text, target_id, llm_model, llm_rationale, tokens_in,
tokens_out) via an injectable async ``audit_sink``. Auditing is best-effort:
a sink failure is logged, never raised — it must not block promotion.

Robustness
----------
litellm / psycopg3 are lazy-imported; ``completion_fn`` and ``audit_sink``
are injectable for tests. Any LLM failure degrades a whole chunk to NOOP
(safe default — never invents ADD/DELETE on error), still audited.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from continuum.core.config import PromoterConfig
from continuum.core.types import ScoredItem
from continuum.extraction.fact_extractor import Fact

log = logging.getLogger(__name__)

#: Valid operations (mirrors the ``memory_promotions.op`` CHECK constraint).
OPERATIONS = ("ADD", "UPDATE", "DELETE", "NOOP")

#: LLM function-calling tool schema (exact spec).
MEMORY_OP_SCHEMA: dict[str, Any] = {
    "name": "memory_operation",
    "description": ("Decide how to handle a candidate fact relative to existing memories"),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["ADD", "UPDATE", "DELETE", "NOOP"],
                "description": (
                    "ADD: new fact; UPDATE: augment existing; "
                    "DELETE: contradicts existing; NOOP: no change needed"
                ),
            },
            "target_id": {
                "type": "string",
                "description": ("UUID of existing fact to update/delete (null for ADD/NOOP)"),
            },
            "rationale": {"type": "string", "description": "Why this operation?"},
            "merged_text": {
                "type": "string",
                "description": "For UPDATE: merged fact text",
            },
        },
        "required": ["operation", "rationale"],
    },
}

#: ``audit_sink(records) -> awaitable`` — best-effort persistence.
AuditSink = Callable[[list[dict[str, Any]]], Awaitable[Any]]
#: ``completion_fn(**kwargs) -> awaitable[response]`` — litellm-shaped.
CompletionFn = Callable[..., Awaitable[Any]]
#: ``cosine_of(scored_item) -> float`` — how to read a neighbour's similarity.
CosineOf = Callable[[ScoredItem], float]


@dataclass
class Decision:
    """
    A promotion decision for one candidate fact.

    ``op`` ∈ :data:`OPERATIONS`. ``target_id`` is the existing LTM fact to
    UPDATE/DELETE (``None`` for ADD/NOOP). ``merged_text`` is set only for
    UPDATE. The trailing fields are audit/observability extras.
    """

    op: str
    target_id: uuid.UUID | None
    rationale: str
    merged_text: str | None = None
    candidate_text: str = ""
    short_circuited: bool = False
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def audit_record(self) -> dict[str, Any]:
        """Row for the ``memory_promotions`` audit table."""
        return {
            "op": self.op,
            "candidate_text": self.candidate_text,
            "target_id": str(self.target_id) if self.target_id else None,
            "llm_model": self.model,
            "llm_rationale": self.rationale,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
        }


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError | ValueError):
        return False
    return True


def _default_cosine(si: ScoredItem) -> float:
    """ScoreBreakdown.relevance is documented as cosine similarity."""
    return float(si.scores.relevance)


def _coerce_uuid(value: Any) -> uuid.UUID | None:
    if value in (None, "", "null"):
        return None
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


class Mem0Promoter:
    """
    Mem0 four-operation promotion decider.

    Parameters
    ----------
    config:
        :class:`continuum.core.config.PromoterConfig` (thresholds, model,
        batch size, timeout).
    completion_fn:
        Async litellm-style completion. Injected in tests; default lazily
        wraps ``litellm.acompletion``.
    audit_sink:
        ``async (records) -> None``. ``None`` → decisions are only logged.
        Use :func:`make_postgres_audit_sink` for the ``memory_promotions``
        table.
    cosine_of:
        How to extract a neighbour's cosine similarity from a
        :class:`ScoredItem` (default ``scores.relevance``).
    """

    def __init__(
        self,
        config: PromoterConfig | None = None,
        *,
        completion_fn: CompletionFn | None = None,
        audit_sink: AuditSink | None = None,
        cosine_of: CosineOf | None = None,
    ) -> None:
        self.config = config or PromoterConfig()
        self._completion_fn = completion_fn
        self._audit_sink = audit_sink
        self._cosine_of = cosine_of or _default_cosine
        self._max_attempts = 3
        self._backoff_initial = 0.3
        self._backoff_max = 8.0

    # ── public API ──────────────────────────────────────────────────────────

    async def decide_operation(self, candidate: Fact, neighbors: list[ScoredItem]) -> Decision:
        """Decide the operation for a single *candidate* (audited)."""
        (decision,) = await self.decide_operations_batch([(candidate, neighbors)])
        return decision

    async def decide_operations_batch(
        self,
        items: Sequence[tuple[Fact, list[ScoredItem]]],
    ) -> list[Decision]:
        """
        Decide operations for many candidates with minimal API calls.

        Short-circuited candidates never reach the LLM. The rest are grouped
        into ``config.batch_size`` chunks; each chunk is one LLM call with
        parallel ``memory_operation`` tool-calls. Every decision is audited.
        Returns decisions in input order.
        """
        decisions: list[Decision | None] = [None] * len(items)
        pending: list[int] = []

        for i, (cand, nbrs) in enumerate(items):
            sc = self._short_circuit(cand, nbrs)
            if sc is not None:
                decisions[i] = sc
            else:
                pending.append(i)

        bs = max(1, self.config.batch_size)
        for start in range(0, len(pending), bs):
            chunk = pending[start : start + bs]
            chunk_items = [items[i] for i in chunk]
            chunk_decisions = await self._decide_via_llm(chunk_items)
            for idx, dec in zip(chunk, chunk_decisions, strict=True):
                decisions[idx] = dec

        final = [d for d in decisions if d is not None]
        await self._audit([d.audit_record() for d in final])
        return final

    # ── short-circuit ───────────────────────────────────────────────────────

    def _max_cos(self, neighbors: Sequence[ScoredItem]) -> tuple[float, ScoredItem | None]:
        best: ScoredItem | None = None
        best_c = 0.0
        for si in neighbors:
            c = self._cosine_of(si)
            if best is None or c > best_c:
                best, best_c = si, c
        return best_c, best

    def _short_circuit(self, candidate: Fact, neighbors: list[ScoredItem]) -> Decision | None:
        max_c, best = self._max_cos(neighbors)
        if not neighbors or max_c < self.config.add_threshold:
            return Decision(
                op="ADD",
                target_id=None,
                rationale=(
                    f"No sufficiently-similar memory "
                    f"(max cosine {max_c:.3f} < {self.config.add_threshold}) "
                    f"→ definitely new."
                ),
                candidate_text=candidate.text,
                short_circuited=True,
                model=None,
            )
        if max_c > self.config.noop_threshold and best is not None:
            tid = _coerce_uuid(best.item.id)
            return Decision(
                op="NOOP",
                target_id=tid,
                rationale=(
                    f"Near-duplicate of {best.item.id} "
                    f"(cosine {max_c:.3f} > {self.config.noop_threshold})."
                ),
                merged_text=None,
                candidate_text=candidate.text,
                short_circuited=True,
                model=None,
            )
        return None

    # ── LLM decision (batch, parallel tool calls) ───────────────────────────

    async def _decide_via_llm(self, chunk: list[tuple[Fact, list[ScoredItem]]]) -> list[Decision]:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": self._user_prompt(chunk)},
        ]
        try:
            resp = await self._complete(messages)
            args_list, tin, tout = self._parse_tool_calls(resp)
        except Exception as exc:
            # Degrade the whole chunk to NOOP — never invent ADD/DELETE on
            # error; promotion must not crash the Promoter.
            log.exception("Mem0 LLM decision failed — NOOP for %d candidates", len(chunk))
            return [
                Decision(
                    op="NOOP",
                    target_id=None,
                    rationale=f"LLM decision unavailable: {exc!r}",
                    candidate_text=cand.text,
                    model=self.config.llm_model,
                )
                for cand, _ in chunk
            ]

        out: list[Decision] = []
        for i, (cand, _nbrs) in enumerate(chunk):
            args = args_list[i] if i < len(args_list) else None
            out.append(self._decision_from_args(args, cand, tin, tout))
        return out

    def _decision_from_args(
        self,
        args: dict[str, Any] | None,
        candidate: Fact,
        tokens_in: int,
        tokens_out: int,
    ) -> Decision:
        if not isinstance(args, dict):
            return Decision(
                op="NOOP",
                target_id=None,
                rationale="No tool-call returned for this candidate.",
                candidate_text=candidate.text,
                model=self.config.llm_model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )
        op = str(args.get("operation", "")).upper()
        if op not in OPERATIONS:
            op = "NOOP"
        target = _coerce_uuid(args.get("target_id"))
        if op in ("ADD", "NOOP"):
            target = None  # schema: null for ADD/NOOP
        merged = args.get("merged_text")
        return Decision(
            op=op,
            target_id=target,
            rationale=str(args.get("rationale", "")),
            merged_text=(str(merged) if op == "UPDATE" and merged else None),
            candidate_text=candidate.text,
            model=self.config.llm_model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    # ── LLM call (retry + per-attempt timeout) ──────────────────────────────

    def _retrying(self) -> AsyncRetrying:
        return AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(initial=self._backoff_initial, max=self._backoff_max),
            retry=retry_if_exception(_is_transient),
            reraise=True,
        )

    async def _complete(self, messages: list[dict[str, str]]) -> Any:
        async for attempt in self._retrying():
            with attempt:
                return await asyncio.wait_for(self._call(messages), timeout=self.config.timeout)
        raise RuntimeError("unreachable retry exit")  # pragma: no cover

    async def _call(self, messages: list[dict[str, str]]) -> Any:
        tools = [{"type": "function", "function": MEMORY_OP_SCHEMA}]
        if self._completion_fn is not None:
            return await self._completion_fn(
                model=self.config.llm_model,
                messages=messages,
                tools=tools,
                tool_choice="required",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        try:
            import litellm
        except ImportError as exc:  # pragma: no cover - via completion_fn
            raise ImportError(
                "litellm is required for Mem0Promoter.\nInstall it with:  pip install litellm"
            ) from exc
        return await litellm.acompletion(
            model=self.config.llm_model,
            messages=messages,
            tools=tools,
            tool_choice="required",
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

    @staticmethod
    def _parse_tool_calls(
        resp: Any,
    ) -> tuple[list[dict[str, Any]], int, int]:
        """Extract parsed tool-call argument dicts + token usage."""
        try:
            msg = resp.choices[0].message
            tool_calls = msg.tool_calls
        except (AttributeError, IndexError, KeyError, TypeError):
            try:
                tool_calls = resp["choices"][0]["message"]["tool_calls"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError(f"no tool_calls in response: {resp!r}") from exc

        args: list[dict[str, Any]] = []
        for tc in tool_calls or []:
            try:
                raw = tc.function.arguments
            except AttributeError:
                raw = tc["function"]["arguments"]
            args.append(json.loads(raw) if isinstance(raw, str) else dict(raw))

        tin, tout = 0, 0
        usage = getattr(resp, "usage", None) or (
            resp.get("usage") if isinstance(resp, dict) else None
        )
        if usage is not None:
            tin = int(
                getattr(usage, "prompt_tokens", 0)
                or (usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0)
            )
            tout = int(
                getattr(usage, "completion_tokens", 0)
                or (usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0)
            )
        return args, tin, tout

    # ── prompt ──────────────────────────────────────────────────────────────

    @staticmethod
    def _user_prompt(chunk: list[tuple[Fact, list[ScoredItem]]]) -> str:
        lines: list[str] = []
        for i, (cand, nbrs) in enumerate(chunk):
            lines.append(f"### Candidate {i}")
            lines.append(f'Fact: "{cand.text}"')
            if nbrs:
                lines.append("Top similar existing facts:")
                for si in nbrs[:10]:
                    lines.append(
                        f"  - id={si.item.id} "
                        f"cos={si.scores.relevance:.3f} "
                        f'text="{si.item.content}"'
                    )
            else:
                lines.append("Top similar existing facts: (none)")
            lines.append("")
        lines.append(
            f"Call `memory_operation` exactly once per candidate, in order "
            f"(candidates 0..{len(chunk) - 1})."
        )
        return "\n".join(lines)

    # ── audit ───────────────────────────────────────────────────────────────

    async def _audit(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        if self._audit_sink is None:
            for r in records:
                log.debug("promotion decision (no audit sink): %s", r)
            return
        try:
            await self._audit_sink(records)
        except Exception:  # audit is best-effort — never block promotion
            log.exception("audit sink failed — decisions NOT persisted")


_SYSTEM_PROMPT = """\
You maintain a long-term memory store. For EACH candidate fact, compare it to
the listed existing facts and choose exactly one operation by calling the
`memory_operation` function:

- ADD    : the candidate is genuinely new information. target_id = null.
- UPDATE : the candidate augments/refines ONE existing fact. Set target_id to
           that fact's id and provide merged_text combining both.
- DELETE : the candidate contradicts ONE existing fact (the old one is now
           wrong). Set target_id to the contradicted fact's id.
- NOOP   : the candidate is redundant / already known. target_id = null.

Call `memory_operation` exactly ONCE per candidate, in the SAME ORDER the
candidates are listed (candidate 0 first). Always include a concise
rationale.
"""


# ============================================================================
# Default audit sink → memory_promotions
# ============================================================================


def make_postgres_audit_sink(
    *,
    dsn: str | None = None,
    conn_factory: Any | None = None,
    table: str = "memory_promotions",
) -> AuditSink:
    """
    Build an :data:`AuditSink` that batch-inserts decisions into
    ``memory_promotions`` (migration 001 schema).

    psycopg3 is imported lazily; pass ``conn_factory`` to unit-test without
    psycopg / PostgreSQL.
    """
    if conn_factory is None and dsn is None:
        raise ValueError("Provide either dsn or conn_factory.")

    async def _sink(records: list[dict[str, Any]]) -> None:
        if not records:
            return
        sql = (
            f"INSERT INTO {table} "
            f"(op, candidate_text, target_id, llm_model, llm_rationale, "
            f" tokens_in, tokens_out) "
            f"VALUES (%(op)s, %(candidate_text)s, %(target_id)s, "
            f"        %(llm_model)s, %(llm_rationale)s, "
            f"        %(tokens_in)s, %(tokens_out)s)"
        )
        if conn_factory is not None:
            async with conn_factory() as conn:
                for r in records:
                    await conn.execute(sql, r)
            return
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - via conn_factory
            raise ImportError(
                "psycopg3 is required for the Postgres audit sink.\n"
                "Install it with:  pip install 'psycopg[binary]>=3.1'"
            ) from exc
        # Guaranteed by the dsn-or-conn_factory check above: this branch is
        # only reached when conn_factory is None, so dsn is set.
        assert dsn is not None
        conn = await psycopg.AsyncConnection.connect(dsn, autocommit=True, row_factory=dict_row)
        try:
            for r in records:
                await conn.execute(sql, r)
        finally:
            await conn.close()

    return _sink


__all__ = [
    "Mem0Promoter",
    "Decision",
    "MEMORY_OP_SCHEMA",
    "OPERATIONS",
    "make_postgres_audit_sink",
]
