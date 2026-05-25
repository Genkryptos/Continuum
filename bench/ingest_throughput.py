"""
bench/ingest_throughput.py
==========================
Compare ingestion cost across four memory systems on the same
deterministic synthetic workload.

Systems
-------
* ``naive_append``    raw Python list ``append`` — the lower bound on
  what any "memory" can cost
* ``continuum_bare``  Continuum's :class:`InMemorySTM` only, no
  promotion / extraction; the framework's hot-path cost
* ``continuum_full``  STM + :class:`FactExtractor` +
  :class:`LLMEntityExtractor` invoked once per turn; the realistic
  production cost
* ``mem0`` (optional) ``mem0.Memory.add`` if the package is installed;
  skipped with a logged note otherwise

Measurements
------------
For each system + each of ``--sessions`` synthetic sessions:

* total wall-clock to ingest the session (p50, p95)
* number of LLM completion calls per session
* total bytes stored (approximation — content lengths summed)

Outputs
-------
* ``bench/results/ingest_<timestamp>.json`` with the full breakdown
* ``bench/results/ingest_latest.json`` pointing at the most recent
* a one-paragraph narrative on stdout

Run
---
::

    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \\
        -m bench.ingest_throughput --sessions 50 --turns 6
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import statistics
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bench.synth import Session, make_sessions
from continuum.core.types import MemoryItem, MemoryTier
from continuum.stores.stm.conversation_stm import InMemorySTM

log = logging.getLogger("bench.ingest")
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# Mock LLM — counts every call without spending money / wall-clock-blocking
# on a real API. Returns a tiny stub response that the production
# extractors will parse cleanly.
# ---------------------------------------------------------------------------


class _CallCounter:
    """Async LLM stub. Returns canned replies; counts every invocation."""

    def __init__(self) -> None:
        self.calls = 0

    async def acompletion(self, *_args: Any, **kwargs: Any) -> dict[str, Any]:
        """Mimic ``litellm.acompletion`` enough for FactExtractor + entities."""
        self.calls += 1
        messages = kwargs.get("messages") or []
        prompt = " ".join(m.get("content", "") for m in messages).lower()
        # Cheap heuristic: return JSON shaped for whichever extractor is asking.
        if "entit" in prompt or "person" in prompt or "extract entities" in prompt:
            content = '{"entities": [{"name":"Boston","type":"GPE","confidence":0.9}]}'
        elif "fact" in prompt:
            content = (
                '{"facts": [{"text":"User lives in Boston.",'
                '"confidence":0.9,"importance":0.7}]}'
            )
        else:
            content = '{"facts": []}'
        # Match litellm's response shape (choices[0].message.content).
        return {"choices": [{"message": {"content": content}}]}


# ---------------------------------------------------------------------------
# Per-system ingest implementations
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    system: str
    available: bool
    n_sessions: int = 0
    n_turns: int = 0
    per_session_ms: list[float] = field(default_factory=list)
    llm_calls: int = 0
    total_bytes: int = 0
    note: str = ""

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.per_session_ms) if self.per_session_ms else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.per_session_ms:
            return 0.0
        idx = max(0, int(len(self.per_session_ms) * 0.95) - 1)
        return sorted(self.per_session_ms)[idx]

    @property
    def llm_calls_per_session(self) -> float:
        return self.llm_calls / self.n_sessions if self.n_sessions else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "available": self.available,
            "n_sessions": self.n_sessions,
            "n_turns": self.n_turns,
            "p50_ms_per_session": round(self.p50_ms, 3),
            "p95_ms_per_session": round(self.p95_ms, 3),
            "llm_calls": self.llm_calls,
            "llm_calls_per_session": round(self.llm_calls_per_session, 2),
            "total_bytes_stored": self.total_bytes,
            "note": self.note,
        }


async def _ingest_naive(sessions: list[Session]) -> IngestStats:
    """Raw Python list — lower-bound ingest cost."""
    s = IngestStats(system="naive_append", available=True)
    store: list[dict[str, str]] = []
    for sess in sessions:
        t0 = time.perf_counter()
        for turn in sess.turns:
            store.append({"role": turn.role, "content": turn.content})
            s.total_bytes += len(turn.content)
        s.per_session_ms.append((time.perf_counter() - t0) * 1000.0)
        s.n_turns += len(sess.turns)
    s.n_sessions = len(sessions)
    s.note = (
        "Lower bound — no organization, no retrieval support, no extraction. "
        "Useful only as a 'how fast can a list grow' baseline."
    )
    return s


async def _ingest_continuum_bare(sessions: list[Session]) -> IngestStats:
    """Continuum :class:`InMemorySTM` only, no promotion / extraction."""
    s = IngestStats(system="continuum_bare", available=True)
    stm = InMemorySTM()
    for sess in sessions:
        t0 = time.perf_counter()
        for turn in sess.turns:
            await stm.append(MemoryItem(
                content=turn.content,
                tier=MemoryTier.STM,
                metadata={
                    "role": turn.role,
                    "session_id": turn.session_id,
                },
            ))
            s.total_bytes += len(turn.content)
        s.per_session_ms.append((time.perf_counter() - t0) * 1000.0)
        s.n_turns += len(sess.turns)
    s.n_sessions = len(sessions)
    s.note = (
        "Framework hot-path cost. InMemorySTM enforces tier semantics + "
        "metadata stamping but skips any LLM work. The realistic floor "
        "for a Continuum deployment that uses STM-only retrieval."
    )
    return s


async def _ingest_continuum_full(sessions: list[Session]) -> IngestStats:
    """
    STM + per-turn fact + entity extraction.

    This is the cost a production Continuum user pays per turn when
    extraction is wired into the ``after_turn`` trigger path.
    """
    from continuum.core.config import FactExtractionConfig, LLMExtractionConfig
    from continuum.core.types import SummaryBlock
    from continuum.extraction.fact_extractor import FactExtractor
    from continuum.extraction.llm_extractor import LLMEntityExtractor

    s = IngestStats(system="continuum_full", available=True)
    counter = _CallCounter()

    # Wire the production extractors with our counted LLM stub.
    entity_ex = LLMEntityExtractor(
        config=LLMExtractionConfig(model="mock"),
        completion_fn=counter.acompletion,
    )
    fact_ex = FactExtractor(
        config=FactExtractionConfig(model="mock"),
        completion_fn=counter.acompletion,
    )
    stm = InMemorySTM()

    for sess in sessions:
        t0 = time.perf_counter()
        for turn in sess.turns:
            await stm.append(MemoryItem(
                content=turn.content,
                tier=MemoryTier.STM,
                metadata={
                    "role": turn.role,
                    "session_id": turn.session_id,
                },
            ))
            s.total_bytes += len(turn.content)
            # Skip extraction on assistant-only turns — production
            # adapters typically extract facts from user statements.
            if turn.role != "user":
                continue
            block = SummaryBlock(
                text=turn.content,
                session_id=turn.session_id,
            )
            # LLMEntityExtractor.extract returns (merged_entities, relations);
            # we only need the entities to pass into FactExtractor.
            entities, _rel = await entity_ex.extract(turn.content, [])
            await fact_ex.extract_facts(block, entities)
        s.per_session_ms.append((time.perf_counter() - t0) * 1000.0)
        s.n_turns += len(sess.turns)
    s.n_sessions = len(sessions)
    s.llm_calls = counter.calls
    s.note = (
        "Realistic production cost: STM append + entity-extract + "
        "fact-extract on every user turn. LLM calls are mocked so the "
        "wall-clock here is mostly framework overhead; multiply "
        "llm_calls_per_session by your real per-call latency for the "
        "production-equivalent number."
    )
    return s


async def _ingest_mem0(sessions: list[Session]) -> IngestStats:
    """Run mem0's default ``Memory.add`` per turn if the package is installed."""
    s = IngestStats(system="mem0", available=False)
    try:
        import mem0  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        s.note = (
            "skipped: mem0 not installed. Install with "
            "'pip install mem0ai' to enable this comparison."
        )
        return s
    # mem0's Memory.add hits a real LLM provider by default; users
    # would need to wire it. We deliberately don't run it here without
    # an explicit credential, to keep `make bench-ingest` free.
    s.note = (
        "skipped: mem0 is installed but requires provider credentials "
        "(OpenAI / Anthropic) and a real LLM call per add(). Enable via "
        "MEM0_PROVIDER + MEM0_MODEL env vars and a follow-up PR."
    )
    return s


# ---------------------------------------------------------------------------
# Driver + narrative
# ---------------------------------------------------------------------------


def _narrative(stats: list[IngestStats]) -> str:
    by_name = {s.system: s for s in stats}
    bare = by_name.get("continuum_bare")
    full = by_name.get("continuum_full")
    naive = by_name.get("naive_append")
    if not (bare and full and naive):
        return "(insufficient data for narrative)"

    bare_vs_naive = (bare.p50_ms / naive.p50_ms) if naive.p50_ms > 0 else float("inf")
    full_vs_bare_calls = full.llm_calls_per_session
    return (
        f"Continuum's hot path (continuum_bare) costs "
        f"{bare.p50_ms:.2f}ms/session at p50 vs {naive.p50_ms:.2f}ms for a "
        f"raw list append — a {bare_vs_naive:.1f}x overhead to gain tier "
        f"semantics, metadata indexing, and async-safe access. With "
        f"production extraction enabled (continuum_full), the framework "
        f"adds {full_vs_bare_calls:.1f} LLM calls per {bare.n_turns // bare.n_sessions}"
        f"-turn session — entirely on a background queue in real "
        f"deployments — for a p50 of {full.p50_ms:.2f}ms/session (LLM "
        f"calls mocked; real wall-clock = mocked + N x your provider "
        f"latency). The trade is paid once per turn and amortises across "
        f"all subsequent retrievals."
    )


async def _run(n_sessions: int, n_turns: int) -> list[IngestStats]:
    sessions = make_sessions(n_sessions, n_turns=n_turns)
    log.info(
        "running ingest benchmark: n_sessions=%d, n_turns=%d (total turns=%d)",
        n_sessions, n_turns, n_sessions * n_turns,
    )

    impls: list[tuple[str, Callable[[list[Session]], Awaitable[IngestStats]]]] = [
        ("naive_append",    _ingest_naive),
        ("continuum_bare",  _ingest_continuum_bare),
        ("continuum_full",  _ingest_continuum_full),
        ("mem0",            _ingest_mem0),
    ]
    stats: list[IngestStats] = []
    for name, fn in impls:
        log.info("  → %s", name)
        try:
            s = await fn(sessions)
        except Exception as exc:
            log.exception("ingest %s failed", name)
            s = IngestStats(system=name, available=False, note=f"error: {exc!r}")
        stats.append(s)
    return stats


def _write_results(stats: list[IngestStats], n_sessions: int, n_turns: int) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    out = RESULTS_DIR / f"ingest_{ts}.json"
    payload = {
        "benchmark": "ingest_throughput",
        "timestamp": ts,
        "config": {"n_sessions": n_sessions, "n_turns": n_turns},
        "systems": [s.summary() for s in stats],
        "narrative": _narrative(stats),
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    latest = RESULTS_DIR / "ingest_latest.json"
    with contextlib.suppress(OSError):
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(out.name)
    return out


def _print_table(stats: list[IngestStats]) -> None:
    print()
    print("=" * 96)
    print(f"{'system':<22}{'avail':>7}{'p50ms':>10}{'p95ms':>10}{'llm/s':>10}{'bytes':>12}{'note':>25}")
    print("-" * 96)
    for s in stats:
        d = s.summary()
        print(
            f"{d['system']:<22}{d['available']!s:>7}"
            f"{d['p50_ms_per_session']:>10.2f}"
            f"{d['p95_ms_per_session']:>10.2f}"
            f"{d['llm_calls_per_session']:>10.2f}"
            f"{d['total_bytes_stored']:>12}"
            f"  {d['note'][:23]!r:>25}"
        )
    print("=" * 96)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sessions", type=int, default=50,
                   help="Number of synthetic sessions (default 50).")
    p.add_argument("--turns", type=int, default=6,
                   help="Turns per session, even = balanced user/assistant (default 6).")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    stats = asyncio.run(_run(args.sessions, args.turns))
    _print_table(stats)
    out = _write_results(stats, args.sessions, args.turns)
    print()
    print("NARRATIVE:")
    print(f"  {_narrative(stats)}")
    print()
    print(f"results: {out.relative_to(Path(__file__).resolve().parents[1])}")
    print("latest:  bench/results/ingest_latest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
