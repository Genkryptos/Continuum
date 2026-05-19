"""
tests/benchmarks/conftest.py
=============================
Fixtures for performance benchmarks driven by ``pytest-benchmark``.

Run with::

    make benchmark
    # or directly:
    pytest tests/benchmarks -v --benchmark-only --benchmark-sort=mean

All benchmark tests should be decorated ``@pytest.mark.benchmark`` and use
the ``benchmark`` fixture injected by pytest-benchmark.

Example::

    def test_score_throughput(benchmark, large_memory_corpus, continuum_config):
        scorer = SimpleScorer(continuum_config.scoring)
        query  = Query(text="user preferences", agent_id="test-agent")
        now    = datetime.now(timezone.utc)

        result = benchmark(scorer.score_batch, large_memory_corpus, query, now)
        assert len(result) == len(large_memory_corpus)
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from continuum.core.types import MemoryItem, MemoryTier

# ---------------------------------------------------------------------------
# large_memory_corpus — 1 000-item candidate pool for throughput benchmarks
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def large_memory_corpus(sample_memories: list[MemoryItem]) -> list[MemoryItem]:
    """
    Extend the 100-item ``sample_memories`` fixture to 1 000 items,
    giving benchmarks a realistic retrieval candidate pool.

    The extra 900 items are LTM facts of varying importance and age so the
    scorer has meaningful work to do (no trivially uniform input).
    """
    now = datetime.now(UTC)
    corpus = list(sample_memories)  # copy so sample_memories stays intact

    phrases = [
        "User language preference is English.",
        "Service SLA is 99.9% uptime.",
        "Default currency is USD.",
        "Rate limit is 1000 req/min.",
        "Retry budget: 3 attempts with exponential back-off.",
        "Data residency: EU West region.",
        "Max context window: 128k tokens.",
        "Embedding model: BAAI/bge-m3, dim=1024.",
        "JWT secret rotates every 90 days.",
        "Audit logs retained for 7 years.",
    ]

    for i in range(900):
        phrase = phrases[i % len(phrases)]
        corpus.append(
            MemoryItem(
                id=str(uuid.uuid4()),
                content=f"[bench-{i:04d}] {phrase}",
                tier=MemoryTier.LTM,
                importance=0.40 + (i % 6) * 0.10,  # 0.4 – 0.9
                confidence=0.60 + (i % 4) * 0.10,  # 0.6 – 0.9
                created_at=now - timedelta(hours=i * 0.5),
                session_id=f"sess-bench-{i % 10:02d}",
                agent_id="bench-agent",
                user_id="bench-user",
            )
        )

    return corpus
