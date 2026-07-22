"""
tests/unit/test_harnesses.py
============================
Tests for the **measurement** scripts, not the product.

Every headline number in `docs/` came out of these harnesses, and this cycle the
instrument was wrong at least five times: a bare `str` passed to `embed()` (which
then embedded per character), a stale DSN file, a `'%tin%'` match that also
matched "meeting", a corpus generator that span forever on an impossible target,
and a DSN regex that swallowed a trailing backslash. Each produced a confident,
wrong intermediate conclusion before it was caught.

A green result is only as trustworthy as the harness behind it, so the harnesses
get tests too.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

pytestmark = pytest.mark.unit

_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    """Import a script by path — `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ── corpus generation ─────────────────────────────────────────────────────────


def test_generator_delivers_exactly_what_was_asked_for() -> None:
    ras = _load("recall_at_scale")
    for shape in ("realistic", "formulaic"):
        got = ras.distractors(300, shape=shape)
        assert len(got) == 300
        assert len(set(got)) == 300, f"{shape} produced duplicates"


def test_generator_refuses_an_impossible_target_instead_of_hanging() -> None:
    """The original spun at 100% CPU for ten minutes when asked for more strings
    than its vocabulary allowed, and looked exactly like a slow embedder."""
    ras = _load("recall_at_scale")
    with pytest.raises(ValueError, match="cannot produce"):
        ras.distractors(10_000_000, shape="formulaic")


def test_generator_is_deterministic() -> None:
    # A measurement you cannot reproduce is an anecdote.
    ras = _load("recall_at_scale")
    assert ras.distractors(50, seed=3) == ras.distractors(50, seed=3)
    assert ras.distractors(50, seed=3) != ras.distractors(50, seed=4)


def test_no_distractor_contains_a_needle_answer() -> None:
    """If a distractor could be mistaken for the answer, recall is unmeasurable.

    This is the check that would have caught the substring-scoring bug: the
    needles are matched by full sentence precisely because short markers like
    "tin" appear inside ordinary words.
    """
    ras = _load("recall_at_scale")
    facts = {fact.strip() for fact, _q in ras.NEEDLES}
    for shape in ("realistic", "formulaic"):
        for sentence in ras.distractors(2000, shape=shape):
            assert sentence.strip() not in facts


def test_needles_are_unique_and_paired_with_a_query() -> None:
    ras = _load("recall_at_scale")
    facts = [f for f, _ in ras.NEEDLES]
    queries = [q for _, q in ras.NEEDLES]
    assert len(set(facts)) == len(facts)
    assert len(set(queries)) == len(queries)
    assert all(f.strip() and q.strip() for f, q in ras.NEEDLES)


def test_enough_needles_are_purely_semantic_probes() -> None:
    """The set is a deliberate mix, and this pins the mix.

    Real questions sometimes share words with the answer ("when is my
    anniversary") and sometimes share none at all ("who does my taxes" ->
    "My accountant is called Filipa Rego"). A set of only the first kind would
    be answerable by the lexical channel alone and would say nothing about
    semantic recall — which is the thing that degrades with store size.

    Half the set carries no content word in common with its answer. If a future
    edit erodes that, the headline recall number quietly starts measuring
    something easier.
    """
    ras = _load("recall_at_scale")
    stop = {"i", "my", "a", "an", "the", "is", "was", "in", "to", "do", "have", "am", "of"}
    semantic = 0
    for fact, query in ras.NEEDLES:
        fact_words = {w.strip(".,'").lower() for w in fact.split()}
        query_words = {w.strip("?.,'").lower() for w in query.split()}
        if not (fact_words & query_words) - stop:
            semantic += 1
    assert semantic >= 8, f"only {semantic}/{len(ras.NEEDLES)} needles are pure semantic probes"


# ── the soak harness ──────────────────────────────────────────────────────────


def test_soak_verdict_fails_on_a_memory_leak() -> None:
    soak = _load("soak_test")
    s = soak.Soak(url="", dsn="", pid=1)
    s.calls, s.errors = 100, 0
    for i in range(10):
        s.samples.append(
            soak.Sample(t=i * 60.0, rss_mb=100 + i * 20, fds=50, pg_conns=4, p95_ms=30)
        )
    s.alive = lambda: True  # type: ignore[method-assign]
    assert soak._verdict(s, None) == 1  # climbing RSS must not pass


def test_soak_verdict_fails_on_leaked_descriptors() -> None:
    soak = _load("soak_test")
    s = soak.Soak(url="", dsn="", pid=1)
    s.calls, s.errors = 100, 0
    for i in range(10):
        s.samples.append(
            soak.Sample(t=i * 60.0, rss_mb=100, fds=50 + i * 20, pg_conns=4, p95_ms=30)
        )
    s.alive = lambda: True  # type: ignore[method-assign]
    assert soak._verdict(s, None) == 1


def test_soak_verdict_passes_a_flat_run() -> None:
    soak = _load("soak_test")
    s = soak.Soak(url="", dsn="", pid=1)
    s.calls, s.errors = 100, 0
    for i in range(10):
        s.samples.append(soak.Sample(t=i * 60.0, rss_mb=100.0, fds=50, pg_conns=4, p95_ms=30))
    s.alive = lambda: True  # type: ignore[method-assign]
    assert soak._verdict(s, None) == 0


def test_soak_verdict_refuses_to_judge_too_few_samples() -> None:
    # Silence is not success: a run that barely started must not report PASS.
    soak = _load("soak_test")
    s = soak.Soak(url="", dsn="", pid=1)
    s.samples.append(soak.Sample(t=0.0, rss_mb=100, fds=50, pg_conns=4, p95_ms=30))
    assert soak._verdict(s, None) == 1
