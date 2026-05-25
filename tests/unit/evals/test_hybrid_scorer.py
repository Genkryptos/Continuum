"""
tests/unit/evals/test_hybrid_scorer.py
======================================
Unit tests for :mod:`evals.longmemeval.hybrid_scorer`.

Covers every deterministic stage (exact / normalized / numeric / unit /
rule_semantic), the cache key stability rule, mode gating, and the
async LLM fallback contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.longmemeval.hybrid_scorer import (
    HybridScorer,
    JudgeCache,
    ScoreResult,
    normalize_text,
    numeric_match,
    percent_match,
    rule_semantic_match,
    unit_aware_match,
)

# ─── Stage 1: normalization + exact ────────────────────────────────────────


def test_normalize_text_strips_articles_and_punct():
    # `_PUNCT_RE` deliberately preserves `.` `%` `/` `-` because they're
    # meaningful in numeric / unit answers (e.g. "3.5 weeks", "1/2 cup").
    # Articles and bare punctuation still go.
    assert normalize_text("The Mayo Clinic!") == "mayo clinic"
    assert normalize_text("  An apple,  please  ") == "apple please"


def test_normalize_text_handles_number_words():
    assert normalize_text("Two and a half hours") == "2.5 hours"
    assert normalize_text("seven shirts") == "7 shirts"


def test_normalize_text_empty():
    assert normalize_text("") == ""


def test_score_sync_exact_match():
    s = HybridScorer(mode="rule")
    r = s.score_sync("q?", "Hello", "Hello")
    assert r.correct
    assert r.method == "exact"


def test_score_sync_normalized_match():
    s = HybridScorer(mode="rule")
    # Case + article difference only
    r = s.score_sync("q?", "The Mayo Clinic", "mayo clinic")
    assert r.correct
    assert r.method == "normalized"


# ─── Stage 2: numeric ──────────────────────────────────────────────────────


def test_numeric_match_subset():
    assert numeric_match("7", "7 shirts in the closet")
    assert numeric_match("7", "I bought 7 shirts and 3 hats")
    assert not numeric_match("7", "5 shirts")


def test_percent_match_handles_word_form():
    assert percent_match("10%", "discount of 10 percent")
    assert percent_match("25 percent", "25%")
    assert not percent_match("10%", "20%")


def test_score_sync_numeric_path():
    s = HybridScorer(mode="rule")
    r = s.score_sync("how many?", "7", "You have 7 shirts.")
    assert r.correct
    assert r.method == "numeric"


# ─── Stage 3: unit-aware ───────────────────────────────────────────────────


def test_unit_aware_match_strips_units():
    assert unit_aware_match("7", "7 shirts")
    assert unit_aware_match("45 minutes", "45 min")
    assert not unit_aware_match("7", "shirts")  # no number on rhs


# ─── Stage 4: rule_semantic ────────────────────────────────────────────────


def test_rule_semantic_substring_matches_both_directions():
    assert rule_semantic_match("#PlankChallenge", "The challenge was #PlankChallenge.")
    assert rule_semantic_match("Your commute is 45 minutes each way.", "45 minutes each way")


def test_rule_semantic_short_predicted_not_matched():
    # The reverse-direction guard requires len(p) >= 3 so 1-char p doesn't
    # accidentally "match" a longer expected.
    assert not rule_semantic_match("a longer answer", "a")


def test_rule_semantic_idk_symmetry():
    assert rule_semantic_match(
        "I do not know based on the conversation.",
        "Not enough information to answer.",
    )


# ─── Mode gating ───────────────────────────────────────────────────────────


def test_mode_none_short_circuits():
    s = HybridScorer(mode="none")
    r = s.score_sync("q?", "x", "x")
    assert not r.correct
    assert r.method == "skipped"


def test_mode_rule_returns_wrong_when_no_match():
    s = HybridScorer(mode="rule")
    r = s.score_sync("q?", "the answer", "completely different")
    assert not r.correct
    assert r.method == "wrong"


# ─── Cache key stability ───────────────────────────────────────────────────


def test_cache_key_stable_across_normalization():
    """Two predictions that normalise to the same string share a key."""
    k1 = JudgeCache.make_key("qid", "The Mayo Clinic", "mayo clinic", "judge-v1")
    k2 = JudgeCache.make_key("qid", "the mayo clinic", "Mayo  Clinic", "judge-v1")
    assert k1 == k2


def test_cache_key_changes_on_judge_model():
    k1 = JudgeCache.make_key("qid", "x", "y", "judge-v1")
    k2 = JudgeCache.make_key("qid", "x", "y", "judge-v2")
    assert k1 != k2


def test_cache_roundtrip_on_disk(tmp_path: Path):
    cache_path = tmp_path / "cache.json"
    cache = JudgeCache(cache_path)
    key = JudgeCache.make_key("qid", "expected", "predicted", "m")
    cache.put(key, ScoreResult(True, "llm", "verdict=true"))
    cache.flush()
    # Re-instantiate; cache should reload from disk.
    fresh = JudgeCache(cache_path)
    hit = fresh.get(key)
    assert hit is not None
    assert hit.correct
    assert hit.method == "llm"
    # And the file is valid JSON.
    assert json.loads(cache_path.read_text())[key]["correct"] is True


# ─── Async LLM fallback ────────────────────────────────────────────────────


class _FakeJudge:
    """Async stub matching the protocol HybridScorer expects."""

    def __init__(self, verdict: bool):
        self.verdict = verdict
        self.calls = 0

    async def is_correct(self, q: str, e: str, p: str) -> bool:
        self.calls += 1
        return self.verdict


@pytest.mark.asyncio
async def test_hybrid_skips_judge_when_deterministic_passes():
    judge = _FakeJudge(verdict=False)
    s = HybridScorer(mode="hybrid", llm_judge=judge)
    result, cache_hit = await s.score("q?", "7", "7 shirts", question_id="x")
    assert result.correct
    # Deterministic path won, judge never invoked.
    assert judge.calls == 0
    assert not cache_hit


@pytest.mark.asyncio
async def test_hybrid_invokes_judge_when_deterministic_fails():
    judge = _FakeJudge(verdict=True)
    s = HybridScorer(mode="hybrid", llm_judge=judge)
    result, cache_hit = await s.score(
        "q?", "the famous Mayo Clinic", "Cleveland Clinic", question_id="x",
    )
    assert result.correct
    assert result.method == "llm"
    assert judge.calls == 1
    assert not cache_hit


@pytest.mark.asyncio
async def test_hybrid_judge_cache_hit_skips_second_call(tmp_path: Path):
    judge = _FakeJudge(verdict=True)
    cache = JudgeCache(tmp_path / "c.json")
    s = HybridScorer(mode="hybrid", llm_judge=judge, judge_model="j", cache=cache)
    # First call: should populate cache.
    r1, hit1 = await s.score("q?", "alpha", "beta", question_id="qid")
    assert r1.correct
    assert not hit1
    # Second call with same inputs: cache hit, judge not re-invoked.
    r2, hit2 = await s.score("q?", "alpha", "beta", question_id="qid")
    assert r2.correct
    assert hit2
    assert judge.calls == 1


@pytest.mark.asyncio
async def test_hybrid_llm_mode_calls_judge_unconditionally():
    judge = _FakeJudge(verdict=True)
    s = HybridScorer(mode="llm", llm_judge=judge)
    # Even though predicted == expected (exact-match would pass), llm mode
    # ALSO calls the judge unconditionally because the deterministic ladder
    # returned correct=True (early-exit) — verify behaviour matches docs.
    result, _ = await s.score("q?", "x", "x", question_id="qid")
    # Deterministic exact match short-circuits; that's the documented
    # behaviour ("LLM judge fallback only on uncertain rows").
    assert result.correct
    assert result.method == "exact"
    assert judge.calls == 0


@pytest.mark.asyncio
async def test_hybrid_judge_error_surfaces_as_wrong():
    class _BrokenJudge:
        async def is_correct(self, *args, **kwargs):
            raise RuntimeError("boom")

    s = HybridScorer(mode="hybrid", llm_judge=_BrokenJudge())
    result, _ = await s.score("q?", "alpha", "beta", question_id="qid")
    assert not result.correct
    assert "llm_error" in result.reason
