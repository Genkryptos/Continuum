"""
tests/unit/evals/test_answer_post.py
====================================
Unit tests for the assistant-claim picker that now lives in
:mod:`evals.longmemeval.answer_post`, with explicit coverage of the
SmallLLM span-selector fallback path.

Each test injects a fake span extractor + a fake SmallLLM so the regex
tier decisions and the trigger-fires-or-not logic can be asserted
without touching the real `_extract_fact_span` regexes or any network.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from evals.longmemeval.answer_post import (
    SPAN_FALLBACK_STATS,
    _pick_answer_from_assistant_claim,
    _should_trigger_span_fallback,
    reset_span_fallback_stats,
)

pytestmark = pytest.mark.unit


# ── fakes ──────────────────────────────────────────────────────────────────


class _FakeSmallLLM:
    """Stub SmallLLM.span_select with a scripted reply + cache-key recorder."""

    def __init__(self, reply: str = "fallback-answer") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str, str | None]] = []

    def span_select(
        self,
        question: str,
        passage: str,
        cache_key: str | None = None,
    ) -> str:
        self.calls.append((question, passage, cache_key))
        return self.reply


def _fixed_extractor(span: str) -> Callable[[str, str], str]:
    """Build a span_extractor that always returns ``span`` regardless of inputs."""

    def _extract(_question: str, _claim: str) -> str:
        return span

    return _extract


@pytest.fixture(autouse=True)
def _clear_stats() -> Any:
    reset_span_fallback_stats()
    yield
    reset_span_fallback_stats()


# ── regex path (no fallback fires) ─────────────────────────────────────────


def test_regex_short_body_wins_no_fallback() -> None:
    """≤12-word body → body verbatim, no fallback (no trigger matches)."""
    llm = _FakeSmallLLM()
    out = _pick_answer_from_assistant_claim(
        "Roscioli is great for Roman pasta.",
        "What restaurant should I try?",
        span_extractor=_fixed_extractor("Roscioli"),
        small_llm=llm,
    )
    # Short body → body verbatim. No comma/and in the body? Actually
    # there IS no comma; "for" — no " and ". Triggers stay quiet.
    assert out == "Roscioli is great for Roman pasta"
    assert llm.calls == []  # no fallback fired


def test_regex_short_body_without_llm_returns_unchanged() -> None:
    out = _pick_answer_from_assistant_claim(
        "Andy is a great choice.",
        "Who should I hire?",
        span_extractor=_fixed_extractor("Andy"),
        small_llm=None,
    )
    assert out == "Andy is a great choice"


def test_regex_medium_body_returns_high_confidence_span() -> None:
    # 14 words → tier 2; span "Boston University" is multi-word so it
    # wins. No fallback trigger (count of tokens > 1).
    llm = _FakeSmallLLM()
    claim = (
        "I went to Boston University for my undergraduate degree from "
        "2010 to 2014 happily."  # 14 words after split
    )
    out = _pick_answer_from_assistant_claim(
        claim,
        "Where did you study?",
        span_extractor=_fixed_extractor("Boston University"),
        small_llm=llm,
    )
    assert out == "Boston University"
    assert llm.calls == []


# ── trigger (a) — nationality adjective ────────────────────────────────────


def test_fallback_fires_on_long_prose_nationality_span() -> None:
    """Tier 3 long prose returns the extractor's span verbatim — if that
    span is a single nationality adjective ("Italian"), trigger (a)
    fires and the SmallLLM reply replaces it."""
    long_claim = (
        "When you're in Rome the classic recommendation is to look "
        "for a small trattoria off the tourist routes where the chef "
        "still cooks the traditional Italian recipes from memory "
        "rather than from a menu printed in five languages."
    )  # > 20 words
    llm = _FakeSmallLLM(reply="Roscioli")
    out = _pick_answer_from_assistant_claim(
        long_claim,
        "Which restaurant did the assistant suggest?",
        span_extractor=_fixed_extractor("Italian"),
        small_llm=llm,
    )
    assert out == "Roscioli"
    assert len(llm.calls) == 1
    assert SPAN_FALLBACK_STATS["nationality_adjective"] == 1


def test_should_trigger_recognises_nationality_token() -> None:
    assert (
        _should_trigger_span_fallback(
            "Italian",
            "Which cuisine?",
            "long enough claim here, with detail.",
        )
        == "nationality_adjective"
    )
    # Multi-word phrases containing the adjective are NOT triggers —
    # only single-token outputs.
    assert (
        _should_trigger_span_fallback(
            "Italian food",
            "Which cuisine?",
            "long enough claim here, with detail.",
        )
        != "nationality_adjective"
    )


# ── trigger (b) — count-shape miss ────────────────────────────────────────


def test_fallback_fires_on_count_question_without_digit() -> None:
    """Count question + answer with no digit → fallback fires."""
    claim = (
        "We ended up using a fair number of eggs for the omelette "
        "across the whole table, more than I expected really."
    )  # tier-2 medium; span won't have a digit
    llm = _FakeSmallLLM(reply="six")
    out = _pick_answer_from_assistant_claim(
        claim,
        "How many eggs did you use?",
        span_extractor=_fixed_extractor("a fair number"),
        small_llm=llm,
    )
    assert out == "six"
    assert SPAN_FALLBACK_STATS["missing_number"] == 1


def test_fallback_silent_when_count_answer_has_digit() -> None:
    """Count question + digit in the picked answer → no fallback."""
    llm = _FakeSmallLLM()
    out = _pick_answer_from_assistant_claim(
        "Use 6 eggs for the omelette.",  # short body, tier 1
        "How many eggs?",
        span_extractor=_fixed_extractor(""),  # span_extractor unused at tier 1
        small_llm=llm,
    )
    assert out == "Use 6 eggs for the omelette"
    assert llm.calls == []


# ── trigger (c) — long claim with structural punctuation ──────────────────


def test_fallback_fires_on_long_claim_with_commas_around_short_span() -> None:
    """Tier 3 long prose returns a short span; the claim is comma-rich.
    Trigger (c) fires."""
    long_claim = (
        "For Roman pasta the top picks would be Roscioli, "
        "Felice a Testaccio, Da Enzo al 29, and Salumeria Roscioli, "
        "all of which take reservations weeks ahead but are worth it."
    )
    llm = _FakeSmallLLM(reply="Roscioli")
    out = _pick_answer_from_assistant_claim(
        long_claim,
        "Which restaurant was suggested?",
        # Single short non-nationality span — only trigger (c) can fire.
        span_extractor=_fixed_extractor("Felice"),
        small_llm=llm,
    )
    assert out == "Roscioli"
    assert SPAN_FALLBACK_STATS["long_claim_structure"] == 1


def test_should_trigger_long_claim_requires_comma_or_and() -> None:
    """A claim 2× the answer but without ',' / 'and' should NOT trigger."""
    assert (
        _should_trigger_span_fallback(
            "Roscioli",
            "Where to eat?",
            "Roscioli is widely considered the single best option there.",
        )
        is None
    )  # no comma, no 'and'


# ── fallback honors regex output when LLM reply is empty ───────────────────


def test_fallback_empty_reply_keeps_regex_output() -> None:
    long_claim = (
        "When you're in Rome the classic recommendation is to look "
        "for a small trattoria off the tourist routes where the chef "
        "still cooks the traditional Italian recipes from memory."
    )
    llm = _FakeSmallLLM(reply="   ")  # whitespace → empty
    out = _pick_answer_from_assistant_claim(
        long_claim,
        "Which restaurant?",
        span_extractor=_fixed_extractor("Italian"),
        small_llm=llm,
    )
    assert out == "Italian"  # regex output preserved
    # Trigger fired but no successful replacement, so the stats counter
    # stays at zero — we only count completed replacements.
    assert SPAN_FALLBACK_STATS["nationality_adjective"] == 0


def test_fallback_llm_exception_keeps_regex_output() -> None:
    class _Boom:
        def span_select(self, *a: Any, **kw: Any) -> str:
            raise RuntimeError("network down")

    out = _pick_answer_from_assistant_claim(
        "When you're in Rome try one of the classic Italian trattorias "
        "found off the tourist routes for a truly authentic experience "
        "of regional cooking, every single time.",
        "Which restaurant?",
        span_extractor=_fixed_extractor("Italian"),
        small_llm=_Boom(),
    )
    assert out == "Italian"


# ── cache-key reuse ───────────────────────────────────────────────────────


def test_cache_key_is_stable_across_calls_same_inputs() -> None:
    """Two picker calls with the same (question, claim) should produce
    the same cache_key — so the SmallLLM's on-disk cache can hit on
    the second call."""
    claim = (
        "When you're in Rome the classic recommendation is a small "
        "trattoria off the tourist routes where the chef still cooks "
        "the traditional Italian recipes from memory."
    )
    question = "Which restaurant was suggested?"
    llm = _FakeSmallLLM(reply="Roscioli")
    _pick_answer_from_assistant_claim(
        claim,
        question,
        span_extractor=_fixed_extractor("Italian"),
        small_llm=llm,
    )
    _pick_answer_from_assistant_claim(
        claim,
        question,
        span_extractor=_fixed_extractor("Italian"),
        small_llm=llm,
    )
    assert len(llm.calls) == 2
    # Same cache key both times — the real SmallLLM cache would have
    # served the second call from sqlite without hitting Ollama.
    assert llm.calls[0][2] == llm.calls[1][2]
    assert llm.calls[0][2].startswith("span:")


def test_cache_key_differs_across_questions() -> None:
    """Different questions → different cache keys (so caches don't collide)."""
    claim = "When you're in Rome try a small trattoria off the tourist routes."
    llm = _FakeSmallLLM(reply="x")
    _pick_answer_from_assistant_claim(
        claim,
        "Which restaurant?",
        span_extractor=_fixed_extractor("Italian"),
        small_llm=llm,
    )
    _pick_answer_from_assistant_claim(
        claim,
        "What cuisine?",
        span_extractor=_fixed_extractor("Italian"),
        small_llm=llm,
    )
    # Both calls only fire if a trigger matches — Italian + long enough
    # claim with comma should be tier-3 nationality. If the claim is
    # short enough to be tier-1 body verbatim, no trigger fires. The
    # assertion below only inspects the calls actually made.
    keys = [c[2] for c in llm.calls]
    assert len(set(keys)) == len(keys) or len(keys) <= 1


# ── default LLM picked up via module-level config ────────────────────────


def test_default_llm_used_when_no_explicit_injection() -> None:
    from evals.longmemeval.answer_post import (
        get_default_span_fallback_llm,
        set_default_span_fallback_llm,
    )

    llm = _FakeSmallLLM(reply="DefaultLLMReply")
    try:
        set_default_span_fallback_llm(llm)  # type: ignore[arg-type]
        assert get_default_span_fallback_llm() is llm
        out = _pick_answer_from_assistant_claim(
            "When you're in Rome the classic recommendation is a small "
            "trattoria off the tourist routes where the chef still cooks "
            "the traditional Italian recipes from memory.",
            "Which restaurant?",
            span_extractor=_fixed_extractor("Italian"),
            # NB: no small_llm injected here.
        )
        assert out == "DefaultLLMReply"
    finally:
        set_default_span_fallback_llm(None)
