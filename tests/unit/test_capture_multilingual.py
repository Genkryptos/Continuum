"""
tests/unit/test_capture_multilingual.py
=======================================
The per-language adversarial set for automatic capture.

This exists **before** any non-English rules, deliberately. Capture's whole
defence is a measured 0-false-capture rate; adding patterns for a language with
no way to measure that would be trading the one property that makes the feature
safe for a hit rate nobody checked.

Today the answer is uniform: capture is English-only and stays silent on
everything here, durable facts and noise alike. That is the *safe* failure —
silence, never a wrong capture — and it is asymmetric with retrieval, which is
genuinely multilingual (an English question retrieves a Portuguese memory).

These tests pin two things:

1. The refusal is total, so nobody believes non-English capture half-works.
2. When rules are added, the noise cases here must still be refused. That is
   the bar English was held to (0 false captures across 47 negatives), and it
   is the bar a new language has to clear before it ships.
"""

from __future__ import annotations

import pytest

from continuum.promotion.capture import extract_durable_facts, looks_like_secret

pytestmark = pytest.mark.unit


# (language, sentence) — statements a speaker would expect to be remembered.
DURABLE: list[tuple[str, str]] = [
    ("pt", "Eu moro em Lisboa."),
    ("pt", "Trabalho na Nimbus Data."),
    ("pt", "Sou alérgico a amendoim."),
    ("es", "Vivo en Madrid."),
    ("es", "Soy alérgico a los cacahuetes."),
    ("es", "Mi portátil es un MacBook Air."),
    ("de", "Ich wohne in Berlin."),
    ("de", "Ich arbeite bei Nimbus Data."),
    ("fr", "Je vis à Paris."),
    ("fr", "Je suis allergique aux arachides."),
    ("it", "Vivo a Roma."),
    ("hi", "मैं दिल्ली में रहता हूँ।"),
    ("hi", "मुझे मूंगफली से एलर्जी है।"),
    ("ja", "私は東京に住んでいます。"),
    ("zh", "我住在北京。"),
]

# The same languages' equivalents of "I ran the tests" — the noise that must
# NEVER be captured, whatever rules arrive later.
NOISE: list[tuple[str, str]] = [
    ("pt", "Corri os testes e falharam."),
    ("pt", "Acabei de fazer deploy."),
    ("pt", "Podes corrigir este erro?"),
    ("es", "Acabo de desplegar a producción."),
    ("es", "He ejecutado las pruebas."),
    ("es", "¿Puedes arreglar el error?"),
    ("de", "Ich habe die Tests ausgeführt."),
    ("de", "Ich habe gerade deployed."),
    ("de", "Kannst du den Fehler beheben?"),
    ("fr", "J'ai lancé les tests."),
    ("fr", "Peux-tu corriger ce bug ?"),
    ("it", "Ho eseguito i test."),
    ("hi", "मैंने टेस्ट चलाए।"),
    ("ja", "テストを実行しました。"),
    ("zh", "我刚刚运行了测试。"),
]

SECRETS: list[tuple[str, str]] = [
    ("pt", "A minha password é hunter2."),
    ("es", "Mi API key es sk-proj-abc123def456ghi789jkl."),
    ("de", "Mein Token ist ghp_16C7e42F292c6912E7710c838347Ae178B4a."),
]


@pytest.mark.parametrize(
    ("lang", "sentence"), NOISE, ids=[f"{lg}-{i}" for i, (lg, _) in enumerate(NOISE)]
)
def test_non_english_noise_is_never_captured(lang: str, sentence: str) -> None:
    """The bar any future language rules must clear.

    English reached 0 false captures across 47 negatives before it shipped. A
    new language that cannot do the same makes memory worse, not better.
    """
    got = extract_durable_facts(sentence)
    assert not got, f"[{lang}] wrongly captured {sentence!r} -> {[f.text for f in got]}"


@pytest.mark.parametrize(("lang", "sentence"), SECRETS, ids=[lg for lg, _ in SECRETS])
def test_credentials_are_refused_regardless_of_language(lang: str, sentence: str) -> None:
    # Secret detection is pattern-based, not grammar-based, so it is the one
    # part of capture that already works in any language. It must stay that way.
    assert looks_like_secret(sentence), f"[{lang}] missed a credential in {sentence!r}"
    assert not extract_durable_facts(sentence)


def test_capture_is_currently_english_only_and_says_so() -> None:
    """Documents today's behaviour honestly rather than aspirationally.

    If this starts failing, some language has gained support — at which point
    the durable cases for it move into a passing set and the noise cases above
    become its regression bar.
    """
    captured = [s for _lang, s in DURABLE if extract_durable_facts(s)]
    assert captured == [], (
        "non-English capture now fires; move those languages' DURABLE cases into "
        f"an asserted set and keep their NOISE cases refused. Fired on: {captured}"
    )


def test_the_set_covers_both_scripts_and_a_spread_of_languages() -> None:
    # A set that is all Romance languages would prove very little about the
    # rules being English-shaped rather than Latin-shaped.
    languages = {lang for lang, _ in DURABLE}
    assert len(languages) >= 6
    assert {"hi", "ja", "zh"} & languages, "no non-Latin script represented"
    assert len(NOISE) >= len(DURABLE) - 1  # weighted toward what must be refused
