"""
tests/unit/test_capture_multilingual.py
=======================================
The per-language adversarial set for automatic capture.

Capture's whole defence is a measured 0-false-capture rate. English earned it on
47 negatives before it shipped; a new language has to clear the same bar. This
file is that bar for the Latin-script languages capture now supports, and an
honest boundary for the ones it does not.

**Supported (regex over word order):** pt, es, fr, de, it. Precision-biased in
exactly the way English is — stative first-person frames accept, and actions,
"just did", questions, imperatives and transient states are refused. The verb
carries the signal, which matters because Portuguese/Spanish/Italian drop the
subject pronoun.

**Not supported:** hi, ja, zh. Word-order regexes do not survive Devanagari or
CJK — there are no spaces to anchor on — and the author cannot validate
adversarial cases in them well enough to claim 0 false captures. They fail
*safe* (silence, never a wrong capture) and are named in `KNOWN_UNSUPPORTED`
rather than left implicit.
"""

from __future__ import annotations

import pytest

from continuum.promotion.capture import extract_durable_facts, looks_like_secret

pytestmark = pytest.mark.unit

SUPPORTED = frozenset({"pt", "es", "fr", "de", "it"})


# (language, sentence) — statements a speaker would expect to be remembered.
DURABLE: list[tuple[str, str]] = [
    # Portuguese (pro-drop: the subject pronoun is usually omitted)
    ("pt", "Eu moro em Lisboa."),
    ("pt", "Moro no Porto há dez anos."),
    ("pt", "Trabalho na Nimbus Data."),
    ("pt", "Sou alérgico a amendoim."),
    ("pt", "Falo português e inglês."),
    ("pt", "Sou vegetariano."),
    # Spanish
    ("es", "Vivo en Madrid."),
    ("es", "Trabajo en Nimbus Data."),
    ("es", "Soy alérgico a los cacahuetes."),
    ("es", "Hablo español e inglés."),
    ("es", "Prefiero PostgreSQL a MySQL."),
    # French
    ("fr", "Je vis à Paris."),
    ("fr", "Je travaille chez Nimbus Data."),
    ("fr", "Je suis allergique aux arachides."),
    ("fr", "Je parle français et anglais."),
    # German
    ("de", "Ich wohne in Berlin."),
    ("de", "Ich arbeite bei Nimbus Data."),
    ("de", "Ich bin allergisch gegen Erdnüsse."),
    ("de", "Ich spreche Deutsch und Englisch."),
    # Italian
    ("it", "Vivo a Roma."),
    ("it", "Lavoro alla Nimbus Data."),
    ("it", "Sono allergico alle arachidi."),
]

# The categories English is held to — actions, "just did", questions,
# imperatives, transient states — in each supported language. NONE may be
# captured; a rule that captures one is worse than a rule that captures nothing.
NOISE: list[tuple[str, str]] = [
    # Portuguese
    ("pt", "Corri os testes e falharam."),
    ("pt", "Acabei de fazer deploy."),
    ("pt", "Corrigi o erro no retriever."),
    ("pt", "Fiz o commit."),
    ("pt", "Podes corrigir este erro?"),
    ("pt", "Onde está o ficheiro de configuração?"),
    ("pt", "Tenho uma reunião às três."),
    # Spanish
    ("es", "Acabo de desplegar a producción."),
    ("es", "He ejecutado las pruebas."),
    ("es", "Arreglé el error de importación."),
    ("es", "¿Puedes arreglar el error?"),
    ("es", "¿Dónde está el archivo?"),
    ("es", "Tengo una reunión a las tres."),
    # French — passé composé (avoir + participle) is the action trap
    ("fr", "J'ai lancé les tests."),
    ("fr", "J'ai corrigé le bug."),
    ("fr", "Je viens de déployer."),
    ("fr", "Peux-tu corriger ce bug ?"),
    ("fr", "Où est le fichier de configuration ?"),
    ("fr", "J'ai une réunion à trois heures."),
    # German — Perfekt (habe + ge-…-t) collides with "ich habe X"
    ("de", "Ich habe die Tests ausgeführt."),
    ("de", "Ich habe gerade deployed."),
    ("de", "Ich habe den Fehler behoben."),
    ("de", "Kannst du den Fehler beheben?"),
    ("de", "Wo ist die Konfigurationsdatei?"),
    ("de", "Ich habe um drei einen Termin."),
    # Italian — passato prossimo (ho + participle)
    ("it", "Ho eseguito i test."),
    ("it", "Ho appena fatto il deploy."),
    ("it", "Ho corretto il bug."),
    ("it", "Puoi correggere questo errore?"),
    ("it", "Dov'è il file di configurazione?"),
    ("it", "Ho una riunione alle tre."),
]

# Explicitly not supported — captured nothing, and the rules do not try.
KNOWN_UNSUPPORTED: list[tuple[str, str]] = [
    ("hi", "मैं दिल्ली में रहता हूँ।"),
    ("hi", "मुझे मूंगफली से एलर्जी है।"),
    ("ja", "私は東京に住んでいます。"),
    ("zh", "我住在北京。"),
]

SECRETS: list[tuple[str, str]] = [
    ("pt", "A minha password é hunter2."),
    ("es", "Mi API key es sk-proj-abc123def456ghi789jkl."),
    ("de", "Mein Token ist ghp_16C7e42F292c6912E7710c838347Ae178B4a."),
]


@pytest.mark.parametrize(
    ("lang", "sentence"), DURABLE, ids=[f"{lg}-{i}" for i, (lg, _) in enumerate(DURABLE)]
)
def test_supported_language_durable_facts_are_captured(lang: str, sentence: str) -> None:
    assert lang in SUPPORTED
    assert extract_durable_facts(sentence), f"[{lang}] missed a durable fact: {sentence!r}"


@pytest.mark.parametrize(
    ("lang", "sentence"), NOISE, ids=[f"{lg}-{i}" for i, (lg, _) in enumerate(NOISE)]
)
def test_noise_is_never_captured(lang: str, sentence: str) -> None:
    """The non-negotiable bar. English cleared 47 negatives; these are the
    equivalents, and a single false capture here fails the whole feature's
    safety story."""
    got = extract_durable_facts(sentence)
    assert not got, f"[{lang}] wrongly captured {sentence!r} -> {[f.text for f in got]}"


def test_zero_false_captures_across_the_whole_non_english_noise_set() -> None:
    # The headline number, asserted as one fact so a regression is unmissable.
    wrong = [s for _lang, s in NOISE if extract_durable_facts(s)]
    assert wrong == [], f"{len(wrong)}/{len(NOISE)} false captures: {wrong}"


@pytest.mark.parametrize(("lang", "sentence"), SECRETS, ids=[lg for lg, _ in SECRETS])
def test_credentials_are_refused_regardless_of_language(lang: str, sentence: str) -> None:
    # Secret detection is pattern-based, not grammar-based, so it already works
    # in any language. It must stay that way.
    assert looks_like_secret(sentence), f"[{lang}] missed a credential in {sentence!r}"
    assert not extract_durable_facts(sentence)


@pytest.mark.parametrize(
    ("lang", "sentence"), KNOWN_UNSUPPORTED, ids=[lg for lg, _ in KNOWN_UNSUPPORTED]
)
def test_unsupported_scripts_stay_silent(lang: str, sentence: str) -> None:
    """Devanagari/CJK are not supported and must fail *safe*.

    If one of these starts capturing, it needs its own adversarial noise set at
    the same bar before it can be trusted — silence is the correct behaviour
    until then, not a bug to 'fix' by loosening the rules.
    """
    assert lang not in SUPPORTED
    assert extract_durable_facts(sentence) == []


def test_the_supported_set_spans_five_languages() -> None:
    assert {lang for lang, _ in DURABLE} == SUPPORTED
    assert {lang for lang, _ in NOISE} == SUPPORTED
    # Weighted toward what must be refused, like the English set.
    assert len(NOISE) >= len(DURABLE)
