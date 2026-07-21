"""
continuum.promotion.capture
===========================
Decide whether a conversational turn contains a **durable fact about the user**
worth writing to long-term memory — deterministically, with no LLM and no
network.

Automatic capture is the feature most likely to make memory worse. A store that
swallows every turn fills with "I ran the tests", "ok do that", and the odd API
key, and then buries the handful of facts that mattered. So this is biased hard
toward **precision**: it fires only on unambiguous first-person statements of a
standing fact, and stays silent on everything else. A missed fact costs a
`remember` call; a captured secret or a store full of noise costs trust.

What it captures
----------------
Stative first-person declaratives — statements that are still true tomorrow::

    "I live in Boston."                 -> captured
    "My daughter is named Mira."        -> captured
    "I use Neovim as my editor."        -> captured
    "I'm allergic to penicillin."       -> captured

What it refuses
---------------
* **Actions and events** — "I ran the tests", "I fixed the bug", "I just
  deployed". True for a moment, noise forever.
* **Questions, requests, imperatives** — "how do I configure this?", "add a
  test for that".
* **Hypotheticals and uncertainty** — "if I move to Berlin", "I might switch
  to Postgres", "I was thinking about buying a car".
* **Anything about the assistant or the code**, not the user.
* **Secrets** — API keys, tokens, passwords, card and ID numbers. These are
  never captured, even in an otherwise perfect sentence, and the *whole*
  sentence is dropped rather than redacted: a sentence built around a secret is
  rarely a fact worth keeping.

Everything here is a heuristic over English surface form. It will miss facts.
That is the intended failure direction.

**English only, and this is asymmetric with the rest of the system.** Retrieval
is multilingual — bge-m3 answers an English question with a Portuguese or Hindi
memory — but these rules are regexes over English word order, so
``"Eu moro em Lisboa."`` captures nothing. It fails *safe* (silence, never a
wrong capture), but a non-English user should know their turns are being
skipped and keep using ``remember`` explicitly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["CapturedFact", "extract_durable_facts", "looks_like_secret"]


# ── secrets: refused outright, before any other consideration ────────────────

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:sk|pk|rk)[-_][A-Za-z0-9_-]{12,}"),  # provider-style keys
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}"),  # GitHub tokens
    re.compile(r"\bAKIA[0-9A-Z]{12,}"),  # AWS access key id
    re.compile(r"\bey[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\."),  # JWT
    re.compile(r"\b[A-Fa-f0-9]{32,}\b"),  # long hex digest
    re.compile(r"\b(?:\d[ -]?){13,19}\b"),  # card-ish number runs
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # US SSN
    re.compile(
        r"\b(?:password|passwd|passphrase|secret|api[\s_-]?key|access[\s_-]?token"
        r"|private[\s_-]?key|credential)s?\b",
        re.I,
    ),
    re.compile(r"postgres(?:ql)?://\S*:\S+@"),  # DSN carrying a password
)


def looks_like_secret(text: str) -> bool:
    """True if *text* contains anything credential-shaped.

    Deliberately trigger-happy: a false positive costs one un-captured fact, a
    false negative writes a live key into a database that gets backed up.
    """
    return any(p.search(text) for p in _SECRET_PATTERNS)


# ── shape filters ────────────────────────────────────────────────────────────

#: Turns that are not the user asserting something about themselves.
_NOT_A_STATEMENT = re.compile(
    r"^\s*(?:"
    r"(?:can|could|would|should|will|do|does|did|is|are|was|were|have|has|am)\b"  # questions
    r"|(?:who|what|when|where|why|how|which)\b"
    r"|(?:please|let'?s|add|write|make|fix|run|show|give|tell|explain|create|update"
    r"|remove|delete|check|try|use|implement|refactor|commit|push|open|close)\b"  # imperatives
    r")",
    re.I,
)

#: Markers of something not (yet) true: hypothetical, planned, uncertain.
_UNCERTAIN = re.compile(
    r"\b(?:if|maybe|perhaps|probably|might|may|could|would|should|thinking\s+about"
    r"|considering|planning\s+to|going\s+to|want\s+to|wish|hope|suppose|guess"
    r"|not\s+sure|unsure|i'?m\s+trying\s+to)\b",
    re.I,
)

#: First-person stative frames — "I <be/stative verb> …" or "My <noun> is …".
#: Each requires a copula or a standing-state verb, which is what separates a
#: fact from an action.
#: Habitual adverbs. "I *always* squash before merging" is a standing practice —
#: the adverb is evidence of durability, yet it sits between the subject and the
#: verb and would otherwise break the anchor and lose the fact. ("never" stays
#: out: it is handled as a negation.)
_HABITUAL = r"(?:always|usually|generally|normally|typically|mostly|often|mainly|still)\s+"

_STATIVE_I = re.compile(
    r"^\s*i\s+" + f"(?:{_HABITUAL})?" + r"(?:"
    r"am|'m|was\s+born|live|reside|work|study|studied|speak|own|have|has|drive"
    r"|prefer|like|love|hate|use|play|need|avoid|keep|maintain|can'?t\s+eat|don'?t\s+eat"
    r"|am\s+allergic|graduated|grew\s+up"
    r")\b",
    re.I,
)
_STATIVE_MY = re.compile(
    r"^\s*my\s+[\w'\- ]{2,40}?\s+(?:is|are|was|were)\b",
    re.I,
)

#: Verbs that describe a moment, not a state — refused even in the "I …" frame.
_EPISODIC = re.compile(
    r"^\s*i\s+(?:just\s+|already\s+|finally\s+)?(?:"
    r"ran|run|fixed|fix|deployed|deploy|pushed|push|committed|commit|merged|merge"
    r"|added|add|removed|remove|deleted|delete|tried|try|tested|test|checked|check"
    r"|opened|open|closed|close|started|start|stopped|stop|clicked|typed|asked"
    r"|updated|update|installed|install|restarted|restart|read|wrote|write|saw|see"
    r")\b",
    re.I,
)

#: "I am running…", "I'm getting…" — a copula plus a participle is an action
#: wearing a stative frame, which is how most episodic noise sneaks through.
_PROGRESSIVE = re.compile(r"^\s*i\s*(?:am|'m)\s+(?:\w+ing)\b", re.I)

#: Negations and retractions. "I don't live in Boston anymore" is a real update,
#: but storing it as if it asserted something is worse than missing it —
#: retraction belongs to supersession, not capture.
_NEGATED = re.compile(
    r"\b(?:don'?t|doesn'?t|didn'?t|do\s+not|never|no\s+longer|not\s+anymore"
    r"|isn'?t|aren'?t|wasn'?t|won'?t|can'?t\s+stand)\b",
    re.I,
)

#: The object is deictic — "I use *this* file", "I love *that* bug". It points
#: at the current screen, not at a standing preference.
_DEICTIC_OBJECT = re.compile(
    r"^\s*i\s+\w+(?:\s+\w+)?\s+(?:this|that|these|those|it|here|there)\b", re.I
)

#: Heads of "My <X> is …" that belong to the work, not the person. Without this
#: every "my build is failing" becomes a permanent memory.
_WORK_SUBJECT = re.compile(
    r"^\s*my\s+(?:build|test|tests|branch|commit|pr|ci|pipeline|code|repo|repository"
    r"|server|terminal|error|errors|output|script|file|files|function|class|config"
    r"|log|logs|diff|patch|deploy|deployment|environment|env|venv|container|process"
    r"|query|request|response|session|task|ticket|issue|bug|change|changes)\b",
    re.I,
)

#: "I am/have <transient>" — true this afternoon, noise forever.
_TRANSIENT_STATE = re.compile(
    r"^\s*i\s+(?:am|'m)\s+(?:done|here|back|ready|stuck|confused|sorry|tired|busy"
    r"|online|offline|off|away|good|fine|ok|okay|sure|curious|lost|blocked"
    # \w+ not \w: a trailing \b after a single char lands mid-word and the
    # whole alternative silently never matches ("I am on the branch").
    r"|on\s+\w+|in\s+the\s+(?:middle|process)|at\s+(?:the\s+)?(?:office|desk))\b",
    re.I,
)
#: ``(?:\w+\s+){0,2}`` matters: without it a single modifier walks straight
#: past the filter — "I have a *dentist* appointment on Thursday" was captured
#: as a durable fact, which is the exact noise this list exists to stop.
_TRANSIENT_HAVE = re.compile(
    r"^\s*i\s+have\s+(?:a|an|some|another|one|two|three)?\s*(?:\w+\s+){0,2}"
    r"(?:meeting|call|appointment|deadline|interview|flight|reservation|booking"
    r"|headache|cold|fever|cough|question|idea|problem|issue|error|bug|thought"
    r"|minute|moment|second|feeling|hunch|plan|task|ticket|errand|chore|demo|review)\b",
    re.I,
)

#: Subjects that are not the user.
_NOT_ABOUT_USER = re.compile(
    r"^\s*(?:you|it|this|that|there|the|we|they|he|she|claude|the\s+code|the\s+test)\b",
    re.I,
)

#: A fact needs some substance; a bare "I am." is not one. Three admits
#: "I am vegetarian." — the transient-state list is what rejects "I am done."
_MIN_WORDS = 3
_MAX_CHARS = 300

#: Everything that disqualifies a sentence, in one place so the rejection
#: reason is greppable and the order is obvious.
_REJECTORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("not-a-statement", _NOT_A_STATEMENT),
    ("not-about-user", _NOT_ABOUT_USER),
    ("work-subject", _WORK_SUBJECT),
    ("episodic", _EPISODIC),
    ("progressive", _PROGRESSIVE),
    ("transient-state", _TRANSIENT_STATE),
    ("transient-have", _TRANSIENT_HAVE),
    ("deictic-object", _DEICTIC_OBJECT),
)


#: "I switched from Neovim to Zed" — a preference replacing a preference. Both
#: sides must be capitalised, the same precision lever the attribute extractor
#: uses: it separates "I switched from Neovim to Zed" from "I switched branches".
_SWITCHED = re.compile(r"^\s*[Ii]\s+(?:switched|changed)\s+from\s+[A-Z][\w.+-]*\s+to\s+[A-Z]")


def _states_a_change(sentence: str) -> bool:
    """True when the sentence announces a *change* to a durable attribute.

    This is the case longitudinal use exposed. The stative frames above accept
    "I live in Porto" but refuse "I moved to Berlin", so a month of daily use
    locked in day-one facts and silently dropped every correction — memory
    drifting further from the truth the longer it ran. The change is the most
    valuable thing to capture, because it is what makes the stored fact stale.

    Reuses :func:`extract_attribute`: if that recognises the sentence it is, by
    construction, a statement about a durable attribute of the user, and it was
    tuned to demand capitalised proper nouns (0 false tags on its adversarial
    set) — so it is a safe second opinion rather than a looser one.
    """
    from continuum.promotion.attribute_extract import extract_attribute

    if _SWITCHED.match(sentence):
        return True
    return extract_attribute(sentence) is not None


@dataclass(frozen=True)
class CapturedFact:
    """A sentence judged durable, with the rule that accepted it."""

    text: str
    rule: str


def _sentences(text: str) -> list[str]:
    """Split on sentence enders and newlines; keep it dumb and predictable."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [p.strip() for p in parts if p and p.strip()]


def extract_durable_facts(text: str) -> list[CapturedFact]:
    """Durable first-person facts in *text*, or ``[]`` — the common answer.

    Sentence-at-a-time: a turn that mixes a fact with chatter yields just the
    fact. Order is preserved and duplicates within the turn are collapsed.
    """
    out: list[CapturedFact] = []
    seen: set[str] = set()
    for raw in _sentences(text):
        sentence = raw.strip()
        if not sentence or len(sentence) > _MAX_CHARS:
            continue
        if sentence.endswith("?") or len(sentence.split()) < _MIN_WORDS:
            continue
        if looks_like_secret(sentence):
            continue
        if _UNCERTAIN.search(sentence) or _NEGATED.search(sentence):
            continue
        if any(pattern.match(sentence) for _name, pattern in _REJECTORS):
            continue

        rule = (
            "stative-i"
            if _STATIVE_I.match(sentence)
            else "stative-my"
            if _STATIVE_MY.match(sentence)
            else "changed-attribute"
            if _states_a_change(sentence)
            else ""
        )
        if not rule:
            continue
        key = sentence.lower().rstrip(".")
        if key in seen:
            continue
        seen.add(key)
        out.append(CapturedFact(text=sentence, rule=rule))
    return out
