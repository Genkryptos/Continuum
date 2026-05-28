"""
evals/longmemeval/scaffold_filter.py
====================================
Central scaffold / prompt-debris detector.

Why this exists
---------------
The wiki retriever renders evidence as a Markdown document with
``## Matched Turn`` / ``## Nearby Context`` headers and
``- (user): …`` bullets. The decomposed-answer prompt emits
``Sub-answer:`` / ``Matching facts:`` prefixes. These tokens leak two
places:

1. Into the **candidate pool** — the regex extractors run over the
   rendered evidence text and happily turn headings like
   ``"Evidence Window"`` into entity candidates.
2. Into the **final answer** — the LLM sometimes echoes a label like
   ``"Sub-answer"`` or ``"Candidates:"`` instead of producing an
   actual answer string.

This module is the single source of truth for "is this string
scaffold". Every gate (extractor, verifier, validator, repair
fallback) calls :func:`is_scaffold_text` so the blacklist is
consistent and one-touch to extend.

Contract
--------
``is_scaffold_text(text)`` is case-insensitive, punctuation-tolerant,
and matches both the exact tokens and common surface variants ("Sub
answer", "sub-answer", "Sub-Answer:").
"""

from __future__ import annotations

import re

#: Hard tokens — case-insensitive substring match on a normalised form.
#: Each entry should be a substring that would never legitimately appear
#: as a final answer or a candidate value on its own.
_SCAFFOLD_TOKENS: tuple[str, ...] = (
    "evidence window",
    "matched turn",
    "nearby context",
    "sub answer",       # also catches "Sub-Answer:" after normalisation
    "subanswer",
    "candidates",       # the "Candidates:" header
    "matching facts",
    "matching fact",
    "reviewing the candidates",
    "reviewing candidates",
    "i dont know",        # normalised form of "I don't know"
    "i do not know",
    "no information",
    "not mentioned",
    "final answer",
    "answer:",          # bare prompt-echo
    "question:",
    "session=",         # leftover from rendered span text
    "evidence:",
    "verified evidence",
    "no evidence",
    "no candidates",
    "n/a",
)

#: Strings that are entire scaffold lines — match on the whole
#: normalised string, not just a substring. Catches "Sub-answer" as a
#: bare value distinct from "Sub-answer: the actual content".
_SCAFFOLD_EXACT: frozenset[str] = frozenset({
    "sub answer",
    "subanswer",
    "candidates",
    "matched turn",
    "nearby context",
    "evidence window",
    "matching facts",
    "matching fact",
    "answer",
    "question",
    "evidence",
    "final answer",
    "summary",
    "facts",
    "context",
    "session",
    "turn",
    "role user",
    "role assistant",
    "user",
    "assistant",
    "system",
})

#: Markdown / wrapper line patterns. Headings, code fences, bullet
#: labels that the extractor's regex would otherwise turn into
#: candidates.
_HEADING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*#{1,6}\s+"),                 # # / ## / ### …
    re.compile(r"^\s*```"),                       # code fences
    re.compile(r"^\s*-{3,}\s*$"),                 # markdown rules
    re.compile(r"^\s*\*{3,}\s*$"),                # * * *
    re.compile(r"^\s*=+\s*$"),                    # ===
    # ``**Sister's Birthday**`` style — bold-only line with no body.
    # Matches lines whose stripped form is a single ``**...**`` chunk.
    re.compile(r"^\s*\*{1,3}[^*\n]+?\*{1,3}\s*$"),
    # ``__Heading__`` underscore variant.
    re.compile(r"^\s*_{2,3}[^_\n]+?_{2,3}\s*$"),
)

#: ``Label: value`` lines where the line is essentially metadata —
#: short label followed by a colon. Catches `Occasion: Sister's
#: birthday`, `Topic: Marketing`, `Subject: ...`. Skipped by the
#: claim extractor.
_LABEL_LINE_RE = re.compile(
    r"^\s*(?:occasion|topic|subject|category|tags?|date|time|location|"
    r"venue|event|type|note|notes?|context|setting|gift|recipient|"
    r"present|item|reminder|attendees?)\s*:\s*(?P<body>.*?)\s*$",
    re.IGNORECASE,
)

#: User-request / question shapes. A sentence that is the user asking
#: the assistant for help is NOT a memory claim, even when it happens
#: to share content words with the benchmark question. The pattern
#: catches "Can you ...", "Could you ...", "Would you ...",
#: "Please ...", "Help me ...", "Tell me ..." opens followed by an
#: imperative / interrogative.
_USER_REQUEST_RE = re.compile(
    # Polite-frame requests ("Can you...", "Please...", etc.)
    r"^\s*(?:can\s+you|could\s+you|would\s+you|will\s+you|"
    r"please|help\s+me|tell\s+me|let\s+me\s+know|"
    r"may\s+I|do\s+you\s+have\s+any|"
    r"any\s+(?:advice|recommendations?|suggestions?|tips)|"
    r"how\s+(?:do|would|can|could)\s+I"
    # Bare-imperative requests addressed to the assistant — the v8
    # failure mode where ``"Brainstorm ideas for work from home jobs"``
    # got returned as an assistant-memory answer. The verb leads the
    # sentence with no subject pronoun (a leading "I"/"we"/"you" means
    # the sentence is content, not a request).
    r"|brainstorm|generate|create|write|make|draft|draw|design|build|"
    r"give\s+me|show\s+me|find\s+me|search\s+for|come\s+up\s+with|"
    r"list\s+(?:out|some|the|all)|suggest\s+(?:some|a\s+few|the)|"
    r"recommend\s+(?:some|a\s+few|the)|explain\s+(?:how|why|the|what)|"
    r"describe\s+(?:how|the|a)|outline\s+(?:the|a|some)|"
    r"prepare\s+(?:a|the|some)|summari[sz]e\s+(?:the|a)|"
    r"compare\s+(?:the|these|a)|provide\s+(?:me\s+with\s+|a\s+|some\s+|the\s+)?"
    r")\b",
    re.IGNORECASE,
)


def is_label_metadata_line(line: str) -> bool:
    """
    True when the line is a label/metadata field — ``Occasion: …`` or
    ``Topic: …``. The label part is matched against a known set; the
    body can be anything (including empty).

    This is a stronger check than ``is_scaffold_line`` because it
    catches lines that look like content but are really structured
    metadata leaking from the bundle.
    """
    if not line or not line.strip():
        return False
    return bool(_LABEL_LINE_RE.match(line.strip()))


#: Bundle-scaffold role prefix — matches the wiki bundle's
#: structural per-turn markers in any of the live shapes:
#:
#:   ``turn 1 (user):``
#:   ``- turn 1 (user):``
#:   ``- (user):``
#:   ``(user):``
#:   ``user:``  (only when it leads the string)
#:
#: A string that BEGINS with one of these is the bundle's structural
#: text, never a memory claim or final answer. Wired into
#: :func:`is_scaffold_text` so the broad scaffold gate catches it
#: across every code path — including the final-answer validator and
#: the candidate-extractor pre-filter. The structural fix for the
#: 8752c811 regression (assistant-memory head returning
#: ``"turn 1 (user): Give me 100 prompt parameters..."`` verbatim).
_BUNDLE_ROLE_PREFIX_RE = re.compile(
    r"^\s*"
    r"(?:-\s+)?"
    r"(?:turn\s+\d+\s+)?"
    r"\(?(?:user|assistant|system)\)?\s*:",
    re.IGNORECASE,
)


def is_bundle_role_prefix(text: str) -> bool:
    """
    True when ``text`` STARTS with a wiki-bundle role/turn prefix —
    i.e., the text is leaked bundle scaffolding rather than real
    content. See :data:`_BUNDLE_ROLE_PREFIX_RE` for the matched shapes.
    """
    if not text or not text.strip():
        return False
    return bool(_BUNDLE_ROLE_PREFIX_RE.match(text))


#: Generic conversational openers that assistants emit before the
#: actual content of a reply. When the candidate answer IS one of
#: these (with no substantive content after), the head picked the
#: assistant's preamble instead of the assistant's answer. Matches
#: the v8 failure modes: ``"Sure, here are 100 prompt parameters..."``,
#: ``"Of course!"``, ``"Here's the list:"``, ``"Great question!"``.
#: The list is short and high-confidence — extend conservatively, the
#: cost of a false positive is dropping a real (if quirky) answer.
_GENERIC_INTRO_RE = re.compile(
    r"^\s*(?:"
    r"sure(?:,?\s+(?:here(?:'?s)?|here\s+(?:are|is)|let\s+me|i'?ll|"
    r"i\s+can|absolutely|of\s+course))?"
    r"|of\s+course(?:!|,)?"
    r"|absolutely(?:!|,)?"
    r"|certainly(?:!|,)?"
    r"|definitely(?:!|,)?"
    r"|great\s+question(?:!|,)?"
    r"|that'?s\s+a\s+great\s+question"
    r"|here(?:'?s|\s+(?:is|are|you\s+go))"
    r"|here\s+are\s+(?:some|a\s+few|the|\d+)"
    r"|i'?d\s+be\s+happy\s+to"
    r"|i\s+would\s+be\s+happy\s+to"
    r"|let\s+me\s+(?:help|know|see|think|check|share|give|provide)"
    r"|good\s+(?:question|point)"
    r"|happy\s+to\s+help"
    r")\b",
    re.IGNORECASE,
)


def is_generic_intro(text: str) -> bool:
    """
    True when ``text`` is (or is dominated by) a generic conversational
    opener — the assistant's preamble rather than its answer. Used by
    the answer-validation gate to walk past intros and pick the
    underlying content span.

    Logic:
      * If the whole text matches the intro regex with ≤ 3 substantive
        tail tokens, treat as scaffold.
      * If the text *starts* with the intro but has a substantive tail,
        let it through — the caller should strip the intro upstream,
        but we don't want to drop a real answer for being a little
        chatty.
    """
    if not text or not text.strip():
        return False
    m = _GENERIC_INTRO_RE.match(text)
    if not m:
        return False
    tail = text[m.end():].strip()
    # Strip trailing punctuation so "Sure!" with no content is caught.
    tail_clean = re.sub(r"[^\w\s]", "", tail).strip()
    # A bare opener or one with only a tiny tail (≤ 3 word tokens) is
    # scaffold. A longer tail probably is the real answer — let it
    # through; the upstream extractor should have peeled the intro.
    return len(tail_clean.split()) <= 3


def is_user_request_sentence(text: str) -> bool:
    """
    True when the sentence is a user asking the assistant for
    something — a request, not a memory.

    Catches two shapes:
      * **Polite frames** — "Can you help me organize...", "Please
        remind me...". The classic user-question.
      * **Bare imperatives** — "Brainstorm ideas for...", "Give me
        100 prompt parameters...". The v8 assistant-memory failure
        where the user's prompt got returned as the answer.

    The lead-word match catches both. A leading subject pronoun
    ("I brainstorm every morning", "We create lots of variants")
    rescues the sentence — see :data:`_USER_REQUEST_RE` for the
    anchor logic.
    """
    if not text or not text.strip():
        return False
    return bool(_USER_REQUEST_RE.match(text.strip()))


#: Token-Jaccard threshold for :func:`is_prompt_echo`. An answer
#: covering this fraction of the question's content tokens is treated
#: as an echo. 0.7 is conservative — high enough that legitimate
#: short factual answers ("Roscioli", "February 14th") don't trip,
#: low enough to catch the case where the head returned a phrase
#: with the same content words as the question.
_PROMPT_ECHO_THRESHOLD = 0.7

#: Stopwords stripped before the Jaccard check — these are content-
#: free tokens whose presence on both sides isn't a real overlap
#: signal.
_ECHO_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at",
    "for", "from", "with", "by", "as", "is", "are", "was", "were",
    "be", "been", "being", "do", "does", "did", "have", "has", "had",
    "i", "me", "my", "mine", "we", "us", "our", "you", "your",
    "he", "she", "it", "they", "them", "their",
    "what", "when", "where", "who", "whom", "which", "why", "how",
    "that", "this", "these", "those", "any", "some",
})


def is_prompt_echo(answer: str, question: str) -> bool:
    """
    True when ``answer`` is a verbatim or near-verbatim restatement
    of ``question``. Belt-and-suspenders for the case where a head
    returns the user's question as the answer.

    Two checks (cheap-first):
      1. **Substring containment** after normalisation — answer is a
         substring of question or vice versa.
      2. **Content-token Jaccard** — fraction of answer's non-stopword
         tokens that appear in question. Fires when ≥
         :data:`_PROMPT_ECHO_THRESHOLD`.

    Short answers without enough tokens to score reliably (≤ 2
    content tokens) bypass the Jaccard check — they're either bare
    proper nouns ("Roscioli") or numbers ("28. Kg3"), neither of
    which can meaningfully overlap a question.
    """
    if not answer or not question:
        return False
    a_norm = _normalise(answer)
    q_norm = _normalise(question)
    if not a_norm or not q_norm:
        return False
    # 1) Substring containment.
    if a_norm in q_norm or q_norm in a_norm:
        return True
    # 2) Content-token Jaccard.
    a_tokens = set(a_norm.split()) - _ECHO_STOPWORDS
    q_tokens = set(q_norm.split()) - _ECHO_STOPWORDS
    if len(a_tokens) <= 2:
        return False
    if not q_tokens:
        return False
    overlap = a_tokens & q_tokens
    return (len(overlap) / len(a_tokens)) >= _PROMPT_ECHO_THRESHOLD

_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Lowercase, drop apostrophes, collapse separators, strip punctuation."""
    if not text:
        return ""
    out = text.lower().strip()
    # Drop apostrophes (straight and curly) so "don't" → "dont".
    for ap in ("'", "’", "ʼ", "`"):
        out = out.replace(ap, "")
    # Treat underscore / dash as separators ("matched_turn" → "matched turn").
    out = out.replace("_", " ").replace("-", " ")
    out = _PUNCT_RE.sub(" ", out)
    out = _SPACE_RE.sub(" ", out).strip()
    return out


def is_scaffold_line(line: str) -> bool:
    """
    True when a *line* of source text is Markdown scaffold (heading,
    fence, rule). Used by extractors to skip lines they should never
    pull candidates from.
    """
    if not line or not line.strip():
        return True
    return any(pat.match(line) for pat in _HEADING_PATTERNS)


def is_scaffold_text(text: str) -> bool:
    """
    True when ``text`` is a scaffold / prompt-debris string that
    should never become a candidate value or a final answer.

    Three signals (any one fires):

    1. **Exact match** — the normalised string IS a known label
       ("sub answer", "candidates", "matched turn", a bare role
       name like "user").
    2. **Substring match** — the normalised string starts with or
       contains a scaffold token followed by a separator. Catches
       "Sub-answer: " when the LLM emits the prefix without a body.
    3. **Empty after normalisation** — degenerate strings like
       just punctuation or whitespace.
    """
    norm = _normalise(text)
    if not norm:
        return True
    if norm in _SCAFFOLD_EXACT:
        return True
    # Any line that *starts* with a scaffold token and has nothing
    # meaningful after it is scaffold ("Sub-answer:" with no body,
    # "Candidates: " with empty list).
    for tok in _SCAFFOLD_TOKENS:
        if norm == tok or norm.startswith(tok + " "):
            tail = norm[len(tok):].strip()
            if not tail or len(tail) <= 2:
                return True
    # Bundle role/turn prefix — defensive catch for the
    # ``turn 1 (user): ...`` leak. See :data:`_BUNDLE_ROLE_PREFIX_RE`.
    if is_bundle_role_prefix(text):
        return True
    # Generic assistant intro masquerading as an answer (the v8
    # "Sure, here are 100 prompt parameters..." failure mode).
    if is_generic_intro(text):
        return True
    return False


def contains_scaffold_token(text: str) -> bool:
    """
    True when ``text`` *contains* any scaffold token. Stronger than
    :func:`is_scaffold_text` — useful for the candidate extractor's
    pre-filter where we want to skip *any* chunk that's part of the
    wrapper, even when it has additional content.
    """
    norm = _normalise(text)
    if not norm:
        return True
    return any(tok in norm for tok in _SCAFFOLD_TOKENS)


__all__ = [
    "contains_scaffold_token",
    "is_label_metadata_line",
    "is_scaffold_line",
    "is_bundle_role_prefix",
    "is_generic_intro",
    "is_prompt_echo",
    "is_scaffold_text",
    "is_user_request_sentence",
]
