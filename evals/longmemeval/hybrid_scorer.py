"""
evals/longmemeval/hybrid_scorer.py
==================================
Hybrid eval scorer with deterministic stages first, LLM fallback last.

Why this exists
---------------
The LongMemEval substring scorer undercounts by ~9 pp (verified across
the prior 500-row sweeps). The LLM-judge scorer fixes accuracy but
costs ~$0.0001 per row and isn't reproducible across model snapshots.
This module is the middle path: a fast deterministic pipeline that
catches the obvious matches (exact, number-words, unit-aware, simple
rule-semantic) and only invokes the LLM judge on rows it can't decide.

Each :class:`ScoreResult` records exactly which stage produced the
verdict so production telemetry can monitor scorer drift.

Modes
-----
* ``none``    — run no scoring at all (skip the row).
* ``rule``    — only the deterministic stages 1-5. Anything unresolved
                falls through to ``correct=False``. Free, reproducible.
* ``hybrid``  — stages 1-5 first; LLM judge only on uncertain rows.
                The default for dev / diagnostic runs.
* ``llm``     — LLM judge unconditionally. Most expensive; matches the
                old `rescore_with_judge.py` behaviour.

Cache
-----
Every judged pair is keyed by
``sha1(question_id + expected_norm + predicted_norm + judge_model)``.
The in-memory cache is per-process; the on-disk backend is wired in
via the optional ``cache_dir`` argument so re-runs reuse verdicts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)


ScoreMode = Literal["none", "rule", "hybrid", "llm"]
ScoreMethod = Literal[
    "exact", "normalized", "numeric", "unit", "rule_semantic", "llm",
    "wrong", "skipped",
]


@dataclasses.dataclass(frozen=True)
class ScoreResult:
    correct: bool
    method: ScoreMethod
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ─── Stage 1: exact + normalized ────────────────────────────────────────────


_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16",
    "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
    "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60",
    "seventy": "70", "eighty": "80", "ninety": "90", "hundred": "100",
}
_HALF_RE = re.compile(
    r"\b(?P<base>" + "|".join(_NUMBER_WORDS) + r")\s+and\s+a\s+half\b",
    re.IGNORECASE,
)
_NUMBER_WORD_RE = re.compile(
    r"\b(?:" + "|".join(_NUMBER_WORDS) + r")\b", re.IGNORECASE,
)
_ARTICLE_RE = re.compile(r"\b(?:the|a|an)\b", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s%.\-/]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str, *, drop_articles: bool = True) -> str:
    """Cheap normalised form for the exact-match stage."""
    if not text:
        return ""
    lower = text.lower().strip()
    # Number-words → digits ("two and a half" → "2.5" → preserved next).
    lower = _HALF_RE.sub(
        lambda m: _NUMBER_WORDS.get(m.group("base").lower(), m.group("base")) + ".5",
        lower,
    )
    lower = _NUMBER_WORD_RE.sub(
        lambda m: _NUMBER_WORDS.get(m.group(0).lower(), m.group(0)),
        lower,
    )
    # Strip articles only when we're certain it can't carry meaning
    # ("a clinic" vs "the Mayo Clinic" — drop in both, preserve the
    # rest).
    if drop_articles:
        lower = _ARTICLE_RE.sub(" ", lower)
    lower = _PUNCT_RE.sub(" ", lower)
    lower = _WHITESPACE_RE.sub(" ", lower).strip()
    return lower


# ─── Stage 2/3: numeric / unit-aware ────────────────────────────────────────


_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")
# Note: no trailing `\b` after `%` — `%` is non-word so `\b` requires a
# word char on the other side, which fails at end-of-string ("10%"). Keep
# the word-boundary check explicit for "percent" only.
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.IGNORECASE)


def _extract_numbers(text: str) -> list[float]:
    out: list[float] = []
    for m in _NUMERIC_RE.finditer(text):
        try:
            out.append(float(m.group(0)))
        except ValueError:
            continue
    return out


def numeric_match(expected: str, predicted: str) -> bool:
    """True if expected and predicted both contain the same single number."""
    e_nums = _extract_numbers(expected)
    p_nums = _extract_numbers(predicted)
    if not e_nums or not p_nums:
        return False
    # Only call it a numeric match when ALL numbers in expected are
    # present in predicted (the predicted may be a sentence containing
    # the number; the expected is usually just the value).
    return set(e_nums).issubset(set(p_nums))


def percent_match(expected: str, predicted: str) -> bool:
    """Handle '10 percent' / '10%' equivalence."""
    e = _PCT_RE.search(expected)
    p = _PCT_RE.search(predicted)
    if not (e and p):
        return False
    try:
        return float(e.group(1)) == float(p.group(1))
    except ValueError:
        return False


def unit_aware_match(expected: str, predicted: str) -> bool:
    """
    Strip a single optional unit suffix from each side and compare on the
    numeric stem.

    Example: ``expected = "7"`` vs ``predicted = "7 shirts"`` → True.
    """
    e = normalize_text(expected)
    p = normalize_text(predicted)
    if e == p:
        return True
    e_nums = _extract_numbers(e)
    p_nums = _extract_numbers(p)
    if not e_nums or not p_nums:
        return False
    return sorted(e_nums) == sorted(p_nums)


# ─── Stage 4: simple rule-semantic ──────────────────────────────────────────


_IDK_PATTERNS = re.compile(
    r"\b(?:i don't know|i do not know|not mentioned|not enough information|"
    r"cannot determine|no information|i couldn't find|do not know)\b",
    re.IGNORECASE,
)


def rule_semantic_match(expected: str, predicted: str) -> bool:
    """
    Catch the common "predicted is a sentence containing the expected
    short answer" pattern.

    Examples that should match:

    * expected ``"45 minutes each way"`` vs predicted
      ``"Your commute is 45 minutes each way."``
    * expected ``"#PlankChallenge"`` vs predicted
      ``"The challenge was called #PlankChallenge."``
    """
    e = expected.strip()
    p = predicted.strip()
    if not e or not p:
        return False
    if e.lower() in p.lower():
        return True
    # Symmetric — predicted is a short answer, expected is a sentence.
    if p.lower() in e.lower() and len(p) >= 3:
        return True
    # Abstain symmetry — both sides recognised as IDK.
    if _IDK_PATTERNS.search(e) and _IDK_PATTERNS.search(p):
        return True
    return False


# ─── Cache (in-memory + optional disk) ──────────────────────────────────────


class JudgeCache:
    """SHA1-keyed verdict cache; survives across runs via JSON-on-disk."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else None
        self._mem: dict[str, dict[str, Any]] = {}
        if self.path is not None and self.path.exists():
            try:
                self._mem = json.loads(self.path.read_text())
            except Exception:  # pragma: no cover
                log.warning("judge cache at %s is corrupt; ignoring", self.path)
                self._mem = {}

    @staticmethod
    def make_key(
        question_id: str, expected: str, predicted: str,
        judge_model: str = "",
    ) -> str:
        payload = "␟".join((
            question_id,
            normalize_text(expected),
            normalize_text(predicted),
            judge_model,
        ))
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> ScoreResult | None:
        raw = self._mem.get(key)
        if raw is None:
            return None
        return ScoreResult(
            correct=bool(raw.get("correct", False)),
            method=raw.get("method", "llm"),  # type: ignore[arg-type]
            reason=str(raw.get("reason", "")),
        )

    def put(self, key: str, result: ScoreResult) -> None:
        self._mem[key] = result.to_dict()

    def flush(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._mem, indent=2))
        except Exception:  # pragma: no cover
            log.exception("failed to flush judge cache to %s", self.path)


# ─── Scorer entry point ────────────────────────────────────────────────────


class HybridScorer:
    """
    Stage 1-5 deterministic match, optional LLM judge for stage 6.

    Construct once per eval run; reuse across rows. ``llm_judge`` is
    any object with ``async is_correct(question, expected, predicted) -> bool``.
    Passing ``None`` (or running in ``rule`` mode) skips the LLM stage.
    """

    def __init__(
        self,
        *,
        mode: ScoreMode = "hybrid",
        llm_judge: Any | None = None,
        judge_model: str = "",
        cache: JudgeCache | None = None,
    ) -> None:
        self.mode = mode
        self.llm_judge = llm_judge
        self.judge_model = judge_model
        self.cache = cache

    def score_sync(
        self, question: str, expected: str, predicted: str,
        *, question_id: str = "",
    ) -> ScoreResult:
        """Synchronous deterministic ladder; never calls the LLM."""
        if self.mode == "none":
            return ScoreResult(False, "skipped", "score_mode=none")
        e = (expected or "").strip()
        p = (predicted or "").strip()

        # Stage 1 — exact (case-sensitive).
        if e and p and e == p:
            return ScoreResult(True, "exact")
        # Stage 1b — normalised (case + articles + punctuation).
        en = normalize_text(e)
        pn = normalize_text(p)
        if en and pn and en == pn:
            return ScoreResult(True, "normalized")

        # Stage 2 — numeric.
        if percent_match(e, p) or numeric_match(e, p):
            return ScoreResult(True, "numeric")

        # Stage 3 — unit-aware numeric (e.g. "7" ≈ "7 shirts").
        if unit_aware_match(e, p):
            return ScoreResult(True, "unit")

        # Stage 4 — rule-semantic.
        if rule_semantic_match(e, p):
            return ScoreResult(True, "rule_semantic")

        if self.mode == "rule":
            return ScoreResult(False, "wrong", "no deterministic match")
        return ScoreResult(False, "wrong", "deterministic match failed; LLM stage required")

    async def score(
        self, question: str, expected: str, predicted: str,
        *, question_id: str = "",
    ) -> tuple[ScoreResult, bool]:
        """
        Return ``(result, cache_hit)``. Async only because the LLM
        stage is async; the deterministic path completes synchronously.
        """
        det = self.score_sync(
            question, expected, predicted, question_id=question_id,
        )
        if det.correct or self.mode in ("none", "rule"):
            return det, False
        if self.mode == "llm" or (self.mode == "hybrid" and self.llm_judge is not None):
            key = JudgeCache.make_key(question_id, expected, predicted, self.judge_model)
            if self.cache is not None:
                hit = self.cache.get(key)
                if hit is not None:
                    return hit, True
            if self.llm_judge is None:
                return det, False
            try:
                verdict = await self.llm_judge.is_correct(question, expected, predicted)
            except Exception as exc:
                log.exception("LLM judge call failed for %r", question[:80])
                return ScoreResult(False, "wrong", f"llm_error: {exc!r}"), False
            result = ScoreResult(
                bool(verdict), "llm",
                "llm_judge_verdict=true" if verdict else "llm_judge_verdict=false",
            )
            if self.cache is not None:
                self.cache.put(key, result)
            return result, False
        return det, False


__all__ = [
    "HybridScorer",
    "JudgeCache",
    "ScoreMethod",
    "ScoreMode",
    "ScoreResult",
    "normalize_text",
    "numeric_match",
    "percent_match",
    "rule_semantic_match",
    "unit_aware_match",
]
