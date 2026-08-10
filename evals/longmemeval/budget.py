"""
evals/longmemeval/budget.py
===========================
Context-budget accounting for the answer prompt.

Two jobs, both prerequisites for the cost–accuracy ablation in
``docs/LOW_BUDGET_PLAN.md``:

1. **Count tokens for real.** Everything downstream of the budget knob
   used to assume "4 chars per token". Measured on LongMemEval turns
   with ``o200k_base`` the ratio is ~3.9, and it drifts by category
   (dated temporal turns carry more punctuation). A curve whose x-axis
   is an assumed constant is not a measurement, so count with a real
   tokenizer and report the ratio per run.

2. **Enforce the budget on whole items.** The previous enforcement was
   ``"\\n".join(lines)[:max_chars]`` — a byte slice that cuts the last
   turn mid-sentence. At 64 000 chars that is cosmetic; at 8 000 it
   hands the reader a truncated fragment and the ablation would score
   that as "the budget was too small" when the real cause is a
   corrupted tail. :func:`admit_lines` admits whole lines and reports
   what it dropped.

Admission is **stop-at-first-overflow**, not skip-and-continue. Items
arrive in relevance order, so stopping keeps that ordering meaningful
and makes the budget monotone: raising the budget only ever *adds*
items, never swaps one for another. That is what makes two points on
the ablation curve comparable.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

log = logging.getLogger(__name__)

#: Encoding used for token counts. ``o200k_base`` is the GPT-4o/o-series
#: vocabulary — the closest widely-available match to gpt-oss-120b's
#: tokenizer. Exact parity is not required: the ablation compares budget
#: points measured the *same* way, and the per-run chars/token ratio is
#: reported so a different tokenizer can be reconciled after the fact.
_ENCODING_NAME = "o200k_base"

#: Fallback when tiktoken is unavailable. Documented as an estimate and
#: surfaced in the stats, so no one mistakes it for a measurement.
_FALLBACK_CHARS_PER_TOKEN = 3.9

_encoder: Any = None
_encoder_ready = False


def _get_encoder() -> Any:
    """Load the tokenizer once; ``None`` when tiktoken isn't installed."""
    global _encoder, _encoder_ready
    if _encoder_ready:
        return _encoder
    _encoder_ready = True
    try:
        import tiktoken

        _encoder = tiktoken.get_encoding(_ENCODING_NAME)
    except Exception:
        log.warning(
            "tiktoken unavailable — token counts fall back to chars/%.1f "
            "estimates. Install tiktoken before running the budget ablation.",
            _FALLBACK_CHARS_PER_TOKEN,
        )
        _encoder = None
    return _encoder


def tokens_measured() -> bool:
    """True when counts come from a real tokenizer rather than an estimate."""
    return _get_encoder() is not None


def count_tokens(text: str) -> int:
    """Token count for ``text`` — measured when possible, estimated otherwise."""
    if not text:
        return 0
    enc = _get_encoder()
    if enc is None:
        return int(len(text) / _FALLBACK_CHARS_PER_TOKEN)
    return len(enc.encode(text))


@dataclass
class BudgetStats:
    """What the budget actually admitted, for one row."""

    context_chars: int = 0
    context_tokens: int = 0
    items_offered: int = 0
    items_admitted: int = 0
    items_dropped: int = 0
    #: Which limit stopped admission: "chars", "tokens", or "" (nothing did).
    budget_bound_by: str = ""
    max_context_chars: int = 0
    max_context_tokens: int = 0
    #: False when counts are chars/N estimates rather than tokenizer output.
    tokens_measured: bool = True

    @property
    def chars_per_token(self) -> float:
        """Measured ratio for this row — the assumed 4.0 is never used."""
        if self.context_tokens <= 0:
            return 0.0
        return round(self.context_chars / self.context_tokens, 3)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["chars_per_token"] = self.chars_per_token
        return out


def admit_lines(
    lines: list[str],
    *,
    max_chars: int = 0,
    max_tokens: int = 0,
) -> tuple[str, BudgetStats]:
    """
    Join ``lines`` into a context string under both budgets.

    Admits whole lines in order and stops at the first line that would
    breach ``max_chars`` or ``max_tokens`` (either limit ``<= 0`` is
    treated as unbounded). Returns the context plus the
    :class:`BudgetStats` the caller should publish on the row.

    The returned ``context_tokens`` is a single encode of the final
    string, not a sum of per-line counts, so it is exact rather than
    off-by-the-separators.
    """
    stats = BudgetStats(
        items_offered=len(lines),
        max_context_chars=max(0, int(max_chars)),
        max_context_tokens=max(0, int(max_tokens)),
        tokens_measured=tokens_measured(),
    )

    char_cap = stats.max_context_chars or None
    token_cap = stats.max_context_tokens or None

    admitted: list[str] = []
    used_chars = 0
    used_tokens = 0

    for line in lines:
        # +1 for the "\n" this line costs once something precedes it.
        sep = 1 if admitted else 0
        line_chars = len(line) + sep
        if char_cap is not None and used_chars + line_chars > char_cap:
            stats.budget_bound_by = "chars"
            break
        line_tokens = count_tokens(line) + sep if token_cap is not None else 0
        if token_cap is not None and used_tokens + line_tokens > token_cap:
            stats.budget_bound_by = "tokens"
            break
        admitted.append(line)
        used_chars += line_chars
        used_tokens += line_tokens

    context = "\n".join(admitted)
    stats.items_admitted = len(admitted)
    stats.items_dropped = len(lines) - len(admitted)
    stats.context_chars = len(context)
    stats.context_tokens = count_tokens(context)
    return context, stats


__all__ = [
    "BudgetStats",
    "admit_lines",
    "count_tokens",
    "tokens_measured",
]
