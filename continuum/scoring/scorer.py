"""
continuum/scoring/scorer.py
===========================
``Scorer`` — the composite memory-scoring formula (``ScorerProtocol``).

    score = w_rel·relevance + w_imp·importance + w_rec·recency + w_conf·confidence
    final = score · layer_boost[tier]

Components (each normalised to [0, 1] before weighting)
-------------------------------------------------------
* **relevance**  — cosine(query.embedding, item.embedding), clamped to
  [0, 1] (negative cosine → 0; a missing embedding → 0).
* **importance** — ``item.importance`` (set at write time).
* **confidence** — ``item.confidence``.
* **recency**    — reinforcement-aware exponential decay (Ebbinghaus):

      ref         = item.last_access or item.created_at
      age_hours   = (now - ref) / 3600
      base        = exp(-age_hours / tau_hours)
      strength    = 1 + log1p(item.access_count)   # recall strengthens memory
      recency     = min(1.0, base * strength)

Layer boost
-----------
A multiplicative tweak keyed by ``MemoryItem.tier.name``
(``config.layer_boost``, default STM 1.05 / MTM 1.0 / LTM 1.1). Because the
boost is multiplicative and LTM > 1.0, ``score()`` may exceed 1.0 for a
maximally-relevant LTM item (e.g. 1.0 × 1.1) — that is the spec's explicit
formula. :meth:`breakdown` exposes the *un-boosted* weighted composite
(clamped to [0, 1], per ``ScoreBreakdown`` semantics) so callers can inspect
the raw dimensions.

Configuration & validation
--------------------------
Weights come from :class:`continuum.core.config.ScoringConfig.weights`
(``ScoringWeights``), which **enforces the four weights sum to 1.0** via its
own pydantic ``model_validator`` — so "weights must sum to 1.0" is validated
at config-construction time, before a ``Scorer`` is ever built.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from continuum.core.config import ScoringConfig
from continuum.core.types import MemoryItem, Query, ScoreBreakdown


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _cosine(a: list[float] | None, b: list[float] | None) -> float:
    """Cosine similarity in [-1, 1]; 0.0 if either vector is missing/zero."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class Scorer:
    """
    Composite scorer. Stateless given its config — one instance is safe to
    share across queries and threads.
    """

    def __init__(self, config: ScoringConfig | None = None) -> None:
        # Constructing ScoringConfig() runs ScoringWeights' sum-to-1.0
        # validator, so an invalid weight set fails fast here.
        self.config = config or ScoringConfig()

    # ── ScorerProtocol ──────────────────────────────────────────────────────

    def score(
        self,
        item: MemoryItem,
        query: Query,
        now: datetime | None = None,
    ) -> float:
        """Composite score × per-tier boost (see module docstring)."""
        composite = self.breakdown(item, query, now).composite
        boost = self.config.layer_boost.get(item.tier.name, 1.0)
        return composite * boost

    def breakdown(
        self,
        item: MemoryItem,
        query: Query,
        now: datetime | None = None,
    ) -> ScoreBreakdown:
        """
        Per-dimension scores. ``composite`` is the **un-boosted** weighted
        sum (using ``config.weights``, not the canonical constants) clamped
        to [0, 1]; ``score()`` applies the layer boost on top.
        """
        ref = now or datetime.now(UTC)
        w = self.config.weights
        relevance = _clamp01(_cosine(query.embedding, item.embedding))
        importance = _clamp01(item.importance)
        confidence = _clamp01(item.confidence)
        recency = self._recency(item, ref)
        composite = _clamp01(
            w.rel * relevance + w.imp * importance + w.rec * recency + w.conf * confidence
        )
        return ScoreBreakdown(
            relevance=relevance,
            importance=importance,
            recency=recency,
            confidence=confidence,
            composite=composite,
        )

    # ── recency (reinforcement-aware exponential decay) ─────────────────────

    def _recency(self, item: MemoryItem, now: datetime) -> float:
        ref = self._recency_basis(item)
        now, ref = _as_utc(now), _as_utc(ref)
        age_hours = max(0.0, (now - ref).total_seconds() / 3600.0)
        base = math.exp(-age_hours / self.config.tau_hours)
        strength = 1.0 + math.log1p(max(0, item.access_count))
        return min(1.0, base * strength)

    @staticmethod
    def _recency_basis(item: MemoryItem) -> datetime:
        """**Transaction** time — how long since we learned or last used this.

        This deliberately does *not* use valid time. Basing decay on
        ``valid_from`` conflates "this describes an old event" with "this
        memory is stale", and the two are not the same: a fact you learned
        yesterday about last January is fresh knowledge. Measured cost of
        getting that wrong — dating a fact buried it:

            query "where do I live now"
            relevance 1.000, recency 0.000 → "I live in Boston."      (rank 5)
            relevance 0.914, recency 0.005 → "I moved … to New York"  (rank 7)
            relevance 0.830, recency 0.999 → "I play the guitar…"     (rank 1)

        The dated facts were the only ones carrying ``valid_from``, so the
        0.20 recency weight outweighed their relevance lead and the undated
        noise won. Stating *when* a fact became true made it harder to recall —
        exactly backwards.

        Choosing between competing versions of the same fact is supersession's
        job, not decay's: ``valid_to`` retires the old version, and
        :meth:`continuum.Memory.current` resolves the live one through an exact
        valid-time lookup. Valid time still drives ``current``, ``timeline`` and
        ``as_of`` — it just no longer distorts generic recall.
        """
        return item.last_access or item.created_at


def _as_utc(dt: datetime) -> datetime:
    """Make *dt* aware (assuming UTC) so mixed naive/aware inputs cannot raise."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


__all__ = ["Scorer"]
