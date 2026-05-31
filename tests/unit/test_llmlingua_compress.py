"""
tests/unit/test_llmlingua_compress.py
=====================================
Unit tests for
:class:`continuum.optimizer.strategies.llmlingua.LLMLinguaCompress`.

LLMLingua-2 is mocked via an injectable fake compressor — these tests
never load the real ~500 MB model. The fake returns a deterministic
substring of the input so we can assert on byte-level expectations.

Covers
------
* Protocol satisfaction + Pydantic config validation
* Skip logic when joined LTM < min_input_tokens
* No-op when no LTM rows at all
* Compression reduces tokens toward the target ratio
* Cache hit on identical input; misses on different ratio
* Quality gate rejects low-similarity outputs
* Quality gate is no-op when no embedder wired
* Compressor failures surface as a no-op (chain robustness)
* `_split_back` aligns when separator survives; fuses otherwise
* `_extract_compressed_text` handles dict/string return shapes
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from pydantic import ValidationError

from continuum.core.types import (
    ContextBundle,
    MemoryItem,
    MemoryTier,
    TokenBudget,
)
from continuum.optimizer.base import estimate_tokens_text
from continuum.optimizer.protocol import Optimizer
from continuum.optimizer.strategies.llmlingua import (
    LLMLinguaCompress,
    LLMLinguaConfig,
    _extract_compressed_text,
    _split_back,
)

_BASE = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)


def _budget() -> TokenBudget:
    return TokenBudget(
        total=8_000,
        stm_reserved=500,
        mtm_reserved=2_000,
        ltm_reserved=2_000,
        response_reserved=500,
    )


def _ltm(content: str, *, offset_seconds: int = 0) -> MemoryItem:
    return MemoryItem(
        id=str(uuid.uuid4()),
        content=content,
        tier=MemoryTier.LTM,
        created_at=_BASE + timedelta(seconds=offset_seconds),
    )


def _bundle(items: list[MemoryItem]) -> ContextBundle:
    return ContextBundle(
        items=items,
        messages=[{"role": "system", "content": i.content} for i in items],
        tokens_used=sum(estimate_tokens_text(i.content) for i in items),
        budget=_budget(),
    )


# A long-ish LTM corpus — comfortably over the 500-token default floor.
_LONG_LTM = [
    (
        "Acme Corp launched their flagship analytics dashboard in Q1 2025. "
        "The dashboard ships with real-time retention metrics, cohort analysis, "
        "and a customisable funnel visualiser. Early enterprise customers "
        "reported a 30% reduction in time-to-insight during the beta period."
    ),
    (
        "Engineering completed a migration from Postgres 14 to Postgres 16 in "
        "early February. The team observed a 40% reduction in p95 query "
        "latency post-migration. JIT compilation is scheduled to be enabled "
        "in the next sprint after a final round of regression testing."
    ),
    (
        "Sales closed three enterprise deals totalling $1.2M in annual "
        "recurring revenue last quarter. The largest of the three deals "
        "includes a multi-year support contract with extended SLA coverage. "
        "Customer success is preparing onboarding materials for next month."
    ),
    (
        "The hiring committee approved two senior engineering offers in "
        "March. Both candidates accepted and will begin in early April. The "
        "engineering organisation expects to ramp them via the standard "
        "30-60-90 onboarding programme."
    ),
    (
        "Product roadmap for the second half of the year prioritises an "
        "analytics dashboard refresh and a brand-new audit log feature. The "
        "audit log requirement surfaced during three separate enterprise "
        "QBR discussions over the past quarter."
    ),
]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeCompressor:
    """
    Deterministic stand-in for ``llmlingua.PromptCompressor``.

    Returns a string with roughly *rate* fraction of the input's words
    so the per-token ratio in our tests is predictable.
    """

    def __init__(self) -> None:
        self.calls = 0

    def compress_prompt(self, text: str, *, rate: float, force_tokens: list[str]) -> dict[str, str]:
        self.calls += 1
        words = text.split()
        keep = max(1, int(len(words) * rate))
        return {"compressed_prompt": " ".join(words[:keep])}


class _SeparatorPreservingCompressor:
    """Like _FakeCompressor but keeps the inter-item separator intact."""

    def __init__(self, sep: str = " ⟦SEP⟧ ") -> None:
        self.sep = sep
        self.calls = 0

    def compress_prompt(self, text: str, *, rate: float, force_tokens: list[str]) -> dict[str, str]:
        self.calls += 1
        pieces = text.split(self.sep)
        compressed_pieces = []
        for p in pieces:
            words = p.split()
            keep = max(1, int(len(words) * rate))
            compressed_pieces.append(" ".join(words[:keep]))
        return {"compressed_prompt": self.sep.join(compressed_pieces)}


class _FakeEmbedder:
    """Maps text → a hashed deterministic 16-dim vector."""

    def __init__(self, fixed: float | None = None) -> None:
        self.fixed = fixed
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        if self.fixed is not None:
            return [self.fixed] * 16
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.normal(size=16).astype(np.float32)
        return v.tolist()


# ---------------------------------------------------------------------------
# Protocol + config
# ---------------------------------------------------------------------------


def test_protocol_satisfaction() -> None:
    assert isinstance(LLMLinguaCompress(), Optimizer)


def test_config_defaults() -> None:
    cfg = LLMLinguaConfig()
    assert cfg.ratio == 0.5
    assert cfg.min_input_tokens == 500
    assert cfg.quality_threshold == 0.85


def test_config_rejects_ratio_outside_zero_one() -> None:
    with pytest.raises(ValidationError):
        LLMLinguaConfig(ratio=0.0)
    with pytest.raises(ValidationError):
        LLMLinguaConfig(ratio=1.0)


# ---------------------------------------------------------------------------
# Skip / no-op paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skip_short_input() -> None:
    """Short LTM (< min_input_tokens) → returned unchanged, compressor untouched."""
    comp = _FakeCompressor()
    items = [_ltm("brief fact")]  # tiny
    out = await LLMLinguaCompress(
        compressor=comp,
        min_input_tokens=100,
    ).apply(_bundle(items), _budget())
    assert out is _bundle(items) or out.items == items
    assert comp.calls == 0


@pytest.mark.asyncio
async def test_no_ltm_at_all_is_noop() -> None:
    comp = _FakeCompressor()
    ctx = _bundle([])
    out = await LLMLinguaCompress(compressor=comp).apply(ctx, _budget())
    assert out is ctx
    assert comp.calls == 0


@pytest.mark.asyncio
async def test_compressor_failure_returns_bundle_unchanged() -> None:
    class _Broken:
        def compress_prompt(self, *_a: object, **_k: object) -> dict[str, str]:
            raise RuntimeError("model crashed")

    items = [_ltm(t, offset_seconds=i) for i, t in enumerate(_LONG_LTM)]
    ctx = _bundle(items)
    out = await LLMLinguaCompress(compressor=_Broken()).apply(ctx, _budget())
    # Same item set survives.
    assert [it.id for it in out.items] == [it.id for it in items]


# ---------------------------------------------------------------------------
# Compression behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compression_reduces_tokens_toward_ratio() -> None:
    items = [_ltm(t, offset_seconds=i) for i, t in enumerate(_LONG_LTM)]
    ctx = _bundle(items)
    before = sum(estimate_tokens_text(it.content) for it in ctx.items)

    out = await LLMLinguaCompress(
        compressor=_SeparatorPreservingCompressor(),
        ratio=0.4,
        min_input_tokens=50,
    ).apply(ctx, _budget())

    after = sum(estimate_tokens_text(it.content) for it in out.items)
    # Fake compressor keeps ~40% of words per item; tiktoken counts roughly
    # 1:1 with words for English prose. Loose [0.30, 0.60] envelope.
    reduction = after / before
    assert 0.30 <= reduction <= 0.60, (
        f"unexpected ratio {reduction:.2f}; fake compressor targeted 0.40"
    )
    # Item count preserved (separator was kept).
    assert len(out.items) == len(items)
    # Debug info recorded.
    info = out.debug_info["llmlingua"]
    assert info["target_ratio"] == 0.4
    assert info["output_tokens"] < info["input_tokens"]


@pytest.mark.asyncio
async def test_fallback_fuses_to_single_item_when_separator_lost() -> None:
    """When the compressor drops the separator, LTM collapses to one row."""
    items = [_ltm(t, offset_seconds=i) for i, t in enumerate(_LONG_LTM)]
    ctx = _bundle(items)
    out = await LLMLinguaCompress(
        compressor=_FakeCompressor(),  # this fake drops the separator
        ratio=0.5,
        min_input_tokens=50,
    ).apply(ctx, _budget())

    ltm_after = [it for it in out.items if it.tier == MemoryTier.LTM]
    assert len(ltm_after) == 1
    assert ltm_after[0].metadata["kind"] == "llmlingua_compressed"
    assert ltm_after[0].metadata["compacted_count"] == len(items)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_hits_on_repeat_input() -> None:
    comp = _SeparatorPreservingCompressor()
    strategy = LLMLinguaCompress(
        compressor=comp,
        ratio=0.5,
        min_input_tokens=50,
    )
    items = [_ltm(t, offset_seconds=i) for i, t in enumerate(_LONG_LTM)]

    first = await strategy.apply(_bundle(items), _budget())
    assert comp.calls == 1
    assert first.debug_info["llmlingua"]["cache_hit"] is False

    second = await strategy.apply(_bundle(items), _budget())
    # No additional compressor invocation.
    assert comp.calls == 1
    assert second.debug_info["llmlingua"]["cache_hit"] is True


@pytest.mark.asyncio
async def test_cache_misses_when_ratio_changes() -> None:
    comp = _SeparatorPreservingCompressor()
    items = [_ltm(t, offset_seconds=i) for i, t in enumerate(_LONG_LTM)]

    s1 = LLMLinguaCompress(compressor=comp, ratio=0.3, min_input_tokens=50)
    s2 = LLMLinguaCompress(compressor=comp, ratio=0.5, min_input_tokens=50)
    await s1.apply(_bundle(items), _budget())
    await s2.apply(_bundle(items), _budget())
    # Different strategy instances → different caches anyway, but the
    # cache key salts on ratio so even reuse would miss.
    assert comp.calls == 2


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quality_gate_rejects_low_similarity() -> None:
    """Embedder returns orthogonal vectors → cosine ~ 0 → reject."""

    class _Orthogonal:
        async def embed(self, text: str) -> list[float]:
            # Different one-hot per call.
            if "compressed_prompt" in text or "Acme" not in text:
                # branch never used; embed only ever gets the strings
                # passed by the strategy, and we just want two distinct
                # one-hots
                return [0.0, 1.0] + [0.0] * 14
            return [1.0, 0.0] + [0.0] * 14

    items = [_ltm(t, offset_seconds=i) for i, t in enumerate(_LONG_LTM)]
    out = await LLMLinguaCompress(
        compressor=_SeparatorPreservingCompressor(),
        embedder=_Orthogonal(),
        quality_threshold=0.9,
        ratio=0.4,
        min_input_tokens=50,
    ).apply(_bundle(items), _budget())

    # Quality gate rejected → original items survive intact.
    assert [it.id for it in out.items] == [it.id for it in items]


@pytest.mark.asyncio
async def test_quality_gate_accepts_high_similarity() -> None:
    """Same vector for both inputs → cosine = 1 → accept."""
    embedder = _FakeEmbedder(fixed=0.5)
    items = [_ltm(t, offset_seconds=i) for i, t in enumerate(_LONG_LTM)]

    out = await LLMLinguaCompress(
        compressor=_SeparatorPreservingCompressor(),
        embedder=embedder,
        quality_threshold=0.9,
        ratio=0.4,
        min_input_tokens=50,
    ).apply(_bundle(items), _budget())

    # Identical embedding vectors → cosine = 1 ≥ 0.9 → compression applied.
    assert embedder.calls == 2  # original + compressed
    after = sum(estimate_tokens_text(it.content) for it in out.items)
    before = sum(estimate_tokens_text(it.content) for it in items)
    assert after < before


@pytest.mark.asyncio
async def test_quality_gate_skipped_when_no_embedder() -> None:
    items = [_ltm(t, offset_seconds=i) for i, t in enumerate(_LONG_LTM)]
    out = await LLMLinguaCompress(
        compressor=_SeparatorPreservingCompressor(),
        embedder=None,
        ratio=0.5,
        min_input_tokens=50,
    ).apply(_bundle(items), _budget())
    after = sum(estimate_tokens_text(it.content) for it in out.items)
    before = sum(estimate_tokens_text(it.content) for it in items)
    assert after < before


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def test_extract_compressed_text_dict() -> None:
    assert _extract_compressed_text({"compressed_prompt": "x"}) == "x"
    assert _extract_compressed_text({"output": "y"}) == "y"
    assert _extract_compressed_text({}) == ""


def test_extract_compressed_text_string() -> None:
    assert _extract_compressed_text("hello") == "hello"


def test_extract_compressed_text_unknown_shape() -> None:
    assert _extract_compressed_text(["nope"]) == ""


def test_split_back_aligns_when_separator_preserved() -> None:
    sep = " ⟦SEP⟧ "
    originals = [_ltm("foo"), _ltm("bar"), _ltm("baz")]
    out = _split_back(f"alpha{sep}beta{sep}gamma", originals, separator=sep)
    assert [it.content for it in out] == ["alpha", "beta", "gamma"]


def test_split_back_fuses_when_separator_dropped() -> None:
    originals = [_ltm("foo"), _ltm("bar")]
    out = _split_back("alpha beta", originals, separator=" ⟦SEP⟧ ")
    assert len(out) == 1
    assert out[0].metadata["kind"] == "llmlingua_compressed"
