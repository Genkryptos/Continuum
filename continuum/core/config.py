"""
continuum/core/config.py
========================
Pydantic v2 configuration system for the Continuum memory framework.

Priority order (highest → lowest)
----------------------------------
1. Programmatic  — pass a ``ContinuumConfig`` (or sub-config) object directly
2. Environment   — ``CONTINUUM_*`` env vars (see each field's ``alias``)
3. YAML file     — ``continuum.yaml`` in the working directory (or override
                   with ``CONTINUUM_CONFIG_FILE`` env var)
4. Defaults      — the values declared on every field

Loading
-------
    # Use all defaults / env vars / continuum.yaml if present:
    cfg = ContinuumConfig.load()

    # Override config file path:
    cfg = ContinuumConfig.load(config_file="path/to/my.yaml")

    # Construct programmatically (no file I/O):
    cfg = ContinuumConfig(
        database=DatabaseConfig(dsn="postgresql://..."),
        scoring=ScoringConfig(weights=ScoringWeights(rel=0.5, imp=0.2, rec=0.2, conf=0.1)),
    )

    # Access a sub-config:
    cfg.database.pool_size
    cfg.scoring.weights.rel

Env-var mapping (sample)
------------------------
    CONTINUUM_DB_DSN          → database.dsn
    CONTINUUM_DB_POOL_SIZE    → database.pool_size
    CONTINUUM_EMBED_MODEL     → embedding.model_name
    CONTINUUM_SCORE_TAU_HOURS → scoring.tau_hours
    CONTINUUM_CONFIG_FILE     → path to the YAML config file
    (full list below, one per field)

YAML schema
-----------
    See ``continuum.yaml.example`` next to this file, or call
    ``ContinuumConfig.yaml_schema()`` to get an annotated template.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    AliasChoices,
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# ---------------------------------------------------------------------------
# Sentinel for "not set" — distinguishable from None
# ---------------------------------------------------------------------------
_UNSET = object()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FLOAT_TOL = 1e-6  # tolerance for the weights-sum-to-1 check


def _abs_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


# =============================================================================
# DatabaseConfig
# =============================================================================


class DatabaseConfig(BaseSettings):
    """
    PostgreSQL connection settings.

    Env vars
    --------
    CONTINUUM_DB_DSN        postgresql://user:pass@host:port/dbname
    CONTINUUM_DB_POOL_SIZE  integer ≥ 1  (default 5)
    CONTINUUM_DB_TIMEOUT    float seconds (default 30.0)
    """

    model_config = SettingsConfigDict(
        env_prefix="CONTINUUM_DB_",
        populate_by_name=True,
        extra="ignore",
    )

    dsn: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/continuum",
        validation_alias=AliasChoices("dsn", "CONTINUUM_DB_DSN"),
        description="Full PostgreSQL DSN including credentials and database name.",
    )
    pool_size: int = Field(
        default=5,
        ge=1,
        le=100,
        validation_alias=AliasChoices("pool_size", "CONTINUUM_DB_POOL_SIZE"),
        description="Number of persistent connections in the psycopg2 pool (1–100).",
    )
    timeout: float = Field(
        default=30.0,
        gt=0,
        validation_alias=AliasChoices("timeout", "CONTINUUM_DB_TIMEOUT"),
        description="Query / connection timeout in seconds.",
    )

    @field_validator("dsn")
    @classmethod
    def dsn_must_be_postgres(cls, v: str) -> str:
        """Reject non-Postgres DSNs early so errors surface at startup."""
        v = v.strip()
        if not (v.startswith("postgresql://") or v.startswith("postgres://")):
            raise ValueError(f"dsn must start with 'postgresql://' or 'postgres://', got: {v!r}")
        return v

    @field_validator("pool_size")
    @classmethod
    def pool_size_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"pool_size must be ≥ 1, got {v}")
        return v


# =============================================================================
# EmbeddingConfig
# =============================================================================


class EmbeddingConfig(BaseSettings):
    """
    Embedding model settings.

    Env vars
    --------
    CONTINUUM_EMBED_MODEL          model name or HuggingFace path
    CONTINUUM_EMBED_BATCH_SIZE     integer (default 64)
    CONTINUUM_EMBED_CACHE_ENABLED  1/true/yes (default true)
    CONTINUUM_EMBED_DIM            embedding dimension (default 1024)
    """

    model_config = SettingsConfigDict(
        env_prefix="CONTINUUM_EMBED_",
        populate_by_name=True,
        extra="ignore",
    )

    model_name: str = Field(
        default="BAAI/bge-m3",
        validation_alias=AliasChoices("model_name", "CONTINUUM_EMBED_MODEL"),
        description=(
            "HuggingFace model id or local path. "
            "Must produce embeddings whose dimension matches the pgvector column. "
            "Default: 'BAAI/bge-m3' (1024-dim, multilingual, MTEB SOTA)."
        ),
    )
    batch_size: int = Field(
        default=64,
        ge=1,
        le=2048,
        validation_alias=AliasChoices("batch_size", "CONTINUUM_EMBED_BATCH_SIZE"),
        description="Number of texts to embed per forward pass (1–2048).",
    )
    cache_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("cache_enabled", "CONTINUUM_EMBED_CACHE_ENABLED"),
        description=(
            "When True, identical texts are served from an LRU cache "
            "instead of re-running the model."
        ),
    )
    dim: int = Field(
        default=1024,
        ge=1,
        validation_alias=AliasChoices("dim", "CONTINUUM_EMBED_DIM"),
        description=(
            "Expected embedding dimensionality. "
            "Must match the pgvector column (memory_nodes.embedding). "
            "Used for validation; does not force the model to change its output."
        ),
    )
    cache_size: int = Field(
        default=10_000,
        ge=1,
        validation_alias=AliasChoices("cache_size", "CONTINUUM_EMBED_CACHE_SIZE"),
        description="Max entries in the in-memory LRU embedding cache.",
    )
    redis_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("redis_url", "CONTINUUM_EMBED_REDIS_URL"),
        description=(
            "Optional redis://… URL for a shared L2 embedding cache. "
            "When unset, only the in-process LRU cache is used."
        ),
    )
    device: Literal["auto", "cuda", "cpu"] = Field(
        default="auto",
        validation_alias=AliasChoices("device", "CONTINUUM_EMBED_DEVICE"),
        description=(
            "Inference device. 'auto' picks cuda when available, else cpu. "
            "On CUDA OOM the service transparently falls back to cpu."
        ),
    )


# =============================================================================
# ExtractionConfig  (GLiNER entity/relation extraction)
# =============================================================================


class ExtractionConfig(BaseSettings):
    """
    Deterministic entity/relation extraction (GLiNER baseline).

    Env vars
    --------
    CONTINUUM_EXTRACT_MODEL       GLiNER model id (default urchade/gliner_multi-v2.1)
    CONTINUUM_EXTRACT_THRESHOLD   float 0–1 (default 0.7)
    CONTINUUM_EXTRACT_DEVICE      auto | cuda | cpu (default auto)
    """

    model_config = SettingsConfigDict(
        env_prefix="CONTINUUM_EXTRACT_",
        populate_by_name=True,
        extra="ignore",
    )

    gliner_model: str = Field(
        default="urchade/gliner_multi-v2.1",
        validation_alias=AliasChoices("gliner_model", "CONTINUUM_EXTRACT_MODEL"),
        description=(
            "HuggingFace GLiNER model id. Default 'urchade/gliner_multi-v2.1' "
            "(multilingual, zero-shot span extraction)."
        ),
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices(
            "confidence_threshold", "CONTINUUM_EXTRACT_THRESHOLD"
        ),
        description="Minimum GLiNER span score to keep an entity (0–1).",
    )
    device: Literal["auto", "cuda", "cpu"] = Field(
        default="auto",
        validation_alias=AliasChoices("device", "CONTINUUM_EXTRACT_DEVICE"),
        description="Inference device. 'auto' picks cuda when available, else cpu.",
    )


# =============================================================================
# LLMExtractionConfig  (LLM fallback/enhancement over GLiNER)
# =============================================================================


class LLMExtractionConfig(BaseSettings):
    """
    LLM-based entity/relation extraction (enhances the GLiNER baseline).

    Env vars
    --------
    CONTINUUM_LLMX_MODEL          litellm model id (default gpt-4o-mini)
    CONTINUUM_LLMX_TEMPERATURE    float (default 0.0 — deterministic)
    CONTINUUM_LLMX_MAX_TOKENS     int   (default 1000)
    CONTINUUM_LLMX_TIMEOUT        seconds per attempt (default 30)
    """

    model_config = SettingsConfigDict(
        env_prefix="CONTINUUM_LLMX_",
        populate_by_name=True,
        extra="ignore",
    )

    model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("model", "CONTINUUM_LLMX_MODEL"),
        description=(
            "litellm model id. Default 'gpt-4o-mini' — the smallest capable "
            "model; 'claude-3-haiku-20240307' is an equivalent alternative."
        ),
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        validation_alias=AliasChoices("temperature", "CONTINUUM_LLMX_TEMPERATURE"),
        description="Sampling temperature; 0.0 keeps extraction deterministic.",
    )
    max_tokens: int = Field(
        default=1000,
        ge=1,
        le=8192,
        validation_alias=AliasChoices("max_tokens", "CONTINUUM_LLMX_MAX_TOKENS"),
        description="Hard cap on completion tokens (cost control).",
    )
    timeout: int = Field(
        default=30,
        ge=1,
        validation_alias=AliasChoices("timeout", "CONTINUUM_LLMX_TIMEOUT"),
        description="Per-attempt LLM call timeout, in seconds.",
    )


# =============================================================================
# FactExtractionConfig  (atomic facts from MTM summary blocks)
# =============================================================================


class FactExtractionConfig(BaseSettings):
    """
    LLM atomic-fact extraction from MTM ``SummaryBlock``s.

    Env vars
    --------
    CONTINUUM_FACT_MODEL          litellm model id (default gpt-4o-mini)
    CONTINUUM_FACT_TEMPERATURE    float (default 0.0)
    CONTINUUM_FACT_MAX_TOKENS     int   (default 1000)
    CONTINUUM_FACT_TIMEOUT        seconds per attempt (default 30)
    CONTINUUM_FACT_MIN_CONF       float (default 0.6)
    CONTINUUM_FACT_MIN_LEN        int   (default 10)
    CONTINUUM_FACT_MAX_LEN        int   (default 500)
    """

    model_config = SettingsConfigDict(
        env_prefix="CONTINUUM_FACT_",
        populate_by_name=True,
        extra="ignore",
    )

    model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("model", "CONTINUUM_FACT_MODEL"),
        description="litellm model id (smallest capable; gpt-4o-mini default).",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        validation_alias=AliasChoices("temperature", "CONTINUUM_FACT_TEMPERATURE"),
        description="0.0 keeps fact extraction deterministic.",
    )
    max_tokens: int = Field(
        default=1000,
        ge=1,
        le=8192,
        validation_alias=AliasChoices("max_tokens", "CONTINUUM_FACT_MAX_TOKENS"),
        description="Hard cap on completion tokens (cost control).",
    )
    timeout: int = Field(
        default=30,
        ge=1,
        validation_alias=AliasChoices("timeout", "CONTINUUM_FACT_TIMEOUT"),
        description="Per-attempt LLM call timeout, in seconds.",
    )
    min_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("min_confidence", "CONTINUUM_FACT_MIN_CONF"),
        description="Drop facts the model is less than this sure of.",
    )
    min_fact_len: int = Field(
        default=10,
        ge=1,
        validation_alias=AliasChoices("min_fact_len", "CONTINUUM_FACT_MIN_LEN"),
        description="Reject trivially short 'facts' (chars).",
    )
    max_fact_len: int = Field(
        default=500,
        ge=1,
        validation_alias=AliasChoices("max_fact_len", "CONTINUUM_FACT_MAX_LEN"),
        description=(
            "Reject over-long facts — a long fact is almost never atomic, "
            "so the cap doubles as an atomicity guard."
        ),
    )


# =============================================================================
# TriggerConfig  (automatic promotion triggers)
# =============================================================================


class TriggerConfig(BaseSettings):
    """
    When the Promoter should fire automatically.

    Env vars
    --------
    CONTINUUM_TRIGGER_ON_NEW_ENTITY   bool (default true)
    CONTINUUM_TRIGGER_BLOCK_THRESHOLD int  (default 20)
    CONTINUUM_TRIGGER_PERIOD_SECONDS  int  (default 21600 = 6 h)
    """

    model_config = SettingsConfigDict(
        env_prefix="CONTINUUM_TRIGGER_",
        populate_by_name=True,
        extra="ignore",
    )

    on_new_entity: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "on_new_entity", "CONTINUUM_TRIGGER_ON_NEW_ENTITY"
        ),
        description="Promote eagerly when a turn mentions an entity not in LTM.",
    )
    block_threshold: int = Field(
        default=20,
        ge=1,
        validation_alias=AliasChoices(
            "block_threshold", "CONTINUUM_TRIGGER_BLOCK_THRESHOLD"
        ),
        description="Promote once this many unprocessed MTM blocks accumulate.",
    )
    periodic_interval_seconds: int = Field(
        default=21_600,  # 6 hours
        ge=1,
        validation_alias=AliasChoices(
            "periodic_interval_seconds", "CONTINUUM_TRIGGER_PERIOD_SECONDS"
        ),
        description="Background sweep cadence in seconds (default 6 hours).",
    )


# =============================================================================
# RerankerConfig  (BGE cross-encoder second-pass)
# =============================================================================


class RerankerConfig(BaseSettings):
    """
    Cross-encoder reranking settings.

    Env vars
    --------
    CONTINUUM_RERANK_MODEL          HF model id (default BAAI/bge-reranker-v2-m3)
    CONTINUUM_RERANK_BATCH_SIZE     int  (default 32)
    CONTINUUM_RERANK_DEVICE         auto | cuda | cpu (default auto)
    CONTINUUM_RERANK_TOP_N          int  (default 50)
    CONTINUUM_RERANK_SKIP_BELOW     int  (default 10)
    """

    model_config = SettingsConfigDict(
        env_prefix="CONTINUUM_RERANK_",
        populate_by_name=True,
        extra="ignore",
    )

    model_name: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        validation_alias=AliasChoices("model_name", "CONTINUUM_RERANK_MODEL"),
        description="HuggingFace cross-encoder id (default bge-reranker-v2-m3).",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        validation_alias=AliasChoices(
            "batch_size", "CONTINUUM_RERANK_BATCH_SIZE"
        ),
        description="(query, doc) pairs scored per cross-encoder forward pass.",
    )
    device: Literal["auto", "cuda", "cpu"] = Field(
        default="auto",
        validation_alias=AliasChoices("device", "CONTINUUM_RERANK_DEVICE"),
        description="'auto' picks cuda when available, else cpu.",
    )
    rerank_top_n: int = Field(
        default=50,
        ge=1,
        validation_alias=AliasChoices("rerank_top_n", "CONTINUUM_RERANK_TOP_N"),
        description=(
            "Only the first N recalled items are reranked; the remainder "
            "are appended after, in their original order."
        ),
    )
    skip_if_fewer_than: int = Field(
        default=10,
        ge=0,
        validation_alias=AliasChoices(
            "skip_if_fewer_than", "CONTINUUM_RERANK_SKIP_BELOW"
        ),
        description=(
            "Below this many items, skip reranking entirely — the cross-"
            "encoder latency is not worth it for a tiny set."
        ),
    )


# =============================================================================
# PolicyEngineConfig  (policy-based memory lifecycle)
# =============================================================================


class PolicyEngineConfig(BaseSettings):
    """
    Policy-based memory lifecycle settings.

    Env vars
    --------
    CONTINUUM_POLICY_ENABLED                       bool (default true)
    CONTINUUM_POLICY_STRICT_SENSITIVITY            bool (default true)
    CONTINUUM_POLICY_REQUIRE_APPROVAL_RESTRICTED   bool (default true)
    CONTINUUM_POLICY_DEFAULT_FACT_TTL_DAYS         int|null (default null)
    CONTINUUM_POLICY_TASK_GRACE_HOURS              int (default 24)
    CONTINUUM_POLICY_MEETING_RAW_TTL_DAYS          int (default 30)
    """

    model_config = SettingsConfigDict(
        env_prefix="CONTINUUM_POLICY_",
        populate_by_name=True,
        extra="ignore",
    )

    enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "enabled", "CONTINUUM_POLICY_ENABLED"
        ),
        description=(
            "Master switch. When False (or no PolicyEngine is wired), the "
            "Promoter falls back to the legacy Mem0 ADD/UPDATE/DELETE/NOOP "
            "path and Retriever skips policy-aware filtering."
        ),
    )
    strict_sensitivity: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "strict_sensitivity", "CONTINUUM_POLICY_STRICT_SENSITIVITY"
        ),
        description=(
            "When True, the SensitivityPolicy always wins on conflict — "
            "never merge a redact/encrypt requirement away."
        ),
    )
    require_approval_for_restricted: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "require_approval_for_restricted",
            "CONTINUUM_POLICY_REQUIRE_APPROVAL_RESTRICTED",
        ),
        description=(
            "When True, RESTRICTED-sensitivity candidates are deferred to "
            "memory_pending_approvals instead of being stored."
        ),
    )
    default_fact_ttl_days: int | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "default_fact_ttl_days", "CONTINUUM_POLICY_DEFAULT_FACT_TTL_DAYS"
        ),
        description=(
            "Fallback TTL (days) the DefaultFactPolicy stamps on plain facts. "
            "``None`` = indefinite retention."
        ),
    )
    task_grace_period_hours: int = Field(
        default=24,
        ge=0,
        validation_alias=AliasChoices(
            "task_grace_period_hours", "CONTINUUM_POLICY_TASK_GRACE_HOURS"
        ),
        description=(
            "Hours past a task's deadline the TaskUrgencyPolicy keeps it "
            "around before compaction/expiry."
        ),
    )
    meeting_raw_transcript_ttl_days: int = Field(
        default=30,
        ge=1,
        validation_alias=AliasChoices(
            "meeting_raw_transcript_ttl_days",
            "CONTINUUM_POLICY_MEETING_RAW_TTL_DAYS",
        ),
        description=(
            "How long the raw meeting transcript is retained in the "
            "evidence store before compaction. Extracted decisions / tasks "
            "are unaffected."
        ),
    )
    preserve_decision_versions: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "preserve_decision_versions",
            "CONTINUUM_POLICY_PRESERVE_DECISION_VERSIONS",
        ),
        description="DecisionPolicy keeps every version (append-only).",
    )
    preserve_preference_versions: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "preserve_preference_versions",
            "CONTINUUM_POLICY_PRESERVE_PREFERENCE_VERSIONS",
        ),
        description="UserPreferencePolicy keeps domain-specific history.",
    )
    enable_code_policy: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "enable_code_policy", "CONTINUUM_POLICY_ENABLE_CODE"
        ),
    )
    enable_procedural_policy: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "enable_procedural_policy", "CONTINUUM_POLICY_ENABLE_PROCEDURAL"
        ),
    )


# =============================================================================
# ScoringConfig  (weights + decay)
# =============================================================================


class ScoringWeights(BaseSettings):
    """
    Coefficients for the composite relevance formula.

        composite = rel·relevance + imp·importance + rec·recency + conf·confidence

    All four weights must sum to exactly 1.0.

    Env vars
    --------
    CONTINUUM_SCORE_W_REL   float (default 0.45)
    CONTINUUM_SCORE_W_IMP   float (default 0.25)
    CONTINUUM_SCORE_W_REC   float (default 0.20)
    CONTINUUM_SCORE_W_CONF  float (default 0.10)
    """

    model_config = SettingsConfigDict(
        env_prefix="CONTINUUM_SCORE_",
        populate_by_name=True,
        extra="ignore",
    )

    rel: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("rel", "CONTINUUM_SCORE_W_REL"),
        description="Weight for vector / keyword relevance (0–1).",
    )
    imp: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("imp", "CONTINUUM_SCORE_W_IMP"),
        description="Weight for item importance (0–1).",
    )
    rec: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("rec", "CONTINUUM_SCORE_W_REC"),
        description="Weight for recency / exponential decay (0–1).",
    )
    conf: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("conf", "CONTINUUM_SCORE_W_CONF"),
        description="Weight for source confidence (0–1).",
    )

    @model_validator(mode="after")
    def weights_must_sum_to_one(self) -> ScoringWeights:
        total = self.rel + self.imp + self.rec + self.conf
        if abs(total - 1.0) > _FLOAT_TOL:
            raise ValueError(
                f"Scoring weights must sum to 1.0 "
                f"(rel={self.rel} + imp={self.imp} + rec={self.rec} + conf={self.conf} = {total:.6f}). "
                f"Difference from 1.0: {abs(total - 1.0):.2e}. "
                f"Tip: adjust one weight so all four add up to exactly 1.0."
            )
        return self

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return (rel, imp, rec, conf) for use in vectorised scoring."""
        return (self.rel, self.imp, self.rec, self.conf)

    def apply(
        self,
        relevance: float,
        importance: float,
        recency: float,
        confidence: float,
    ) -> float:
        """Compute the composite score from four normalised dimensions."""
        raw = (
            self.rel * relevance
            + self.imp * importance
            + self.rec * recency
            + self.conf * confidence
        )
        return min(max(raw, 0.0), 1.0)


class ScoringConfig(BaseSettings):
    """
    Scoring formula configuration.

    Env vars
    --------
    CONTINUUM_SCORE_TAU_HOURS  float hours for recency half-life (default 168 = 7 days)
    """

    model_config = SettingsConfigDict(
        env_prefix="CONTINUUM_SCORE_",
        populate_by_name=True,
        extra="ignore",
    )

    weights: ScoringWeights = Field(
        default_factory=ScoringWeights,
        description="Per-dimension coefficients. Must sum to 1.0.",
    )
    tau_hours: float = Field(
        default=168.0,  # 7 days
        gt=0,
        validation_alias=AliasChoices("tau_hours", "CONTINUUM_SCORE_TAU_HOURS"),
        description=(
            "Recency decay half-life in hours. "
            "After tau_hours, an item's recency score is 0.5. "
            "After 2×tau_hours it is 0.25, etc. "
            "Default 168 h = 7 days. Set lower (e.g. 24) for fast-moving domains."
        ),
    )
    layer_boost: dict[str, float] = Field(
        default_factory=lambda: {"STM": 1.05, "MTM": 1.0, "LTM": 1.1},
        description=(
            "Multiplicative post-score boost per tier (keyed by "
            "MemoryTier.name). LTM facts are slightly favoured for "
            "durability, STM for recency; unknown tiers default to 1.0."
        ),
    )

    def recency_score(self, age_hours: float) -> float:
        """
        Exponential decay: score = 2^(−age / tau_hours).

        Returns 1.0 for brand-new items and approaches 0.0 asymptotically.
        Equivalent to: exp(−age × ln2 / tau).
        """
        return math.pow(2.0, -age_hours / self.tau_hours)


# =============================================================================
# RetrieverConfig
# =============================================================================


class RetrieverConfig(BaseSettings):
    """
    Fan-out retrieval parameters.

    Env vars
    --------
    CONTINUUM_RETRIEVER_K1           int  (STM + MTM candidate pool, default 50)
    CONTINUUM_RETRIEVER_K_GRAPH      int  (LTM graph-neighbour hops, default 2)
    CONTINUUM_RETRIEVER_LTM_TOP_K    int  (LTM results after re-ranking, default 10)
    CONTINUUM_RETRIEVER_STM_TURNS    int  (how many STM turns to include, default 6)
    CONTINUUM_RETRIEVER_MTM_TOP_K    int  (MTM results after re-ranking, default 5)
    """

    model_config = SettingsConfigDict(
        env_prefix="CONTINUUM_RETRIEVER_",
        populate_by_name=True,
        extra="ignore",
    )

    k1: int = Field(
        default=50,
        ge=1,
        validation_alias=AliasChoices("k1", "CONTINUUM_RETRIEVER_K1"),
        description=(
            "Initial candidate pool size fetched from the vector index "
            "before scoring and re-ranking. Larger = better recall, slower."
        ),
    )
    k_graph: int = Field(
        default=2,
        ge=0,
        le=5,
        validation_alias=AliasChoices("k_graph", "CONTINUUM_RETRIEVER_K_GRAPH"),
        description=(
            "Number of hops to traverse in the LTM knowledge graph "
            "starting from top-scoring seed nodes (0 = graph disabled)."
        ),
    )
    ltm_top_k: int = Field(
        default=10,
        ge=1,
        validation_alias=AliasChoices("ltm_top_k", "CONTINUUM_RETRIEVER_LTM_TOP_K"),
        description="Maximum LTM items returned after scoring and re-ranking.",
    )
    stm_turns: int = Field(
        default=6,
        ge=1,
        validation_alias=AliasChoices("stm_turns", "CONTINUUM_RETRIEVER_STM_TURNS"),
        description=(
            "Number of most-recent STM conversation turns to include in "
            "the context bundle (each turn = 1 user + 1 assistant message)."
        ),
    )
    mtm_top_k: int = Field(
        default=5,
        ge=1,
        validation_alias=AliasChoices("mtm_top_k", "CONTINUUM_RETRIEVER_MTM_TOP_K"),
        description="Maximum MTM items returned after scoring and re-ranking.",
    )
    graph_expand_n: int = Field(
        default=10,
        ge=0,
        validation_alias=AliasChoices(
            "graph_expand_n", "CONTINUUM_RETRIEVER_GRAPH_EXPAND_N"
        ),
        description=(
            "How many top entity-kind hits to expand via 1-hop LTM graph "
            "neighbours (0 disables graph expansion). Distinct from "
            "``k_graph`` which is the hop *depth*."
        ),
    )
    hyde_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "hyde_enabled", "CONTINUUM_RETRIEVER_HYDE_ENABLED"
        ),
        description=(
            "Enable HyDE query rewriting (a hypothetical answer is generated "
            "and used as the search query). Off by default — needs an "
            "injected rewrite function and adds an LLM call."
        ),
    )


# =============================================================================
# PromoterConfig
# =============================================================================


class PromoterConfig(BaseSettings):
    """
    MTM → LTM promotion settings.

    Env vars
    --------
    CONTINUUM_PROMOTER_CONFIDENCE_THRESHOLD  float 0–1  (default 0.75)
    CONTINUUM_PROMOTER_BATCH_SIZE            int        (default 20)
    CONTINUUM_PROMOTER_TRIGGER_ON_NEW_ENTITY bool       (default true)
    CONTINUUM_PROMOTER_LLM_MODEL             str        (default "gpt-4o-mini")
    """

    model_config = SettingsConfigDict(
        env_prefix="CONTINUUM_PROMOTER_",
        populate_by_name=True,
        extra="ignore",
    )

    confidence_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices(
            "confidence_threshold", "CONTINUUM_PROMOTER_CONFIDENCE_THRESHOLD"
        ),
        description=(
            "Minimum LLM-assigned confidence for a candidate to be promoted "
            "to LTM. Candidates below this score receive a NOOP decision. "
            "Range [0.0, 1.0]; raise it to reduce noisy promotions."
        ),
    )
    batch_size: int = Field(
        default=20,
        ge=1,
        le=500,
        validation_alias=AliasChoices("batch_size", "CONTINUUM_PROMOTER_BATCH_SIZE"),
        description=(
            "Number of MTM items evaluated per Promoter.promote() invocation. "
            "Smaller batches = lower latency per call, more calls needed."
        ),
    )
    trigger_on_new_entity: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "trigger_on_new_entity", "CONTINUUM_PROMOTER_TRIGGER_ON_NEW_ENTITY"
        ),
        description=(
            "When True, the Promoter runs eagerly whenever a new 'entity' "
            "kind is added to MTM (e.g. a new project, person, or decision "
            "is mentioned), in addition to its scheduled batch runs."
        ),
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("llm_model", "CONTINUUM_PROMOTER_LLM_MODEL"),
        description=(
            "LiteLLM-compatible model identifier used for promotion decisions. "
            "Accepts any provider supported by litellm (openai/, anthropic/, ollama/, …)."
        ),
    )

    # ── Mem0 four-operation (ADD/UPDATE/DELETE/NOOP) knobs ───────────────────

    max_neighbors: int = Field(
        default=10,
        ge=1,
        le=100,
        validation_alias=AliasChoices(
            "max_neighbors", "CONTINUUM_PROMOTER_MAX_NEIGHBORS"
        ),
        description=(
            "Top-K similar LTM facts retrieved per candidate to ground the "
            "ADD/UPDATE/DELETE/NOOP decision."
        ),
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        validation_alias=AliasChoices(
            "temperature", "CONTINUUM_PROMOTER_TEMPERATURE"
        ),
        description="0.0 keeps ADD/UPDATE/DELETE/NOOP decisions deterministic.",
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=8192,
        validation_alias=AliasChoices(
            "max_tokens", "CONTINUUM_PROMOTER_MAX_TOKENS"
        ),
        description="Completion-token cap for the decision call (cost control).",
    )
    timeout: int = Field(
        default=30,
        ge=1,
        validation_alias=AliasChoices("timeout", "CONTINUUM_PROMOTER_TIMEOUT"),
        description="Per-attempt LLM call timeout, in seconds.",
    )
    add_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices(
            "add_threshold", "CONTINUUM_PROMOTER_ADD_THRESHOLD"
        ),
        description=(
            "Short-circuit: if the best neighbour's cosine is below this, "
            "the candidate is definitely new → ADD without calling the LLM."
        ),
    )
    noop_threshold: float = Field(
        default=0.97,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices(
            "noop_threshold", "CONTINUUM_PROMOTER_NOOP_THRESHOLD"
        ),
        description=(
            "Short-circuit: if the best neighbour's cosine exceeds this, the "
            "candidate is a near-duplicate → NOOP without calling the LLM."
        ),
    )


# =============================================================================
# OptimizerConfig
# =============================================================================

_VALID_STRATEGIES = frozenset({"truncation", "compression", "hybrid"})


class OptimizerConfig(BaseSettings):
    """
    Token-budget optimisation settings.

    Env vars
    --------
    CONTINUUM_OPTIMIZER_COMPRESS_RATIO      float 0–1  (default 0.5)
    CONTINUUM_OPTIMIZER_STRATEGIES_ENABLED  comma-separated list (default "truncation,compression")
    """

    model_config = SettingsConfigDict(
        env_prefix="CONTINUUM_OPTIMIZER_",
        populate_by_name=True,
        extra="ignore",
    )

    compress_ratio: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        validation_alias=AliasChoices("compress_ratio", "CONTINUUM_OPTIMIZER_COMPRESS_RATIO"),
        description=(
            "Target compression ratio for the 'compression' strategy. "
            "0.5 means aim to halve the token count of compressed items. "
            "Only used when 'compression' is in strategies_enabled."
        ),
    )
    strategies_enabled: list[Literal["truncation", "compression", "hybrid"]] = Field(
        default=["truncation", "compression"],
        validation_alias=AliasChoices(
            "strategies_enabled", "CONTINUUM_OPTIMIZER_STRATEGIES_ENABLED"
        ),
        description=(
            "Ordered list of optimisation strategies the Optimizer may apply. "
            "truncation — drop lowest-scored items until budget fits. "
            "compression — shorten item content via LLMlingua. "
            "hybrid — compress low-priority, keep high-priority verbatim."
        ),
    )

    @field_validator("strategies_enabled", mode="before")
    @classmethod
    def parse_comma_list(cls, v: Any) -> list[str]:
        """Accept either a list or a comma-separated string from env vars."""
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return list(v)

    @field_validator("strategies_enabled")
    @classmethod
    def strategies_must_be_valid(cls, v: list[str]) -> list[str]:
        invalid = set(v) - _VALID_STRATEGIES
        if invalid:
            raise ValueError(
                f"Unknown strategy names: {invalid}. Valid choices: {sorted(_VALID_STRATEGIES)}"
            )
        if not v:
            raise ValueError("strategies_enabled must contain at least one strategy.")
        return v


# =============================================================================
# ContinuumCodeConfig
# =============================================================================

_VALID_LANGUAGES = frozenset(
    {
        "python",
        "javascript",
        "typescript",
        "rust",
        "go",
        "java",
        "c",
        "cpp",
        "ruby",
        "kotlin",
        "swift",
    }
)


class ContinuumCodeConfig(BaseSettings):
    """
    Code knowledge graph settings (continuum-code sub-package).

    Env vars
    --------
    CONTINUUM_CODE_LANGUAGES        comma-separated (default "python,javascript,typescript")
    CONTINUUM_CODE_INDEX_BATCH_SIZE int (default 32)
    CONTINUUM_CODE_INCREMENTAL      bool (default true)
    CONTINUUM_CODE_WATCH_GLOBS      comma-separated glob patterns
    """

    model_config = SettingsConfigDict(
        env_prefix="CONTINUUM_CODE_",
        populate_by_name=True,
        extra="ignore",
    )

    languages: list[str] = Field(
        default=["python", "javascript", "typescript"],
        validation_alias=AliasChoices("languages", "CONTINUUM_CODE_LANGUAGES"),
        description=(
            "Programming languages the code knowledge graph will index. "
            f"Supported: {sorted(_VALID_LANGUAGES)}."
        ),
    )
    index_batch_size: int = Field(
        default=32,
        ge=1,
        le=1024,
        validation_alias=AliasChoices("index_batch_size", "CONTINUUM_CODE_INDEX_BATCH_SIZE"),
        description=(
            "Number of files processed per indexing batch. "
            "Larger batches use more RAM but reduce overhead."
        ),
    )
    incremental_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("incremental_enabled", "CONTINUUM_CODE_INCREMENTAL"),
        description=(
            "When True, only files changed since the last index run are "
            "re-processed (tracked via content hashes). "
            "Set to False for a full re-index on every run."
        ),
    )
    watch_globs: list[str] = Field(
        default=["**/*.py", "**/*.js", "**/*.ts"],
        validation_alias=AliasChoices("watch_globs", "CONTINUUM_CODE_WATCH_GLOBS"),
        description="Glob patterns for files to include in the index.",
    )

    @field_validator("languages", "watch_globs", mode="before")
    @classmethod
    def parse_comma_list(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return list(v)

    @field_validator("languages")
    @classmethod
    def languages_must_be_supported(cls, v: list[str]) -> list[str]:
        invalid = {lang.lower() for lang in v} - _VALID_LANGUAGES
        if invalid:
            raise ValueError(
                f"Unsupported languages: {invalid}. Supported: {sorted(_VALID_LANGUAGES)}"
            )
        return [lang.lower() for lang in v]


# =============================================================================
# ContinuumConfig  (root)
# =============================================================================


class ContinuumConfig(BaseSettings):
    """
    Root configuration object for the Continuum memory framework.

    Merges all sub-configs and exposes a single ``.load()`` factory method
    that handles env-var, YAML-file, and programmatic sources in the correct
    priority order.

    Priority (highest → lowest)
    ---------------------------
    1. Values passed directly to the constructor / ``load()``
    2. CONTINUUM_* environment variables
    3. ``continuum.yaml`` (or the file at CONTINUUM_CONFIG_FILE)
    4. Built-in defaults

    Example
    -------
    >>> cfg = ContinuumConfig.load()
    >>> cfg.database.pool_size
    5
    >>> cfg.scoring.weights.apply(0.8, 0.6, 0.9, 1.0)
    0.79
    """

    model_config = SettingsConfigDict(
        env_prefix="CONTINUUM_",
        populate_by_name=True,
        extra="ignore",
        # pydantic-settings reads env vars even without explicit env_prefix
        # on the root; sub-configs carry their own prefixes.
    )

    # ── Sub-configs ──────────────────────────────────────────────────────────
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    llm_extraction: LLMExtractionConfig = Field(
        default_factory=LLMExtractionConfig
    )
    fact_extraction: FactExtractionConfig = Field(
        default_factory=FactExtractionConfig
    )
    trigger: TriggerConfig = Field(default_factory=TriggerConfig)
    policy_engine: PolicyEngineConfig = Field(default_factory=PolicyEngineConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    promoter: PromoterConfig = Field(default_factory=PromoterConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    code: ContinuumCodeConfig = Field(default_factory=ContinuumCodeConfig)

    # ── Top-level knobs ───────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        validation_alias=AliasChoices("log_level", "CONTINUUM_LOG_LEVEL"),
        description="Root log level for the continuum logger hierarchy.",
    )
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        validation_alias=AliasChoices("environment", "CONTINUUM_ENV"),
        description=(
            "Deployment environment. In 'production', validation errors are "
            "fatal and telemetry is enabled by default."
        ),
    )

    # ── Class-level: which YAML file to read ─────────────────────────────────
    _config_file: Path = Path("continuum.yaml")  # overridden by load()

    # ─────────────────────────────────────────────────────────────────────────
    # pydantic-settings source ordering
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,  # pydantic-settings ≥ 2.4
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Source priority (left = highest):
            init_settings  (programmatic kwargs)
            env_settings   (CONTINUUM_* env vars)
            yaml_settings  (continuum.yaml or $CONTINUUM_CONFIG_FILE)
            dotenv_settings (.env file)
        """
        config_file = Path(os.environ.get("CONTINUUM_CONFIG_FILE", "continuum.yaml"))
        yaml_settings = YamlConfigSettingsSource(
            settings_cls,
            yaml_file=config_file if config_file.exists() else None,
        )
        return (
            init_settings,
            env_settings,
            yaml_settings,
            dotenv_settings,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public factory
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        config_file: str | Path | None = None,
        **overrides: Any,
    ) -> ContinuumConfig:
        """
        Load configuration from all sources and return a validated instance.

        Parameters
        ----------
        config_file:
            Path to a YAML config file.  When provided, sets
            ``CONTINUUM_CONFIG_FILE`` in the environment so
            ``settings_customise_sources`` picks it up.
            Defaults to ``"continuum.yaml"`` in the working directory.
        **overrides:
            Arbitrary keyword arguments forwarded to the constructor,
            letting callers override individual fields programmatically.
            Example: ``ContinuumConfig.load(log_level="DEBUG")``.

        Returns
        -------
        ContinuumConfig
            Fully validated configuration object.

        Raises
        ------
        pydantic.ValidationError
            If any value fails validation (weights don't sum to 1.0,
            pool_size < 1, confidence_threshold out of range, …).
        """
        if config_file is not None:
            os.environ["CONTINUUM_CONFIG_FILE"] = str(_abs_path(config_file))
        return cls(**overrides)

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience helpers
    # ─────────────────────────────────────────────────────────────────────────

    def is_production(self) -> bool:
        return self.environment == "production"

    def scoring_weights_tuple(self) -> tuple[float, float, float, float]:
        """Return (rel, imp, rec, conf) — handy for vectorised scoring loops."""
        return self.scoring.weights.as_tuple()

    def to_yaml(self) -> str:
        """
        Serialise the current (fully resolved) configuration to YAML.

        Useful for dumping the effective config to a log or file so the exact
        settings used for a run are reproducible.
        """
        data = self.model_dump(mode="json")
        return yaml.dump(data, default_flow_style=False, sort_keys=True)

    @classmethod
    def yaml_schema(cls) -> str:
        """
        Return an annotated YAML template showing every field, its default,
        and a one-line description.  Pipe this into ``continuum.yaml`` to
        start a new project config.

        Usage::

            python -c "
            from continuum.core.config import ContinuumConfig
            print(ContinuumConfig.yaml_schema())
            " > continuum.yaml
        """
        lines: list[str] = [
            "# Continuum configuration — generated by ContinuumConfig.yaml_schema()",
            "# All values shown are defaults.  Remove lines you don't need to override.",
            "",
        ]
        schema = cls.model_json_schema()
        instance = cls()
        data = instance.model_dump(mode="json")
        _append_yaml_section(lines, data, schema, indent=0)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal: annotated YAML generation
# ---------------------------------------------------------------------------


def _append_yaml_section(
    lines: list[str],
    data: dict[str, Any],
    schema: dict[str, Any],
    indent: int,
) -> None:
    """Recursively build annotated YAML lines from a model_dump dict."""
    pad = "  " * indent
    props = schema.get("properties", {})
    defs = schema.get("$defs", {})

    for key, value in data.items():
        field_schema = props.get(key, {})
        # Resolve $ref
        if "$ref" in field_schema:
            ref_name = field_schema["$ref"].split("/")[-1]
            field_schema = defs.get(ref_name, field_schema)

        description = field_schema.get("description", "")
        if description:
            # Wrap long descriptions at 72 chars
            for chunk in _wrap(description, 72 - len(pad)):
                lines.append(f"{pad}# {chunk}")

        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            _append_yaml_section(lines, value, field_schema, indent + 1)
        elif isinstance(value, list):
            lines.append(f"{pad}{key}:")
            for item in value:
                lines.append(f"{pad}  - {item!r}" if isinstance(item, str) else f"{pad}  - {item}")
        else:
            lines.append(f"{pad}{key}: {value!r}")
        lines.append("")  # blank line between fields


def _wrap(text: str, width: int) -> list[str]:
    """Very simple word-wrap that preserves existing newlines."""
    words, line, result = text.split(), "", []
    for word in words:
        if len(line) + len(word) + 1 > width:
            if line:
                result.append(line)
            line = word
        else:
            line = word if not line else f"{line} {word}"
    if line:
        result.append(line)
    return result or [""]
