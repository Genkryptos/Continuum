# Config reference

Continuum reads configuration from four sources, merged in priority
order (highest wins):

1. **Constructor args** — `ContinuumConfig(...)` or `.load(**kwargs)`.
2. **Environment variables** with the `CONTINUUM_` prefix.
3. **YAML file** at `continuum.yaml` (or wherever `CONTINUUM_CONFIG_FILE` points).
4. **Built-in defaults** baked into each sub-config dataclass.

Everything is pydantic-settings under the hood, so type validation
is automatic and error messages point at the offending field.

## Top-level config

```python
from continuum.core.config import ContinuumConfig
cfg = ContinuumConfig.load()       # picks up env + yaml + defaults
print(cfg.database.pool_size)      # → 5
```

Two top-level knobs:

| Field | Default | Notes |
|---|---|---|
| `log_level` | `INFO` | Root logger level. Set via `CONTINUUM_LOG_LEVEL` or YAML. |
| `environment` | `development` | One of `development` / `staging` / `production`. In `production`, validation errors are fatal and telemetry defaults to enabled. |

## The 13 sub-configs

Each sub-config is a separate pydantic-settings model with its own
env-var prefix. All are optional — defaults are reasonable for
in-memory + local dev.

### `database` (`CONTINUUM_DB_*`)

| Field | Default | Notes |
|---|---|---|
| `url` | `postgresql://localhost:5432/continuum` | Async DSN. Required only when using `PostgresSTM/MTM/LTM`. |
| `pool_size` | `5` | psycopg3 connection pool size. |
| `max_overflow` | `10` | Extra connections under load. |
| `pool_timeout` | `30.0` | Seconds to wait for a free connection. |
| `max_retries` | `3` | Transient-failure retry count. |
| `application_name` | `continuum` | Visible in `pg_stat_activity`. |

### `embedding` (`CONTINUUM_EMBEDDING_*`)

| Field | Default | Notes |
|---|---|---|
| `model` | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim, fast on CPU. |
| `device` | `auto` | `auto` / `cpu` / `mps` / `cuda`. Use `cpu` on Macs when running alongside LMStudio. |
| `batch_size` | `32` | Encoder batch size. |
| `normalize` | `True` | L2-normalize so cosine == dot product. |

### `extraction` (`CONTINUUM_EXTRACTION_*`)

Top-level extraction toggles + the GLiNER (NER) model config.

| Field | Default | Notes |
|---|---|---|
| `enabled` | `True` | Master switch. |
| `gliner_model` | `urchade/gliner_small-v2.1` | NER model. ~200MB download. |
| `gliner_threshold` | `0.5` | Confidence floor for entity acceptance. |
| `entity_types` | `["PER","ORG","LOC","DATE","EVENT"]` | What GLiNER tags. |

### `llm_extraction` (`CONTINUUM_LLM_EXTRACTION_*`)

LLM-based entity / relation extractor (`continuum/extraction/llm_extractor.py`).

| Field | Default | Notes |
|---|---|---|
| `enabled` | `True` | |
| `model` | `gpt-4o-mini` | OpenAI-shaped completion provider. |
| `temperature` | `0.0` | Deterministic by default. |
| `max_tokens` | `512` | Cap on the structured-output response. |

### `fact_extraction` (`CONTINUUM_FACT_EXTRACTION_*`)

Atomic-fact extractor over MTM summaries.

| Field | Default | Notes |
|---|---|---|
| `enabled` | `True` | |
| `model` | `gpt-4o-mini` | |
| `temperature` | `0.0` | |
| `max_tokens` | `1024` | |
| `min_confidence` | `0.5` | Facts below this are dropped before write. |
| `min_importance` | `0.3` | Same for importance. |

### `trigger` (`CONTINUUM_TRIGGER_*`)

When to fire MTM → LTM promotion.

| Field | Default | Notes |
|---|---|---|
| `enabled` | `False` | Off by default; opt in when you wire a promoter. |
| `mtm_token_threshold` | `8000` | Promote when MTM exceeds this. |
| `mtm_block_threshold` | `20` | …or this many blocks. |
| `idle_minutes` | `30` | Idle-flush gap. |

### `policy_engine` (`CONTINUUM_POLICY_*`)

Policy-engine config (migration 004 + the 8 default policies).

| Field | Default | Notes |
|---|---|---|
| `enabled` | `True` | |
| `default_policies` | All 8 (see [`continuum/policies/`](../continuum/policies/)) | |
| `evaluation_mode` | `lenient` | `strict` rejects on any policy fail; `lenient` warns. |

### `scoring` (`CONTINUUM_SCORING_*`)

Composite scorer (the four-component weighted score).

| Field | Default | Notes |
|---|---|---|
| `weights.relevance` | `0.45` | Cosine score weight. |
| `weights.importance` | `0.25` | LLM-extracted importance weight. |
| `weights.recency` | `0.20` | Exponential-decay weight. |
| `weights.confidence` | `0.10` | Extraction-confidence weight. |
| `tau_hours` | `168.0` | Recency decay half-life (7 days). |
| `layer_boost` | `{stm: 1.0, mtm: 0.95, ltm: 0.90}` | Tier preferences. |

### `retriever` (`CONTINUUM_RETRIEVER_*`)

| Field | Default | Notes |
|---|---|---|
| `top_k` | `8` | Final result count. |
| `k1` | `40` | Cosine candidate pool. |
| `mode` | `topk` | `topk` / `long` / `auto`. |
| `max_context_tokens` | `100000` | Long-context cutoff. |

### `reranker` (`CONTINUUM_RERANKER_*`)

Cross-encoder reranker (off by default).

| Field | Default | Notes |
|---|---|---|
| `enabled` | `False` | |
| `model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | |
| `top_n` | `4` | How many to keep after reranking. |

### `promoter` (`CONTINUUM_PROMOTER_*`)

`Mem0Promoter` config.

| Field | Default | Notes |
|---|---|---|
| `enabled` | `False` | Off until you wire it. |
| `batch_size` | `10` | Promotion batch size. |
| `dedup_threshold` | `0.92` | Cosine threshold above which facts dedupe. |
| `contradiction_threshold` | `0.85` | Below this, two facts are considered NOT contradicting (left as parallel). |

### `optimizer` (`CONTINUUM_OPTIMIZER_*`)

Token-budget compression chain (5 strategies).

| Field | Default | Notes |
|---|---|---|
| `enabled` | `False` | Off by default; the chain is most useful when you're context-bound. |
| `stm_trim.keep_last` | `10` | Verbatim recent turns. |
| `mtm_summarize.keep_recent` | `3` | Recent summary blocks left untouched. |
| `mtm_summarize.method` | `extractive` | `extractive` / `llm`. |
| `semantic_dedupe.threshold` | `0.92` | |
| `llmlingua.ratio` | `0.5` | Compression target (LLMLingua only). |
| `score_aware_prune.preserve_mtm_count` | `3` | |

### `code` (`CONTINUUM_CODE_*`)

Code-indexing config for an LSP-style subsystem that's *defined but
not yet implemented*. Lives in `ContinuumCodeConfig` for forward
compatibility; setting it does nothing today. Leave at defaults.

## YAML example

`continuum.yaml`:

```yaml
log_level: INFO
environment: production

database:
  url: postgresql://localhost:5432/continuum
  pool_size: 10
  pool_timeout: 60.0

embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  device: cuda
  batch_size: 64

scoring:
  weights:
    relevance: 0.55
    importance: 0.20
    recency: 0.15
    confidence: 0.10
  tau_hours: 96.0

trigger:
  enabled: true
  mtm_token_threshold: 12000

promoter:
  enabled: true
  dedup_threshold: 0.90
```

Load with `ContinuumConfig.load()` and pydantic-settings merges
defaults + env + this file in the right order.

## Env-var examples

```bash
export CONTINUUM_LOG_LEVEL=DEBUG
export CONTINUUM_ENV=production
export CONTINUUM_DB_URL='postgresql://continuum_user:secret@db:5432/continuum'
export CONTINUUM_DB_POOL_SIZE=10
export CONTINUUM_EMBEDDING_DEVICE=cpu
export CONTINUUM_TRIGGER_ENABLED=true
export CONTINUUM_PROMOTER_ENABLED=true
```

Nested fields use `__` (double underscore) as the separator in
pydantic-settings, e.g.:

```bash
export CONTINUUM_SCORING__WEIGHTS__RELEVANCE=0.55
export CONTINUUM_SCORING__TAU_HOURS=96
```

## Recommended starting points

### Local development (no infra)

Defaults work. Optionally set:

```bash
export CONTINUUM_LOG_LEVEL=DEBUG
export CONTINUUM_EMBEDDING_DEVICE=cpu
```

### Small production deployment

```yaml
environment: production
database:
  pool_size: 10
trigger:
  enabled: true
promoter:
  enabled: true
extraction:
  enabled: true
fact_extraction:
  enabled: true
```

### Large production deployment

Add to the small config:

```yaml
database:
  pool_size: 50
  max_overflow: 100
optimizer:
  enabled: true       # context-bound at this scale
reranker:
  enabled: true       # diminishing returns on the cheap path
```

…and see [Operations](operations.md) for monitoring + scaling.

## Validating a config without running

```python
from continuum.core.config import ContinuumConfig
try:
    cfg = ContinuumConfig.load(_config_file="my_config.yaml")
    print("OK")
except Exception as e:
    print("invalid:", e)
```

In `environment=production`, any validation error is fatal at
`load()` time — which is the intended behaviour. In development the
defaults absorb most missing values.
