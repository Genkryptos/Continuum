# Operations

This doc covers what you need to put Continuum in production:
Postgres + pgvector setup, scaling, observability, and what to
monitor. The in-memory defaults are great for prototyping; everything
here is about durable, multi-process deployment.

## 1 · Postgres + pgvector setup

Continuum's `PostgresSTM` / `PostgresMTM` / `PostgresLTM` need:

* PostgreSQL **≥ 15**
* `pgvector` extension **≥ 0.5**
* `pg_trgm` extension (for lexical hybrid search; migration 003)
* psycopg3 client (already in `pyproject.toml` deps)

### Quick local setup

```bash
# Using Docker (recommended for dev):
docker run -d \
  --name continuum-pg \
  -e POSTGRES_PASSWORD=secret \
  -e POSTGRES_DB=continuum \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

Then point Continuum at it:

```bash
export CONTINUUM_DB_URL='postgresql://postgres:secret@localhost:5432/continuum'
```

### Production-equivalent setup

* Managed Postgres with pgvector (AWS RDS, GCP Cloud SQL, Supabase,
  Crunchy Bridge, Neon — all support pgvector).
* Connection pooler in front (pgBouncer or built-in pool) — Continuum
  uses psycopg3's pool but a transaction-mode pgBouncer in front
  helps at scale.
* Separate read-replica for analytics queries against the LTM table;
  the write path is mostly background promotion.

### Running the migrations

The `stm_messages` table self-bootstraps (`PostgresSTM.ensure_schema()` runs on
`session.start()`). Everything else — the `vector` / `pg_trgm` extensions, the
`memory_nodes` graph, the HNSW index, and the policy tables — lives in
`migrations/*.sql` and is **not** auto-applied. Run it explicitly:

```bash
make db-migrate            # applies all pending migrations to CONTINUUM_DB_DSN
make db-migrate-dry        # list what would be applied, without applying
```

`make db-migrate` is just `python -m continuum.db.migrate` (pass `--dsn` to
override). Each file is idempotent and records itself into `schema_migrations`,
so re-running is safe and only pending migrations execute. Equivalent manual
path with `psql`:

```bash
psql $CONTINUUM_DB_DSN -f migrations/001_ltm_schema.sql
psql $CONTINUUM_DB_DSN -f migrations/002_pgvector_upgrade.sql
psql $CONTINUUM_DB_DSN -f migrations/003_lexical_search.sql
psql $CONTINUUM_DB_DSN -f migrations/004_policy_engine.sql
```

* **001** `CREATE EXTENSION vector / pg_trgm / uuid-ossp`, the bi-temporal
  `memory_nodes` / `memory_edges` / `memory_episodes` tables, and the HNSW index.
* **002** upgrades the `vector(1024)` embedding column to `halfvec` + rebuilds
  the HNSW index (requires pgvector ≥ 0.8.0).
* **003** adds the trigram (`pg_trgm`) index that powers hybrid lexical search.
* **004** adds the policy-engine tables (`memory_decision_traces`,
  `memory_pending_approvals`) and the candidate/sensitivity/urgency columns.

After migrating, confirm with `make check-env` — it reports `LTM schema present`
once `memory_nodes` exists.

## 2 · Scaling

### Single-process, small scale (< 100K facts in LTM)

* Defaults work.
* `database.pool_size=5` plenty.
* Single Postgres instance.
* All extraction synchronous on the background queue is fine.

### Multi-process, medium scale (100K-10M facts)

* Bump `database.pool_size=20-50`.
* Add a transaction-mode pgBouncer.
* Move extraction off the same process — the `Mem0Promoter` is
  designed to run as a background worker pool consuming the
  `promotion_queue` table.
* Enable the reranker for retrieval (off by default to keep base
  latency low; at this scale the cross-encoder quality is worth the
  ~15 ms it adds).

### Large scale (> 10M facts)

* Postgres alone is no longer the cheapest path; consider:
  * pgvector with **HNSW index** (`migrations/002_pgvector_upgrade.sql`
    uses `ivfflat`; switch to `hnsw` for sub-linear ANN at the cost
    of build time).
  * Partition LTM by `user_id` if you're multi-tenant.
  * Move embeddings to a dedicated vector DB (Qdrant / Pinecone /
    Weaviate) and let Postgres own only the structured columns
    (supersession, bi-temporal). The schema supports this split.
* The optimizer chain becomes load-bearing — large LTM means
  retrieval can pull large bundles; the chain compresses to budget.

## 3 · Observability

### Logs

Continuum logs through the standard `logging` hierarchy:

```python
logging.getLogger("continuum")             # root
logging.getLogger("continuum.session")
logging.getLogger("continuum.promotion")
logging.getLogger("continuum.retrieval")
logging.getLogger("continuum.extraction")
```

Set `CONTINUUM_LOG_LEVEL=DEBUG` to see every retrieval, promotion,
and extraction step. INFO is the production default.

### Metrics — what to expose

`ContinuumSession` doesn't ship a Prometheus integration directly,
but every component publishes the right hook points:

| Metric | Source | Why |
|---|---|---|
| `process_turn_ms` (histogram) | `session.process_turn` timing | Foreground latency. Target p95 < 1 s on a healthy deployment. |
| `retrieval_latency_ms` (histogram) | retriever timing | Where most of `process_turn_ms` goes when STM grows large. |
| `promotion_lag_seconds` (gauge) | `trigger_manager` queue depth | Promotion is async — but if it falls behind, LTM is stale. Alert at > 5 min. |
| `ltm_total_facts` (gauge) | `SELECT COUNT(*) FROM ltm_facts` | Capacity planning. |
| `ltm_superseded_ratio` (gauge) | `count(*) FILTER (WHERE superseded_by IS NOT NULL) / count(*)` | Sanity — if this is 0 % the supersession path isn't firing. If it's > 80 % the user is contradicting themselves a lot or the detector is over-firing. |
| `llm_calls_per_turn` (histogram) | `process_turn` instrumentation | Cost driver. Each extraction call is one LLM-call per user turn. |
| `embedder_latency_ms` (histogram) | `EmbeddingService.encode` timing | Often the dominant retrieval cost. |
| `background_queue_depth` (gauge) | `session.background.depth()` | If this grows unbounded, you've outrun your background workers — `BackgroundQueue` will start dropping with warnings. |

### Health checks

The `ContinuumSession` exposes `health()`:

```python
async with ContinuumSession(config) as session:
    h = await session.health()
    # → {"stm": "ok", "mtm": "ok", "ltm": "ok", "background_workers": "ok"}
```

Wire to your readiness probe. Each tier reports `"ok"` /
`"degraded"` / `"down"` based on whether reads work in under 2 s.

### Tracing

Continuum's foreground path is short (5 awaits per turn). If you
need traces, instrument at the `session.process_turn` boundary —
OpenTelemetry spans on the responder + retriever calls give you the
full picture. Background promotion benefits from its own trace
context but isn't latency-critical.

## 4 · What to alert on

| Alert | Threshold | Why |
|---|---|---|
| `process_turn_ms.p95 > 2 s` | sustained 5 min | User-facing latency degradation. |
| `promotion_lag_seconds > 300` | sustained 5 min | LTM is going stale; new facts won't be retrievable. |
| `background_queue_depth > 800` | for 1 min | Queue capacity is 1000 by default — within 20 % of dropping. |
| `ltm_superseded_ratio = 0 over 1 hr` | exact | The supersession detector isn't firing — likely a config or model regression. |
| `Postgres connections_used / pool_size > 0.8` | sustained 1 min | Bump `database.pool_size` or your pgBouncer config. |
| `embedder error rate > 1 %` | 1 min | Usually transient (GPU contention, OOM); investigate if sustained. |

## 5 · Backup + recovery

Standard Postgres practice:

* Continuous archiving (WAL) with PITR.
* Daily logical backups (`pg_dump`) of the `continuum` database.
* Test restore quarterly.

What's unique:

* **STM is durable in `PostgresSTM`.** Don't truncate the STM table to
  reclaim space — it'll break in-flight sessions. Set a retention
  policy via the `trigger.idle_minutes` knob instead, which promotes
  + clears STM gracefully.
* **LTM with `superseded_by` is append-only by design.** Don't
  `DELETE` superseded rows — they're the audit trail. If space is
  the concern, archive rows where `superseded_by IS NOT NULL AND
  superseded_at < now() - interval '1 year'` to cold storage.

## 6 · Cost model

Per-user, per-month, ballpark:

| Component | Cost driver | Order of magnitude |
|---|---|---|
| Postgres | rows × indexes | dominated by LTM; ~1 MB per 1K facts. |
| Embedder (CPU) | turns × embedder latency | usually < 1 % of total compute. |
| Embedder (GPU) | turns × GPU-hour | meaningful at scale; consider serving via vLLM. |
| LLM extraction | promoted turns × extraction calls | **the largest variable cost.** ~2 calls per user turn. On gpt-4o-mini: ~$0.0001 / turn. |
| LLM responder | turns × your model | depends entirely on your model choice. |

The framework's job is to *minimise* the extraction LLM cost
(promotion runs in batches, off the hot path) and the
responder LLM context cost (the optimizer chain). Both knobs are
tunable via the configs in [Config](config.md).

## 7 · Common production issues

### "Supersession-correctness drops to ~50 % in production"

The schema benchmark hits 100 % given correct contradiction
*detection*. If detection is unreliable in your deployment:

1. Check your LLM extractor's confidence threshold
   (`fact_extraction.min_confidence`). Too low → noisy facts;
   too high → contradictions missed.
2. Check that `Mem0Promoter.contradiction_threshold` isn't too low —
   below 0.85 it can mis-flag unrelated facts as contradictions.
3. Inspect a sample of `superseded_by` edges manually. If they look
   wrong, the LLM contradiction prompt is the lever, not the schema.

### "Retrieval latency spikes to seconds"

Usually one of:

* **Embedder is on the wrong device.** On Macs with LMStudio running,
  the embedder may OOM on MPS and fall back to CPU silently with
  corrupt output. Force `embedding.device=cpu` — see the
  `--embedder-device` CLI flag in the eval and the LongMemEval
  findings § "GPU contention" for the receipts.
* **pgvector index isn't being used.** Run `EXPLAIN ANALYZE` on a
  retrieval query. If you see a sequential scan, your `ivfflat`
  lists parameter is too low (or you skipped migration 002).
* **STM table has grown unbounded.** Enable `trigger` so old turns
  promote out of STM. Production STM should stay under ~10K rows.

### "Background queue is dropping work"

`BackgroundQueue` is bounded (`queue_maxsize=1000` by default). At
that point new jobs are dropped *with warnings, not silently*. If
this is happening:

1. Bump `background_workers` (default 2) to 4-8 — promotion is
   I/O bound, more workers help.
2. Bump `queue_maxsize` if bursts are short — but if depth is
   sustained > 500, you have an actual throughput problem.
3. Reduce promotion frequency by tuning `trigger.mtm_token_threshold`
   upward.

## 8 · Deploying — recipes

### Docker Compose

```yaml
version: "3"
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_PASSWORD: ${PG_PASSWORD}
      POSTGRES_DB: continuum
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports: ["5432:5432"]

  continuum:
    build: .
    environment:
      CONTINUUM_ENV: production
      CONTINUUM_DB_URL: postgresql://postgres:${PG_PASSWORD}@postgres:5432/continuum
      CONTINUUM_DB_POOL_SIZE: "10"
      CONTINUUM_TRIGGER_ENABLED: "true"
      CONTINUUM_PROMOTER_ENABLED: "true"
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    depends_on: [postgres]

volumes:
  pgdata:
```

### Kubernetes

* StatefulSet for Postgres (or use a managed Postgres + Secret for the URL).
* Deployment for Continuum's API surface (FastAPI / your wrapper).
* HorizontalPodAutoscaler scaling on `process_turn_ms.p95`.
* PodDisruptionBudget = 1 for any single-replica services.

## 9 · Where to look in the code

| Concern | Module |
|---|---|
| Postgres pool + connection management | [`continuum/db/`](../continuum/db/) |
| Schema migrations | [`migrations/`](../migrations/) |
| Background queue | [`continuum/core/background.py`](../continuum/core/background.py) |
| Health checks | `ContinuumSession.health()` in [`continuum/core/session.py`](../continuum/core/session.py) |
| Promotion / triggers | [`continuum/promotion/`](../continuum/promotion/) |

## See also

* [Architecture](architecture.md) — why these components exist.
* [Config reference](config.md) — every knob.
* [Quickstart](quickstart.md) — the dev-loop without infra.
