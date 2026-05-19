-- =============================================================================
-- Migration : 001_ltm_schema.sql
-- Project   : Continuum v1 — LTM schema upgrade
-- Author    : Continuum team
-- Requires  : PostgreSQL ≥ 14, pgvector ≥ 0.5.0
-- Safe to re-run: YES (all changes are idempotent)
-- =============================================================================
--
-- BI-TEMPORAL DESIGN
-- ==================
-- Every LTM record tracks two independent time axes:
--
--   VALID TIME (world time)   — when the fact was true in the real world.
--     valid_from  : The moment the fact became true.
--     valid_to    : The moment it stopped being true (NULL = still true).
--
--   TRANSACTION TIME (system time) — when Continuum stored/invalidated it.
--     created_at    : When the row was first inserted (existing column).
--     invalidated_at: When the system marked this version obsolete.
--                     NULL = current live version.
--
-- This separation lets us answer two distinct questions:
--   "What did we KNOW at time T?"          → filter on created_at ≤ T AND (invalidated_at IS NULL OR invalidated_at > T)
--   "What was TRUE at time T in the world?" → filter on valid_from ≤ T AND (valid_to IS NULL OR valid_to > T)
--
-- EMBEDDING NOTES
-- ===============
-- This migration uses vector(1024) which is compatible with pgvector ≥ 0.5.0.
-- To upgrade to the more memory-efficient halfvec(1024) (50 % storage saving),
-- upgrade pgvector to ≥ 0.7.0 first, then apply the PHASE 2 block at the
-- bottom of this file.
--
-- =============================================================================

BEGIN;

-- =============================================================================
-- EXTENSIONS
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector: vector similarity
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- trigram indexes for fuzzy text search
CREATE EXTENSION IF NOT EXISTS "uuid-ossp"; -- uuid_generate_v4() fallback
                                            -- (gen_random_uuid() is built-in on PG ≥ 13)

-- =============================================================================
-- MIGRATION BOOKKEEPING
-- =============================================================================
-- Tracks which migrations have been applied.  Other tooling (Alembic, Flyway)
-- can replace this table — it is used only when running migrations manually.

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT        PRIMARY KEY,
    description TEXT        NOT NULL,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO schema_migrations (version, description)
VALUES ('001', 'LTM schema upgrade: bi-temporal columns, HNSW index, episodes table')
ON CONFLICT (version) DO NOTHING;


-- =============================================================================
-- STEP 1 — ESTABLISH V0 BASELINES (no-ops if tables already exist)
-- =============================================================================
-- These CREATE TABLE IF NOT EXISTS statements establish the minimum v0 schema
-- so that the ALTER statements in STEP 2+ are always valid, regardless of
-- whether this is a fresh install or an upgrade of an existing v0 database.

CREATE TABLE IF NOT EXISTS memory_nodes (
    id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    layer      TEXT,
    text       TEXT,
    embedding  vector(1024),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memory_edges (
    id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id  UUID        NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
    target_id  UUID        NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memory_access_log (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id     UUID        NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
    accessed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memory_promotions (
    id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);


-- =============================================================================
-- STEP 2 — ALTER memory_nodes (all non-destructive, idempotent)
-- =============================================================================

-- ── Classification ───────────────────────────────────────────────────────────

ALTER TABLE memory_nodes
    ADD COLUMN IF NOT EXISTS kind TEXT
        CHECK (kind IN (
            'fact',         -- a discrete, verifiable piece of knowledge
            'entity',       -- a named thing (person, project, library …)
            'episode',      -- a conversation excerpt promoted from MTM
            'preference',   -- a user or agent preference
            'decision',     -- a documented architectural / product decision
            'summary',      -- a compressed digest of several episodes
            'code_symbol'   -- a code entity (function, class, module …)
        ));

-- ── Scoring dimensions (see ScorerProtocol) ──────────────────────────────────

ALTER TABLE memory_nodes
    ADD COLUMN IF NOT EXISTS confidence REAL
        CHECK (confidence >= 0.0 AND confidence <= 1.0);
        -- 0.0 = highly uncertain / inferred   1.0 = directly observed / verified

ALTER TABLE memory_nodes
    ADD COLUMN IF NOT EXISTS importance REAL
        CHECK (importance >= 0.0 AND importance <= 1.0);
        -- Normalised from the 4-level STM Importance enum:
        --   LOW=0.25  NORMAL=0.50  HIGH=0.75  CRITICAL=1.0

-- ── Access statistics ────────────────────────────────────────────────────────

ALTER TABLE memory_nodes
    ADD COLUMN IF NOT EXISTS access_count INT NOT NULL DEFAULT 0;

ALTER TABLE memory_nodes
    ADD COLUMN IF NOT EXISTS last_access TIMESTAMPTZ;

-- ── Bi-temporal: VALID TIME (when the fact was true in the world) ─────────────
-- See design note at the top of this file.

ALTER TABLE memory_nodes
    ADD COLUMN IF NOT EXISTS valid_from TIMESTAMPTZ;
        -- NULL on creation means "valid since we first recorded it";
        -- populate explicitly when the world-time is known.

ALTER TABLE memory_nodes
    ADD COLUMN IF NOT EXISTS valid_to TIMESTAMPTZ;
        -- NULL = the fact is still true as of now.
        -- Set when a fact is superseded (e.g. user changed their preference).

-- ── Bi-temporal: TRANSACTION TIME (when the system marked it obsolete) ────────

ALTER TABLE memory_nodes
    ADD COLUMN IF NOT EXISTS invalidated_at TIMESTAMPTZ;
        -- NULL  = this is the current live version of the fact.
        -- Set   = a newer version exists; this row is historical.
        -- Never DELETE rows; set invalidated_at instead.

-- ── Provenance ───────────────────────────────────────────────────────────────

ALTER TABLE memory_nodes
    ADD COLUMN IF NOT EXISTS source_ids UUID[];
        -- IDs of the memory_episodes rows this fact was promoted from.
        -- Enables "show me the original episodes that led to this belief".

-- ── Metadata / tags ──────────────────────────────────────────────────────────

ALTER TABLE memory_nodes
    ADD COLUMN IF NOT EXISTS tags JSONB NOT NULL DEFAULT '{}';
        -- Arbitrary key-value annotations.  Examples:
        --   code_symbol: {"lang":"python","path":"src/foo.py","symbol":"MyClass"}
        --   preference:  {"domain":"ui","agent_id":"agent-42"}

-- ── Embedding dimension normalisation ────────────────────────────────────────
-- Ensures the embedding column is vector(1024).  Uses a DO block so we can
-- inspect the current dimension before deciding whether to ALTER.
-- WARNING: This rewrites the column; run VACUUM ANALYZE afterwards.

DO $$
DECLARE
    v_atttypmod  int;
    v_target_dim int := 1024;
    -- pgvector stores the raw dimension directly in atttypmod (unlike
    -- varchar/text which adds 4).  So vector(1024) → atttypmod = 1024.
BEGIN
    SELECT a.atttypmod
      INTO v_atttypmod
      FROM pg_attribute  a
      JOIN pg_class      c ON c.oid = a.attrelid
     WHERE c.relname  = 'memory_nodes'
       AND c.relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
       AND a.attname  = 'embedding'
       AND a.attnum   > 0
       AND NOT a.attisdropped;

    -- Column absent → nothing to do (ADD COLUMN in STEP 1 already set the type)
    IF v_atttypmod IS NULL THEN
        RAISE NOTICE 'memory_nodes.embedding not found; skipping dimension check.';
        RETURN;
    END IF;

    -- Already the target dimension → skip
    IF v_atttypmod = v_target_dim THEN
        RAISE NOTICE 'memory_nodes.embedding is already vector(%); no change.', v_target_dim;
        RETURN;
    END IF;

    -- -1 means untyped vector (no fixed dim) → safe to set dimension
    IF v_atttypmod = -1 THEN
        RAISE NOTICE 'Fixing untyped vector → vector(%) on memory_nodes.embedding.', v_target_dim;
        EXECUTE format(
            'ALTER TABLE memory_nodes ALTER COLUMN embedding TYPE vector(%s) USING embedding::vector(%s)',
            v_target_dim, v_target_dim
        );
        RETURN;
    END IF;

    -- Different fixed dimension → requires manual intervention; warn the operator.
    RAISE WARNING
        'memory_nodes.embedding has dimension % (expected %). '
        'Automatic conversion is risky when rows contain data. '
        'Back up the table, then run manually: '
        'ALTER TABLE memory_nodes ALTER COLUMN embedding TYPE vector(1024) '
        'USING embedding::vector(1024);',
        v_atttypmod, v_target_dim;
END
$$;


-- =============================================================================
-- STEP 3 — ALTER memory_edges
-- =============================================================================

ALTER TABLE memory_edges
    ADD COLUMN IF NOT EXISTS predicate TEXT;
        -- Relationship type in SCREAMING_SNAKE_CASE.  Examples:
        --   CALLS, IMPORTS, EXTENDS, IMPLEMENTS, REFERENCES,
        --   DEPENDS_ON, CONTRADICTS, SUPPORTS, SUPERSEDES

ALTER TABLE memory_edges
    ADD COLUMN IF NOT EXISTS fact_id UUID
        REFERENCES memory_nodes(id) ON DELETE SET NULL;
        -- The LTM fact node this edge was derived from (optional).
        -- Enables "show me all graph edges that support this belief".

ALTER TABLE memory_edges
    ADD COLUMN IF NOT EXISTS weight REAL
        CHECK (weight >= 0.0 AND weight <= 1.0);
        -- Edge strength / confidence; 1.0 = certain, 0.0 = speculative.

-- Bi-temporal columns (same semantics as memory_nodes — see design note)
ALTER TABLE memory_edges
    ADD COLUMN IF NOT EXISTS valid_from     TIMESTAMPTZ;

ALTER TABLE memory_edges
    ADD COLUMN IF NOT EXISTS valid_to       TIMESTAMPTZ;

ALTER TABLE memory_edges
    ADD COLUMN IF NOT EXISTS invalidated_at TIMESTAMPTZ;


-- =============================================================================
-- STEP 4 — ALTER memory_access_log
-- =============================================================================
-- The access log is append-only; no existing columns are modified.
-- Adding agent_id and kind lets analytics queries filter by layer/agent.

ALTER TABLE memory_access_log
    ADD COLUMN IF NOT EXISTS agent_id  UUID;
ALTER TABLE memory_access_log
    ADD COLUMN IF NOT EXISTS kind      TEXT;  -- mirrors memory_nodes.kind


-- =============================================================================
-- STEP 5 — ALTER memory_promotions
-- =============================================================================

ALTER TABLE memory_promotions
    ADD COLUMN IF NOT EXISTS op TEXT
        CHECK (op IN (
            'ADD',      -- a new LTM fact was created
            'UPDATE',   -- an existing LTM fact was revised
            'DELETE',   -- a fact was invalidated (valid_to closed)
            'NOOP'      -- the LLM decided the candidate was not promotion-worthy
        ));

ALTER TABLE memory_promotions
    ADD COLUMN IF NOT EXISTS candidate_text TEXT;
        -- The raw MTM content that was evaluated for promotion.

ALTER TABLE memory_promotions
    ADD COLUMN IF NOT EXISTS target_id UUID
        REFERENCES memory_nodes(id) ON DELETE SET NULL;
        -- The memory_nodes row that was created/updated (NULL for NOOP/DELETE).

ALTER TABLE memory_promotions
    ADD COLUMN IF NOT EXISTS llm_model     TEXT;
        -- The model that produced the promotion decision (e.g. "gpt-4o-mini").

ALTER TABLE memory_promotions
    ADD COLUMN IF NOT EXISTS llm_rationale TEXT;
        -- The model's natural-language justification for its decision.

ALTER TABLE memory_promotions
    ADD COLUMN IF NOT EXISTS tokens_in  INT;  -- prompt tokens consumed
ALTER TABLE memory_promotions
    ADD COLUMN IF NOT EXISTS tokens_out INT;  -- completion tokens generated


-- =============================================================================
-- STEP 6 — CREATE memory_episodes (new table)
-- =============================================================================
-- Episodes are the raw material that flows from STM → MTM → LTM.
-- They are never modified once written (append-only), so there are no
-- bi-temporal columns — just the ingestion timestamp and the original
-- world-time (occurred_at) if known.

CREATE TABLE IF NOT EXISTS memory_episodes (
    -- Identity
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Routing
    session_id  UUID        NOT NULL,   -- which conversation this came from
    agent_id    UUID        NOT NULL,   -- which agent observed it

    -- Content
    content     TEXT        NOT NULL,   -- the raw text of the episode
    meta        JSONB       NOT NULL DEFAULT '{}',
                -- Arbitrary metadata.  Common keys:
                --   role       : "user" | "assistant" | "system"
                --   source     : "stm_evict" | "stm_compress" | "manual"
                --   importance : 1-4 (raw STM Importance enum value)
                --   token_count: number of tokens in content

    -- Time
    occurred_at  TIMESTAMPTZ,           -- world-time: when the event happened
                                        -- (NULL if only ingestion time is known)
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                                        -- system-time: when we stored it
);

COMMENT ON TABLE memory_episodes IS
    'Append-only log of raw conversational episodes promoted from STM/MTM. '
    'These are the source material for LTM fact extraction. Never UPDATE or DELETE rows.';

COMMENT ON COLUMN memory_episodes.occurred_at IS
    'When the event actually happened in the world. '
    'Distinct from ingested_at (when Continuum stored it). '
    'Use this for temporal queries about what was said/known at a given time.';


-- =============================================================================
-- STEP 7 — INDEXES
-- =============================================================================
-- All index names are prefixed with the table name so pg_indexes is readable.
-- All CREATE INDEX use IF NOT EXISTS (requires PG 9.5+).

-- ── memory_nodes: HNSW vector similarity ─────────────────────────────────────
-- Replaces the IVFFlat index if it exists.  HNSW offers better recall and
-- does not require a training step (no need to pre-populate the table).
-- m=16: 16 bi-directional links per node (higher = better recall, more RAM)
-- ef_construction=64: beam width during index build (trade-off: recall vs speed)

CREATE INDEX IF NOT EXISTS memory_nodes_embedding_hnsw_idx
    ON memory_nodes
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ── memory_nodes: full-text trigram search ────────────────────────────────────
-- Enables LIKE / ILIKE / similarity() queries on the text column without a
-- sequential scan.  Requires pg_trgm (installed above).

CREATE INDEX IF NOT EXISTS memory_nodes_text_trgm_idx
    ON memory_nodes
    USING gin ("text" gin_trgm_ops);

-- ── memory_nodes: live-rows-only partial indexes ──────────────────────────────
-- These covering indexes skip historical/invalidated rows, so the query
-- planner can satisfy "WHERE invalidated_at IS NULL" with an index scan.

CREATE INDEX IF NOT EXISTS memory_nodes_live_kind_idx
    ON memory_nodes (kind, created_at DESC)
    WHERE invalidated_at IS NULL;
    -- Supports: SELECT … WHERE kind = 'fact' AND invalidated_at IS NULL
    --           ORDER BY created_at DESC

CREATE INDEX IF NOT EXISTS memory_nodes_live_importance_idx
    ON memory_nodes (importance DESC)
    WHERE invalidated_at IS NULL;
    -- Supports top-K retrieval by importance on live rows only.

-- ── memory_nodes: bi-temporal range lookup ────────────────────────────────────
-- Used by point-in-time queries: "what was true at timestamp T?"

CREATE INDEX IF NOT EXISTS memory_nodes_valid_from_idx
    ON memory_nodes (valid_from)
    WHERE invalidated_at IS NULL;

CREATE INDEX IF NOT EXISTS memory_nodes_valid_to_idx
    ON memory_nodes (valid_to)
    WHERE invalidated_at IS NULL AND valid_to IS NOT NULL;

-- ── memory_nodes: JSONB tags ──────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS memory_nodes_tags_gin_idx
    ON memory_nodes
    USING gin (tags);
    -- Supports: WHERE tags @> '{"lang":"python"}'

-- ── memory_edges: traversal by source and target ─────────────────────────────

CREATE INDEX IF NOT EXISTS memory_edges_source_live_idx
    ON memory_edges (source_id, predicate)
    WHERE invalidated_at IS NULL;
    -- "all live out-edges of node X with predicate P"

CREATE INDEX IF NOT EXISTS memory_edges_target_live_idx
    ON memory_edges (target_id, predicate)
    WHERE invalidated_at IS NULL;
    -- "all live in-edges to node X with predicate P"

-- ── memory_episodes: retrieval by session / agent / time ─────────────────────

CREATE INDEX IF NOT EXISTS memory_episodes_session_time_idx
    ON memory_episodes (session_id, ingested_at DESC);

CREATE INDEX IF NOT EXISTS memory_episodes_agent_time_idx
    ON memory_episodes (agent_id, ingested_at DESC);

CREATE INDEX IF NOT EXISTS memory_episodes_meta_gin_idx
    ON memory_episodes
    USING gin (meta);

-- ── memory_promotions: audit queries ─────────────────────────────────────────

CREATE INDEX IF NOT EXISTS memory_promotions_target_idx
    ON memory_promotions (target_id)
    WHERE target_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS memory_promotions_op_time_idx
    ON memory_promotions (op, created_at DESC);


-- =============================================================================
-- STEP 8 — TABLE & COLUMN COMMENTS
-- =============================================================================

COMMENT ON TABLE memory_nodes IS
    'LTM fact store. Each row is one version of a fact. '
    'Live facts have invalidated_at IS NULL. '
    'Historical facts (superseded, contradicted, corrected) have invalidated_at set. '
    'Never DELETE rows — set invalidated_at instead.';

COMMENT ON COLUMN memory_nodes.valid_from IS
    'VALID TIME start: when the described fact became true in the world. '
    'NULL means "at least as far back as created_at".';

COMMENT ON COLUMN memory_nodes.valid_to IS
    'VALID TIME end: when the described fact stopped being true in the world. '
    'NULL = still true as of now (open-ended interval).';

COMMENT ON COLUMN memory_nodes.invalidated_at IS
    'TRANSACTION TIME: when this version was superseded in Continuum''s store. '
    'NULL = current live version. '
    'When promoting an UPDATE, set invalidated_at = now() on the old row '
    'and INSERT a new row with the revised content.';

COMMENT ON COLUMN memory_nodes.source_ids IS
    'UUIDs of the memory_episodes rows whose content led to this fact. '
    'Populated by the Promoter; used for provenance traces.';

COMMENT ON TABLE memory_edges IS
    'Knowledge-graph edges between memory_nodes. '
    'Live edges have invalidated_at IS NULL. '
    'predicate uses SCREAMING_SNAKE_CASE (CALLS, IMPORTS, SUPPORTS, …).';

COMMENT ON TABLE memory_promotions IS
    'Audit log of every MTM→LTM promotion decision made by the Promoter. '
    'One row per evaluation, whether the decision was ADD, UPDATE, DELETE, or NOOP.';


-- =============================================================================
-- COMMIT
-- =============================================================================

COMMIT;


-- =============================================================================
-- PHASE 2 — halfvec upgrade (requires pgvector ≥ 0.7.0)
-- =============================================================================
-- Run this SEPARATELY after upgrading pgvector.  Do NOT include in this
-- transaction — type changes acquire an ACCESS EXCLUSIVE lock and may timeout.
--
-- Check your version first:
--   SELECT extversion FROM pg_extension WHERE extname = 'vector';
--
-- Steps:
--   1. DROP the HNSW index (cannot coexist with a type change).
--   2. ALTER the column type.
--   3. Recreate the index using halfvec_cosine_ops.
--
-- BEGIN;
--
-- DROP INDEX IF EXISTS memory_nodes_embedding_hnsw_idx;
--
-- ALTER TABLE memory_nodes
--     ALTER COLUMN embedding TYPE halfvec(1024)
--     USING embedding::halfvec(1024);
--
-- CREATE INDEX memory_nodes_embedding_hnsw_idx
--     ON memory_nodes
--     USING hnsw (embedding halfvec_cosine_ops)
--     WITH (m = 16, ef_construction = 64);
--
-- COMMIT;
-- =============================================================================


-- =============================================================================
-- ROLLBACK GUIDE
-- =============================================================================
-- To undo this migration cleanly, run the block below.
-- Note: DROP COLUMN is irreversible if rows have been inserted.
-- Always take a pg_dump backup before applying migrations in production.
--
-- BEGIN;
--
-- -- Drop new table
-- DROP TABLE IF EXISTS memory_episodes;
--
-- -- Drop new indexes (DROP INDEX is safe even without IF NOT EXISTS in PG 15)
-- DROP INDEX IF EXISTS memory_nodes_embedding_hnsw_idx;
-- DROP INDEX IF EXISTS memory_nodes_text_trgm_idx;
-- DROP INDEX IF EXISTS memory_nodes_live_kind_idx;
-- DROP INDEX IF EXISTS memory_nodes_live_importance_idx;
-- DROP INDEX IF EXISTS memory_nodes_valid_from_idx;
-- DROP INDEX IF EXISTS memory_nodes_valid_to_idx;
-- DROP INDEX IF EXISTS memory_nodes_tags_gin_idx;
-- DROP INDEX IF EXISTS memory_edges_source_live_idx;
-- DROP INDEX IF EXISTS memory_edges_target_live_idx;
-- DROP INDEX IF EXISTS memory_episodes_session_time_idx;
-- DROP INDEX IF EXISTS memory_episodes_agent_time_idx;
-- DROP INDEX IF EXISTS memory_episodes_meta_gin_idx;
-- DROP INDEX IF EXISTS memory_promotions_target_idx;
-- DROP INDEX IF EXISTS memory_promotions_op_time_idx;
--
-- -- Remove columns added to memory_nodes
-- ALTER TABLE memory_nodes
--     DROP COLUMN IF EXISTS kind,
--     DROP COLUMN IF EXISTS confidence,
--     DROP COLUMN IF EXISTS importance,
--     DROP COLUMN IF EXISTS access_count,
--     DROP COLUMN IF EXISTS last_access,
--     DROP COLUMN IF EXISTS valid_from,
--     DROP COLUMN IF EXISTS valid_to,
--     DROP COLUMN IF EXISTS invalidated_at,
--     DROP COLUMN IF EXISTS source_ids,
--     DROP COLUMN IF EXISTS tags;
--
-- -- Remove columns added to memory_edges
-- ALTER TABLE memory_edges
--     DROP COLUMN IF EXISTS predicate,
--     DROP COLUMN IF EXISTS fact_id,
--     DROP COLUMN IF EXISTS weight,
--     DROP COLUMN IF EXISTS valid_from,
--     DROP COLUMN IF EXISTS valid_to,
--     DROP COLUMN IF EXISTS invalidated_at;
--
-- -- Remove columns added to memory_access_log
-- ALTER TABLE memory_access_log
--     DROP COLUMN IF EXISTS agent_id,
--     DROP COLUMN IF EXISTS kind;
--
-- -- Remove columns added to memory_promotions
-- ALTER TABLE memory_promotions
--     DROP COLUMN IF EXISTS op,
--     DROP COLUMN IF EXISTS candidate_text,
--     DROP COLUMN IF EXISTS target_id,
--     DROP COLUMN IF EXISTS llm_model,
--     DROP COLUMN IF EXISTS llm_rationale,
--     DROP COLUMN IF EXISTS tokens_in,
--     DROP COLUMN IF EXISTS tokens_out;
--
-- -- Remove bookkeeping row
-- DELETE FROM schema_migrations WHERE version = '001';
--
-- COMMIT;
-- =============================================================================
