-- =============================================================================
-- Migration : 003_lexical_search.sql
-- Project   : Continuum v1 — lexical (trigram) search for hybrid retrieval
-- Requires  : PostgreSQL >= 14
-- Depends on: 001_ltm_schema.sql  (memory_nodes.text, invalidated_at)
-- Safe to re-run: YES — every statement is IF NOT EXISTS / idempotent.
-- =============================================================================
--
-- WHY
-- ===
-- Dense vector search (pgvector / halfvec, migration 002) is great at
-- semantic similarity but blind to exact tokens: identifiers, error codes,
-- product names, rare proper nouns, and typo'd-but-literal queries. pg_trgm
-- adds a sparse LEXICAL signal so the Retriever can fuse both (RRF) for
-- materially better recall than either alone.
--
-- RELATIONSHIP TO 001
-- ===================
-- 001 already runs `CREATE EXTENSION IF NOT EXISTS pg_trgm` and creates
-- `memory_nodes_text_trgm_idx`. This migration is intentionally a SUPERSET
-- of that: it re-declares them idempotently (so a DB that skipped/edited
-- 001 still converges), adds a LIVE-rows partial index, and documents the
-- EXPLAIN evidence + operator semantics the Python retrieval layer relies on.
--
-- TRIGRAM PRIMER
-- ==============
-- pg_trgm breaks text into 3-char shingles ("hello" -> {"  h"," he","hel",
-- "ell","llo","lo "}). similarity(a,b) = |trigrams(a) ∩ trigrams(b)| /
-- |trigrams(a) ∪ trigrams(b)|  ∈ [0,1]. Because matching is on shared
-- shingles, a one-character typo only drops a couple of trigrams, so fuzzy
-- queries still rank their intended target highly.
--
-- Operators (all GIN-accelerated by gin_trgm_ops):
--   a %  b   → boolean: similarity(a,b) >= pg_trgm.similarity_threshold
--   a <-> b  → real:    1 - similarity(a,b)         (distance; ORDER BY this)
--   ILIKE / LIKE '%x%'  → also uses the trigram index
-- =============================================================================

BEGIN;

-- -----------------------------------------------------------------------------
-- STEP 1 — Extension
-- -----------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- -----------------------------------------------------------------------------
-- STEP 2 — GIN trigram indexes on memory_nodes.text
-- -----------------------------------------------------------------------------
-- GIN + gin_trgm_ops is the right call for read-heavy LTM:
--   * GIN  → smaller, faster lookups, slightly slower writes. Ideal here:
--            facts are written once and read many times.
--   * GiST → supports k-NN ordering by <-> as an index scan and is cheaper
--            to update, but larger and slower to probe. Prefer only for
--            high-churn text or when you need index-ordered <-> streaming.
-- The column is quoted ("text") because `text` is also a type name.
CREATE INDEX IF NOT EXISTS memory_nodes_text_trgm_idx
    ON memory_nodes
    USING gin ("text" gin_trgm_ops);

-- A partial twin restricted to LIVE rows. Lexical retrieval always filters
-- `invalidated_at IS NULL`, so this smaller index is what the planner will
-- prefer for the hot path (historical / superseded rows are excluded).
CREATE INDEX IF NOT EXISTS memory_nodes_text_trgm_live_idx
    ON memory_nodes
    USING gin ("text" gin_trgm_ops)
    WHERE invalidated_at IS NULL;

-- -----------------------------------------------------------------------------
-- STEP 3 — Migration bookkeeping
-- -----------------------------------------------------------------------------
INSERT INTO schema_migrations (version, description)
VALUES ('003',
        'Lexical search: pg_trgm + GIN(gin_trgm_ops) on memory_nodes.text '
        '(full + live-partial) for hybrid RRF retrieval')
ON CONFLICT (version) DO NOTHING;

COMMIT;


-- =============================================================================
-- POST-MIGRATION — refresh planner statistics (run OUTSIDE the txn)
-- =============================================================================
--     ANALYZE memory_nodes;
--
-- SIMILARITY THRESHOLD (the `%` operator's cutoff)
-- ------------------------------------------------
-- Default is 0.3. Continuum's retrieval layer lowers it PER SESSION so
-- genuine typo matches survive the boolean prefilter, without changing the
-- global default for other workloads:
--
--     SET LOCAL pg_trgm.similarity_threshold = 0.25;   -- continuum/.../retrieval.py
--
-- (A DB-wide default, if you want one, is set out of band by an operator:
--  `ALTER DATABASE <dbname> SET pg_trgm.similarity_threshold = 0.25;`
--  — it needs the literal database name so it is intentionally NOT in this
--  migration.)


-- =============================================================================
-- EXPLAIN ANALYZE — proof the GIN trigram index is used
-- =============================================================================
-- Run these against a populated table. Expected highlights are noted; the
-- key line is "Bitmap Index Scan on memory_nodes_text_trgm*_idx".
--
-- (1) Fuzzy ranked search — the exact shape lexical_search() issues:
--
--     SET pg_trgm.similarity_threshold = 0.25;
--     EXPLAIN (ANALYZE, BUFFERS)
--     SELECT id, similarity("text", 'postgres conection pooling') AS sim
--     FROM   memory_nodes
--     WHERE  invalidated_at IS NULL
--       AND  "text" % 'postgres conection pooling'   -- note the typo
--     ORDER  BY sim DESC
--     LIMIT  10;
--
--   Expected plan (shape):
--     Limit
--       ->  Sort                       (Sort Key: (similarity(...)) DESC)
--             ->  Bitmap Heap Scan on memory_nodes
--                   Recheck Cond: ("text" % 'postgres conection pooling')
--                   Filter: (invalidated_at IS NULL)
--                   ->  Bitmap Index Scan on memory_nodes_text_trgm_live_idx
--                         Index Cond: ("text" % 'postgres conection pooling')
--   → Bitmap Index Scan == the GIN trigram index served the candidate set;
--     a "Seq Scan on memory_nodes" instead means the index was NOT used.
--
-- (2) Substring / ILIKE also rides the trigram index:
--
--     EXPLAIN (ANALYZE, BUFFERS)
--     SELECT id FROM memory_nodes
--     WHERE invalidated_at IS NULL AND "text" ILIKE '%conection%';
--
--   Expected: Bitmap Index Scan on memory_nodes_text_trgm_live_idx.
--
-- (3) Distance-ordered variant (alternative ranking):
--
--     EXPLAIN (ANALYZE, BUFFERS)
--     SELECT id, "text" <-> 'conection pooling' AS dist
--     FROM   memory_nodes
--     WHERE  invalidated_at IS NULL AND "text" % 'conection pooling'
--     ORDER  BY dist
--     LIMIT  10;
--
-- Forcing a comparison (demonstration only — do not leave this set):
--     SET enable_seqscan = off;   -- then re-run (1); compare cost/time
--     RESET enable_seqscan;
-- =============================================================================


-- =============================================================================
-- ROLLBACK PLAN
-- =============================================================================
-- The trigram indexes are pure accelerators — dropping them only makes
-- lexical queries slower (seq scan), it does not lose data. The extension
-- may be shared by other migrations/objects, so dropping it is OPTIONAL and
-- guarded.
--
--   BEGIN;
--   DROP INDEX IF EXISTS memory_nodes_text_trgm_live_idx;
--   -- Only drop the base index if you are NOT relying on 001's copy:
--   -- DROP INDEX IF EXISTS memory_nodes_text_trgm_idx;
--   DELETE FROM schema_migrations WHERE version = '003';
--   COMMIT;
--
--   -- Optional, ONLY if nothing else uses pg_trgm (will error if dependent
--   -- objects exist unless CASCADE — do not CASCADE blindly):
--   -- DROP EXTENSION IF EXISTS pg_trgm RESTRICT;
--
-- TRIGGERS FOR ROLLBACK
--   * Write throughput regresses unacceptably (GIN maintenance cost) on a
--     suddenly write-heavy memory_nodes — reconsider GiST or drop the
--     live-partial twin.
--   * EXPLAIN shows the planner ignoring the index AND ANALYZE + threshold
--     tuning do not fix it.
-- =============================================================================
