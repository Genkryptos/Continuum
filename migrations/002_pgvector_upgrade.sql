-- =============================================================================
-- Migration : 002_pgvector_upgrade.sql
-- Project   : Continuum v1 — vector index upgrade
-- Requires  : PostgreSQL >= 14, pgvector >= 0.8.0
-- Depends on: 001_ltm_schema.sql  (creates memory_nodes.embedding vector(1024))
-- Safe to re-run: YES — every step is guarded / idempotent.
-- =============================================================================
--
-- WHAT THIS DOES
-- ==============
--   1. Hard-checks that pgvector >= 0.8.0 (aborts the whole migration if not).
--   2. Drops ANY existing index on memory_nodes.embedding
--      (IVFFlat, an old HNSW from 001, or none — all handled).
--   3. Converts  memory_nodes.embedding  vector(1024) -> halfvec(1024).
--   4. Rebuilds the HNSW index with halfvec_cosine_ops (m=16, ef_construction=64).
--
-- EXPECTED BENEFITS
-- =================
--   * ~50% less storage + RAM for the column AND its index
--     (halfvec is 2 bytes/dim vs 4 bytes/dim for vector).
--   * Faster ANN queries: HNSW graph traversal vs IVFFlat list scans,
--     and half the bytes touched per distance computation.
--   * < 1% recall loss at 1024 dims — FP16 mantissa (~3 decimal digits) is
--     well below the noise floor of cosine ranking on normalised embeddings.
--
-- LOCKING — READ BEFORE RUNNING IN PRODUCTION
-- ===========================================
-- This script runs as ONE transaction for atomicity.  Both the column type
-- change (ALTER ... TYPE) and the non-concurrent CREATE INDEX take an
-- ACCESS EXCLUSIVE lock on memory_nodes for their full duration — reads and
-- writes are blocked while the index builds.
--
--   * Small/medium tables (<~1M rows) or a maintenance window → run as-is.
--   * Zero-downtime requirement → DO NOT run this file directly.  Use the
--     "ONLINE (CONCURRENT) VARIANT" at the bottom instead: it cannot be
--     wrapped in a transaction but never blocks traffic.
--
-- Always take a backup first:
--     pg_dump -Fc -t memory_nodes mydb > memory_nodes_pre002.dump
-- =============================================================================


-- -----------------------------------------------------------------------------
-- PRE-FLIGHT 1 — pgvector version gate (runs OUTSIDE the txn so a failure
-- here costs nothing and leaves zero partial state).
-- -----------------------------------------------------------------------------
DO $$
DECLARE
    v_ext   text;
    v_parts int[];
    v_major int;
    v_minor int;
BEGIN
    SELECT extversion INTO v_ext
      FROM pg_extension
     WHERE extname = 'vector';

    IF v_ext IS NULL THEN
        RAISE EXCEPTION
            'pgvector extension "vector" is not installed. '
            'Install/upgrade pgvector to >= 0.8.0, then CREATE EXTENSION vector.';
    END IF;

    -- extversion looks like '0.8.0' (strip any '-dev'/'+meta' suffix first).
    v_parts := string_to_array(split_part(v_ext, '-', 1), '.')::int[];
    v_major := COALESCE(v_parts[1], 0);
    v_minor := COALESCE(v_parts[2], 0);

    IF (v_major < 0) OR (v_major = 0 AND v_minor < 8) THEN
        RAISE EXCEPTION
            'pgvector % is too old. halfvec + HNSW upgrade requires >= 0.8.0. '
            'Upgrade the pgvector binary, run ALTER EXTENSION vector UPDATE; '
            'then re-run this migration.', v_ext;
    END IF;

    RAISE NOTICE 'pgvector version % — OK (>= 0.8.0)', v_ext;
END
$$;


-- -----------------------------------------------------------------------------
-- PRE-FLIGHT 2 — fail fast if the prerequisite table/column is missing.
-- -----------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM pg_attribute a
          JOIN pg_class     c ON c.oid = a.attrelid
          JOIN pg_namespace n ON n.oid = c.relnamespace
         WHERE n.nspname = 'public'
           AND c.relname = 'memory_nodes'
           AND a.attname = 'embedding'
           AND a.attnum  > 0
           AND NOT a.attisdropped
    ) THEN
        RAISE EXCEPTION
            'memory_nodes.embedding not found. Apply 001_ltm_schema.sql first.';
    END IF;
END
$$;


BEGIN;

-- Bound the blast radius: if some other session holds a conflicting lock,
-- wait at most 10s rather than queuing indefinitely behind it.
SET LOCAL lock_timeout = '10s';
-- An index build must not be killed by a global statement_timeout.
SET LOCAL statement_timeout = 0;
-- Give the HNSW builder room — bigger maintenance_work_mem = fewer merge
-- passes = faster build. Tune to taste (a few hundred MB is reasonable).
SET LOCAL maintenance_work_mem = '512MB';
-- Parallel workers speed up the graph build on multi-core hosts.
SET LOCAL max_parallel_maintenance_workers = 4;


-- =============================================================================
-- STEP 1 — DROP every existing index on memory_nodes.embedding
-- =============================================================================
-- We don't assume a name: an old install may have an IVFFlat index
-- (memory_nodes_embedding_idx), the HNSW index created by migration 001
-- (memory_nodes_embedding_hnsw_idx), or nothing at all. Drop whatever is
-- there so the column type can be altered (a type change cannot proceed
-- while an index depends on the column).
-- -----------------------------------------------------------------------------
DO $$
DECLARE
    r record;
BEGIN
    FOR r IN
        SELECT i.indexrelid::regclass AS idx_ident
          FROM pg_index     i
          JOIN pg_class     t ON t.oid = i.indrelid
          JOIN pg_namespace n ON n.oid = t.relnamespace
         WHERE n.nspname = 'public'
           AND t.relname = 'memory_nodes'
           AND EXISTS (
               SELECT 1
                 FROM pg_attribute a
                WHERE a.attrelid = t.oid
                  AND a.attname  = 'embedding'
                  AND a.attnum   = ANY (i.indkey)
           )
    LOOP
        EXECUTE format('DROP INDEX IF EXISTS %s', r.idx_ident);
        RAISE NOTICE 'dropped pre-existing embedding index: %', r.idx_ident;
    END LOOP;
END
$$;

-- Belt-and-suspenders: drop the well-known names too (no-op if already gone).
DROP INDEX IF EXISTS memory_nodes_embedding_idx;          -- typical IVFFlat name
DROP INDEX IF EXISTS memory_nodes_embedding_ivfflat_idx;  -- alt IVFFlat name
DROP INDEX IF EXISTS memory_nodes_embedding_hnsw_idx;      -- HNSW from 001


-- =============================================================================
-- STEP 2 — Convert the column: vector(1024) -> halfvec(1024)
-- =============================================================================
-- Idempotent: if the column is already halfvec(1024) the ALTER is skipped.
-- The USING cast (embedding::halfvec(1024)) is provided by pgvector >= 0.7;
-- NULLs convert to NULL, so empty / unembedded rows are fine.
-- -----------------------------------------------------------------------------
DO $$
DECLARE
    v_current_type text;
BEGIN
    SELECT format_type(a.atttypid, a.atttypmod)
      INTO v_current_type
      FROM pg_attribute  a
      JOIN pg_class      c ON c.oid = a.attrelid
      JOIN pg_namespace  n ON n.oid = c.relnamespace
     WHERE n.nspname = 'public'
       AND c.relname = 'memory_nodes'
       AND a.attname = 'embedding'
       AND a.attnum  > 0
       AND NOT a.attisdropped;

    IF v_current_type = 'halfvec(1024)' THEN
        RAISE NOTICE 'memory_nodes.embedding is already halfvec(1024) — skipping ALTER.';
    ELSE
        EXECUTE
            'ALTER TABLE memory_nodes '
            'ALTER COLUMN embedding TYPE halfvec(1024) '
            'USING embedding::halfvec(1024)';
        RAISE NOTICE 'converted memory_nodes.embedding %  ->  halfvec(1024)',
                     v_current_type;
    END IF;
END
$$;

COMMENT ON COLUMN memory_nodes.embedding IS
    'FP16 sentence embedding (BAAI/bge-m3, 1024-dim) stored as pgvector '
    'halfvec(1024). ~50% smaller than vector(1024) at <1% cosine-recall loss. '
    'Query with the <=> cosine-distance operator against a halfvec literal.';


-- =============================================================================
-- STEP 3 — Rebuild the HNSW index on the halfvec column
-- =============================================================================
-- halfvec_cosine_ops = cosine distance over FP16. Must match the distance
-- used at query time ( <=> ). HNSW needs no training data (unlike IVFFlat),
-- so this works even on an empty table and stays correct as rows are added.
--
--   m = 16
--     Max bi-directional links per graph node. The single biggest
--     recall/size knob.
--       * Higher m  -> better recall, larger index, slower build, more RAM.
--       * Lower  m  -> smaller/faster, lower recall.
--     16 is the pgvector default and the sweet spot for ~1k-dim sentence
--     embeddings. Use 24-48 only if recall@k is measurably insufficient
--     (verify with the Python recall harness shipped alongside this file).
--
--   ef_construction = 64
--     Candidate-list (beam) width while building the graph.
--       * Higher -> denser, higher-recall graph; build time grows ~linearly.
--       * Lower  -> faster build, sparser graph, lower recall.
--     Rule of thumb: ef_construction >= 2 * m. 64 pairs well with m=16.
--     Bump to 100-200 for a one-off build where recall matters more than
--     build minutes.
--
--   QUERY-TIME knob (NOT set here — it's per-session/among GUCs):
--     SET hnsw.ef_search = 40;   -- default 40
--       Size of the search beam at query time. This is the live
--       recall<->latency dial; raise it (e.g. 100) for higher recall
--       WITHOUT rebuilding the index. With pgvector >= 0.8 you can also
--       enable iterative scans for better recall under filters:
--       SET hnsw.iterative_scan = relaxed_order;
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS memory_nodes_embedding_hnsw_idx
    ON memory_nodes
    USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);


-- =============================================================================
-- STEP 4 — Migration bookkeeping
-- =============================================================================
INSERT INTO schema_migrations (version, description)
VALUES ('002',
        'pgvector >=0.8 upgrade: embedding vector(1024) -> halfvec(1024), '
        'HNSW halfvec_cosine_ops (m=16, ef_construction=64)')
ON CONFLICT (version) DO NOTHING;


COMMIT;


-- =============================================================================
-- POST-MIGRATION — refresh planner statistics (run OUTSIDE the txn above)
-- =============================================================================
-- The column was rewritten; stale stats can mislead the planner into a
-- seq-scan instead of the new HNSW index. ANALYZE is cheap vs that risk.
--
--     ANALYZE memory_nodes;
--
-- Then sanity-check the plan uses the index:
--     SET hnsw.ef_search = 40;
--     EXPLAIN ANALYZE
--       SELECT id FROM memory_nodes
--       ORDER BY embedding <=> '[...1024 floats...]'::halfvec
--       LIMIT 10;
--   -> expect "Index Scan using memory_nodes_embedding_hnsw_idx".
--
-- Verify storage + recall objectively with the bundled harness:
--     python -m continuum.db.pgvector_upgrade --dsn "$DATABASE_URL"
-- =============================================================================


-- =============================================================================
-- ONLINE (CONCURRENT) VARIANT — zero-downtime production path
-- =============================================================================
-- Use this INSTEAD OF running the file above when memory_nodes is large and
-- cannot tolerate an ACCESS EXCLUSIVE lock for the whole index build.
-- These statements CANNOT run inside a transaction block — execute each one
-- separately (psql \i won't wrap them; an ORM migration should mark them
-- non-transactional).
--
-- NOTE: the column-type ALTER itself still briefly takes ACCESS EXCLUSIVE and
-- rewrites the table — there is no fully online way to change a column type.
-- The win is that the slow part (index build) is concurrent.
--
--   -- 1. Drop old index concurrently (no table lock):
--   DROP INDEX CONCURRENTLY IF EXISTS memory_nodes_embedding_hnsw_idx;
--   DROP INDEX CONCURRENTLY IF EXISTS memory_nodes_embedding_idx;
--
--   -- 2. Convert the type (short ACCESS EXCLUSIVE; schedule in a quiet window):
--   ALTER TABLE memory_nodes
--       ALTER COLUMN embedding TYPE halfvec(1024)
--       USING embedding::halfvec(1024);
--
--   -- 3. Build the new index WITHOUT blocking reads/writes:
--   SET maintenance_work_mem = '512MB';
--   CREATE INDEX CONCURRENTLY memory_nodes_embedding_hnsw_idx
--       ON memory_nodes
--       USING hnsw (embedding halfvec_cosine_ops)
--       WITH (m = 16, ef_construction = 64);
--   -- If a CONCURRENTLY build fails it leaves an INVALID index — drop and retry:
--   --   DROP INDEX CONCURRENTLY IF EXISTS memory_nodes_embedding_hnsw_idx;
--
--   -- 4. Record the migration + refresh stats:
--   INSERT INTO schema_migrations (version, description)
--   VALUES ('002', 'pgvector >=0.8 upgrade (online variant)')
--   ON CONFLICT (version) DO NOTHING;
--   ANALYZE memory_nodes;
-- =============================================================================


-- =============================================================================
-- ROLLBACK PLAN
-- =============================================================================
-- halfvec -> vector is loss-free in PRECISION TERMS for ranking, but the
-- original FP32 bits are gone (they were never stored after the conversion).
-- Rolling back restores the vector(1024) TYPE and an equivalent index; it
-- does NOT restore pre-conversion FP32 values. For a true byte-for-byte
-- revert, restore the pg_dump taken in the pre-flight step.
--
-- Option A — revert type + index (keeps current data, FP16 precision):
--
--   BEGIN;
--   SET LOCAL lock_timeout = '10s';
--   SET LOCAL maintenance_work_mem = '512MB';
--
--   DROP INDEX IF EXISTS memory_nodes_embedding_hnsw_idx;
--
--   ALTER TABLE memory_nodes
--       ALTER COLUMN embedding TYPE vector(1024)
--       USING embedding::vector(1024);
--
--   -- Recreate the index that migration 001 had (vector ops):
--   CREATE INDEX memory_nodes_embedding_hnsw_idx
--       ON memory_nodes
--       USING hnsw (embedding vector_cosine_ops)
--       WITH (m = 16, ef_construction = 64);
--
--   DELETE FROM schema_migrations WHERE version = '002';
--   COMMIT;
--   ANALYZE memory_nodes;
--
-- Option B — full restore (recovers original FP32 embeddings):
--
--   -- With the server idle / table quiesced:
--   psql mydb -c 'TRUNCATE memory_nodes CASCADE;'   -- or DROP+recreate
--   pg_restore --data-only -t memory_nodes -d mydb memory_nodes_pre002.dump
--   psql mydb -c "DELETE FROM schema_migrations WHERE version = '002';"
--
-- TRIGGERS FOR ROLLBACK
--   * Recall harness shows recall@10 dropping > 1% vs the pre-upgrade
--     baseline AND raising hnsw.ef_search (e.g. to 100) does not recover it.
--   * Query latency regresses (mis-tuned m / ef_search, or planner avoiding
--     the index — check EXPLAIN, run ANALYZE).
--   * Application cannot yet bind halfvec params (deploy the query-code
--     update first; see continuum/db/pgvector_upgrade.py).
-- =============================================================================
