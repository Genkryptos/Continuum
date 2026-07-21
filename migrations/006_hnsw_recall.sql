-- =============================================================================
-- 006 — Rebuild the HNSW index for recall
-- =============================================================================
-- Why
-- ---
-- 002 built the vector index with `m = 16, ef_construction = 64` — pgvector's
-- speed-first defaults. Measured on 50,020 real embedded memories with 20
-- needles whose answers an exact scan ranks at position 1:
--
--   index build              ef_search=400   query p50
--   m=16, ef_construction=64      9/20         4.2 ms
--   m=32, ef_construction=200    12/20         2.1 ms   <- this migration
--   (exact scan, no index)       20/20        71.0 ms
--
-- The denser graph is better on BOTH axes: it finds more and it answers
-- faster, because a well-connected graph converges instead of wandering. The
-- only costs are build time (~12s for 50k rows) and index size.
--
-- Honest limit: HNSW stays *approximate*. Even at m=32 it loses outlier
-- memories among tightly clustered neighbours — the corpus above is close to a
-- worst case (50k formulaic sentences around 20 semantic outliers), and real
-- stores cluster far less brutally, but the shape of the failure is real. If
-- you need exact recall and your store is small enough to afford it (71ms at
-- 50k), drop the index and let Postgres scan.
--
-- The companion query-time knob is `hnsw.ef_search`, which PostgresLTM now sets
-- per query (DEFAULT_HNSW_EF_SEARCH). pgvector's default of 40 is far too low:
-- it found 4/20 here.
--
-- Large live store?
-- -----------------
-- This runs inside the migration's transaction and takes an ACCESS EXCLUSIVE
-- lock for the duration of the build. On a big table, do it by hand instead,
-- outside a transaction, so readers keep working:
--
--   CREATE INDEX CONCURRENTLY memory_nodes_embedding_hnsw_idx_new
--       ON memory_nodes USING hnsw (embedding halfvec_cosine_ops)
--       WITH (m = 32, ef_construction = 200);
--   DROP INDEX CONCURRENTLY memory_nodes_embedding_hnsw_idx;
--   ALTER INDEX memory_nodes_embedding_hnsw_idx_new
--       RENAME TO memory_nodes_embedding_hnsw_idx;
--
-- Build memory: HNSW wants the graph in maintenance_work_mem; if the build
-- spills to disk it gets much slower. Raise it for the build if you can.
-- =============================================================================

SET maintenance_work_mem = '512MB';

DROP INDEX IF EXISTS memory_nodes_embedding_hnsw_idx;

CREATE INDEX IF NOT EXISTS memory_nodes_embedding_hnsw_idx
    ON memory_nodes
    USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 32, ef_construction = 200);

INSERT INTO schema_migrations (version, description)
VALUES ('006', 'HNSW rebuilt with m=32, ef_construction=200 for recall')
ON CONFLICT (version) DO NOTHING;
