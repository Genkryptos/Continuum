-- =============================================================================
-- 007 — One live row per (namespace, text)
-- =============================================================================
-- Why
-- ---
-- `PostgresLTM.touch_duplicate` collapses a restated fact into the existing row
-- instead of storing a copy, but it cannot see a row that is not yet committed.
-- Writers racing to store a brand-new fact all find nothing and all insert:
-- measured, 8 simultaneous identical writes left 5 rows. Sequential writes —
-- one user, one hook per prompt — already collapse correctly, so this is
-- storage waste rather than a correctness bug, and the read path dedups. It
-- still grows forever.
--
-- Only the database can actually close it, via a uniqueness constraint.
--
-- Why this could not simply be added
-- ----------------------------------
-- The index cannot be built on a table that already holds duplicates, and every
-- store written before `touch_duplicate` existed holds them. Adding the index
-- alone would make `make db-migrate` fail on upgrade, which is worse than the
-- defect. So STEP 1 clears the way first.
--
-- md5(text), not text: a btree entry is capped near 2704 bytes and a memory may
-- be up to 8000 characters. A hash collision would wrongly merge two distinct
-- memories, which is why the write path re-checks the actual text rather than
-- trusting the index alone.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- STEP 1 — Retire pre-existing duplicates. NOTHING IS DELETED.
--
-- Rows are closed with `invalidated_at`, the same bi-temporal retirement
-- supersession and `Memory.forget()` use, so every copy stays on disk and is
-- recoverable by hand. The survivor is the most-reinforced row (highest
-- access_count), tie-broken by the most recent — that is the copy carrying the
-- restatement history, so keeping it preserves the most information.
-- -----------------------------------------------------------------------------
WITH ranked AS (
    SELECT id,
           row_number() OVER (
               PARTITION BY namespace, md5("text")
               ORDER BY access_count DESC, created_at DESC, id
           ) AS rn
    FROM   memory_nodes
    WHERE  invalidated_at IS NULL
)
UPDATE memory_nodes m
SET    invalidated_at = now()
FROM   ranked r
WHERE  m.id = r.id
  AND  r.rn > 1;

-- -----------------------------------------------------------------------------
-- STEP 2 — Enforce it from here on.
--
-- Partial, so retired rows and bi-temporal history are unaffected: a superseded
-- fact and its replacement may share text, and only the live one is constrained.
-- -----------------------------------------------------------------------------
CREATE UNIQUE INDEX IF NOT EXISTS memory_nodes_live_text_uniq
    ON memory_nodes (namespace, md5("text"))
    WHERE invalidated_at IS NULL;

INSERT INTO schema_migrations (version, description)
VALUES ('007', 'One live row per (namespace, text): retire duplicates + partial unique index')
ON CONFLICT (version) DO NOTHING;
