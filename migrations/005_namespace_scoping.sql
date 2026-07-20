-- =============================================================================
-- Migration : 005_namespace_scoping.sql
-- Project   : Continuum v2 — tenant scoping for memory_nodes
-- Requires  : PostgreSQL >= 14
-- Depends on: 001 (memory_nodes)
-- Safe to re-run: YES — every statement is IF NOT EXISTS / additive.
-- =============================================================================
--
-- Adds a ``namespace`` scope key so one database can hold isolated stores.
-- Before this, memory_nodes had no tenant column and no query filtered by one,
-- so ``recall`` and ``current`` returned facts from EVERY session/user/project
-- sharing the database. The column is NOT NULL DEFAULT 'default': all existing
-- rows land in the 'default' namespace, and legacy code (which never sets it)
-- keeps seeing exactly that one namespace — non-destructive and backward
-- compatible.
--
-- The store filters every read (search_hybrid, by_tags, neighbors) and stamps
-- every write with its configured namespace.
-- =============================================================================

BEGIN;

-- -----------------------------------------------------------------------------
-- STEP 1 — Scope column (additive, defaulted so all existing rows are valid)
-- -----------------------------------------------------------------------------
ALTER TABLE memory_nodes
    ADD COLUMN IF NOT EXISTS namespace TEXT NOT NULL DEFAULT 'default';

-- -----------------------------------------------------------------------------
-- STEP 2 — Indexes. Every scoped query filters on namespace first, so it must
-- lead the composite indexes to stay selective.
-- -----------------------------------------------------------------------------

-- Live-row lookups (search_hybrid over-fetch, by_tags, neighbors) all filter
-- namespace + not-yet-invalidated. A partial btree keeps that selective; the
-- planner bitmap-ANDs it with the existing tags GIN for by_tags, so no
-- btree_gin extension is needed.
CREATE INDEX IF NOT EXISTS memory_nodes_ns_live_idx
    ON memory_nodes (namespace)
    WHERE invalidated_at IS NULL;

COMMIT;
