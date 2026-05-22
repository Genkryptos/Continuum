-- =============================================================================
-- Migration : 004_policy_engine.sql
-- Project   : Continuum v1 — Policy-Based Memory Lifecycle Engine
-- Requires  : PostgreSQL >= 14
-- Depends on: 001 (memory_nodes / memory_promotions),
--             002 (halfvec — optional but recommended),
--             003 (pg_trgm — optional but recommended).
-- Safe to re-run: YES — every statement is IF NOT EXISTS / additive.
-- =============================================================================
--
-- This migration is **non-destructive**. It widens ``memory_nodes`` with the
-- behavioural attributes the Policy Engine writes (candidate_type, urgency,
-- volatility, sensitivity, source_authority, retention/retrieval/privacy
-- JSONB blocks, expires_at, supersedes link, status) and adds two new
-- tables for governance:
--
--   * ``memory_decision_traces``    — one row per evaluated candidate
--   * ``memory_pending_approvals``  — items the engine needs human sign-off on
--
-- The columns are all nullable / defaulted so EVERY existing row stays
-- valid: legacy code paths (Mem0-style promotion, old retrieval) see
-- exactly what they did before; only the new policy-aware code reads the
-- new columns.
-- =============================================================================

BEGIN;

-- -----------------------------------------------------------------------------
-- STEP 1 — Widen memory_nodes (additive, defaulted)
-- -----------------------------------------------------------------------------

ALTER TABLE memory_nodes
    ADD COLUMN IF NOT EXISTS candidate_type   TEXT,
    ADD COLUMN IF NOT EXISTS urgency          TEXT,
    ADD COLUMN IF NOT EXISTS volatility       TEXT,
    ADD COLUMN IF NOT EXISTS sensitivity      TEXT,
    ADD COLUMN IF NOT EXISTS source_authority TEXT,
    ADD COLUMN IF NOT EXISTS policy_ids       TEXT[] NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS retention        JSONB,
    ADD COLUMN IF NOT EXISTS retrieval_policy JSONB,
    ADD COLUMN IF NOT EXISTS privacy_policy   JSONB,
    ADD COLUMN IF NOT EXISTS update_policy    JSONB,
    ADD COLUMN IF NOT EXISTS source_ref       TEXT,
    ADD COLUMN IF NOT EXISTS source_span      TEXT,
    ADD COLUMN IF NOT EXISTS speaker          TEXT,
    ADD COLUMN IF NOT EXISTS expires_at       TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS superseded_by    UUID
        REFERENCES memory_nodes(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS status           TEXT NOT NULL DEFAULT 'active';

COMMENT ON COLUMN memory_nodes.candidate_type IS
    'Behavioural type assigned by the policy engine (fact, user_preference, '
    'task, decision, meeting_episode, procedure, code_symbol, sensitive_data, …).';
COMMENT ON COLUMN memory_nodes.sensitivity IS
    'public / private / confidential / restricted — drives privacy filtering '
    'in retrieval and approval requirements in storage.';
COMMENT ON COLUMN memory_nodes.expires_at IS
    'Retention-policy expiry. Items past this time are filtered from "current" '
    'retrieval (the row stays; bi-temporal history is preserved).';
COMMENT ON COLUMN memory_nodes.status IS
    'active / superseded / expired / pending_approval / redacted.';

-- -----------------------------------------------------------------------------
-- STEP 2 — Decision-trace table (one row per policy evaluation)
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS memory_decision_traces (
    id                UUID         PRIMARY KEY,
    candidate_id      UUID         NOT NULL,
    candidate_text    TEXT         NOT NULL,
    selected_action   TEXT         NOT NULL,
    selected_scope    TEXT         NOT NULL,
    applied_policies  TEXT[]       NOT NULL DEFAULT '{}',
    rejected_policies TEXT[]       NOT NULL DEFAULT '{}',
    reasons           JSONB        NOT NULL DEFAULT '[]'::jsonb,
    final_plan        JSONB        NOT NULL,
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT now()
);

COMMENT ON TABLE memory_decision_traces IS
    'Audit trail for the Policy Engine — one row per candidate evaluated, '
    'capturing which policies matched, which were overridden, and the final '
    'MemoryHandlingPlan applied. Never modified; query-only forensics.';

-- -----------------------------------------------------------------------------
-- STEP 3 — Pending-approval queue (items the engine wants human sign-off on)
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS memory_pending_approvals (
    id              UUID         PRIMARY KEY,
    candidate_id    UUID         NOT NULL,
    candidate_text  TEXT         NOT NULL,
    proposed_plan   JSONB        NOT NULL,
    reason          TEXT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    decided_at      TIMESTAMPTZ,
    decided_by      TEXT,
    status          TEXT         NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'approved', 'rejected', 'expired'))
);

COMMENT ON TABLE memory_pending_approvals IS
    'Items the policy engine deferred (e.g. detected secret, restricted '
    'sensitivity) pending explicit user/operator approval. status moves '
    'from "pending" → "approved" / "rejected" / "expired".';

-- -----------------------------------------------------------------------------
-- STEP 4 — Indexes for the hot retrieval / governance queries
-- -----------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_memory_nodes_candidate_type
    ON memory_nodes (candidate_type);
CREATE INDEX IF NOT EXISTS idx_memory_nodes_sensitivity
    ON memory_nodes (sensitivity);
CREATE INDEX IF NOT EXISTS idx_memory_nodes_urgency
    ON memory_nodes (urgency);
CREATE INDEX IF NOT EXISTS idx_memory_nodes_status_live
    ON memory_nodes (status)
    WHERE invalidated_at IS NULL;
-- Partial index — most queries want only "current, unexpired" rows.
CREATE INDEX IF NOT EXISTS idx_memory_nodes_expires_at_live
    ON memory_nodes (expires_at)
    WHERE invalidated_at IS NULL AND expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memory_nodes_policy_ids
    ON memory_nodes USING GIN (policy_ids);
CREATE INDEX IF NOT EXISTS idx_memory_decision_traces_candidate_id
    ON memory_decision_traces (candidate_id);
CREATE INDEX IF NOT EXISTS idx_memory_pending_approvals_status
    ON memory_pending_approvals (status, created_at DESC);

-- -----------------------------------------------------------------------------
-- STEP 5 — Bookkeeping
-- -----------------------------------------------------------------------------

INSERT INTO schema_migrations (version, description)
VALUES (
    '004',
    'Policy Engine: memory_nodes policy columns + memory_decision_traces + '
    'memory_pending_approvals + indexes'
)
ON CONFLICT (version) DO NOTHING;

COMMIT;


-- =============================================================================
-- ROLLBACK PLAN (run separately, NOT in this transaction)
-- =============================================================================
-- The columns and tables added here only carry policy-engine metadata —
-- dropping them does not lose canonical memory content. Existing legacy
-- code paths never read these columns.
--
--   BEGIN;
--   DROP INDEX IF EXISTS idx_memory_pending_approvals_status;
--   DROP INDEX IF EXISTS idx_memory_decision_traces_candidate_id;
--   DROP INDEX IF EXISTS idx_memory_nodes_policy_ids;
--   DROP INDEX IF EXISTS idx_memory_nodes_expires_at_live;
--   DROP INDEX IF EXISTS idx_memory_nodes_status_live;
--   DROP INDEX IF EXISTS idx_memory_nodes_urgency;
--   DROP INDEX IF EXISTS idx_memory_nodes_sensitivity;
--   DROP INDEX IF EXISTS idx_memory_nodes_candidate_type;
--   DROP TABLE IF EXISTS memory_pending_approvals;
--   DROP TABLE IF EXISTS memory_decision_traces;
--   ALTER TABLE memory_nodes
--       DROP COLUMN IF EXISTS candidate_type,
--       DROP COLUMN IF EXISTS urgency,
--       DROP COLUMN IF EXISTS volatility,
--       DROP COLUMN IF EXISTS sensitivity,
--       DROP COLUMN IF EXISTS source_authority,
--       DROP COLUMN IF EXISTS policy_ids,
--       DROP COLUMN IF EXISTS retention,
--       DROP COLUMN IF EXISTS retrieval_policy,
--       DROP COLUMN IF EXISTS privacy_policy,
--       DROP COLUMN IF EXISTS update_policy,
--       DROP COLUMN IF EXISTS source_ref,
--       DROP COLUMN IF EXISTS source_span,
--       DROP COLUMN IF EXISTS speaker,
--       DROP COLUMN IF EXISTS expires_at,
--       DROP COLUMN IF EXISTS superseded_by,
--       DROP COLUMN IF EXISTS status;
--   DELETE FROM schema_migrations WHERE version = '004';
--   COMMIT;
-- =============================================================================
