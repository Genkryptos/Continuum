# Continuum — Hardening Plan

**Status:** open · **Owner:** Mayank · **Context:** `docs/MCP_PRODUCTION_PLAN.md`
is complete and its Definition of Done is met. This plan covers what six rounds
of user testing left behind — the things that stop the answer to "is it
production ready?" being an unqualified yes.

Ordered by what gates *confidence*, not by what is easy. Every item below is
grounded in a measurement from this cycle, not a hunch.

---

## The two meta-problems

Individual bugs are not the main risk. Two structural facts are:

**A. The defect curve never converged.** Six rounds of user testing produced
3, 3, 2, 2, 2, 2 real defects. Flat, not trending to zero — so the honest
prior is that a seventh round finds roughly two more. Twice the bugs were in
code written the same day, so risk concentrates in recent work.

**B. The measurement layer is less reliable than the code.** At least five
times this cycle the *instrument* was wrong, not the product: a bare string
passed to `embed()` (which then embedded per character), a stale DSN file, a
`'%tin%'` substring that matched "meeting", a corpus generator that span forever
on an impossible target, and a DSN regex that swallowed a trailing backslash.
Each was caught — but every one produced a confident, wrong intermediate
conclusion first, and there is no reason to believe all of them were caught.
Green results are only as trustworthy as the harness that produced them.

Phase 0 addresses both. Nothing later matters as much.

---

## Phase 0 — Make the evidence trustworthy

### 0.1 Soak test — the only entirely unmeasured dimension
**Problem:** nothing in this project has run for more than ~8 minutes. Every
test is a burst. Connection-pool churn, RSS growth, file-descriptor leaks and
prepared-statement accumulation over hours are completely unknown — and
`continuum-mcp --http` is a **daemon**. This is where a failure bites quietly at
3am instead of loudly in CI.
**Fix:** drive the HTTP server for 4–8 hours at a realistic rate (a few
requests/minute, mixed `remember`/`recall`/`current`), sampling RSS, open file
descriptors, pool size and p95 latency every minute. Include at least one
Postgres restart mid-run to prove the pool recovers.
**Acceptance:** RSS flat after warm-up (no monotonic climb), fd count bounded,
p95 latency at hour 8 within 20% of hour 1, and full recovery from the restart
without operator action.
**Effort:** S to write, hours to run. **Do this first.**

### 0.2 Promote the throwaway harnesses to real tooling — ✅ DONE
**Problem:** the corpus generators, the recall scorers and the month-long daily
simulation live in a scratch directory as one-off scripts. They produced every
headline number in the docs, and they are exactly where the five instrument bugs
were. They are also unrepeatable by anyone else, which fails the project's own
"a stranger can reproduce it" bar.
**Fix:** move them into `scripts/` with the same standards as the code — typed,
linted, and with a handful of unit tests over the *harness itself* (does the
generator actually produce N distinct strings? does the scorer match a needle
exactly rather than by substring?). Add `make soak` and `make recall-at-scale`.
**Acceptance:** every number quoted in `docs/` is reproducible by one make
target on a clean machine.
**Effort:** M.

**Shipped:** `scripts/recall_at_scale.py` (`make recall-at-scale`) and
`scripts/soak_test.py` (`make soak`), both typed and linted, plus
`tests/unit/test_harnesses.py` — ten tests over the *instruments*: the generator
delivers exactly N distinct sentences, **raises instead of hanging** on an
impossible target, is deterministic, and never emits a sentence that could be
mistaken for a needle; the soak verdict fails on climbing RSS, on leaked
descriptors, and on too few samples to judge.

Reproducibility checked against the published figures: 3k formulaic gives
**19/20 = 95%**, missing exactly `"can I eat prawns"` — the documented number and
the documented miss. 3k realistic gives 20/20, which is why the shape is now an
explicit flag rather than an accident of which script you ran.

Writing those tests immediately caught a flaw in the needle set itself: half the
queries share content words with their answers, so the set was never a pure
semantic probe. That is *fine* — real questions are a mix — but it now has a
test asserting at least 8 of 20 carry no shared content word, so the headline
number cannot quietly drift to measuring something easier.

### 0.3 One more testing round, on the dimensions never touched
**Problem:** the flat defect curve. Rounds so far covered: personal facts,
messy/multilingual input, a simulated month, concurrency and failure recovery,
scale, and the fixes themselves. **Never covered:** the upgrade path on a large
populated store, backup/restore, clock skew and DST boundaries, a store that
hits disk limits, and a from-scratch install on a machine that has never seen
this project.
**Fix:** one round against those.
**Acceptance:** the round completes with findings recorded either way — a clean
round is itself the result that has never yet occurred.
**Effort:** M.

**Upgrade path — ✅ verified**, on a store built from migrations 001–005 with
12,020 real embedded memories (79 MB). `make db-migrate` applied `006` in
**6.0s**, lost nothing, needed no manual step.

**And it corrected a claim of mine.** The first measurement said 70% → 80%, one
index build each. Repeating it five times per configuration says something
different and more useful:

| index | runs | median | range |
|---|---|---:|---|
| old `m=16, ef_construction=64` | 16, 12, 14, 19, 15 | 15/20 | **12–19** |
| new `m=32, ef_construction=200` | 16, 17, 16, 16, 16 | 16/20 | **16–17** |

The median barely moves. What the denser graph actually buys is **consistency**:
with `m=16` a user could rebuild their index and silently lose seven needles'
worth of recall, or gain four, purely on the luck of the build. That is the real
argument for `006`, and it is a better one.

**Methodological consequence, which applies to every recall figure in this
repo:** HNSW assigns node levels randomly, so an index build is a *sample*, not
a measurement. Single-build comparisons can report a difference that is entirely
build noise — as mine did. `recall_at_scale.py --rebuilds N` now reports a
median and range, and a single-build run prints a warning saying so.

**Backup/restore — ✅ verified.** `pg_dump -Fc` of the 12k store (51 MB) and
`pg_restore` into a fresh database: 0 errors, all 12,020 rows and vectors
intact, migration history preserved, HNSW parameters and all four extensions
recreated, and retrieval works on the restored copy. One operational gotcha
worth documenting: the `pg_dump` on `PATH` was 14.18 against a 16.10 server and
**refused to run at all** — back up with client tools matching your server.

Remaining in this item: clock skew and DST, disk-full, and a from-scratch
install on a clean machine.

---

## Phase 1 — Known defects with real user impact

### 1.1 Concurrent first-insert duplicates — ✅ DONE (migration 007)
**Problem (measured):** 8 simultaneous identical writes leave 5 rows.
`touch_duplicate` cannot see a row that is not yet committed, so racing writers
all insert. Sequential writes — one user, one hook per prompt — already collapse
correctly, so severity is storage-only and the read path dedups.
**Why it is still open:** the clean fix is a partial unique index on
`(namespace, md5(text)) WHERE invalidated_at IS NULL`, and that index **cannot
be built on a store that already holds duplicates** — which is every store
written before dedup existed. The migration would hard-fail on upgrade.
**Fix:** a two-step migration — retire (never delete) existing duplicates,
keeping the most-reinforced row per `(namespace, text)`, then build the index;
plus catch the unique violation in `upsert` and fall back to `touch_duplicate`.
Must be safe to run on a live store, and must report what it retired.
**Acceptance:** the concurrency test yields exactly 1 row; migration verified on
a populated store that already contains duplicates.
**Effort:** M. **Risk:** touches user data — dry-run first, and print the plan.

**Shipped as migration 007.** Two steps, in one transaction:

1. Retire pre-existing duplicates — `invalidated_at`, the same bi-temporal
   close supersession and `forget()` use, so **nothing is deleted** and every
   copy stays recoverable. The survivor is the most-reinforced row (highest
   `access_count`, tie-broken by newest), because that is the copy carrying the
   restatement history.
2. `CREATE UNIQUE INDEX … (namespace, md5(text)) WHERE invalidated_at IS NULL`.
   Partial, so a superseded fact and its replacement may still share text.

`upsert` now catches that specific violation — matched on the index *name*, so
an unrelated unique violation still surfaces as a bug — and reinforces the
winner instead of failing the caller. Losing the race means someone stored this
exact fact a moment ago, which is the outcome we wanted.

Verified: on a store deliberately seeded with duplicates (5 live rows, 3
distinct across two namespaces), the migration left 3 live rows, kept the
`access_count=7` copy, preserved the other namespace's, and retired 2 without
deleting. **8 concurrent writers of a brand-new sentence now leave exactly 1 row
with `access_count=7`** — previously 5 rows. Sequential writes unchanged.

### 1.2 Supersession reordering only works for tagged facts
**Problem (measured):** `_prefer_current_versions` fixed residence and employer,
but "I switched from Neovim to Zed" still ranks below "I use Neovim with a tmux
setup" — that pair carries no `attribute` tag, so there is no group to reorder.
**Fix:** either extend `extract_attribute` to derive a tag for
"switched from X to Y" patterns, or group on a second signal (a shared
distinctive proper noun) — the latter is fuzzier and needs its own precision
measurement before it can be trusted.
**Acceptance:** the Neovim/Zed case orders correctly with no regression in the
0-false-tag attribute set.
**Effort:** M.

---

## Phase 2 — Quality ceilings (characteristics, not bugs)

### 2.1 Recall falls with store size
**Measured:** ~100% at tens of memories, 95% at 3k, **75% at 47k** (k=8, real
hybrid path). Every tuning lever is exhausted: `ef_search` fixed a real loss,
candidate pool 24→250 changed nothing, k 8→20 changed nothing, the cross-encoder
reranker made it worse. The residual is the embedder's semantic limit — "who
does my taxes" → *"My accountant is called Filipa Rego"* needs world knowledge
bge-m3 does not supply against 47k competitors.
**Fix (ranked):** (a) evaluate a stronger embedding model on the same needle
set — this is the single biggest lever left; (b) query expansion — `hyde_fn`
already exists in the Retriever and is unused, at the cost of an LLM call per
query, so it must be opt-in and measured against the reranker's cautionary tale.
**Acceptance:** a published curve of recall vs store size, and a decision on
each lever with numbers.
**Effort:** M. **Until then:** publish the 75%-at-47k figure in the README.

### 2.2 Capture is English-only
**Problem:** retrieval is multilingual (an English question retrieves a
Portuguese or Hindi memory — verified), but the capture rules are regexes over
English word order, so a non-English user's turns are silently skipped. It fails
*safe* — silence, never a wrong capture — but it is an asymmetry a user will
notice and not understand.
**Fix:** the prerequisite is measurement, not rules. Build a per-language
adversarial set (durable facts + noise) before writing a single pattern;
without it there is no way to know whether new rules preserve the 0-false-capture
property that makes this feature defensible.
**Acceptance:** measured precision per supported language, at the same bar.
**Effort:** L. **Until then:** documented in `docs/mcp.md`.

### 2.3 One ambiguous capture verb
"I review every PR myself" is refused because *review* is as often an action
("I review the diff") as a habit. Low value, listed for completeness.
**Effort:** S, and possibly correct to leave alone.

---

## Phase 3 — Release mechanics

### 3.1 Version and tag
`pyproject` still says 2.0.0, and the `v2.0.0` tag points at the wheel whose
documented install **crashed** (`rank-bm25` missing). Nothing was published, so
this is still cheap: bump to 2.1.0 and let that tag be the first real one.
**Effort:** S. **Blocks:** any PyPI upload.

### 3.2 CI has not run in 29 commits
`origin/main` is at `da9ac1a`. The matrix was reproduced locally in Docker
(Linux + Python 3.11/3.12, both green), which is a strong proxy but does not
exercise the coverage gate, the acceptance gate, or the build steps.
**Effort:** S — a push and a merge.

### 3.3 Lint debt outside CI's scope
~190 ruff findings in `scripts/` and `benchmarks/`; CI only checks
`continuum tests`. Zero user impact, but 0.2 moves harnesses into `scripts/`,
so it is worth clearing at the same time.
**Effort:** S.

---

## Suggested order

1. **0.1 soak** — the only unmeasured dimension, and the one that fails quietly.
2. **0.2 harness tooling** — everything else is measured *with* these; they had
   five bugs.
3. **0.3 one more round** — the defect curve is flat; find out if it is still flat.
4. **1.1 / 1.2** — the two real defects left.
5. **2.1** — the biggest remaining quality lever, once the harness is trustworthy.
6. **3.x** — mechanical, do last.

Phase 0 is what turns "everything I found is fixed" into "I know what I have
not found." That distinction is the whole difference between shipping with
confidence and shipping with hope.
