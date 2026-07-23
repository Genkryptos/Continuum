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

### 0.1 Soak test — ✅ DONE (6h clean, restart recovered)
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

**Result:** 5.96h, 1,435 calls, 358 minute-samples, every check PASS.
RSS 68.9 → 53.5 MB (it *fell* — no leak, GC settling after warm-up), fds 75 →
74, pool 4 → 3, p95 21 → 24 ms (+14%, inside the 50% bar). Postgres was
restarted at the 3-hour mark; the single error of the whole run was that one
call, and the pool re-established itself with no operator action. The daemon
dimension — the one place a slow failure could hide — is now measured, not
assumed.

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

### 1.2 Supersession reordering only works for tagged facts — ✅ DONE
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

**Shipped:** the second option, kept narrow. A memory that names what it
replaced — `switched|moved|migrated|changed|upgraded from X to Y` — supersedes
the memories still asserting X. Both sides must be **capitalised**, the same
precision lever the attribute extractor uses, which is what keeps "I switched
from the kitchen to the office" and "I moved the file from src to lib" out of
it entirely.

The announcement leads by construction rather than by clock: it may well have
been written before the fact it displaces was last touched, so a timestamp
comparison would get this backwards.

Verified on a real store: `"what editor do I use?"` now returns *"I switched
from Neovim to Zed"* first and *"I use Neovim with a tmux setup"* second —
previously the reverse. Unrelated neighbours do not move, a replacement with
nothing to displace changes nothing, and two replacements do not reorder each
other.

---

## Phase 2 — Quality ceilings (characteristics, not bugs)

### 2.1 Recall falls with store size — ✅ DONE (ef_search raised to 1000; curve measured)
**Measured:** ~100% at tens of memories, 95% at 3k, **75% at 47k** (k=8, real
hybrid path). Every tuning lever is exhausted: `ef_search` fixed a real loss,
candidate pool 24→250 changed nothing, k 8→20 changed nothing, the cross-encoder
reranker made it worse. The residual is the embedder's semantic limit — "who
does my taxes" → *"My accountant is called Filipa Rego"* needs world knowledge
bge-m3 does not supply against 47k competitors.
**This entry was wrong, and the correction matters more than the entry.**

Two measurements settle it, both on the same 45,020-memory realistic store:

| | recall@8 | p50 |
|---|---:|---:|
| with the HNSW index (what ships) | 13/20 = 65% | 132 ms |
| index dropped — exact scan | **18/20 = 90%** | 157 ms |

**The index is discarding a quarter of retrievable memories to save 25 ms.**
The embedder was never the bottleneck: exact cosine over the same stored
vectors finds 19/20, and `bge-m3` beat both alternatives in a bake-off
(mxbai-embed-large 19/20, multilingual-e5-large 16/20 at 20k). Nothing is wrong
with the model or the data — the approximate index is dropping them.

`hnsw.ef_search` cannot fix it: pgvector **caps it at 1000**, 400 and 1000 give
the same 13/20 through the product path, and `iterative_scan` does not help.

**Recommendation:** an exact scan is the better default for a store of this
size. 157 ms against a ~130 ms baseline is a rounding error next to losing 25
points of recall, and the earlier "latency at scale" work measured the index's
*speed*, never what it cost in answers. The open question is where the crossover
actually is — the index must win eventually, and finding that row count is the
next piece of work.

**Fix (ranked):** (a) ~~evaluate a stronger embedding model~~ — done, no
candidate beats the current one; (b) query expansion — `hyde_fn`
already exists in the Retriever and is unused, at the cost of an LLM call per
query, so it must be opt-in and measured against the reranker's cautionary tale.
**Acceptance:** a published curve of recall vs store size, and a decision on
each lever with numbers.
**Effort:** M.

**Measurement caveat, stated plainly:** getting to that answer took several
wrong turns of my own — an ad-hoc SQL sweep that mis-scored the exact baseline,
a monkey-patch that silently measured the same setting four times (a signature
default binds at definition, not at call), and an earlier corpus whose
composition, not its size, produced the original "75% at 47k". The numbers
above come from the audited harness and from plain numpy over the stored
vectors, which agree. `CONTINUUM_HNSW_EF_SEARCH` now exists precisely so this
can be swept without patching anything.

**Crossover measured, and it revised the conclusion above.** A clean per-size
sweep (`scripts/index_crossover.py`, one process, fresh store per size) through
the real product path:

| rows | exact recall | exact p50 | index @ef=1000 | ef=400 (old default) |
|---|---:|---:|---:|---:|
| 3,000 | 100% | 3ms | 100% | 100% |
| 25,000 | 100% | 29ms | 100% | 100% |
| 45,000 | 90% | 59ms | 85% | 75% |

The earlier "65% vs 90%" was a confound — a poor HNSW build on a store rebuilt
many times, not the index's true behaviour. Reconciled, the index at
`ef_search=1000` nearly matches an exact scan (85 vs 90 at 45k, identical below
25k) while staying ~2ms at any size; an exact scan grows O(rows). **Decision:
raise the default `ef_search` 400 → 1000** — two-thirds of the gap recovered for
~0ms. The remaining ~5 points at 45k are available by dropping the HNSW index
for an exact scan, documented as an ops choice because a sequential scan does
not stay affordable as the store grows. No "no index below N rows" switch is
needed: the planner already seq-scans tiny tables, and above that the tuned
index is the right tool.

### 2.2 Capture is English-only — ✅ DONE for Latin-script (pt/es/fr/de/it); CJK/Devanagari out of scope
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

**Prerequisite built:** `tests/unit/test_capture_multilingual.py` — 15 durable
statements and 15 pieces of noise across pt/es/de/fr/it/hi/ja/zh, weighted
toward what must be refused, plus credential cases.

Deliberately built *before* any rules. Capture's entire defence is a measured
0-false-capture rate; writing patterns for a language with no way to measure
that would trade the one property making the feature safe for a hit rate nobody
checked.

**Then the rules, held to the same bar.** pt/es/fr/de/it now capture
first-person stative statements (22 durable cases) with **0 false captures
across 33 noise cases** — the categories English is held to: past-tense actions,
"just did", questions, imperatives, transient states. The design is the same as
English (accept only present-tense stative verbs) and leans on the auxiliaries
being different words: French `j'ai`, German `habe`, Italian `ho`, Spanish `he`
front both "I have a meeting" and the perfect tense, so none are on the accept
list and both stay refused for free. Portuguese/Spanish/Italian drop the subject
pronoun, so the verb anchors, and the word-count floor dropped to 2 to reach
"Sou vegetariano".

Guarded against overfitting with 8 unseen adversarial noise cases (futures,
modals, `estar`/`stare` progressives, "I'm done"/"I'm tired" transients) and
unseen durable cases — 0 false captures, 0 misses. Verified end to end: a mixed
Portuguese turn stores the fact and drops the imperative.

**hi/ja/zh stay unsupported, by decision not omission.** Word-order regexes do
not survive scripts without spaces to anchor on, and adversarial cases in them
cannot be author-validated to the 0-false-capture bar. `KNOWN_UNSUPPORTED` names
them and a test asserts they stay silent; loosening the rules to capture them is
explicitly the wrong move. Credential detection works in every language already,
because it is pattern-based; that is asserted too.

### 2.3 One ambiguous capture verb — ✅ DONE (narrowly, after the open version proved unsafe)
"I review every PR myself" is refused because *review* is as often an action
("I review the diff") as a habit. Low value, listed for completeness.
**Effort:** S, and possibly correct to leave alone.

**Resolved by a closed allowlist, not the obvious open rule.** The first cut —
"any non-episodic verb governing every/each/all <noun>" — leaked: "I broke every
build today" and "I built every feature" are irregular pasts that neither the
-ed guard nor the episodic list catches, and one false capture is worse than the
missed facts an allowlist costs. So the accept is a closed set of role verbs
(review/handle/manage/maintain/oversee/own/lead/coordinate/administer/curate/
moderate) in present tense over a universal quantifier — a standing
responsibility. Bare "want"/"wanted" also became a desire marker, so "I want
every feature" is refused. Measured 5/5 durable, **0/13 noise** including the
irregular-past traps. This is a case where "possibly leave alone" was nearly
right: the value is small and the open version would have broken the feature's
one guarantee.

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
