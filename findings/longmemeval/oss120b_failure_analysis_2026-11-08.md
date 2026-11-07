# LongMemEval-S: why the last 64 rows still fail (gpt-oss-120b, 2026-11-08)

Source run: `results/arm_audit_fixes_oss120b_20261107` plus the 17-row rerun
at `--answer-max-tokens 8192`. The run used the audit-fix code on
`experiment/lexical-channel-gating` (the turn-selection fix, 35878a9); that
code is **not on main yet**. Opus manual grade: 436/500 = **87.2%**; the llama
judge gives 87.8%. Hindsight reports 89.0% at gpt-oss-120b. The run is
non_comparable: it used top-k 48, LTM off, and mixed answer caps.

Per-row buckets: [`oss120b_failure_buckets_2026-11-08.json`](oss120b_failure_buckets_2026-11-08.json).

## The headline: the residual is now a reader problem

| | rows | failing |
|---|---:|---:|
| every gold turn retrieved | 456 | **46** (89.9% correct) |
| gold turns partially or not retrieved | 23 | 16 |
| no gold-turn annotation | 21 | 2 |

The turn-selection fix closed the retrieval gap. 46 of the 64 remaining
failures had every gold turn in the prompt. The 2026-11-07 diagnosis
("turn-selection-bound, not reader-bound") was right about the run *before*
the fix. It no longer describes this one.

## Buckets (each failing row has exactly one primary bucket)

| bucket | rows | evidence | what goes wrong |
|---|---:|---|---|
| **A. False "not mentioned"** | 15 | complete | answers "the conversation does not mention X… it only mentions Y" |
| G. Retrieval incomplete | 13 | partial/none | 3 are the user's own age; 3 are orderings missing one event |
| B. Count wrong | 9 | complete | duplicate mentions across sessions (over-count), missed items (under-count) |
| F. Preference | 8 | mostly complete | generic answer that ignores the stated preference |
| D. Temporal | 7 | complete | 3 pick the wrong event for "a week ago" or "past weekend"; 3 interval arithmetic; 1 ordering |
| C. KU stale | 6 | complete, dated | takes the earlier value although both turns are dated |
| E. Empty answer | 4 | complete | hidden reasoning used all 8192 answer tokens |
| H. `_abs` answered | 2 | — | answered a question with a false premise |

**Cross-cutting: 7 rows hinge on a relative date anchor** ("last Saturday",
"last Friday", "two weeks ago", "past weekend", "Valentine's Day", "last
week"). They are spread over A, D and one multi-session row.

### A in detail: the presupposition clause over-fires

All 18 refusal-style answers on non-`_abs` rows are wrong (15 have complete
evidence; 3 more are the missing-age rows in G). They copy the wording of
`_PRESUPPOSITION_CLAUSE`: *"say the conversation does not mention it, and
name what it does mention instead."* That clause goes into every prompt
branch. In each of these rows the answer needs one step the reader would not
take:

- **bridge two turns:** "submitted to ACL" + "ACL's submission date was
  February 1st" (51c32626); "20 of the leadership positions" + "100
  leadership positions" → 20% (099778bb)
- **one sum:** "left home at 7 AM" + "took two hours to get to the clinic" →
  9 AM (73d42213); $30 list price + impulse-buy price → 20% (d905b33f)
- **alias:** "my flea market find" is the sunset painting (b86304ba);
  "Valentine's Day" is February 14th (58ef2f1c, gpt4_f420262d)
- **"today" in a turn means that turn's date:** "started enjoying their music
  today", said on a Friday, answers "last Friday" (gpt4_fa19884d, 71017277)

On the other side of the trade, 27 of the 28 correct `_abs` rows are
refusal-style. So the clause can't simply be removed: it has to stop firing
when the evidence supports one inference step.

### Why the date stamps aren't enough

On this branch, KU, aggregate and temporal turns are already stamped
`[YYYY-MM-DD]` and the prompt gets today's date. But the stamp has **no
weekday**, and nothing turns "last Saturday" or "two weeks ago" into a date
range. The model has to compute weekdays and offsets in its head, which is
exactly what fails. The dataset already has the weekday (`2023/03/04 (Sat)`);
`_parse_lme_date` drops it.

## Levers, ranked by expected rows per unit of cost

Expected gains are guesses from reading the rows, not measurements. A
prompt change also flips rows the other way: the last arm gained 65 and lost
19. So every lever gets a targeted subset run first and a full-500
regression run after.

1. **Verify refusals (A, maybe 3 of G):** when the first answer is a
   refusal, make one extra reasoning call. It allows bridging two turns, one
   sum, alias resolution and "today = this turn's date", and it needs a quoted
   reason before it may keep "not mentioned". It fires on about 46/500 rows,
   so it's cheap. Expected **+6 to +10 rows**. Guard: keep 27 or more of the
   28 `_abs` rows. Test set: the 46 refusal rows, about $0.05.
2. **Weekday and offset in the stamp, relative anchors resolved in code (the 7
   anchor rows):** use `[2023-03-04 Sat, 5 days before today]` and parse
   "last <weekday>", "N weeks ago" and "past weekend" against `question_date`.
   This is deterministic and adds no LLM call. Expected **+3 to +5**.
3. **Fallback for empty answers (E):** retry an empty answer at low
   reasoning effort. Also pin 8192 as the cap for reasoning answerers, so the
   run has one config instead of a 2048/8192 mix. Expected **+2 to +3**, and
   the run stops being a mixed-cap deviation.
4. **Run with `--use-ltm` (C):** this is the pinned config, and the only way
   the supersession layer (`[CURRENT FACT]`) takes part. It is also the
   paper's thesis, still unmeasured on LongMemEval. Expected **+2 to +4** on KU.
   The cost is about 17 hours of ingest.
5. **Count in code (B):** have the reader emit a JSON list of distinct items
   with source dates, then dedupe and count in code (the same pattern as
   `ORDER:`). `--aggregation-v2` came out null on an older non-reasoning
   reader at 60% overall. It has not been tested with complete evidence and a
   reasoning reader. Expected **+2 to +4**; low confidence.
6. **A user-profile retrieval channel (3 rows in G):** questions about the
   user's own age need the turn that states it, and that turn shares no words
   with "when Alex was born". Expected +2 to +3; more work.
7. **Preference (F):** 8 rows, and the judge is noisy here (Opus was
   stricter on 2). Last in line.

Levers 1 to 3 together target about 26 rows. Recovering half of them takes
the Opus grade from 87.2% to about 89 to 90%. That is a projection; nothing
here has been run yet. The comparability gaps (top-k 48, LTM off, judge model)
still have to close before any claim against Hindsight.
