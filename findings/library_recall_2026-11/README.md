# Library recall on LongMemEval-S haystacks (2026-11-08)

The README's LongMemEval accuracy comes from the eval harness in
`evals/longmemeval/`, which has its own retrieval. These runs measure the code a
`pip install continuum-mcp` user runs instead: `Memory.recall()`, which the MCP
`recall` tool calls.

`scripts/lme_library_recall.py` writes every haystack turn with `Memory.add`
(dated with its session's date) into a Postgres + bge-m3 store, asks the question
with `Memory.recall(k=48)`, and counts how many of the turns LongMemEval flags
`has_answer` come back in the top *k*. No LLM and no judge, so it is
deterministic. It is a retrieval number, not an accuracy number.

Sample: 10 questions per type (60, seed 0, abstention rows skipped). All three
runs read the same stored rows and vectors; only the retrieval code differs.

## Result

Recall of gold turns (in brackets: share of questions with *every* gold turn
retrieved):

| | @8 | @24 | @48 |
|---|---|---|---|
| `main` (d1abb3b) | 0.80 (0.72) | 0.85 (0.78) | 0.85 (0.78) |
| `recall(k)` honours `k` | 0.80 (0.72) | **0.98 (0.97)** | **0.99 (0.98)** |

By type at @48, before → after: temporal-reasoning 0.62 → 1.00,
single-session-preference 0.65 → 1.00, knowledge-update 0.95 → 1.00,
multi-session 0.91 → 0.95; single-session-user and -assistant were already 1.00.
No question got worse at any cut-off.

**Why:** the retriever kept at most `ltm_top_k` (10) long-term hits whatever the
caller asked for, and filled the remaining slots with the most recent STM turns.
On `main`, `recall(k=48)` returned 14–16 items, and only 10 of them were ranked by
relevance. With the fix it returns 48.

**Unchanged:** @8, because `k=8` (the MCP tool's default) was always under the
cap. Relevance-only scorer weights (`fixed_relevance_only.json`, the setting the
eval harness uses) give identical numbers: importance and confidence are the
same for every turn here, so the default weights already order by relevance.
`make mcp-eval` scores the same before and after (recall@1 100%, paraphrase
recall@1 90%, supersession 2/2, timeline 1/1).

## Files

| file | code |
|---|---|
| `main_d1abb3b.json` | clean `main` |
| `fixed.json` | retriever honours `query.top_k` |
| `fixed_relevance_only.json` | the fix + `CONTINUUM_SCORE_W_REL=1`, other weights 0 |

Haystack vectors were computed with the library's model and settings on the
GPU (CPU for chunks that did not fit); query vectors go through the library's
own `EmbeddingService`.
