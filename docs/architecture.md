# Architecture

Continuum has three memory tiers, a promotion lifecycle that moves
content between them, and two schema-level features (supersession +
bi-temporal) that nobody else has built in. Once these four ideas
are clear, the rest of the framework follows.

## 1 · The three tiers

| Tier | What lives there | Lifetime | Default store |
|---|---|---|---|
| **STM** (short-term memory) | Raw conversational turns — exactly the messages the user typed and the agent's replies. | Hours to days, capped by token budget. | `InMemorySTM` (volatile) or `PostgresSTM` (durable). |
| **MTM** (mid-term memory) | LLM-generated *summaries* of session chunks. Compressed, episode-grained. Tokens-per-MB is much higher than STM. | Weeks to months. | `PostgresMTM` (pgvector for retrieval). |
| **LTM** (long-term memory) | Extracted *atomic facts* and *entities* with confidence, supersession, and bi-temporal columns. | Indefinite. | `PostgresLTM` with the migration-004 schema (`superseded_by`, `valid_from`, `recorded_at`). |

```
   ┌──────────┐   summarisation    ┌──────────┐    extraction     ┌──────────┐
   │   STM    │ ─────────────────► │   MTM    │ ─────────────────►│   LTM    │
   │  (raw    │  (Mem0Promoter     │ (summary │  (FactExtractor   │  (facts, │
   │  turns)  │   when STM full)   │  blocks) │   over MTM blocks)│  entities│
   └──────────┘                    └──────────┘                   │  with    │
                                                                  │  superseded_by) │
                                                                  └──────────┘
```

Why three tiers and not one? Because retrieval cost grows linearly
with the size of the store you're searching, and the *value density*
differs by tier:

* STM is high-detail / low-density — a single turn might be 30 tokens
  of "yeah, sounds good" that retrieval should almost never return.
* MTM is medium-detail / medium-density — a session summary is 200
  tokens compressing several turns' worth of decisions.
* LTM is low-detail / high-density — a single atomic fact is 15 tokens
  and is the answer to many queries.

Routing retrieval through all three lets the retriever favour LTM
(cheap, dense) when a fact-card answers the question, fall back to
MTM (mid-detail) when the user wants context, and reach into STM
(raw) only for in-session continuity.

## 2 · The promotion lifecycle

`session.process_turn(user_msg)` does five things in order:

```
   user_msg ──► STM.append("user", msg)
                       │
                       ▼
              Retriever.retrieve(msg, budget)
                       │
                       ▼
              Optimizer chain (compress to budget)
                       │
                       ▼
              Responder(user_msg, context) ──► reply
                       │
                       ▼
              STM.append("assistant", reply)
                       │
                       ▼
              [background] trigger_manager.after_turn()
              [background] incremental_index(ctx)
              [background] access_log
```

The bottom three lines run on a `BackgroundQueue` so the response
path stays under the foreground latency budget. The
`trigger_manager` is what fires promotion:

```
   trigger condition fires
        │   (e.g. "STM has 4K tokens, promote oldest chunk")
        ▼
   Mem0Promoter walks the chunk
        │   1. summarize chunk into a SummaryBlock → write to MTM
        │   2. extract entities + facts from summary → write to LTM
        │   3. for each new fact: detect contradictions against existing
        │      LTM rows on the same (entity, attribute); if a match exists,
        │      set OLD_FACT.superseded_by = NEW_FACT.id
        ▼
   chunk removed from STM
```

This is the architectural diagram for the supersession-correctness
benchmark you saw at 100 %: the supersession edge is set *at promotion
time*, off the foreground path, by the same component that's doing
the LLM-driven fact extraction. Retrieval doesn't have to do any
work to honour it — the `WHERE superseded_by IS NULL` filter on the
LTM query is a one-line addition that the schema makes possible.

## 3 · Supersession in detail

Every row in the LTM table has a `superseded_by UUID NULL` column.
A NULL means "this fact is currently valid"; a non-NULL points at the
fact that replaced it.

```
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ id   │ subject   │ attribute      │ value      │ superseded_by │ ...   │
   ├──────┼───────────┼────────────────┼────────────┼───────────────┼───────┤
   │ a1.. │ user      │ location       │ NYC        │ d9..          │ ...   │
   │ b5.. │ user      │ pets.dog       │ Rex        │ NULL          │ ...   │
   │ d9.. │ user      │ location       │ Boston     │ NULL          │ ...   │
   └─────────────────────────────────────────────────────────────────────────┘
```

Retrieval queries that want "the current value" filter
`superseded_by IS NULL`. Queries that want the full history
(`/show ltm --all` in the demo) skip that filter. The same table
serves both reads.

### Why this beats append-only memory

An append-only memory has both NYC and Boston facts with no link.
At query time:

* a vector store ranks by cosine similarity — both facts may rank
  highly, the model sees both, has to disambiguate. With careful
  prompting this works ~38 % of the time
  ([`bench/supersession_correctness.py`](../bench/supersession_correctness.py)).
* an LLM-as-router asked "which fact is current?" gives the right
  answer most of the time but spends an extra inference per query.
* **Continuum** filters at the SQL layer, costs one extra `AND`
  clause, returns only Boston. 100 % correct by construction.

### Why this isn't just "delete the old fact"

Two reasons:

1. **Audit / history queries.** "When did the user move from NYC to
   Boston?" needs the NYC row to still exist.
2. **Retroactive corrections.** If the user later says *"actually I
   was wrong — I never moved to Boston, I moved to Cambridge"*, you
   need to roll back the Boston-supersedes-NYC edge and replace it
   with Cambridge-supersedes-NYC. That's only possible if NYC is
   still there.

Soft-delete via `superseded_by` is strictly more capable than
hard-delete + reinsert.

## 4 · Bi-temporal in detail

Two timestamp columns on every fact:

| Column | Meaning |
|---|---|
| `valid_from` | The point in *real-world* time when this fact became true. |
| `recorded_at` | The point in *system* time when Continuum learned about this fact. |

These two are almost always equal — facts are usually recorded right
after they become true. But they can diverge in two important ways:

### Point-in-time queries

`"Where did the user live in March 2024?"` — we want the fact whose
`valid_from` ≤ `2024-03-01` AND that wasn't superseded by an earlier
fact.

```sql
SELECT * FROM ltm
WHERE attribute = 'user.location'
  AND valid_from <= '2024-03-01'
ORDER BY valid_from DESC LIMIT 1;
```

A naive store without `valid_from` can only answer "what's the
current location" — wrong for any historical question.

### Retroactive corrections

The user says "Actually I moved to Boston in *June* 2023, not
August like I told you before."

* `valid_from` = `2023-06-15` (when the move actually happened)
* `recorded_at` = today (when we learned the correction)

A naive "most recently recorded fact" filter would miss this — it
would say the August fact is the most recently recorded for the June
date. Continuum's bi-temporal lookup correctly returns the corrected
fact.

The benchmark at
[`bench/bi_temporal.py`](../bench/bi_temporal.py) measures
this directly: 100 % vs the best naive baseline's 75 %.

## 5 · Putting it together

A request that hits a healthy Continuum deployment:

```
   ┌──────────────────────────────────────────────────────────────────┐
   │ session.process_turn("Where do I live now?")                      │
   └──────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                ┌───────────────────────────────────┐
                │  Retriever — pulls from all tiers │
                │  ┌────────┐  ┌────────┐  ┌──────┐ │
                │  │  STM   │  │  MTM   │  │ LTM  │ │
                │  │ recent │  │ summ-  │  │ current
                │  │ turns  │  │ aries  │  │ facts│ │
                │  └────────┘  └────────┘  └──────┘ │
                │       │          │          │     │
                │       ▼          ▼          ▼     │
                │   ┌──────────────────────────┐    │
                │   │ Composite scorer         │    │
                │   │ relevance / importance / │    │
                │   │ recency / confidence     │    │
                │   └──────────┬───────────────┘    │
                │              ▼                    │
                │   ┌──────────────────────────┐    │
                │   │ Optimizer chain — pack   │    │
                │   │ into context budget      │    │
                │   └──────────────────────────┘    │
                └───────────────────┬───────────────┘
                                    ▼
                          ContextBundle
                                    │
                                    ▼
                            Responder(LLM)
                                    │
                                    ▼
                                 reply
```

Every box is a swappable component behind a Protocol; the default
implementations are the ones the benchmarks use. Swap them out via
the constructor (`ContinuumSession(stm=..., retriever=..., responder=...)`)
or via the YAML config file (see [Config](config.md)).

## 6 · What this architecture is *not*

* **Not a reasoner.** No agentic loop, no multi-hop retrieval, no
  chain-of-thought orchestration. Bring your own
  (LangGraph / AutoGen / hand-rolled) and plug Continuum in as the
  memory layer it calls.
* **Not a single source of truth for everything.** Facts live in LTM
  because we extracted them from conversation. Source documents, code
  artifacts, structured records — those belong in your application
  database; Continuum should *reference* them, not own them.
* **Not "free" supersession.** The benchmark measures the schema's
  ability to surface the current fact *given correct detection*. The
  contradiction-detection step itself runs through an LLM at promotion
  time and has its own quality axis. See
  [`bench/supersession_correctness.py`](../bench/supersession_correctness.py)
  docstring for the honest split.

## 7 · Where to look in the code

| Concept | Module |
|---|---|
| STM tier | [`continuum/stores/stm/`](../continuum/stores/stm/) |
| MTM tier | [`continuum/stores/postgres/mtm.py`](../continuum/stores/postgres/mtm.py) |
| LTM tier (supersession + bi-temporal columns) | [`continuum/stores/postgres/ltm.py`](../continuum/stores/postgres/ltm.py) + [`migrations/004_policies.sql`](../migrations/) |
| Session orchestration | [`continuum/core/session.py`](../continuum/core/session.py) |
| Retriever + scorer | [`continuum/retrieval/`](../continuum/retrieval/) |
| Promotion | [`continuum/promotion/`](../continuum/promotion/) |
| Fact / entity extraction | [`continuum/extraction/`](../continuum/extraction/) |
| Optimizer chain | [`continuum/optimizer/`](../continuum/optimizer/) |

Every public class has docstrings; the Protocol-based interfaces are
under [`continuum/core/protocols.py`](../continuum/core/protocols.py).
