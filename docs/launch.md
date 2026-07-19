# Launch content (drafts)

Drafts for the owner to post. Every number here is sourced from `docs/report.md`
/ `docs/limitations.md` / `findings/roadmap_v3.md` — do not add uncited claims.

---

## Long-form launch post

### Memory for AI agents is a state problem, not a retrieval problem

Every "memory" layer I tried for agents was really just retrieval with extra
steps: a vector store that happily returns a fact the user retracted last week,
or a chat log that grows until it's unaffordable. The hard part was never
*finding* the memory — it was knowing **which version is true now**.

So I built **Continuum**: a memory layer with first-class **supersession** and
**bi-temporal** history. When a user says "I moved to Boston" and later "actually
I'm in NYC now", Continuum knows NYC is current *and* can still answer "where did
I live as of March?". Superseded facts are stamped `invalidated_at`, never
deleted; "current" reads filter them out; `valid_from`/`valid_to` make "as of
date Y" a real query.

It's five verbs (`add` / `recall` / `current` / `timeline` / `remember`) or an
MCP server you drop into Claude Code / Cursor with zero glue.

**The honest part.** On LongMemEval-S (500 Q, `gpt-oss-120b` reader,
`llama-3.3-70b` judge) Continuum lands ~**74%**, and on the scripted
supersession / as-of-date benches it's **100%** (vs 38% / 75% for a naive store).
But I also spent a week proving four "obvious" ideas *wrong* — having the memory
layer count in code, route deterministic answers, distill context, compute dates
in code. All net-negative: you can't bolt deterministic machinery onto a
mid-size model's unreliable intermediate outputs. The prompt/sampling levers I
kept are about accuracy-neutral (within run-to-run noise); the robust,
reproducible wins are the deterministic supersession/as-of benches.

And a methodology lesson I paid for: **measuring gains only on known failures
overstates them** (a failure can't regress). It took a same-setup control to see
the true, honest number.

Repo, benchmarks, and the full negative-results writeup are open. If your agent
forgets or trusts stale facts, this is the layer you're missing.

---

## Show HN

**Title:** Show HN: Continuum – a memory layer for AI agents with supersession
and bi-temporal recall

**Body:**

I kept hitting the same wall building agents: the "memory" was really just
retrieval, and it would confidently return facts the user had already retracted.
The hard problem isn't finding a memory — it's knowing which version is current.

Continuum is a memory layer built around **supersession** (superseded facts are
invalidated, not deleted, so "current" reads always know the live value) and
**bi-temporal** columns (answer "what was true as of date Y?", including
retroactive corrections). Five-verb Python API, or an MCP server for Claude
Code / Cursor.

It's ~74% on LongMemEval-S (gpt-oss-120b reader, llama-3.3 judge) and 100% on
scripted supersession/as-of benches. I also wrote up, in full, four ideas that
*didn't* work (deterministic counting/router/distillation/date-math — all
net-negative) and the measurement mistake that made them look good at first
(failure-only evaluation overstates gains). The negative results are in the repo.

Not a reasoning engine — it's the state layer under the reasoner. Feedback
welcome, especially on the supersession semantics.

Repo: https://github.com/Genkryptos/Continuum

---

## Communities to post (from RELEASE_PLAN WS-E)
- Show HN
- r/LocalLLaMA
- LangChain Discord
- AI-engineering communities

## Posting checklist (owner)
- [ ] Keys rotated, `release-3.0` merged to `main`, tag `v2.0.0`.
- [ ] PyPI `continuum-memory` published; `pip install continuum-memory` verified
      on a clean machine.
- [ ] README numbers match a fresh benchmark run (RELEASE_PLAN WS-B).
- [ ] Links in this file resolve.
