# Continuum docs

Four pages, each readable independently. Pick whichever matches what
you're trying to do.

| Doc | Read when |
|---|---|
| [**Quickstart**](quickstart.md) | You want a working `ContinuumSession` in five commands. |
| [**Architecture**](architecture.md) | You want to understand the three tiers, the promotion lifecycle, and what supersession + bi-temporal actually mean. |
| [**Config reference**](config.md) | You want to know what every knob does — env vars, YAML, the 13 sub-configs. |
| [**Operations**](operations.md) | You're putting Continuum into production: Postgres setup, scaling, observability, what to monitor. |

Looking for the *value-prop summary*, benchmarks, or the LongMemEval
report? Those live at the top level:

* [`README.md`](../README.md) — landing page with headline numbers.
* [`bench/`](../bench/) — `make bench-all` runs the four memory-op benchmarks.
* [`findings/longmemeval_2026-05.md`](../findings/longmemeval_2026-05.md) — the honest LongMemEval-S evaluation.
* [`examples/chat_agent/`](../examples/chat_agent/) — `make demo-chat` runs the 60-second walkthrough.

## What's *not* documented here

* **Internal API reference**. Every public class has docstrings; run
  `pydoc continuum.core.session` etc. for the ground truth. We
  deliberately don't generate stale HTML — read the code.
* **The eval harness.** That lives at [`evals/longmemeval/`](../evals/longmemeval/)
  with its own README and the reproducibility artifact at
  [`findings/longmemeval/repro/`](../findings/longmemeval/repro/).
* **Phase planning / decisions.** The full project history is in
  git — these docs cover the current state.

## Reading order if you're new

1. The top-level [`README.md`](../README.md) — 60-second value-prop pitch.
2. Run `make demo-chat` — see it work.
3. [Architecture](architecture.md) — internalize the tier model.
4. [Quickstart](quickstart.md) — write your first `ContinuumSession`.
5. [Config](config.md) + [Operations](operations.md) — when you're ready to wire it to Postgres.

Roughly **30 minutes** if you read every page; **10 minutes** if you
skim quickstart + architecture and come back to the rest when needed.
