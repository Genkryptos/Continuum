# LongMemEval-S reproducibility artifact

This directory reproduces the two headline LongMemEval-S numbers from
[`findings/longmemeval_2026-05.md`](../../longmemeval_2026-05.md) — the
gpt-4o-mini top-k baseline and the `DecompositionRetriever` variant. It
is intentionally narrow: just enough to confirm the report's numbers
land in the same ±2pp band on a fresh run.

## What gets reproduced

| run name | what it tests | report section | expected accuracy |
|---|---|---|---|
| `gpt4omini_topk` | the gpt-4o-mini baseline retriever (single-shot top-k=4 + optimizer chain) | §2 row 6 | **26.2 % / 77.4 % recall** |
| `gpt4omini_decompose` | `DecompositionRetriever` — splits question into atomic sub-questions, retrieves per part | §2 row 4 | **29.2 % / 76.3 % recall** |

The Llama-3.3-70B long-context run (§2 row 3) is **not** reproduced
here — it requires an NVIDIA Build account, takes ~10 hours, and the
finding it underpins (the 32 % ceiling at 100 % recall) is already
recorded in the report's source JSON. We chose to keep the repro path
under $0.20 and 30 minutes.

## Requirements

- **OpenAI key** in env (`OPENAI_API_KEY`) or `.env` (`OPEN_AI_KEY=…`,
  the legacy name from the original runs — `run_repro.sh` accepts
  both).
- **Python 3.12** framework interpreter at
  `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`.
  Override with `PY3=/path/to/python3 make repro-longmemeval`.
- **Deps** from `requirements-eval.txt`:

  ```bash
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
      -m pip install -r findings/longmemeval/repro/requirements-eval.txt
  ```

- **Dataset** at
  `evals/longmemeval/LongMemEval/data/longmemeval_s_cleaned.json` — vendored
  in the repo, no download needed.

## How to run

```bash
# From the repo root
make repro-longmemeval
```

That's the entire interface. The Make target shells out to
`run_repro.sh`, which:

1. Locates the OpenAI key (env → `.env` fallback).
2. Runs each configuration in `expected_results.json` as a full 500-question
   sweep through `evals/longmemeval/bootstrap_ollama.py`.
3. Writes raw results to `results/repro/<run-name>/baseline_*.json`.
4. Calls `verify.py` on each, comparing actual numbers against
   the expected values with ±2pp tolerance on percentages and ±10 on
   absolute counts (`missing_fact`, `wrong_retrieval`).
5. Exits 0 if everything is inside tolerance, 1 otherwise.

Total wall-clock: ~25-35 min. API cost: ~$0.10.

## What "within tolerance" means

The ±2pp band is empirically calibrated. Across our 7 prior runs, the
sources of variance are:

- **gpt-4o-mini snapshot drift.** OpenAI updates model snapshots without
  ID changes; same prompt at `temperature=0` can drift ~1pp.
- **Cosine ties.** all-MiniLM-L6-v2 produces enough near-tied scores
  that floating-point order can flip which top-4 candidate wins. We
  saw ±0.5pp from this between MPS and CPU embedder devices.
- **OpenAI rate-limit jitter.** Adaptive throttle may slightly change
  retry timing, but the answers are temperature-0 deterministic so
  this only affects latency, not accuracy.

If a run **fails** the verifier:

- Diff exceeds 5pp → likely a code regression. Check the most recent
  commit touching `evals/longmemeval/` or `continuum/optimizer/`.
- Diff in the 2-5pp range → likely a model-snapshot shift. Note the
  drift in the verifier output and consider rebasing the expected
  numbers if the new value reproduces stably.

## Layout

```
findings/longmemeval/repro/
├── README.md                  ← this file
├── expected_results.json      ← the contract: configs + expected numbers + tolerance
├── requirements-eval.txt      ← pinned deps that produced the report's numbers
├── run_repro.sh               ← driver; called by `make repro-longmemeval`
└── verify.py                  ← compares a baseline_*.json against expected, ±2pp
```

The actual eval code lives in `evals/longmemeval/` (unchanged from the
original runs). This artifact is a thin reproducibility wrapper over
that code — not a fork.

## When to update `expected_results.json`

Update the `expected` block for a run when:

1. The model snapshot shifts and a new value is stably reproduced
   across ≥2 fresh runs.
2. Code changes intentionally move the number (e.g. a real retrieval
   improvement). Record the rationale in the commit message and
   regenerate the report's tables via `findings/charts/extract_tables.py`.

**Do not** update the expected numbers to mask a regression. If the
verifier fails because of a real code change, fix the code or revert
it; the contract here is *load-bearing* for the report.

## See also

- [`findings/longmemeval_2026-05.md`](../../longmemeval_2026-05.md) —
  the full technical report whose numbers this artifact reproduces.
- [`findings/charts/extract_tables.py`](../../charts/extract_tables.py) —
  regenerates the report's tables from raw JSON.
- [`findings/charts/render_charts.py`](../../charts/render_charts.py) —
  regenerates the report's figures.
- [`evals/longmemeval/bootstrap_ollama.py`](../../../evals/longmemeval/bootstrap_ollama.py) —
  the actual eval driver.
