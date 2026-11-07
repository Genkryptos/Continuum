# Preserved benchmark artifacts

`bench/results/` is gitignored (`.gitignore:47`), so raw run output does not
survive a clone. These are the specific runs the findings documents rest on,
copied here so their citations resolve.

| file | what it shows |
|---|---|
| `scale_real_50000.json` | 50k real LongMemEval turns — the corrected headline (85% recall@10) |
| `ann_vs_exact_real_50000.json` | **the decisive one** — HNSW ties exact at 85%, 3.7ms vs 71.5ms |
| `scale_synthetic_50000.json` | the adversarial control (40% recall@10) |
| `ann_vs_exact_synthetic_50000.json` | the control's −75pp ANN gap, which did not survive real data |
| `scale_synthetic_5000.json` | 5k synthetic, for the ingestion scaling curve |

Each records server version, migrations, index parameters, embedding coverage,
and corpus source, so the environment behind a number is checked rather than
assumed. See `findings/scale_retrieval_2026-08.md`.

Regenerate with the commands in that document's §7.
