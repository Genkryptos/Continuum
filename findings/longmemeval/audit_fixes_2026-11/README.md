# LongMemEval-S audit-fixes runs (2026-11-07)

`results/` is gitignored, so these are the artifacts the §3.5 / §7 entries
in `evals/PREREGISTRATION.md` rest on. All runs: LongMemEval-S 500q, judge
`meta-llama/llama-3.3-70b-instruct` pinned to DeepInfra, `--top-k 48`,
`--use-ltm` off. All are `non_comparable: true`.

| file | what | accuracy |
|---|---|---|
| `muse_wonder_500.json` | `run_arm_audit_fixes.sh`, answerer `muse-wonder-1.0` | 90.8% |
| `oss120b_500.json` | `run_arm_audit_fixes_oss120b.sh`, answerer `openai/gpt-oss-120b` | 85.8% |
| `oss120b_500_opus_graded.json` | every row of the above regraded by hand (Opus 5.5) | 85.2% |
| `oss120b_empty17_8192.json` | the 17 rows that were empty at `--answer-max-tokens 2048`, rerun at 8192 | 10/17 |
| `oss120b_500_opus_graded_merged17.json` | Opus grade with the 17 rerun rows merged in (mixed caps) | 87.2% (llama 87.8%) |

Retrieval is identical between the muse and gpt-oss runs, so the gap between
them is the reader. Every empty gpt-oss answer used exactly the token cap.
