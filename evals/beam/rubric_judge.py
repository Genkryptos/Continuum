#!/usr/bin/env python3
"""
evals/beam/rubric_judge.py
==========================
Score BEAM answers with BEAM's *own* rubric-nugget judge, not entailment.

Entailment / substring judging is invalid for BEAM: a confident factual answer
that *ignores* a contradiction entails the gold facts and scores "correct",
which is the exact opposite of what BEAM tests. BEAM instead ships a per-question
``rubric`` (a list of nuggets) and judges each nugget independently 0 / 0.5 / 1.0,
then:

    question_score = mean(nugget_scores)          # PASS if >= 0.5
    category_acc   = fraction of questions with score >= 0.5

which is exactly the number Mem0 reports (contradiction 25%, event_ordering 15%).
This script re-scores *already-generated* answers (no re-inference): it joins a
bootstrap_ollama results file (question_id -> answer) against the converted BEAM
dataset (question_id -> question, question_type, rubric) and reports per-category
pass-rate + mean score.

The nugget prompt + system prompt are vendored verbatim from BEAM's
``unified_llm_judge_base_prompt`` (via the public Mem0 memory-benchmarks port) so
the numbers are comparable to published BEAM results.

    set -a && source .env && set +a
    python3.12 -m evals.beam.rubric_judge \
        --answers results/beam_topk/baseline_2026-07-26.json \
        --dataset scratchpad/beam_100k_c_o.json \
        --judge-model openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

BEAM_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator assessing whether an AI assistant's response "
    "satisfies specific rubric criteria. You must be objective, fair, and "
    "consistent. Return ONLY valid JSON with the exact format requested."
)

# Vendored verbatim from BEAM's unified judge (get_beam_nugget_judge_prompt).
NUGGET_PROMPT = """Evaluate whether the following LLM response demonstrates compliance with the specified RUBRIC CRITERION.

QUESTION:
{question}

LLM RESPONSE:
{response}

RUBRIC CRITERION:
{nugget}

SCORING GUIDELINES:

First, determine whether the rubric criterion is a POSITIVE requirement (the response SHOULD include something) or a NEGATIVE constraint (the response SHOULD NOT include something).

**For POSITIVE requirements** (response should contain, mention, or demonstrate something):
- **1.0 (Complete Compliance)**: The required element is present, accurate, and complete.
- **0.5 (Partial Compliance)**: The required element is partially present, has minor inaccuracies, or is incomplete.
- **0.0 (No Compliance)**: The required element is missing, incorrect, or the response is entirely off-topic / non-responsive.

**For NEGATIVE constraints** (response should NOT contain or should avoid something):
- **1.0**: The response is responsive AND the prohibited element is absent.
- **0.5**: The response is responsive but contains a borderline/ambiguous reference to the prohibited element.
- **0.0**: The prohibited element is present, OR the response is non-responsive.

**Compound statement handling**: If the rubric criterion joins multiple required elements with "and" or commas: all present = 1.0, some = 0.5, none = 0.0.

EVALUATION RULES:
1. Semantic tolerance: paraphrases and synonyms are acceptable.
2. Numeric/date equivalence: "$68,000" = "68k" = "sixty-eight thousand dollars".
3. Ignore case / punctuation / whitespace differences.
4. Do not penalize hedging, passive voice, or verbosity if the substance satisfies the criterion.
5. Do not penalize tone/format/length unless the criterion requires a format.
6. If the response is off-topic or refuses, score 0.0.
7. Evaluate this criterion in isolation.
8. Vague/generic answers score lower than specific ones.

Return your evaluation as a JSON object with exactly two fields:
{{"score": <0.0 or 0.5 or 1.0>, "reason": "<one concise sentence>"}}"""


def _clamp(raw: float) -> float:
    if raw >= 0.75:
        return 1.0
    if raw >= 0.25:
        return 0.5
    return 0.0


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a completion, tolerating prose/fences."""
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # last-ditch: look for a bare score token
    for tok, val in (("1.0", 1.0), ("0.5", 0.5), ("0.0", 0.0)):
        if tok in text:
            return {"score": val, "reason": "regex-fallback"}
    return {"score": 0.0, "reason": f"parse-error: {text[:120]}"}


async def _judge_nugget(
    client: httpx.AsyncClient, model: str, key: str,
    question: str, nugget: str, response: str, sem: asyncio.Semaphore,
) -> float:
    prompt = NUGGET_PROMPT.format(question=question, nugget=nugget, response=response)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": BEAM_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 200,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    async with sem:
        for attempt in range(3):
            try:
                r = await client.post(OPENROUTER_URL, json=payload, headers=headers, timeout=90)
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
                raw = _extract_json(content)
                return _clamp(float(raw.get("score", 0.0)))
            except Exception:
                if attempt == 2:
                    return 0.0
                await asyncio.sleep(1.5 * (attempt + 1))
    return 0.0


async def _judge_question(
    client, model, key, q: dict, answer: str, sem,
) -> dict[str, Any]:
    nuggets = q.get("rubric") or []
    if not nuggets:
        return {"question_id": q["question_id"], "score": 0.0,
                "pass": False, "error": "no-rubric", "n_nuggets": 0}
    scores = await asyncio.gather(*[
        _judge_nugget(client, model, key, q["question"], n, answer, sem)
        for n in nuggets
    ])
    mean = statistics.mean(scores) if scores else 0.0
    return {
        "question_id": q["question_id"],
        "question_type": q["question_type"],
        "score": round(mean, 4),
        "pass": mean >= 0.5,
        "n_nuggets": len(nuggets),
        "nugget_scores": [round(s, 2) for s in scores],
    }


async def run(answers_path: Path, dataset_path: Path, model: str, out_path: Path) -> int:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("[judge] set OPENROUTER_API_KEY (source .env)", file=sys.stderr)
        return 1
    ds = {r["question_id"]: r for r in json.loads(dataset_path.read_text())}
    payload = json.loads(answers_path.read_text())
    rows = payload["rows"] if isinstance(payload, dict) else payload
    ans = {r["question_id"]: (r.get("answer") or "") for r in rows}

    todo = [(qid, ds[qid], a) for qid, a in ans.items() if qid in ds]
    print(f"  judging {len(todo)} answered questions with rubric ({model}) …")

    sem = asyncio.Semaphore(8)
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[
            _judge_question(client, model, key, q, a, sem) for _, q, a in todo
        ])

    by_type_pass: dict[str, list[bool]] = defaultdict(list)
    by_type_score: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_type_pass[r["question_type"]].append(r["pass"])
        by_type_score[r["question_type"]].append(r["score"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"model": model, "answers": str(answers_path),
                                    "results": results}, indent=2))

    print("\n  ── BEAM rubric-nugget scoring (PASS = mean nugget score ≥ 0.5) ──")
    print(f"  {'category':28s} {'PASS-rate (acc)':>18s}   {'mean score':>10s}")
    allp: list[bool] = []
    for t in sorted(by_type_pass):
        p = by_type_pass[t]; s = by_type_score[t]
        allp += p
        print(f"  {t:28s} {sum(p):2d}/{len(p):2d} = {100*sum(p)/len(p):5.1f}%     "
              f"{statistics.mean(s):6.3f}")
    print(f"  {'OVERALL':28s} {sum(allp):2d}/{len(allp):2d} = {100*sum(allp)/len(allp):5.1f}%")
    print(f"\n  wrote {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--answers", type=Path, required=True,
                   help="bootstrap_ollama baseline results JSON (question_id -> answer).")
    p.add_argument("--dataset", type=Path, required=True,
                   help="Converted BEAM JSON carrying question/question_type/rubric.")
    p.add_argument("--judge-model", default="openai/gpt-4o-mini")
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args(argv)
    out = a.output or a.answers.parent / "rubric_judged.json"
    return asyncio.run(run(a.answers, a.dataset, a.judge_model, out))


if __name__ == "__main__":
    raise SystemExit(main())
