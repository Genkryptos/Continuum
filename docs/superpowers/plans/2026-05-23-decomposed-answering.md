# Decomposed Answering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace retrieval-only decomposition with a full `decompose -> retrieve per subquestion -> answer per subquestion -> synthesize final answer` path for LongMemEval.

**Architecture:** Keep the existing `DecompositionRetriever` as the retrieval-union baseline. Add a separate decomposed-answering adapter path that decomposes the original question, retrieves evidence independently per subquestion, answers each subquestion against its own evidence, and then synthesizes a final answer from the structured intermediate answers. Atomic questions should fall back to the normal answer path to avoid the single-session regressions observed in the current decomposer run.

**Tech Stack:** Python 3.12, asyncio, dataclasses, existing `ContinuumAdapter`, existing LongMemEval bootstrap harness, pytest.

---

## File Structure

- Modify `evals/longmemeval/decompose.py`
  - Export a public `build_decompose_prompt(question: str) -> str`.
  - Keep `_build_decompose_prompt` as a backward-compatible alias if desired.

- Create `evals/longmemeval/decomposed_answer.py`
  - Own pure prompt helpers and data structures:
    - `SubAnswer`
    - `extract_session_ids(ctx)`
    - `build_subanswer_prompt(subquestion, ctx)`
    - `build_final_synthesis_prompt(question, subanswers)`

- Modify `evals/longmemeval/bootstrap_ollama.py`
  - Add `_DecomposedAnsweringAdapter`.
  - Add `decompose_answer` parameters to `make_adapter_factory`.
  - Add CLI flag `--decompose-answer`.
  - Log decomposed-answer mode separately from the existing `--decompose` retrieval-union mode.

- Create `tests/unit/evals/test_decomposed_answer.py`
  - Test pure helpers and the adapter behavior with fake LLM/retriever objects.

- Modify `tests/unit/evals/test_decompose.py`
  - Add coverage for the public prompt builder.

---

### Task 1: Make Decomposition Prompt Public

**Files:**
- Modify: `evals/longmemeval/decompose.py`
- Test: `tests/unit/evals/test_decompose.py`

- [ ] **Step 1: Write the failing test**

Add this to `tests/unit/evals/test_decompose.py`:

```python
from evals.longmemeval.decompose import build_decompose_prompt


def test_build_decompose_prompt_contains_question() -> None:
    prompt = build_decompose_prompt("Did X happen before Y?")

    assert "Question: Did X happen before Y?" in prompt
    assert prompt.rstrip().endswith("Sub-questions:")
    assert "Return between 1 and 4 sub-questions" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/pytest tests/unit/evals/test_decompose.py::test_build_decompose_prompt_contains_question -q
```

Expected: FAIL with `ImportError` or `cannot import name 'build_decompose_prompt'`.

- [ ] **Step 3: Implement the public helper**

In `evals/longmemeval/decompose.py`, replace `_build_decompose_prompt` with:

```python
def build_decompose_prompt(question: str) -> str:
    return (
        DECOMPOSE_SYSTEM_PROMPT
        + f"\nQuestion: {question}\nSub-questions:"
    )


_build_decompose_prompt = build_decompose_prompt
```

Then update `_decompose` to call the public helper:

```python
reply = await self.llm.complete(
    prompt=build_decompose_prompt(question),
    max_tokens=self.decompose_max_tokens,
)
```

Add `build_decompose_prompt` to `__all__`:

```python
__all__ = [
    "DecompositionRetriever",
    "parse_subquestions",
    "build_decompose_prompt",
    "DECOMPOSE_SYSTEM_PROMPT",
]
```

- [ ] **Step 4: Run tests**

Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/pytest tests/unit/evals/test_decompose.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evals/longmemeval/decompose.py tests/unit/evals/test_decompose.py
git commit -m "refactor: expose longmemeval decomposition prompt builder"
```

---

### Task 2: Add Decomposed Answer Prompt Helpers

**Files:**
- Create: `evals/longmemeval/decomposed_answer.py`
- Create: `tests/unit/evals/test_decomposed_answer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/evals/test_decomposed_answer.py` with:

```python
from __future__ import annotations

from continuum.core.types import ContextBundle, MemoryItem, MemoryTier, TokenBudget
from evals.longmemeval.decomposed_answer import (
    SubAnswer,
    build_final_synthesis_prompt,
    build_subanswer_prompt,
    extract_session_ids,
)


def _budget() -> TokenBudget:
    return TokenBudget(
        total=8000,
        stm_reserved=500,
        mtm_reserved=500,
        ltm_reserved=2000,
        response_reserved=500,
    )


def _ctx() -> ContextBundle:
    return ContextBundle(
        items=[
            MemoryItem(
                id="a",
                content="I started using the Fitbit Charge 3 in March.",
                tier=MemoryTier.STM,
                metadata={"role": "user", "session_id": "s1"},
            ),
            MemoryItem(
                id="b",
                content="It is now December, so I have used it for 9 months.",
                tier=MemoryTier.STM,
                metadata={"role": "assistant", "session_id": "s2"},
            ),
        ],
        messages=[],
        tokens_used=20,
        budget=_budget(),
        tier_breakdown={"stm": 20, "mtm": 0, "ltm": 0},
    )


def test_extract_session_ids_dedupes_in_order() -> None:
    ctx = _ctx()
    ctx.items.append(
        MemoryItem(
            id="c",
            content="duplicate session",
            tier=MemoryTier.STM,
            metadata={"session_id": "s1"},
        )
    )

    assert extract_session_ids(ctx) == ["s1", "s2"]


def test_build_subanswer_prompt_uses_only_subquestion_and_context() -> None:
    prompt = build_subanswer_prompt("How long have I used the Fitbit?", _ctx())

    assert "Sub-question: How long have I used the Fitbit?" in prompt
    assert "I started using the Fitbit Charge 3 in March." in prompt
    assert "Answer the sub-question using only the evidence." in prompt
    assert "Original question:" not in prompt


def test_build_final_synthesis_prompt_contains_structured_subanswers() -> None:
    prompt = build_final_synthesis_prompt(
        "How long have I used the Fitbit?",
        [
            SubAnswer(
                subquestion="When did I start using the Fitbit?",
                answer="March",
                evidence_session_ids=["s1"],
                evidence_text="I started using the Fitbit Charge 3 in March.",
                hit_count=1,
            ),
            SubAnswer(
                subquestion="What month is it now?",
                answer="December",
                evidence_session_ids=["s2"],
                evidence_text="It is now December.",
                hit_count=1,
            ),
        ],
    )

    assert "Original question: How long have I used the Fitbit?" in prompt
    assert "Sub-question 1: When did I start using the Fitbit?" in prompt
    assert "Sub-answer 1: March" in prompt
    assert "Sessions 1: s1" in prompt
    assert "Return only the final answer" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/pytest tests/unit/evals/test_decomposed_answer.py -q
```

Expected: FAIL because `evals.longmemeval.decomposed_answer` does not exist.

- [ ] **Step 3: Implement helper module**

Create `evals/longmemeval/decomposed_answer.py`:

```python
"""
Structured subquestion answering helpers for LongMemEval.

The retrieval-only decomposer improves context coverage slightly but still
asks the final model to compose all reasoning in one pass. These helpers
support the stronger path: answer each atomic sub-question first, then
synthesize the final answer from those intermediate answers.
"""

from __future__ import annotations

from dataclasses import dataclass

from continuum.core.types import ContextBundle


@dataclass(frozen=True)
class SubAnswer:
    subquestion: str
    answer: str
    evidence_session_ids: list[str]
    evidence_text: str
    hit_count: int


def extract_session_ids(ctx: ContextBundle | None) -> list[str]:
    if ctx is None:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in getattr(ctx, "items", []) or []:
        sid = (getattr(item, "metadata", {}) or {}).get("session_id")
        if not sid:
            continue
        sid_s = str(sid)
        if sid_s in seen:
            continue
        seen.add(sid_s)
        out.append(sid_s)
    return out


def _context_text(ctx: ContextBundle | None) -> str:
    if ctx is None or not getattr(ctx, "items", None):
        return ""
    chunks: list[str] = []
    for idx, item in enumerate(ctx.items, start=1):
        role = str((item.metadata or {}).get("role", "user"))
        sid = str((item.metadata or {}).get("session_id", ""))
        prefix = f"[{idx}]"
        if sid:
            prefix += f" session={sid}"
        chunks.append(f"{prefix} role={role}\n{item.content}")
    return "\n\n".join(chunks)


def build_subanswer_prompt(subquestion: str, ctx: ContextBundle | None) -> str:
    evidence = _context_text(ctx)
    return (
        "Answer the sub-question using only the evidence. "
        "If the evidence does not contain the answer, say \"I don't know\".\n\n"
        f"Evidence:\n{evidence or '[no evidence retrieved]'}\n\n"
        f"Sub-question: {subquestion}\n"
        "Sub-answer:"
    )


def build_final_synthesis_prompt(
    question: str,
    subanswers: list[SubAnswer],
) -> str:
    blocks: list[str] = []
    for idx, sub in enumerate(subanswers, start=1):
        sessions = ", ".join(sub.evidence_session_ids) or "none"
        blocks.append(
            f"Sub-question {idx}: {sub.subquestion}\n"
            f"Sub-answer {idx}: {sub.answer}\n"
            f"Sessions {idx}: {sessions}\n"
            f"Evidence {idx}: {sub.evidence_text or '[no evidence]'}"
        )
    return (
        "Use the sub-answers to answer the original question. "
        "Resolve comparisons, arithmetic, before/after ordering, and updates "
        "explicitly from the sub-answers. If the sub-answers are insufficient, "
        "say \"I don't know\". Return only the final answer.\n\n"
        f"Original question: {question}\n\n"
        + "\n\n".join(blocks)
        + "\n\nFinal answer:"
    )
```

- [ ] **Step 4: Run helper tests**

Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/pytest tests/unit/evals/test_decomposed_answer.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evals/longmemeval/decomposed_answer.py tests/unit/evals/test_decomposed_answer.py
git commit -m "feat: add decomposed answer prompt helpers"
```

---

### Task 3: Implement Decomposed Answering Adapter

**Files:**
- Modify: `evals/longmemeval/bootstrap_ollama.py`
- Modify: `tests/unit/evals/test_decomposed_answer.py`

- [ ] **Step 1: Add adapter behavior tests**

Append these tests to `tests/unit/evals/test_decomposed_answer.py`:

```python
import pytest

from evals.longmemeval.bootstrap_ollama import _DecomposedAnsweringAdapter


class _FakeRetriever:
    def __init__(self, table: dict[str, ContextBundle]) -> None:
        self.table = table
        self.queries: list[str] = []

    async def retrieve(self, query, budget):
        self.queries.append(query.text)
        return self.table.get(
            query.text,
            ContextBundle(
                items=[],
                messages=[],
                tokens_used=0,
                budget=budget,
                tier_breakdown={"stm": 0, "mtm": 0, "ltm": 0},
            ),
        )


class _FakeSession:
    def __init__(self, retriever: _FakeRetriever) -> None:
        self.retriever = retriever


class _ScriptedLLM:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def complete(self, *, prompt: str, max_tokens: int) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("unexpected LLM call")
        return self.replies.pop(0)


@pytest.mark.asyncio
async def test_decomposed_adapter_answers_subquestions_then_synthesizes() -> None:
    retriever = _FakeRetriever(
        {
            "When did I start using the Fitbit?": _ctx(),
            "What month is it now?": _ctx(),
        }
    )
    llm = _ScriptedLLM(
        [
            "When did I start using the Fitbit?\nWhat month is it now?",
            "March",
            "December",
            "9 months",
        ]
    )
    adapter = _DecomposedAnsweringAdapter(
        session=_FakeSession(retriever),
        llm=llm,
        answer_max_tokens=100,
        subanswer_max_tokens=40,
    )

    answer = await adapter.answer_question("How long have I used the Fitbit?")

    assert answer == "9 months"
    assert retriever.queries == [
        "When did I start using the Fitbit?",
        "What month is it now?",
    ]
    assert adapter.last_ctx is not None
    assert len(adapter.last_ctx.items) == 2
    assert adapter.last_decomposition_stats["n_sub_questions"] == 2
    assert adapter.last_decomposition_stats["sub_answers"] == ["March", "December"]


@pytest.mark.asyncio
async def test_decomposed_adapter_atomic_question_uses_normal_answer_path() -> None:
    retriever = _FakeRetriever({"What degree did I graduate with?": _ctx()})
    llm = _ScriptedLLM(
        [
            "What degree did I graduate with?",
            "Business Administration",
        ]
    )
    adapter = _DecomposedAnsweringAdapter(
        session=_FakeSession(retriever),
        llm=llm,
        answer_max_tokens=100,
    )

    answer = await adapter.answer_question("What degree did I graduate with?")

    assert answer == "Business Administration"
    assert retriever.queries == ["What degree did I graduate with?"]
    assert adapter.last_decomposition_stats["mode"] == "atomic_fallback"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/pytest tests/unit/evals/test_decomposed_answer.py -q
```

Expected: FAIL because `_DecomposedAnsweringAdapter` does not exist.

- [ ] **Step 3: Add imports to bootstrap**

In `evals/longmemeval/bootstrap_ollama.py`, add:

```python
from evals.longmemeval.decompose import (
    DecompositionRetriever,
    build_decompose_prompt,
    parse_subquestions,
)
from evals.longmemeval.decomposed_answer import (
    SubAnswer,
    build_final_synthesis_prompt,
    build_subanswer_prompt,
)
```

Replace the existing single import:

```python
from evals.longmemeval.decompose import DecompositionRetriever
```

- [ ] **Step 4: Implement `_DecomposedAnsweringAdapter`**

Add this class after `_OptimizingAdapter` in `evals/longmemeval/bootstrap_ollama.py`:

```python
class _DecomposedAnsweringAdapter(_IngestingAdapter):
    """
    Full decomposed answering path for LongMemEval.

    Unlike DecompositionRetriever, this does not only merge retrieved
    snippets. It answers each sub-question against its own evidence, then
    synthesizes the final answer from those intermediate answers.
    """

    def __init__(
        self,
        *,
        decompose_max_tokens: int = 160,
        subanswer_max_tokens: int = 80,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.decompose_max_tokens = decompose_max_tokens
        self.subanswer_max_tokens = subanswer_max_tokens
        self.last_decomposition_stats: dict[str, Any] = {}

    async def _decompose_question(self, question: str) -> list[str]:
        try:
            reply = await self.llm.complete(
                prompt=build_decompose_prompt(question),
                max_tokens=self.decompose_max_tokens,
            )
        except Exception:
            log.exception("decomposed-answer decomposition failed")
            return [question]
        return parse_subquestions(reply, original=question)

    @staticmethod
    def _is_atomic(question: str, subquestions: list[str]) -> bool:
        if len(subquestions) != 1:
            return False
        return subquestions[0].strip().lower() == question.strip().lower()

    @staticmethod
    def _merge_contexts(contexts: list[ContextBundle]) -> ContextBundle | None:
        if not contexts:
            return None
        first = contexts[0]
        seen: set[str] = set()
        items: list[MemoryItem] = []
        for ctx in contexts:
            for item in ctx.items:
                if item.id in seen:
                    continue
                seen.add(item.id)
                items.append(item)
        token_count = sum(estimate_tokens_text(item.content) for item in items)
        messages = [
            {
                "role": str(item.metadata.get("role", "user"))
                if item.metadata else "user",
                "content": item.content,
            }
            for item in items
        ]
        return dataclasses_replace(
            first,
            items=items,
            messages=messages,
            tokens_used=token_count,
            tier_breakdown={"stm": token_count, "mtm": 0, "ltm": 0},
            debug_info={
                "retrieval_mode": "decompose_answer",
                "merged_items": len(items),
            },
        )

    async def answer_question(self, question: str) -> str:
        subquestions = await self._decompose_question(question)
        if self._is_atomic(question, subquestions):
            answer = await super().answer_question(question)
            self.last_decomposition_stats = {
                "mode": "atomic_fallback",
                "n_sub_questions": 1,
                "sub_questions": subquestions,
                "sub_answers": [],
            }
            return answer

        retriever = getattr(self.session, "retriever", None)
        contexts: list[ContextBundle] = []
        subanswers: list[SubAnswer] = []
        if retriever is None:
            self.last_ctx = None
            return "I don't know"

        for subquestion in subquestions:
            try:
                ctx = await retriever.retrieve(Query(text=subquestion), self.budget)
            except Exception:
                log.exception("sub-question retrieve failed for %r", subquestion)
                ctx = ContextBundle(
                    items=[],
                    messages=[],
                    tokens_used=0,
                    budget=self.budget,
                    tier_breakdown={"stm": 0, "mtm": 0, "ltm": 0},
                )
            contexts.append(ctx)
            prompt = build_subanswer_prompt(subquestion, ctx)
            try:
                subanswer = await self.llm.complete(
                    prompt=prompt,
                    max_tokens=self.subanswer_max_tokens,
                )
            except Exception as exc:
                log.exception("sub-question answer failed for %r", subquestion)
                subanswer = f"I don't know ({exc!r})"
            evidence_text = "\n".join(item.content for item in ctx.items[:4])
            subanswers.append(
                SubAnswer(
                    subquestion=subquestion,
                    answer=str(subanswer).strip(),
                    evidence_session_ids=[
                        str((item.metadata or {}).get("session_id", ""))
                        for item in ctx.items
                        if (item.metadata or {}).get("session_id")
                    ],
                    evidence_text=evidence_text,
                    hit_count=len(ctx.items),
                )
            )

        self.last_ctx = self._merge_contexts(contexts)
        final_prompt = build_final_synthesis_prompt(question, subanswers)
        try:
            answer = await self.llm.complete(
                prompt=final_prompt,
                max_tokens=self.answer_max_tokens,
            )
        except Exception as exc:
            log.exception("final synthesis failed for %r", question)
            answer = f"[error: {exc!r}]"

        self.last_decomposition_stats = {
            "mode": "decompose_answer",
            "n_sub_questions": len(subquestions),
            "sub_questions": subquestions,
            "sub_answers": [s.answer for s in subanswers],
            "subquestion_hits": [s.hit_count for s in subanswers],
        }
        return str(answer).strip()
```

- [ ] **Step 5: Run adapter tests**

Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/pytest tests/unit/evals/test_decomposed_answer.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add evals/longmemeval/bootstrap_ollama.py tests/unit/evals/test_decomposed_answer.py
git commit -m "feat: add decomposed answering adapter"
```

---

### Task 4: Wire CLI and Factory Flag

**Files:**
- Modify: `evals/longmemeval/bootstrap_ollama.py`
- Modify: `tests/unit/evals/test_decomposed_answer.py`

- [ ] **Step 1: Add factory tests**

Append this test to `tests/unit/evals/test_decomposed_answer.py`:

```python
from evals.longmemeval.bootstrap_ollama import FlatHaystackStore, make_adapter_factory


class _FakeEmbedder:
    def encode(self, texts: list[str]):
        import numpy as np

        rows = []
        for idx, _ in enumerate(texts):
            row = np.zeros(4, dtype=np.float32)
            row[idx % 4] = 1.0
            rows.append(row)
        return np.stack(rows, axis=0)


def test_factory_can_build_decomposed_answering_adapter() -> None:
    llm = _ScriptedLLM(["What degree did I graduate with?", "Business Administration"])
    factory = make_adapter_factory(
        llm=llm,
        embedder=_FakeEmbedder(),
        decompose_answer=True,
    )

    adapter = factory()

    assert isinstance(adapter, _DecomposedAnsweringAdapter)
    assert isinstance(adapter.session.stm, FlatHaystackStore)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/pytest tests/unit/evals/test_decomposed_answer.py::test_factory_can_build_decomposed_answering_adapter -q
```

Expected: FAIL because `make_adapter_factory` has no `decompose_answer` argument.

- [ ] **Step 3: Extend `make_adapter_factory` signature**

In `evals/longmemeval/bootstrap_ollama.py`, add parameters:

```python
    decompose_answer: bool = False,
    decompose_answer_subanswer_max_tokens: int = 80,
```

Place them next to the existing `decompose` parameters.

- [ ] **Step 4: Return the new adapter from the factory**

In `make_adapter_factory.factory`, before the existing `if chain is None:` block, add:

```python
        if decompose_answer:
            return _DecomposedAnsweringAdapter(
                session=session,
                llm=llm,
                answer_max_tokens=answer_max_tokens,
                subanswer_max_tokens=decompose_answer_subanswer_max_tokens,
            )
```

Do not wrap the retriever in `DecompositionRetriever` when `decompose_answer` is true. Use the base `STMSemanticRetriever`, because `_DecomposedAnsweringAdapter` owns decomposition itself.

- [ ] **Step 5: Add CLI arguments**

In `_parse_args`, add:

```python
    p.add_argument(
        "--decompose-answer",
        action="store_true",
        help=(
            "Enable full decomposed answering: decompose the question, "
            "retrieve and answer each sub-question, then synthesize the final "
            "answer. This is stronger than --decompose, which only merges "
            "retrieved context."
        ),
    )
    p.add_argument(
        "--subanswer-max-tokens",
        type=int,
        default=80,
        help="Token cap for each decomposed sub-question answer. Default 80.",
    )
```

- [ ] **Step 6: Add mode validation and logging**

In `main_async`, after the existing `if args.decompose:` log, add:

```python
    if args.decompose and args.decompose_answer:
        raise ValueError(
            "Use either --decompose or --decompose-answer, not both. "
            "--decompose is retrieval-only; --decompose-answer performs "
            "sub-question answering and final synthesis."
        )
    if args.decompose_answer:
        log.info(
            "decomposed answering ENABLED - split question, retrieve and "
            "answer each part, synthesize final answer"
        )
```

- [ ] **Step 7: Pass factory arguments**

In the `make_adapter_factory(...)` call, add:

```python
        decompose_answer=args.decompose_answer,
        decompose_answer_subanswer_max_tokens=args.subanswer_max_tokens,
```

- [ ] **Step 8: Run focused tests**

Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/pytest tests/unit/evals/test_decomposed_answer.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add evals/longmemeval/bootstrap_ollama.py tests/unit/evals/test_decomposed_answer.py
git commit -m "feat: wire decomposed answering benchmark mode"
```

---

### Task 5: Add Telemetry to Baseline Rows

**Files:**
- Modify: `evals/longmemeval/baseline.py`
- Modify: `tests/unit/evals/test_longmemeval_baseline.py`

- [ ] **Step 1: Add test for optional decomposition stats**

In `tests/unit/evals/test_longmemeval_baseline.py`, add a fake adapter test that asserts optional stats persist:

```python
@pytest.mark.asyncio
async def test_run_one_records_decomposition_stats_when_available() -> None:
    from evals.longmemeval.baseline import EvalRow, _run_one

    class Adapter:
        last_ctx = None
        last_decomposition_stats = {
            "mode": "decompose_answer",
            "n_sub_questions": 2,
            "sub_questions": ["Q1?", "Q2?"],
            "sub_answers": ["A1", "A2"],
            "subquestion_hits": [4, 3],
        }

        async def process_conversation(self, messages):
            return None

        async def answer_question(self, question):
            return "final"

    result = await _run_one(
        row=EvalRow(
            question_id="q",
            question="Q?",
            expected_answer="final",
            messages=[],
            answer_session_ids=[],
        ),
        adapter=Adapter(),
        scorer=lambda answer, expected: answer == expected,
        answerer="gpt-4o-mini",
        prices={"gpt-4o-mini": 0.0},
        count_tokens=lambda text: 1,
    )

    assert result.decomposition_stats["mode"] == "decompose_answer"
    assert result.decomposition_stats["sub_answers"] == ["A1", "A2"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/pytest tests/unit/evals/test_longmemeval_baseline.py::test_run_one_records_decomposition_stats_when_available -q
```

Expected: FAIL because `RowResult` has no `decomposition_stats` field.

- [ ] **Step 3: Add field to `RowResult`**

In `evals/longmemeval/baseline.py`, add to `RowResult`:

```python
    decomposition_stats: dict[str, Any] = dataclasses.field(default_factory=dict)
```

Place it after `strategy_savings`.

- [ ] **Step 4: Populate field in `_run_one`**

In the `RowResult(...)` construction, add:

```python
        decomposition_stats=dict(
            getattr(adapter, "last_decomposition_stats", None) or {}
        ),
```

- [ ] **Step 5: Run baseline tests**

Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/pytest tests/unit/evals/test_longmemeval_baseline.py tests/unit/evals/test_decomposed_answer.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add evals/longmemeval/baseline.py tests/unit/evals/test_longmemeval_baseline.py
git commit -m "feat: record decomposed answering telemetry"
```

---

### Task 6: Smoke Benchmark and Acceptance Check

**Files:**
- No source changes expected.
- Output: `results/gpt4omini_decompose_answer/smoke/baseline_YYYY-MM-DD.json`

- [ ] **Step 1: Run unit test suite for eval changes**

Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/pytest tests/unit/evals/test_decompose.py tests/unit/evals/test_decomposed_answer.py tests/unit/evals/test_longmemeval_baseline.py -q
```

Expected: PASS.

- [ ] **Step 2: Run a 5-question smoke benchmark**

Run:

```bash
python3 -m evals.longmemeval.bootstrap_ollama --provider openai --model gpt-4o-mini --optimizer --decompose-answer --output results/gpt4omini_decompose_answer --limit 5
```

Expected:
- CLI logs `decomposed answering ENABLED`.
- Smoke run completes.
- `results/gpt4omini_decompose_answer/smoke/baseline_YYYY-MM-DD.json` exists.
- Smoke JSON rows contain `decomposition_stats`.

- [ ] **Step 3: Inspect smoke telemetry**

Run:

```bash
python3 -m json.tool results/gpt4omini_decompose_answer/smoke/baseline_2026-05-23.json
```

Expected:
- Atomic rows have `"mode": "atomic_fallback"`.
- Multi-subquestion rows have `"mode": "decompose_answer"`.
- `sub_questions`, `sub_answers`, and `subquestion_hits` are present.

- [ ] **Step 4: Run full benchmark only after smoke passes**

Run:

```bash
python3 -m evals.longmemeval.bootstrap_ollama --provider openai --model gpt-4o-mini --optimizer --decompose-answer --output results/gpt4omini_decompose_answer --full --yes
```

Expected:
- Accuracy should exceed the retrieval-only decomposer baseline of `29.2%`.
- Temporal reasoning should exceed `14.3%`.
- Multi-session should exceed `15.0%`.
- Single-session-user should not regress below the top-k baseline of `58.6%`; if it does, tighten atomic fallback before accepting the run.

- [ ] **Step 5: Commit benchmark notes**

If a benchmark summary file is added, commit it:

```bash
git add results/gpt4omini_decompose_answer
git commit -m "test: record decomposed answering benchmark results"
```

If benchmark artifacts are intentionally not committed, skip this commit and record the metrics in the PR description.

---

## Self-Review

- Spec coverage: The plan implements the missing architecture identified in the analysis: subquestion retrieval plus subquestion answering plus final composition. It also preserves the existing retrieval-only decomposer for direct comparison.
- Regression control: Atomic fallback directly targets the observed regressions on single-session questions.
- Evaluation control: New CLI flag keeps `--decompose` and `--decompose-answer` comparisons clean.
- Telemetry: Baseline rows record subquestions, subanswers, and hit counts so failures can be inspected without rerunning.
- Test coverage: Pure helper tests, adapter tests, factory tests, baseline telemetry tests, and smoke/full benchmark acceptance are included.
