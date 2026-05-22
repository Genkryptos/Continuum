"""
tests/unit/evals/test_longmemeval_harness.py
============================================
Unit tests for the LongMemEval harness modules. Network, git, pip, and
the real benchmark library are all stubbed — every test runs offline
under a second.

Coverage
--------
* ``setup.setup_longmemeval`` — fresh clone, re-clone with --force,
  idempotent pull, missing requirements.txt
* ``adapter.ContinuumAdapter`` — process_conversation dispatches to
  session.process_turn for user turns and STM-appends assistants;
  answer_question retrieves + calls LLM + handles errors;
  format_prompt buckets by tier and omits empty sections
* ``run.run_benchmark`` — wires adapter into factory-provided
  benchmark, normalises results, writes JSON to disk
"""
from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path
from typing import Any

import pytest

from continuum.core.types import (
    ContextBundle,
    MemoryItem,
    MemoryTier,
    TokenBudget,
)
from evals.longmemeval import adapter as adapter_mod
from evals.longmemeval import run as run_mod
from evals.longmemeval import setup as setup_mod

# ---------------------------------------------------------------------------
# setup.py
# ---------------------------------------------------------------------------


class _FakeRunner:
    """subprocess stand-in that records every call."""

    def __init__(self, *, fail_on: list[str] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_on = fail_on or []

    def run(self, args: list[str], check: bool = False) -> Any:
        self.calls.append(list(args))
        if any(token in args for token in self.fail_on):
            raise subprocess.CalledProcessError(1, args)
        return subprocess.CompletedProcess(args, 0)


def test_setup_clones_when_missing(tmp_path: Path) -> None:
    target = tmp_path / "LongMemEval"
    runner = _FakeRunner()

    # Simulate the clone creating the directory + requirements.txt that
    # the second step needs. Our fake runner doesn't actually clone, so
    # create the files manually before that step would inspect them.
    def fake_run(args: list[str], check: bool = False) -> Any:
        runner.calls.append(list(args))
        if args[0] == "git" and args[1] == "clone":
            target.mkdir(parents=True, exist_ok=True)
            (target / "requirements.txt").write_text("# none yet\n")
        return subprocess.CompletedProcess(args, 0)

    runner.run = fake_run  # type: ignore[method-assign]
    result = setup_mod.setup_longmemeval(target=target, runner=runner)
    assert result == target
    # First call clones, second installs.
    assert runner.calls[0][:2] == ["git", "clone"]
    assert "pip" in runner.calls[1] and "install" in runner.calls[1]


def test_setup_pulls_when_already_cloned(tmp_path: Path) -> None:
    target = tmp_path / "LongMemEval"
    target.mkdir(parents=True)
    (target / "requirements.txt").write_text("")
    runner = _FakeRunner()
    setup_mod.setup_longmemeval(target=target, runner=runner)
    # Should issue `git -C <target> pull --ff-only`, not clone.
    assert runner.calls[0][:3] == ["git", "-C", str(target)]
    assert "pull" in runner.calls[0]


def test_setup_force_reclones(tmp_path: Path) -> None:
    target = tmp_path / "LongMemEval"
    target.mkdir(parents=True)
    (target / "stale.txt").write_text("old")
    runner = _FakeRunner()

    def fake_run(args: list[str], check: bool = False) -> Any:
        runner.calls.append(list(args))
        if args[0] == "git" and args[1] == "clone":
            target.mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(args, 0)

    runner.run = fake_run  # type: ignore[method-assign]
    setup_mod.setup_longmemeval(target=target, force=True, runner=runner)
    assert not (target / "stale.txt").exists()
    assert runner.calls[0][:2] == ["git", "clone"]


def test_setup_skips_pip_when_no_requirements(tmp_path: Path) -> None:
    target = tmp_path / "LongMemEval"
    runner = _FakeRunner()

    def fake_run(args: list[str], check: bool = False) -> Any:
        runner.calls.append(list(args))
        if args[0] == "git" and args[1] == "clone":
            target.mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(args, 0)

    runner.run = fake_run  # type: ignore[method-assign]
    setup_mod.setup_longmemeval(target=target, runner=runner)
    # Only the clone call should have run.
    assert len(runner.calls) == 1


def test_setup_propagates_git_failure(tmp_path: Path) -> None:
    target = tmp_path / "LongMemEval"
    runner = _FakeRunner(fail_on=["clone"])
    with pytest.raises(subprocess.CalledProcessError):
        setup_mod.setup_longmemeval(target=target, runner=runner)


# ---------------------------------------------------------------------------
# adapter.py — fakes
# ---------------------------------------------------------------------------


class _FakeStm:
    def __init__(self) -> None:
        self.appended: list[MemoryItem] = []

    async def append(self, item: MemoryItem) -> None:
        self.appended.append(item)


class _FakeRetriever:
    def __init__(self, ctx: ContextBundle | None) -> None:
        self.ctx = ctx
        self.calls = 0
        self.raise_ = False

    async def retrieve(
        self, query: Any, budget: TokenBudget
    ) -> ContextBundle | None:
        self.calls += 1
        if self.raise_:
            raise RuntimeError("retrieve broken")
        return self.ctx


class _FakeSession:
    def __init__(self, retriever: Any, stm: Any) -> None:
        self.retriever = retriever
        self.stm = stm
        self.turns: list[str] = []
        self.raise_on_turn = False

    async def process_turn(
        self, user_message: str, *, context_budget: TokenBudget
    ) -> str:
        if self.raise_on_turn:
            raise RuntimeError("turn boom")
        self.turns.append(user_message)
        return "ok"


class _FakeLLM:
    def __init__(self, output: str = "the answer", raise_: bool = False) -> None:
        self.output = output
        self.raise_ = raise_
        self.calls: list[dict[str, Any]] = []

    async def complete(self, *, prompt: str, max_tokens: int) -> str:
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})
        if self.raise_:
            raise RuntimeError("llm down")
        return self.output


def _bundle(stm: list[str], mtm: list[str], ltm: list[str]) -> ContextBundle:
    items: list[MemoryItem] = []
    for c in ltm:
        items.append(MemoryItem(id=str(uuid.uuid4()), content=c, tier=MemoryTier.LTM))
    for c in mtm:
        items.append(MemoryItem(id=str(uuid.uuid4()), content=c, tier=MemoryTier.MTM))
    for c in stm:
        items.append(MemoryItem(id=str(uuid.uuid4()), content=c, tier=MemoryTier.STM))
    return ContextBundle(
        items=items,
        messages=[],
        tokens_used=0,
        budget=TokenBudget(
            total=4000, stm_reserved=1000, mtm_reserved=1000,
            ltm_reserved=2000, response_reserved=200,
        ),
    )


# ---------------------------------------------------------------------------
# adapter.py — tests
# ---------------------------------------------------------------------------


def test_adapter_format_prompt_buckets_by_tier() -> None:
    a = adapter_mod.ContinuumAdapter(
        session=_FakeSession(retriever=None, stm=None),
        llm=_FakeLLM(),
    )
    ctx = _bundle(["stm1"], ["mtm1"], ["ltm1", "ltm2"])
    prompt = a.format_prompt("What is X?", ctx)
    assert "Long-term knowledge:\nltm1\nltm2" in prompt
    assert "Project summary:\nmtm1" in prompt
    assert "Recent conversation:\nstm1" in prompt
    assert "Question: What is X?\nAnswer:" in prompt


def test_adapter_format_prompt_omits_empty_sections() -> None:
    a = adapter_mod.ContinuumAdapter(
        session=_FakeSession(retriever=None, stm=None),
        llm=_FakeLLM(),
    )
    ctx = _bundle(stm=[], mtm=[], ltm=["only ltm"])
    prompt = a.format_prompt("Q?", ctx)
    assert "Long-term knowledge:" in prompt
    assert "Recent conversation:" not in prompt
    assert "Project summary:" not in prompt


def test_adapter_format_prompt_handles_none_ctx() -> None:
    a = adapter_mod.ContinuumAdapter(
        session=_FakeSession(retriever=None, stm=None),
        llm=_FakeLLM(),
    )
    prompt = a.format_prompt("Q?", None)
    # No tier sections; question + answer still rendered.
    assert "Long-term knowledge:" not in prompt
    assert "Question: Q?" in prompt


@pytest.mark.asyncio
async def test_process_conversation_dispatches_user_turns() -> None:
    stm = _FakeStm()
    session = _FakeSession(retriever=None, stm=stm)
    a = adapter_mod.ContinuumAdapter(session=session, llm=_FakeLLM())
    await a.process_conversation([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "how are you"},
    ])
    assert session.turns == ["hello", "how are you"]
    # Assistant turn was STM-appended without firing process_turn.
    assistant_rows = [i for i in stm.appended if i.metadata.get("role") == "assistant"]
    assert len(assistant_rows) == 1
    assert assistant_rows[0].content == "hi"


@pytest.mark.asyncio
async def test_process_conversation_skips_empty_messages() -> None:
    stm = _FakeStm()
    session = _FakeSession(retriever=None, stm=stm)
    a = adapter_mod.ContinuumAdapter(session=session, llm=_FakeLLM())
    await a.process_conversation([
        {"role": "user", "content": ""},
        {"role": "user", "content": "  "},
    ])
    assert session.turns == []


@pytest.mark.asyncio
async def test_process_conversation_swallows_turn_errors() -> None:
    stm = _FakeStm()
    session = _FakeSession(retriever=None, stm=stm)
    session.raise_on_turn = True
    a = adapter_mod.ContinuumAdapter(session=session, llm=_FakeLLM())
    # Must not raise.
    await a.process_conversation([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_answer_question_runs_retrieve_and_llm() -> None:
    ctx = _bundle(["recent stm"], [], ["durable fact"])
    retriever = _FakeRetriever(ctx)
    llm = _FakeLLM(output="the answer is X")
    session = _FakeSession(retriever=retriever, stm=_FakeStm())
    a = adapter_mod.ContinuumAdapter(session=session, llm=llm)

    answer = await a.answer_question("What is X?")
    assert answer == "the answer is X"
    assert retriever.calls == 1
    assert "durable fact" in llm.calls[0]["prompt"]
    assert llm.calls[0]["max_tokens"] == 100


@pytest.mark.asyncio
async def test_answer_question_degrades_on_retrieve_failure() -> None:
    retriever = _FakeRetriever(None)
    retriever.raise_ = True
    llm = _FakeLLM()
    a = adapter_mod.ContinuumAdapter(
        session=_FakeSession(retriever=retriever, stm=_FakeStm()), llm=llm,
    )
    answer = await a.answer_question("Q?")
    # Still produced an answer using a no-context prompt.
    assert answer == "the answer"
    assert "Long-term knowledge:" not in llm.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_answer_question_returns_error_string_on_llm_failure() -> None:
    llm = _FakeLLM(raise_=True)
    a = adapter_mod.ContinuumAdapter(
        session=_FakeSession(retriever=_FakeRetriever(None), stm=_FakeStm()),
        llm=llm,
    )
    answer = await a.answer_question("Q?")
    assert answer.startswith("[error:")


# ---------------------------------------------------------------------------
# run.py
# ---------------------------------------------------------------------------


class _FakeBenchmark:
    def __init__(self, *, dataset: str, adapter: Any, answerer: str) -> None:
        self.dataset = dataset
        self.adapter = adapter
        self.answerer = answerer
        self.run_called = False

    async def run(self) -> dict[str, Any]:
        self.run_called = True
        return {
            "accuracy": 0.732,
            "avg_tokens": 1234,
            "total_cost": 1.25,
        }


def _factory_calls() -> list[dict[str, Any]]:
    """Capture how the factory is invoked."""
    return []


@pytest.mark.asyncio
async def test_run_benchmark_wires_factory_and_persists(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []
    fake_bench: dict[str, _FakeBenchmark] = {}

    def factory(**kwargs: Any) -> _FakeBenchmark:
        captured.append(kwargs)
        fake_bench["b"] = _FakeBenchmark(**kwargs)
        return fake_bench["b"]

    adapter_obj = object()  # opaque — factory just passes it through

    results = await run_mod.run_benchmark(
        dataset="longmemeval-s",
        answerer="gpt-4o-mini",
        adapter=adapter_obj,
        benchmark_factory=factory,
        out_dir=tmp_path,
        results_filename="run.json",
    )
    assert fake_bench["b"].run_called
    assert captured[0]["dataset"] == "longmemeval-s"
    assert captured[0]["answerer"] == "gpt-4o-mini"
    assert captured[0]["adapter"] is adapter_obj
    assert results["accuracy"] == pytest.approx(0.732)

    written = (tmp_path / "run.json").read_text()
    loaded = json.loads(written)
    assert loaded["accuracy"] == pytest.approx(0.732)
    assert loaded["avg_tokens"] == 1234


@pytest.mark.asyncio
async def test_run_benchmark_default_filename_uses_clock(tmp_path: Path) -> None:
    import datetime as dt

    def factory(**kwargs: Any) -> _FakeBenchmark:
        return _FakeBenchmark(**kwargs)

    frozen = dt.datetime(2026, 5, 20, 12, 0, 0, tzinfo=dt.UTC)
    await run_mod.run_benchmark(
        dataset="d", answerer="a", adapter=object(),
        benchmark_factory=factory,
        out_dir=tmp_path,
        now=lambda: frozen,
    )
    files = list(tmp_path.glob("longmemeval_run_*.json"))
    assert len(files) == 1
    assert "20260520T120000Z" in files[0].name


@pytest.mark.asyncio
async def test_run_benchmark_normalises_object_results(tmp_path: Path) -> None:
    class _Obj:
        def __init__(self) -> None:
            self.accuracy = 0.5
            self.avg_tokens = 100
            self.total_cost = 0.1

        async def run(self) -> _Obj:
            return self

    def factory(**_kwargs: Any) -> _Obj:
        return _Obj()

    results = await run_mod.run_benchmark(
        dataset="d", answerer="a", adapter=object(),
        benchmark_factory=factory,
        out_dir=tmp_path,
        results_filename="r.json",
    )
    assert results["accuracy"] == 0.5
    assert results["avg_tokens"] == 100
