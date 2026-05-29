"""
tests/reasoning/conftest.py
===========================
Reusable fakes for :class:`IterativeReasoner` tests. None of these
touch disk, network, or any real LLM — they record calls so tests
can assert on call counts, cache-key shapes, and ordering.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest


# ── Fake LLMs ──────────────────────────────────────────────────────────────


class FakeComposerLLM:
    """
    Stand-in for the gpt-4o-mini-style composer. Returns scripted
    replies from a queue; each call appends to ``self.calls`` so
    tests can assert order and prompt content.
    """

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies: list[str] = list(replies or [])
        self.calls: list[dict[str, Any]] = []

    async def complete(self, prompt: str, max_tokens: int) -> str:
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})
        if not self.replies:
            return ""
        return self.replies.pop(0)


class FakeSmallLLM:
    """
    Records every classify_intent / span_select call and returns
    scripted replies. Mirrors the public surface of
    :class:`continuum.extraction.small_llm.SmallLLM` the reasoner
    actually uses.
    """

    def __init__(
        self,
        intent_replies: list[str] | None = None,
        span_replies: list[str] | None = None,
    ) -> None:
        self.intent_replies = list(intent_replies or [])
        self.span_replies = list(span_replies or [])
        self.intent_calls: list[dict[str, Any]] = []
        self.span_calls: list[dict[str, Any]] = []

    async def classify_intent(self, question: str, cache_key: str | None = None) -> str:
        self.intent_calls.append({"question": question, "cache_key": cache_key})
        if not self.intent_replies:
            return "lookup"
        return self.intent_replies.pop(0)

    async def span_select(
        self, question: str, passage: str, cache_key: str | None = None,
    ) -> str:
        self.span_calls.append({
            "question": question, "passage": passage, "cache_key": cache_key,
        })
        if not self.span_replies:
            return ""
        return self.span_replies.pop(0)


# ── Fake retriever + ctx ───────────────────────────────────────────────────


@dataclass
class _FakeCtx:
    """Object the reasoner passes into extract_*_fn callables."""
    payload: Any = None


class FakeRetriever:
    """
    Returns a scripted ctx per call. Maintains a query log so tests
    can assert rewrites flow through. If ``ctx_for_query`` is given,
    look up by query string; otherwise return ``default_ctx``.
    """

    def __init__(
        self,
        default_ctx: Any | None = None,
        ctx_for_query: dict[str, Any] | None = None,
    ) -> None:
        self._default_ctx = default_ctx if default_ctx is not None else _FakeCtx()
        self._by_query = dict(ctx_for_query or {})
        self.queries: list[str] = []

    async def retrieve(self, query: str) -> Any:
        self.queries.append(query)
        return self._by_query.get(query, self._default_ctx)


# ── Fake "candidate" + "claim" + "verifier result" ─────────────────────────


@dataclass
class FakeCandidate:
    """Minimal candidate shape — enough for verify/filter/packet."""
    value: str
    claim: str = ""
    confidence: float = 0.5
    source_role: str = "user"


@dataclass
class FakeClaim:
    text: str
    source_role: str = "user"
    confidence: float = 0.5


@dataclass
class FakeVerifierResult:
    candidate: FakeCandidate
    verdict: str  # "PASS" or "FAIL"
    reason: str = ""


@dataclass
class FakePacket:
    question: str
    claims: list[Any] = field(default_factory=list)
    excluded_noise_count: int = 0


# ── Pytest fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def fake_composer() -> FakeComposerLLM:
    """Default composer with two scripted replies (decompose + synthesis)."""
    return FakeComposerLLM(replies=[
        "1. sub-q one\n2. sub-q two",  # decompose
        "synthesised final answer",     # synthesis
    ])


@pytest.fixture
def fake_small() -> FakeSmallLLM:
    return FakeSmallLLM()


@pytest.fixture
def fake_retriever() -> FakeRetriever:
    return FakeRetriever()


# ── Reasoner factory shared across tests ──────────────────────────────────


def _identity_extract(ctx: Any) -> list[Any]:
    """Default extractor: read .candidates / .claims off the ctx payload."""
    return list(getattr(ctx, "payload", None) or [])


def _passthrough_verify(question: str, candidates: list[Any], **kw: Any) -> list[Any]:
    """Default verifier: mark every candidate PASS."""
    return [FakeVerifierResult(candidate=c, verdict="PASS") for c in candidates]


def _filter_passing(results: list[Any]) -> list[Any]:
    return [r.candidate for r in results if getattr(r, "verdict", "") == "PASS"]


def _build_packet(question: str, verified: list[Any], **kw: Any) -> FakePacket:
    return FakePacket(question=question, claims=list(verified))


def _render_packet(packet: FakePacket) -> str:
    return f"PACKET:{len(packet.claims)}"


def _build_final_prompt(question: str, **kw: Any) -> str:
    return f"FINAL:{question}"


@dataclass
class _StubIntent:
    matched: bool
    answer_shape: str = "lookup"


def _matched_intent(question: str) -> _StubIntent:
    return _StubIntent(matched=True, answer_shape="lookup")


def _unmatched_intent(question: str) -> _StubIntent:
    return _StubIntent(matched=False, answer_shape="unknown")


class _StubMode:
    name = "FACT_LOOKUP"


def _route_to(mode: Any) -> Callable[..., Any]:
    def _route(question: str, **kw: Any) -> Any:
        return mode
    return _route


def _decompose_two(question: str) -> tuple[str, Callable[..., list[str]]]:
    return ("decompose-prompt", lambda reply, original: ["sub-q one", "sub-q two"])


def _decompose_empty(question: str) -> tuple[str, Callable[..., list[str]]]:
    return ("decompose-prompt", lambda reply, original: [])


__all__ = [
    "FakeCandidate",
    "FakeClaim",
    "FakeComposerLLM",
    "FakeCtx",
    "FakePacket",
    "FakeRetriever",
    "FakeSmallLLM",
    "FakeVerifierResult",
    "_StubIntent",
    "_StubMode",
    "_build_final_prompt",
    "_build_packet",
    "_decompose_empty",
    "_decompose_two",
    "_filter_passing",
    "_identity_extract",
    "_matched_intent",
    "_passthrough_verify",
    "_render_packet",
    "_route_to",
    "_unmatched_intent",
]

FakeCtx = _FakeCtx  # alias for direct construction in tests
