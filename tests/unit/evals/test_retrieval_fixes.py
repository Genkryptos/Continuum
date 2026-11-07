"""
Fixes from the 2026-11-07 gap audit, tested through the composed path.

The previous chrono-sort fix was tested on the bare session-aware
retriever, while the pinned configuration wraps it (``--retriever
hybrid``) in reciprocal-rank fusion, which re-sorted its output and undid
the fix. So the tests here build adapters with the real
``make_adapter_factory`` and assert on the prompt the reader receives,
not on an intermediate bundle.
"""
from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pytest

from continuum.core.types import MemoryItem, Query, TokenBudget
from continuum.retrieval.bm25 import BM25Index
from continuum.retrieval.rrf import ReciprocalRankFusion
from evals.longmemeval.bootstrap_ollama import (
    _PRESUPPOSITION_CLAUSE,
    FlatHaystackStore,
    SessionAwareSemanticRetriever,
    _chrono_key,
    _effective_turn_budget,
    _is_prior_value_question,
    _parse_args,
    _versions_block,
    make_adapter_factory,
)
from evals.longmemeval.hybrid_scorer import HybridScorer

pytestmark = pytest.mark.unit


class _ScriptedEmbedder:
    """Known texts get a vector whose cosine to the query is a fixed score;
    anything else (the question) is the query direction."""

    model_name = "scripted"

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            s = self.scores.get(t)
            if s is None:
                rows.append([1.0, 0.0])
            else:
                rows.append([s, float(np.sqrt(max(0.0, 1.0 - s * s)))])
        return np.asarray(rows, dtype=np.float64)


class _CaptureLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(self, *, prompt: str, max_tokens: int) -> str:
        self.prompts.append(prompt)
        return "ok"


def _budget() -> TokenBudget:
    return TokenBudget(
        total=10_000, stm_reserved=10_000, mtm_reserved=0,
        ltm_reserved=0, response_reserved=0,
    )


# Haystack listed OUT of date order — as 211 of the 500 LongMemEval-S rows
# are. Session "late" comes first in the haystack but was said last.
_TURNS = [
    # (content, session, date, cosine)
    ("late turn about the bike", "late", "2023-05-20T10:00:00", 0.90),
    ("early turn about the bike", "early", "2023-03-01T09:00:00", 0.80),
    ("middle turn about the bike", "mid", "2023-04-10T09:00:00", 0.85),
    ("unrelated chatter", "noise", "2023-04-11T09:00:00", 0.10),
]


def _messages() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": c, "session_id": sid, "date": d, "haystack_index": i}
        for i, (c, sid, d, _s) in enumerate(_TURNS)
    ]


async def _prompt_for(question: str, *, qtype: str, **factory_kwargs: Any) -> str:
    llm = _CaptureLLM()
    kwargs: dict[str, Any] = {
        "top_k": 3,
        "direct_answer": True,
        "direct_max_context_chars": 10_000,
        "session_aware_retrieval": True,
        "session_top_k": 4,
        "turns_per_session": 1,
        "turn_rerank": "floor",
        "retriever_kind": "hybrid",
        "chrono_sort": True,
        "score_weights": {"relevance": 1.0, "importance": 0.0, "recency": 0.0, "confidence": 0.0},
    }
    kwargs.update(factory_kwargs)
    factory = make_adapter_factory(
        llm, _ScriptedEmbedder({c: s for c, _sid, _d, s in _TURNS}), **kwargs,
    )
    adapter = factory()
    await adapter.process_conversation(_messages())
    adapter.dataset_question_type = qtype
    adapter.dataset_question_date = "2023/05/30 (Tue) 10:00"
    await adapter.answer_question(question)
    return llm.prompts[-1]


# ── chronological presentation ─────────────────────────────────────────────


async def test_pinned_path_presents_turns_in_date_order_not_haystack_order() -> None:
    prompt = await _prompt_for("What did I do with the bike?", qtype="single-session-user")
    early = prompt.index("early turn")
    middle = prompt.index("middle turn")
    late = prompt.index("late turn")
    assert early < middle < late


async def test_pinned_path_fuses_inside_the_session_aware_retriever() -> None:
    factory = make_adapter_factory(
        _CaptureLLM(), _ScriptedEmbedder({}), session_aware_retrieval=True,
        retriever_kind="hybrid", direct_answer=True, chrono_sort=True,
    )
    retriever = factory().session.retriever
    # Not wrapped: a HybridRetriever would re-sort the selection by fused rank.
    assert isinstance(retriever, SessionAwareSemanticRetriever)
    assert retriever.lexical is True


def test_plain_hybrid_child_does_not_feed_chronological_order_to_rrf() -> None:
    factory = make_adapter_factory(
        _CaptureLLM(), _ScriptedEmbedder({}), retriever_kind="hybrid",
        direct_answer=True, chrono_sort=True,
    )
    adapter = factory()
    hybrid = adapter.session.retriever
    assert isinstance(hybrid, ReciprocalRankFusion)
    # RRF reads child position as relevance rank; the child must stay in
    # relevance order, and the adapter applies time order after fusion.
    assert hybrid._retrievers[0].chrono_sort is False
    assert adapter._chrono_sort is True


def test_chrono_key_orders_by_date_then_turn_position() -> None:
    a = MemoryItem(content="x", metadata={"date": "2023-05-01T10:00:00", "haystack_index": 0})
    b = MemoryItem(content="y", metadata={"date": "2023-03-01T10:00:00", "haystack_index": 9})
    c = MemoryItem(content="z", metadata={"date": "2023-03-01T10:00:00", "haystack_index": 10})
    assert sorted([a, c, b], key=lambda it: _chrono_key(it)) == [b, c, a]


# ── turn selection ─────────────────────────────────────────────────────────


_FLOOR_SCORES = {
    "a0": 0.99, "a1": 0.98, "a2": 0.97, "a3": 0.96,
    "b0": 0.40, "b1": 0.10,
    "c0": 0.30, "c1": 0.05,
}


def _floor_bundle(turn_rerank: str, max_items: int = 4, lexical: bool = False):
    store = FlatHaystackStore()
    for name in _FLOOR_SCORES:
        store.items.append(MemoryItem(id=name, content=name, metadata={"session_id": name[0]}))
    r = SessionAwareSemanticRetriever(
        store=store, embedder=_ScriptedEmbedder(_FLOOR_SCORES), top_k=max_items,
        session_top_k=3, turns_per_session=1, max_items=max_items,
        turn_rerank=turn_rerank, lexical=lexical,
        score_weights={"relevance": 1.0, "importance": 0.0, "recency": 0.0, "confidence": 0.0},
    )
    r._refresh_index()
    return [it.content for it in r._bundle_topk(Query(text="QUERY"), _budget()).items]


def test_floor_guarantees_every_selected_session_then_fills_globally() -> None:
    picked = _floor_bundle("floor")
    # One per session (a0, b0, c0), then the strongest remaining (a1).
    assert set(picked) == {"a0", "b0", "c0", "a1"}


def test_global_lets_one_session_take_the_whole_budget() -> None:
    assert set(_floor_bundle("global")) == {"a0", "a1", "a2", "a3"}


def test_quota_leaves_budget_unused() -> None:
    # 3 sessions x 1 turn = 3, although the budget is 4.
    assert len(_floor_bundle("quota")) == 3


def test_unknown_turn_rerank_is_rejected() -> None:
    with pytest.raises(ValueError):
        _floor_bundle("bogus")


def test_scorer_weights_are_honoured_under_session_aware_retrieval() -> None:
    # Two sessions: an old, more relevant turn and a recent, less relevant one.
    store = FlatHaystackStore()
    old = MemoryItem(content="old", metadata={"session_id": "o"})
    new = MemoryItem(content="new", metadata={"session_id": "n"})
    import datetime as dt

    old.created_at = dt.datetime(2023, 1, 1)
    new.created_at = dt.datetime(2023, 5, 1)
    store.items.extend([old, new])

    def top(weights: dict[str, float]) -> str:
        r = SessionAwareSemanticRetriever(
            store=store, embedder=_ScriptedEmbedder({"old": 0.9, "new": 0.6}),
            top_k=1, session_top_k=1, turns_per_session=1, max_items=1,
            score_weights=weights, tau_hours=168.0,
        )
        r._refresh_index()
        return r._bundle_topk(Query(text="QUERY"), _budget()).items[0].content

    assert top({"relevance": 1.0, "importance": 0.0, "recency": 0.0, "confidence": 0.0}) == "old"
    # Before the fix the override used raw cosine, so this was also "old".
    assert top({"relevance": 0.3, "importance": 0.0, "recency": 0.7, "confidence": 0.0}) == "new"


def test_lexical_fusion_admits_a_rare_token_match_cosine_misses() -> None:
    scores = {"ordinary words here": 0.9, "Roscioli booking": 0.05, "more ordinary text": 0.8}
    store = FlatHaystackStore()
    for i, text in enumerate(scores):
        store.items.append(MemoryItem(content=text, metadata={"session_id": f"s{i}"}))

    def picked(lexical: bool) -> list[str]:
        r = SessionAwareSemanticRetriever(
            store=store, embedder=_ScriptedEmbedder(scores), top_k=2,
            session_top_k=2, turns_per_session=1, max_items=2, lexical=lexical,
        )
        r._refresh_index()
        return [it.content for it in r._bundle_topk(Query(text="Roscioli"), _budget()).items]

    assert "Roscioli booking" not in picked(lexical=False)
    assert "Roscioli booking" in picked(lexical=True)


def test_bm25_score_all_covers_every_item_in_order() -> None:
    items = [MemoryItem(content="apple pie"), MemoryItem(content="banana"), MemoryItem(content="apple")]
    scores = BM25Index(items).score_all("apple")
    assert len(scores) == 3
    assert scores[1] == 0.0 and scores[0] > 0 and scores[2] > 0


# ── prompt routing ─────────────────────────────────────────────────────────


async def test_windowed_count_question_gets_dated_aggregate_prompt() -> None:
    prompt = await _prompt_for(
        "How many bike repairs did I do in April?", qtype="multi-session",
    )
    assert "combine information" in prompt
    assert "[2023-04-10" in prompt  # turns are date-stamped
    assert "Today's date is 2023/05/30" in prompt


async def test_date_delta_question_still_gets_temporal_prompt() -> None:
    prompt = await _prompt_for(
        "How many days ago did I fix the bike?", qtype="temporal-reasoning",
    )
    assert "compute the difference" in prompt


async def test_prior_value_ku_question_is_not_told_latest_wins() -> None:
    prompt = await _prompt_for(
        "Where did I initially keep the bike?", qtype="knowledge-update",
    )
    assert "EARLIER state" in prompt
    assert "never answer with a value the user has since changed" not in prompt


async def test_current_value_ku_question_keeps_latest_wins() -> None:
    prompt = await _prompt_for("Where do I keep the bike now?", qtype="knowledge-update")
    assert "LATEST date" in prompt


async def test_preference_question_is_not_pulled_into_aggregate_prompt() -> None:
    prompt = await _prompt_for(
        "Can you list some bike accessories I might like?",
        qtype="single-session-preference", preference_conditioning=True,
    )
    assert "preference this user has stated" in prompt


async def test_default_prompt_carries_presupposition_check() -> None:
    prompt = await _prompt_for("What colour is my bike?", qtype="single-session-user")
    assert _PRESUPPOSITION_CLAUSE in prompt


@pytest.mark.parametrize(
    ("question", "prior"),
    [
        ("Where do I initially keep my old sneakers?", True),
        ("What was my previous personal best time?", True),
        ("How many coins did I collect in the first three months of collection?", True),
        ("Where do I keep my old sneakers now?", False),
        ("What is my current personal best?", False),
    ],
)
def test_prior_value_detector(question: str, prior: bool) -> None:
    assert _is_prior_value_question(question) is prior


# ── scorer ─────────────────────────────────────────────────────────────────


def test_second_person_answer_matches_first_person_gold() -> None:
    # 07741c45 — the one clear scorer false negative in the 500-row run.
    result = HybridScorer(mode="rule").score_sync(
        "Where do I keep my sneakers?",
        "in a shoe rack in my closet",
        "In a shoe rack in your closet",
    )
    assert result.correct and result.method == "normalized"


def test_person_swap_does_not_make_different_answers_match() -> None:
    from evals.longmemeval.hybrid_scorer import _first_person, normalize_text

    a = _first_person(normalize_text("in my closet"))
    b = _first_person(normalize_text("in your garage"))
    assert a != b


# ── CLI and provenance ─────────────────────────────────────────────────────


def test_rerank_to_defaults_to_keeping_top_k() -> None:
    args = _parse_args(["--top-k", "24", "--rerank"])
    assert args.rerank_to == 0
    args.session_aware_retrieval = True
    args.turn_rerank = "floor"
    assert _effective_turn_budget(args) == 24


def test_quota_budget_reports_the_real_cap() -> None:
    args = _parse_args(["--top-k", "24", "--session-top-k", "8", "--turns-per-session", "2"])
    args.turn_rerank = "quota"
    assert _effective_turn_budget(args) == 16


def test_versions_block_derives_non_comparable_from_deviations() -> None:
    args = _parse_args([
        "--provider", "meta", "--top-k", "24", "--reasoner", "direct",
        "--retriever", "hybrid", "--session-aware-retrieval",
    ])
    llm = argparse.Namespace(model="llama")
    block = _versions_block(args, "muse-wonder-1.0", llm, _ScriptedEmbedder({}))
    assert block["non_comparable"] is True
    joined = " ".join(block["deviations"])
    assert "top_k=24" in joined and "answerer=meta/muse-wonder-1.0" in joined
    assert block["judge_prompt_sha256"].startswith("60222fd975")


# ── round 2: preference routing, user-turns-first, paraphrase wording ──────


async def test_factual_question_with_preference_wording_keeps_factual_prompt() -> None:
    # "for my" is a preference wording term; the dataset label says otherwise.
    prompt = await _prompt_for(
        "What did I buy for my bike?", qtype="single-session-user",
        preference_conditioning=True,
    )
    assert "preference this user has stated" not in prompt
    assert _PRESUPPOSITION_CLAUSE in prompt


def test_preference_wording_applies_only_without_a_label() -> None:
    from evals.longmemeval.bootstrap_ollama import _is_preference

    q = "Can you recommend resources for my trip?"
    assert _is_preference("", q) is True
    assert _is_preference("multi-session", q) is False
    assert _is_preference("single-session-preference", "anything") is True


def _role_bundle(question: str, user_first: bool) -> list[str]:
    turns = {
        # Assistant replies outscore the user turn that carries the answer.
        "a-reply-1": ("assistant", 0.95), "a-reply-2": ("assistant", 0.93),
        "u-evidence": ("user", 0.60), "u-other": ("user", 0.20),
    }
    store = FlatHaystackStore()
    for text, (role, _s) in turns.items():
        store.items.append(MemoryItem(content=text, metadata={"session_id": "s", "role": role}))
    r = SessionAwareSemanticRetriever(
        store=store, embedder=_ScriptedEmbedder({t: s for t, (_r, s) in turns.items()}),
        top_k=2, session_top_k=1, turns_per_session=2, max_items=2,
        turn_rerank="floor", user_first=user_first,
    )
    r._refresh_index()
    return [it.content for it in r._bundle_topk(Query(text=question), _budget()).items]


def test_user_turns_first_ranks_user_turns_ahead_of_assistant_replies() -> None:
    assert set(_role_bundle("What bike did I buy?", user_first=False)) == {"a-reply-1", "a-reply-2"}
    assert set(_role_bundle("What bike did I buy?", user_first=True)) == {"u-evidence", "u-other"}


def test_user_turns_first_steps_aside_for_questions_about_the_assistant() -> None:
    picked = _role_bundle("Remind me what you suggested for the bike?", user_first=True)
    assert set(picked) == {"a-reply-1", "a-reply-2"}


@pytest.mark.parametrize(
    ("question", "about_assistant"),
    [
        ("Can you remind me what you suggested for my garden?", True),
        ("I'm looking back at our previous conversation about tea.", True),
        ("What was the move you made after 27. Kg2?", True),
        ("What did I buy for my sister's birthday?", False),
        ("How many days ago did you know I adopted a cat?", False),
    ],
)
def test_asks_about_assistant_detector(question: str, about_assistant: bool) -> None:
    from evals.longmemeval.bootstrap_ollama import _asks_about_assistant

    assert _asks_about_assistant(question) is about_assistant


def test_presupposition_clause_says_paraphrases_count_as_mentioned() -> None:
    assert "counts as mentioned" in _PRESUPPOSITION_CLAUSE
