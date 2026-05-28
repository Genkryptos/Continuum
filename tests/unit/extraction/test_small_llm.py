"""
tests/unit/extraction/test_small_llm.py
=======================================
Unit tests for :class:`continuum.extraction.small_llm.SmallLLM`.

``requests.post`` is monkeypatched in every test — no Ollama, no network.
Each test uses a per-test sqlite cache under ``tmp_path`` so cache state
never leaks across cases.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from continuum.extraction.small_llm import DEFAULT_MODEL, SmallLLM

pytestmark = pytest.mark.unit


# ── helpers ─────────────────────────────────────────────────────────────────


def _resp(payload: Any, *, status: int = 200) -> MagicMock:
    """Build a stand-in for ``requests.Response``."""
    r = MagicMock()
    r.status_code = status
    if isinstance(payload, str):
        r.json.side_effect = ValueError("not json")
        r.text = payload
    else:
        r.json.return_value = payload
    return r


@pytest.fixture
def cache_path(tmp_path):
    return tmp_path / "small_llm.db"


@pytest.fixture
def llm(cache_path):
    return SmallLLM(model="m1", url="http://x", cache_path=cache_path)


# ── cache hit / miss ────────────────────────────────────────────────────────


def test_cache_miss_then_hit(llm, monkeypatch):
    post = MagicMock(return_value=_resp({"response": "Boston"}))
    monkeypatch.setattr("continuum.extraction.small_llm.requests.post", post)

    out1 = llm.span_select("where?", "I live in Boston.")
    out2 = llm.span_select("where?", "I live in Boston.")

    assert out1 == "Boston"
    assert out2 == "Boston"
    assert post.call_count == 1  # second call served from cache


def test_cache_key_separates_methods(llm, monkeypatch):
    post = MagicMock(
        side_effect=[
            _resp({"response": "Boston"}),
            _resp({"response": '{"supported": true, "confidence": 0.9}'}),
        ]
    )
    monkeypatch.setattr("continuum.extraction.small_llm.requests.post", post)

    span = llm.span_select("q", "Boston is nice.", cache_key="same")
    ok, conf = llm.verify_claim("c", "Boston is nice.", cache_key="same")

    assert span == "Boston"
    assert (ok, conf) == (True, 0.9)
    assert post.call_count == 2  # different methods → different keys


def test_explicit_cache_key_reused(llm, monkeypatch):
    post = MagicMock(return_value=_resp({"response": "X"}))
    monkeypatch.setattr("continuum.extraction.small_llm.requests.post", post)

    llm.span_select("q1", "p1", cache_key="k")
    llm.span_select("q2-different", "p2-different", cache_key="k")

    assert post.call_count == 1  # cache_key collapses both calls


# ── env override ────────────────────────────────────────────────────────────


def test_env_overrides_model_and_url(cache_path, monkeypatch):
    monkeypatch.setenv("CONTINUUM_SMALL_LLM_MODEL", "tiny-test")
    monkeypatch.setenv("CONTINUUM_SMALL_LLM_URL", "http://env-host:9999")
    post = MagicMock(return_value=_resp({"response": "ok"}))
    monkeypatch.setattr("continuum.extraction.small_llm.requests.post", post)

    llm = SmallLLM(cache_path=cache_path)
    assert llm.model == "tiny-test"
    assert llm.url == "http://env-host:9999"

    llm.span_select("q", "p")
    sent_url = post.call_args.args[0] if post.call_args.args else post.call_args.kwargs["url"]
    sent_json = post.call_args.kwargs["json"]
    assert sent_url == "http://env-host:9999/api/generate"
    assert sent_json["model"] == "tiny-test"


def test_constructor_overrides_env(cache_path, monkeypatch):
    monkeypatch.setenv("CONTINUUM_SMALL_LLM_MODEL", "from-env")
    llm = SmallLLM(model="from-ctor", cache_path=cache_path)
    assert llm.model == "from-ctor"


def test_default_model_when_no_env(cache_path, monkeypatch):
    monkeypatch.delenv("CONTINUUM_SMALL_LLM_MODEL", raising=False)
    llm = SmallLLM(cache_path=cache_path)
    assert llm.model == DEFAULT_MODEL


# ── malformed response handling ─────────────────────────────────────────────


def test_non_200_returns_safe_default_and_does_not_cache(llm, monkeypatch):
    post = MagicMock(
        side_effect=[
            _resp({"error": "model not loaded"}, status=500),
            _resp({"response": "Boston"}),
        ]
    )
    monkeypatch.setattr("continuum.extraction.small_llm.requests.post", post)

    assert llm.span_select("q", "p") == ""  # safe default on failure
    assert llm.span_select("q", "p") == "Boston"  # retried — not cached
    assert post.call_count == 2


def test_non_json_body_returns_safe_default(llm, monkeypatch):
    post = MagicMock(return_value=_resp("not-json-at-all"))
    monkeypatch.setattr("continuum.extraction.small_llm.requests.post", post)
    assert llm.span_select("q", "p") == ""


def test_missing_response_field_returns_safe_default(llm, monkeypatch):
    post = MagicMock(return_value=_resp({"other": "field"}))
    monkeypatch.setattr("continuum.extraction.small_llm.requests.post", post)
    assert llm.span_select("q", "p") == ""


def test_request_exception_returns_safe_default(llm, monkeypatch):
    import requests as _r

    post = MagicMock(side_effect=_r.ConnectionError("refused"))
    monkeypatch.setattr("continuum.extraction.small_llm.requests.post", post)

    assert llm.span_select("q", "p") == ""
    assert llm.verify_claim("c", "e") == (False, 0.0)
    assert llm.classify_intent("q?") == "lookup"


def test_verify_claim_parses_json_with_fences(llm, monkeypatch):
    fenced = '```json\n{"supported": false, "confidence": 0.3}\n```'
    monkeypatch.setattr(
        "continuum.extraction.small_llm.requests.post",
        MagicMock(return_value=_resp({"response": fenced})),
    )
    assert llm.verify_claim("c", "e") == (False, 0.3)


def test_verify_claim_garbage_returns_default(llm, monkeypatch):
    monkeypatch.setattr(
        "continuum.extraction.small_llm.requests.post",
        MagicMock(return_value=_resp({"response": "definitely not json"})),
    )
    assert llm.verify_claim("c", "e") == (False, 0.0)


def test_verify_claim_clamps_confidence(llm, monkeypatch):
    monkeypatch.setattr(
        "continuum.extraction.small_llm.requests.post",
        MagicMock(
            return_value=_resp(
                {"response": json.dumps({"supported": True, "confidence": 5.0})}
            )
        ),
    )
    ok, conf = llm.verify_claim("c", "e")
    assert ok is True
    assert conf == 1.0


def test_classify_intent_valid_label(llm, monkeypatch):
    monkeypatch.setattr(
        "continuum.extraction.small_llm.requests.post",
        MagicMock(return_value=_resp({"response": "temporal"})),
    )
    assert llm.classify_intent("when did I move?") == "temporal"


def test_classify_intent_strips_punctuation(llm, monkeypatch):
    monkeypatch.setattr(
        "continuum.extraction.small_llm.requests.post",
        MagicMock(return_value=_resp({"response": "Preference."})),
    )
    assert llm.classify_intent("do I prefer X?") == "preference"


def test_classify_intent_unknown_label_falls_back(llm, monkeypatch):
    monkeypatch.setattr(
        "continuum.extraction.small_llm.requests.post",
        MagicMock(return_value=_resp({"response": "something-else"})),
    )
    assert llm.classify_intent("?") == "lookup"


# ── cache key derivation ────────────────────────────────────────────────────


def test_normalized_inputs_collide(llm, monkeypatch):
    """Whitespace-only differences in inputs should hit the same cache row."""
    post = MagicMock(return_value=_resp({"response": "Boston"}))
    monkeypatch.setattr("continuum.extraction.small_llm.requests.post", post)

    llm.span_select("where?", "I live in Boston.")
    llm.span_select("  where?  ", "  I live in Boston.  ")
    assert post.call_count == 1


def test_different_models_have_separate_caches(cache_path, monkeypatch):
    post = MagicMock(return_value=_resp({"response": "Boston"}))
    monkeypatch.setattr("continuum.extraction.small_llm.requests.post", post)

    a = SmallLLM(model="a", url="http://x", cache_path=cache_path)
    b = SmallLLM(model="b", url="http://x", cache_path=cache_path)
    a.span_select("q", "p")
    b.span_select("q", "p")
    assert post.call_count == 2
