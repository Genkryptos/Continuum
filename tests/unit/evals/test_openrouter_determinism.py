"""
tests/unit/evals/test_openrouter_determinism.py
===============================================
OpenRouterLLM gained determinism knobs (provider pin + seed) because the
model routes across many backends, giving ~44% answer variance run-to-run
even at temperature 0 — which makes small ablations unmeasurable.

These tests assert the request payload carries the right keys; the network
call itself is never made.
"""

from __future__ import annotations

import pytest

from evals.longmemeval.bootstrap_ollama import OpenRouterLLM

pytestmark = pytest.mark.unit


def test_payload_includes_seed_and_pinned_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("OPENROUTER_PROVIDER", raising=False)
    llm = OpenRouterLLM(model="openai/gpt-oss-120b", provider_pin="DeepInfra", seed=0)
    p = llm._build_payload("hi", 16)
    assert p["seed"] == 0
    assert p["temperature"] == 0.0
    assert p["provider"]["order"] == ["DeepInfra"]
    assert p["provider"]["allow_fallbacks"] is False
    assert p["provider"]["require_parameters"] is True


def test_unpinned_no_seed_omits_routing_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("OPENROUTER_PROVIDER", raising=False)
    llm = OpenRouterLLM(model="x", seed=None)
    p = llm._build_payload("hi", 16)
    assert "provider" not in p  # default behaviour unchanged when unpinned
    assert "seed" not in p


def test_provider_pin_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_PROVIDER", "Fireworks")
    llm = OpenRouterLLM(model="x")  # no explicit pin → env
    assert llm.provider_pin == "Fireworks"
    assert llm._build_payload("hi", 16)["provider"]["order"] == ["Fireworks"]


def test_explicit_pin_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_PROVIDER", "Fireworks")
    llm = OpenRouterLLM(model="x", provider_pin="Together")
    assert llm.provider_pin == "Together"
