"""
tests/unit/evals/test_bedrock_provider.py
=========================================
Unit coverage for Amazon Bedrock API-key wiring in the LongMemEval runner.

The tests avoid real network calls and real credentials; they verify the
provider is accepted by argparse and BedrockLLM uses the native Boto3
Converse API shape shown in AWS's examples.
"""

from __future__ import annotations

from typing import Any

import pytest

from evals.longmemeval import bootstrap_ollama as boot

pytestmark = pytest.mark.unit


class _FakeConverseClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.response


class _FakeResponses:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


class _FakeOpenAIClient:
    def __init__(self, response: Any) -> None:
        self.responses = _FakeResponses(response)


class _FakeOpenAIResponse:
    output_text = "oss hello"


def test_parse_args_accepts_bedrock_provider() -> None:
    args = boot._parse_args(
        [
            "--provider",
            "bedrock",
            "--model",
            "anthropic.claude-3-haiku-20240307-v1:0",
        ]
    )

    assert args.provider == "bedrock"
    assert args.model == "anthropic.claude-3-haiku-20240307-v1:0"


async def test_bedrock_llm_uses_boto3_converse_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeConverseClient(
        {
            "output": {
                "message": {
                    "content": [{"text": "bedrock hello"}],
                },
            },
            "usage": {
                "inputTokens": 7,
                "outputTokens": 2,
            },
        }
    )
    created: dict[str, Any] = {}

    def fake_boto3_client(
        region: str,
        endpoint_url: str | None = None,
    ) -> _FakeConverseClient:
        created["region"] = region
        created["endpoint_url"] = endpoint_url
        return fake_client

    async def fail_adaptive_complete(**_kwargs: Any) -> str:
        raise AssertionError("BedrockLLM should use boto3 converse")

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-api-key-test")
    monkeypatch.setattr(boot, "_bedrock_boto3_client", fake_boto3_client, raising=False)
    monkeypatch.setattr(boot, "_adaptive_complete", fail_adaptive_complete)

    llm = boot.BedrockLLM(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        region="us-east-1",
        temperature=0.25,
        rpm=17,
    )
    try:
        reply = await llm.complete(prompt="hello", max_tokens=12)
    finally:
        await llm.aclose()

    assert reply == "bedrock hello"
    assert created == {"region": "us-east-1", "endpoint_url": None}
    assert fake_client.calls == [
        {
            "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": "hello"}],
                },
            ],
            "inferenceConfig": {
                "maxTokens": 12,
                "temperature": 0.25,
            },
        }
    ]
    assert llm._throttle.interval == pytest.approx(60.0 / 17.0)


async def test_bedrock_llm_uses_optional_endpoint_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeConverseClient(
        {
            "output": {
                "message": {
                    "content": [{"text": "ok"}],
                },
            },
        }
    )
    created: dict[str, Any] = {}

    def fake_boto3_client(
        region: str,
        endpoint_url: str | None = None,
    ) -> _FakeConverseClient:
        created["region"] = region
        created["endpoint_url"] = endpoint_url
        return fake_client

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-api-key-test")
    monkeypatch.setattr(boot, "_bedrock_boto3_client", fake_boto3_client, raising=False)

    llm = boot.BedrockLLM(
        model="meta.llama3-8b-instruct-v1:0",
        region="us-west-2",
        base_url="https://bedrock-runtime.us-west-2.amazonaws.com",
    )
    try:
        reply = await llm.complete(prompt="hello", max_tokens=4)
    finally:
        await llm.aclose()

    assert reply == "ok"
    assert created == {
        "region": "us-west-2",
        "endpoint_url": "https://bedrock-runtime.us-west-2.amazonaws.com",
    }


async def test_bedrock_gpt_oss_uses_openai_responses_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeOpenAIClient(_FakeOpenAIResponse())
    created: dict[str, Any] = {}

    def fail_boto3_client(
        region: str,
        endpoint_url: str | None = None,
    ) -> _FakeConverseClient:
        raise AssertionError("gpt-oss Bedrock models should use OpenAI Responses")

    def fake_responses_client(
        region: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> _FakeOpenAIClient:
        created["region"] = region
        created["base_url"] = base_url
        created["api_key"] = api_key
        return fake_client

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-api-key-test")
    monkeypatch.setattr(boot, "_bedrock_boto3_client", fail_boto3_client)
    monkeypatch.setattr(
        boot,
        "_bedrock_openai_responses_client",
        fake_responses_client,
        raising=False,
    )

    llm = boot.BedrockLLM(
        model="openai.gpt-oss-120b",
        region="us-east-1",
        temperature=0.2,
        rpm=17,
    )
    try:
        reply = await llm.complete(prompt="hello", max_tokens=12)
    finally:
        await llm.aclose()

    assert reply == "oss hello"
    assert created == {
        "region": "us-east-1",
        "base_url": None,
        "api_key": None,
    }
    assert fake_client.responses.calls == [
        {
            "model": "openai.gpt-oss-120b",
            "input": [{"role": "user", "content": "hello"}],
            "max_output_tokens": 12,
            "temperature": 0.2,
        }
    ]
    assert llm._throttle.interval == pytest.approx(60.0 / 17.0)
