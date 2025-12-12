import os
import types
import unittest

from agent.agent import LocalAgent


class FakeResponse:
    def __init__(self):
        self.choices = [
            types.SimpleNamespace(message=types.SimpleNamespace(content="hello"))
        ]
        self.usage = types.SimpleNamespace(
            prompt_tokens=3, completion_tokens=2, total_tokens=5
        )


class FakeCompletions:
    def __init__(self, parent):
        self.parent = parent

    def create(self, model, messages, temperature):
        self.parent.calls.append({
            "model": model,
            "messages": messages,
            "temperature": temperature,
        })
        return FakeResponse()


class FakeOpenAIClient:
    def __init__(self):
        self.calls = []
        self.chat = types.SimpleNamespace(completions=FakeCompletions(self))
        self.api_key = "test-key"


class TestLocalAgent(unittest.TestCase):
    def setUp(self):
        # ensure openai branch does not fail on missing key
        os.environ.setdefault("OPEN_AI_KEY", "test-key")

    def tearDown(self):
        os.environ.pop("OPEN_AI_KEY", None)

    def test_openai_provider_handles_gpt_models(self):
        client = FakeOpenAIClient()
        agent = LocalAgent(model="gpt-4o-mini", openai_client=client)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        result = agent.call_model(messages)

        self.assertTrue(result["success"])
        self.assertEqual(result["response"], "hello")
        self.assertEqual(client.calls[0]["model"], "gpt-4o-mini")
        self.assertEqual(client.calls[0]["messages"], messages)
        self.assertGreater(result["tokens_used"]["total"], 0)


if __name__ == "__main__":
    unittest.main()
