import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# Provide lightweight stubs so importing the module does not require external
# dependencies at test time. Tests patch these stubs as needed.
class _DummyEncoding:
    def encode(self, text):  # pragma: no cover - trivial stub
        return [0 for _ in str(text)]


anthropic_stub = types.SimpleNamespace(Anthropic=lambda *args, **kwargs: None)
tiktoken_stub = types.SimpleNamespace(
    encoding_for_model=lambda *_args, **_kwargs: _DummyEncoding(),
    get_encoding=lambda *_args, **_kwargs: _DummyEncoding(),
)
litellm_stub = types.SimpleNamespace(token_counter=lambda **_kwargs: 0)

sys.modules.setdefault("anthropic", anthropic_stub)
sys.modules.setdefault("tiktoken", tiktoken_stub)
sys.modules.setdefault("litellm", litellm_stub)

import helper.tknCounter as counter_mod
from helper.tknCounter import TknCounter


class TestTknCounter(unittest.TestCase):
    def test_encoding_fallback(self):
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2]

        counter = TknCounter()
        with patch("tiktoken.encoding_for_model", side_effect=ValueError):
            with patch("tiktoken.get_encoding", return_value=mock_encoding) as get_encoding:
                encoding = counter._encoding_for_model("unknown-model")

        get_encoding.assert_called_once_with(counter.default_encoding)
        self.assertIs(encoding, mock_encoding)

    def test_gpt_rules_selection(self):
        counter = TknCounter()

        self.assertEqual(counter._gpt_rules_for_model("gpt-3.5-turbo-0613"), (3, 1))
        self.assertEqual(counter._gpt_rules_for_model("gpt-4-unknown"), (3, 1))
        self.assertEqual(counter._gpt_rules_for_model("gpt-3.5-turbo-0301"), (4, -1))

    def test_gpt_counting_uses_rules(self):
        counter = TknCounter()
        counter._encoding_for_model = MagicMock()
        counter._encoding_for_model.return_value.encode.side_effect = (
            lambda text: list(range(len(str(text))))
        )

        messages = [
            {"role": "system", "content": "hi"},
            {"role": "user", "content": "there", "name": "alice"},
        ]

        tokens = counter._count_gpt_tokens("gpt-3.5-turbo-0613", messages)
        expected = (2 * 3) + 6 + 2 + 4 + 5 + 5 + 1 + 3
        self.assertEqual(tokens, expected)

    def test_claude_count_raises_when_client_missing(self):
        with patch.object(counter_mod, "anthropic") as anthropic_mod:
            anthropic_mod.Anthropic.side_effect = RuntimeError("missing creds")
            counter = TknCounter(use_provider_api=True)

        with self.assertRaises(RuntimeError):
            counter._count_claude_tokens("claude-3", messages=[{"role": "user", "content": "hi"}])

    def test_claude_count_uses_provider_api_when_available(self):
        mock_response = types.SimpleNamespace(input_tokens=42)

        with patch.object(counter_mod, "anthropic") as anthropic_mod:
            anthropic_instance = MagicMock()
            anthropic_instance.beta.messages.count_tokens.return_value = mock_response
            anthropic_mod.Anthropic.return_value = anthropic_instance
            counter = TknCounter(use_provider_api=True)

        tokens = counter._count_claude_tokens("claude-3", messages=[{"role": "user", "content": "hi"}])

        anthropic_instance.beta.messages.count_tokens.assert_called_once()
        self.assertEqual(tokens, 42)

    def test_non_claude_routes_through_token_counter(self):
        counter = TknCounter()
        with patch.object(counter_mod, "token_counter", return_value=7) as token_counter_mock:
            tokens = counter.count_tokens("gemini-pro", messages=[{"role": "user", "content": "hi"}])

        token_counter_mock.assert_called_once()
        self.assertEqual(tokens, 7)


if __name__ == "__main__":
    unittest.main()
