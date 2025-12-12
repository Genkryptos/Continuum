import logging
import sys
import types
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

try:
    import anthropic
except ImportError:  # pragma: no cover - optional dependency
    anthropic = None
try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    class _FallbackEncoding:
        def encode(self, value: str):
            return list(value.encode("utf-8"))

        def decode(self, tokens):  # pragma: no cover - minimal safety net
            if isinstance(tokens, (bytes, bytearray)):
                return tokens.decode("utf-8", errors="ignore")

            try:
                return bytes(tokens).decode("utf-8", errors="ignore")
            except Exception:  # noqa: BLE001
                return ""

    def _fallback_encoding_for_model(_model: str):  # pragma: no cover - tests patch behavior
        raise ValueError("tiktoken not installed")

    def _fallback_get_encoding(_default: str):  # pragma: no cover - tests patch behavior
        return _FallbackEncoding()

    tiktoken = types.SimpleNamespace(
        encoding_for_model=_fallback_encoding_for_model,
        get_encoding=_fallback_get_encoding,
    )
    sys.modules["tiktoken"] = tiktoken
try:
    from litellm import token_counter
except ImportError:  # pragma: no cover - minimal fallback when dependency missing
    def token_counter(model: str, messages):
        return sum(len(str(value)) for message in messages for value in message.values())


logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _resolve_encoding(model: str, default_encoding: str):
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:  # noqa: BLE001
        logger.debug(
            "Falling back to default encoding for model %s; token counts may be approximate.",
            model,
        )
        return tiktoken.get_encoding(default_encoding)


class TknCounter:
    """Provider-aware token counter for chat-style messages.

    The class prefers provider/tokenizer-aware accounting but degrades to
    heuristic fallbacks when dependencies like ``tiktoken`` are unavailable.
    Those fallbacks intentionally over-estimate to avoid under-counting but are
    less accurate for non-ASCII text. When ``use_provider_api`` is enabled, an
    Anthropic client is instantiated lazily and requires valid credentials for
    Claude models; otherwise counting those models will raise at call time.
    """

    # Token accounting rules adapted from OpenAI's chat tokenization guidance.
    _GPT_RULES: Dict[str, Tuple[int, int]] = {
        # Rules derived from https://github.com/openai/openai-cookbook for chat models.
        "gpt-3.5-turbo-0301": (4, -1),
        "gpt-3.5-turbo-0613": (3, 1),
        "gpt-3.5-turbo-1106": (3, 1),
        "gpt-4-0314": (3, 1),
        "gpt-4-0613": (3, 1),
        "gpt-4-1106-preview": (3, 1),
        "gpt-4o": (3, 1),
    }

    def __init__(self, use_provider_api: bool = False, default_encoding: str = "cl100k_base"):
        self.use_provider_api = use_provider_api
        self.default_encoding = default_encoding
        self.anthropic_client: Optional[anthropic.Anthropic] = None

        if use_provider_api:
            if anthropic is None:
                raise RuntimeError("Anthropic SDK is not installed")
            try:
                self.anthropic_client = anthropic.Anthropic()
            except Exception as exc:  # noqa: BLE001
                # Defer raising until a Claude-specific count is requested so the
                # caller gets a targeted error message.
                self.anthropic_client = None
                self._anthropic_error = exc
            else:
                self._anthropic_error = None

    def _gpt_rules_for_model(self, model: str) -> Tuple[int, int]:
        """Return the tokens-per-message and tokens-per-name tuple for the model."""

        for known_model, rules in self._GPT_RULES.items():
            if model.startswith(known_model):
                return rules

        # Default rules align with modern GPT chat models if no exact match is found.
        return 3, 1

    def _encoding_for_model(self, model: str):
        return _resolve_encoding(model, self.default_encoding)

    def _count_gpt_tokens(self, model: str, messages: List[Dict[str, str]]) -> int:
        encoding = self._encoding_for_model(model)
        tokens_per_message, tokens_per_name = self._gpt_rules_for_model(model)
        num_tokens = 0

        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name

        # Every reply is primed with <|start|>assistant<|message|>
        num_tokens += 3
        return num_tokens

    def _count_claude_tokens(self, model: str, messages: List[Dict[str, str]]) -> int:
        if self.use_provider_api:
            if self.anthropic_client is None:
                raise RuntimeError(
                    "Anthropic client is not initialized. Ensure API credentials are configured "
                    "or disable use_provider_api."
                ) from getattr(self, "_anthropic_error", None)

            response = self.anthropic_client.beta.messages.count_tokens(model=model, messages=messages)
            return response.input_tokens

        return token_counter(model=model, messages=messages)

    def count_tokens(self, model: str, messages: List[Dict[str, str]]) -> int:
        """Count request tokens for the given model and messages."""

        if model.startswith("gpt-"):
            return self._count_gpt_tokens(model=model, messages=messages)

        if model.startswith("claude-"):
            return self._count_claude_tokens(model=model, messages=messages)

        # Explicitly route other known families through litellm for clarity.
        if model.startswith(("gemini-", "cohere-", "mistral-", "mixtral-", "command-")):
            return token_counter(model=model, messages=messages)

        return token_counter(model=model, messages=messages)

    def count_request_tokens(self, model: str, messages: List[Dict[str, str]]) -> int:
        """Public alias for counting input/request tokens."""

        return self.count_tokens(model=model, messages=messages)

    def estimate_total_tokens(self, model: str, messages: List[Dict[str, str]], max_output_tokens: Optional[int] = None) -> int:
        """
        Estimate total tokens for a request by combining input tokens with a
        projected output budget.
        """

        input_tokens = self.count_request_tokens(model=model, messages=messages)
        return input_tokens + (max_output_tokens or 0)

    def truncate_text(self, model: str, text: str, max_tokens: int) -> str:
        """Truncate ``text`` to fit within ``max_tokens`` using model encoding."""

        if max_tokens <= 0:
            return ""

        encoding = self._encoding_for_model(model)
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens]
        try:
            return encoding.decode(truncated_tokens)
        except Exception:  # noqa: BLE001
            return ""
