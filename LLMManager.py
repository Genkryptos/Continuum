"""
Lightweight wrapper around the OpenAI SDK with a built-in dummy client.

The module allows the rest of the codebase to call OpenAI-compatible methods
without hard dependency on API credentials or the real SDK; when unavailable,
it supplies predictable stub responses and embeddings for local development and
tests.
"""

import os
from textwrap import shorten
from typing import Any, Dict, Iterable, List, Optional

try:
    import openai as real_openai # official SDK from site-packages
    HAS_REAL_OPENAI = True
    print(HAS_REAL_OPENAI)
except ImportError:
    real_openai = None
    HAS_REAL_OPENAI = False


class _AttrDict:
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class _DummyMessage:
    def __init__(self, content: str, role: str = "assistant"):
        self.content = content
        self.role = role


class _DummyChoice(_AttrDict):
    def __init__(self, message: _DummyMessage, *, index: int = 0, finish_reason: str = "stop"):
        self.index = index
        self.message = message
        self.finish_reason = finish_reason


class _DummyUsage(_AttrDict):
    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0, **extras: Any):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        for key, value in extras.items():
            setattr(self, key, value)


class _DummyCompletionResponse(_AttrDict):
    def __init__(self, *, choices: List[_DummyChoice], usage: _DummyUsage, model: str = "dummy-model"):
        self.id = "dummy-chatcmpl-0"
        self.object = "chat.completion"
        self.created = 0
        self.model = model
        self.choices = choices
        self.usage = usage


class _DummyCompletions:
    def _extract_content(self, messages: Iterable[Dict[str, Any]]) -> str:
        msgs = list(messages)
        for msg in reversed(msgs):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def _build_reply(self, last_user_message: str) -> str:
        if not last_user_message:
            return "Dummy response from placeholder OpenAI client."
        preview = shorten(str(last_user_message), width=120, placeholder="...")
        return (
            "This is a dummy response from the placeholder OpenAI client. "
            f'I received your message: "{preview}"'
        )

    def create(self, *args: Any, **kwargs: Any):
        messages = kwargs.get("messages") or []
        prompt_tokens = 0
        for msg in messages:
            try:
                prompt_tokens += len(msg.get("content", ""))
            except AttributeError:
                continue

        completion_text = self._build_reply(self._extract_content(messages))
        message = _DummyMessage(completion_text)
        usage = _DummyUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=len(message.content),
        )
        choice = _DummyChoice(message)
        return _DummyCompletionResponse(
            choices=[choice],
            usage=usage,
            model=kwargs.get("model", "dummy-model"),
        )


class _DummyChat:
    def __init__(self):
        self.completions = _DummyCompletions()


# ---------- Dummy embeddings ----------

class _DummyEmbeddingItem(_AttrDict):
    def __init__(self, index: int, embedding: List[float]):
        self.index = index
        self.embedding = embedding
        self.object = "embedding"


class _DummyEmbeddingResponse(_AttrDict):
    def __init__(self, vectors: List[_DummyEmbeddingItem], model: str = "dummy-embedding"):
        self.object = "list"
        self.data = vectors
        self.model = model
        self.usage = _DummyUsage(prompt_tokens=0, completion_tokens=0)


class _DummyEmbeddings:
    def __init__(self, embedding_dim: int = 1536):
        self.embedding_dim = embedding_dim

    def create(self, *args: Any, **kwargs: Any):
        inputs = kwargs.get("input") or []
        if not isinstance(inputs, list):
            inputs = [inputs]

        vectors = [
            _DummyEmbeddingItem(index=i, embedding=[0.0] * self.embedding_dim)
            for i, _ in enumerate(inputs)
        ]
        return _DummyEmbeddingResponse(vectors)


# ---------- Public wrapper ----------

class LLM:
    """
    Wrapper around real openai.OpenAI when available,
    otherwise falls back to dummy chat + embeddings.
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs: Any) -> None:
        env_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY")
        force_dummy = os.getenv("CONTINUUM_FORCE_DUMMY_OPENAI", "0") == "1"
        debug = os.getenv("CONTINUUM_DEBUG_OPENAI", "0") == "1"

        if debug:
            print(
                f"[continuum_openai] api_key_present={bool(env_key)} "
                f"force_dummy={force_dummy} HAS_REAL_OPENAI={HAS_REAL_OPENAI}"
            )

        if HAS_REAL_OPENAI and env_key and not force_dummy:
            client = real_openai.OpenAI(api_key=env_key, **kwargs)
            self._client = client
            self.api_key = env_key
            self.chat = client.chat
            self.embeddings = client.embeddings
            if debug:
                print("[continuum_openai] Using REAL OpenAI client")
        else:
            self._client = None
            self.api_key = env_key
            self.chat = _DummyChat()
            self.embeddings = _DummyEmbeddings()
            if debug:
                print("[continuum_openai] Using DUMMY OpenAI client")


class ChatCompletion:
    @staticmethod
    def create(*args: Any, **kwargs: Any):
        force_dummy = os.getenv("CONTINUUM_FORCE_DUMMY_OPENAI", "0") == "1"
        env_key = (
            kwargs.get("api_key")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("OPEN_AI_KEY")
        )
        debug = os.getenv("CONTINUUM_DEBUG_OPENAI", "0") == "1"

        if HAS_REAL_OPENAI and env_key and not force_dummy:
            if debug:
                print("[continuum_openai.ChatCompletion] Using REAL completion")
            client = real_openai.OpenAI(api_key=env_key)
            return client.chat.completions.create(*args, **kwargs)

        if debug:
            print("[continuum_openai.ChatCompletion] Using DUMMY completion")
        return _DummyCompletions().create(*args, **kwargs)
