"""
Adapter class that hides whether an LLM call is served by a local Ollama model
or a hosted OpenAI endpoint. It normalizes responses and basic error handling
so the higher-level agents can treat both providers the same.
"""

import json
import logging
import time
from typing import Dict, List, Optional

import requests
from LLMManager import LLM
import LLMManager

from settings import LOCAL_MODEL_TIMEOUT, OPENAI_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)


class LocalAgent:
    """Thin wrapper around either an OpenAI chat model or a local Ollama model."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434/",
        model: str = LLM_MODEL,
        provider: Optional[str] = None,
        openai_client: Optional[LLM] = None,
        request_timeout: Optional[float] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.provider = provider or self._infer_provider(model)
        self._client = openai_client
        self.request_timeout = (
            request_timeout if request_timeout is not None else LOCAL_MODEL_TIMEOUT
        )

        self._validate_provider()
        self._validate_timeout()

        if self.provider == "openai":
            self._client = self._client or LLM(api_key=OPENAI_API_KEY)
            self._verify_openai_configuration()
        else:
            self.verify_connection()

    def _infer_provider(self, model: str) -> str:
        """Heuristically determine provider from model name."""
        if model.lower().startswith("gpt-"):
            return "openai"
        return "ollama"

    def _validate_provider(self) -> None:
        if self.provider not in {"openai", "ollama"}:
            raise ValueError(
                "Invalid provider. Supported providers are 'openai' and 'ollama'."
            )

    def _validate_timeout(self) -> None:
        if self.request_timeout is None:
            return
        try:
            timeout_val = float(self.request_timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("request_timeout must be a number") from exc

        if timeout_val <= 0:
            raise ValueError("request_timeout must be positive")

        self.request_timeout = timeout_val

    def _verify_openai_configuration(self) -> bool:
        if not OPENAI_API_KEY and not getattr(self._client, "api_key", None):
            raise ValueError("OPEN_AI_KEY environment variable is not set.")
        return True

    def _create_openai_completion(self, messages: List[Dict]):
        if hasattr(self._client, "chat") and hasattr(self._client.chat, "completions"):
            return self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
            )

        if hasattr(LLMManager, "ChatCompletion"):
            return LLMManager.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
            )

        raise RuntimeError(
            "OpenAI client does not support chat completions. Update the SDK or "
            "provide a compatible client."
        )

    def verify_connection(self) -> bool:
        """Probe the local Ollama server to verify connectivity."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags", timeout=self.request_timeout
            )
            response.raise_for_status()
            print(
                f"✅ Ollama connected! Available models: "
                f"{len(response.json().get('models', []))}"
            )
            return True
        except requests.Timeout as exc:
            logger.exception("Local model connection attempt timed out")
            raise RuntimeError("Timeout connecting to Ollama") from exc
        except requests.ConnectionError as exc:
            logger.exception("Local model connection error")
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. Make sure ollama serve is running."
            ) from exc
        except requests.HTTPError as exc:
            details = self._format_http_error(exc.response, exc)
            logger.exception("Local model connection HTTP error")
            raise RuntimeError(details) from exc

    def call_model(self, messages: List[Dict]) -> Dict:
        """Dispatch to the configured provider and normalize its response."""
        start = time.time()

        if self.provider == "openai":
            return self._call_openai(messages, start)

        return self._call_local(messages, start)

    def _call_openai(self, messages: List[Dict], start: float) -> Dict:
        """Invoke OpenAI chat completions via the configured client."""
        try:
            response = self._create_openai_completion(messages)

            latency = time.time() - start
            usage = getattr(response, "usage", None)
            return {
                "response": response.choices[0].message.content.strip(),
                "tokens_used": {
                    "input": getattr(usage, "prompt_tokens", 0),
                    "output": getattr(usage, "completion_tokens", 0),
                    "total": getattr(usage, "total_tokens", 0),
                },
                "latency": latency,
                "success": True,
            }
        except Exception as e:  # noqa: BLE001
            logger.exception("OpenAI model call failed")
            return {
                "response": None,
                "error": str(e),
                "success": False,
            }

    def _call_local(self, messages: List[Dict], start: float) -> Dict:
        """Invoke a local Ollama model and extract latency/token metadata."""
        prompt = self._format_prompt(messages)

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                },
                timeout=self.request_timeout,
            )
            response.raise_for_status()

            latency = time.time() - start
            result = response.json()

            prompt_evals = result.get("prompt_eval_count", 0)
            evals = result.get("eval_count", 0)

            return {
                "response": result.get("response", "").strip(),
                "tokens_used": {
                    "input": prompt_evals,
                    "output": evals,
                    "total": prompt_evals + evals,
                },
                "latency": latency,
                "success": True,
            }

        except requests.Timeout:
            logger.exception("Local model request timed out")
            return {
                "response": None,
                "error": "Request timeout",
                "success": False,
            }
        except requests.ConnectionError as e:
            logger.exception("Local model connection error")
            return {
                "response": None,
                "error": f"Connection error: {e}",
                "success": False,
            }
        except requests.HTTPError as e:
            details = self._format_http_error(e.response, e)
            logger.exception("Local model HTTP error")
            return {
                "response": None,
                "error": details,
                "success": False,
            }
        except Exception as e:  # noqa: BLE001
            logger.exception("Unexpected local model error")
            return {
                "response": None,
                "error": str(e),
                "success": False,
            }

    def _format_http_error(self, response: Optional[requests.Response], error: Exception) -> str:
        if response is None:
            return str(error)

        status = getattr(response, "status_code", "unknown")
        reason = getattr(response, "reason", "")
        body = getattr(response, "text", "")
        details = "; ".join(
            part
            for part in [
                f"HTTP error {status}",
                f"reason: {reason}" if reason else "",
                f"body: {body.strip()}" if body and body.strip() else "",
            ]
            if part
        )
        return details or f"HTTP error {status}: {error}"

    def _format_prompt(self, messages: List[Dict]) -> str:
        prompt = ""
        for message in messages:
            role = message["role"].upper()
            content = message["content"]
            prompt += f"{role}: {content}\n\n"
        prompt += "ASSISTANT: "
        return prompt


if __name__ == "__main__":

    llm = LocalAgent(model=LLM_MODEL)

    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what Python is in 3 short sentences."},
    ]

    result = llm.call_model(test_messages)

    if result["success"]:
        print("\n=== MODEL RESPONSE ===")
        print(result["response"])
        print("\n=== METRICS ===")
        print(f"Latency: {result['latency']:.2f}s")
        print(f"Tokens (in/out/total): {result['tokens_used']}")
    else:
        print("\n❌ CALL FAILED")
        print("Error:", result["error"])
