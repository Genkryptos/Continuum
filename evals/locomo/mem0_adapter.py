"""
evals/locomo/mem0_adapter.py
============================
Mem0 side of the LOCOMO head-to-head, using the ``mem0ai`` package in
**standard config** (not handicapped).

Fairness contract
-----------------
The *answerer* is identical to the Continuum side — the same shared LLM
(OpenRouter ``gpt-oss-120b``) turns retrieved memory into the final
answer. What differs is the **memory layer**:

* Mem0 ``add()`` runs its own extraction → stores distilled facts.
* Mem0 ``search(query)`` returns the top-k relevant facts.
* The shared answerer then composes the answer from those facts.

So we measure *Mem0's memory* vs *Continuum's memory*, holding the
answerer + judge constant.

Version sensitivity
-------------------
``mem0ai``'s config schema and return shapes have shifted across
releases. This adapter targets the common ``Memory.from_config`` /
``add`` / ``search`` surface and reads results defensively (handles
both ``{"results": [...]}`` and bare-list returns, and both ``memory``
and ``text`` keys). **Verify against your installed mem0ai version on
the first 50-question smoke** — if the config or result keys differ,
adjust ``_build_memory`` / ``_extract_memories`` here.

Mem0's internal LLM + embedder are pointed at OpenRouter (OpenAI-
compatible) via ``OPENROUTER_API_KEY`` so it runs without an OpenAI
key. Override the model with ``MEM0_LLM_MODEL`` /
``MEM0_EMBED_MODEL`` if needed.
"""
from __future__ import annotations

import os
import time
from typing import Any

from evals.locomo.loader import LocomoConversation

# OpenRouter endpoint reused for Mem0's internal extraction LLM.
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _build_memory() -> Any:
    """
    Construct a Mem0 ``Memory`` that runs without an OpenAI key:

    * **LLM** (fact extraction) → OpenRouter, OpenAI-compatible. Uses
      ``gpt-oss-120b`` by default (``MEM0_LLM_MODEL`` to override).
    * **Embedder** → a LOCAL HuggingFace sentence-transformer
      (``all-MiniLM-L6-v2`` by default, ``MEM0_EMBED_MODEL`` to
      override). Mem0's *default* embedder is OpenAI, which needs an
      OPENAI_API_KEY you don't have — and OpenRouter doesn't serve
      embeddings. The local embedder is also the *fair* choice: it's
      the same model Continuum uses, so both systems share an embedder
      and only the memory layer differs.

    This is "standard Mem0" minus the OpenAI dependency, not a handicap.

    Version note: mem0ai's config schema shifts between releases. If
    ``from_config`` rejects these keys on your version, adjust the
    config dict here (LLM ``openai_base_url`` / embedder ``model`` are
    the version-sensitive bits). Requires ``sentence-transformers``
    (already a Continuum dep) for the local embedder.
    """
    from mem0 import Memory  # lazy so the module loads without mem0ai

    # Mem0's add() asks the LLM to emit structured JSON memory ops.
    # REASONING models (gpt-oss-120b) are bad at this: they burn tokens
    # thinking and return truncated/unterminated JSON ("Unterminated
    # string" → mem0's 'list has no attribute get'). Use a NON-reasoning
    # instruct model for extraction — what mem0's own default
    # (gpt-4o-mini) assumes. This is mem0's memory-layer internals, NOT
    # the answerer (the answerer stays gpt-oss-120b on both sides for
    # fairness), so a clean-JSON model here is correct, not a handicap.
    llm_model = os.environ.get("MEM0_LLM_MODEL", "meta-llama/llama-3.3-70b-instruct")
    embed_model = os.environ.get(
        "MEM0_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    # all-MiniLM-L6-v2 → 384 dims. Mem0 otherwise defaults the vector
    # store collection to 1536 (OpenAI's size), causing a shape
    # mismatch (0,1536) vs (384,). Tell BOTH the embedder and the
    # vector store the real dimension. Override MEM0_EMBED_DIMS if you
    # swap MEM0_EMBED_MODEL for a model of a different size.
    dims = int(os.environ.get("MEM0_EMBED_DIMS", "384"))
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    config: dict[str, Any] = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": llm_model,
                "openai_base_url": _OPENROUTER_BASE,
                "api_key": api_key,
                # Headroom so the extraction JSON completes, not truncates.
                "max_tokens": 2048,
                "temperature": 0.0,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {"model": embed_model, "embedding_dims": dims},
        },
        # In-memory Qdrant sized to the local embedder. on_disk=False
        # keeps it fresh per process (no stale 1536-dim collection
        # persisting between runs).
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "locomo_mem0",
                "embedding_model_dims": dims,
                "on_disk": False,
            },
        },
    }
    return Memory.from_config(config)


def _extract_memories(search_result: Any) -> list[str]:
    """Normalise Mem0's search return into a list of memory strings."""
    rows = search_result
    if isinstance(search_result, dict):
        rows = search_result.get("results", search_result.get("memories", []))
    out: list[str] = []
    for r in rows or []:
        if isinstance(r, str):
            out.append(r)
        elif isinstance(r, dict):
            text = r.get("memory") or r.get("text") or r.get("content")
            if text:
                out.append(str(text))
    return out


class Mem0LocomoAnswerer:
    """
    Ingests one LOCOMO conversation into Mem0, then answers its
    questions: ``search()`` for relevant memories → shared answerer
    composes the final answer. One instance per sample.
    """

    def __init__(
        self,
        conversation: LocomoConversation,
        *,
        llm: Any,
        search_limit: int = 10,
        answer_max_tokens: int = 2048,
    ) -> None:
        self._conversation = conversation
        self._llm = llm
        self._search_limit = search_limit
        self._answer_max_tokens = answer_max_tokens
        self._memory: Any | None = None
        self._user_id = conversation.sample_id or "locomo_user"

    async def _ensure_ingested(self) -> Any:
        if self._memory is not None:
            return self._memory
        memory = _build_memory()
        # Add the conversation as OpenAI-style messages so Mem0's
        # extractor sees speaker structure. Mem0.add is sync.
        messages = [
            {"role": "user", "content": f"{t.speaker}: {t.text}"}
            for t in self._conversation.turns
        ]
        # Add in chunks — Mem0's extractor over a 200-turn conversation
        # in one call can blow context; chunk to keep extraction sane.
        chunk = 20
        for i in range(0, len(messages), chunk):
            memory.add(messages[i : i + chunk], user_id=self._user_id)
        self._memory = memory
        return memory

    async def answer(
        self, question: str, *, category: int | None = None,
    ) -> dict[str, Any]:
        memory = await self._ensure_ingested()
        t0 = time.perf_counter()
        try:
            search_result = memory.search(
                question, user_id=self._user_id, limit=self._search_limit,
            )
        except TypeError:
            # Older mem0 signatures don't take ``limit``.
            search_result = memory.search(question, user_id=self._user_id)
        memories = _extract_memories(search_result)
        context = "\n".join(f"- {m}" for m in memories) or "(no memories found)"
        prompt = (
            "Answer the question using ONLY these retrieved memories. "
            "Reply with just the answer — no explanation. If the answer "
            "isn't present, reply 'I don't know.'\n\n"
            f"Memories:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )
        answer = await self._llm.complete(
            prompt=prompt, max_tokens=self._answer_max_tokens,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "answer": (answer or "").strip(),
            "retrieved_memories": memories,
            "latency_ms": latency_ms,
            "llm_calls": 1,
        }


__all__ = ["Mem0LocomoAnswerer"]
