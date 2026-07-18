"""
examples/langgraph_node.py
==========================
Continuum as a memory node in a LangGraph graph: before the model runs, recall
relevant memories and inject them into the prompt; after, remember the exchange.

Requires LangGraph:  pip install langgraph
(Continuum itself does not depend on LangGraph — this is an integration example.)

Run:  python examples/langgraph_node.py
"""

from __future__ import annotations

import asyncio
from typing import Any, TypedDict

from continuum import Memory


class ChatState(TypedDict):
    user: str        # the incoming user message
    memories: list[str]
    reply: str


def build_graph(mem: Memory) -> Any:
    """Wire a minimal recall → respond → remember graph around a Continuum Memory."""
    from langgraph.graph import END, StateGraph

    async def recall_node(state: ChatState) -> ChatState:
        hits = await mem.recall(state["user"], k=5)
        return {**state, "memories": [h.content for h in hits]}

    async def respond_node(state: ChatState) -> ChatState:
        # Your LLM call goes here; we stub it to keep the example dependency-free.
        ctx = "; ".join(state["memories"]) or "(no memories yet)"
        return {**state, "reply": f"[using memory: {ctx}] ..."}

    async def remember_node(state: ChatState) -> ChatState:
        await mem.remember(state["user"])
        return state

    g = StateGraph(ChatState)
    g.add_node("recall", recall_node)
    g.add_node("respond", respond_node)
    g.add_node("remember", remember_node)
    g.set_entry_point("recall")
    g.add_edge("recall", "respond")
    g.add_edge("respond", "remember")
    g.add_edge("remember", END)
    return g.compile()


async def main() -> None:
    mem = Memory.in_memory()
    async with mem:
        await mem.add("My favorite editor is Neovim")
        graph = build_graph(mem)
        out = await graph.ainvoke({"user": "what editor do I use?", "memories": [], "reply": ""})
        print("memories recalled:", out["memories"])
        print("reply:", out["reply"])


if __name__ == "__main__":
    asyncio.run(main())
