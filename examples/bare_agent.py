"""
examples/bare_agent.py
======================
The smallest possible Continuum agent: a memory-backed loop with no framework,
no Postgres, no API key. Run it:

    python examples/bare_agent.py

It shows the five verbs of the public API — add / recall / current / timeline —
including supersession (the "current" residence updates when the user changes
their mind). Doubles as a smoke test for the Memory facade.
"""

from __future__ import annotations

import asyncio

from continuum import Memory


async def main() -> None:
    mem = Memory.in_memory()  # zero-config: in-process stores, no DB, no model
    async with mem:
        # The user tells the agent things over time.
        await mem.add("My residence is Boston")
        await mem.add("My favorite rice is Japanese short-grain")
        await mem.add("My residence is now New York")  # updates residence

        # Retrieve memories relevant to a query (in-memory = recency/lexical;
        # attach an embedder or use Postgres for dense semantic recall).
        hits = await mem.recall("residence", k=5)
        print("recall(residence):", [h.content for h in hits])

        # The CURRENT resolved value of an attribute — latest wins.
        print("current residence:", await mem.current("user", "residence"))

        # History for an entity, oldest -> newest (bi-temporal on Postgres).
        history = await mem.timeline("residence")
        print("residence timeline:", [h.content for h in history])


if __name__ == "__main__":
    asyncio.run(main())
