#!/usr/bin/env python3
"""
bench/supersession_e2e.py
=========================
Supersession correctness **through the shipped Postgres store**, not the
in-memory schema simulation in ``supersession_correctness.py``.

That sibling benchmark validates the LTM *schema* (superseded_by links) with a
numpy model — enough to argue the architecture, but a reviewer will ask whether
the *deployed* path actually behaves the same. This harness answers that: it
replays the **same 50 scenarios** through ``Memory.from_postgres`` — planting
each fact with ``add(text, attribute=…, occurred_at=…)`` and then asking
``current("user", attribute)`` — on the real ``invalidated_at`` / bi-temporal
store.

Attribute-tagged supersession is a **deterministic exact-tag + valid-time
lookup** (see ``Memory.current``), so this needs **no LLM decider and no API
key** — it isolates the store's supersession semantics, which is exactly the
claim the paper makes.

    CONTINUUM_DB_DSN=postgresql://…/throwaway python3 -m bench.supersession_e2e --scenarios 50

Reports two numbers per run:
  * **current()**    — the deterministic exact-tag path (the MCP ``current`` tool)
  * **recall top-1** — the retrieval path, which must filter superseded facts
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime, timedelta

from bench.supersession_correctness import _build_scenarios


def _clear(dsn: str, namespace: str) -> None:
    import psycopg

    with psycopg.connect(dsn) as c, c.cursor() as cur:
        cur.execute("DELETE FROM memory_nodes WHERE namespace = %s", (namespace,))
        c.commit()


def _hit(text: str, current: str, stale: tuple[str, ...]) -> bool:
    """The returned fact carries the current value (and not, instead, a stale one).

    `current()`/recall return a single fact, so the current value being present
    is the signal; the stale guard only fires when current is absent."""
    t = text or ""
    if current in t:
        return True
    return False  # current value not present → wrong (stale or empty)


async def run(dsn: str, n: int) -> int:
    from continuum.memory import Memory

    scenarios = _build_scenarios(n)
    ns = "ss_e2e"
    mem = Memory.from_postgres(dsn, embeddings=True, namespace=ns)
    await mem.start()
    cur_ok = rec_ok = 0
    stale_current = 0
    try:
        await mem.recall("warm up", k=1)
        for sc in scenarios:
            _clear(dsn, ns)
            tag = next((t.attribute_tag for t in sc.turns if t.attribute_tag), None)
            for i, turn in enumerate(sc.turns):
                await mem.add(
                    turn.text,
                    occurred_at=datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i),
                    attribute=turn.attribute_tag,
                )
            got = (await mem.current("user", tag) if tag else None) or ""
            if _hit(got, sc.current_answer, sc.stale_answers):
                cur_ok += 1
            elif any(s in got for s in sc.stale_answers):
                stale_current += 1
            top = (await mem.recall(sc.query, k=1))
            top_text = (top[0].content if top else "") or ""
            if _hit(top_text, sc.current_answer, sc.stale_answers):
                rec_ok += 1
    finally:
        await mem.aclose()

    total = len(scenarios)
    print(f"  end-to-end supersession through Postgres · {total} scenarios\n")
    print(f"  current() correct   : {cur_ok}/{total} = {100 * cur_ok / total:.1f}%"
          f"   (stale returned: {stale_current})")
    print(f"  recall top-1 correct: {rec_ok}/{total} = {100 * rec_ok / total:.1f}%")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios", type=int, default=50)
    args = p.parse_args(argv)
    dsn = os.environ.get("CONTINUUM_DB_DSN")
    if not dsn:
        print("[e2e] set CONTINUUM_DB_DSN to a THROWAWAY migrated db.", file=sys.stderr)
        return 1
    return asyncio.run(run(dsn, args.scenarios))


if __name__ == "__main__":
    raise SystemExit(main())
