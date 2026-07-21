#!/usr/bin/env python3
"""
scripts/mcp_eval.py
===================
Deterministic **retrieval quality** eval for the Continuum MCP server — the
reproducible version of "how does it perform with Claude?". It drives the same
tools Claude calls (`remember` / `recall` / `current` / `timeline`), over a
fixed scenario with known-correct answers, and scores:

  • recall@1 / recall@3  — did the RIGHT memory come back, ranked well?
  • current accuracy      — supersession resolves to the latest value?
  • timeline accuracy      — bi-temporal history returned in order?

No LLM / no judge: if the right memory is retrieved, Claude answers correctly —
so retrieval accuracy is the honest proxy. The scenario has distractors, so
recency-only ranking (the in-memory backend) scores low on recall@1, while the
Postgres + embedder backend ranks the relevant fact first.

Backend is whatever the server picks: set CONTINUUM_DB_DSN (after `make db-up &&
make db-migrate`) to eval the production stack; unset for the in-memory baseline.

    python3 scripts/mcp_eval.py                 # auto-locate continuum-mcp
    python3 scripts/mcp_eval.py --k 3 --min-recall1 0.7
    CONTINUUM_DB_DSN=postgresql://... python3 scripts/mcp_eval.py

Exit 0 if it clears the thresholds, else 1.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading

# ── scenario: facts (with distractors) + queries with known answers ───────────

FACTS: list[dict[str, str]] = [
    {"text": "My name is Mayank and I build Continuum solo."},
    {"text": "I live in Boston.", "occurred_at": "2026-01-10", "attribute": "residence"},
    {
        "text": "I moved from Boston to New York City.",
        "occurred_at": "2026-06-15",
        "attribute": "residence",
    },
    {"text": "My favorite programming language is Python."},
    {"text": "My favorite database is PostgreSQL with the pgvector extension."},
    {"text": "I drive a red Tesla Model 3."},
    {"text": "My dog is a corgi named Pixel."},
    {"text": "My preferred code editor is Neovim."},
    {"text": "I studied computer science at IIT."},
    {"text": "My favorite cuisine is Japanese, especially ramen."},
    {"text": "I play the acoustic guitar on weekends."},
    {"text": "My coffee order is a flat white with oat milk."},
    {"text": "Continuum v2.0.0 was released on 2026-07-19."},
    {"text": "My laptop is a 14-inch MacBook Pro."},
]

# query -> substring that must appear in the retrieved memory to count as a hit
RECALL_QUERIES: list[dict[str, str]] = [
    {"query": "what car do I drive", "expect": "Tesla"},
    {"query": "what is my pet's name", "expect": "Pixel"},
    {"query": "which text editor do I use for coding", "expect": "Neovim"},
    {"query": "what is my favorite programming language", "expect": "Python"},
    {"query": "where did I go to university", "expect": "IIT"},
    {"query": "which database do I prefer", "expect": "PostgreSQL"},
    {"query": "what kind of food do I like", "expect": "Japanese"},
    {"query": "what instrument do I play", "expect": "guitar"},
    {"query": "how do I take my coffee", "expect": "flat white"},
    {"query": "what laptop do I own", "expect": "MacBook"},
]

# The set above shares content words with its facts ("what car do I *drive*" →
# "I *drive* a red Tesla"), so the lexical channel alone can carry it — it says
# little about semantic recall. These ask the same questions with **no word in
# common** with the target fact, so only the dense channel can answer them. This
# is the honest paraphrase number.
PARAPHRASE_QUERIES: list[dict[str, str]] = [
    {"query": "what do I commute in", "expect": "Tesla"},
    {"query": "tell me about my pet", "expect": "Pixel"},
    {"query": "what do I write software in", "expect": "Neovim"},
    {"query": "which coding language do I reach for", "expect": "Python"},
    {"query": "where did I get my degree", "expect": "IIT"},
    {"query": "how do I store my data", "expect": "PostgreSQL"},
    {"query": "what should we order for dinner", "expect": "Japanese"},
    {"query": "do I have any musical hobbies", "expect": "guitar"},
    {"query": "what is my usual morning drink", "expect": "flat white"},
    {"query": "what machine do I work on", "expect": "MacBook"},
]

CURRENT_CHECKS: list[dict[str, str]] = [
    {"subject": "user", "attribute": "residence", "expect": "New York"},
    # bi-temporal: before the June move, the answer must be the OLD value
    {"subject": "user", "attribute": "residence", "as_of": "2026-03-01", "expect": "Boston"},
]

TIMEOUT_S = 180.0  # watchdog: kill the server if a reply never arrives

TIMELINE_CHECKS: list[dict[str, object]] = [
    {"entity": "Boston", "expect_order": ["Boston", "New York"]},
]


# ── minimal MCP stdio client (newline-delimited JSON-RPC) ─────────────────────


def _server_cmd(explicit: str | None) -> list[str]:
    if explicit:
        return [explicit]
    found = shutil.which("continuum-mcp")
    return [found] if found else [sys.executable, "-m", "continuum.mcp.server"]


def _rpc(mid: int, method: str, params: dict | None = None) -> str:
    msg: dict = {"jsonrpc": "2.0", "method": method}
    if mid:
        msg["id"] = mid
    if params is not None:
        msg["params"] = params
    return json.dumps(msg)


def _run_session(cmd: list[str], k: int) -> dict[int, dict]:
    """Store all facts, then issue every query, in one server process."""
    reqs: list[str] = [
        _rpc(
            1,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "mcp-eval", "version": "0"},
            },
        ),
        _rpc(0, "notifications/initialized"),
    ]
    mid = 100
    id_map: dict[int, tuple[str, int]] = {}  # request id -> (kind, index)
    write_ids: set[int] = set()

    for f in FACTS:
        args = {"text": f["text"]}
        for opt in ("occurred_at", "attribute"):
            if opt in f:
                args[opt] = f[opt]
        reqs.append(_rpc(mid, "tools/call", {"name": "remember", "arguments": args}))
        write_ids.add(mid)
        mid += 1

    # Queries are issued only AFTER every write has been acknowledged. MCP
    # servers handle requests concurrently, so piping writes+reads together lets
    # the fast reads race ahead of the slower writes — which measures a race,
    # not retrieval quality.
    queries: list[str] = []
    for i, q in enumerate(RECALL_QUERIES):
        queries.append(
            _rpc(mid, "tools/call", {"name": "recall", "arguments": {"query": q["query"], "k": k}})
        )
        id_map[mid] = ("recall", i)
        mid += 1
    for i, q in enumerate(PARAPHRASE_QUERIES):
        queries.append(
            _rpc(mid, "tools/call", {"name": "recall", "arguments": {"query": q["query"], "k": k}})
        )
        id_map[mid] = ("paraphrase", i)
        mid += 1
    for i, c in enumerate(CURRENT_CHECKS):
        queries.append(
            _rpc(
                mid,
                "tools/call",
                {
                    "name": "current",
                    "arguments": {
                        "subject": c["subject"],
                        "attribute": c["attribute"],
                        **({"as_of": c["as_of"]} if "as_of" in c else {}),
                    },
                },
            )
        )
        id_map[mid] = ("current", i)
        mid += 1
    for i, t in enumerate(TIMELINE_CHECKS):
        queries.append(
            _rpc(mid, "tools/call", {"name": "timeline", "arguments": {"entity": t["entity"]}})
        )
        id_map[mid] = ("timeline", i)
        mid += 1

    # Read replies as they stream rather than waiting for the process to exit:
    # the Postgres backend starts background workers + a connection pool, so the
    # server legitimately stays alive after stdin EOF. A watchdog kills it if a
    # reply never arrives (killing makes readline() return '' → clean break).
    replies: dict[int, dict] = {}
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    watchdog = threading.Timer(TIMEOUT_S, proc.kill)
    watchdog.start()

    def _drain(want: set[int]) -> set[int]:
        """Read replies until every id in *want* has arrived (or EOF)."""
        assert proc.stdout
        while want:
            line = proc.stdout.readline()
            if not line:  # EOF (or watchdog killed us)
                break
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(m, dict) and isinstance(m.get("id"), int) and m["id"] in want:
                replies[m["id"]] = m
                want.discard(m["id"])
        return want

    missing: set[int] = set()
    try:
        assert proc.stdin
        # Phase 1 — writes, and wait for every ack before reading anything.
        proc.stdin.write("\n".join(reqs) + "\n")
        proc.stdin.flush()
        unacked = _drain(set(write_ids))
        if unacked:
            sys.stderr.write(f"[mcp-eval] warning: {len(unacked)} write(s) unacknowledged\n")
        # Phase 2 — queries, now that memory is fully populated.
        proc.stdin.write("\n".join(queries) + "\n")
        proc.stdin.flush()
        missing = _drain(set(id_map))
    finally:
        watchdog.cancel()
        proc.kill()
        stderr = (proc.stderr.read() if proc.stderr else "") or ""
        proc.wait(timeout=10)

    if missing:
        sys.stderr.write(f"\n[mcp-eval] missing {len(missing)} repl(ies) — server stderr tail:\n")
        sys.stderr.write("\n".join(stderr.splitlines()[-12:]) + "\n")
    return {mid_: {"reply": replies.get(mid_), "meta": meta} for mid_, meta in id_map.items()}


def _structured(reply: dict | None) -> object:
    if not reply or "result" not in reply:
        return None
    return reply["result"].get("structuredContent", {}).get("result")


# ── scoring ───────────────────────────────────────────────────────────────────


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="mcp-eval", description="Continuum MCP retrieval eval")
    p.add_argument("--server-cmd", default=None, help="path to continuum-mcp (default: auto)")
    p.add_argument("--k", type=int, default=3, help="recall depth (recall@k), default 3")
    p.add_argument("--min-recall1", type=float, default=0.0, help="fail below this recall@1")
    args = p.parse_args(argv)

    cmd = _server_cmd(args.server_cmd)
    backend = (
        "postgres"
        if (os.environ.get("CONTINUUM_DB_DSN") or os.environ.get("DATABASE_URL"))
        else "in-memory"
    )
    embeds = os.environ.get("CONTINUUM_MCP_EMBEDDINGS", "1") not in {"0", "false", "no", "off"}
    print(f"[mcp-eval] server: {' '.join(cmd)}")
    print(
        f"[mcp-eval] backend: {backend}"
        + (f" (embeddings={'on' if embeds else 'off'})" if backend == "postgres" else "")
    )
    print(
        f"[mcp-eval] scenario: {len(FACTS)} facts, {len(RECALL_QUERIES)} recall queries, k={args.k}\n"
    )

    out = _run_session(cmd, args.k)

    recall1 = recall_k = current_ok = timeline_ok = 0
    para1 = para_k = 0
    recall_rows: list[str] = []
    para_rows: list[str] = []
    for entry in out.values():
        kind, idx = entry["meta"]
        data = _structured(entry["reply"])
        if kind in ("recall", "paraphrase"):
            q = (RECALL_QUERIES if kind == "recall" else PARAPHRASE_QUERIES)[idx]
            hits = [str(d.get("content", "")) for d in data] if isinstance(data, list) else []
            at1 = bool(hits) and q["expect"].lower() in hits[0].lower()
            atk = any(q["expect"].lower() in h.lower() for h in hits[: args.k])
            mark = "✓@1" if at1 else ("·@k" if atk else "✗  ")
            row = f"  {mark}  {q['query'][:38]:38s} → want '{q['expect']}'"
            if kind == "recall":
                recall1 += at1
                recall_k += atk
                recall_rows.append(row)
            else:
                para1 += at1
                para_k += atk
                para_rows.append(row)
        elif kind == "current":
            c = CURRENT_CHECKS[idx]
            current_ok += bool(isinstance(data, str) and c["expect"].lower() in data.lower())
        elif kind == "timeline":
            t = TIMELINE_CHECKS[idx]
            contents = [str(d.get("content", "")) for d in data] if isinstance(data, list) else []
            blob = " || ".join(contents).lower()
            order = [str(s).lower() for s in t["expect_order"]]  # type: ignore[union-attr]
            positions = [blob.find(s) for s in order]
            timeline_ok += bool(
                all(pos >= 0 for pos in positions) and positions == sorted(positions)
            )

    n_recall = len(RECALL_QUERIES) or 1
    print("recall (relevant memory ranked well?):")
    print("\n".join(recall_rows))
    r1 = recall1 / n_recall
    rk = recall_k / n_recall
    print(f"\n  recall@1 = {recall1}/{n_recall} = {r1:.0%}   (the fact ranked FIRST)")
    print(f"  recall@{args.k} = {recall_k}/{n_recall} = {rk:.0%}   (in the top {args.k})")
    n_para = len(PARAPHRASE_QUERIES) or 1
    p1, pk = para1 / n_para, para_k / n_para
    print("\nparaphrase (no word shared with the fact \u2014 dense channel only):")
    print("\n".join(para_rows))
    print(f"\n  recall@1 = {para1}/{n_para} = {p1:.0%}")
    print(f"  recall@{args.k} = {para_k}/{n_para} = {pk:.0%}")

    print(f"\n  supersession (current) = {current_ok}/{len(CURRENT_CHECKS)}")
    print(f"  timeline (ordered)     = {timeline_ok}/{len(TIMELINE_CHECKS)}")

    passed = r1 >= args.min_recall1
    print(
        f"\n[mcp-eval] {'PASS ✓' if passed else 'FAIL ✗'}"
        + (f"  (recall@1 {r1:.0%} < {args.min_recall1:.0%} gate)" if not passed else "")
    )
    if backend == "in-memory":
        print(
            "[mcp-eval] note: in-memory has NO retriever (recency only) — low recall@1 is"
            " expected. Set CONTINUUM_DB_DSN for relevance-ranked recall."
        )
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except subprocess.TimeoutExpired:
        print("[mcp-eval] ✗ server timed out")
        raise SystemExit(1) from None
