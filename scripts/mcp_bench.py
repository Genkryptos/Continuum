#!/usr/bin/env python3
"""
scripts/mcp_bench.py
====================
Latency benchmark for the Continuum MCP server — how *fast* each tool answers,
measured through the real stdio server (the same binary an MCP client spawns).

Reports per-tool p50/p95/mean, the one-off cold start, and a breakdown that
separates the embedder's forward pass from the database work — because on this
stack the embedder dominates everything semantic:

    recall, unique query each time (embedder runs)  ~82 ms
    recall, same query repeated   (embed cache hit)  ~7 ms   -> embedder ~75 ms
    current (exact tag lookup, never embeds)         ~1.5 ms

**It never touches your live memory.** `CONTINUUM_DB_DSN` / `DATABASE_URL` are
stripped from the server's environment; the benchmark writes facts, so pointing
it at a real store would pollute it. Pass `--dsn` for a throwaway Postgres:

    python3 scripts/mcp_bench.py                      # in-memory (no DB needed)
    python3 scripts/mcp_bench.py --dsn postgresql://…/continuum_bench
    make mcp-bench MCP_BENCH_DSN=postgresql://…/continuum_bench
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any

TIMEOUT_S = 120.0


class _Server:
    """A spawned MCP stdio server; `call` blocks until that request's reply."""

    def __init__(self, cmd: list[str], dsn: str | None) -> None:
        # Strip any live DSN, then opt in explicitly — a benchmark that writes
        # must never inherit the store the user actually relies on.
        env = {k: v for k, v in os.environ.items() if k not in ("CONTINUUM_DB_DSN", "DATABASE_URL")}
        if dsn:
            env["CONTINUUM_DB_DSN"] = dsn
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        self._id = 0

    def call(self, method: str, params: dict[str, Any]) -> float:
        """Send one request, wait for its reply, return elapsed seconds."""
        assert self.proc.stdin and self.proc.stdout
        self._id += 1
        mid = self._id
        t0 = time.perf_counter()
        self.proc.stdin.write(
            json.dumps({"jsonrpc": "2.0", "id": mid, "method": method, "params": params}) + "\n"
        )
        self.proc.stdin.flush()
        deadline = t0 + TIMEOUT_S
        while time.perf_counter() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("server closed the stream")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == mid:
                return time.perf_counter() - t0
        raise TimeoutError(f"no reply to {method} within {TIMEOUT_S}s")

    def notify(self, method: str) -> None:
        assert self.proc.stdin
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
        self.proc.stdin.flush()

    def tool(self, name: str, args: dict[str, Any]) -> float:
        return self.call("tools/call", {"name": name, "arguments": args})

    def close(self) -> None:
        self.proc.kill()


def _stats(samples_ms: list[float]) -> tuple[float, float, float]:
    s = sorted(samples_ms)
    p95 = s[max(0, int(0.95 * len(s)) - 1)]
    return statistics.median(s), p95, statistics.fmean(s)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="mcp-bench", description="Continuum MCP latency benchmark")
    p.add_argument("--dsn", default=None, help="throwaway Postgres DSN (default: in-memory)")
    p.add_argument("--n", type=int, default=20, help="iterations per tool (default 20)")
    p.add_argument("--server-cmd", default=None, help="path to continuum-mcp (default: auto)")
    args = p.parse_args(argv)

    cmd = [args.server_cmd] if args.server_cmd else [shutil.which("continuum-mcp") or ""]
    if not cmd[0]:
        cmd = [sys.executable, "-m", "continuum.mcp.server"]

    backend = f"postgres ({args.dsn.rsplit('/', 1)[-1]})" if args.dsn else "in-memory"
    print(f"[mcp-bench] server : {' '.join(cmd)}")
    print(f"[mcp-bench] backend: {backend}   (live CONTINUUM_DB_DSN is never inherited)")
    print(f"[mcp-bench] n      : {args.n} per tool\n")

    srv = _Server(cmd, args.dsn)
    try:
        srv.call(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "mcp-bench", "version": "0"},
            },
        )
        srv.notify("notifications/initialized")

        cold = srv.tool("remember", {"text": "warmup fact zero", "attribute": "warmup"})
        print(f"cold start (first call: pool open + embedder load) = {cold * 1000:,.0f} ms\n")

        n = args.n
        measured = {
            "remember": [srv.tool("remember", {"text": f"benchmark fact {i}"}) for i in range(n)],
            # Distinct from the remembered texts on purpose: reusing them would
            # hit the embedding cache those writes just populated and report a
            # ~10x optimistic recall latency.
            "recall": [
                srv.tool("recall", {"query": f"cold retrieval probe {i}", "k": 3}) for i in range(n)
            ],
            "current": [
                srv.tool("current", {"subject": "user", "attribute": "warmup"}) for _ in range(n)
            ],
            "timeline": [srv.tool("timeline", {"entity": "benchmark"}) for _ in range(n)],
        }
        print(f"{'tool':10s} {'p50':>9s} {'p95':>9s} {'mean':>9s}   (warm)")
        for name, samples in measured.items():
            p50, p95, mean = _stats([s * 1000 for s in samples])
            print(f"{name:10s} {p50:7.1f}ms {p95:7.1f}ms {mean:7.1f}ms")

        # Where does the time actually go? A unique query embeds; a repeated one
        # hits the in-process embedding cache. The delta IS the forward pass.
        uniq = [srv.tool("recall", {"query": f"unique probe {i}", "k": 3}) * 1000 for i in range(n)]
        same = [srv.tool("recall", {"query": "identical probe", "k": 3}) * 1000 for _ in range(n)]
        u50, s50 = statistics.median(uniq), statistics.median(same)
        print("\nwhere the time goes:")
        print(f"  recall, unique query each time (embedder runs) : {u50:6.1f} ms")
        print(f"  recall, same query repeated  (embed cache hit) : {s50:6.1f} ms")
        print(f"  -> embedding forward pass                      : ~{u50 - s50:.0f} ms")
        print("  -> `current` skips the model entirely, which is why it is ~50x faster")
    finally:
        srv.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (TimeoutError, RuntimeError) as exc:
        print(f"[mcp-bench] ✗ {exc}")
        raise SystemExit(1) from None
