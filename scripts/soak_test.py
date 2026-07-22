#!/usr/bin/env python3
"""
scripts/soak_test.py
====================
Run the Continuum HTTP server for hours and watch for the failures that only
show up slowly: memory that climbs, file descriptors that leak, a connection
pool that grows without bound, latency that drifts.

Every other test in this project is a burst — nothing had run longer than about
eight minutes. ``continuum-mcp --http`` is a **daemon**, and daemons fail
quietly at 3am rather than loudly in CI.

    python3 scripts/soak_test.py --hours 6 --dsn postgresql://…/throwaway

It drives a realistic mix (``remember`` / ``recall`` / ``current``) at a modest
rate, samples the server process every minute, and — with ``--restart-container``
— restarts Postgres at the midpoint to prove the pool recovers on its own.

Point it at a **throwaway** database: it writes. It never restarts anything you
did not name explicitly.

Exit 0 if every check holds, 1 otherwise, so it can gate a release.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

FACTS = [
    "I live in Porto.",
    "I work at Nimbus Data on the platform team.",
    "My laptop is an M3 MacBook Air.",
    "I am allergic to peanuts.",
    "I prefer pytest over unittest.",
]
QUERIES = ["where do I live?", "who do I work for?", "what am I allergic to?", "what laptop?"]


@dataclass
class Sample:
    t: float
    rss_mb: float
    fds: int
    pg_conns: int
    p95_ms: float


@dataclass
class Soak:
    url: str
    dsn: str
    pid: int
    samples: list[Sample] = field(default_factory=list)
    errors: int = 0
    calls: int = 0
    session_id: str | None = None

    # ── MCP over Streamable-HTTP ──────────────────────────────────────────────

    def _post(self, payload: dict, timeout: float = 60.0) -> dict | None:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        req = urllib.request.Request(self.url, json.dumps(payload).encode(), headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if not self.session_id:
                self.session_id = r.headers.get("mcp-session-id")
            body = r.read().decode()
        for line in body.splitlines():
            if line.startswith("data: "):
                return json.loads(line[6:])
        return None

    def handshake(self) -> None:
        self.session_id = None
        self._post(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "soak", "version": "0"},
                },
            }
        )
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def tool(self, name: str, **args: object) -> float:
        """Call a tool; return latency in ms. Counts errors rather than raising —
        a soak that dies on the first blip measures nothing."""
        started = time.perf_counter()
        try:
            reply = self._post(
                {
                    "jsonrpc": "2.0",
                    "id": self.calls + 100,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": args},
                }
            )
            if reply is None or (reply.get("result") or {}).get("isError"):
                self.errors += 1
        except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
            self.errors += 1
            self.session_id = None  # force a fresh handshake next tick
        self.calls += 1
        return (time.perf_counter() - started) * 1000

    # ── process + database observation ────────────────────────────────────────

    def rss_mb(self) -> float:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(self.pid)], capture_output=True, text=True
        ).stdout.strip()
        return int(out) / 1024 if out.isdigit() else 0.0

    def fd_count(self) -> int:
        r = subprocess.run(["lsof", "-p", str(self.pid)], capture_output=True, text=True)
        return max(0, len(r.stdout.splitlines()) - 1)

    def pg_conns(self) -> int:
        try:
            import psycopg

            with psycopg.connect(self.dsn, connect_timeout=5) as c, c.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()"
                )
                row = cur.fetchone()
                return int(row[0]) if row else 0
        except Exception:
            return -1

    def alive(self) -> bool:
        return subprocess.run(["kill", "-0", str(self.pid)], capture_output=True).returncode == 0


def _verdict(s: Soak, restarted_at: float | None) -> int:
    """Report, and decide. Growth is judged against the FIRST sample after
    warm-up, not the very first — a young process is still filling caches."""
    if len(s.samples) < 3:
        print("\n[soak] too few samples to judge", file=sys.stderr)
        return 1

    warm = s.samples[1]
    last = s.samples[-1]
    hours = (last.t - warm.t) / 3600 or 1e-9

    rss_growth = (last.rss_mb - warm.rss_mb) / max(warm.rss_mb, 1) * 100
    fd_growth = last.fds - warm.fds
    early = [x.p95_ms for x in s.samples[1 : max(2, len(s.samples) // 4)]]
    late = [x.p95_ms for x in s.samples[-max(2, len(s.samples) // 4) :]]
    lat_drift = (statistics.median(late) / max(statistics.median(early), 1e-9) - 1) * 100

    print("\n" + "=" * 66)
    print(f"  soak: {hours:.2f}h, {s.calls} calls, {s.errors} errors")
    print(f"  RSS      {warm.rss_mb:7.1f} MB -> {last.rss_mb:7.1f} MB   ({rss_growth:+.1f}%)")
    print(f"  fds      {warm.fds:7d}    -> {last.fds:7d}      ({fd_growth:+d})")
    print(f"  pg conns {warm.pg_conns:7d}    -> {last.pg_conns:7d}")
    print(
        f"  p95      {statistics.median(early):7.0f} ms -> {statistics.median(late):7.0f} ms"
        f"   ({lat_drift:+.1f}%)"
    )
    if restarted_at:
        print("  postgres was restarted mid-run")

    checks = [
        ("server still alive", s.alive()),
        ("RSS growth < 25%", rss_growth < 25),
        ("fd growth < 50", fd_growth < 50),
        ("pool bounded (<= 25 conns)", last.pg_conns <= 25),
        ("p95 drift < 50%", lat_drift < 50),
        ("error rate < 2%", s.errors <= max(1, s.calls * 0.02)),
    ]
    print()
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    passed = all(ok for _, ok in checks)
    print(f"\n[soak] {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--url", default="http://127.0.0.1:8799/mcp")
    p.add_argument("--dsn", default=os.environ.get("CONTINUUM_DB_DSN", ""))
    p.add_argument("--pid", type=int, required=True, help="pid of the continuum-mcp server")
    p.add_argument("--rate", type=float, default=4.0, help="calls per minute")
    p.add_argument("--restart-container", default=None, help="docker container to restart midway")
    p.add_argument("--out", default=None, help="write samples as JSONL here")
    args = p.parse_args(argv)

    s = Soak(url=args.url, dsn=args.dsn, pid=args.pid)
    s.handshake()

    deadline = time.time() + args.hours * 3600
    midpoint = time.time() + args.hours * 1800
    restarted_at: float | None = None
    interval = 60.0 / max(args.rate, 0.1)
    next_sample = time.time() + 60
    window: list[float] = []
    i = 0

    while time.time() < deadline:
        if not s.alive():
            print("[soak] server process died", file=sys.stderr)
            break
        if s.session_id is None:
            # Re-handshake after a blip (e.g. the mid-run Postgres restart).
            with contextlib.suppress(Exception):
                s.handshake()

        i += 1
        if i % 4 == 0:
            window.append(s.tool("remember", text=FACTS[i % len(FACTS)]))
        elif i % 4 == 3:
            window.append(s.tool("current", subject="user", attribute="residence"))
        else:
            window.append(s.tool("recall", query=QUERIES[i % len(QUERIES)], k=8))

        if args.restart_container and restarted_at is None and time.time() >= midpoint:
            print("[soak] restarting postgres…", flush=True)
            subprocess.run(["docker", "restart", args.restart_container], capture_output=True)
            restarted_at = time.time()
            time.sleep(10)

        if time.time() >= next_sample:
            ordered = sorted(window) or [0.0]
            sample = Sample(
                t=time.time(),
                rss_mb=s.rss_mb(),
                fds=s.fd_count(),
                pg_conns=s.pg_conns(),
                p95_ms=ordered[int(len(ordered) * 0.95) - 1] if len(ordered) > 1 else ordered[0],
            )
            s.samples.append(sample)
            print(
                f"[soak] +{(sample.t - s.samples[0].t) / 60:6.1f}m  rss={sample.rss_mb:7.1f}MB "
                f"fds={sample.fds:4d} conns={sample.pg_conns:3d} p95={sample.p95_ms:6.0f}ms "
                f"calls={s.calls} errors={s.errors}",
                flush=True,
            )
            if args.out:
                with open(args.out, "a") as fh:
                    fh.write(json.dumps(sample.__dict__) + "\n")
            window.clear()
            next_sample = time.time() + 60

        time.sleep(max(0.0, interval))

    return _verdict(s, restarted_at)


if __name__ == "__main__":
    raise SystemExit(main())
