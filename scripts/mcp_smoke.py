#!/usr/bin/env python3
"""
scripts/mcp_smoke.py
====================
Standalone smoke test for the Continuum MCP server — proves it works with **zero
Claude involvement**. Spawns `continuum-mcp` (stdio), performs the MCP handshake,
lists the tools, then does a `remember` → `recall` round-trip and asserts the
stored fact comes back.

This is the honest answer to "is the MCP server actually working?" — it exercises
the same binary Claude Code spawns, but from a plain subprocess we control.

Exit code 0 = healthy; non-zero = something is broken (prints why).

    python3 scripts/mcp_smoke.py            # auto-locates continuum-mcp
    python3 scripts/mcp_smoke.py /path/to/continuum-mcp
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

_MARKER = "continuum-mcp-smoke: The launch date is 2026-07-19"


def _server_cmd(argv: list[str]) -> list[str]:
    """continuum-mcp if on PATH / given explicitly, else `python -m` fallback."""
    if len(argv) > 1:
        return [argv[1]]
    found = shutil.which("continuum-mcp")
    if found:
        return [found]
    return [sys.executable, "-m", "continuum.mcp.server"]


def _rpc(mid: int, method: str, params: dict | None = None) -> str:
    msg: dict = {"jsonrpc": "2.0", "method": method}
    if mid:
        msg["id"] = mid
    if params is not None:
        msg["params"] = params
    return json.dumps(msg)


def main(argv: list[str]) -> int:
    cmd = _server_cmd(argv)
    requests = "\n".join(
        [
            _rpc(
                1,
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "mcp-smoke", "version": "0"},
                },
            ),
            _rpc(0, "notifications/initialized"),
            _rpc(2, "tools/list", {}),
            _rpc(3, "tools/call", {"name": "remember", "arguments": {"text": _MARKER}}),
            _rpc(
                4, "tools/call", {"name": "recall", "arguments": {"query": "launch date", "k": 3}}
            ),
        ]
    )

    print(f"[mcp-smoke] spawning: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(
        cmd,
        input=requests + "\n",
        capture_output=True,
        text=True,
        timeout=60,
    )

    replies: dict[int, dict] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(m, dict) and isinstance(m.get("id"), int):
            replies[m["id"]] = m

    # ── assertions ────────────────────────────────────────────────────────────
    ok = True

    tools = None
    if 2 in replies and "result" in replies[2]:
        tools = sorted(t["name"] for t in replies[2]["result"].get("tools", []))
    expected = ["current", "recall", "remember", "timeline"]
    if tools == expected:
        print(f"[mcp-smoke] ✓ tools exposed: {tools}")
    else:
        print(f"[mcp-smoke] ✗ tools/list mismatch: got {tools}, want {expected}")
        ok = False

    if 3 not in replies:
        print("[mcp-smoke] ✗ no remember reply")
        ok = False
    elif replies[3].get("result", {}).get("isError"):
        print("[mcp-smoke] ✗ remember returned isError")
        ok = False
    else:
        print("[mcp-smoke] ✓ remember stored a fact")

    recalled: list[str] = []
    if 4 in replies and "result" in replies[4]:
        structured = replies[4]["result"].get("structuredContent", {})
        recalled = [d.get("content", "") for d in structured.get("result", [])]
    if any(_MARKER in c for c in recalled):
        print(f"[mcp-smoke] ✓ recall round-trip returned the stored fact ({len(recalled)} hit(s))")
    else:
        print(f"[mcp-smoke] ✗ recall did not return the stored fact; got: {recalled}")
        ok = False

    if not ok and proc.stderr:
        print("\n[mcp-smoke] server stderr (tail):", file=sys.stderr)
        print("\n".join(proc.stderr.splitlines()[-10:]), file=sys.stderr)

    print(f"\n[mcp-smoke] {'PASS ✓' if ok else 'FAIL ✗'}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except subprocess.TimeoutExpired:
        print("[mcp-smoke] ✗ server did not respond within 60s")
        raise SystemExit(1) from None
