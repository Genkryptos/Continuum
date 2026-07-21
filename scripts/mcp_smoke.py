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
import threading

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


def _exchange(cmd: list[str], requests: list[str], want: set[int], timeout: float = 60.0):
    """Speak stdio JSON-RPC to `cmd` and return ({id: reply}, stderr).

    Reads replies as they arrive and only then closes the session, the way a real
    client does. Do NOT write everything and wait for EOF: mcp>=1.28's stdio
    server drops in-flight replies when stdin closes, so the last call of the
    run silently comes back missing — which looked exactly like a broken recall.
    """
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    assert proc.stdin and proc.stdout and proc.stderr
    err: list[str] = []
    threading.Thread(target=lambda: err.append(proc.stderr.read()), daemon=True).start()

    replies: dict[int, dict] = {}
    reading_done = threading.Event()

    def _read() -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            try:
                m = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if isinstance(m, dict) and isinstance(m.get("id"), int):
                replies[m["id"]] = m
                if want <= replies.keys():
                    reading_done.set()
                    return
        reading_done.set()

    reader = threading.Thread(target=_read, daemon=True)
    reader.start()
    try:
        proc.stdin.write("\n".join(requests) + "\n")
        proc.stdin.flush()
        reading_done.wait(timeout)
    finally:
        proc.stdin.close()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    reader.join(timeout=5)
    return replies, "".join(err)


def main(argv: list[str]) -> int:
    cmd = _server_cmd(argv)
    requests = [
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
        _rpc(4, "tools/call", {"name": "recall", "arguments": {"query": "launch date", "k": 3}}),
    ]

    print(f"[mcp-smoke] spawning: {' '.join(cmd)}", file=sys.stderr)
    replies, stderr = _exchange(cmd, requests, want={1, 2, 3, 4})

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

    if not ok and stderr:
        print("\n[mcp-smoke] server stderr (tail):", file=sys.stderr)
        print("\n".join(stderr.splitlines()[-10:]), file=sys.stderr)

    print(f"\n[mcp-smoke] {'PASS ✓' if ok else 'FAIL ✗'}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except subprocess.TimeoutExpired:
        print("[mcp-smoke] ✗ server did not respond within 60s")
        raise SystemExit(1) from None
