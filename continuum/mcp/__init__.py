"""Continuum MCP server — expose memory as tools for any MCP client.

    continuum-mcp        # run over stdio

See :mod:`continuum.mcp.server`.
"""

from __future__ import annotations

from continuum.mcp.server import build_server, main

__all__ = ["build_server", "main"]
