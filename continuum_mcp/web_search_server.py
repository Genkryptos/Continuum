"""
Model Context Protocol server that exposes live web search + fetch tools.

Run this module directly to start the server:

    python -m continuum_mcp.web_search_server

Environment knobs:
  - WEB_SEARCH_MAX_RESULTS: default number of search hits to return (3)
  - WEB_SEARCH_TIMEOUT: HTTP/search timeout in seconds (10)
  - WEB_SEARCH_ALLOW_NETWORK: set to "0" to disable network access
"""

import logging
import os
from typing import Annotated, Dict, List

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "The 'mcp' package is required for the web search server. "
        "Install with `pip install mcp` (or `pip install -r requirements.txt`)."
    ) from exc
from helper.web_search import WebSearchError, WebSearchResult, WebSearchService


def _env_bool(name: str, default: str = "1") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value not in {"0", "false", "no", "off", ""}


def build_server() -> FastMCP:
    """Construct a FastMCP server wired to the shared WebSearchService."""
    max_results = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "3"))
    timeout = float(os.getenv("WEB_SEARCH_TIMEOUT", "10"))
    allow_network = _env_bool("WEB_SEARCH_ALLOW_NETWORK", "1")

    web = WebSearchService(
        max_results=max_results,
        timeout=timeout,
        allow_network=allow_network,
    )

    server = FastMCP("continuum-web-search")

    @server.tool()
    def search_web(
        query: Annotated[str, "Natural language query to search the live web."],
        max_results_override: Annotated[
            int, "Override the default number of results for this request."
        ] = max_results,
    ) -> List[Dict[str, str]]:
        """Search the web and return structured results with titles, urls and snippets."""
        try:
            results: List[WebSearchResult] = web.search(
                query, max_results=max_results_override
            )
        except WebSearchError as exc:
            raise RuntimeError(str(exc)) from exc
        return [item.as_dict() for item in results]

    @server.tool()
    def fetch_page(
        url: Annotated[str, "HTTP or HTTPS URL to fetch"],
        max_chars: Annotated[int, "Truncate the body to this many characters"] = 4000,
    ) -> str:
        """Fetch a page body (lightly cleaned) for grounding."""
        try:
            return web.fetch_url(url, max_chars=max_chars)
        except WebSearchError as exc:
            raise RuntimeError(str(exc)) from exc

    return server


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    server = build_server()
    server.run()


if __name__ == "__main__":
    main()
