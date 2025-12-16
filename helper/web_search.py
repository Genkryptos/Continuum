"""
Lightweight web search + fetch utilities shared by the MCP server and agents.

The helpers favor defensive defaults (network opt-in, bounded result counts)
so they remain safe to import in offline or test environments.
"""

from dataclasses import dataclass
import logging
import re
import warnings
from typing import Callable, Dict, List, Optional

import requests

# Prefer the renamed package `ddgs` but fall back to the legacy name. Suppress
# the deprecation warning emitted by older duckduckgo_search releases.
DDGS = None
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    try:  # pragma: no cover - optional dependency
        from ddgs import DDGS  # type: ignore
    except Exception:  # noqa: BLE001  # pragma: no cover
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except Exception:  # noqa: BLE001  # pragma: no cover
            DDGS = None

logger = logging.getLogger(__name__)


class WebSearchError(Exception):
    """Raised when a live web lookup fails."""


@dataclass
class WebSearchResult:
    """Structured representation of a single web search hit."""

    title: str
    url: str
    snippet: str = ""

    def as_dict(self) -> Dict[str, str]:
        """Return a plain dict payload suitable for serialization."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
        }


class WebSearchService:
    """
    Fetch live web snippets using DuckDuckGo and optionally pull page bodies.

    A slim interface is intentional so agents can inject a stub implementation
    in unit tests or disable network access entirely.
    """

    def __init__(
        self,
        max_results: int = 3,
        timeout: float = 10.0,
        allow_network: bool = True,
        http_get: Optional[Callable[..., object]] = None,
    ) -> None:
        self.max_results = max_results
        self.timeout = timeout
        self.allow_network = allow_network
        self.http_get = http_get or requests.get
        logging.getLogger("primp").setLevel(logging.WARNING)
        logging.getLogger("ddgs.ddgs").setLevel(logging.WARNING)

    def search(self, query: str, *, max_results: Optional[int] = None) -> List[WebSearchResult]:
        """
        Run a DuckDuckGo text search and return structured hits.

        The call is guarded by ``allow_network`` and will raise WebSearchError
        on failures so callers can fall back gracefully.
        """
        if not self.allow_network:
            logger.info("Web search skipped (network disabled)")
            return []

        limit = max_results or self.max_results
        if limit <= 0:
            return []

        if DDGS is None:
            raise WebSearchError(
                "duckduckgo_search is not installed. Install it or disable web search."
            )

        try:
            with DDGS(timeout=self.timeout) as ddg:
                raw_results = ddg.text(query, max_results=limit)
        except Exception as exc:  # noqa: BLE001
            raise WebSearchError(f"Web search failed: {exc}") from exc

        results: List[WebSearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("href") or item.get("url") or "").strip()
            snippet = str(item.get("body") or item.get("snippet") or "").strip()
            results.append(WebSearchResult(title=title, url=url, snippet=self._clean(snippet)))
        return results

    def fetch_url(self, url: str, *, max_chars: Optional[int] = None) -> str:
        """Return a cleaned, plain-text body for the provided URL."""
        if not self.allow_network:
            raise WebSearchError("Network access is disabled for web fetches.")

        try:
            resp = self.http_get(url, timeout=self.timeout)
            # requests compat: both the real library and the lightweight shim expose raise_for_status/text
            if hasattr(resp, "raise_for_status"):
                resp.raise_for_status()
            text = getattr(resp, "text", "") or ""
        except Exception as exc:  # noqa: BLE001
            raise WebSearchError(f"Failed to fetch url {url}: {exc}") from exc

        cleaned = self._clean(text)
        if max_chars and max_chars > 0:
            return cleaned[:max_chars]
        return cleaned

    def format_results(self, results: List[WebSearchResult]) -> str:
        """Render results into a compact bullet list for prompt inclusion."""
        lines = []
        for idx, item in enumerate(results, start=1):
            title = item.title or "Untitled result"
            snippet = item.snippet or ""
            lines.append(f"[{idx}] {title} — {snippet} ({item.url})")
        return "\n".join(lines)

    def _clean(self, text: str) -> str:
        """Collapse whitespace so we feed models compact context."""
        return re.sub(r"\s+", " ", text or "").strip()
