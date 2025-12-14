"""
Tiny requests-style wrapper built on urllib, used to avoid pulling in requests.

Only the minimal surface area needed by the project is implemented; callers
should expect limited features compared to the real ``requests`` library.
"""

import json as _json
import urllib.error
import urllib.request
from typing import Any, Optional
import ssl
import certifi


class Timeout(Exception):
    """Raised when an HTTP operation exceeds the provided timeout."""
    pass


class ConnectionError(Exception):
    """Raised when a network failure occurs before a response is returned."""
    pass


class HTTPError(Exception):
    def __init__(self, response: "Response") -> None:
        super().__init__(f"HTTP {response.status_code}")
        self.response = response


class Response:
    def __init__(self, status_code: int, reason: str, content: bytes, encoding: str = "utf-8") -> None:
        self.status_code = status_code
        self.reason = reason
        self.content = content
        self.text = content.decode(encoding, errors="ignore") if content is not None else ""
    def json(self) -> Any:
        return _json.loads(self.text or "null")

    def raise_for_status(self) -> None:
        if 400 <= self.status_code:
            raise HTTPError(self)


_DEF_TIMEOUT = 30


def _request(method: str, url: str, *, json: Optional[Any] = None, timeout: Optional[float] = None, **kwargs: Any) -> Response:
    """Internal helper mirroring the core requests.request signature."""
    data = None
    headers = kwargs.get("headers", {})
    if json is not None:
        data = _json.dumps(json).encode()
        headers = {**headers, "Content-Type": "application/json"}

    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())

    ssl_context = (
        ssl.create_default_context(cafile=certifi.where()) if certifi else ssl.create_default_context()
    )

    try:
       with urllib.request.urlopen(req, timeout=timeout or _DEF_TIMEOUT, context=ssl_context) as resp:
            raw = resp.read()
            return Response(status_code=resp.getcode() or 0, reason=resp.reason, content=raw)
    except urllib.error.HTTPError as exc:  # noqa: PERF203
        body = exc.read()
        response = Response(status_code=exc.code, reason=exc.reason, content=body)
        raise HTTPError(response) from None
    except urllib.error.URLError as exc:  # noqa: PERF203
        if isinstance(exc.reason, TimeoutError):
            raise Timeout(str(exc))
        raise ConnectionError(str(exc))


def get(url: str, *, timeout: Optional[float] = None, **kwargs: Any) -> Response:
    """Perform a GET request and return a lightweight Response object."""
    return _request("GET", url, timeout=timeout, **kwargs)


def post(url: str, *, json: Optional[Any] = None, timeout: Optional[float] = None, **kwargs: Any) -> Response:
    """Perform a POST request with optional JSON payload."""
    return _request("POST", url, json=json, timeout=timeout, **kwargs)
