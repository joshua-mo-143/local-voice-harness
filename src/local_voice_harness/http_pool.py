"""Process-level keep-alive HTTPS for Venice LLM and TTS requests."""

from __future__ import annotations

import http.client
import threading
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Callable
from typing import Any

_LOCK = threading.Lock()
_IDLE: dict[tuple[str, int, str], list[http.client.HTTPConnection]] = defaultdict(list)

_STALE = (
    BrokenPipeError,
    ConnectionResetError,
    http.client.BadStatusLine,
    http.client.RemoteDisconnected,
)


def clear() -> None:
    """Close idle pooled connections. Tests use this to isolate cases."""
    with _LOCK:
        for connections in _IDLE.values():
            for connection in connections:
                connection.close()
        _IDLE.clear()


def _origin(url: str) -> tuple[str, int, str]:
    parsed = urllib.parse.urlsplit(url)
    scheme = parsed.scheme or "https"
    host = parsed.hostname or ""
    if parsed.port is not None:
        port = parsed.port
    else:
        port = 443 if scheme == "https" else 80
    return scheme, port, host


def _checkout(
    origin: tuple[str, int, str],
    conn_class: Callable[..., http.client.HTTPConnection],
    timeout: Any,
) -> tuple[http.client.HTTPConnection, bool]:
    with _LOCK:
        idle = _IDLE[origin]
        connection = idle.pop() if idle else None
    if connection is None:
        _scheme, port, host = origin
        return conn_class(host, port=port, timeout=timeout), False
    connection.timeout = timeout
    return connection, True


def _release(
    origin: tuple[str, int, str], connection: http.client.HTTPConnection
) -> None:
    with _LOCK:
        _IDLE[origin].append(connection)


class _PooledResponse:
    def __init__(
        self,
        connection: http.client.HTTPConnection,
        response: http.client.HTTPResponse,
        origin: tuple[str, int, str],
    ) -> None:
        self._connection = connection
        self._response = response
        self._origin = origin
        self.status = response.status
        self.code = response.status
        self.reason = response.reason
        self.headers = response.headers
        self.msg = response.reason
        self.url = ""
        self._closed = False

    def read(self, amt: int | None = None) -> bytes:
        return self._response.read(amt)

    def readline(self, limit: int = -1) -> bytes:
        return self._response.readline(limit)

    def __iter__(self) -> _PooledResponse:
        return self

    def __next__(self) -> bytes:
        line = self._response.readline()
        if not line:
            raise StopIteration
        return line

    def getcode(self) -> int:
        return self.status

    def geturl(self) -> str:
        return self.url

    def info(self) -> Any:
        return self.headers

    def __enter__(self) -> _PooledResponse:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if not self._response.isclosed():
                self._response.read()
            if self._response.will_close:
                self._connection.close()
            else:
                _release(self._origin, self._connection)
        except Exception:
            self._connection.close()


class PooledHTTPSHandler(urllib.request.HTTPSHandler):
    def http_open(self, req: urllib.request.Request) -> Any:
        return self._pooled_open(req, http.client.HTTPConnection)

    def https_open(self, req: urllib.request.Request) -> Any:  # type: ignore[override]
        return self._pooled_open(req, http.client.HTTPSConnection)

    def _pooled_open(
        self,
        req: urllib.request.Request,
        conn_class: Callable[..., http.client.HTTPConnection],
    ) -> _PooledResponse:
        origin = _origin(req.full_url)
        timeout = req.timeout
        headers = dict(req.unredirected_hdrs)
        headers.update(
            {key: value for key, value in req.headers.items() if key not in headers}
        )
        headers = {key.title(): value for key, value in headers.items()}
        headers["Connection"] = "keep-alive"
        method = req.get_method()
        selector = req.selector
        data = req.data
        connection, reused = _checkout(origin, conn_class, timeout)
        try:
            try:
                connection.request(method, selector, data, headers)
                response = connection.getresponse()
            except _STALE:
                connection.close()
                if not reused:
                    raise
                connection, _reused = _checkout(origin, conn_class, timeout)
                connection.request(method, selector, data, headers)
                response = connection.getresponse()
        except Exception:
            connection.close()
            raise
        wrapped = _PooledResponse(connection, response, origin)
        wrapped.url = req.full_url
        return wrapped


def install() -> None:
    urllib.request.install_opener(urllib.request.build_opener(PooledHTTPSHandler))


install()
