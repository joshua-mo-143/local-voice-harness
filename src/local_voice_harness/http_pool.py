"""Scoped keep-alive HTTPS transport for Venice requests."""

from __future__ import annotations

import http.client
import threading
import urllib.request
from collections import defaultdict
from collections.abc import Callable
from typing import Any

_LOCK = threading.Lock()
_PoolKey = tuple[str, str, str]
_IDLE: dict[_PoolKey, list[http.client.HTTPConnection]] = defaultdict(list)


def clear() -> None:
    """Close idle pooled connections. Tests use this to isolate cases."""
    with _LOCK:
        for connections in _IDLE.values():
            for connection in connections:
                connection.close()
        _IDLE.clear()


def _pool_key(req: urllib.request.Request) -> _PoolKey:
    return (req.type, req.host, str(getattr(req, "_tunnel_host", "") or ""))


def _checkout(
    key: _PoolKey,
    conn_class: Callable[..., http.client.HTTPConnection],
    timeout: Any,
) -> tuple[http.client.HTTPConnection, bool]:
    with _LOCK:
        idle = _IDLE[key]
        connection = idle.pop() if idle else None
    if connection is None:
        return conn_class(key[1], timeout=timeout), False
    connection.timeout = timeout
    if connection.sock is not None:
        connection.sock.settimeout(timeout)
    return connection, True


def _release(key: _PoolKey, connection: http.client.HTTPConnection) -> None:
    with _LOCK:
        _IDLE[key].append(connection)


class _PooledResponse:
    def __init__(
        self,
        connection: http.client.HTTPConnection,
        response: http.client.HTTPResponse,
        key: _PoolKey,
    ) -> None:
        self._connection = connection
        self._response = response
        self._key = key
        self.status = response.status
        self.code = response.status
        self.reason = response.reason
        self.headers = response.headers
        self.msg = response.reason
        self.url = ""
        self._closed = False
        self._complete = False

    def read(self, amt: int | None = None) -> bytes:
        data = self._response.read(amt)
        if amt is None or len(data) < amt:
            self._complete = True
        return data

    def readline(self, limit: int = -1) -> bytes:
        line = self._response.readline(limit)
        if not line:
            self._complete = True
        return line

    def __iter__(self) -> _PooledResponse:
        return self

    def __next__(self) -> bytes:
        line = self.readline()
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
            if not self._complete or self._response.will_close:
                self._connection.close()
            else:
                _release(self._key, self._connection)
        except Exception:
            self._connection.close()
        finally:
            self._response.close()


class PooledHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req: urllib.request.Request) -> Any:  # type: ignore[override]
        return self._pooled_open(req, http.client.HTTPSConnection)

    def _pooled_open(
        self,
        req: urllib.request.Request,
        conn_class: Callable[..., http.client.HTTPConnection],
    ) -> _PooledResponse:
        key = _pool_key(req)
        timeout = getattr(req, "timeout", None)
        headers = dict(req.unredirected_hdrs)
        headers.update(
            {key: value for key, value in req.headers.items() if key not in headers}
        )
        headers = {name.title(): value for name, value in headers.items()}
        headers["Connection"] = "keep-alive"
        tunnel_headers: dict[str, str] = {}
        tunnel_host = str(getattr(req, "_tunnel_host", "") or "")
        if tunnel_host:
            proxy_authorization = headers.pop("Proxy-Authorization", None)
            if proxy_authorization is not None:
                tunnel_headers["Proxy-Authorization"] = proxy_authorization
        method = req.get_method()
        selector = req.selector
        data = req.data
        connection, reused = _checkout(key, conn_class, timeout)
        if tunnel_host and not reused:
            connection.set_tunnel(tunnel_host, headers=tunnel_headers)
        try:
            connection.request(
                method,
                selector,
                data,
                headers,
                encode_chunked=req.has_header("Transfer-encoding"),
            )
            response = connection.getresponse()
        except Exception:
            connection.close()
            raise
        wrapped = _PooledResponse(connection, response, key)
        wrapped.url = req.full_url
        return wrapped


_OPENER = urllib.request.build_opener(PooledHTTPSHandler)


def urlopen(request: urllib.request.Request, *, timeout: float) -> Any:
    """Open one Venice request without replacing urllib's global opener."""
    return _OPENER.open(request, timeout=timeout)
