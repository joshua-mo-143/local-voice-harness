from __future__ import annotations

import multiprocessing
import socket
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import TracebackType
from typing import Any, Generic, TypeVar

T = TypeVar("T")
_CONTEXT = multiprocessing.get_context("fork")
_STOP_TIMEOUT = 0.25


@dataclass(frozen=True)
class ConcurrentOutcomes(Generic[T]):
    values: tuple[T | None, ...]
    errors: tuple[BaseException | None, ...]


def _stop_process(process: BaseProcess, *, timeout: float = _STOP_TIMEOUT) -> None:
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout)
    if process.is_alive():
        process.kill()
        process.join(timeout)
    if process.is_alive():
        raise RuntimeError(f"could not stop test process {process.pid}")


def _run_call(call: Callable[[], object], start: Any, status: Connection) -> None:
    try:
        status.send(("ready", None))
        start.wait()
        try:
            status.send(("value", call()))
        except BaseException as exc:
            try:
                status.send(("error", exc))
            except BaseException:
                status.send(("error", RuntimeError(repr(exc))))
    finally:
        status.close()


def run_concurrently(
    calls: Sequence[Callable[[], T]], *, timeout: float = 2
) -> ConcurrentOutcomes[T]:
    """Run calls together in bounded, killable Linux subprocesses."""

    if not calls:
        return ConcurrentOutcomes((), ())
    start = _CONTEXT.Event()
    values: list[T | None] = [None] * len(calls)
    errors: list[BaseException | None] = [None] * len(calls)
    processes: list[BaseProcess] = []
    statuses: list[Connection] = []
    for call in calls:
        parent, child = _CONTEXT.Pipe(duplex=False)
        process = _CONTEXT.Process(
            target=_run_call,
            args=(call, start, child),
            name="concurrent-test-call",
        )
        process.start()
        child.close()
        processes.append(process)
        statuses.append(parent)

    deadline = time.monotonic() + timeout
    try:
        for status in statuses:
            if not status.poll(max(0, deadline - time.monotonic())):
                raise TimeoutError("concurrent call did not become ready")
            kind, _ = status.recv()
            if kind != "ready":
                raise RuntimeError("concurrent call sent an invalid ready message")
        start.set()
        for index, status in enumerate(statuses):
            if not status.poll(max(0, deadline - time.monotonic())):
                raise TimeoutError("concurrent calls did not finish")
            kind, value = status.recv()
            if kind == "value":
                values[index] = value
            elif kind == "error" and isinstance(value, BaseException):
                errors[index] = value
            else:
                raise RuntimeError("concurrent call sent an invalid result")
        return ConcurrentOutcomes(tuple(values), tuple(errors))
    finally:
        start.set()
        for process in processes:
            _stop_process(process)
            process.close()
        for status in statuses:
            status.close()


def _serve_unix_socket(
    path: str,
    handler: Callable[[socket.socket], None],
    stop: Any,
    status: Connection,
) -> None:
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection: socket.socket | None = None
    try:
        listener.bind(path)
        listener.listen(1)
        listener.settimeout(0.05)
        status.send(("ready", None))
        while not stop.is_set():
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            handler(connection)
            break
    except BaseException as exc:
        try:
            status.send(("error", repr(exc)))
        except BaseException:
            pass
    finally:
        if connection is not None:
            connection.close()
        listener.close()
        status.close()


class UnixSocketServer:
    """One-shot Unix fixture isolated in a bounded, killable subprocess."""

    def __init__(
        self,
        handler: Callable[[socket.socket], None],
        *,
        shutdown_timeout: float = _STOP_TIMEOUT,
    ) -> None:
        self.handler = handler
        self.shutdown_timeout = shutdown_timeout
        self._temporary = tempfile.TemporaryDirectory()
        self.path = Path(self._temporary.name) / "fake.sock"
        self._process: BaseProcess | None = None
        self._stop: Any = None
        self._status: Connection | None = None

    def __enter__(self) -> UnixSocketServer:
        stop = _CONTEXT.Event()
        parent, child = _CONTEXT.Pipe(duplex=False)
        process = _CONTEXT.Process(
            target=_serve_unix_socket,
            args=(str(self.path), self.handler, stop, child),
            name="fake-unix-server",
        )
        process.start()
        child.close()
        self._process = process
        self._stop = stop
        self._status = parent
        if not parent.poll(2):
            self._cleanup()
            raise TimeoutError("fake Unix server did not become ready")
        kind, detail = parent.recv()
        if kind != "ready":
            self._cleanup()
            raise RuntimeError(f"fake Unix server failed to start: {detail}")
        return self

    def _cleanup(self) -> str | None:
        error: str | None = None
        if self._stop is not None:
            self._stop.set()
        if self._process is not None:
            _stop_process(self._process, timeout=self.shutdown_timeout)
            self._process.close()
            self._process = None
        if self._status is not None:
            if self._status.poll():
                try:
                    kind, detail = self._status.recv()
                except EOFError:
                    pass
                else:
                    if kind == "error":
                        error = str(detail)
            self._status.close()
            self._status = None
        self._temporary.cleanup()
        return error

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        error = self._cleanup()
        if exception_type is None and error is not None:
            raise RuntimeError(f"fake Unix server handler failed: {error}")


def receive_all(connection: socket.socket, *, chunk_size: int = 64 * 1024) -> bytes:
    chunks: list[bytes] = []
    while chunk := connection.recv(chunk_size):
        chunks.append(chunk)
    return b"".join(chunks)
