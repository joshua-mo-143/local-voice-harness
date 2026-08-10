from __future__ import annotations

import collections
import contextlib
import json
import os
import select
import shutil
import socket
import subprocess
import threading
import time
import uuid
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import HarnessError
from ..ipc import unix_request
from .client import PLAYBACK_LATENCY, playback_slot
from .stream import STREAM_POLL_SECONDS, STREAM_TIMEOUT_SECONDS, TTSStreamParser

PREFETCH_JOIN_SECONDS = 3.0
PREFETCH_CONNECT_SECONDS = 2.0


@dataclass
class PrefetchedUtterance:
    sample_rate: int
    chunks: list[Path]
    chunk_texts: list[str]
    done_meta: dict[str, object] = field(default_factory=dict)
    error: BaseException | None = None

    @property
    def played_text(self) -> str:
        return " ".join(self.chunk_texts).strip()


@dataclass
class PlaybackRequest:
    text: str
    job_id: str | None = None
    delivery_token: str | None = None
    job_status: str | None = None


class PrefetchHandle:
    def __init__(self, text: str) -> None:
        self.text = text
        self.request_id = uuid.uuid4().hex
        self._event = threading.Event()
        self._result: PrefetchedUtterance | None = None
        self._discard_lock = threading.Lock()
        self._discarded = False
        self._cancelled = threading.Event()
        self._cancel_sent = threading.Event()
        self._submitted = threading.Event()
        self._cancel_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._socket: socket.socket | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"voice-prefetch-{self.request_id[:8]}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            self._result = _prefetch_utterance(
                self.text,
                request_id=self.request_id,
                cancelled=self._cancelled,
                register_socket=self._register_socket,
                submit_request=self._submit_request,
                clear_socket=self._clear_socket,
            )
        except BaseException as exc:
            self.cancel()
            self._result = PrefetchedUtterance(
                sample_rate=0,
                chunks=[],
                chunk_texts=[],
                error=exc,
            )
        finally:
            self._event.set()

    def wait(
        self,
        timeout: float = STREAM_TIMEOUT_SECONDS,
        *,
        should_interrupt: Callable[[], bool] | None = None,
        on_poll: Callable[[], None] | None = None,
    ) -> PrefetchedUtterance | None:
        deadline = time.monotonic() + timeout
        if on_poll is not None:
            on_poll()
        while not self._event.wait(
            min(STREAM_POLL_SECONDS, max(0.0, deadline - time.monotonic()))
        ):
            if on_poll is not None:
                on_poll()
            if should_interrupt is not None and should_interrupt():
                self.discard()
                return None
            if time.monotonic() >= deadline:
                self.discard()
                raise HarnessError("TTS prefetch timed out")
        if on_poll is not None:
            on_poll()
        assert self._result is not None
        if self._result.error is not None:
            raise HarnessError(
                f"TTS prefetch failed: {type(self._result.error).__name__}: "
                f"{self._result.error}"
            )
        return self._result

    def _register_socket(self, stream_socket: socket.socket) -> None:
        with self._state_lock:
            self._socket = stream_socket
        if self._cancelled.is_set():
            with contextlib.suppress(OSError):
                stream_socket.shutdown(socket.SHUT_RDWR)

    def _clear_socket(self, stream_socket: socket.socket) -> None:
        with self._state_lock:
            if self._socket is stream_socket:
                self._socket = None

    def _submit_request(self, stream_socket: socket.socket, request: bytes) -> None:
        with self._cancel_lock:
            if self._cancelled.is_set():
                raise HarnessError("TTS prefetch was cancelled")
            self._submitted.set()
            stream_socket.sendall(request)
            stream_socket.shutdown(socket.SHUT_WR)

    def _send_cancel(self) -> None:
        if self._cancel_sent.is_set():
            return
        self._cancel_sent.set()
        from ..config import TTS_SOCKET

        request = (
            json.dumps({"op": "cancel", "request_id": self.request_id}).encode() + b"\n"
        )
        for attempt in range(3):
            try:
                response = json.loads(unix_request(TTS_SOCKET, request, timeout=2))
            except (OSError, ValueError, json.JSONDecodeError):
                return
            if isinstance(response, dict) and response.get("cancelled") is True:
                return
            if attempt < 2:
                time.sleep(STREAM_POLL_SECONDS)

    def cancel(self) -> None:
        with self._cancel_lock:
            if self._cancelled.is_set():
                return
            self._cancelled.set()
            submitted = self._submitted.is_set()
        with self._state_lock:
            stream_socket = self._socket
        if stream_socket is not None:
            with contextlib.suppress(OSError):
                stream_socket.shutdown(socket.SHUT_RDWR)
        if submitted:
            self._send_cancel()

    def discard(self) -> None:
        with self._discard_lock:
            if self._discarded:
                return
            self._discarded = True
        if not self._event.is_set():
            self.cancel()
        self._thread.join(timeout=PREFETCH_JOIN_SECONDS)
        if self._thread.is_alive():
            with self._discard_lock:
                self._discarded = False
            raise HarnessError("TTS prefetch worker did not stop after cancellation")
        assert self._result is not None
        _cleanup_chunks(self._result.chunks)


def _cleanup_chunks(chunks: list[Path]) -> None:
    parents = {
        output.parent for output in chunks if output.parent.name.startswith("stream-")
    }
    for output in chunks:
        with contextlib.suppress(OSError):
            output.unlink(missing_ok=True)
    for parent in parents:
        shutil.rmtree(parent, ignore_errors=True)


def _prefetch_utterance(
    text: str,
    *,
    request_id: str,
    cancelled: threading.Event,
    register_socket: Callable[[socket.socket], None],
    submit_request: Callable[[socket.socket, bytes], None],
    clear_socket: Callable[[socket.socket], None],
) -> PrefetchedUtterance:
    from ..config import TTS_SOCKET

    chunks: list[Path] = []
    chunk_texts: list[str] = []
    parser = TTSStreamParser(request_id)
    last_response = time.monotonic()
    stream_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    register_socket(stream_socket)
    try:
        if cancelled.is_set():
            raise HarnessError("TTS prefetch was cancelled")
        stream_socket.settimeout(PREFETCH_CONNECT_SECONDS)
        stream_socket.connect(str(TTS_SOCKET))
        stream_socket.setblocking(True)
        request = {
            "text": text,
            "voice": os.environ.get("VOICE_HARNESS_VOICE", ""),
            "stream": True,
            "request_id": request_id,
        }
        submit_request(stream_socket, json.dumps(request).encode() + b"\n")
        while True:
            if cancelled.is_set():
                raise HarnessError("TTS prefetch was cancelled")
            readable, _, _ = select.select([stream_socket], [], [], STREAM_POLL_SECONDS)
            if not readable:
                if time.monotonic() - last_response >= STREAM_TIMEOUT_SECONDS:
                    raise HarnessError("TTS prefetch stream timed out")
                continue
            data = stream_socket.recv(64 * 1024)
            if not data:
                break
            last_response = time.monotonic()
            for event in parser.feed(data):
                kind = event.get("event")
                if kind == "chunk":
                    chunks.append(Path(str(event["output"])))
                    chunk_texts.append(str(event["text"]))
        done_meta = parser.finish()
        return PrefetchedUtterance(
            sample_rate=parser.sample_rate,
            chunks=chunks,
            chunk_texts=chunk_texts,
            done_meta=done_meta,
        )
    except BaseException:
        _cleanup_chunks(chunks)
        raise
    finally:
        clear_socket(stream_socket)
        stream_socket.close()
        # Chunk files live under stream-{request_id}; playback owns cleanup after play.


def _open_playback(sample_rate: int) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            "pw-play",
            "--raw",
            "--channels=1",
            f"--rate={sample_rate}",
            "--format=s16",
            f"--latency={PLAYBACK_LATENCY}",
            "-",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _wait_for_process(
    process: subprocess.Popen[bytes],
    on_poll: Callable[[], None] | None,
) -> int:
    if on_poll is None:
        return process.wait()
    while True:
        on_poll()
        try:
            return process.wait(timeout=STREAM_POLL_SECONDS)
        except subprocess.TimeoutExpired:
            continue


def play_prefetched(
    prefetched: PrefetchedUtterance,
    *,
    should_interrupt: Callable[[], bool] | None = None,
) -> tuple[dict[str, object], bool]:
    started = time.perf_counter()
    interrupted = False
    played_texts: list[str] = []
    with playback_slot():
        process = _open_playback(prefetched.sample_rate)
        if process.stdin is None:
            raise HarnessError("pw-play stdin is unavailable")
        try:
            for index, output in enumerate(prefetched.chunks):
                if should_interrupt is not None and should_interrupt():
                    interrupted = True
                    break
                with wave.open(str(output), "rb") as source:
                    if (
                        source.getnchannels() != 1
                        or source.getsampwidth() != 2
                        or source.getframerate() != prefetched.sample_rate
                    ):
                        raise HarnessError(f"unexpected streaming WAV format: {output}")
                    while True:
                        if should_interrupt is not None and should_interrupt():
                            interrupted = True
                            break
                        audio = source.readframes(4096)
                        if not audio:
                            break
                        process.stdin.write(audio)
                if interrupted:
                    break
                played_texts.append(prefetched.chunk_texts[index])
            if interrupted:
                process.terminate()
                process.wait(timeout=1)
            else:
                process.stdin.close()
                returncode = process.wait()
                if returncode:
                    detail = (
                        process.stderr.read().decode(errors="replace").strip()
                        if process.stderr is not None
                        else ""
                    )
                    raise HarnessError(f"pw-play failed: {detail or returncode}")
        finally:
            if process.poll() is None:
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.terminate()
                    process.wait(timeout=1)
            _cleanup_chunks(prefetched.chunks)
    result = {
        **prefetched.done_meta,
        "ok": True,
        "stage": "tts",
        "request_seconds": round(time.perf_counter() - started, 3),
        "interrupted": interrupted,
        "played_text": " ".join(played_texts).strip(),
    }
    print(json.dumps(result))
    return result, interrupted


class PlaybackQueue:
    def __init__(self) -> None:
        self._items: collections.deque[
            tuple[PlaybackRequest, PrefetchHandle | None]
        ] = collections.deque()
        self._lock = threading.Lock()

    def enqueue(self, request: PlaybackRequest) -> None:
        with self._lock:
            self._items.append((request, None))

    def start_prefetch(self, *, limit: int = 1) -> None:
        if limit <= 0:
            raise ValueError("prefetch limit must be positive")
        with self._lock:
            active = sum(handle is not None for _, handle in self._items)
            for index, (request, handle) in enumerate(self._items):
                if active >= limit:
                    break
                if handle is None:
                    self._items[index] = (request, PrefetchHandle(request.text))
                    active += 1

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def clear(self) -> None:
        with self._lock:
            for _, handle in self._items:
                if handle is not None:
                    handle.discard()
            self._items.clear()

    def peek_text(self) -> str:
        with self._lock:
            if not self._items:
                return ""
            return self._items[0][0].text

    def queued_text(self) -> str:
        """Return all queued speech for wake-echo suppression."""
        with self._lock:
            return " ".join(request.text for request, _handle in self._items)

    def drain(
        self,
        *,
        should_interrupt: Callable[[], bool] | None = None,
        on_poll: Callable[[], None] | None = None,
        on_played: (
            Callable[[dict[str, object], bool, PlaybackRequest], None] | None
        ) = None,
    ) -> list[tuple[dict[str, object], bool, PlaybackRequest]]:
        """Play every queued item through one continuous pw-play stream."""
        played: list[tuple[dict[str, object], bool, PlaybackRequest]] = []
        deferred_error: Exception | None = None
        started = time.perf_counter()
        with playback_slot():
            process: subprocess.Popen[bytes] | None = None
            sample_rate = 0
            interrupted = False
            try:
                while not interrupted:
                    with self._lock:
                        if not self._items:
                            break
                        request, handle = self._items[0]
                        if handle is None:
                            handle = PrefetchHandle(request.text)
                            self._items[0] = (request, handle)
                    try:
                        prefetched = handle.wait(
                            should_interrupt=should_interrupt,
                            on_poll=on_poll,
                        )
                    except Exception as exc:
                        handle.discard()
                        with self._lock:
                            if self._items and self._items[0][0] is request:
                                self._items.popleft()
                        deferred_error = exc
                        break
                    if prefetched is None:
                        interrupted = True
                        handle.discard()
                        with self._lock:
                            if self._items and self._items[0][0] is request:
                                self._items.popleft()
                        result = {
                            "ok": True,
                            "stage": "tts",
                            "request_seconds": round(time.perf_counter() - started, 3),
                            "interrupted": True,
                            "played_text": "",
                        }
                        print(json.dumps(result))
                        played.append((result, True, request))
                        break
                    item_started = time.perf_counter()
                    chunk_texts: list[str] = []
                    try:
                        if process is None:
                            sample_rate = prefetched.sample_rate
                            process = _open_playback(sample_rate)
                        if process.stdin is None:
                            raise HarnessError("pw-play stdin is unavailable")
                        for index, output in enumerate(prefetched.chunks):
                            if on_poll is not None:
                                on_poll()
                            if should_interrupt is not None and should_interrupt():
                                interrupted = True
                                break
                            with wave.open(str(output), "rb") as source:
                                if (
                                    source.getnchannels() != 1
                                    or source.getsampwidth() != 2
                                    or source.getframerate() != sample_rate
                                ):
                                    raise HarnessError(
                                        f"unexpected streaming WAV format: {output}"
                                    )
                                while True:
                                    if on_poll is not None:
                                        on_poll()
                                    if (
                                        should_interrupt is not None
                                        and should_interrupt()
                                    ):
                                        interrupted = True
                                        break
                                    audio = source.readframes(4096)
                                    if not audio:
                                        break
                                    process.stdin.write(audio)
                            if interrupted:
                                break
                            chunk_texts.append(prefetched.chunk_texts[index])
                    except Exception as exc:
                        deferred_error = exc
                    finally:
                        _cleanup_chunks(prefetched.chunks)
                        with self._lock:
                            if self._items and self._items[0][0] is request:
                                self._items.popleft()
                    if deferred_error is not None:
                        break
                    result = {
                        **prefetched.done_meta,
                        "ok": True,
                        "stage": "tts",
                        "request_seconds": round(time.perf_counter() - item_started, 3),
                        "interrupted": interrupted,
                        "played_text": " ".join(chunk_texts).strip(),
                    }
                    print(json.dumps(result))
                    played.append((result, interrupted, request))
                    if interrupted:
                        break
                if process is not None:
                    if interrupted:
                        process.terminate()
                        process.wait(timeout=1)
                    else:
                        if process.stdin is not None:
                            process.stdin.close()
                        returncode = _wait_for_process(process, on_poll)
                        if returncode:
                            detail = (
                                process.stderr.read().decode(errors="replace").strip()
                                if process.stderr is not None
                                else ""
                            )
                            raise HarnessError(
                                f"pw-play failed: {detail or returncode}"
                            )
            finally:
                if process is not None and process.poll() is None:
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        process.terminate()
                        process.wait(timeout=1)
        if played:
            played[-1][0]["request_seconds"] = round(time.perf_counter() - started, 3)
        # Delivery callbacks may acknowledge durable announcements. Do not run
        # them until pw-play has exited (or an intentional interruption has
        # terminated it), because a late sink failure invalidates the batch.
        if on_played is not None:
            for result, interrupted, request in played:
                on_played(result, interrupted, request)
        if deferred_error is not None:
            raise deferred_error
        return played

    def play_next(
        self,
        *,
        should_interrupt: Callable[[], bool] | None = None,
    ) -> tuple[dict[str, object], bool, PlaybackRequest | None]:
        batch = self.drain(should_interrupt=should_interrupt)
        if not batch:
            return {}, False, None
        result, interrupted, request = batch[0]
        return result, interrupted, request
