from __future__ import annotations

import collections
import contextlib
import json
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
from .client import PLAYBACK_LATENCY, playback_slot

STREAM_POLL_SECONDS = 0.08


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
        self._event = threading.Event()
        self._result: PrefetchedUtterance | None = None
        threading.Thread(
            target=self._run,
            name=f"voice-prefetch-{uuid.uuid4().hex[:8]}",
            daemon=True,
        ).start()

    def _run(self) -> None:
        try:
            self._result = _prefetch_utterance(self.text)
        except BaseException as exc:
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
        timeout: float = 120.0,
        *,
        should_interrupt: Callable[[], bool] | None = None,
    ) -> PrefetchedUtterance | None:
        deadline = time.monotonic() + timeout
        while not self._event.wait(
            min(STREAM_POLL_SECONDS, max(0.0, deadline - time.monotonic()))
        ):
            if should_interrupt is not None and should_interrupt():
                return None
            if time.monotonic() >= deadline:
                raise HarnessError("Chatterbox prefetch timed out")
        assert self._result is not None
        if self._result.error is not None:
            raise HarnessError(
                f"Chatterbox prefetch failed: {type(self._result.error).__name__}: "
                f"{self._result.error}"
            )
        return self._result

    def discard(self) -> None:
        def cleanup() -> None:
            self._event.wait()
            assert self._result is not None
            for output in self._result.chunks:
                output.unlink(missing_ok=True)
            if self._result.chunks:
                shutil.rmtree(self._result.chunks[0].parent, ignore_errors=True)

        threading.Thread(
            target=cleanup,
            name=f"voice-prefetch-cleanup-{uuid.uuid4().hex[:8]}",
            daemon=True,
        ).start()


def _prefetch_utterance(text: str) -> PrefetchedUtterance:
    from ..config import TTS_SOCKET

    request_id = uuid.uuid4().hex
    chunks: list[Path] = []
    chunk_texts: list[str] = []
    done_meta: dict[str, object] = {}
    sample_rate = 0
    buffer = bytearray()
    stream_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        stream_socket.connect(str(TTS_SOCKET))
        request = {
            "text": text,
            "voice": __import__("os").environ.get("VOICE_HARNESS_VOICE", ""),
            "stream": True,
            "request_id": request_id,
        }
        stream_socket.sendall(json.dumps(request).encode() + b"\n")
        stream_socket.shutdown(socket.SHUT_WR)
        while True:
            readable, _, _ = select.select([stream_socket], [], [], STREAM_POLL_SECONDS)
            if not readable:
                continue
            data = stream_socket.recv(64 * 1024)
            if not data:
                break
            buffer.extend(data)
            while b"\n" in buffer:
                line, _, remainder = buffer.partition(b"\n")
                buffer[:] = remainder
                if not line:
                    continue
                event = json.loads(line)
                if not event.get("ok"):
                    raise HarnessError(
                        f"Chatterbox failed: {event.get('error', 'unknown error')}"
                    )
                kind = event.get("event")
                if kind == "start":
                    sample_rate = int(str(event["sample_rate"]))
                elif kind == "chunk":
                    chunks.append(Path(str(event["output"])))
                    chunk_texts.append(str(event.get("text", "")))
                elif kind == "done":
                    done_meta = event
        if not chunks or not sample_rate:
            raise HarnessError("Chatterbox prefetch returned no audio")
        return PrefetchedUtterance(
            sample_rate=sample_rate,
            chunks=chunks,
            chunk_texts=chunk_texts,
            done_meta=done_meta,
        )
    finally:
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
            for output in prefetched.chunks:
                output.unlink(missing_ok=True)
            if prefetched.chunks:
                shutil.rmtree(prefetched.chunks[0].parent, ignore_errors=True)
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
        self._items: collections.deque[tuple[PlaybackRequest, PrefetchHandle]] = (
            collections.deque()
        )
        self._lock = threading.Lock()

    def enqueue(self, request: PlaybackRequest) -> None:
        handle = PrefetchHandle(request.text)
        with self._lock:
            self._items.append((request, handle))

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def clear(self) -> None:
        with self._lock:
            for _, handle in self._items:
                handle.discard()
            self._items.clear()

    def peek_text(self) -> str:
        with self._lock:
            if not self._items:
                return ""
            return self._items[0][0].text

    def drain(
        self,
        *,
        should_interrupt: Callable[[], bool] | None = None,
        on_played: (
            Callable[[dict[str, object], bool, PlaybackRequest], None] | None
        ) = None,
    ) -> list[tuple[dict[str, object], bool, PlaybackRequest]]:
        """Play every queued item through one continuous pw-play stream."""
        played: list[tuple[dict[str, object], bool, PlaybackRequest]] = []
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
                    prefetched = handle.wait(should_interrupt=should_interrupt)
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
                        if on_played is not None:
                            on_played(result, True, request)
                        break
                    if process is None:
                        sample_rate = prefetched.sample_rate
                        process = _open_playback(sample_rate)
                    if process.stdin is None:
                        raise HarnessError("pw-play stdin is unavailable")
                    item_started = time.perf_counter()
                    chunk_texts: list[str] = []
                    for index, output in enumerate(prefetched.chunks):
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
                                if should_interrupt is not None and should_interrupt():
                                    interrupted = True
                                    break
                                audio = source.readframes(4096)
                                if not audio:
                                    break
                                process.stdin.write(audio)
                        if interrupted:
                            break
                        chunk_texts.append(prefetched.chunk_texts[index])
                    for output in prefetched.chunks:
                        output.unlink(missing_ok=True)
                    if prefetched.chunks:
                        shutil.rmtree(prefetched.chunks[0].parent, ignore_errors=True)
                    with self._lock:
                        if self._items and self._items[0][0] is request:
                            self._items.popleft()
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
                    if on_played is not None:
                        on_played(result, interrupted, request)
                    if interrupted:
                        break
                if process is not None:
                    if interrupted:
                        process.terminate()
                        process.wait(timeout=1)
                    else:
                        if process.stdin is not None:
                            process.stdin.close()
                        returncode = process.wait()
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
