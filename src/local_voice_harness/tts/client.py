from __future__ import annotations

import contextlib
import fcntl
import json
import os
import queue
import select
import shutil
import socket
import subprocess
import threading
import time
import uuid
import wave
from collections.abc import Callable, Iterator
from pathlib import Path

from ..config import STATE_DIR, TTS_SOCKET
from ..errors import HarnessError
from ..ipc import unix_request
from .stream import STREAM_POLL_SECONDS, STREAM_TIMEOUT_SECONDS, TTSStreamParser

PLAYBACK_LATENCY = os.environ.get("VOICE_HARNESS_PLAYBACK_LATENCY", "100ms")


@contextlib.contextmanager
def playback_slot() -> Iterator[None]:
    STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (STATE_DIR / "playback.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def synthesize_and_play(text: str) -> dict[str, object]:
    output = STATE_DIR / f"reply-{uuid.uuid4().hex}.wav"
    request = (
        json.dumps(
            {
                "text": text,
                "output": str(output),
                "voice": os.environ.get("VOICE_HARNESS_VOICE", ""),
            }
        ).encode()
        + b"\n"
    )
    try:
        started = time.perf_counter()
        try:
            response = unix_request(TTS_SOCKET, request, timeout=120)
        except OSError as exc:
            raise HarnessError(f"TTS request failed: {exc}") from exc
        try:
            decoded = json.loads(response)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HarnessError("TTS backend returned an invalid response") from exc
        if not isinstance(decoded, dict):
            raise HarnessError("TTS backend returned an invalid response")
        result: dict[str, object] = decoded
        if not result.get("ok"):
            raise HarnessError(
                f"TTS backend failed: {result.get('error', 'unknown error')}"
            )
        result.update(
            {
                "stage": "tts",
                "request_seconds": round(time.perf_counter() - started, 3),
            }
        )
        print(json.dumps(result))
        with playback_slot():
            subprocess.run(["pw-play", str(output)], check=True)
        return result
    finally:
        output.unlink(missing_ok=True)


class StreamingPlayback:
    """One cancellable, chunked TTS request feeding one PipeWire stream."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.request_id = uuid.uuid4().hex
        self.cancelled = threading.Event()
        self._cancel_sent = threading.Event()
        self._socket: socket.socket | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._state_lock = threading.Lock()
        self._played: list[str] = []
        self._worker_error: BaseException | None = None
        self._paths: set[Path] = set()
        self._ran = False

    @property
    def played_text(self) -> str:
        with self._state_lock:
            return " ".join(self._played).strip()

    def _send_cancel(self) -> None:
        if self._cancel_sent.is_set():
            return
        self._cancel_sent.set()
        request = (
            json.dumps({"op": "cancel", "request_id": self.request_id}).encode() + b"\n"
        )
        with contextlib.suppress(OSError, ValueError, json.JSONDecodeError):
            unix_request(TTS_SOCKET, request, timeout=2)

    def _terminate_process(self) -> None:
        with self._state_lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)

    def cancel(self) -> None:
        """Stop audible/queued output and ask the server to stop after its current chunk."""
        if self.cancelled.is_set():
            return
        self.cancelled.set()
        self._terminate_process()
        self._send_cancel()
        with self._state_lock:
            stream_socket = self._socket
        if stream_socket is not None:
            with contextlib.suppress(OSError):
                stream_socket.shutdown(socket.SHUT_RDWR)

    def _open_playback(self, sample_rate: int) -> subprocess.Popen[bytes]:
        process = subprocess.Popen(
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
        with self._state_lock:
            self._process = process
        if self.cancelled.is_set():
            self._terminate_process()
        return process

    def _play_chunks(
        self,
        chunks: queue.Queue[dict[str, object] | None],
        sample_rate: int,
    ) -> None:
        process: subprocess.Popen[bytes] | None = None
        pending_text: str | None = None
        try:
            process = self._open_playback(sample_rate)
            if process.stdin is None:
                raise HarnessError("pw-play stdin is unavailable")
            while not self.cancelled.is_set():
                item = chunks.get()
                if item is None:
                    break
                output = Path(str(item["output"]))
                self._paths.add(output)
                with wave.open(str(output), "rb") as source:
                    if (
                        source.getnchannels() != 1
                        or source.getsampwidth() != 2
                        or source.getframerate() != sample_rate
                    ):
                        raise HarnessError(f"unexpected streaming WAV format: {output}")
                    while not self.cancelled.is_set():
                        audio = source.readframes(4096)
                        if not audio:
                            break
                        process.stdin.write(audio)
                output.unlink(missing_ok=True)
                self._paths.discard(output)
                if not self.cancelled.is_set():
                    # Keep the newest fully-written chunk pending: pw-play may still
                    # have it buffered when an interruption terminates the stream.
                    with self._state_lock:
                        if pending_text:
                            self._played.append(pending_text)
                    pending_text = str(item.get("text", ""))
            if self.cancelled.is_set():
                self._terminate_process()
                return
            process.stdin.close()
            returncode = process.wait()
            if returncode:
                detail = (
                    process.stderr.read().decode(errors="replace").strip()
                    if process.stderr is not None
                    else ""
                )
                raise HarnessError(f"pw-play failed: {detail or returncode}")
            if pending_text:
                with self._state_lock:
                    self._played.append(pending_text)
        except Exception as exc:
            if not self.cancelled.is_set():
                self._worker_error = exc
                self.cancelled.set()
                self._send_cancel()
        finally:
            if process is not None and process.poll() is None:
                self._terminate_process()

    def run(
        self,
        *,
        should_interrupt: Callable[[], bool] | None = None,
    ) -> dict[str, object]:
        if self._ran:
            raise RuntimeError("a streaming playback session can only run once")
        self._ran = True
        started = time.perf_counter()
        chunks: queue.Queue[dict[str, object] | None] = queue.Queue()
        worker: threading.Thread | None = None
        done: dict[str, object] = {}
        sentinel_sent = False
        parser = TTSStreamParser(self.request_id)
        last_response = time.monotonic()
        stream_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        with self._state_lock:
            self._socket = stream_socket
        try:
            stream_socket.connect(str(TTS_SOCKET))
            request = {
                "text": self.text,
                "voice": os.environ.get("VOICE_HARNESS_VOICE", ""),
                "stream": True,
                "request_id": self.request_id,
            }
            stream_socket.sendall(json.dumps(request).encode() + b"\n")
            stream_socket.shutdown(socket.SHUT_WR)
            while not self.cancelled.is_set():
                if self._worker_error is not None:
                    self.cancel()
                    break
                if should_interrupt is not None and should_interrupt():
                    self.cancel()
                    break
                readable, _, _ = select.select(
                    [stream_socket], [], [], STREAM_POLL_SECONDS
                )
                if not readable:
                    if time.monotonic() - last_response >= STREAM_TIMEOUT_SECONDS:
                        raise HarnessError("TTS stream timed out")
                    continue
                data = stream_socket.recv(64 * 1024)
                if not data:
                    if self.cancelled.is_set():
                        break
                    done = parser.finish()
                    break
                last_response = time.monotonic()
                for event in parser.feed(data):
                    kind = event.get("event")
                    if kind == "start":
                        worker = threading.Thread(
                            target=self._play_chunks,
                            args=(chunks, parser.sample_rate),
                            name=f"voice-playback-{self.request_id[:8]}",
                            daemon=True,
                        )
                        worker.start()
                    elif kind == "chunk":
                        path = Path(str(event["output"]))
                        self._paths.add(path)
                        chunks.put(event)
                    elif kind == "done":
                        if not sentinel_sent:
                            chunks.put(None)
                            sentinel_sent = True
            if not sentinel_sent:
                chunks.put(None)
            if worker is not None:
                worker.join()
            if self._worker_error is not None:
                raise HarnessError(f"streaming playback failed: {self._worker_error}")
            result = {
                **done,
                "ok": True,
                "stage": "tts",
                "request_seconds": round(time.perf_counter() - started, 3),
                "interrupted": self.cancelled.is_set(),
                "played_text": self.played_text,
            }
            print(json.dumps(result))
            return result
        except Exception:
            self.cancel()
            chunks.put(None)
            if worker is not None:
                worker.join(timeout=2)
            raise
        finally:
            with self._state_lock:
                self._socket = None
            stream_socket.close()
            for output in list(self._paths):
                output.unlink(missing_ok=True)
            stream_dir = STATE_DIR / f"stream-{self.request_id}"
            shutil.rmtree(stream_dir, ignore_errors=True)


def stream_and_play(
    text: str,
    *,
    should_interrupt: Callable[[], bool] | None = None,
) -> dict[str, object]:
    with playback_slot():
        return StreamingPlayback(text).run(should_interrupt=should_interrupt)
