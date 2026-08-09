from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import STT_SOCKET
from ..errors import HarnessError, NoSpeechError
from ..ipc import unix_request

REQUEST_DEADLINE_SECONDS = 120.0
BUSY_BACKOFF_SECONDS = 0.1
MAX_BUSY_BACKOFF_SECONDS = 1.0


def _busy_timeout_error(audio_path: Path) -> HarnessError:
    return HarnessError(
        "STT remained busy; audio was preserved. Retry with "
        f"`voice-harness transcribe --generation {audio_path}`"
    )


def transcribe(audio_path: Path) -> str:
    started = time.perf_counter()
    deadline = time.monotonic() + REQUEST_DEADLINE_SECONDS
    backoff = BUSY_BACKOFF_SECONDS
    first_attempt = True
    while True:
        if first_attempt:
            timeout = REQUEST_DEADLINE_SECONDS
        else:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                raise _busy_timeout_error(audio_path)
        first_attempt = False
        try:
            response = unix_request(
                STT_SOCKET, f"{audio_path}\n".encode(), timeout=timeout
            )
        except OSError as exc:
            raise HarnessError(f"STT request failed: {exc}") from exc
        text = response.decode(errors="replace").strip()
        try:
            protocol = json.loads(text)
        except json.JSONDecodeError:
            protocol = None
        if not (
            isinstance(protocol, dict)
            and protocol.get("ok") is False
            and isinstance(protocol.get("error"), dict)
        ):
            break
        error = protocol["error"]
        code = str(error.get("code", "protocol_error"))
        message = str(error.get("message", "STT request failed"))
        if code != "server_busy":
            raise HarnessError(f"{code}: {message}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _busy_timeout_error(audio_path)
        delay = min(backoff, remaining)
        time.sleep(delay)
        backoff = min(backoff * 2, MAX_BUSY_BACKOFF_SECONDS)
    if text.startswith("__DICTATION_ERROR__:"):
        raise HarnessError(text.removeprefix("__DICTATION_ERROR__:"))
    if not text:
        raise NoSpeechError("STT did not recognize any speech")
    print(
        json.dumps({"stage": "stt", "seconds": round(time.perf_counter() - started, 3)})
    )
    return text
