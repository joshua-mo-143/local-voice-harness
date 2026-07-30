from __future__ import annotations

import json
import time

from ..config import STT_SOCKET, WAV_PATH
from ..errors import HarnessError
from ..ipc import unix_request


def transcribe() -> str:
    started = time.perf_counter()
    response = unix_request(STT_SOCKET, f"{WAV_PATH}\n".encode(), timeout=120)
    text = response.decode(errors="replace").strip()
    if text.startswith("__DICTATION_ERROR__:"):
        raise HarnessError(text.removeprefix("__DICTATION_ERROR__:"))
    if not text:
        raise HarnessError("Whisper did not recognize any speech")
    print(json.dumps({"stage": "stt", "seconds": round(time.perf_counter() - started, 3)}))
    return text
