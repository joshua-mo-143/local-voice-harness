from __future__ import annotations

import json
import os
import subprocess
import time

from ..config import STATE_DIR, TTS_SOCKET
from ..errors import HarnessError
from ..ipc import unix_request


def synthesize_and_play(text: str) -> dict[str, object]:
    output = STATE_DIR / "reply.wav"
    request = json.dumps(
        {
            "text": text,
            "output": str(output),
            "voice": os.environ.get("VOICE_HARNESS_VOICE", ""),
        }
    ).encode() + b"\n"
    started = time.perf_counter()
    result = json.loads(unix_request(TTS_SOCKET, request, timeout=120))
    if not result.get("ok"):
        raise HarnessError(f"Chatterbox failed: {result.get('error', 'unknown error')}")
    result.update(
        {
            "stage": "tts",
            "request_seconds": round(time.perf_counter() - started, 3),
        }
    )
    print(json.dumps(result))
    subprocess.run(["pw-play", str(output)], check=True)
    return result
