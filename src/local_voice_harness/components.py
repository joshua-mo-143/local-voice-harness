from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request

from .config import LLM_HEALTH, TTS_SOCKET
from .errors import HarnessError
from .ipc import socket_ready


def llm_ready() -> bool:
    try:
        with urllib.request.urlopen(LLM_HEALTH, timeout=0.5) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def start_components(timeout: float = 30.0) -> None:
    subprocess.run(
        [
            "systemctl",
            "--user",
            "start",
            "voice-harness-llm.service",
            "voice-harness-tts.service",
        ],
        check=True,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if llm_ready() and socket_ready(TTS_SOCKET):
            return
        time.sleep(0.25)
    raise HarnessError("Qwen or Chatterbox did not become ready within 30 seconds")


def stop_components() -> None:
    subprocess.run(
        [
            "systemctl",
            "--user",
            "stop",
            "voice-harness-llm.service",
            "voice-harness-tts.service",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
