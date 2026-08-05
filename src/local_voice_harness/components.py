from __future__ import annotations

import contextlib
import fcntl
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator

from .config import LLM_HEALTH, STATE_DIR, TTS_SOCKET
from .errors import HarnessError
from .ipc import socket_ready


def llm_ready() -> bool:
    try:
        with urllib.request.urlopen(LLM_HEALTH, timeout=0.5) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


@contextlib.contextmanager
def component_usage() -> Iterator[None]:
    STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (STATE_DIR / "components.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


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
    raise HarnessError(
        f"Qwen or Chatterbox did not become ready within {timeout:g} seconds"
    )


def stop_components() -> None:
    STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (STATE_DIR / "components.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
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
