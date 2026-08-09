from __future__ import annotations

import contextlib
import fcntl
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator

from .config import (
    LLM_HEALTH,
    STATE_DIR,
    TTS_SOCKET,
    load_backend_settings,
)
from .credentials import CredentialError, get_venice_api_key
from .errors import HarnessError
from .ipc import socket_ready


def llm_ready() -> bool:
    settings = load_backend_settings()
    if settings.llm_provider == "venice":
        try:
            get_venice_api_key()
        except CredentialError:
            return False
        return True
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
    settings = load_backend_settings()
    services = ["voice-harness-tts.service"]
    if settings.llm_provider == "local":
        services.insert(0, "voice-harness-llm.service")
    subprocess.run(
        ["systemctl", "--user", "start", *services],
        check=True,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if llm_ready() and socket_ready(TTS_SOCKET):
            return
        time.sleep(0.25)
    raise HarnessError(
        f"LLM or TTS backend did not become ready within {timeout:g} seconds"
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
