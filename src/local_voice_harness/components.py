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
    BackendSettings,
)
from .credentials import CredentialError, get_venice_api_key
from .errors import HarnessError
from .ipc import socket_ready
from .user_config import default_user_config


def llm_ready(settings: BackendSettings | None = None) -> bool:
    resolved = settings or default_user_config().providers
    if resolved.llm_provider == "venice":
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


def start_components(
    settings: BackendSettings | None = None, timeout: float = 30.0
) -> None:
    resolved = settings or default_user_config().providers
    if "venice" in (resolved.llm_provider, resolved.tts_provider):
        get_venice_api_key()
    services = ["voice-harness-tts.service"]
    if resolved.llm_provider == "local":
        services.insert(0, "voice-harness-llm.service")
    subprocess.run(
        ["systemctl", "--user", "start", *services],
        check=True,
    )
    deadline = time.monotonic() + timeout
    llm_is_ready = False
    tts_is_ready = False
    while time.monotonic() < deadline:
        llm_is_ready = resolved.llm_provider == "venice" or llm_ready(resolved)
        tts_is_ready = socket_ready(TTS_SOCKET) if llm_is_ready else False
        if llm_is_ready and tts_is_ready:
            return
        time.sleep(0.25)
    unready = [
        name
        for name, ready in (("LLM", llm_is_ready), ("TTS", tts_is_ready))
        if not ready
    ]
    subject = f"{' and '.join(unready)} backend"
    if len(unready) > 1:
        subject += "s"
    raise HarnessError(f"{subject} did not become ready within {timeout:g} seconds")


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
