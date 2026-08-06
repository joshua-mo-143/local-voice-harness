from __future__ import annotations

import os
import signal
import subprocess
import time

from .components import start_components
from .config import (
    DEFAULT_SOURCE,
    PID_PATH,
    RECORDER_LOG,
    STATE_DIR,
    STT_SOCKET,
    WAV_PATH,
)
from .errors import HarnessError
from .ipc import socket_ready
from .notifications import notify


def start_recording() -> None:
    STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    if PID_PATH.exists():
        try:
            os.kill(int(PID_PATH.read_text()), 0)
            print("already recording")
            return
        except (OSError, ValueError):
            PID_PATH.unlink(missing_ok=True)
    if not socket_ready(STT_SOCKET):
        raise HarnessError("STT is stopped or still loading")
    start_components()
    source = os.environ.get("VOICE_HARNESS_SOURCE", DEFAULT_SOURCE)
    command = ["pw-record"]
    if source:
        command.extend(("--target", source))
    command.extend(("--channels=1", "--rate=16000", "--format=s16", str(WAV_PATH)))
    log_handle = RECORDER_LOG.open("wb")
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()
    PID_PATH.write_text(str(process.pid))
    time.sleep(0.2)
    if process.poll() is not None:
        PID_PATH.unlink(missing_ok=True)
        detail = RECORDER_LOG.read_text(errors="replace").strip()
        raise HarnessError(f"pw-record failed: {detail or process.returncode}")
    notify("Listening…")
    print("recording")


def stop_recording() -> None:
    if not PID_PATH.exists():
        raise HarnessError("not currently recording")
    try:
        pid = int(PID_PATH.read_text())
    except ValueError as exc:
        raise HarnessError("invalid recorder PID") from exc
    finally:
        PID_PATH.unlink(missing_ok=True)
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError as exc:
        raise HarnessError("recorder exited unexpectedly") from exc
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(pid, signal.SIGTERM)
        raise HarnessError("recorder did not stop cleanly")
    if not WAV_PATH.is_file() or WAV_PATH.stat().st_size <= 44:
        raise HarnessError("no microphone audio was captured")


def cancel_recording() -> None:
    if PID_PATH.exists():
        try:
            os.kill(int(PID_PATH.read_text()), signal.SIGTERM)
        except (ProcessLookupError, ValueError):
            pass
        PID_PATH.unlink(missing_ok=True)
    WAV_PATH.unlink(missing_ok=True)
    print("cancelled")
