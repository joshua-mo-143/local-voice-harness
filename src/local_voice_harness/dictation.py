from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import time
from pathlib import Path

from .config import DEFAULT_SOURCE, RUNTIME, STT_SOCKET
from .errors import HarnessError
from .ipc import socket_ready
from .notifications import notify
from .stt.client import transcribe

TERMINAL_CLASSES = (
    "alacritty",
    "xterm",
    "kitty",
    "urxvt",
    "rxvt",
    "termite",
    "konsole",
    "wezterm",
    "foot",
    "tilix",
    "terminator",
    "gnome-terminal",
    "xfce4-terminal",
)
BLOCKED_WINDOW_CLASSES = ("net-runelite-client-runelite",)
INJECT_MODES = {"auto", "paste", "type", "stdout"}
STATE_DIR = RUNTIME / "dictation"
WAV_PATH = STATE_DIR / "recording.wav"
PID_PATH = STATE_DIR / "recording.pid"
RECORDER_LOG = STATE_DIR / "pw-record.log"


def recording_active() -> bool:
    if not PID_PATH.exists():
        return False
    try:
        os.kill(int(PID_PATH.read_text()), 0)
    except (OSError, ValueError):
        PID_PATH.unlink(missing_ok=True)
        return False
    return True


def start_recording() -> None:
    _ensure_dictation_allowed()
    STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    if recording_active():
        print("already recording")
        return
    if not socket_ready(STT_SOCKET):
        raise HarnessError("Whisper STT is stopped or still loading")
    source = os.environ.get(
        "DICTATION_SOURCE",
        os.environ.get("VOICE_HARNESS_SOURCE", DEFAULT_SOURCE),
    )
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
        raise HarnessError("dictation is not currently recording")
    try:
        pid = int(PID_PATH.read_text())
    except ValueError as exc:
        raise HarnessError("invalid dictation recorder PID") from exc
    finally:
        PID_PATH.unlink(missing_ok=True)
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError as exc:
        raise HarnessError("dictation recorder exited unexpectedly") from exc
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(pid, signal.SIGTERM)
        raise HarnessError("dictation recorder did not stop cleanly")
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


def _run(
    command: list[str],
    *,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )


def _active_window_pid() -> int | None:
    if shutil.which("xdotool") is None:
        return None
    process = _run(["xdotool", "getactivewindow", "getwindowpid"])
    try:
        return int(process.stdout.strip()) if process.returncode == 0 else None
    except ValueError:
        return None


def _process_tree(root: int) -> list[int]:
    result: list[int] = []
    pending = [root]
    while pending:
        pid = pending.pop()
        result.append(pid)
        children = Path(f"/proc/{pid}/task/{pid}/children")
        try:
            pending.extend(int(child) for child in children.read_text().split())
        except (OSError, ValueError):
            continue
    return result


def _focused_window_has_herdr() -> bool:
    pid = _active_window_pid()
    if pid is None:
        return False
    for process_id in _process_tree(pid):
        try:
            command = Path(f"/proc/{process_id}/comm").read_text().strip()
        except OSError:
            continue
        if command.startswith("herdr"):
            return True
    return False


def _send_to_herdr(text: str) -> bool:
    if shutil.which("herdr") is None or not _focused_window_has_herdr():
        return False
    current = _run(["herdr", "pane", "current"])
    if current.returncode:
        return False
    try:
        pane = json.loads(current.stdout)["result"]["pane"]
        pane_id = pane["pane_id"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return False
    if pane.get("agent") != "cursor" or not pane.get("focused"):
        return False
    return _run(["herdr", "pane", "send-text", pane_id, text]).returncode == 0


def _active_window_class() -> str:
    if shutil.which("xdotool") is None:
        return ""
    process = _run(["xdotool", "getactivewindow", "getwindowclassname"])
    return process.stdout.strip().lower() if process.returncode == 0 else ""


def _ensure_dictation_allowed() -> None:
    window_class = _active_window_class()
    if any(name in window_class for name in BLOCKED_WINDOW_CLASSES):
        raise HarnessError(
            "Dictation is disabled for RuneLite because simulated input may "
            "violate Jagex's rules."
        )


def _copy_to_clipboard(text: str) -> None:
    if shutil.which("xclip") is None:
        raise HarnessError("xclip is required to paste recognized text")
    if _run(["xclip", "-selection", "clipboard"], input_text=text).returncode:
        raise HarnessError("could not copy recognized text to the clipboard")


def _xdotool(*arguments: str) -> None:
    if shutil.which("xdotool") is None:
        raise HarnessError("xdotool is required to insert recognized text")
    if _run(["xdotool", *arguments]).returncode:
        raise HarnessError("could not insert recognized text into the active window")


def inject(text: str) -> None:
    _ensure_dictation_allowed()
    mode = os.environ.get("DICTATION_INJECT", "auto").lower()
    if mode not in INJECT_MODES:
        raise HarnessError(
            f"invalid DICTATION_INJECT mode {mode!r}; "
            f"expected one of {', '.join(sorted(INJECT_MODES))}"
        )
    if mode == "stdout":
        print(text)
        return
    if mode != "type" and _send_to_herdr(text):
        return
    if mode == "auto":
        mode = (
            "type"
            if any(name in _active_window_class() for name in TERMINAL_CLASSES)
            else "paste"
        )
    if mode == "type" or shutil.which("xclip") is None:
        time.sleep(0.12)
        _xdotool("type", "--clearmodifiers", "--", text)
        return
    _copy_to_clipboard(text)
    time.sleep(0.12)
    shortcut = (
        "ctrl+shift+v"
        if any(name in _active_window_class() for name in TERMINAL_CLASSES)
        else "ctrl+v"
    )
    _xdotool("key", "--clearmodifiers", shortcut)


def transcribe_and_type() -> None:
    inject(transcribe(WAV_PATH))


def run(command: str) -> None:
    if command == "begin":
        start_recording()
    elif command == "end":
        stop_recording()
        _ensure_dictation_allowed()
        transcribe_and_type()
    elif command == "toggle":
        if recording_active():
            stop_recording()
            _ensure_dictation_allowed()
            transcribe_and_type()
        else:
            start_recording()
    elif command == "transcribe":
        _ensure_dictation_allowed()
        transcribe_and_type()
    elif command == "cancel":
        cancel_recording()
    else:
        raise HarnessError(f"unknown dictation command: {command}")
