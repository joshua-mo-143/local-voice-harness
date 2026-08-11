from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from . import config, recorder
from .config import (
    DEFAULT_SOURCE,
    DICTATION_PID_PATH,
    DICTATION_RECORDER_LOG,
    DICTATION_STATE_DIR,
    DICTATION_WAV_PATH,
    RECORDING_LOCK,
    STT_SOCKET,
)
from .desktop import Desktop, DesktopError, Window, get_desktop
from .errors import HarnessError, NoSpeechError
from .ipc import socket_ready
from .notifications import notify
from .stt.client import transcribe
from .vad import SpeechDetector, VadCaptureSettings, capture_vad_audio

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
STATE_DIR = DICTATION_STATE_DIR
WAV_PATH = DICTATION_WAV_PATH
PID_PATH = DICTATION_PID_PATH
RECORDER_LOG = DICTATION_RECORDER_LOG
PATHS = recorder.RecorderPaths(
    STATE_DIR, WAV_PATH, PID_PATH, RECORDER_LOG, RECORDING_LOCK
)
MANUAL_PATHS = recorder.RecorderPaths(
    config.STATE_DIR,
    config.WAV_PATH,
    config.PID_PATH,
    config.RECORDER_LOG,
    RECORDING_LOCK,
)


def recording_active() -> bool:
    return recorder.recording_active(PATHS)


def start_recording() -> None:
    _ensure_dictation_allowed()
    source = os.environ.get(
        "DICTATION_SOURCE",
        os.environ.get("VOICE_HARNESS_SOURCE", DEFAULT_SOURCE),
    )
    recorder.start_recording(
        PATHS,
        source=source,
        ready=lambda: socket_ready(STT_SOCKET),
        conflicts=(MANUAL_PATHS,),
    )
    notify("Listening…")
    print("recording")


def stop_recording() -> Path:
    return recorder.stop_recording(
        PATHS, missing_message="dictation is not currently recording"
    )


def cancel_recording() -> None:
    recorder.cancel_recording(PATHS)
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
    desktop = get_desktop()
    window = desktop.active_window() if desktop is not None else None
    return window.pid if window is not None else None


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


def _active_window() -> Window | None:
    desktop = get_desktop()
    if desktop is None:
        return None
    return desktop.active_window()


def _window_is_blocked(window: Window) -> bool:
    return any(name in window.window_class for name in BLOCKED_WINDOW_CLASSES)


def _ensure_dictation_allowed() -> None:
    window = _active_window()
    if window is not None and _window_is_blocked(window):
        raise HarnessError(
            "Dictation is disabled for RuneLite because simulated input may "
            "violate Jagex's rules."
        )


def _ensure_window_allowed(window: Window) -> None:
    if _window_is_blocked(window):
        raise HarnessError(
            "Dictation is disabled for RuneLite because simulated input may "
            "violate Jagex's rules."
        )


def _require_focus(desktop: Desktop, window: Window) -> None:
    if desktop.active_window() != window:
        raise HarnessError("dictation cancelled because the focused window changed")


def _copy_to_clipboard(desktop: Desktop, window: Window, text: str) -> None:
    _require_focus(desktop, window)
    try:
        copied = desktop.write_clipboard(text)
    except DesktopError as exc:
        raise HarnessError(str(exc)) from exc
    if not copied:
        raise HarnessError("could not copy recognized text to the clipboard")


def _type_text(desktop: Desktop, window: Window, text: str) -> None:
    _require_focus(desktop, window)
    try:
        desktop.type_text(text, window=window)
    except DesktopError as exc:
        raise HarnessError(str(exc)) from exc


def _send_key(desktop: Desktop, window: Window, key: str) -> None:
    _require_focus(desktop, window)
    try:
        sent = desktop.send_key(key, window=window)
    except DesktopError as exc:
        raise HarnessError(str(exc)) from exc
    if not sent:
        raise HarnessError("could not insert recognized text into the active window")


def _restore_clipboard(
    desktop: Desktop,
    *,
    copied_text: str,
    clipboard_existed: bool,
    previous_clipboard: str,
) -> None:
    current_exists, current_clipboard = desktop.read_clipboard()
    if not current_exists or current_clipboard != copied_text:
        return
    try:
        desktop.write_clipboard(previous_clipboard if clipboard_existed else "")
    except DesktopError:
        pass


def inject(text: str) -> None:
    mode = os.environ.get("DICTATION_INJECT", "auto").lower()
    if mode not in INJECT_MODES:
        raise HarnessError(
            f"invalid DICTATION_INJECT mode {mode!r}; "
            f"expected one of {', '.join(sorted(INJECT_MODES))}"
        )
    if mode == "stdout":
        print(text)
        return

    desktop = get_desktop()
    window = desktop.active_window() if desktop is not None else None
    if window is not None:
        _ensure_window_allowed(window)

    if mode != "type" and _send_to_herdr(text):
        return

    if desktop is None:
        raise HarnessError(
            "focused-window automation supports X11, Hyprland, and Sway only"
        )
    if window is None:
        raise HarnessError("could not determine the focused window for dictation")

    if mode == "auto":
        mode = (
            "type"
            if any(name in window.window_class for name in TERMINAL_CLASSES)
            else "paste"
        )

    if mode == "type" or not desktop.has_clipboard():
        time.sleep(0.12)
        _type_text(desktop, window, text)
        return

    clipboard_existed, previous_clipboard = desktop.read_clipboard()
    copied = False
    try:
        _copy_to_clipboard(desktop, window, text)
        copied = True
        time.sleep(0.12)
        shortcut = (
            "ctrl+shift+v"
            if any(name in window.window_class for name in TERMINAL_CLASSES)
            else "ctrl+v"
        )
        _send_key(desktop, window, shortcut)
    finally:
        if copied:
            _restore_clipboard(
                desktop,
                copied_text=text,
                clipboard_existed=clipboard_existed,
                previous_clipboard=previous_clipboard,
            )


def transcribe_and_type(audio_path: Path) -> None:
    inject(transcribe(audio_path))


def run_vad() -> None:
    if recorder.request_vad_stop(PATHS):
        notify("Stopping always-on VAD dictation…")
        print("stopping")
        return

    _ensure_dictation_allowed()
    if not socket_ready(STT_SOCKET):
        raise HarnessError("STT is stopped or still loading")
    settings = VadCaptureSettings.from_environment()
    detector = SpeechDetector(minimum_rms=settings.minimum_rms)
    stop_requested = threading.Event()
    previous_terminate = signal.getsignal(signal.SIGTERM)

    def request_stop(_signum: int, _frame: object) -> None:
        stop_requested.set()

    signal.signal(signal.SIGTERM, request_stop)
    claimed = False
    try:
        recorder.claim_current_process(PATHS, mode="vad", conflicts=(MANUAL_PATHS,))
        claimed = True
        source = os.environ.get(
            "DICTATION_SOURCE",
            os.environ.get("VOICE_HARNESS_SOURCE", DEFAULT_SOURCE),
        )
        notify("Always-on VAD dictation enabled")
        print("listening")
        while not stop_requested.is_set():
            try:
                capture_vad_audio(
                    PATHS.audio,
                    source=source,
                    settings=settings,
                    detector=detector,
                    stop_requested=stop_requested,
                )
            except HarnessError:
                if stop_requested.is_set():
                    break
                raise
            generation = recorder.complete_current_recording(
                PATHS,
                mode="vad",
                retain_owner=True,
            )
            notify("Transcribing…")
            _ensure_dictation_allowed()
            try:
                transcribe_and_type(generation)
            except NoSpeechError:
                print("no speech detected; listening")
    finally:
        if claimed:
            recorder.abandon_current_recording(PATHS, mode="vad")
        signal.signal(signal.SIGTERM, previous_terminate)


def run(command: str) -> None:
    if command == "begin":
        start_recording()
    elif command == "end":
        audio_path = stop_recording()
        _ensure_dictation_allowed()
        transcribe_and_type(audio_path)
    elif command == "toggle":
        if recording_active():
            audio_path = stop_recording()
            _ensure_dictation_allowed()
            transcribe_and_type(audio_path)
        else:
            start_recording()
    elif command == "transcribe":
        _ensure_dictation_allowed()
        audio_path = recorder.handoff_recording(
            PATHS, active_message="cannot transcribe while dictation is recording"
        )
        transcribe_and_type(audio_path)
    elif command == "cancel":
        cancel_recording()
    elif command == "vad":
        run_vad()
    else:
        raise HarnessError(f"unknown dictation command: {command}")
