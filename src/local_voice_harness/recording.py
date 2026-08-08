from __future__ import annotations

import os
from pathlib import Path

from . import recorder
from .components import start_components
from .config import (
    DEFAULT_SOURCE,
    DICTATION_PID_PATH,
    DICTATION_RECORDER_LOG,
    DICTATION_STATE_DIR,
    DICTATION_WAV_PATH,
    PID_PATH,
    RECORDER_LOG,
    RECORDING_LOCK,
    STATE_DIR,
    STT_SOCKET,
    WAV_PATH,
)
from .ipc import socket_ready
from .notifications import notify

PATHS = recorder.RecorderPaths(
    STATE_DIR, WAV_PATH, PID_PATH, RECORDER_LOG, RECORDING_LOCK
)
DICTATION_PATHS = recorder.RecorderPaths(
    DICTATION_STATE_DIR,
    DICTATION_WAV_PATH,
    DICTATION_PID_PATH,
    DICTATION_RECORDER_LOG,
    RECORDING_LOCK,
)


def start_recording() -> None:
    source = os.environ.get("VOICE_HARNESS_SOURCE", DEFAULT_SOURCE)
    recorder.start_recording(
        PATHS,
        source=source,
        ready=lambda: socket_ready(STT_SOCKET),
        prepare=start_components,
        conflicts=(DICTATION_PATHS,),
    )
    notify("Listening…")
    print("recording")


def stop_recording() -> Path:
    return recorder.stop_recording(PATHS, missing_message="not currently recording")


def handoff_recording() -> Path:
    return recorder.handoff_recording(
        PATHS, active_message="cannot transcribe while recording is active"
    )


def retry_generation(generation: Path) -> Path:
    selected, conflict = (
        (PATHS, DICTATION_PATHS)
        if recorder.is_generation_path(PATHS, generation)
        else (DICTATION_PATHS, PATHS)
    )
    return recorder.retry_generation(
        selected,
        generation,
        active_message="cannot retry transcription while recording is active",
        conflicts=(conflict,),
    )


def cancel_recording() -> None:
    recorder.cancel_recording(PATHS)
    print("cancelled")
