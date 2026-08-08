from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(
    os.environ.get("VOICE_HARNESS_ROOT", PACKAGE_ROOT.parents[1])
).resolve()
REPOSITORY_ROOT = Path(
    os.environ.get("VOICE_HARNESS_PROJECT_ROOT", Path.home())
).resolve()
GITHUB_ROOT = Path(
    os.environ.get("VOICE_HARNESS_GITHUB_ROOT", Path.home() / "src")
).resolve()
RUNTIME = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
RECORDING_LOCK = RUNTIME / "voice-harness-recording.lock"
STATE_DIR = RUNTIME / "voice-harness"
LEGACY_JOBS_DIR = STATE_DIR / "jobs"
JOB_LOGS_DIR = STATE_DIR / "jobs"


def xdg_state_home(
    environment: Mapping[str, str] = os.environ,
    *,
    home: Path | None = None,
) -> Path:
    configured = environment.get("XDG_STATE_HOME")
    path = Path(configured) if configured else None
    if path is not None and path.is_absolute():
        return path
    return (home or Path.home()) / ".local/state"


def systemd_state_directory(
    environment: Mapping[str, str] = os.environ,
) -> Path | None:
    configured = environment.get("STATE_DIRECTORY", "")
    candidates = [
        Path(value)
        for value in configured.split(":")
        if value and Path(value).is_absolute()
    ]
    named = [path for path in candidates if path.name == "voice-harness"]
    if len(named) == 1:
        return named[0]
    if len(candidates) == 1:
        return candidates[0]
    return None


def durable_state_dir(
    environment: Mapping[str, str] = os.environ,
    *,
    home: Path | None = None,
) -> Path:
    return systemd_state_directory(environment) or (
        xdg_state_home(environment, home=home) / "voice-harness"
    )


DURABLE_STATE_DIR = durable_state_dir()
JOBS_DIR = DURABLE_STATE_DIR / "jobs"
WAV_PATH = STATE_DIR / "request.wav"
PID_PATH = STATE_DIR / "recording.pid"
RECORDER_LOG = STATE_DIR / "pw-record.log"
DICTATION_STATE_DIR = RUNTIME / "dictation"
DICTATION_WAV_PATH = DICTATION_STATE_DIR / "recording.wav"
DICTATION_PID_PATH = DICTATION_STATE_DIR / "recording.pid"
DICTATION_RECORDER_LOG = DICTATION_STATE_DIR / "pw-record.log"
STT_SOCKET = RUNTIME / "dictation.sock"
TTS_SOCKET = RUNTIME / "voice-harness-tts.sock"
LLM_HEALTH = "http://127.0.0.1:8090/health"
LLM_CHAT = "http://127.0.0.1:8090/v1/chat/completions"
DEFAULT_LLM_MODEL = "qwen3.5-4b"
LLM_MODEL = os.environ.get("VOICE_HARNESS_LLM_MODEL", DEFAULT_LLM_MODEL)
CURSOR_FOREGROUND_SECONDS = float(
    os.environ.get("VOICE_HARNESS_CURSOR_FOREGROUND_SECONDS", "5")
)
DEFAULT_SOURCE = "alsa_input.pci-0000_00_1f.3-platform-sof_sdw.HiFi__Mic__source"
CURSOR_PATTERN = re.compile(
    r"^\s*(?P<verb>use|ask|call)\s+(?:cursor|curser|cursa)\b", re.IGNORECASE
)


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


def _env_int(name: str, *, default: int, minimum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return value if value >= minimum else default


def _env_classes(name: str, defaults: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None:
        return defaults
    return tuple(
        part for part in (piece.strip().casefold() for piece in raw.split(",")) if part
    )


# Focused-application context capture (GitHub issue #32). Capture only runs when
# the trusted utterance explicitly asks for it, and is fenced by a deny-list of
# sensitive or unsupported application window classes plus a total size cap.
FOCUSED_APP_CONTEXT_ENABLED = _env_flag(
    "VOICE_HARNESS_FOCUSED_APP_CONTEXT", default=True
)
# Substrings matched (case-insensitively) against the focused window class. Any
# match denies capture. Defaults cover password managers, secret stores, and the
# RuneLite client that already blocks simulated input elsewhere in the harness.
DEFAULT_FOCUSED_APP_DENY_CLASSES = (
    "keepassxc",
    "keepass",
    "bitwarden",
    "1password",
    "1passwd",
    "enpass",
    "gnome-keyring",
    "seahorse",
    "polkit",
    "net-runelite-client-runelite",
)
FOCUSED_APP_DENY_CLASSES = _env_classes(
    "VOICE_HARNESS_FOCUSED_APP_DENY", DEFAULT_FOCUSED_APP_DENY_CLASSES
)
# Upper bound on the combined, provenance-labelled focused-app context appended
# to a routed request. Individual sources enforce their own tighter limits.
MAX_FOCUSED_APP_CONTEXT_CHARS = _env_int(
    "VOICE_HARNESS_FOCUSED_APP_MAX_CHARS", default=12_000, minimum=1
)

SYSTEMD_USER_DIR = Path.home() / ".config" / "systemd" / "user"
SERVICE_FILES = (
    "dictation.service",
    "voice-harness-llm.service",
    "voice-harness-tts.service",
    "voice-harness-wake.service",
)
START_SERVICES = ("dictation.service", "voice-harness-wake.service")
STOP_SERVICES = (
    "voice-harness-wake.service",
    "voice-harness-llm.service",
    "voice-harness-tts.service",
    "dictation.service",
)
