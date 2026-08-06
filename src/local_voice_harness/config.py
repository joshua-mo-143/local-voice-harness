from __future__ import annotations

import os
import re
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
STATE_DIR = RUNTIME / "voice-harness"
JOBS_DIR = STATE_DIR / "jobs"
WAV_PATH = STATE_DIR / "request.wav"
PID_PATH = STATE_DIR / "recording.pid"
RECORDER_LOG = STATE_DIR / "pw-record.log"
STT_SOCKET = RUNTIME / "dictation.sock"
TTS_SOCKET = RUNTIME / "voice-harness-tts.sock"
LLM_HEALTH = "http://127.0.0.1:8090/health"
LLM_CHAT = "http://127.0.0.1:8090/v1/chat/completions"
CURSOR_FOREGROUND_SECONDS = float(
    os.environ.get("VOICE_HARNESS_CURSOR_FOREGROUND_SECONDS", "5")
)
DEFAULT_SOURCE = "alsa_input.pci-0000_00_1f.3-platform-sof_sdw.HiFi__Mic__source"
CURSOR_PATTERN = re.compile(
    r"^\s*(?P<verb>use|ask|call)\s+(?:cursor|curser|cursa)\b", re.IGNORECASE
)
FORK_PATTERN = re.compile(r"\bfork(?:ed|ing|s)?\b", re.IGNORECASE)

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
