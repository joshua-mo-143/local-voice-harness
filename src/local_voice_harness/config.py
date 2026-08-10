from __future__ import annotations

import math
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
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


def xdg_config_home(
    environment: Mapping[str, str] = os.environ,
    *,
    home: Path | None = None,
) -> Path:
    configured = environment.get("XDG_CONFIG_HOME")
    path = Path(configured) if configured else None
    if path is not None and path.is_absolute():
        return path
    return (home or Path.home()) / ".config"


def vocabulary_path(
    environment: Mapping[str, str] = os.environ,
    *,
    home: Path | None = None,
) -> Path:
    return xdg_config_home(environment, home=home) / "voice-harness" / "vocabulary.json"


class BackendConfigurationError(ValueError):
    """A backend selector, model, endpoint, or timeout is invalid."""


@dataclass(frozen=True)
class BackendSettings:
    llm_provider: str
    llm_model: str
    llm_endpoint: str
    llm_timeout: float
    tts_provider: str
    tts_model: str
    tts_voice: str
    tts_speed: float
    tts_endpoint: str
    tts_timeout: float


def backend_config_path(
    environment: Mapping[str, str] = os.environ,
    *,
    home: Path | None = None,
) -> Path:
    return xdg_config_home(environment, home=home) / "voice-harness" / "backends.toml"


def _backend_value(
    section: object,
    key: str,
    default: object,
    environment: Mapping[str, str],
    environment_key: str,
) -> object:
    if environment_key in environment:
        return environment[environment_key]
    if isinstance(section, dict):
        return section.get(key, default)
    return default


def _provider(value: object, *, label: str) -> str:
    provider = str(value).strip().casefold()
    if provider not in {"local", "venice"}:
        raise BackendConfigurationError(f"{label} provider must be local or venice")
    return provider


def _nonempty(value: object, *, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise BackendConfigurationError(f"{label} must not be empty")
    return text


def _positive_float(value: object, *, label: str) -> float:
    try:
        number = float(str(value))
    except (TypeError, ValueError) as exc:
        raise BackendConfigurationError(f"{label} must be a positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise BackendConfigurationError(f"{label} must be a positive number")
    return number


def _tts_speed(value: object) -> float:
    speed = _positive_float(value, label="TTS speed")
    if speed > 4:
        raise BackendConfigurationError("TTS speed must be between 0.25 and 4")
    if speed < 0.25:
        raise BackendConfigurationError("TTS speed must be between 0.25 and 4")
    return speed


def read_toml_table(
    config_path: Path,
    *,
    error: type[ValueError] = BackendConfigurationError,
    label: str = "configuration",
) -> dict[str, object]:
    """Read ``config_path`` into a TOML table, tolerating a missing file."""

    try:
        raw = tomllib.loads(config_path.read_text()) if config_path.exists() else {}
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise error(f"could not read {label} {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise error(f"{label} must be a TOML table")
    return raw


def reject_file_based_credentials(
    venice: Mapping[str, object],
    environment: Mapping[str, str],
) -> None:
    """Raise if Venice credentials are supplied through a file instead of the store."""

    if "api_key_file" in venice or "VOICE_HARNESS_VENICE_API_KEY_FILE" in environment:
        raise BackendConfigurationError(
            "file-based Venice credentials are unsupported; run "
            "`voice-harness credentials set`"
        )


def backend_settings_from_tables(
    llm: Mapping[str, object],
    tts: Mapping[str, object],
    environment: Mapping[str, str] = os.environ,
) -> BackendSettings:
    """Validate already-merged ``[llm]``/``[tts]`` tables into typed settings.

    Environment variables override the supplied table values, matching the
    precedence enforced by :func:`load_backend_settings`.
    """

    llm_provider = _provider(
        _backend_value(
            llm, "provider", "local", environment, "VOICE_HARNESS_LLM_PROVIDER"
        ),
        label="LLM",
    )
    tts_provider = _provider(
        _backend_value(
            tts, "provider", "local", environment, "VOICE_HARNESS_TTS_PROVIDER"
        ),
        label="TTS",
    )
    return BackendSettings(
        llm_provider=llm_provider,
        llm_model=_nonempty(
            _backend_value(
                llm,
                "model",
                "qwen3.5-4b" if llm_provider == "local" else "venice-uncensored",
                environment,
                "VOICE_HARNESS_LLM_MODEL",
            ),
            label="LLM model",
        ),
        llm_endpoint=_nonempty(
            _backend_value(
                llm,
                "endpoint",
                (
                    "http://127.0.0.1:8090/v1/chat/completions"
                    if llm_provider == "local"
                    else "https://api.venice.ai/api/v1/chat/completions"
                ),
                environment,
                "VOICE_HARNESS_LLM_ENDPOINT",
            ),
            label="LLM endpoint",
        ),
        llm_timeout=_positive_float(
            _backend_value(
                llm, "timeout", 60, environment, "VOICE_HARNESS_LLM_TIMEOUT"
            ),
            label="LLM timeout",
        ),
        tts_provider=tts_provider,
        tts_model=_nonempty(
            _backend_value(
                tts,
                "model",
                "chatterbox-turbo" if tts_provider == "local" else "tts-kokoro",
                environment,
                "VOICE_HARNESS_TTS_MODEL",
            ),
            label="TTS model",
        ),
        tts_voice=_nonempty(
            _backend_value(
                tts,
                "voice",
                "default" if tts_provider == "local" else "af_sky",
                environment,
                "VOICE_HARNESS_TTS_VOICE",
            ),
            label="TTS voice",
        ),
        tts_speed=_tts_speed(
            _backend_value(
                tts,
                "speed",
                1.25 if tts_provider == "venice" else 1,
                environment,
                "VOICE_HARNESS_TTS_SPEED",
            )
        ),
        tts_endpoint=_nonempty(
            _backend_value(
                tts,
                "endpoint",
                "https://api.venice.ai/api/v1/audio/speech",
                environment,
                "VOICE_HARNESS_TTS_ENDPOINT",
            ),
            label="TTS endpoint",
        ),
        tts_timeout=_positive_float(
            _backend_value(
                tts, "timeout", 120, environment, "VOICE_HARNESS_TTS_TIMEOUT"
            ),
            label="TTS timeout",
        ),
    )


def load_backend_settings(
    environment: Mapping[str, str] = os.environ,
    *,
    path: Path | None = None,
    home: Path | None = None,
) -> BackendSettings:
    config_path = path or backend_config_path(environment, home=home)
    raw = read_toml_table(config_path, label="backend configuration")
    llm = raw.get("llm", {})
    tts = raw.get("tts", {})
    venice = raw.get("venice", {})
    if not (
        isinstance(llm, dict) and isinstance(tts, dict) and isinstance(venice, dict)
    ):
        raise BackendConfigurationError(
            "llm, tts, and venice configuration entries must be TOML tables"
        )
    reject_file_based_credentials(venice, environment)
    return backend_settings_from_tables(llm, tts, environment)


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
VOCABULARY_PATH = vocabulary_path()
WAV_PATH = STATE_DIR / "request.wav"
PID_PATH = STATE_DIR / "recording.pid"
WAKE_PID_PATH = STATE_DIR / "wake.pid"
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


AGENT_JOB_START_CONCURRENCY = _env_int(
    "VOICE_HARNESS_AGENT_JOB_START_CONCURRENCY",
    default=3,
    minimum=1,
)


def _env_nonnegative_float(
    name: str,
    *,
    default: float,
    environment: Mapping[str, str] = os.environ,
) -> float:
    raw = environment.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return value


def _env_positive_float(
    name: str,
    *,
    default: float,
    environment: Mapping[str, str] = os.environ,
) -> float:
    value = _env_nonnegative_float(name, default=default, environment=environment)
    if value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


CURSOR_AGENT_INACTIVITY_SECONDS = _env_positive_float(
    "VOICE_HARNESS_CURSOR_AGENT_INACTIVITY_SECONDS",
    default=15 * 60,
)
CURSOR_AGENT_MAX_RUNTIME_SECONDS = _env_positive_float(
    "VOICE_HARNESS_CURSOR_AGENT_MAX_RUNTIME_SECONDS",
    default=60 * 60,
)


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

# Completed-job follow-up context (GitHub issue #follow-on-jobs). When enabled,
# the wake daemon retains the last successfully announced completed Cursor job as
# a bounded, one-shot follow-up target. The kill switch disables the whole
# feature without affecting clarification replies or fresh submissions.
CURSOR_FOLLOWUP_ENABLED = _env_flag("VOICE_HARNESS_CURSOR_FOLLOWUP", default=True)
# Absolute lifetime of the retained completed-job reference, in seconds. It does
# not slide when unrelated conversation extends the conversation deadline.
CURSOR_FOLLOWUP_WINDOW_SECONDS = _env_nonnegative_float(
    "VOICE_HARNESS_CURSOR_FOLLOWUP_WINDOW_SECONDS",
    default=60,
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
