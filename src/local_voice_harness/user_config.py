"""Unified, validated user configuration.

This module introduces a single typed model for user-facing defaults instead of
spreading settings across ``backends.toml``, environment variables, backend
environment files, and systemd drop-ins. Existing installations that rely on
``backends.toml`` and environment overrides keep working unchanged; runtime
composition roots consume only the resolved typed snapshot.

Precedence, from lowest to highest, is::

    built-in defaults < config.toml < backends.toml < environment

Credentials are intentionally never read from configuration files; the Venice
API key stays in the desktop credential store (``voice-harness credentials``).
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import shlex
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import config
from .config import (
    BackendSettings,
    backend_settings_from_tables,
    read_toml_table,
    reject_file_based_credentials,
    xdg_config_home,
)

_TOP_LEVEL_SECTIONS = (
    "providers",
    "integrations",
    "compute",
    "audio",
    "dictation",
    "platform",
    "announcements",
)
_PROVIDER_TABLES = ("llm", "tts", "venice")
_LLM_KEYS = ("provider", "model", "endpoint", "timeout")
_TTS_KEYS = ("provider", "model", "voice", "speed", "endpoint", "timeout")
_INTEGRATION_KEYS = ("github", "zendesk", "linear")
_COMPUTE_KEYS = (
    "cuda_device",
    "llm_device",
    "tts_device",
    "dictation_device",
    "dictation_backend",
    "dictation_model",
    "dictation_quantization",
    "dictation_compute",
    "dictation_language",
)
_AUDIO_KEYS = (
    "source",
    "voice",
    "wake_threshold",
    "min_speech_rms",
    "barge_in_mode",
    "barge_in_speech_frames",
    "playback_quiet_frames",
    "playback_quiet_timeout_seconds",
    "playback_latency",
)
_DICTATION_KEYS = (
    "source",
    "inject",
    "prompt",
    "replacements",
    "vad_end_silence_ms",
    "vad_max_seconds",
    "vad_min_speech_rms",
    "vad_start_speech_frames",
)
_PLATFORM_KEYS = (
    "project_root",
    "github_root",
    "herdr_worktree_root",
    "gh_bin",
    "git_bin",
    "herdr_bin",
    "github_timeout_seconds",
    "herdr_timeout_seconds",
    "focused_app_context",
    "focused_app_deny_classes",
    "focused_app_max_chars",
    "cursor_followup",
    "cursor_followup_window_seconds",
    "cursor_foreground_seconds",
    "cursor_agent_inactivity_seconds",
    "cursor_agent_max_runtime_seconds",
    "cursor_mcp_auth_source",
    "agent_job_start_concurrency",
)
_ANNOUNCEMENT_KEYS = (
    "mode",
    "quiet_hours_start",
    "quiet_hours_end",
    "timezone",
)
_ANNOUNCEMENT_MODES = {"all", "action-required", "desktop-only", "quiet"}
_CLOCK_TIME = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}
_BARGE_IN_MODES = {"wake", "vad", "off"}
_DICTATION_BACKENDS = {"parakeet", "whisper"}
_DICTATION_DEVICES = {"auto", "cpu", "cuda"}
_DICTATION_LANGUAGES = {"en", "zh", "english", "chinese", "auto"}
_DICTATION_INJECT_MODES = {"auto", "paste", "type", "stdout"}
_PLAYBACK_LATENCY = re.compile(r"^\d+(?:\.\d+)?(?:us|ms|s)$")
_CREDENTIAL_KEYS = {"api_key", "api_key_file"}
_PLAN_APPROVAL_VERSION = 1


class UserConfigurationError(ValueError):
    """A unified configuration value is missing, malformed, or out of range."""


class PlanApprovalMode(StrEnum):
    """How reviewed Cursor Plan Mode Build gates are handled."""

    ASK = "ask"
    AUTO = "auto"


class AnnouncementMode(StrEnum):
    """When background job results may interrupt with speech."""

    ALL = "all"
    ACTION_REQUIRED = "action-required"
    DESKTOP_ONLY = "desktop-only"
    QUIET = "quiet"


class ComputeDevice(StrEnum):
    """Typed compute selector for local LLM, TTS, and dictation."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


DictationDevice = ComputeDevice


@dataclass(frozen=True)
class PlanApprovalPreferences:
    """Crash-safe learning state for Cursor plan approval."""

    version: int = _PLAN_APPROVAL_VERSION
    mode: PlanApprovalMode = PlanApprovalMode.ASK
    explicit_approval_ids: tuple[str, ...] = ()
    offer_pending_id: str | None = None
    offer_completed: bool = False

    @property
    def explicit_approval_count(self) -> int:
        return len(self.explicit_approval_ids)


@dataclass(frozen=True)
class IntegrationSettings:
    """Optional context and routing integrations.

    Optional integrations default to disabled on fresh installations except for
    GitHub, which remains enabled for compatibility. The integration registry
    enforces these flags before constructing providers.
    """

    github_enabled: bool = True
    zendesk_enabled: bool = False
    linear_enabled: bool = False


@dataclass(frozen=True)
class ComputeSettings:
    """GPU device and local LLM, TTS, and dictation compute selectors."""

    cuda_device: str = "CUDA0"
    llm_device: ComputeDevice = ComputeDevice.AUTO
    tts_device: ComputeDevice = ComputeDevice.AUTO
    dictation_device: DictationDevice = DictationDevice.AUTO
    dictation_backend: str = "parakeet"
    dictation_model: str = "nemo-parakeet-tdt-0.6b-v2"
    dictation_quantization: str = "int8"
    dictation_compute: str = "float16"
    dictation_language: str = "auto"


def resolve_local_compute(
    requested: ComputeDevice,
    *,
    cuda_available: bool,
    label: str,
) -> Literal["cpu", "cuda"]:
    """Resolve a typed local compute selector without probing on the CPU path.

    Callers must not query CUDA when ``requested`` is ``cpu``. ``auto`` and
    ``cuda`` receive a precomputed availability flag so tests can stay
    hardware-free.
    """

    if requested is ComputeDevice.CPU:
        return "cpu"
    if requested is ComputeDevice.CUDA and not cuda_available:
        raise RuntimeError(
            f"CUDA {label} was requested, but CUDA is unavailable; install the "
            "matching CUDA dependency profile and verify the NVIDIA driver"
        )
    if requested is ComputeDevice.CUDA:
        return "cuda"
    return "cuda" if cuda_available else "cpu"


@dataclass(frozen=True)
class AudioSettings:
    """Microphone, wake, barge-in, and playback tuning."""

    source: str = config.DEFAULT_SOURCE
    voice: str = ""
    wake_threshold: float = 0.55
    min_speech_rms: float = 1100.0
    barge_in_mode: str = "wake"
    barge_in_speech_frames: int = 5
    playback_quiet_frames: int = 4
    playback_quiet_timeout_seconds: float = 2.0
    playback_latency: str = "100ms"


DEFAULT_DICTATION_PROMPT = (
    "Technical software engineering dictation that may mention Cursor, Herdr, "
    "code, files, functions, terminals, and command-line tools."
)
DEFAULT_DICTATION_REPLACEMENTS = (
    ("herder", "herdr"),
    ("cursa", "Cursor"),
    ("curser", "Cursor"),
    ("service", "Jarvis"),
    ("jarvus", "Jarvis"),
    ("jervis", "Jarvis"),
)


@dataclass(frozen=True)
class DictationSettings:
    """Focused-window injection, transcription prompting, and VAD tuning."""

    source: str = config.DEFAULT_SOURCE
    inject: str = "auto"
    prompt: str = DEFAULT_DICTATION_PROMPT
    replacements: tuple[tuple[str, str], ...] = DEFAULT_DICTATION_REPLACEMENTS
    vad_end_silence_ms: float = 900.0
    vad_max_seconds: float = 120.0
    vad_min_speech_rms: float = 1100.0
    vad_start_speech_frames: int = 3


@dataclass(frozen=True)
class PlatformSettings:
    """Local trust boundaries and desktop capability toggles."""

    project_root: Path
    github_root: Path
    herdr_worktree_root: Path
    gh_bin: Path
    git_bin: Path
    herdr_bin: Path
    github_timeout_seconds: float = 30.0
    herdr_timeout_seconds: float = 30.0
    focused_app_context_enabled: bool = True
    focused_app_deny_classes: tuple[str, ...] = config.DEFAULT_FOCUSED_APP_DENY_CLASSES
    focused_app_max_chars: int = 12_000
    cursor_followup_enabled: bool = True
    cursor_followup_window_seconds: float = 60.0
    cursor_foreground_seconds: float = 5.0
    cursor_agent_inactivity_seconds: float = 15 * 60
    cursor_agent_max_runtime_seconds: float = 60 * 60
    cursor_mcp_auth_source: Path | None = None
    agent_job_start_concurrency: int = 3


@dataclass(frozen=True)
class AnnouncementSettings:
    """When and how background job results interrupt the user."""

    mode: AnnouncementMode = AnnouncementMode.ALL
    quiet_hours_start: str = ""
    quiet_hours_end: str = ""
    timezone: str = ""


@dataclass(frozen=True)
class UserConfig:
    """The complete, validated user configuration."""

    providers: BackendSettings
    integrations: IntegrationSettings
    compute: ComputeSettings
    audio: AudioSettings
    dictation: DictationSettings
    platform: PlatformSettings
    announcements: AnnouncementSettings


def user_config_path(
    environment: Mapping[str, str] = os.environ,
    *,
    home: Path | None = None,
) -> Path:
    """Return the path to the unified ``config.toml``."""

    return xdg_config_home(environment, home=home) / "voice-harness" / "config.toml"


def _home(home: Path | None) -> Path:
    return home if home is not None else Path.home()


def _section(raw: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise UserConfigurationError(f"[{name}] must be a TOML table")
    return value


def _table(section: Mapping[str, object], key: str, *, label: str) -> dict[str, object]:
    value = section.get(key, {})
    if not isinstance(value, Mapping):
        raise UserConfigurationError(f"[{label}] must be a TOML table")
    return {str(name): item for name, item in value.items()}


def _reject_unknown(
    section: Mapping[str, object], allowed: Iterable[str], *, label: str
) -> None:
    permitted = set(allowed)
    unknown = sorted(key for key in section if key not in permitted)
    if unknown:
        raise UserConfigurationError(f"unknown {label} key(s): {', '.join(unknown)}")


def _reject_file_credentials(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in _CREDENTIAL_KEYS:
                raise UserConfigurationError(
                    "credentials must not be stored in configuration files; run "
                    "`voice-harness credentials set`"
                )
            _reject_file_credentials(nested)


def _resolve(
    environment: Mapping[str, str],
    env_key: str,
    section: Mapping[str, object],
    key: str,
    default: object,
) -> object:
    if env_key in environment:
        return environment[env_key]
    if key in section:
        return section[key]
    return default


def _as_bool(value: object, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    raise UserConfigurationError(f"{label} must be a boolean")


def _as_positive_int(value: object, *, label: str) -> int:
    number = _as_int(value, label=label)
    if number < 1:
        raise UserConfigurationError(f"{label} must be a positive integer")
    return number


def _as_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise UserConfigurationError(f"{label} must be an integer")
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise UserConfigurationError(f"{label} must be an integer") from exc


def _as_nonnegative_float(value: object, *, label: str) -> float:
    try:
        number = float(str(value))
    except (TypeError, ValueError) as exc:
        raise UserConfigurationError(
            f"{label} must be a finite non-negative number"
        ) from exc
    if not math.isfinite(number) or number < 0:
        raise UserConfigurationError(f"{label} must be a finite non-negative number")
    return number


def _as_positive_float(value: object, *, label: str) -> float:
    number = _as_nonnegative_float(value, label=label)
    if number <= 0:
        raise UserConfigurationError(f"{label} must be a finite positive number")
    return number


def _as_float_in_range(value: object, *, label: str, low: float, high: float) -> float:
    try:
        number = float(str(value))
    except (TypeError, ValueError) as exc:
        raise UserConfigurationError(
            f"{label} must be a number between {low} and {high}"
        ) from exc
    if not math.isfinite(number) or number < low or number > high:
        raise UserConfigurationError(
            f"{label} must be a number between {low} and {high}"
        )
    return number


def _as_nonempty(value: object, *, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise UserConfigurationError(f"{label} must not be empty")
    return text


def _as_choice(value: object, choices: set[str], *, label: str) -> str:
    text = str(value).strip().casefold()
    if text not in choices:
        options = ", ".join(sorted(choices))
        raise UserConfigurationError(f"{label} must be one of: {options}")
    return text


def _as_path(value: object, *, label: str) -> Path:
    return Path(_as_nonempty(value, label=label)).expanduser()


def _as_optional_path(value: object, *, label: str) -> Path | None:
    text = str(value).strip()
    return Path(text).expanduser() if text else None


def _as_classes(value: object, *, label: str) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        parts: Iterable[str] = (str(item) for item in value)
    else:
        parts = str(value).split(",")
    return tuple(part for part in (piece.strip().casefold() for piece in parts) if part)


def _as_replacements(value: object, *, label: str) -> tuple[tuple[str, str], ...]:
    if isinstance(value, Mapping):
        entries = ((str(source), str(target)) for source, target in value.items())
    else:
        entries = (
            tuple(entry.split(":", 1))
            for entry in str(value).split(";")
            if entry.strip() and ":" in entry
        )
    replacements = tuple(
        (source.strip(), target.strip()) for source, target in entries if source.strip()
    )
    return replacements


def backend_environment_path(
    environment: Mapping[str, str] = os.environ,
    *,
    home: Path | None = None,
) -> Path:
    """Return the legacy dictation ``backend.env`` path."""

    return xdg_config_home(environment, home=home) / "dictation" / "backend.env"


def load_backend_environment(path: Path) -> dict[str, str]:
    """Parse the allowlisted legacy dictation selectors."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError) as exc:
        raise UserConfigurationError(
            f"could not read backend environment {path}: {exc}"
        ) from exc
    environment: dict[str, str] = {}
    allowed = {
        "DICTATION_BACKEND",
        "DICTATION_DEVICE",
        "DICTATION_MODEL",
        "DICTATION_LANGUAGE",
        "DICTATION_COMPUTE",
        "DICTATION_QUANTIZATION",
    }
    for line_number, raw_line in enumerate(lines, start=1):
        try:
            fields = shlex.split(raw_line, comments=True, posix=True)
        except ValueError as exc:
            raise UserConfigurationError(
                f"{path}:{line_number}: invalid environment assignment: {exc}"
            ) from exc
        if not fields:
            continue
        if len(fields) != 1 or "=" not in fields[0]:
            raise UserConfigurationError(
                f"{path}:{line_number}: invalid environment assignment"
            )
        key, value = fields[0].split("=", 1)
        if key not in allowed:
            raise UserConfigurationError(
                f"{path}:{line_number}: unsupported backend environment key {key!r}"
            )
        environment[key] = value
    return environment


def _resolve_legacy(
    environment: Mapping[str, str],
    legacy: Mapping[str, str],
    env_key: str,
    section: Mapping[str, object],
    key: str,
    default: object,
) -> object:
    if env_key in environment:
        return environment[env_key]
    if env_key in legacy:
        return legacy[env_key]
    return section.get(key, default)


def _load_integrations(
    section: Mapping[str, object], environment: Mapping[str, str]
) -> IntegrationSettings:
    _reject_unknown(section, _INTEGRATION_KEYS, label="[integrations]")
    return IntegrationSettings(
        github_enabled=_as_bool(
            _resolve(
                environment,
                "VOICE_HARNESS_INTEGRATION_GITHUB",
                section,
                "github",
                True,
            ),
            label="integrations.github",
        ),
        zendesk_enabled=_as_bool(
            _resolve(
                environment,
                "VOICE_HARNESS_INTEGRATION_ZENDESK",
                section,
                "zendesk",
                False,
            ),
            label="integrations.zendesk",
        ),
        linear_enabled=_as_bool(
            _resolve(
                environment,
                "VOICE_HARNESS_INTEGRATION_LINEAR",
                section,
                "linear",
                False,
            ),
            label="integrations.linear",
        ),
    )


def _load_compute(
    section: Mapping[str, object],
    environment: Mapping[str, str],
    legacy: Mapping[str, str],
) -> ComputeSettings:
    _reject_unknown(section, _COMPUTE_KEYS, label="[compute]")
    return ComputeSettings(
        cuda_device=_as_nonempty(
            _resolve(
                environment,
                "VOICE_HARNESS_CUDA_DEVICE",
                section,
                "cuda_device",
                "CUDA0",
            ),
            label="compute.cuda_device",
        ),
        llm_device=ComputeDevice(
            _as_choice(
                _resolve(
                    environment,
                    "VOICE_HARNESS_LLM_DEVICE",
                    section,
                    "llm_device",
                    ComputeDevice.AUTO,
                ),
                _DICTATION_DEVICES,
                label="compute.llm_device",
            )
        ),
        tts_device=ComputeDevice(
            _as_choice(
                _resolve(
                    environment,
                    "VOICE_HARNESS_TTS_DEVICE",
                    section,
                    "tts_device",
                    ComputeDevice.AUTO,
                ),
                _DICTATION_DEVICES,
                label="compute.tts_device",
            )
        ),
        dictation_device=DictationDevice(
            _as_choice(
                _resolve_legacy(
                    environment,
                    legacy,
                    "DICTATION_DEVICE",
                    section,
                    "dictation_device",
                    DictationDevice.AUTO,
                ),
                _DICTATION_DEVICES,
                label="compute.dictation_device",
            )
        ),
        dictation_backend=_as_choice(
            _resolve_legacy(
                environment,
                legacy,
                "DICTATION_BACKEND",
                section,
                "dictation_backend",
                "parakeet",
            ),
            _DICTATION_BACKENDS,
            label="compute.dictation_backend",
        ),
        dictation_model=_as_nonempty(
            _resolve_legacy(
                environment,
                legacy,
                "DICTATION_MODEL",
                section,
                "dictation_model",
                "nemo-parakeet-tdt-0.6b-v2",
            ),
            label="compute.dictation_model",
        ),
        dictation_quantization=_as_nonempty(
            _resolve_legacy(
                environment,
                legacy,
                "DICTATION_QUANTIZATION",
                section,
                "dictation_quantization",
                "int8",
            ),
            label="compute.dictation_quantization",
        ),
        dictation_compute=_as_nonempty(
            _resolve_legacy(
                environment,
                legacy,
                "DICTATION_COMPUTE",
                section,
                "dictation_compute",
                "float16",
            ),
            label="compute.dictation_compute",
        ),
        dictation_language=_as_choice(
            _resolve_legacy(
                environment,
                legacy,
                "DICTATION_LANGUAGE",
                section,
                "dictation_language",
                "auto",
            ),
            _DICTATION_LANGUAGES,
            label="compute.dictation_language",
        ),
    )


def _load_dictation(
    section: Mapping[str, object],
    environment: Mapping[str, str],
    *,
    default_source: str,
) -> DictationSettings:
    _reject_unknown(section, _DICTATION_KEYS, label="[dictation]")
    return DictationSettings(
        source=str(
            _resolve(
                environment,
                "DICTATION_SOURCE",
                section,
                "source",
                default_source,
            ),
        ).strip(),
        inject=_as_choice(
            _resolve(environment, "DICTATION_INJECT", section, "inject", "auto"),
            _DICTATION_INJECT_MODES,
            label="dictation.inject",
        ),
        prompt=str(
            _resolve(
                environment,
                "DICTATION_PROMPT",
                section,
                "prompt",
                DEFAULT_DICTATION_PROMPT,
            ),
        ).strip(),
        replacements=_as_replacements(
            _resolve(
                environment,
                "DICTATION_REPLACEMENTS",
                section,
                "replacements",
                ";".join(
                    f"{source}:{target}"
                    for source, target in DEFAULT_DICTATION_REPLACEMENTS
                ),
            ),
            label="dictation.replacements",
        ),
        vad_end_silence_ms=_as_positive_float(
            _resolve(
                environment,
                "DICTATION_VAD_END_SILENCE_MS",
                section,
                "vad_end_silence_ms",
                900,
            ),
            label="dictation.vad_end_silence_ms",
        ),
        vad_max_seconds=_as_positive_float(
            _resolve(
                environment,
                "DICTATION_VAD_MAX_SECONDS",
                section,
                "vad_max_seconds",
                120,
            ),
            label="dictation.vad_max_seconds",
        ),
        vad_min_speech_rms=_as_nonnegative_float(
            _resolve(
                environment,
                "DICTATION_VAD_MIN_SPEECH_RMS",
                section,
                "vad_min_speech_rms",
                1100,
            ),
            label="dictation.vad_min_speech_rms",
        ),
        vad_start_speech_frames=_as_positive_int(
            _resolve(
                environment,
                "DICTATION_VAD_START_SPEECH_FRAMES",
                section,
                "vad_start_speech_frames",
                3,
            ),
            label="dictation.vad_start_speech_frames",
        ),
    )


def _load_audio(
    section: Mapping[str, object], environment: Mapping[str, str]
) -> AudioSettings:
    _reject_unknown(section, _AUDIO_KEYS, label="[audio]")
    playback_latency = str(
        _resolve(
            environment,
            "VOICE_HARNESS_PLAYBACK_LATENCY",
            section,
            "playback_latency",
            "100ms",
        )
    ).strip()
    if not _PLAYBACK_LATENCY.fullmatch(playback_latency):
        raise UserConfigurationError(
            "audio.playback_latency must be a non-negative duration ending in "
            "us, ms, or s"
        )
    return AudioSettings(
        source=_as_nonempty(
            _resolve(
                environment,
                "VOICE_HARNESS_SOURCE",
                section,
                "source",
                config.DEFAULT_SOURCE,
            ),
            label="audio.source",
        ),
        voice=str(
            _resolve(environment, "VOICE_HARNESS_VOICE", section, "voice", "")
        ).strip(),
        wake_threshold=_as_float_in_range(
            _resolve(
                environment,
                "VOICE_HARNESS_WAKE_THRESHOLD",
                section,
                "wake_threshold",
                0.55,
            ),
            label="audio.wake_threshold",
            low=0.0,
            high=1.0,
        ),
        min_speech_rms=_as_nonnegative_float(
            _resolve(
                environment,
                "VOICE_HARNESS_MIN_SPEECH_RMS",
                section,
                "min_speech_rms",
                1100,
            ),
            label="audio.min_speech_rms",
        ),
        barge_in_mode=_as_choice(
            _resolve(
                environment,
                "VOICE_HARNESS_BARGE_IN_MODE",
                section,
                "barge_in_mode",
                "wake",
            ),
            _BARGE_IN_MODES,
            label="audio.barge_in_mode",
        ),
        barge_in_speech_frames=_as_positive_int(
            _resolve(
                environment,
                "VOICE_HARNESS_BARGE_IN_SPEECH_FRAMES",
                section,
                "barge_in_speech_frames",
                5,
            ),
            label="audio.barge_in_speech_frames",
        ),
        playback_quiet_frames=_as_positive_int(
            _resolve(
                environment,
                "VOICE_HARNESS_PLAYBACK_QUIET_FRAMES",
                section,
                "playback_quiet_frames",
                4,
            ),
            label="audio.playback_quiet_frames",
        ),
        playback_quiet_timeout_seconds=_as_nonnegative_float(
            _resolve(
                environment,
                "VOICE_HARNESS_PLAYBACK_QUIET_TIMEOUT_SECONDS",
                section,
                "playback_quiet_timeout_seconds",
                2,
            ),
            label="audio.playback_quiet_timeout_seconds",
        ),
        playback_latency=playback_latency,
    )


def _load_platform(
    section: Mapping[str, object],
    environment: Mapping[str, str],
    *,
    home: Path,
) -> PlatformSettings:
    _reject_unknown(section, _PLATFORM_KEYS, label="[platform]")
    settings = PlatformSettings(
        project_root=_as_path(
            _resolve(
                environment, "VOICE_HARNESS_PROJECT_ROOT", section, "project_root", home
            ),
            label="platform.project_root",
        ),
        github_root=_as_path(
            _resolve(
                environment,
                "VOICE_HARNESS_GITHUB_ROOT",
                section,
                "github_root",
                home / "src",
            ),
            label="platform.github_root",
        ),
        herdr_worktree_root=_as_path(
            _resolve(
                environment,
                "VOICE_HARNESS_HERDR_WORKTREE_ROOT",
                section,
                "herdr_worktree_root",
                home / ".herdr" / "worktrees",
            ),
            label="platform.herdr_worktree_root",
        ),
        gh_bin=_as_path(
            _resolve(
                environment,
                "VOICE_HARNESS_GH_BIN",
                section,
                "gh_bin",
                Path("gh"),
            ),
            label="platform.gh_bin",
        ),
        git_bin=_as_path(
            _resolve(
                environment,
                "VOICE_HARNESS_GIT_BIN",
                section,
                "git_bin",
                Path("git"),
            ),
            label="platform.git_bin",
        ),
        herdr_bin=_as_path(
            _resolve(
                environment,
                "VOICE_HARNESS_HERDR_BIN",
                section,
                "herdr_bin",
                home / ".local" / "bin" / "herdr",
            ),
            label="platform.herdr_bin",
        ),
        github_timeout_seconds=_as_positive_float(
            _resolve(
                environment,
                "VOICE_HARNESS_GITHUB_TIMEOUT_SECONDS",
                section,
                "github_timeout_seconds",
                30,
            ),
            label="platform.github_timeout_seconds",
        ),
        herdr_timeout_seconds=_as_positive_float(
            _resolve(
                environment,
                "VOICE_HARNESS_HERDR_TIMEOUT_SECONDS",
                section,
                "herdr_timeout_seconds",
                30,
            ),
            label="platform.herdr_timeout_seconds",
        ),
        focused_app_context_enabled=_as_bool(
            _resolve(
                environment,
                "VOICE_HARNESS_FOCUSED_APP_CONTEXT",
                section,
                "focused_app_context",
                True,
            ),
            label="platform.focused_app_context",
        ),
        focused_app_deny_classes=_as_classes(
            _resolve(
                environment,
                "VOICE_HARNESS_FOCUSED_APP_DENY",
                section,
                "focused_app_deny_classes",
                list(config.DEFAULT_FOCUSED_APP_DENY_CLASSES),
            ),
            label="platform.focused_app_deny_classes",
        ),
        focused_app_max_chars=_as_positive_int(
            _resolve(
                environment,
                "VOICE_HARNESS_FOCUSED_APP_MAX_CHARS",
                section,
                "focused_app_max_chars",
                12_000,
            ),
            label="platform.focused_app_max_chars",
        ),
        cursor_followup_enabled=_as_bool(
            _resolve(
                environment,
                "VOICE_HARNESS_CURSOR_FOLLOWUP",
                section,
                "cursor_followup",
                True,
            ),
            label="platform.cursor_followup",
        ),
        cursor_followup_window_seconds=_as_nonnegative_float(
            _resolve(
                environment,
                "VOICE_HARNESS_CURSOR_FOLLOWUP_WINDOW_SECONDS",
                section,
                "cursor_followup_window_seconds",
                60,
            ),
            label="platform.cursor_followup_window_seconds",
        ),
        cursor_foreground_seconds=_as_nonnegative_float(
            _resolve(
                environment,
                "VOICE_HARNESS_CURSOR_FOREGROUND_SECONDS",
                section,
                "cursor_foreground_seconds",
                5,
            ),
            label="platform.cursor_foreground_seconds",
        ),
        cursor_agent_inactivity_seconds=_as_positive_float(
            _resolve(
                environment,
                "VOICE_HARNESS_CURSOR_AGENT_INACTIVITY_SECONDS",
                section,
                "cursor_agent_inactivity_seconds",
                15 * 60,
            ),
            label="platform.cursor_agent_inactivity_seconds",
        ),
        cursor_agent_max_runtime_seconds=_as_positive_float(
            _resolve(
                environment,
                "VOICE_HARNESS_CURSOR_AGENT_MAX_RUNTIME_SECONDS",
                section,
                "cursor_agent_max_runtime_seconds",
                60 * 60,
            ),
            label="platform.cursor_agent_max_runtime_seconds",
        ),
        cursor_mcp_auth_source=_as_optional_path(
            _resolve(
                environment,
                "VOICE_HARNESS_CURSOR_MCP_AUTH_SOURCE",
                section,
                "cursor_mcp_auth_source",
                "",
            ),
            label="platform.cursor_mcp_auth_source",
        ),
        agent_job_start_concurrency=_as_positive_int(
            _resolve(
                environment,
                "VOICE_HARNESS_AGENT_JOB_START_CONCURRENCY",
                section,
                "agent_job_start_concurrency",
                3,
            ),
            label="platform.agent_job_start_concurrency",
        ),
    )
    for label, path in (
        ("platform.project_root", settings.project_root),
        ("platform.github_root", settings.github_root),
        ("platform.herdr_worktree_root", settings.herdr_worktree_root),
        ("platform.herdr_bin", settings.herdr_bin),
    ):
        if not path.is_absolute():
            raise UserConfigurationError(f"{label} must be an absolute path")
    if (
        settings.cursor_mcp_auth_source is not None
        and not settings.cursor_mcp_auth_source.is_absolute()
    ):
        raise UserConfigurationError(
            "platform.cursor_mcp_auth_source must be an absolute path"
        )
    try:
        settings.github_root.resolve().relative_to(settings.project_root.resolve())
    except ValueError as exc:
        raise UserConfigurationError(
            "platform.github_root must be inside platform.project_root"
        ) from exc
    return settings


def _load_announcements(
    section: Mapping[str, object], environment: Mapping[str, str]
) -> AnnouncementSettings:
    _reject_unknown(section, _ANNOUNCEMENT_KEYS, label="[announcements]")
    mode_text = (
        str(
            _resolve(
                environment,
                "VOICE_HARNESS_ANNOUNCEMENT_MODE",
                section,
                "mode",
                AnnouncementMode.ALL.value,
            )
        )
        .strip()
        .casefold()
    )
    if mode_text not in _ANNOUNCEMENT_MODES:
        raise UserConfigurationError(
            "announcements.mode must be one of: all, action-required, "
            "desktop-only, quiet"
        )
    start = str(
        _resolve(
            environment,
            "VOICE_HARNESS_QUIET_HOURS_START",
            section,
            "quiet_hours_start",
            "",
        )
    ).strip()
    end = str(
        _resolve(
            environment,
            "VOICE_HARNESS_QUIET_HOURS_END",
            section,
            "quiet_hours_end",
            "",
        )
    ).strip()
    if bool(start) != bool(end):
        raise UserConfigurationError(
            "announcements.quiet_hours_start and quiet_hours_end must be set together"
        )
    for label, value in (
        ("announcements.quiet_hours_start", start),
        ("announcements.quiet_hours_end", end),
    ):
        if value and not _CLOCK_TIME.fullmatch(value):
            raise UserConfigurationError(f"{label} must be HH:MM in 24-hour local time")
    timezone = str(
        _resolve(
            environment,
            "VOICE_HARNESS_ANNOUNCEMENT_TIMEZONE",
            section,
            "timezone",
            "",
        )
    ).strip()
    if timezone:
        try:
            ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, KeyError, ValueError) as exc:
            raise UserConfigurationError(
                "announcements.timezone must be a valid IANA timezone name"
            ) from exc
    return AnnouncementSettings(
        mode=AnnouncementMode(mode_text),
        quiet_hours_start=start,
        quiet_hours_end=end,
        timezone=timezone,
    )


def _load_providers(
    section: Mapping[str, object],
    environment: Mapping[str, str],
    *,
    backends_path: Path,
) -> BackendSettings:
    _reject_unknown(section, _PROVIDER_TABLES, label="[providers]")
    config_llm = _table(section, "llm", label="providers.llm")
    config_tts = _table(section, "tts", label="providers.tts")
    _reject_unknown(config_llm, _LLM_KEYS, label="[providers.llm]")
    _reject_unknown(config_tts, _TTS_KEYS, label="[providers.tts]")

    backends = read_toml_table(
        backends_path,
        error=UserConfigurationError,
        label="backend configuration",
    )
    backends_llm = _table(backends, "llm", label="backends llm")
    backends_tts = _table(backends, "tts", label="backends tts")
    backends_venice = _table(backends, "venice", label="backends venice")

    merged_llm = {**config_llm, **backends_llm}
    merged_tts = {**config_tts, **backends_tts}
    try:
        reject_file_based_credentials(backends_venice, environment)
        return backend_settings_from_tables(merged_llm, merged_tts, environment)
    except config.BackendConfigurationError as exc:
        raise UserConfigurationError(str(exc)) from exc


def load_user_config(
    environment: Mapping[str, str] = os.environ,
    *,
    path: Path | None = None,
    backends_path: Path | None = None,
    backend_env_path: Path | None = None,
    home: Path | None = None,
) -> UserConfig:
    """Load and validate the unified configuration.

    Precedence is built-in defaults, then ``config.toml``, then legacy
    ``backends.toml``/``backend.env`` inputs, then environment overrides.
    """

    resolved_home = _home(home)
    config_path = path or user_config_path(environment, home=resolved_home)
    backends = backends_path or config.backend_config_path(
        environment, home=resolved_home
    )
    backend_env = backend_env_path or backend_environment_path(
        environment, home=resolved_home
    )

    raw = read_toml_table(
        config_path, error=UserConfigurationError, label="user configuration"
    )
    _reject_unknown(raw, _TOP_LEVEL_SECTIONS, label="configuration section")
    _reject_file_credentials(raw)
    legacy_compute = load_backend_environment(backend_env)
    audio = _load_audio(_section(raw, "audio"), environment)

    return UserConfig(
        providers=_load_providers(
            _section(raw, "providers"), environment, backends_path=backends
        ),
        integrations=_load_integrations(_section(raw, "integrations"), environment),
        compute=_load_compute(_section(raw, "compute"), environment, legacy_compute),
        audio=audio,
        dictation=_load_dictation(
            _section(raw, "dictation"), environment, default_source=audio.source
        ),
        platform=_load_platform(
            _section(raw, "platform"), environment, home=resolved_home
        ),
        announcements=_load_announcements(_section(raw, "announcements"), environment),
    )


def default_user_config(home: Path | None = None) -> UserConfig:
    """Return the built-in configuration with no files or environment applied."""

    empty = Path(os.devnull)
    return load_user_config(
        {}, path=empty, backends_path=empty, backend_env_path=empty, home=home
    )


def _toml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _toml_array(values: Iterable[object]) -> str:
    return "[" + ", ".join(_toml_scalar(value) for value in values) + "]"


def _render_table(name: str, entries: Mapping[str, object]) -> list[str]:
    lines = [f"[{name}]"]
    for key, value in entries.items():
        if isinstance(value, (list, tuple)):
            lines.append(f"{key} = {_toml_array(value)}")
        else:
            lines.append(f"{key} = {_toml_scalar(value)}")
    lines.append("")
    return lines


def render_user_config(user_config: UserConfig) -> str:
    """Serialize a :class:`UserConfig` to deterministic TOML text."""

    providers = user_config.providers
    audio = user_config.audio
    dictation = user_config.dictation
    compute = user_config.compute
    platform = user_config.platform
    integrations = user_config.integrations

    lines: list[str] = []
    lines += _render_table(
        "providers.llm",
        {
            "provider": providers.llm_provider,
            "model": providers.llm_model,
            "endpoint": providers.llm_endpoint,
            "timeout": providers.llm_timeout,
        },
    )
    lines += _render_table(
        "providers.tts",
        {
            "provider": providers.tts_provider,
            "model": providers.tts_model,
            "voice": providers.tts_voice,
            "speed": providers.tts_speed,
            "endpoint": providers.tts_endpoint,
            "timeout": providers.tts_timeout,
        },
    )
    lines += _render_table(
        "integrations",
        {
            "github": integrations.github_enabled,
            "zendesk": integrations.zendesk_enabled,
            "linear": integrations.linear_enabled,
        },
    )
    lines += _render_table(
        "compute",
        {
            "cuda_device": compute.cuda_device,
            "llm_device": compute.llm_device,
            "tts_device": compute.tts_device,
            "dictation_device": compute.dictation_device,
            "dictation_backend": compute.dictation_backend,
            "dictation_model": compute.dictation_model,
            "dictation_quantization": compute.dictation_quantization,
            "dictation_compute": compute.dictation_compute,
            "dictation_language": compute.dictation_language,
        },
    )
    lines += _render_table(
        "audio",
        {
            "source": audio.source,
            "voice": audio.voice,
            "wake_threshold": audio.wake_threshold,
            "min_speech_rms": audio.min_speech_rms,
            "barge_in_mode": audio.barge_in_mode,
            "barge_in_speech_frames": audio.barge_in_speech_frames,
            "playback_quiet_frames": audio.playback_quiet_frames,
            "playback_quiet_timeout_seconds": audio.playback_quiet_timeout_seconds,
            "playback_latency": audio.playback_latency,
        },
    )
    lines += _render_table(
        "dictation",
        {
            "source": dictation.source,
            "inject": dictation.inject,
            "prompt": dictation.prompt,
            "replacements": ";".join(
                f"{source}:{target}" for source, target in dictation.replacements
            ),
            "vad_end_silence_ms": dictation.vad_end_silence_ms,
            "vad_max_seconds": dictation.vad_max_seconds,
            "vad_min_speech_rms": dictation.vad_min_speech_rms,
            "vad_start_speech_frames": dictation.vad_start_speech_frames,
        },
    )
    lines += _render_table(
        "platform",
        {
            "project_root": str(platform.project_root),
            "github_root": str(platform.github_root),
            "herdr_worktree_root": str(platform.herdr_worktree_root),
            "gh_bin": str(platform.gh_bin),
            "git_bin": str(platform.git_bin),
            "herdr_bin": str(platform.herdr_bin),
            "github_timeout_seconds": platform.github_timeout_seconds,
            "herdr_timeout_seconds": platform.herdr_timeout_seconds,
            "focused_app_context": platform.focused_app_context_enabled,
            "focused_app_deny_classes": list(platform.focused_app_deny_classes),
            "focused_app_max_chars": platform.focused_app_max_chars,
            "cursor_followup": platform.cursor_followup_enabled,
            "cursor_followup_window_seconds": platform.cursor_followup_window_seconds,
            "cursor_foreground_seconds": platform.cursor_foreground_seconds,
            "cursor_agent_inactivity_seconds": (
                platform.cursor_agent_inactivity_seconds
            ),
            "cursor_agent_max_runtime_seconds": platform.cursor_agent_max_runtime_seconds,
            "cursor_mcp_auth_source": (
                str(platform.cursor_mcp_auth_source)
                if platform.cursor_mcp_auth_source is not None
                else ""
            ),
            "agent_job_start_concurrency": platform.agent_job_start_concurrency,
        },
    )
    announcements = user_config.announcements
    lines += _render_table(
        "announcements",
        {
            "mode": announcements.mode.value,
            "quiet_hours_start": announcements.quiet_hours_start,
            "quiet_hours_end": announcements.quiet_hours_end,
            "timezone": announcements.timezone,
        },
    )
    return "\n".join(lines).rstrip("\n") + "\n"


def write_user_config(
    user_config: UserConfig,
    path: Path,
) -> None:
    """Atomically write ``user_config`` to ``path`` with owner-only permissions.

    Providers are serialized into the ``[providers.*]`` tables of the unified
    file, so credentials are never written; the Venice key stays in the store.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = render_user_config(user_config)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def _fsync_directory(directory: Path) -> None:
    """Flush a directory entry so a rename survives a crash (best effort)."""

    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def plan_approval_preferences_path(
    environment: Mapping[str, str] = os.environ,
    *,
    home: Path | None = None,
) -> Path:
    """Return the private learned-preference store path."""

    override = environment.get("VOICE_HARNESS_PLAN_APPROVAL_FILE", "").strip()
    if override:
        return Path(override).expanduser()
    return (
        xdg_config_home(environment, home=home) / "voice-harness" / "plan-approval.json"
    )


def _plan_approval_path(path: Path | None) -> Path:
    return path if path is not None else plan_approval_preferences_path()


def _parse_plan_approval_preferences(raw: object) -> PlanApprovalPreferences:
    if not isinstance(raw, dict):
        raise UserConfigurationError("plan-approval preferences must be an object")
    if raw.get("version") != _PLAN_APPROVAL_VERSION:
        raise UserConfigurationError("unsupported plan-approval preference version")
    try:
        mode = PlanApprovalMode(str(raw.get("mode") or PlanApprovalMode.ASK))
    except ValueError as exc:
        raise UserConfigurationError("plan-approval mode must be ask or auto") from exc
    ids = raw.get("explicit_approval_ids", [])
    if (
        not isinstance(ids, list)
        or any(not isinstance(value, str) or not value for value in ids)
        or len(ids) != len(set(ids))
        or len(ids) > 3
    ):
        raise UserConfigurationError(
            "plan-approval explicit approval IDs must be up to three unique strings"
        )
    pending = raw.get("offer_pending_id")
    if pending is not None and (not isinstance(pending, str) or not pending):
        raise UserConfigurationError(
            "plan-approval pending offer ID must be non-empty text"
        )
    completed = raw.get("offer_completed", False)
    if not isinstance(completed, bool):
        raise UserConfigurationError(
            "plan-approval offer completion flag must be a boolean"
        )
    if pending is not None and (len(ids) < 3 or completed):
        raise UserConfigurationError(
            "plan-approval pending offer requires three approvals and no completion"
        )
    if len(ids) == 3 and not completed and pending is None:
        raise UserConfigurationError(
            "plan-approval threshold requires a pending preference offer"
        )
    if mode == PlanApprovalMode.AUTO and (len(ids) < 3 or not completed):
        raise UserConfigurationError(
            "automatic plan approval requires an accepted threshold offer"
        )
    return PlanApprovalPreferences(
        mode=mode,
        explicit_approval_ids=tuple(ids),
        offer_pending_id=pending,
        offer_completed=completed,
    )


def _read_plan_approval_unlocked(path: Path) -> PlanApprovalPreferences:
    try:
        payload = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return PlanApprovalPreferences()
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise UserConfigurationError(
            f"plan-approval preferences {path} are not valid JSON"
        ) from exc
    return _parse_plan_approval_preferences(raw)


@contextmanager
def _locked_plan_approval(path: Path) -> Iterator[None]:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    with lock_path.open("a+b") as lock:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _write_plan_approval_unlocked(
    preferences: PlanApprovalPreferences, path: Path
) -> None:
    payload = (
        json.dumps(
            {
                "version": preferences.version,
                "mode": preferences.mode.value,
                "explicit_approval_ids": list(preferences.explicit_approval_ids),
                "offer_pending_id": preferences.offer_pending_id,
                "offer_completed": preferences.offer_completed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=".plan-approval-",
        suffix=".json",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_plan_approval_preferences(
    path: Path | None = None,
) -> PlanApprovalPreferences:
    """Load learned plan-approval state, defaulting safely to ask mode."""

    location = _plan_approval_path(path)
    with _locked_plan_approval(location):
        return _read_plan_approval_unlocked(location)


PlanApprovalChange = Callable[[PlanApprovalPreferences], PlanApprovalPreferences]


def update_plan_approval_preferences(
    change: PlanApprovalChange,
    *,
    path: Path | None = None,
) -> PlanApprovalPreferences:
    """Atomically update learned plan-approval state under a file lock."""

    location = _plan_approval_path(path)
    with _locked_plan_approval(location):
        current = _read_plan_approval_unlocked(location)
        updated = change(current)
        if updated.version != _PLAN_APPROVAL_VERSION:
            raise UserConfigurationError(
                "plan-approval update changed the preference version"
            )
        _parse_plan_approval_preferences(
            {
                "version": updated.version,
                "mode": updated.mode.value,
                "explicit_approval_ids": list(updated.explicit_approval_ids),
                "offer_pending_id": updated.offer_pending_id,
                "offer_completed": updated.offer_completed,
            }
        )
        if updated != current:
            _write_plan_approval_unlocked(updated, location)
        return updated


def record_explicit_plan_approval(
    approval_id: str,
    *,
    path: Path | None = None,
) -> PlanApprovalPreferences:
    """Record one accepted explicit Herdr submission exactly once."""

    if not approval_id:
        raise UserConfigurationError("plan approval ID must not be empty")

    def record(current: PlanApprovalPreferences) -> PlanApprovalPreferences:
        if approval_id in current.explicit_approval_ids:
            return current
        ids = current.explicit_approval_ids
        if len(ids) < 3:
            ids = (*ids, approval_id)
        pending = current.offer_pending_id
        if len(ids) == 3 and not current.offer_completed and pending is None:
            pending = approval_id
        return PlanApprovalPreferences(
            mode=current.mode,
            explicit_approval_ids=ids,
            offer_pending_id=pending,
            offer_completed=current.offer_completed,
        )

    return update_plan_approval_preferences(record, path=path)


def resolve_plan_approval_offer(
    offer_id: str,
    *,
    approved: bool,
    path: Path | None = None,
) -> PlanApprovalPreferences:
    """Resolve the one threshold offer, rejecting stale offer identities."""

    def resolve(current: PlanApprovalPreferences) -> PlanApprovalPreferences:
        if current.offer_pending_id != offer_id:
            raise UserConfigurationError(
                "that plan-approval preference offer is no longer pending"
            )
        return PlanApprovalPreferences(
            mode=PlanApprovalMode.AUTO if approved else PlanApprovalMode.ASK,
            explicit_approval_ids=current.explicit_approval_ids,
            offer_pending_id=None,
            offer_completed=True,
        )

    return update_plan_approval_preferences(resolve, path=path)


def set_plan_approval_mode(
    mode: PlanApprovalMode,
    *,
    path: Path | None = None,
) -> PlanApprovalPreferences:
    """Set the user-selected mode without disturbing the approval ledger."""

    return update_plan_approval_preferences(
        lambda current: PlanApprovalPreferences(
            mode=mode,
            explicit_approval_ids=current.explicit_approval_ids,
            offer_pending_id=current.offer_pending_id,
            offer_completed=current.offer_completed,
        ),
        path=path,
    )
