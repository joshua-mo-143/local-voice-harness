"""Typed, read-only conversational self-management operations."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from .config_management import (
    ConfigChangeResult,
    ConfigPaths,
    apply_config_values,
    commit_config_change,
    config_paths,
    describe_config_value,
    load_managed_config,
    restart_services_for_keys,
)
from .responses import AssistantResponse
from .user_config import UserConfig, UserConfigurationError

UNSUPPORTED_INSPECTION_RESPONSE = (
    "I can only inspect one supported harness setting at a time: voice, "
    "speaking speed, barge-in mode, announcement policy, or whether GitHub, "
    "Linear, or Zendesk is enabled."
)
AMBIGUOUS_INSPECTION_RESPONSE = (
    "I couldn't identify exactly one supported harness setting. Ask about voice, "
    "speaking speed, barge-in mode, announcement policy, or one supported "
    "integration."
)
_MAX_SPOKEN_VALUE_CHARS = 80
_SAFE_VOICE_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_SENSITIVE_OR_EXECUTABLE = re.compile(
    r"\b(?:api\s*key|credential|secret|password|token|path|executable|binary|command)\b",
    re.IGNORECASE,
)


class SelfManagementIntent(StrEnum):
    """Provider-neutral first-party operations supported by the harness."""

    INSPECT_CONFIG = "inspect_config"
    CHANGE_CONFIG = "change_config"


class InspectionStatus(StrEnum):
    """Outcome of resolving and executing a configuration inspection."""

    OK = "ok"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"


class ChangePreparationStatus(StrEnum):
    """Outcome of extracting and validating a requested configuration change."""

    READY = "ready"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"
    MALFORMED = "malformed"
    INVALID = "invalid"
    CONFLICT = "conflict"
    NO_CHANGE = "no_change"


class ConfirmationDecision(StrEnum):
    """Strict whole-utterance decisions accepted for a pending write."""

    CONFIRM = "confirm"
    CANCEL = "cancel"
    AMBIGUOUS = "ambiguous"


class SettingKey(StrEnum):
    """Configuration keys intentionally exposed to conversation."""

    VOICE = "audio.voice"
    TTS_SPEED = "providers.tts.speed"
    BARGE_IN_MODE = "audio.barge_in_mode"
    ANNOUNCEMENT_MODE = "announcements.mode"
    GITHUB = "integrations.github"
    LINEAR = "integrations.linear"
    ZENDESK = "integrations.zendesk"


@dataclass(frozen=True, slots=True)
class SelfManagementRequest:
    """One operation authorized only by its original user utterance."""

    intent: SelfManagementIntent
    utterance: str


@dataclass(frozen=True, slots=True)
class InspectionResult:
    """Typed result of a bounded allow-listed configuration read."""

    status: InspectionStatus
    setting: SettingKey | None = None
    value: str | bool | None = None


@dataclass(frozen=True, slots=True)
class ConfigChangeRequest:
    """One extracted change tied to the trusted original utterance."""

    utterance: str
    raw_value: str | None


@dataclass(frozen=True, slots=True)
class PendingConfigChange:
    """One validated change awaiting an explicit user decision."""

    trusted_utterance: str
    setting: SettingKey
    raw_value: str
    old_value: str | bool
    new_value: str | bool
    stored_value: str | bool | float
    affected_services: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ChangePreparation:
    """Typed preflight result; only ``READY`` carries pending state."""

    status: ChangePreparationStatus
    pending: PendingConfigChange | None = None


@dataclass(frozen=True, slots=True)
class SettingSpec:
    """User-facing aliases and typed snapshot accessor for one setting."""

    key: SettingKey
    aliases: tuple[str, ...]
    describe: Callable[[UserConfig], str | bool]


SETTING_REGISTRY = (
    SettingSpec(
        SettingKey.VOICE,
        ("voice", "assistant voice", "speaking voice", "audio voice"),
        lambda config: config.audio.voice,
    ),
    SettingSpec(
        SettingKey.TTS_SPEED,
        (
            "speaking speed",
            "tts speed",
            "speech speed",
            "speed",
        ),
        lambda config: format(config.providers.tts_speed, "g"),
    ),
    SettingSpec(
        SettingKey.BARGE_IN_MODE,
        (
            "barge in",
            "barge in mode",
            "interruption mode",
            "interrupt mode",
        ),
        lambda config: config.audio.barge_in_mode,
    ),
    SettingSpec(
        SettingKey.ANNOUNCEMENT_MODE,
        (
            "announcement policy",
            "announcement mode",
            "background announcements",
            "quiet mode",
        ),
        lambda config: config.announcements.mode.value,
    ),
    SettingSpec(
        SettingKey.GITHUB,
        ("github", "git hub"),
        lambda config: config.integrations.github_enabled,
    ),
    SettingSpec(
        SettingKey.LINEAR,
        ("linear",),
        lambda config: config.integrations.linear_enabled,
    ),
    SettingSpec(
        SettingKey.ZENDESK,
        ("zendesk", "zen desk"),
        lambda config: config.integrations.zendesk_enabled,
    ),
)
_SPECS_BY_KEY = {spec.key: spec for spec in SETTING_REGISTRY}


def _normalize(value: str) -> str:
    value = re.sub(r"[_-]+", " ", value.casefold())
    value = re.sub(r"[^\w\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def resolve_setting_alias(utterance: str) -> SettingKey | InspectionStatus:
    """Resolve exactly one allow-listed alias without semantic guessing."""

    normalized = f" {_normalize(utterance)} "
    matches = {
        spec.key
        for spec in SETTING_REGISTRY
        if any(f" {_normalize(alias)} " in normalized for alias in spec.aliases)
    }
    if not matches:
        return InspectionStatus.UNSUPPORTED
    if len(matches) != 1:
        return InspectionStatus.AMBIGUOUS
    return next(iter(matches))


def inspect_config(
    request: SelfManagementRequest,
    config: UserConfig,
) -> InspectionResult:
    """Read one setting from the supplied immutable runtime snapshot."""

    if request.intent != SelfManagementIntent.INSPECT_CONFIG:
        return InspectionResult(InspectionStatus.UNSUPPORTED)
    resolution = resolve_setting_alias(request.utterance)
    if isinstance(resolution, InspectionStatus):
        return InspectionResult(resolution)
    spec = _SPECS_BY_KEY[resolution]
    return InspectionResult(InspectionStatus.OK, resolution, spec.describe(config))


def read_setting(config: UserConfig, setting: SettingKey) -> str | bool:
    """Read one allow-listed value from a typed configuration snapshot."""

    return _SPECS_BY_KEY[setting].describe(config)


def _normalize_raw_value(setting: SettingKey, raw_value: str) -> str:
    value = raw_value.strip().strip("\"'")
    normalized = _normalize(value)
    if setting in {SettingKey.GITHUB, SettingKey.LINEAR, SettingKey.ZENDESK}:
        aliases = {
            "enable": "true",
            "enabled": "true",
            "disable": "false",
            "disabled": "false",
        }
        return aliases.get(normalized, value)
    if setting == SettingKey.BARGE_IN_MODE and normalized == "wake word":
        return "wake"
    if setting == SettingKey.ANNOUNCEMENT_MODE:
        aliases = {
            "all": "all",
            "everything": "all",
            "speak all": "all",
            "action required": "action-required",
            "action-required": "action-required",
            "desktop only": "desktop-only",
            "desktop-only": "desktop-only",
            "notifications only": "desktop-only",
            "quiet": "quiet",
            "quiet mode": "quiet",
        }
        return aliases.get(normalized, value)
    return value


def prepare_config_change(
    request: ConfigChangeRequest,
    active_config: UserConfig,
    *,
    paths: ConfigPaths | None = None,
    environment: Mapping[str, str] | None = None,
) -> ChangePreparation:
    """Resolve and preflight one allow-listed change without persisting it."""

    resolution = resolve_setting_alias(request.utterance)
    if resolution == InspectionStatus.UNSUPPORTED:
        return ChangePreparation(ChangePreparationStatus.UNSUPPORTED)
    if resolution == InspectionStatus.AMBIGUOUS:
        return ChangePreparation(ChangePreparationStatus.AMBIGUOUS)
    assert isinstance(resolution, SettingKey)
    if (
        request.raw_value is None
        or not request.raw_value.strip()
        or len(request.raw_value) > 100
        or _SENSITIVE_OR_EXECUTABLE.search(request.utterance)
    ):
        return ChangePreparation(ChangePreparationStatus.MALFORMED)
    raw_value = _normalize_raw_value(resolution, request.raw_value)
    if resolution == SettingKey.VOICE and not _SAFE_VOICE_VALUE.fullmatch(raw_value):
        return ChangePreparation(ChangePreparationStatus.INVALID)
    resolved_paths = paths or config_paths(environment)
    try:
        stored = load_managed_config(resolved_paths, environment)
        updated = apply_config_values(stored, {resolution.value: raw_value})
    except (OSError, UserConfigurationError):
        return ChangePreparation(ChangePreparationStatus.INVALID)
    active_value = read_setting(active_config, resolution)
    stored_value = read_setting(stored, resolution)
    new_value = read_setting(updated, resolution)
    if active_value != stored_value:
        return ChangePreparation(ChangePreparationStatus.CONFLICT)
    if new_value == stored_value:
        return ChangePreparation(ChangePreparationStatus.NO_CHANGE)
    fenced_value = describe_config_value(stored, resolution.value)
    if not isinstance(fenced_value, (str, bool, float)):
        return ChangePreparation(ChangePreparationStatus.INVALID)
    return ChangePreparation(
        ChangePreparationStatus.READY,
        PendingConfigChange(
            trusted_utterance=request.utterance,
            setting=resolution,
            raw_value=raw_value,
            old_value=active_value,
            new_value=new_value,
            stored_value=fenced_value,
            affected_services=restart_services_for_keys(
                updated,
                (resolution.value,),
            ),
        ),
    )


_CONFIRMATIONS = frozenset(
    {
        "yes",
        "yes confirm",
        "confirm",
        "apply it",
        "save it",
        "go ahead",
        "do it",
    }
)
_CANCELLATIONS = frozenset(
    {
        "no",
        "no cancel",
        "cancel",
        "cancel it",
        "never mind",
        "nevermind",
        "do not",
        "don't",
    }
)


def resolve_confirmation(utterance: str) -> ConfirmationDecision:
    """Accept only an unambiguous whole-utterance confirmation or cancellation."""

    normalized = _normalize(utterance)
    if normalized in _CONFIRMATIONS:
        return ConfirmationDecision.CONFIRM
    if normalized in _CANCELLATIONS:
        return ConfirmationDecision.CANCEL
    return ConfirmationDecision.AMBIGUOUS


def _display_value(setting: SettingKey, value: str | bool) -> str:
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    if setting == SettingKey.VOICE and not value:
        return "the configured default"
    return _bounded_spoken_value(value)


def render_change_preparation(preparation: ChangePreparation) -> AssistantResponse:
    """Render exact old/new readback or a bounded fail-closed result."""

    if preparation.status == ChangePreparationStatus.AMBIGUOUS:
        return AssistantResponse.from_text(AMBIGUOUS_INSPECTION_RESPONSE)
    if preparation.status == ChangePreparationStatus.NO_CHANGE:
        return AssistantResponse.from_text(
            "That setting already has the requested value, so I didn't write anything."
        )
    if preparation.status == ChangePreparationStatus.CONFLICT:
        return AssistantResponse.from_text(
            "The active and stored values differ, so I won't change that setting "
            "conversationally. Resolve the configuration override first."
        )
    if preparation.status != ChangePreparationStatus.READY:
        return AssistantResponse.from_text(
            "I couldn't validate that allow-listed setting change, so I didn't write "
            "anything."
        )
    pending = preparation.pending
    assert pending is not None
    old_value = _display_value(pending.setting, pending.old_value)
    new_value = _display_value(pending.setting, pending.new_value)
    spoken = (
        f"Change {pending.setting.value} from {old_value} to {new_value}? "
        "Say yes to confirm or no to cancel."
    )
    return AssistantResponse(
        spoken_text=spoken,
        display_text=spoken,
    )


def commit_pending_change(
    pending: PendingConfigChange,
    *,
    paths: ConfigPaths | None = None,
    environment: Mapping[str, str] | None = None,
) -> ConfigChangeResult:
    """Persist one previously validated change with stale-value fencing."""

    return commit_config_change(
        {pending.setting.value: pending.raw_value},
        paths=paths,
        environment=environment,
        expected_values={pending.setting.value: pending.stored_value},
    )


def render_change_committed(
    pending: PendingConfigChange,
    result: ConfigChangeResult,
) -> AssistantResponse:
    """Report persistence and manual activation without mutating runtime state."""

    value = _display_value(pending.setting, pending.new_value)
    affected = ", ".join(pending.affected_services)
    active = ", ".join(result.restart_services)
    if active:
        active_notice = f"Active affected services: {active}."
    else:
        active_notice = "No affected managed service is currently active."
    activation = (
        f" Restart {affected} manually to apply it."
        if affected
        else " Start the next owning process to apply it."
    )
    spoken = (
        f"Saved {pending.setting.value} as {value}. "
        "The running configuration snapshot is unchanged."
        f"{activation} {active_notice}"
    )
    return AssistantResponse(spoken_text=spoken, display_text=spoken)


def _bounded_spoken_value(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip()
    if not normalized:
        return "the configured default"
    if len(normalized) <= _MAX_SPOKEN_VALUE_CHARS:
        return normalized
    return normalized[: _MAX_SPOKEN_VALUE_CHARS - 1].rstrip() + "…"


def render_inspection_result(result: InspectionResult) -> AssistantResponse:
    """Render a bounded answer without exposing any unlisted configuration."""

    if result.status == InspectionStatus.AMBIGUOUS:
        return AssistantResponse.from_text(AMBIGUOUS_INSPECTION_RESPONSE)
    if result.status != InspectionStatus.OK or result.setting is None:
        return AssistantResponse.from_text(UNSUPPORTED_INSPECTION_RESPONSE)
    if result.setting == SettingKey.VOICE:
        value = _bounded_spoken_value(str(result.value or ""))
        return AssistantResponse(
            spoken_text=f"My configured voice is {value}.",
            display_text=f"audio.voice: {value}",
        )
    if result.setting == SettingKey.TTS_SPEED:
        value = _bounded_spoken_value(str(result.value or ""))
        return AssistantResponse(
            spoken_text=f"Speaking speed is {value}.",
            display_text=f"providers.tts.speed: {value}",
        )
    if result.setting == SettingKey.BARGE_IN_MODE:
        value = _bounded_spoken_value(str(result.value or ""))
        return AssistantResponse(
            spoken_text=f"Barge-in mode is {value}.",
            display_text=f"audio.barge_in_mode: {value}",
        )
    if result.setting == SettingKey.ANNOUNCEMENT_MODE:
        value = _bounded_spoken_value(str(result.value or ""))
        return AssistantResponse(
            spoken_text=f"Background announcement policy is {value}.",
            display_text=f"announcements.mode: {value}",
        )
    enabled = result.value is True
    name = {
        SettingKey.GITHUB: "GitHub",
        SettingKey.LINEAR: "Linear",
        SettingKey.ZENDESK: "Zendesk",
    }[result.setting]
    state = "enabled" if enabled else "disabled"
    return AssistantResponse(
        spoken_text=f"{name} is {state}.",
        display_text=f"{result.setting.value}: {state}",
    )


def inspect_config_utterance(
    utterance: str,
    config: UserConfig,
) -> AssistantResponse:
    """Execute and render one trusted read-only inspection request."""

    request = SelfManagementRequest(
        SelfManagementIntent.INSPECT_CONFIG,
        utterance,
    )
    return render_inspection_result(inspect_config(request, config))


__all__ = [
    "AMBIGUOUS_INSPECTION_RESPONSE",
    "ChangePreparation",
    "ChangePreparationStatus",
    "ConfigChangeRequest",
    "ConfirmationDecision",
    "InspectionResult",
    "InspectionStatus",
    "PendingConfigChange",
    "SETTING_REGISTRY",
    "SelfManagementIntent",
    "SelfManagementRequest",
    "SettingKey",
    "UNSUPPORTED_INSPECTION_RESPONSE",
    "commit_pending_change",
    "inspect_config",
    "inspect_config_utterance",
    "prepare_config_change",
    "read_setting",
    "render_change_committed",
    "render_change_preparation",
    "render_inspection_result",
    "resolve_confirmation",
    "resolve_setting_alias",
]
