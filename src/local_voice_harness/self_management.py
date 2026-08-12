"""Typed, read-only conversational self-management operations."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from .responses import AssistantResponse
from .user_config import UserConfig

UNSUPPORTED_INSPECTION_RESPONSE = (
    "I can only inspect one supported harness setting at a time: voice, "
    "barge-in mode, or whether GitHub, Linear, or Zendesk is enabled."
)
AMBIGUOUS_INSPECTION_RESPONSE = (
    "I couldn't identify exactly one supported harness setting. Ask about voice, "
    "barge-in mode, or one supported integration."
)
_MAX_SPOKEN_VALUE_CHARS = 80


class SelfManagementIntent(StrEnum):
    """Provider-neutral first-party operations supported by the harness."""

    INSPECT_CONFIG = "inspect_config"


class InspectionStatus(StrEnum):
    """Outcome of resolving and executing a configuration inspection."""

    OK = "ok"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"


class SettingKey(StrEnum):
    """Configuration keys intentionally exposed to conversation."""

    VOICE = "audio.voice"
    BARGE_IN_MODE = "audio.barge_in_mode"
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
    if result.setting == SettingKey.BARGE_IN_MODE:
        value = _bounded_spoken_value(str(result.value or ""))
        return AssistantResponse(
            spoken_text=f"Barge-in mode is {value}.",
            display_text=f"audio.barge_in_mode: {value}",
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
    "InspectionResult",
    "InspectionStatus",
    "SETTING_REGISTRY",
    "SelfManagementIntent",
    "SelfManagementRequest",
    "SettingKey",
    "UNSUPPORTED_INSPECTION_RESPONSE",
    "inspect_config",
    "inspect_config_utterance",
    "render_inspection_result",
    "resolve_setting_alias",
]
