"""Herdr integration split into transport, workspace, and repository components."""

from __future__ import annotations

from ..rofi import choose_repository, confirm_clone
from .client import HerdrClient
from .repository import HerdrRepository
from .session import HerdrSession
from .transport import HerdrTransport
from .types import (
    AGENT_COMPLETION_POLL_SECONDS,
    AGENT_COMPLETION_QUIET_SECONDS,
    AGENT_PROMPT_WAIT_SECONDS,
    HERDR_UNIT,
    OBSERVABLE_AGENT_STATES,
    SETTLED,
    AgentSelection,
    HerdrError,
    PromptOutcome,
    agent_session_identity,
    extract_marker,
    normalize_name,
    repository_name_from_url,
)
from .workspace import HerdrWorkspace

__all__ = [
    "AGENT_COMPLETION_POLL_SECONDS",
    "AGENT_COMPLETION_QUIET_SECONDS",
    "AGENT_PROMPT_WAIT_SECONDS",
    "HERDR_UNIT",
    "OBSERVABLE_AGENT_STATES",
    "SETTLED",
    "AgentSelection",
    "choose_repository",
    "confirm_clone",
    "HerdrClient",
    "HerdrError",
    "HerdrRepository",
    "HerdrSession",
    "HerdrTransport",
    "HerdrWorkspace",
    "PromptOutcome",
    "agent_session_identity",
    "extract_marker",
    "normalize_name",
    "repository_name_from_url",
]
