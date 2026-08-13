from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from ...agents.harness import Checkpoint as HarnessCheckpoint
from ...agents.harness import HarnessSession, SessionRequest

HERDR_UNIT = "voice-harness-herdr.service"
SETTLED = {"idle", "done"}
OBSERVABLE_AGENT_STATES = SETTLED | {"blocked", "unknown"}
AGENT_COMPLETION_POLL_SECONDS = 1.0
AGENT_COMPLETION_QUIET_SECONDS = 5.0
AGENT_PROMPT_WAIT_SECONDS = 5.0
AGENT_START_READY_POLL_SECONDS = 0.2
AGENT_START_READY_TIMEOUT_SECONDS = 15.0
MAX_MARKER_BYTES = 64 * 1024
SCP_GIT_URL = re.compile(
    r"^(?P<user>[A-Za-z0-9._-]+)@(?P<host>[A-Za-z0-9.-]+):(?P<path>[^:\s]+)$"
)

Checkpoint = Callable[[], None]
PromptBoundary = Callable[[dict[str, Any]], None]
ReserveAgent = Callable[["AgentSelection", bool], None]
SettleAgent = Callable[["AgentSelection"], None]
ReserveWorktree = Callable[[Path, str, Path, str], None]
SettleWorktree = Callable[[Path, str | None, str | None], None]
FailOperation = Callable[["HerdrError"], None]
BeforePromptSubmit = Callable[[int], None]
PromptAccepted = Callable[[], None]
BeforePaneSubmit = Callable[[], None]
PaneAccepted = Callable[[str, str], None]
PlanParticipant = Callable[[str, str, str | None, Path], None]


class HerdrOperations(Protocol):
    def run_json(self, *args: str, timeout: float | None = None) -> dict[str, Any]: ...

    def get_agent(self, target: str) -> dict[str, Any]: ...

    def list_agents(self) -> list[dict[str, Any]]: ...

    def list_workspaces(self) -> list[dict[str, Any]]: ...

    def live_agents(self) -> list[dict[str, Any]]: ...

    def find_agent(
        self,
        *,
        repository: Path | None = None,
        checkout: Path | None = None,
        agent_hint: str | None = None,
        reserved: set[str] | None = None,
    ) -> AgentSelection | None: ...

    def workspace_for(self, checkout: Path) -> dict[str, Any] | None: ...

    def planned_worktree_path(self, repository: Path, branch: str) -> Path: ...

    def new_pane(
        self,
        checkout: Path,
        label: str,
        workspace_id: str | None = None,
        *,
        checkpoint: Checkpoint | None = None,
        before_submit: BeforePaneSubmit | None = None,
        accepted: PaneAccepted | None = None,
    ) -> tuple[str, str]: ...

    def start_agent(
        self,
        checkout: Path,
        label: str,
        pane: str,
        workspace: str,
        *,
        name: str | None = None,
        mode: str | None = None,
        checkpoint: Checkpoint | None = None,
    ) -> AgentSelection: ...

    def create_session(
        self,
        request: SessionRequest,
        *,
        checkpoint: HarnessCheckpoint | None = None,
    ) -> HarnessSession: ...


class HerdrError(RuntimeError):
    def __init__(self, message: str, *, code: str = "") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AgentSelection:
    target: str
    pane_id: str
    workspace_id: str
    cwd: str
    name: str
    worktree_path: str | None = None
    provider: str | None = None
    provider_session_id: str | None = None
    state_sequence: int | None = None


@dataclass(frozen=True)
class PromptOutcome:
    status: str
    summary: str | None
    question: str | None
    output: str
    boundary_marker: str | None = None
    agent_session: str | None = None
    state_change_sequence: int | None = None
    revision: int | None = None


def agent_session_identity(value: object) -> str | None:
    """Return one canonical durable identity for a Herdr agent session."""

    if value is None or value == "":
        return None
    if isinstance(value, dict):
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            return None
    return str(value)


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def repository_name_from_url(value: str) -> str | None:
    value = value.strip()
    match = SCP_GIT_URL.fullmatch(value)
    if match:
        path = match.group("path")
    else:
        try:
            parsed = urlsplit(value)
        except ValueError:
            return None
        if (
            parsed.scheme not in {"https", "ssh"}
            or not parsed.hostname
            or parsed.password is not None
        ):
            return None
        path = parsed.path
    name = path.rstrip("/").rsplit("/", 1)[-1]
    if name.casefold().endswith(".git"):
        name = name[:-4]
    if (
        not name
        or name.startswith(".")
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name)
    ):
        return None
    return name


def extract_marker(output: str, marker: str, token: str) -> str | None:
    prefix = re.compile(rf"^\s*{re.escape(marker)}\[{re.escape(token)}\]:\s*(.*)$")
    matches: list[str] = []
    lines = output.splitlines()
    for index, line in enumerate(lines):
        match = prefix.match(line)
        if match is None:
            continue
        parts = [match.group(1).strip()]
        for continuation in lines[index + 1 :]:
            stripped = continuation.strip()
            if not stripped or re.match(
                r"^(?:VOICE_|ROUTE_|WORKFLOW_)[A-Z_]+\[", stripped
            ):
                break
            parts.append(stripped)
        value = " ".join(filter(None, parts)).strip()
        if (
            value
            and not value.startswith("<")
            and len(value.encode()) <= MAX_MARKER_BYTES
        ):
            matches.append(value)
    return matches[-1] if matches else None
