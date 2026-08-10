from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..config import (
    CURSOR_AGENT_INACTIVITY_SECONDS,
    CURSOR_AGENT_MAX_RUNTIME_SECONDS,
    REPOSITORY_ROOT,
)
from .rofi import choose_repository, confirm_clone

HERDR_BIN = os.environ.get(
    "VOICE_HARNESS_HERDR_BIN", str(Path.home() / ".local/bin/herdr")
)
HERDR_UNIT = "voice-harness-herdr.service"
HERDR_WORKTREE_ROOT = Path(
    os.environ.get(
        "VOICE_HARNESS_HERDR_WORKTREE_ROOT",
        str(Path.home() / ".herdr/worktrees"),
    )
).expanduser()
HOME_ROOT = REPOSITORY_ROOT
SETTLED = {"idle", "done"}
OBSERVABLE_AGENT_STATES = SETTLED | {"blocked", "unknown"}
AGENT_COMPLETION_POLL_SECONDS = 1.0
AGENT_COMPLETION_QUIET_SECONDS = 5.0
AGENT_PROMPT_WAIT_SECONDS = 5.0
MAX_MARKER_BYTES = 64 * 1024
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
PlanParticipant = Callable[[str, str, str | None], None]
LINEAR_ISSUE = re.compile(r"\b([A-Z][A-Z0-9]+)(?:\s*-\s*|\s+)(\d+)\b", re.IGNORECASE)
SCP_GIT_URL = re.compile(
    r"^(?P<user>[A-Za-z0-9._-]+)@(?P<host>[A-Za-z0-9.-]+):(?P<path>[^:\s]+)$"
)


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
        # Terminal history includes the prompt itself. Placeholder examples must
        # never become accepted output when the agent omits its required marker.
        if (
            value
            and not value.startswith("<")
            and len(value.encode()) <= MAX_MARKER_BYTES
        ):
            matches.append(value)
    return matches[-1] if matches else None


class HerdrClient:
    def __init__(self, executable: str = HERDR_BIN) -> None:
        self.executable = executable

    def command(self, *args: str) -> list[str]:
        return [self.executable, *args]

    @staticmethod
    def decode(text: str) -> dict[str, Any]:
        try:
            envelope = json.loads(text.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            raise HerdrError("Herdr returned malformed JSON") from exc
        if "error" in envelope:
            error = envelope["error"]
            raise HerdrError(
                str(error.get("message") or "Herdr command failed"),
                code=str(error.get("code") or ""),
            )
        result = envelope.get("result")
        if not isinstance(result, dict):
            raise HerdrError("Herdr response did not include a result")
        return result

    def run(
        self, *args: str, timeout: float = 30, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        try:
            process = subprocess.run(
                self.command(*args),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except OSError as exc:
            raise HerdrError(
                f"Herdr command failed: {exc}", code="operation_spawn_failed"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise HerdrError(
                f"Herdr command failed: {exc}", code="operation_timeout"
            ) from exc
        if check and process.returncode:
            text = process.stdout.strip() or process.stderr.strip()
            try:
                self.decode(text)
            except HerdrError as exc:
                raise exc
            raise HerdrError(text or f"Herdr exited with status {process.returncode}")
        return process

    def run_json(self, *args: str, timeout: float = 30) -> dict[str, Any]:
        return self.decode(self.run(*args, timeout=timeout).stdout)

    def run_text(self, *args: str, timeout: float = 30) -> str:
        return self.run(*args, timeout=timeout).stdout

    def is_running(self) -> bool:
        process = self.run("status", "server", check=False)
        return process.returncode == 0 and "status: running" in process.stdout

    def ensure_server(self, timeout: float = 15) -> None:
        if self.is_running():
            return
        subprocess.run(
            ["systemctl", "--user", "start", HERDR_UNIT],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if not self.is_running():
            process = subprocess.run(
                [
                    "systemd-run",
                    "--user",
                    "--unit=voice-harness-herdr",
                    "--collect",
                    self.executable,
                    "server",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if (
                process.returncode
                and "already exists" not in (process.stdout + process.stderr).casefold()
            ):
                raise HerdrError(process.stderr.strip() or "Could not start Herdr")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_running():
                return
            time.sleep(0.2)
        raise HerdrError("Herdr did not become ready")

    def list_agents(self) -> list[dict[str, Any]]:
        return list(self.run_json("agent", "list").get("agents") or [])

    def list_workspaces(self) -> list[dict[str, Any]]:
        return list(self.run_json("workspace", "list").get("workspaces") or [])

    def get_agent(self, target: str) -> dict[str, Any]:
        return dict(self.run_json("agent", "get", target).get("agent") or {})

    @staticmethod
    def allowed_repository(path: Path) -> bool:
        try:
            path.relative_to(HOME_ROOT)
        except ValueError:
            return False
        return path != HOME_ROOT and (path / ".git").exists()

    def repository_roots(self) -> list[Path]:
        roots: dict[str, Path] = {}
        for workspace in self.list_workspaces():
            root = (workspace.get("worktree") or {}).get("repo_root")
            if root:
                path = Path(str(root)).resolve()
                if self.allowed_repository(path):
                    roots[str(path)] = path
        try:
            children = HOME_ROOT.iterdir()
        except OSError:
            children = []
        for child in children:
            try:
                if (
                    not child.name.startswith(".")
                    and child.is_dir()
                    and (child / ".git").exists()
                ):
                    path = child.resolve()
                    if self.allowed_repository(path):
                        roots[str(path)] = path
            except OSError:
                continue
        return sorted(roots.values(), key=lambda path: path.name.casefold())

    def resolve_repository(
        self, hint: str | None, task: str, repositories: list[Path] | None = None
    ) -> tuple[Path | None, list[Path]]:
        repositories = self.repository_roots() if repositories is None else repositories
        if hint:
            candidate = Path(hint).expanduser()
            if candidate.is_absolute():
                resolved = candidate.resolve()
                exact = [path for path in repositories if path == resolved]
                return (exact[0], exact) if exact else (None, [])
            normalized = normalize_name(hint)
            exact = [p for p in repositories if normalize_name(p.name) == normalized]
            if len(exact) == 1:
                return exact[0], exact
            partial = [
                p
                for p in repositories
                if normalized
                and (
                    normalized in normalize_name(p.name)
                    or normalize_name(p.name) in normalized
                )
            ]
            return (partial[0], partial) if len(partial) == 1 else (None, partial)
        normalized_task = normalize_name(task)
        matches = [
            path
            for path in repositories
            if normalize_name(path.name) in normalized_task
        ]
        return (matches[0], matches) if len(matches) == 1 else (None, matches)

    def clone_repository(
        self, url: str, *, checkpoint: Checkpoint | None = None
    ) -> Path:
        name = repository_name_from_url(url)
        if name is None:
            raise HerdrError("Only Git HTTPS and SSH repository URLs are supported")
        destination = (HOME_ROOT / name).resolve()
        if destination.parent != HOME_ROOT:
            raise HerdrError(
                "Repository destination is outside the configured project root"
            )
        if destination.exists():
            if self.allowed_repository(destination):
                return destination
            raise HerdrError(
                f"{destination} already exists and is not a Git repository"
            )
        try:
            if checkpoint is not None:
                checkpoint()
            with tempfile.TemporaryDirectory(
                dir=HOME_ROOT, prefix=f".{name}-clone-"
            ) as temporary:
                if checkpoint is not None:
                    checkpoint()
                staging = Path(temporary) / name
                if checkpoint is not None:
                    checkpoint()
                process = subprocess.run(
                    ["git", "clone", "--", url, str(staging)],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )
                if checkpoint is not None:
                    checkpoint()
                if process.returncode:
                    message = process.stderr.strip() or process.stdout.strip()
                    raise HerdrError(message or "Could not clone repository")
                if not self.allowed_repository(staging):
                    raise HerdrError(
                        "Cloned destination is not an allowed Git repository"
                    )
                if checkpoint is not None:
                    checkpoint()
                staging.rename(destination)
                if checkpoint is not None:
                    checkpoint()
        except HerdrError:
            raise
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise HerdrError(f"Could not clone repository: {exc}") from exc
        return destination

    def choose_or_clone_repository(
        self,
        repositories: list[Path],
        *,
        checkpoint: Checkpoint | None = None,
    ) -> tuple[Path | None, str]:
        if checkpoint is not None:
            checkpoint()
        selected = choose_repository([path.name for path in repositories])
        if checkpoint is not None:
            checkpoint()
        if selected is None:
            return None, ""
        repository, _matches = self.resolve_repository(selected, "", repositories)
        if repository is not None:
            return repository, ""
        if repository_name_from_url(selected) is None:
            return None, "The Rofi selection was not a local repository or Git URL."
        if checkpoint is not None:
            checkpoint()
        confirmed = confirm_clone(selected)
        if checkpoint is not None:
            checkpoint()
        if not confirmed:
            return None, "Repository cloning was cancelled."
        try:
            return self.clone_repository(selected, checkpoint=checkpoint), ""
        except HerdrError as exc:
            return None, f"Repository cloning failed: {exc}."

    @staticmethod
    def target(agent: dict[str, Any]) -> str:
        return str(agent.get("name") or agent.get("pane_id") or "")

    def live_agents(self) -> list[dict[str, Any]]:
        return [
            agent
            for agent in self.list_agents()
            if agent.get("agent") == "cursor"
            and agent.get("interactive_ready") is not False
            and agent.get("agent_status") in SETTLED
        ]

    def selection(
        self, agent: dict[str, Any], worktree: str | None = None
    ) -> AgentSelection:
        target = self.target(agent)
        if not target:
            raise HerdrError("Herdr agent has no usable target")
        return AgentSelection(
            target=target,
            pane_id=str(agent.get("pane_id") or ""),
            workspace_id=str(agent.get("workspace_id") or ""),
            cwd=str(agent.get("cwd") or ""),
            name=str(agent.get("name") or target),
            worktree_path=worktree,
        )

    def find_agent(
        self,
        *,
        repository: Path | None = None,
        checkout: Path | None = None,
        agent_hint: str | None = None,
        reserved: set[str] | None = None,
    ) -> AgentSelection | None:
        reserved = reserved or set()
        expected = checkout or repository
        matches = []
        for agent in self.live_agents():
            if self.target(agent) in reserved:
                continue
            if agent_hint and normalize_name(
                str(agent.get("name") or "")
            ) != normalize_name(agent_hint):
                continue
            if (
                expected is not None
                and Path(str(agent.get("cwd") or "")).resolve() != expected.resolve()
            ):
                continue
            matches.append(agent)
        if len(matches) > 1:
            targets = ", ".join(sorted(self.target(agent) for agent in matches))
            raise HerdrError(
                f"multiple settled Cursor agents match the requested checkout: {targets}",
                code="agent_ambiguous",
            )
        if not matches:
            return None
        return self.selection(matches[0], str(expected) if expected else None)

    def workspace_for(self, checkout: Path) -> dict[str, Any] | None:
        for workspace in self.list_workspaces():
            path = (workspace.get("worktree") or {}).get("checkout_path")
            if path and Path(str(path)).resolve() == checkout.resolve():
                return workspace
        return None

    @staticmethod
    def planned_worktree_path(repository: Path, branch: str) -> Path:
        repository = repository.resolve()
        repository_key = hashlib.sha256(str(repository).encode()).hexdigest()[:8]
        repository_directory = (
            f"{normalize_name(repository.name) or 'repository'}-{repository_key}"
        )
        branch_directory = normalize_name(branch) or "worktree"
        return (HERDR_WORKTREE_ROOT / repository_directory / branch_directory).resolve()

    def new_pane(
        self,
        checkout: Path,
        label: str,
        workspace_id: str | None = None,
        *,
        checkpoint: Checkpoint | None = None,
        before_submit: BeforePaneSubmit | None = None,
        accepted: PaneAccepted | None = None,
    ) -> tuple[str, str]:
        if checkpoint is not None:
            checkpoint()
        if before_submit is not None:
            before_submit()
        if workspace_id:
            result = self.run_json(
                "tab",
                "create",
                "--workspace",
                workspace_id,
                "--cwd",
                str(checkout),
                "--label",
                label,
                "--no-focus",
            )
        else:
            result = self.run_json(
                "workspace",
                "create",
                "--cwd",
                str(checkout),
                "--label",
                label,
                "--no-focus",
            )
        if checkpoint is not None:
            checkpoint()
        pane = str((result.get("root_pane") or {}).get("pane_id") or "")
        workspace = str(
            (result.get("workspace") or {}).get("workspace_id") or workspace_id or ""
        )
        if not pane or not workspace:
            raise HerdrError("Herdr did not return a root pane")
        if accepted is not None:
            accepted(pane, workspace)
        return pane, workspace

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
    ) -> AgentSelection:
        if mode not in {None, "plan", "ask"}:
            raise HerdrError("invalid Cursor mode")
        if name is None:
            suffix = uuid.uuid4().hex[:10]
            name = f"voice-{normalize_name(label)[:15] or 'task'}-{suffix}"
        deadline = time.monotonic() + 15
        while True:
            try:
                if checkpoint is not None:
                    checkpoint()
                agent_args = ["--trust"]
                if mode is not None:
                    agent_args.extend(["--mode", mode])
                result = self.run_json(
                    "agent",
                    "start",
                    name,
                    "--kind",
                    "cursor",
                    "--pane",
                    pane,
                    "--timeout",
                    "60000",
                    "--",
                    *agent_args,
                    timeout=70,
                )
                selection = self.selection(
                    dict(result.get("agent") or {}), str(checkout)
                )
                return selection
            except HerdrError as exc:
                if exc.code != "agent_pane_busy" or time.monotonic() >= deadline:
                    raise
                time.sleep(0.4)

    def ensure_agent(
        self,
        repository: Path,
        *,
        issue_key: str | None,
        agent_hint: str | None,
        reserved: set[str],
        worktree_branch: str | None = None,
        worktree_label: str | None = None,
        mode: str | None = None,
        checkpoint: Checkpoint | None = None,
        reserve: ReserveAgent | None = None,
        settle: SettleAgent | None = None,
        fail_agent: FailOperation | None = None,
        reserve_worktree: ReserveWorktree | None = None,
        settle_worktree: SettleWorktree | None = None,
        fail_worktree: FailOperation | None = None,
        plan_participant: PlanParticipant | None = None,
        before_pane_submit: BeforePaneSubmit | None = None,
        pane_accepted: PaneAccepted | None = None,
        participant_name: str | None = None,
    ) -> AgentSelection:
        repository = repository.resolve()
        checkout = repository
        workspace_id = None
        root_pane = None
        label = worktree_label or repository.name
        branch = worktree_branch
        if issue_key and branch is None:
            label = issue_key.casefold()
            branch = f"voice/{label}"
        if branch:
            if not re.fullmatch(r"voice/[a-z0-9][a-z0-9._/-]{0,100}", branch):
                raise HerdrError("invalid voice worktree branch")
            if checkpoint is not None:
                checkpoint()
            listing = self.run_json(
                "worktree", "list", "--cwd", str(repository), "--json"
            )
            if checkpoint is not None:
                checkpoint()
            existing = next(
                (
                    item
                    for item in listing.get("worktrees") or []
                    if item.get("branch") == branch
                ),
                None,
            )
            if existing:
                checkout = Path(str(existing["path"])).resolve()
                if reserve_worktree is not None:
                    reserve_worktree(repository, branch, checkout, "ready")
                workspace_id = str(existing.get("open_workspace_id") or "") or None
                if not workspace_id:
                    if checkpoint is not None:
                        checkpoint()
                    opened = self.run_json(
                        "worktree",
                        "open",
                        "--cwd",
                        str(repository),
                        "--path",
                        str(checkout),
                        "--label",
                        label,
                        "--no-focus",
                        "--json",
                    )
                    if checkpoint is not None:
                        checkpoint()
                    workspace_id = str(
                        (opened.get("workspace") or {}).get("workspace_id") or ""
                    )
                    root_pane = str(
                        (opened.get("root_pane") or {}).get("pane_id") or ""
                    )
                if settle_worktree is not None:
                    settle_worktree(checkout, workspace_id, root_pane)
            else:
                checkout = self.planned_worktree_path(repository, branch)
                if reserve_worktree is not None:
                    reserve_worktree(repository, branch, checkout, "planned")
                if checkpoint is not None:
                    checkpoint()
                if reserve_worktree is not None:
                    reserve_worktree(repository, branch, checkout, "dispatching")
                try:
                    created = self.run_json(
                        "worktree",
                        "create",
                        "--cwd",
                        str(repository),
                        "--branch",
                        branch,
                        "--path",
                        str(checkout),
                        "--label",
                        label,
                        "--no-focus",
                        "--json",
                        timeout=120,
                    )
                    created_checkout = Path(
                        str((created.get("worktree") or {}).get("path") or "")
                    ).resolve()
                    if created_checkout != checkout:
                        raise HerdrError(
                            "Herdr created a worktree outside its reserved path",
                            code="operation_ambiguous",
                        )
                except HerdrError as exc:
                    if fail_worktree is not None:
                        fail_worktree(exc)
                    raise
                workspace_id = str(
                    (created.get("workspace") or {}).get("workspace_id") or ""
                )
                root_pane = str((created.get("root_pane") or {}).get("pane_id") or "")
                if settle_worktree is not None:
                    settle_worktree(checkout, workspace_id, root_pane)
                if checkpoint is not None:
                    checkpoint()
        existing_agent = (
            None
            if mode is not None
            else self.find_agent(
                repository=repository,
                checkout=checkout,
                agent_hint=agent_hint,
                reserved=reserved,
            )
        )
        if existing_agent:
            if reserve is not None:
                reserve(existing_agent, False)
            return existing_agent
        workspace = self.workspace_for(checkout)
        workspace_id = workspace_id or (
            str(workspace.get("workspace_id") or "") if workspace else None
        )
        suffix = uuid.uuid4().hex[:10]
        name = participant_name or (
            f"voice-{normalize_name(label)[:15] or 'task'}-{suffix}"
        )
        if not root_pane:
            if plan_participant is not None:
                plan_participant(name, label, workspace_id)
            root_pane, workspace_id = self.new_pane(
                checkout,
                label,
                workspace_id,
                checkpoint=checkpoint,
                before_submit=before_pane_submit,
                accepted=pane_accepted,
            )
        provisional = AgentSelection(
            target=name,
            pane_id=root_pane,
            workspace_id=str(workspace_id),
            cwd=str(checkout),
            name=name,
            worktree_path=str(checkout),
        )
        if reserve is not None:
            reserve(provisional, True)
        try:
            if mode is None:
                selection = self.start_agent(
                    checkout,
                    label,
                    root_pane,
                    str(workspace_id),
                    name=name,
                    checkpoint=checkpoint,
                )
            else:
                selection = self.start_agent(
                    checkout,
                    label,
                    root_pane,
                    str(workspace_id),
                    name=name,
                    mode=mode,
                    checkpoint=checkpoint,
                )
        except HerdrError as exc:
            if fail_agent is not None:
                fail_agent(exc)
            raise
        if settle is not None:
            settle(selection)
        if checkpoint is not None:
            checkpoint()
        return selection

    def start_fresh_agent(
        self,
        checkout: Path,
        label: str,
        workspace_id: str,
        *,
        role: str,
        mode: str | None,
        name: str | None = None,
        checkpoint: Checkpoint | None = None,
        reserve: ReserveAgent | None = None,
        settle: SettleAgent | None = None,
        fail_agent: FailOperation | None = None,
        before_pane_submit: BeforePaneSubmit | None = None,
        pane_accepted: PaneAccepted | None = None,
    ) -> AgentSelection:
        """Start a distinct participant in a new pane with a durable start fence."""
        if name is None:
            suffix = uuid.uuid4().hex[:10]
            name = (
                f"voice-{normalize_name(role)[:8] or 'agent'}-"
                f"{normalize_name(label)[:7] or 'task'}-{suffix}"
            )[:32]
        pane, workspace = self.new_pane(
            checkout,
            f"{label}-{role}",
            workspace_id,
            checkpoint=checkpoint,
            before_submit=before_pane_submit,
            accepted=pane_accepted,
        )
        provisional = AgentSelection(
            name,
            pane,
            workspace,
            str(checkout),
            name,
            str(checkout),
        )
        if reserve is not None:
            reserve(provisional, True)
        try:
            selection = self.start_agent(
                checkout,
                f"{label}-{role}",
                pane,
                workspace,
                name=name,
                mode=mode,
                checkpoint=checkpoint,
            )
        except HerdrError as exc:
            if fail_agent is not None:
                fail_agent(exc)
            raise
        if settle is not None:
            settle(selection)
        if checkpoint is not None:
            checkpoint()
        return selection

    def ensure_router(
        self, reserved: set[str], *, checkpoint: Checkpoint | None = None
    ) -> AgentSelection:
        existing = self.find_agent(agent_hint="voice-router", reserved=reserved)
        if existing:
            return existing
        workspace = next(
            (
                item
                for item in self.list_workspaces()
                if normalize_name(str(item.get("label") or "")) == "voice-router"
            ),
            None,
        )
        workspace_id = str(workspace.get("workspace_id") or "") if workspace else None
        pane, workspace_id = self.new_pane(
            HOME_ROOT,
            "voice-router",
            workspace_id or None,
            checkpoint=checkpoint,
        )
        return self.start_agent(
            HOME_ROOT,
            "router",
            pane,
            workspace_id,
            name="voice-router",
            checkpoint=checkpoint,
        )

    def prompt_and_wait(
        self,
        target: str,
        text: str,
        *,
        token: str,
        timeout: float = CURSOR_AGENT_INACTIVITY_SECONDS,
        max_runtime: float = CURSOR_AGENT_MAX_RUNTIME_SECONDS,
        checkpoint: Checkpoint | None = None,
        baseline_sequence: int | None = None,
        expected_agent_session: str | None = None,
        before_submit: BeforePromptSubmit | None = None,
        accepted: PromptAccepted | None = None,
        before_agent: PromptBoundary | None = None,
        after_submit: PromptBoundary | None = None,
        active_marker: str | None = None,
        allow_enter_fallback: bool = True,
    ) -> PromptOutcome:
        started_at = time.monotonic()
        if checkpoint is not None:
            checkpoint()
        before = self.get_agent(target)
        before_session = agent_session_identity(before.get("agent_session"))
        if (
            expected_agent_session is not None
            and before_session != expected_agent_session
        ):
            raise HerdrError(
                f"Herdr agent {target} no longer has the expected session",
                code="agent_session_changed",
            )
        if before.get("interactive_ready") is False:
            raise HerdrError(
                f"Herdr agent {target} is showing an interactive questionnaire",
                code="interactive_questionnaire",
            )
        observed_baseline = int(before.get("state_change_seq") or 0)
        if baseline_sequence is not None and observed_baseline != baseline_sequence:
            raise HerdrError(
                "Herdr agent changed before the planned prompt was submitted",
                code="operation_ambiguous",
            )
        if before_agent is not None:
            before_agent(before)
        if checkpoint is not None:
            checkpoint()
            checkpoint()
        if before_submit is not None:
            before_submit(observed_baseline)
        process = subprocess.Popen(
            self.command(
                "agent",
                "prompt",
                target,
                text,
                "--wait",
                "--timeout",
                str(int(AGENT_PROMPT_WAIT_SECONDS * 1000)),
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        acceptance_recorded = False
        observed_acceptance = False
        try:
            if after_submit is not None:
                after_submit(before)
            if checkpoint is not None:
                checkpoint()
            time.sleep(0.35)
            if checkpoint is not None:
                checkpoint()
            current = self.get_agent(target)
            if (
                expected_agent_session is not None
                and agent_session_identity(current.get("agent_session"))
                != expected_agent_session
            ):
                raise HerdrError(
                    f"Herdr agent {target} changed sessions during prompt submission",
                    code="agent_session_changed",
                )
            observed_acceptance = current.get("state_change_seq") != before.get(
                "state_change_seq"
            )
            if checkpoint is not None:
                checkpoint()
            if current.get("interactive_ready") is False:
                raise HerdrError(
                    f"Herdr agent {target} opened an interactive questionnaire",
                    code="interactive_questionnaire",
                )
            if (
                current.get("state_change_seq") == before.get("state_change_seq")
                and current.get("agent_status") in SETTLED
                and allow_enter_fallback
            ):
                if checkpoint is not None:
                    checkpoint()
                self.run_json("agent", "send-keys", target, "enter")
                if checkpoint is not None:
                    checkpoint()
            elif observed_acceptance and accepted is not None:
                accepted()
                acceptance_recorded = True
            deadline = time.monotonic() + AGENT_PROMPT_WAIT_SECONDS + 5
            while process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    process.wait(timeout=min(1.0, remaining))
                except subprocess.TimeoutExpired:
                    if checkpoint is not None:
                        checkpoint()
                    current = self.get_agent(target)
                    if (
                        expected_agent_session is not None
                        and agent_session_identity(current.get("agent_session"))
                        != expected_agent_session
                    ):
                        raise HerdrError(
                            f"Herdr agent {target} changed sessions during "
                            "prompt submission",
                            code="agent_session_changed",
                        ) from None
                    observed_acceptance = observed_acceptance or (
                        current.get("state_change_seq")
                        != before.get("state_change_seq")
                    )
                    if current.get("interactive_ready") is False:
                        raise HerdrError(
                            f"Herdr agent {target} opened an interactive questionnaire",
                            code="interactive_questionnaire",
                        ) from None
            stdout, stderr = process.communicate(timeout=AGENT_PROMPT_WAIT_SECONDS + 5)
            if checkpoint is not None:
                checkpoint()
        except Exception:
            process.kill()
            process.wait()
            raise
        if process.returncode:
            try:
                self.decode(stdout or stderr)
            except HerdrError as exc:
                if exc.code not in {"operation_timeout", "timeout"}:
                    raise
        else:
            self.decode(stdout)
        if expected_agent_session is not None or (
            accepted is not None and not acceptance_recorded
        ):
            if checkpoint is not None:
                checkpoint()
            current = self.get_agent(target)
            if (
                expected_agent_session is not None
                and agent_session_identity(current.get("agent_session"))
                != expected_agent_session
            ):
                raise HerdrError(
                    f"Herdr agent {target} changed sessions during prompt submission",
                    code="agent_session_changed",
                )
            if accepted is not None and not acceptance_recorded:
                observed_acceptance = observed_acceptance or (
                    current.get("state_change_seq") != before.get("state_change_seq")
                )
            if checkpoint is not None:
                checkpoint()
        if accepted is not None and not acceptance_recorded and observed_acceptance:
            accepted()
            acceptance_recorded = True
        if accepted is not None and not acceptance_recorded:
            raise HerdrError(
                f"Herdr did not accept the prompt for agent {target}",
                code="agent_prompt_stalled",
            )
        return self.wait_for_stable_completion(
            target,
            token=token,
            inactivity_timeout=timeout,
            max_runtime=max_runtime,
            started_at=started_at,
            checkpoint=checkpoint,
            expected_agent_session=expected_agent_session,
            active_marker=active_marker,
        )

    def wait_for_stable_completion(
        self,
        target: str,
        *,
        token: str,
        inactivity_timeout: float = CURSOR_AGENT_INACTIVITY_SECONDS,
        max_runtime: float = CURSOR_AGENT_MAX_RUNTIME_SECONDS,
        quiet_period: float = AGENT_COMPLETION_QUIET_SECONDS,
        started_at: float | None = None,
        checkpoint: Checkpoint | None = None,
        expected_agent_session: str | None = None,
        active_marker: str | None = None,
    ) -> PromptOutcome:
        """Wait for stable completion or an opt-in active marker boundary."""

        started = time.monotonic() if started_at is None else started_at
        last_activity = started
        last_signature: tuple[object, str, str] | None = None
        initial_session: str | None = None
        settled_since: float | None = None
        boundary_since: float | None = None
        marker_output: str | None = None

        while True:
            if checkpoint is not None:
                checkpoint()
            agent = self.get_agent(target)
            if checkpoint is not None:
                checkpoint()
            output = self.run_text(
                "agent",
                "read",
                target,
                "--source",
                "recent-unwrapped",
                "--lines",
                "160",
            )
            if checkpoint is not None:
                checkpoint()

            now = time.monotonic()
            status = str(agent.get("agent_status") or "unknown")
            session = agent_session_identity(agent.get("agent_session"))
            if expected_agent_session is not None and session != expected_agent_session:
                raise HerdrError(
                    f"Herdr agent {target} no longer has the expected session",
                    code="agent_session_changed",
                )
            if initial_session is None:
                initial_session = session
            elif session is not None and session != initial_session:
                raise HerdrError(
                    f"Herdr agent {target} changed sessions while work was active",
                    code="agent_session_changed",
                )

            signature = (
                agent.get("state_change_seq"),
                status,
                hashlib.sha256(output.encode()).hexdigest(),
            )
            if signature != last_signature:
                last_signature = signature
                last_activity = now
                settled_since = now if status in OBSERVABLE_AGENT_STATES else None
                boundary_since = None

            summary = extract_marker(output, "VOICE_SUMMARY", token)
            question = extract_marker(output, "VOICE_QUESTION", token)
            if summary or question:
                marker_output = output
            boundary = (
                extract_marker(output, active_marker, token) if active_marker else None
            )
            if agent.get("interactive_ready") is False:
                raise HerdrError(
                    f"Herdr agent {target} opened an interactive questionnaire",
                    code="interactive_questionnaire",
                )
            if boundary and not question and status in {"working", "blocked"}:
                marker_output = output
                if boundary_since is None:
                    boundary_since = now
                if now - boundary_since >= quiet_period:
                    sequence = agent.get("state_change_seq")
                    revision = agent.get("revision")
                    return PromptOutcome(
                        status=status,
                        summary=summary,
                        question=None,
                        output=output,
                        boundary_marker=active_marker,
                        agent_session=session,
                        state_change_sequence=(
                            sequence
                            if isinstance(sequence, int)
                            and not isinstance(sequence, bool)
                            else None
                        ),
                        revision=(
                            revision
                            if isinstance(revision, int)
                            and not isinstance(revision, bool)
                            else None
                        ),
                    )
            else:
                boundary_since = None

            if status in OBSERVABLE_AGENT_STATES:
                if settled_since is None:
                    settled_since = now
                if now - settled_since >= quiet_period:
                    selected_output = marker_output or output
                    return PromptOutcome(
                        status=status,
                        summary=extract_marker(selected_output, "VOICE_SUMMARY", token),
                        question=extract_marker(
                            selected_output, "VOICE_QUESTION", token
                        ),
                        output=selected_output,
                        boundary_marker=(
                            active_marker
                            if active_marker
                            and extract_marker(selected_output, active_marker, token)
                            and not extract_marker(
                                selected_output, "VOICE_QUESTION", token
                            )
                            else None
                        ),
                        agent_session=session,
                        state_change_sequence=(
                            agent.get("state_change_seq")
                            if isinstance(agent.get("state_change_seq"), int)
                            and not isinstance(agent.get("state_change_seq"), bool)
                            else None
                        ),
                        revision=(
                            agent.get("revision")
                            if isinstance(agent.get("revision"), int)
                            and not isinstance(agent.get("revision"), bool)
                            else None
                        ),
                    )
            else:
                settled_since = None

            inactivity_expired = now - last_activity >= inactivity_timeout
            runtime_expired = now - started >= max_runtime
            if inactivity_expired or runtime_expired:
                reason = (
                    "inactivity timeout" if inactivity_expired else "maximum runtime"
                )
                raise HerdrError(
                    f"Herdr agent {target} exceeded its {reason}",
                    code="agent_stalled",
                )

            remaining = min(
                inactivity_timeout - (now - last_activity),
                max_runtime - (now - started),
                AGENT_COMPLETION_POLL_SECONDS,
            )
            time.sleep(max(0.0, remaining))
            if checkpoint is not None:
                checkpoint()

    def cancel_agent(self, target: str) -> None:
        self.run_json("agent", "send-keys", target, "ctrl+c")
        result = self.run_json("agent", "wait", target, "--timeout", "5000", timeout=10)
        agent = dict(result.get("agent") or {})
        if agent.get("agent_status") == "working":
            raise HerdrError(f"Herdr agent {target} did not stop")
