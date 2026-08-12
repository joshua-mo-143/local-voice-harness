from __future__ import annotations

from pathlib import Path
from typing import Any

from .repository import HerdrRepository
from .session import HerdrSession
from .transport import HerdrTransport
from .types import (
    AGENT_COMPLETION_QUIET_SECONDS,
    AgentSelection,
    BeforePaneSubmit,
    BeforePromptSubmit,
    Checkpoint,
    FailOperation,
    PaneAccepted,
    PlanParticipant,
    PromptAccepted,
    PromptBoundary,
    PromptOutcome,
    ReserveAgent,
    ReserveWorktree,
    SettleAgent,
    SettleWorktree,
)
from .workspace import HerdrWorkspace


class HerdrClient:
    """Facade composing Herdr transport, workspace, and repository components."""

    def __init__(
        self,
        executable: str = "herdr",
        *,
        repository_root: Path | None = None,
        worktree_root: Path | None = None,
        cursor_mcp_auth_source: Path | None = None,
        cursor_projects_root: Path | None = None,
        timeout: float = 30,
        agent_inactivity_timeout: float = 15 * 60,
        agent_max_runtime: float = 60 * 60,
    ) -> None:
        root = (repository_root or Path.home()).expanduser().resolve()
        worktrees = (
            (worktree_root or Path.home() / ".herdr" / "worktrees")
            .expanduser()
            .resolve()
        )
        self.transport = HerdrTransport(executable, timeout=timeout)
        self.executable = executable
        self.timeout = timeout
        self.agent_inactivity_timeout = agent_inactivity_timeout
        self.agent_max_runtime = agent_max_runtime
        self.session = HerdrSession(self)
        self.repository = HerdrRepository(self, root)
        self.workspace = HerdrWorkspace(
            self,
            repository_root=root,
            worktree_root=worktrees,
            cursor_mcp_auth_source=cursor_mcp_auth_source,
            cursor_projects_root=cursor_projects_root,
        )

    def command(self, *args: str) -> list[str]:
        return self.transport.command(*args)

    @staticmethod
    def decode(text: str) -> dict[str, Any]:
        return HerdrTransport.decode(text)

    def run(self, *args: str, timeout: float | None = None, check: bool = True):
        return self.transport.run(*args, timeout=timeout, check=check)

    def run_json(self, *args: str, timeout: float | None = None) -> dict[str, Any]:
        return self.transport.run_json(*args, timeout=timeout)

    def run_text(self, *args: str, timeout: float | None = None) -> str:
        return self.transport.run_text(*args, timeout=timeout)

    def is_running(self) -> bool:
        return self.transport.is_running()

    def ensure_server(self, timeout: float | None = None) -> None:
        self.transport.ensure_server(timeout=timeout)

    def list_agents(self) -> list[dict[str, Any]]:
        return list(self.run_json("agent", "list").get("agents") or [])

    def list_workspaces(self) -> list[dict[str, Any]]:
        return list(
            self.transport.run_json("workspace", "list").get("workspaces") or []
        )

    def focused_checkout(self) -> Path | None:
        return self.workspace.focused_checkout()

    def get_agent(self, target: str) -> dict[str, Any]:
        return dict(self.run_json("agent", "get", target).get("agent") or {})

    def allowed_repository(self, path: Path) -> bool:
        return self.repository.allowed_repository(path)

    def repository_roots(self) -> list[Path]:
        return self.repository.repository_roots()

    def resolve_repository(
        self, hint: str | None, task: str, repositories: list[Path] | None = None
    ) -> tuple[Path | None, list[Path]]:
        return self.repository.resolve_repository(hint, task, repositories)

    def clone_repository(
        self, url: str, *, checkpoint: Checkpoint | None = None
    ) -> Path:
        return self.repository.clone_repository(url, checkpoint=checkpoint)

    def choose_or_clone_repository(
        self,
        repositories: list[Path],
        *,
        checkpoint: Checkpoint | None = None,
    ) -> tuple[Path | None, str]:
        return self.repository.choose_or_clone_repository(
            repositories, checkpoint=checkpoint
        )

    @staticmethod
    def target(agent: dict[str, Any]) -> str:
        return HerdrWorkspace.target(agent)

    def live_agents(self) -> list[dict[str, Any]]:
        return HerdrWorkspace.live_agents(self.workspace)

    def selection(
        self, agent: dict[str, Any], worktree: str | None = None
    ) -> AgentSelection:
        return self.workspace.selection(agent, worktree)

    def find_agent(
        self,
        *,
        repository: Path | None = None,
        checkout: Path | None = None,
        agent_hint: str | None = None,
        reserved: set[str] | None = None,
    ) -> AgentSelection | None:
        return HerdrWorkspace.find_agent(
            self.workspace,
            repository=repository,
            checkout=checkout,
            agent_hint=agent_hint,
            reserved=reserved,
        )

    def workspace_for(self, checkout: Path) -> dict[str, Any] | None:
        return HerdrWorkspace.workspace_for(self.workspace, checkout)

    def planned_worktree_path(self, repository: Path, branch: str) -> Path:
        return self.workspace.planned_worktree_path(repository, branch)

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
        return HerdrWorkspace.new_pane(
            self.workspace,
            checkout,
            label,
            workspace_id,
            checkpoint=checkpoint,
            before_submit=before_submit,
            accepted=accepted,
        )

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
        return HerdrWorkspace.start_agent(
            self.workspace,
            checkout,
            label,
            pane,
            workspace,
            name=name,
            mode=mode,
            checkpoint=checkpoint,
        )

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
        return self.workspace.ensure_agent(
            repository,
            issue_key=issue_key,
            agent_hint=agent_hint,
            reserved=reserved,
            worktree_branch=worktree_branch,
            worktree_label=worktree_label,
            mode=mode,
            checkpoint=checkpoint,
            reserve=reserve,
            settle=settle,
            fail_agent=fail_agent,
            reserve_worktree=reserve_worktree,
            settle_worktree=settle_worktree,
            fail_worktree=fail_worktree,
            plan_participant=plan_participant,
            before_pane_submit=before_pane_submit,
            pane_accepted=pane_accepted,
            participant_name=participant_name,
        )

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
        return self.workspace.start_fresh_agent(
            checkout,
            label,
            workspace_id,
            role=role,
            mode=mode,
            name=name,
            checkpoint=checkpoint,
            reserve=reserve,
            settle=settle,
            fail_agent=fail_agent,
            before_pane_submit=before_pane_submit,
            pane_accepted=pane_accepted,
        )

    def ensure_router(
        self, reserved: set[str], *, checkpoint: Checkpoint | None = None
    ) -> AgentSelection:
        return self.workspace.ensure_router(reserved, checkpoint=checkpoint)

    def prompt_and_wait(
        self,
        target: str,
        text: str,
        *,
        token: str,
        timeout: float = -1,
        max_runtime: float = -1,
        checkpoint: Checkpoint | None = None,
        baseline_sequence: int | None = None,
        expected_agent_session: str | None = None,
        before_submit: BeforePromptSubmit | None = None,
        accepted: PromptAccepted | None = None,
        before_agent: PromptBoundary | None = None,
        after_submit: PromptBoundary | None = None,
        active_marker: str | None = None,
        allow_interactive_plan_boundary: bool = False,
        allow_enter_fallback: bool = True,
    ) -> PromptOutcome:
        return self.session.prompt_and_wait(
            target,
            text,
            token=token,
            timeout=self.agent_inactivity_timeout if timeout < 0 else timeout,
            max_runtime=self.agent_max_runtime if max_runtime < 0 else max_runtime,
            checkpoint=checkpoint,
            baseline_sequence=baseline_sequence,
            expected_agent_session=expected_agent_session,
            before_submit=before_submit,
            accepted=accepted,
            before_agent=before_agent,
            after_submit=after_submit,
            active_marker=active_marker,
            allow_interactive_plan_boundary=allow_interactive_plan_boundary,
            allow_enter_fallback=allow_enter_fallback,
        )

    def wait_for_stable_completion(
        self,
        target: str,
        *,
        token: str,
        inactivity_timeout: float = -1,
        max_runtime: float = -1,
        quiet_period: float = AGENT_COMPLETION_QUIET_SECONDS,
        started_at: float | None = None,
        checkpoint: Checkpoint | None = None,
        expected_agent_session: str | None = None,
        active_marker: str | None = None,
        allow_interactive_plan_boundary: bool = False,
    ) -> PromptOutcome:
        return self.session.wait_for_stable_completion(
            target,
            token=token,
            inactivity_timeout=(
                self.agent_inactivity_timeout
                if inactivity_timeout < 0
                else inactivity_timeout
            ),
            max_runtime=self.agent_max_runtime if max_runtime < 0 else max_runtime,
            quiet_period=quiet_period,
            started_at=started_at,
            checkpoint=checkpoint,
            expected_agent_session=expected_agent_session,
            active_marker=active_marker,
            allow_interactive_plan_boundary=allow_interactive_plan_boundary,
        )

    def cancel_agent(self, target: str) -> None:
        self.session.cancel_agent(target)

    def close_owned_pane(
        self,
        target: str,
        pane_id: str,
        workspace_id: str,
    ) -> None:
        self.session.close_owned_pane(target, pane_id, workspace_id)
