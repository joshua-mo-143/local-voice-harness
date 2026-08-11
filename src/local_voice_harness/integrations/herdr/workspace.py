from __future__ import annotations

import hashlib
import re
import time
import uuid
from pathlib import Path
from typing import Any

from .types import (
    SETTLED,
    AgentSelection,
    BeforePaneSubmit,
    Checkpoint,
    FailOperation,
    HerdrError,
    HerdrOperations,
    PaneAccepted,
    PlanParticipant,
    ReserveAgent,
    ReserveWorktree,
    SettleAgent,
    SettleWorktree,
    normalize_name,
)


class HerdrWorkspace:
    """Workspace, pane, worktree, and target reservation management."""

    def __init__(
        self,
        operations: HerdrOperations,
        *,
        repository_root: Path | None = None,
        worktree_root: Path | None = None,
    ) -> None:
        self._operations = operations
        self.repository_root = (repository_root or Path.home()).expanduser().resolve()
        self.worktree_root = (
            (worktree_root or Path.home() / ".herdr" / "worktrees")
            .expanduser()
            .resolve()
        )

    def list_workspaces(self) -> list[dict[str, Any]]:
        return list(self._operations.list_workspaces())

    def focused_checkout(self) -> Path | None:
        """Return the checkout for exactly one focused Herdr workspace."""

        try:
            focused = [
                workspace
                for workspace in self.list_workspaces()
                if isinstance(workspace, dict) and workspace.get("focused") is True
            ]
        except (HerdrError, TypeError):
            return None
        if len(focused) != 1:
            return None
        worktree = focused[0].get("worktree")
        if not isinstance(worktree, dict):
            return None
        value = worktree.get("checkout_path")
        if not isinstance(value, str) or not value.strip():
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            return None
        try:
            return path.resolve()
        except OSError:
            return None

    @staticmethod
    def target(agent: dict[str, Any]) -> str:
        return str(agent.get("name") or agent.get("pane_id") or "")

    def live_agents(self) -> list[dict[str, Any]]:
        return [
            agent
            for agent in self._operations.list_agents()
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
        for agent in self._operations.live_agents():
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

    def planned_worktree_path(self, repository: Path, branch: str) -> Path:
        repository = repository.resolve()
        repository_key = hashlib.sha256(str(repository).encode()).hexdigest()[:8]
        repository_directory = (
            f"{normalize_name(repository.name) or 'repository'}-{repository_key}"
        )
        branch_directory = normalize_name(branch) or "worktree"
        return (self.worktree_root / repository_directory / branch_directory).resolve()

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
            result = self._operations.run_json(
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
            result = self._operations.run_json(
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
                result = self._operations.run_json(
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
            listing = self._operations.run_json(
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
                    opened = self._operations.run_json(
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
                checkout = self._operations.planned_worktree_path(repository, branch)
                if reserve_worktree is not None:
                    reserve_worktree(repository, branch, checkout, "planned")
                if checkpoint is not None:
                    checkpoint()
                if reserve_worktree is not None:
                    reserve_worktree(repository, branch, checkout, "dispatching")
                try:
                    created = self._operations.run_json(
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
            else self._operations.find_agent(
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
        workspace = self._operations.workspace_for(checkout)
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
            root_pane, workspace_id = self._operations.new_pane(
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
            selection = self._operations.start_agent(
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
        pane, workspace = self._operations.new_pane(
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
            selection = self._operations.start_agent(
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
        existing = self._operations.find_agent(
            agent_hint="voice-router", reserved=reserved
        )
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
        pane, workspace_id = self._operations.new_pane(
            self.repository_root,
            "voice-router",
            workspace_id or None,
            checkpoint=checkpoint,
        )
        return self._operations.start_agent(
            self.repository_root,
            "router",
            pane,
            workspace_id,
            name="voice-router",
            checkpoint=checkpoint,
        )
