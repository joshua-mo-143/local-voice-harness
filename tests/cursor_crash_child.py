from __future__ import annotations

import json
import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest import mock

from local_voice_harness.cursor import provisioning, worker_lifecycle
from local_voice_harness.cursor.delivery import acknowledge_delivery, claim_delivery
from local_voice_harness.cursor.model import CursorJob, WorkflowParticipant
from local_voice_harness.cursor.recovery import recover_jobs
from local_voice_harness.cursor.store import JobStore, migrate_legacy_jobs
from local_voice_harness.integrations.github import GitHubClient, GitHubRepository
from local_voice_harness.integrations.herdr import (
    AgentSelection,
    HerdrClient,
    PromptOutcome,
)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class PersistentEffects:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "agents": {},
                "calls": {},
                "forks": {},
                "launches": [],
                "panes": {},
                "playbacks": [],
                "worktrees": {},
            }
        value = json.loads(self.path.read_text())
        if not isinstance(value, dict):
            raise RuntimeError("persistent fake state must be an object")
        return value

    def write(self, value: dict[str, Any]) -> None:
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w") as output:
            json.dump(value, output, allow_nan=False, sort_keys=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, self.path)
        _fsync_directory(self.path.parent)

    def update(self, change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        value = self.read()
        change(value)
        self.write(value)
        return value

    def record(self, kind: str, key: str, value: dict[str, Any]) -> dict[str, Any]:
        def change(state: dict[str, Any]) -> None:
            resources = cast(dict[str, Any], state[kind])
            calls = cast(dict[str, int], state["calls"])
            calls[f"{kind}:{key}"] = calls.get(f"{kind}:{key}", 0) + 1
            resources[key] = value

        return self.update(change)

    def record_launch(self, job_id: str) -> None:
        def change(state: dict[str, Any]) -> None:
            launches = cast(list[str], state["launches"])
            launches.append(job_id)

        self.update(change)

    def record_playback(self, job_id: str) -> None:
        def change(state: dict[str, Any]) -> None:
            playbacks = cast(list[str], state["playbacks"])
            playbacks.append(job_id)

        self.update(change)


class PersistentHerdr(HerdrClient):
    def __init__(self, effects: PersistentEffects) -> None:
        self.effects = effects

    def ensure_server(self, timeout: float = 15) -> None:
        return None

    def get_agent(self, target: str) -> dict[str, Any]:
        agents = cast(dict[str, dict[str, Any]], self.effects.read()["agents"])
        return dict(agents.get(target) or {})

    def run_json(self, *arguments: str, **_kwargs: object) -> dict[str, object]:
        if arguments[:2] == ("worktree", "list"):
            worktrees = cast(
                dict[str, dict[str, Any]], self.effects.read()["worktrees"]
            )
            return {"worktrees": list(worktrees.values())}
        raise RuntimeError(f"unsupported fake Herdr call: {arguments!r}")

    def cancel_agent(self, target: str) -> None:
        def change(state: dict[str, Any]) -> None:
            agents = cast(dict[str, Any], state["agents"])
            agents.pop(target, None)

        self.effects.update(change)


class PersistentGitHub(GitHubClient):
    def __init__(self, effects: PersistentEffects) -> None:
        self.effects = effects

    def reconcile_fork(
        self, source: GitHubRepository, target_name: str
    ) -> GitHubRepository | None:
        forks = cast(dict[str, dict[str, Any]], self.effects.read()["forks"])
        value = forks.get(target_name)
        if value is None:
            return None
        return GitHubRepository(
            name_with_owner=target_name,
            url=str(value["url"]),
            is_private=False,
            default_branch="main",
            parent=source.name_with_owner,
        )


class PromptHerdr(PersistentHerdr):
    def __init__(
        self,
        effects: PersistentEffects,
        kill: Callable[[str], None],
    ) -> None:
        super().__init__(effects)
        self.kill = kill

    def prompt_and_wait(
        self,
        target: str,
        text: str,
        *,
        token: str,
        timeout: float = 300,
        max_runtime: float = 1800,
        checkpoint: Callable[[], None] | None = None,
        baseline_sequence: int | None = None,
        expected_agent_session: str | None = None,
        before_submit: Callable[[int], None] | None = None,
        accepted: Callable[[], None] | None = None,
        before_agent: Callable[[dict[str, Any]], None] | None = None,
        after_submit: Callable[[dict[str, Any]], None] | None = None,
        active_marker: str | None = None,
        allow_enter_fallback: bool = True,
    ) -> PromptOutcome:
        del (
            text,
            timeout,
            max_runtime,
            checkpoint,
            expected_agent_session,
            before_agent,
            after_submit,
            active_marker,
            allow_enter_fallback,
        )
        if before_submit is None or accepted is None:
            raise RuntimeError("fake prompt requires durable boundary callbacks")
        before = self.get_agent(target)
        sequence = int(before.get("state_change_seq") or 0)
        if baseline_sequence != sequence:
            raise RuntimeError("fake prompt baseline changed")
        before_submit(sequence)
        self.kill("pre-submit")

        def submit(state: dict[str, Any]) -> None:
            agents = cast(dict[str, dict[str, Any]], state["agents"])
            calls = cast(dict[str, int], state["calls"])
            calls[f"prompts:{token}"] = calls.get(f"prompts:{token}", 0) + 1
            agent = dict(agents[target])
            agent["state_change_seq"] = sequence + 1
            agents[target] = agent

        self.effects.update(submit)
        self.kill("post-submit")
        accepted()
        self.kill("post-settle")
        return PromptOutcome("idle", None, None, "output")


def _kill_at(expected: str) -> Callable[[str], None]:
    def kill(point: str) -> None:
        if point == expected:
            os.kill(os.getpid(), signal.SIGKILL)

    return kill


def _fd_path(descriptor: int) -> Path | None:
    try:
        return Path(os.readlink(f"/proc/self/fd/{descriptor}"))
    except OSError:
        return None


def _run_migration(root: Path, point: str) -> None:
    from local_voice_harness.cursor import store as store_module

    legacy = root / "legacy"
    durable = root / "jobs"
    kill = _kill_at(point)
    original_fsync = os.fsync
    original_replace = os.replace
    original_unlink = Path.unlink
    original_directory_fsync = store_module._fsync_directory

    def fsync(descriptor: int) -> None:
        target = _fd_path(descriptor)
        original_fsync(descriptor)
        if (
            target is not None
            and target.parent == durable
            and target.name.endswith(".tmp")
        ):
            kill("file-fsync")

    def replace(source: os.PathLike[str] | str, target: os.PathLike[str] | str) -> None:
        destination = Path(target)
        if destination.parent == durable and destination.suffix == ".json":
            kill("pre-rename")
        original_replace(source, target)
        if destination.parent == durable and destination.suffix == ".json":
            kill("post-rename")

    def unlink(path: Path, missing_ok: bool = False) -> None:
        if path.parent == legacy and path.suffix == ".json":
            kill("pre-source-unlink")
        original_unlink(path, missing_ok=missing_ok)
        if path.parent == legacy and path.suffix == ".json":
            kill("post-source-unlink")

    def fsync_directory(path: Path) -> None:
        original_directory_fsync(path)
        if path == durable:
            kill("directory-fsync")

    with (
        mock.patch.object(os, "fsync", fsync),
        mock.patch.object(os, "replace", replace),
        mock.patch.object(Path, "unlink", unlink),
        mock.patch.object(store_module, "_fsync_directory", fsync_directory),
    ):
        migrate_legacy_jobs(
            legacy,
            durable,
            inspect_worker=lambda _job: "stopped",
        )


def _run_quarantine(root: Path, point: str) -> None:
    from local_voice_harness.cursor import store as store_module

    durable = root / "jobs"
    kill = _kill_at(point)
    original_atomic_json = store_module._atomic_json
    original_replace = os.replace
    original_directory_fsync = store_module._fsync_directory

    def atomic_json(path: Path, value: dict[str, object]) -> None:
        original_atomic_json(path, value)
        if path.parent == durable / ".quarantine":
            kill("quarantine-metadata")

    def replace(source: os.PathLike[str] | str, target: os.PathLike[str] | str) -> None:
        original_replace(source, target)
        if Path(source).parent == durable and Path(target).parent == (
            durable / ".quarantine"
        ):
            kill("quarantine-move")

    def fsync_directory(path: Path) -> None:
        original_directory_fsync(path)
        if path in {durable, durable / ".quarantine"}:
            kill("quarantine-fsync")

    store = JobStore(durable, root / "legacy")
    with (
        mock.patch.object(store_module, "_atomic_json", atomic_json),
        mock.patch.object(os, "replace", replace),
        mock.patch.object(store_module, "_fsync_directory", fsync_directory),
    ):
        store.list()


def _begin(store: JobStore, job_id: str) -> tuple[CursorJob, str]:
    claimed = worker_lifecycle.begin_worker(
        store,
        job_id,
        None,
        pid=os.getpid(),
        get_boot_identity=lambda: "crash-worker",
        get_process_identity=lambda _pid: "crash-start",
    )
    if claimed is None:
        raise RuntimeError("could not claim crash-test job")
    return claimed


def _run_effect(root: Path, effect: str, point: str) -> None:
    store = JobStore(root / "jobs", root / "legacy")
    effects = PersistentEffects(root / "effects.json")
    kill = _kill_at(point)
    job, token = _begin(store, "123456789abc")
    repository = root / "repository"
    checkout = root / "worktree"

    if effect == "worktree":
        provisioning._reserve_worker_worktree(
            store,
            job.id,
            token,
            repository,
            "voice/issue-100",
            checkout,
            state="planned",
        )
        provisioning._reserve_worker_worktree(
            store,
            job.id,
            token,
            repository,
            "voice/issue-100",
            checkout,
            state="dispatching",
        )
        kill("pre-submit")
        effects.record(
            "worktrees",
            "voice/issue-100",
            {
                "branch": "voice/issue-100",
                "path": str(checkout.resolve()),
                "open_workspace_id": "workspace-1",
            },
        )
        kill("post-submit")
        provisioning._settle_worker_worktree(
            store, job.id, token, checkout, "workspace-1", "pane-1"
        )
        kill("post-settle")
        return

    if effect == "agent":
        selection = AgentSelection(
            "voice-issue-100",
            "pane-1",
            "workspace-1",
            str(checkout.resolve()),
            "voice-issue-100",
            str(checkout.resolve()),
        )
        provisioning._reserve_worker_target(
            store,
            job.id,
            token,
            selection,
            repository,
            None,
            dispatching=True,
        )
        kill("pre-submit")
        effects.record(
            "agents",
            selection.target,
            {
                "name": selection.name,
                "pane_id": selection.pane_id,
                "workspace_id": selection.workspace_id,
                "cwd": selection.cwd,
                "state_change_seq": 1,
            },
        )
        kill("post-submit")
        provisioning._settle_worker_agent(store, job.id, token, selection)
        kill("post-settle")
        return

    if effect == "fork":
        source = GitHubRepository(
            "upstream/project",
            "https://github.com/upstream/project",
            False,
            "main",
            None,
        )
        provisioning._begin_fork_operation(
            store, job.id, token, source, "voice-user", "voice-user/project"
        )
        provisioning._mark_fork_dispatching(store, job.id, token)
        kill("pre-submit")
        effects.record(
            "forks",
            "voice-user/project",
            {"url": "https://github.com/voice-user/project"},
        )
        kill("post-submit")
        fork = GitHubRepository(
            "voice-user/project",
            "https://github.com/voice-user/project",
            False,
            "main",
            "upstream/project",
        )
        provisioning._settle_fork_operation(store, job.id, token, fork)
        kill("post-settle")
        return

    if effect == "pane":
        planned = provisioning._plan_participant_creation(
            store,
            job,
            token,
            WorkflowParticipant.REVIEWER,
            target="voice-reviewer",
            label="issue-100-reviewer",
            workspace_id="workspace-1",
        )
        before_submit, accepted = provisioning._participant_pane_callbacks(
            store, planned.id, token, "voice-reviewer"
        )
        before_submit()
        kill("pre-submit")
        effects.record(
            "panes",
            "pane-reviewer",
            {"pane_id": "pane-reviewer", "workspace_id": "workspace-1"},
        )
        kill("post-submit")
        accepted("pane-reviewer", "workspace-1")
        kill("post-settle")
        return

    if effect == "prompt":
        if not cast(dict[str, Any], effects.read()["agents"]).get("planner"):
            effects.record(
                "agents",
                "planner",
                {
                    "name": "planner",
                    "pane_id": "pane-1",
                    "workspace_id": "workspace-1",
                    "cwd": str(checkout.resolve()),
                    "state_change_seq": 7,
                },
            )
        client = PromptHerdr(effects, kill)
        provisioning._execute_phase_prompt(
            store,
            job,
            token,
            client,
            lambda: None,
            target="planner",
            prompt="implement issue 100",
            token="123456789abc-1",
        )
        return

    raise RuntimeError(f"unknown effect {effect!r}")


def _run_recovery(root: Path) -> None:
    store = JobStore(root / "jobs", root / "legacy")
    effects = PersistentEffects(root / "effects.json")

    def herdr() -> PersistentHerdr:
        return PersistentHerdr(effects)

    def github() -> PersistentGitHub:
        return PersistentGitHub(effects)

    def is_worker_alive(job: CursorJob) -> bool:
        return bool(
            job.worker_token
            and job.worker_pid == os.getpid()
            and job.worker_boot_id == "recovery-worker"
        )

    def launch(job_id: str) -> None:
        effects.record_launch(job_id)
        worker_lifecycle.begin_worker(
            store,
            job_id,
            None,
            pid=os.getpid(),
            get_boot_identity=lambda: "recovery-worker",
            get_process_identity=lambda _pid: "recovery-start",
        )

    snapshots: list[tuple[list[dict[str, object]], dict[str, Any]]] = []
    for pass_number in range(7):
        recover_jobs(
            store,
            launch_worker=launch,
            herdr_factory=herdr,
            github_factory=github,
            is_worker_alive=is_worker_alive,
            inspect_legacy_worker=lambda _job: "stopped",
            get_boot_identity=lambda: "recovery-worker",
            get_process_identity=lambda _pid: "recovery-start",
            now=1000.0 * (pass_number + 1),
        )
        snapshots.append(([job.to_record() for job in store.list()], effects.read()))
    if snapshots[-1] != snapshots[-2]:
        raise RuntimeError("repeated recovery was not idempotent")


def _run_delivery(root: Path, point: str, now: float) -> None:
    store = JobStore(root / "jobs", root / "legacy")
    effects = PersistentEffects(root / "effects.json")
    kill = _kill_at(point)
    claim = claim_delivery(store, "123456789abc", foreground=True, now=now)
    if claim is None:
        return
    kill("after-claim")
    effects.record_playback(claim.job.id)
    kill("after-playback")
    acknowledge_delivery(store, claim.job.id, claim.token, now=now + 1)
    kill("after-ack")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: cursor_crash_child.py COMMAND ROOT [ARG ...]")
    command = sys.argv[1]
    root = Path(sys.argv[2]).resolve()
    if command == "migrate":
        _run_migration(root, sys.argv[3])
    elif command == "quarantine":
        _run_quarantine(root, sys.argv[3])
    elif command == "effect":
        _run_effect(root, sys.argv[3], sys.argv[4])
    elif command == "recover":
        _run_recovery(root)
    elif command == "delivery":
        _run_delivery(root, sys.argv[3], float(sys.argv[4]))
    else:
        raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main()
