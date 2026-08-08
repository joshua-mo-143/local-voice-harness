from __future__ import annotations

import os
import secrets
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import cast

from ..errors import HarnessError
from ..integrations.github import GitHubClient, GitHubError, GitHubRepository
from ..integrations.herdr import HerdrClient, HerdrError
from .model import CursorJob, JobStatus
from .store import JobStore, LegacyWorkerInspector
from .worker_lifecycle import (
    boot_identity,
    has_legacy_worker_claim,
    inspect_and_stop_legacy_worker,
    process_identity,
    stop_legacy_worker,
    stop_worker,
    worker_is_alive,
)

FAILED_RECONCILE_MAX_ATTEMPTS = 3
UNCERTAIN_RECONCILE_MAX_ATTEMPTS = 6
OPERATION_RECONCILE_BASE_SECONDS = 5.0
OPERATION_RECONCILE_MAX_SECONDS = 60.0
DELIVERY_RETRY_SECONDS = 5.0

HerdrFactory = Callable[[], HerdrClient]
GitHubFactory = Callable[[], GitHubClient]
LaunchWorker = Callable[[str], None]


def _agent_not_found(exc: HerdrError) -> bool:
    return exc.code in {"agent_not_found", "not_found"}


def _reconciliation_due(job: CursorJob, prefix: str, now: float) -> bool:
    return now >= job.operation_reconcile_at(prefix)


def _record_reconciliation_observation(
    store: JobStore,
    job_id: str,
    prefix: str,
    state_key: str,
    expected_states: frozenset[str],
    *,
    now: float,
    observed_absent: bool,
) -> None:
    def observe(job: CursorJob) -> CursorJob | None:
        return job.record_operation_observation(
            prefix,
            state_key,
            expected_states,
            now=now,
            observed_absent=observed_absent,
            failed_max_attempts=FAILED_RECONCILE_MAX_ATTEMPTS,
            uncertain_max_attempts=UNCERTAIN_RECONCILE_MAX_ATTEMPTS,
            base_seconds=OPERATION_RECONCILE_BASE_SECONDS,
            max_seconds=OPERATION_RECONCILE_MAX_SECONDS,
        )

    store.update(job_id, observe)


def reconcile_uncertain_agent(
    store: JobStore,
    job: CursorJob,
    *,
    now: float,
    herdr_factory: HerdrFactory = HerdrClient,
) -> None:
    states = frozenset({"dispatching", "ambiguous", "failed_observing"})
    if job.agent_dispatch_state not in states or not _reconciliation_due(
        job, "agent", now
    ):
        return
    target = job.herdr_target or ""
    if not target:
        _record_reconciliation_observation(
            store,
            job.id,
            "agent",
            "agent_dispatch_state",
            states,
            now=now,
            observed_absent=False,
        )
        return
    try:
        client = herdr_factory()
        client.ensure_server()
        agent = client.get_agent(target)
    except HerdrError as exc:
        _record_reconciliation_observation(
            store,
            job.id,
            "agent",
            "agent_dispatch_state",
            states,
            now=now,
            observed_absent=_agent_not_found(exc),
        )
        return
    if not agent:
        _record_reconciliation_observation(
            store,
            job.id,
            "agent",
            "agent_dispatch_state",
            states,
            now=now,
            observed_absent=True,
        )
        return

    def visible(current: CursorJob) -> CursorJob | None:
        if current.agent_dispatch_state not in states or current.herdr_target != target:
            return None
        return current.evolve_recovery(
            now=now,
            agent_dispatch_state="ready",
            agent_name=str(agent.get("name") or target),
            herdr_pane_id=str(agent.get("pane_id") or current.herdr_pane_id or ""),
            herdr_workspace_id=str(
                agent.get("workspace_id") or current.herdr_workspace_id or ""
            ),
            worker_operation=None,
            agent_next_reconcile_at=None,
        )

    store.update(job.id, visible)


def _fork_source(job: CursorJob) -> GitHubRepository | None:
    name = job.fork_operation_source or ""
    url = job.fork_operation_source_url or ""
    if not name or not url:
        return None
    return GitHubRepository(
        name_with_owner=name,
        url=url,
        is_private=job.fork_operation_source_private,
        default_branch=job.fork_operation_source_default_branch or "",
        parent=job.fork_operation_source_parent,
    )


def reconcile_uncertain_fork(
    store: JobStore,
    job: CursorJob,
    *,
    now: float,
    github_factory: GitHubFactory = GitHubClient,
) -> None:
    states = frozenset({"submitted", "ambiguous", "failed_observing"})
    if job.fork_operation_state not in states or not _reconciliation_due(
        job, "fork", now
    ):
        return
    source = _fork_source(job)
    target = job.fork_operation_target or ""
    if source is None or not target:
        _record_reconciliation_observation(
            store,
            job.id,
            "fork",
            "fork_operation_state",
            states,
            now=now,
            observed_absent=False,
        )
        return
    try:
        fork = github_factory().reconcile_fork(source, target)
    except GitHubError:
        _record_reconciliation_observation(
            store,
            job.id,
            "fork",
            "fork_operation_state",
            states,
            now=now,
            observed_absent=False,
        )
        return
    if fork is None:
        _record_reconciliation_observation(
            store,
            job.id,
            "fork",
            "fork_operation_state",
            states,
            now=now,
            observed_absent=True,
        )
        return

    def visible(current: CursorJob) -> CursorJob | None:
        if current.fork_operation_state not in states:
            return None
        return current.evolve_recovery(
            now=now,
            fork_operation_state="exists",
            fork_exists=True,
            fork_repository=fork.name_with_owner,
            worker_operation=None,
            fork_next_reconcile_at=None,
        )

    store.update(job.id, visible)


def reconcile_uncertain_worktree(
    store: JobStore,
    job: CursorJob,
    *,
    now: float,
    herdr_factory: HerdrFactory = HerdrClient,
) -> None:
    states = frozenset({"dispatching", "ambiguous", "failed_observing"})
    if job.worktree_provision_state not in states or not _reconciliation_due(
        job, "worktree", now
    ):
        return
    repository = job.repository or ""
    branch = job.worktree_branch or ""
    checkout_value = job.worktree_path or ""
    if not repository or not branch or not checkout_value:
        _record_reconciliation_observation(
            store,
            job.id,
            "worktree",
            "worktree_provision_state",
            states,
            now=now,
            observed_absent=False,
        )
        return
    checkout = Path(checkout_value).resolve()
    try:
        client = herdr_factory()
        client.ensure_server()
        listing = client.run_json("worktree", "list", "--cwd", repository, "--json")
    except HerdrError:
        _record_reconciliation_observation(
            store,
            job.id,
            "worktree",
            "worktree_provision_state",
            states,
            now=now,
            observed_absent=False,
        )
        return
    worktrees = cast(list[dict[str, object]], listing.get("worktrees") or [])
    match = next(
        (
            item
            for item in worktrees
            if item.get("branch") == branch
            and Path(str(item.get("path") or "")).resolve() == checkout
        ),
        None,
    )
    if match is None and not checkout.exists():
        _record_reconciliation_observation(
            store,
            job.id,
            "worktree",
            "worktree_provision_state",
            states,
            now=now,
            observed_absent=True,
        )
        return

    def reconciled(current: CursorJob) -> CursorJob | None:
        if current.worktree_provision_state not in states:
            return None
        if match is None:
            return current.evolve_recovery(
                now=now,
                worktree_provision_state="quarantined",
                worktree_provision_error=(
                    "reserved worktree path exists but Herdr did not report the "
                    "expected branch"
                ),
                worktree_manual_inspection_required=True,
                worker_operation=None,
                cancellation_reconciliation_pending=False,
            )
        return current.evolve_recovery(
            now=now,
            worktree_provision_state="retained",
            worktree_workspace_id=str(match.get("open_workspace_id") or "") or None,
            worktree_provision_error=None,
            worker_operation=None,
            worktree_manual_inspection_required=False,
            worktree_next_reconcile_at=None,
            cancellation_reconciliation_pending=False,
        )

    store.update(job.id, reconciled)


def reconcile_uncertain_operations(
    store: JobStore,
    job: CursorJob,
    *,
    now: float,
    herdr_factory: HerdrFactory = HerdrClient,
    github_factory: GitHubFactory = GitHubClient,
) -> None:
    reconcile_uncertain_agent(store, job, now=now, herdr_factory=herdr_factory)
    current = store.get(job.id)
    reconcile_uncertain_fork(store, current, now=now, github_factory=github_factory)
    current = store.get(job.id)
    reconcile_uncertain_worktree(store, current, now=now, herdr_factory=herdr_factory)


def cancel_target_and_release(
    store: JobStore,
    job_id: str,
    target: str,
    release_token: str,
    *,
    worker_stopped: bool = True,
    herdr_factory: HerdrFactory = HerdrClient,
) -> None:
    current = store.get(job_id)
    if (
        current.status not in {JobStatus.CANCELLED, JobStatus.FAILED}
        or current.target_release_token != release_token
    ):
        return
    interrupted = not target
    if target:
        try:
            client = herdr_factory()
            client.ensure_server()
            client.cancel_agent(target)
            interrupted = True
        except HerdrError as exc:
            interrupted = _agent_not_found(exc)

    def release(job: CursorJob) -> CursorJob | None:
        if (
            job.status not in {JobStatus.CANCELLED, JobStatus.FAILED}
            or job.target_release_token != release_token
        ):
            return None
        if interrupted and worker_stopped and not job.has_uncertain_operation():
            return job.evolve(
                target_release_pending=False,
                target_release_token=None,
                target_release_owner_pid=None,
                target_release_owner_boot_id=None,
                target_release_owner_start=None,
                cancellation_reconciliation_pending=False,
                worker_pid=None,
                worker_boot_id=None,
                worker_process_start=None,
                worker_token=None,
                result=(
                    f"Cursor job {job_id} was cancelled."
                    if job.status == JobStatus.CANCELLED
                    else job.result
                ),
            )
        return job.evolve(
            target_release_owner_pid=None,
            target_release_owner_boot_id=None,
            target_release_owner_start=None,
        )

    store.update(job_id, release)


def acknowledge_worktree_quarantine(
    store: JobStore, job_id: str, *, now: float | None = None
) -> None:
    acknowledged_at = time.time() if now is None else now

    def acknowledge(job: CursorJob) -> CursorJob | None:
        if (
            job.worktree_provision_state != "quarantined"
            or not job.worktree_manual_inspection_required
        ):
            return None
        return job.evolve(
            worktree_provision_state="retained",
            worktree_manual_inspection_required=False,
            worktree_quarantine_acknowledged_at=acknowledged_at,
        )

    if store.update(job_id, acknowledge) is None:
        raise HarnessError(
            f"Cursor job {job_id} has no worktree quarantine to acknowledge"
        )


def resolve_manual_reconciliation(
    store: JobStore,
    job_id: str,
    operation: str,
    token: str,
    outcome: str,
    *,
    now: float | None = None,
) -> CursorJob:
    if operation not in {"agent", "fork", "worktree"}:
        raise HarnessError("manual reconciliation operation is invalid")
    if outcome not in {"confirmed_absent", "materialized"}:
        raise HarnessError("manual reconciliation outcome is invalid")
    resolved_at = time.time() if now is None else now

    def resolve(job: CursorJob) -> CursorJob | None:
        if (
            job.manual_reconcile_operation != operation
            or not secrets.compare_digest(job.manual_reconcile_token or "", token)
            or job.operation_state(operation) != "manual_required"
        ):
            return None
        return job.resolve_manual_operation(operation, outcome, resolved_at=resolved_at)

    resolved = store.update(job_id, resolve)
    if resolved is None:
        raise HarnessError(f"Cursor job {job_id} manual reconciliation fence is stale")
    return resolved


def _target_release_owner_alive(
    job: CursorJob,
    *,
    get_boot_identity: Callable[[], str | None],
    get_process_identity: Callable[[int], str | None],
) -> bool:
    pid = job.target_release_owner_pid or 0
    expected = job.target_release_owner_start or ""
    expected_boot = job.target_release_owner_boot_id or ""
    return bool(
        pid
        and expected
        and expected_boot
        and get_boot_identity() == expected_boot
        and get_process_identity(pid) == expected
    )


def recover_jobs(
    store: JobStore,
    *,
    launch_worker: LaunchWorker,
    herdr_factory: HerdrFactory = HerdrClient,
    github_factory: GitHubFactory = GitHubClient,
    is_worker_alive: Callable[[CursorJob], bool] = worker_is_alive,
    stop_owned_worker: Callable[[CursorJob], bool] = stop_worker,
    stop_unfenced_worker: Callable[[JobStore, str], bool] = stop_legacy_worker,
    get_boot_identity: Callable[[], str | None] = boot_identity,
    get_process_identity: Callable[[int], str | None] = process_identity,
    inspect_legacy_worker: LegacyWorkerInspector = inspect_and_stop_legacy_worker,
    now: float | None = None,
) -> None:
    blocked_legacy_jobs = store.migrate_legacy(inspect_worker=inspect_legacy_worker)
    store.prune(now=now)
    recovered_at = time.time() if now is None else now

    for existing in store.list():
        if not has_legacy_worker_claim(existing):
            continue
        disposition = inspect_legacy_worker(existing)
        if disposition == "unsafe":
            blocked_legacy_jobs.add(existing.id)
            continue

        def clear_legacy_owner(job: CursorJob) -> CursorJob | None:
            if not has_legacy_worker_claim(job):
                return None
            active = job.status in {
                JobStatus.QUEUED,
                JobStatus.ROUTING,
                JobStatus.RUNNING,
                JobStatus.RECONCILING,
            }
            return job.evolve(
                status=JobStatus.QUEUED if active else job.status,
                queued_at=(job.queued_at or recovered_at) if active else job.queued_at,
                reconcile=bool(
                    active and (job.herdr_target or job.has_uncertain_operation())
                ),
                worker_token=None,
                worker_pid=None,
                worker_boot_id=None,
                worker_process_start=None,
            )

        store.update(existing.id, clear_legacy_owner)

    for existing in store.list():
        if existing.id in blocked_legacy_jobs:
            continue
        if existing.has_uncertain_operation():
            reconcile_uncertain_operations(
                store,
                existing,
                now=recovered_at,
                herdr_factory=herdr_factory,
                github_factory=github_factory,
            )
    for existing in store.list():
        if (
            existing.status
            in {JobStatus.ROUTING, JobStatus.RUNNING, JobStatus.RECONCILING}
            and not existing.worker_token
            and is_worker_alive(existing)
        ):
            stop_unfenced_worker(store, existing.id)
        elif (
            existing.status == JobStatus.CANCELLED
            and existing.target_release_pending
            and existing.worker_token
            and is_worker_alive(existing)
        ):
            stop_owned_worker(existing)

    launches: list[str] = []
    releases: list[tuple[str, str, str]] = []
    for snapshot in store.list():
        if snapshot.id in blocked_legacy_jobs:
            continue
        should_launch = False
        release: tuple[str, str, str] | None = None

        def recover(job: CursorJob) -> CursorJob | None:
            nonlocal should_launch, release
            take_release = (
                job.target_release_pending
                and not _target_release_owner_alive(
                    job,
                    get_boot_identity=get_boot_identity,
                    get_process_identity=get_process_identity,
                )
                and not is_worker_alive(job)
                and not job.has_uncertain_operation()
                and not (
                    job.agent_dispatch_state == "manual_required"
                    and bool(job.herdr_target)
                )
            )
            release_token = job.target_release_token
            owner_pid = job.target_release_owner_pid
            owner_boot_id = job.target_release_owner_boot_id
            owner_start = job.target_release_owner_start
            if take_release:
                release_token = uuid.uuid4().hex
                owner_pid = os.getpid()
                owner_start = get_process_identity(owner_pid)
                owner_pid = owner_pid if owner_start else None
                owner_boot_id = get_boot_identity() if owner_start else None
                release = (job.id, job.herdr_target or "", release_token)
            if (
                job.has_uncertain_operation()
                or job.manual_reconcile_operation
                or job.worktree_provision_state in {"quarantined", "manual_required"}
                or job.pull_request_worktree_state == "quarantined"
            ):
                if take_release:
                    return job.evolve(
                        target_release_token=release_token,
                        target_release_owner_pid=owner_pid,
                        target_release_owner_boot_id=owner_boot_id,
                        target_release_owner_start=owner_start,
                        worker_pid=None,
                        worker_boot_id=None,
                        worker_process_start=None,
                        worker_token=None,
                    )
                return None
            if job.status == JobStatus.BLOCKED:
                if (
                    job.delivered
                    and job.herdr_target
                    and recovered_at >= job.next_reconcile_at
                ):
                    should_launch = True
                    return job.evolve(
                        status=JobStatus.QUEUED,
                        reconcile=True,
                        queued_at=recovered_at,
                        next_reconcile_at=recovered_at + DELIVERY_RETRY_SECONDS,
                        target_release_token=release_token,
                        target_release_owner_pid=owner_pid,
                        target_release_owner_boot_id=owner_boot_id,
                        target_release_owner_start=owner_start,
                        worker_pid=None,
                        worker_boot_id=None,
                        worker_process_start=None,
                        worker_token=None,
                    )
            elif job.status == JobStatus.QUEUED:
                if not job.worker_token or not is_worker_alive(job):
                    should_launch = True
                    return job.evolve(
                        queued_at=job.queued_at or recovered_at,
                        target_release_token=release_token,
                        target_release_owner_pid=owner_pid,
                        target_release_owner_boot_id=owner_boot_id,
                        target_release_owner_start=owner_start,
                        worker_pid=None,
                        worker_boot_id=None,
                        worker_process_start=None,
                        worker_token=None,
                    )
            elif job.status == JobStatus.ROUTING:
                if not is_worker_alive(job):
                    should_launch = True
                    return job.evolve(
                        status=JobStatus.QUEUED,
                        remove=frozenset({"reconcile"}),
                        queued_at=recovered_at,
                        target_release_token=release_token,
                        target_release_owner_pid=owner_pid,
                        target_release_owner_boot_id=owner_boot_id,
                        target_release_owner_start=owner_start,
                        worker_pid=None,
                        worker_boot_id=None,
                        worker_process_start=None,
                        worker_token=None,
                    )
            elif job.status in {JobStatus.RUNNING, JobStatus.RECONCILING}:
                if not is_worker_alive(job):
                    if job.herdr_target:
                        should_launch = True
                        return job.evolve(
                            status=JobStatus.QUEUED,
                            reconcile=True,
                            queued_at=recovered_at,
                            target_release_token=release_token,
                            target_release_owner_pid=owner_pid,
                            target_release_owner_boot_id=owner_boot_id,
                            target_release_owner_start=owner_start,
                            worker_pid=None,
                            worker_boot_id=None,
                            worker_process_start=None,
                            worker_token=None,
                        )
                    message = "Cursor job was interrupted before an agent started"
                    return job.evolve_recovery(
                        now=recovered_at,
                        status=JobStatus.FAILED,
                        remove=frozenset({"reconcile"}),
                        prepare_delivery=True,
                        error=message,
                        result=message,
                        completed_at=recovered_at,
                        target_release_token=release_token,
                        target_release_owner_pid=owner_pid,
                        target_release_owner_boot_id=owner_boot_id,
                        target_release_owner_start=owner_start,
                        worker_pid=None,
                        worker_boot_id=None,
                        worker_process_start=None,
                        worker_token=None,
                    )
            if take_release:
                return job.evolve(
                    target_release_token=release_token,
                    target_release_owner_pid=owner_pid,
                    target_release_owner_boot_id=owner_boot_id,
                    target_release_owner_start=owner_start,
                    worker_pid=None,
                    worker_boot_id=None,
                    worker_process_start=None,
                    worker_token=None,
                )
            return None

        updated = store.update(snapshot.id, recover)
        if updated is not None:
            if release is not None:
                releases.append(release)
            if should_launch:
                launches.append(updated.id)

    for job_id, target, token in releases:
        cancel_target_and_release(
            store,
            job_id,
            target,
            token,
            herdr_factory=herdr_factory,
        )
    for job_id in launches:
        try:
            launch_worker(job_id)
        except Exception:
            # The launch boundary persists a deliverable failure.
            continue
