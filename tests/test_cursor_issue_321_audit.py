from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, cast
from unittest import mock

import pytest

from local_voice_harness.agents.harness import (
    HarnessSession,
    ReconciliationState,
    SessionReconciliation,
)
from local_voice_harness.cursor.model import (
    CURRENT_SCHEMA_VERSION,
    CursorJob,
    JobStatus,
    JobValidationError,
    WorkflowParticipant,
)
from local_voice_harness.cursor.operations import OperationState, WorkerOwnership
from local_voice_harness.cursor.provisioning import (
    _fence_participant_creation,
    _participant_pane_callbacks,
    _plan_participant_creation,
    _settle_worker_agent,
    _worker_change,
)
from local_voice_harness.cursor.recovery import (
    cancel_target_and_release,
    reconcile_uncertain_agent,
    reconcile_uncertain_worktree,
)
from local_voice_harness.cursor.store import JobStore
from local_voice_harness.cursor.worker_lifecycle import WorkerCancelled, WorkerContext
from local_voice_harness.integrations.herdr import AgentSelection, HerdrError


def _current_job(**changes: object) -> CursorJob:
    values: dict[str, object] = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "harness_kind": "cursor",
        "id": "123456789abc",
        "revision": 0,
        "status": "routing",
        "request": "audit",
        "created_at": 1,
        "delivered": False,
        "worker_token": "worker",
        "worker_pid": 42,
        "worker_boot_id": "boot",
        "worker_process_start": "start",
        "worker_claim_operation": "routing",
        "worker_claimed_at": 2,
        "repository": "/repo",
        "worktree_branch": "voice/audit",
        "worktree_path": "/checkout",
        "worktree_workspace_id": "workspace",
        "worktree_root_pane_id": "root-pane",
        "worktree_provision_state": "ready",
        "herdr_target": "agent",
        "herdr_pane_id": "pane",
        "herdr_workspace_id": "workspace",
        "agent_dispatch_state": "dispatching",
    }
    values.update(changes)
    return CursorJob.from_dict(values)


def _owner(job: CursorJob) -> WorkerOwnership:
    owner = job.worker_ownership
    assert owner is not None
    return owner


def _store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "jobs", tmp_path / "legacy")


def test_v17_worker_claims_migrate_without_weakening_v18() -> None:
    values = _current_job().to_dict()
    values["worker_claim_operation"] = None
    values["worker_claimed_at"] = None
    with pytest.raises(JobValidationError, match="complete worker ownership"):
        CursorJob.from_dict(values)

    values["schema_version"] = CURRENT_SCHEMA_VERSION - 1
    migrated = CursorJob.from_dict(values)
    assert migrated.schema_version == CURRENT_SCHEMA_VERSION
    assert migrated.worker_ownership == WorkerOwnership(
        "worker", 42, "boot", "start", "legacy:routing", 1
    )

    values["worker_process_start"] = None
    incomplete = CursorJob.from_dict(values)
    with pytest.raises(JobValidationError, match="worker ownership is incomplete"):
        _ = incomplete.worker_ownership
    assert incomplete.loaded_schema_version == CURRENT_SCHEMA_VERSION - 1
    assert incomplete.worker_pid == 42


def test_v17_incomplete_operations_migrate_to_safe_typed_states() -> None:
    values = _current_job().to_dict()
    values.update(
        schema_version=17,
        worktree_workspace_id=None,
        worktree_root_pane_id=None,
        agent_dispatch_state="ready",
        agent_provider=None,
        agent_provider_session_id=None,
        agent_state_sequence=None,
        participant_creation_state="created",
        participant_creation_participant="planner",
        participant_creation_target="agent",
        participant_creation_label="planner",
        participant_creation_workspace_id="workspace",
        participant_creation_pane_id="pane",
        participant_creation_checkout=None,
        fork_operation_state="exists",
        fork_operation_source="owner/repo",
        fork_operation_source_url=None,
        fork_operation_source_default_branch="main",
        fork_operation_source_private=False,
        fork_operation_login="owner",
        fork_operation_target="owner/repo",
    )

    migrated = CursorJob.from_dict(values)

    assert migrated.worktree_provision_state == "ready"
    assert migrated.agent_dispatch_state == "ready"
    assert migrated.agent_operation_checkout == "/checkout"
    assert migrated.participant_creation_checkout == "/checkout"
    assert migrated.fork_operation_state == "ambiguous"
    assert migrated.checkout_operation is not None
    assert migrated.checkout_operation.state == OperationState.UNKNOWN
    assert migrated.agent_session_operation is not None
    assert migrated.agent_session_operation.state == OperationState.UNKNOWN
    assert migrated.participant_pane_operation is not None
    assert migrated.fork_operation is not None


def test_worker_mutations_compare_complete_owner_and_reject_token_only(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    job = store.create(_current_job())
    owner = _owner(job)
    stale = WorkerOwnership(
        owner.token,
        owner.pid + 1,
        owner.boot_id,
        owner.process_start,
        owner.operation,
        owner.claimed_at,
    )

    def change(current: CursorJob) -> CursorJob:
        return current.evolve(status=JobStatus.RUNNING)

    assert _worker_change(store, job.id, "worker", {JobStatus.ROUTING}, change) is None
    assert _worker_change(store, job.id, stale, {JobStatus.ROUTING}, change) is None
    assert _worker_change(store, job.id, owner, {JobStatus.ROUTING}, change) is not None


def test_worker_context_rejects_legacy_incomplete_claim(tmp_path: Path) -> None:
    values = _current_job().to_dict()
    values.update(
        schema_version=CURRENT_SCHEMA_VERSION - 1,
        worker_process_start=None,
        worker_claim_operation=None,
        worker_claimed_at=None,
    )
    job = CursorJob.from_dict(values)
    store = mock.Mock()
    store.get_unless_maintenance.return_value = job
    context = WorkerContext(cast(JobStore, store), job, "worker", threading.Event())
    with pytest.raises(WorkerCancelled):
        context.checkpoint()


def test_current_dispatch_without_provider_identity_requires_manual(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    job = store.create(
        _current_job(
            agent_dispatch_state="ambiguous",
            agent_provider=None,
            agent_provider_session_id=None,
            agent_state_sequence=None,
        )
    )
    client = mock.Mock()

    reconcile_uncertain_agent(store, job, now=10, herdr_factory=lambda: client)

    current = store.get(job.id)
    assert current.agent_dispatch_state == "manual_required"
    assert current.manual_reconcile_operation == "agent"
    client.get_agent.assert_not_called()
    client.reconcile_session.assert_not_called()


def test_cleanup_rejects_replacement_session_and_retains_reservation(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    job = store.create(
        _current_job(
            status="cancelled",
            completed_at=3,
            result="cancelled",
            worker_token=None,
            worker_pid=None,
            worker_boot_id=None,
            worker_process_start=None,
            worker_claim_operation=None,
            worker_claimed_at=None,
            agent_dispatch_state="ready",
            agent_provider="cursor/herdr",
            agent_provider_session_id="owned-session",
            agent_state_sequence=7,
            target_release_pending=True,
            target_release_token="release",
        )
    )
    client = mock.Mock()
    client.reconcile_session.return_value = SessionReconciliation(
        ReconciliationState.ACTIVE,
        HarnessSession("cursor/herdr", "replacement", "agent", 8),
        "replacement",
        True,
    )

    cancel_target_and_release(
        store,
        job.id,
        "agent",
        "release",
        herdr_factory=lambda: client,
    )

    client.close_owned_pane.assert_not_called()
    client.reconcile_session.assert_called_once_with(
        "agent", expected_session_id="owned-session"
    )
    assert store.get(job.id).target_release_pending


def test_v17_cleanup_requires_live_binding_and_known_root_pane(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    values = _current_job().to_dict()
    values.update(
        schema_version=17,
        status="cancelled",
        completed_at=3,
        result="cancelled",
        worker_token=None,
        worker_pid=None,
        worker_boot_id=None,
        worker_process_start=None,
        worker_claim_operation=None,
        worker_claimed_at=None,
        agent_dispatch_state="ready",
        worktree_root_pane_id=None,
        target_release_pending=True,
        target_release_token="release",
    )
    job = store.create(CursorJob.from_dict(values))
    client = mock.Mock()
    client.get_agent.return_value = {
        "pane_id": "pane",
        "workspace_id": "workspace",
        "cwd": "/checkout",
    }

    cancel_target_and_release(
        store,
        job.id,
        "agent",
        "release",
        herdr_factory=lambda: client,
    )

    client.get_agent.assert_called_once_with("agent")
    client.close_owned_pane.assert_not_called()
    current = store.get(job.id)
    assert current.target_release_pending
    assert current.target_release_manual_required
    assert current.target_release_unverified_targets == ("agent",)


def test_multi_target_cleanup_retains_unverified_historical_target(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    job = store.create(
        _current_job(
            status="cancelled",
            completed_at=3,
            result="cancelled",
            worker_token=None,
            worker_pid=None,
            worker_boot_id=None,
            worker_process_start=None,
            worker_claim_operation=None,
            worker_claimed_at=None,
            agent_dispatch_state="ready",
            agent_provider="cursor/herdr",
            agent_provider_session_id="owned-session",
            agent_state_sequence=7,
            planner_target="historical-agent",
            target_release_pending=True,
            target_release_token="release",
        )
    )
    client = mock.Mock()
    client.reconcile_session.return_value = SessionReconciliation(
        ReconciliationState.ACTIVE,
        HarnessSession(
            "cursor/herdr",
            "owned-session",
            "agent",
            8,
            {"pane_id": "pane", "workspace_id": "workspace", "cwd": "/checkout"},
        ),
        "active",
        True,
    )

    cancel_target_and_release(
        store,
        job.id,
        "agent",
        "release",
        herdr_factory=lambda: client,
    )

    client.close_owned_pane.assert_called_once_with("agent", "pane", "workspace")
    current = store.get(job.id)
    assert current.target_release_pending
    assert current.target_release_manual_required
    assert current.target_release_unverified_targets == ("historical-agent",)
    assert current.participant_target(WorkflowParticipant.PLANNER) == "historical-agent"


def test_multi_target_cleanup_uses_each_persisted_session_owner(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    job = store.create(
        _current_job(
            status="cancelled",
            completed_at=3,
            result="cancelled",
            worker_token=None,
            worker_pid=None,
            worker_boot_id=None,
            worker_process_start=None,
            worker_claim_operation=None,
            worker_claimed_at=None,
            agent_dispatch_state="ready",
            agent_provider="cursor/herdr",
            agent_provider_session_id="current-session",
            agent_state_sequence=7,
            planner_target="historical-agent",
            participant_session_owners=[
                {
                    "provider": "cursor/herdr",
                    "session_id": "historical-session",
                    "target": "historical-agent",
                    "state_sequence": 5,
                    "checkout": "/checkout",
                    "workspace_id": "workspace",
                    "pane_id": "historical-pane",
                }
            ],
            target_release_pending=True,
            target_release_token="release",
            target_release_owner_pid=42,
            target_release_owner_boot_id="release-boot",
            target_release_owner_start="release-start",
        )
    )
    client = mock.Mock()

    def reconcile(target: str, *, expected_session_id: str) -> SessionReconciliation:
        pane = "pane" if target == "agent" else "historical-pane"
        sequence = 8 if target == "agent" else 6
        return SessionReconciliation(
            ReconciliationState.ACTIVE,
            HarnessSession(
                "cursor/herdr",
                expected_session_id,
                target,
                sequence,
                {
                    "pane_id": pane,
                    "workspace_id": "workspace",
                    "cwd": "/checkout",
                },
            ),
            "active",
            True,
        )

    client.reconcile_session.side_effect = reconcile

    cancel_target_and_release(
        store,
        job.id,
        "agent",
        "release",
        herdr_factory=lambda: client,
    )

    assert client.close_owned_pane.call_args_list == [
        mock.call("agent", "pane", "workspace"),
        mock.call("historical-agent", "historical-pane", "workspace"),
    ]
    current = store.get(job.id)
    assert not current.target_release_pending
    assert not current.target_release_manual_required


def test_current_settled_checkout_requires_workspace_and_root_pane() -> None:
    values = _current_job().to_dict()
    values["worktree_root_pane_id"] = None
    with pytest.raises(
        JobValidationError, match="workspace and root-pane identity must be paired"
    ):
        CursorJob.from_dict(values)


@pytest.mark.parametrize(
    ("job", "changes", "message"),
    [
        (
            _current_job(
                fork_operation_state="exists",
                fork_operation_source="owner/repo",
                fork_operation_source_url="https://example.test/owner/repo",
                fork_operation_source_default_branch="main",
                fork_operation_source_private=False,
                fork_operation_login="owner",
                fork_operation_target="owner/repo",
            ),
            {"fork_operation_state": "submitted"},
            "fork cannot transition",
        ),
        (
            _current_job(),
            {"worktree_provision_state": "dispatching"},
            "checkout cannot transition",
        ),
        (
            _current_job(
                agent_dispatch_state="ready",
                agent_provider="cursor/herdr",
                agent_provider_session_id="session",
                agent_state_sequence=7,
            ),
            {"agent_dispatch_state": "dispatching"},
            "agent session cannot transition",
        ),
    ],
)
def test_durable_operations_cannot_regress_settled_state(
    job: CursorJob,
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(JobValidationError, match=message):
        cast(Any, job).evolve(**changes)


def test_uncertain_checkout_without_root_pane_is_quarantined(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    job = store.create(
        _current_job(
            worktree_provision_state="ambiguous",
            worktree_workspace_id=None,
            worktree_root_pane_id=None,
        )
    )
    client = mock.Mock()
    client.run_json.return_value = {
        "worktrees": [
            {
                "branch": "voice/audit",
                "path": "/checkout",
                "open_workspace_id": "workspace",
            }
        ]
    }

    reconcile_uncertain_worktree(store, job, now=10, herdr_factory=lambda: client)

    current = store.get(job.id)
    assert current.worktree_provision_state == "quarantined"
    assert current.worktree_manual_inspection_required


def test_participant_callbacks_require_owner_and_retain_settled_spec(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    job = store.create(_current_job())
    owner = _owner(job)
    planned = _plan_participant_creation(
        store,
        job,
        owner,
        WorkflowParticipant.PLANNER,
        target="agent",
        label="planner",
        workspace_id="workspace",
    )
    before_submit, accepted = _participant_pane_callbacks(store, job.id, owner, "agent")
    before_submit()

    stale = WorkerOwnership(
        owner.token,
        owner.pid + 1,
        owner.boot_id,
        owner.process_start,
        owner.operation,
        owner.claimed_at,
    )
    _before, stale_accepted = _participant_pane_callbacks(store, job.id, stale, "agent")
    with pytest.raises(WorkerCancelled):
        stale_accepted("pane", "workspace")

    accepted("pane", "workspace")
    selection = AgentSelection(
        "agent",
        "pane",
        "workspace",
        "/checkout",
        "agent",
        "/checkout",
        "cursor/herdr",
        "session",
        7,
    )
    settled = _settle_worker_agent(store, planned.id, owner, selection)
    assert settled is not None
    assert settled.participant_creation_state == "created"
    assert settled.participant_creation_target == "agent"
    assert settled.participant_creation_checkout == "/checkout"
    assert settled.participant_creation_pane_id == "pane"


def test_known_participant_failure_retains_complete_spec(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job = store.create(_current_job())
    owner = _owner(job)
    _plan_participant_creation(
        store,
        job,
        owner,
        WorkflowParticipant.PLANNER,
        target="new-agent",
        label="planner",
        workspace_id="workspace",
    )

    _fence_participant_creation(
        store,
        job.id,
        owner,
        HerdrError("rejected", code="invalid_request"),
    )

    failed = store.get(job.id)
    assert failed.participant_creation_state == "failed"
    assert failed.participant_creation_target == "new-agent"
    assert failed.participant_creation_checkout == "/checkout"
