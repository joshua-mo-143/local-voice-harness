from __future__ import annotations

import pytest

from local_voice_harness.cursor.operations import (
    AgentSessionOperation,
    AgentSessionSpec,
    CheckoutOperation,
    CheckoutSpec,
    ForkOperation,
    ForkSpec,
    OperationState,
    OperationTransitionError,
    ParticipantPaneOperation,
    ParticipantPaneSpec,
    SessionIdentity,
    WorkerOwnership,
    load_worker_ownership,
)


@pytest.mark.parametrize(
    ("operation", "next_state"),
    [
        (
            CheckoutOperation(
                OperationState.PLANNED,
                CheckoutSpec("/repo", "voice/task", "/worktree"),
            ),
            OperationState.ACTIVE,
        ),
        (
            ForkOperation(
                OperationState.ACTIVE,
                ForkSpec(
                    "owner/repo",
                    "https://github.com/owner/repo",
                    "main",
                    False,
                    "me",
                    "me/repo",
                ),
            ),
            OperationState.UNKNOWN,
        ),
        (
            ParticipantPaneOperation(
                OperationState.UNKNOWN,
                ParticipantPaneSpec(
                    "reviewer",
                    "reviewer-target",
                    "task-reviewer",
                    "/worktree",
                    "workspace",
                ),
            ),
            OperationState.MANUAL,
        ),
    ],
)
def test_operation_state_machines_accept_legal_transitions(
    operation: CheckoutOperation | ForkOperation | ParticipantPaneOperation,
    next_state: OperationState,
) -> None:
    assert operation.transition(next_state).state == next_state


@pytest.mark.parametrize(
    ("operation", "next_state"),
    [
        (
            CheckoutOperation(
                OperationState.PLANNED,
                CheckoutSpec("/repo", "voice/task", "/worktree"),
            ),
            OperationState.MANUAL,
        ),
        (
            ForkOperation(
                OperationState.SETTLED,
                ForkSpec(
                    "owner/repo",
                    "https://github.com/owner/repo",
                    "main",
                    False,
                    "me",
                    "me/repo",
                ),
            ),
            OperationState.ACTIVE,
        ),
        (
            ParticipantPaneOperation(
                OperationState.FAILED,
                ParticipantPaneSpec(
                    "reviewer",
                    "reviewer-target",
                    "task-reviewer",
                    "/worktree",
                    "workspace",
                ),
            ),
            OperationState.SETTLED,
        ),
    ],
)
def test_operation_state_machines_reject_illegal_transitions(
    operation: CheckoutOperation | ForkOperation | ParticipantPaneOperation,
    next_state: OperationState,
) -> None:
    with pytest.raises(OperationTransitionError):
        operation.transition(next_state)


@pytest.mark.parametrize(
    "missing",
    ["token", "pid", "boot_id", "process_start", "operation", "claimed_at"],
)
def test_worker_ownership_is_unclaimed_or_complete(missing: str) -> None:
    values: dict[str, object] = {
        "token": "claim",
        "pid": 42,
        "boot_id": "boot",
        "process_start": "start",
        "operation": "routing",
        "claimed_at": 10.0,
    }
    values[missing] = None
    with pytest.raises(OperationTransitionError, match="incomplete"):
        load_worker_ownership(
            token=values["token"] if isinstance(values["token"], str) else None,
            pid=values["pid"] if isinstance(values["pid"], int) else None,
            boot_id=(values["boot_id"] if isinstance(values["boot_id"], str) else None),
            process_start=(
                values["process_start"]
                if isinstance(values["process_start"], str)
                else None
            ),
            operation=(
                values["operation"] if isinstance(values["operation"], str) else None
            ),
            claimed_at=(
                values["claimed_at"]
                if isinstance(values["claimed_at"], float)
                else None
            ),
        )


def test_worker_ownership_rejects_stale_callback_identity() -> None:
    owner = WorkerOwnership("claim", 42, "boot", "start", "routing", 10)
    replacement = WorkerOwnership("claim", 43, "boot", "other", "routing", 10)
    assert owner.matches(owner)
    assert not owner.matches(replacement)


@pytest.mark.parametrize(
    ("provider", "session_id", "target", "sequence", "accepted"),
    [
        ("cursor/herdr", "session", "agent", 7, True),
        ("cursor/herdr", "session", "agent", 8, True),
        ("cursor/herdr", "session", "agent", 6, False),
        ("cursor/herdr", "replacement", "agent", 8, False),
        ("cursor/herdr", "session", "other", 8, False),
        ("other", "session", "agent", 8, False),
    ],
)
def test_agent_reconciliation_uses_complete_identity_and_sequence_fence(
    provider: str,
    session_id: str,
    target: str,
    sequence: int,
    accepted: bool,
) -> None:
    identity = SessionIdentity("cursor/herdr", "session", "agent", 7)
    operation = AgentSessionOperation(
        OperationState.SETTLED,
        AgentSessionSpec("agent", "/worktree", "workspace", "pane"),
        identity,
    )
    observed = SessionIdentity(provider, session_id, target, sequence)
    assert operation.accepts_observation(observed) is accepted


def test_operation_specs_survive_uncertain_and_manual_states() -> None:
    spec = CheckoutSpec("/repo", "voice/task", "/worktree")
    active = CheckoutOperation(OperationState.ACTIVE, spec)
    unknown = active.transition(OperationState.UNKNOWN)
    manual = unknown.transition(OperationState.MANUAL)
    assert unknown.spec == spec
    assert manual.spec == spec
