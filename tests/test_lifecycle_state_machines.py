from __future__ import annotations

from collections.abc import Callable

import pytest

from local_voice_harness.cursor.lifecycle import (
    AnnouncementAck,
    AnnouncementState,
    ClaimedDelivery,
    CleanupOwned,
    CleanupPending,
    CleanupSettled,
    Delivered,
    LifecycleTransitionError,
    MaterializedTerminalOutcome,
    PendingDelivery,
    TerminalIntent,
    abandon_cleanup_owner,
    acknowledge_delivery,
    begin_cleanup,
    claim_cleanup,
    claim_delivery,
    dismiss_announcement,
    finish_cleanup_reconciliation,
    load_cleanup_state,
    load_delivery_state,
    load_terminal_state,
    release_delivery,
    renew_delivery,
    repeat_announcement,
    settle_cleanup,
    take_over_cleanup,
)
from local_voice_harness.prompt_operations import (
    AmbiguousPrompt,
    IdlePrompt,
    PlannedPrompt,
    PromptIdentity,
    PromptOperation,
    PromptOperationError,
    SubmittedPrompt,
    SubmittingPrompt,
    accept_prompt_submission,
    begin_prompt_submission,
    legacy_prompt_fields,
    load_prompt_operation,
    mark_prompt_ambiguous,
    observe_prompt_submission,
    plan_prompt,
)
from local_voice_harness.questions import (
    PromptOperationState,
    Question,
    QuestionError,
    QuestionIdentity,
    QuestionKind,
    QuestionOrigin,
    QuestionSensitivity,
    QuestionState,
    transition_question,
)


def _pending_delivery() -> PendingDelivery:
    return PendingDelivery(
        2,
        0,
        0,
        AnnouncementState(AnnouncementAck.PENDING, False, False),
    )


def test_flat_lifecycle_adapters_restore_typed_sum_states() -> None:
    assert load_terminal_state(
        status="reconciling",
        result=None,
        error=None,
        completed_at=None,
        intent_status="cancelled",
        intent_result="cancelled",
        intent_error=None,
        intent_completed_at=2,
    ) == TerminalIntent("cancelled", "cancelled", None, 2)
    assert load_terminal_state(
        status="completed",
        result="done",
        error=None,
        completed_at=3,
        intent_status=None,
        intent_result=None,
        intent_error=None,
        intent_completed_at=None,
    ) == MaterializedTerminalOutcome("completed", "done", None, 3)
    assert isinstance(
        load_cleanup_state(
            pending=False,
            token=None,
            owner_pid=None,
            owner_boot_id=None,
            owner_start=None,
            reconciliation_pending=False,
        ),
        CleanupSettled,
    )
    assert load_delivery_state(
        delivered=False,
        generation=2,
        claim_token="claim",
        claimed_at=10,
        retry_at=0,
        attempts=1,
        delivered_at=None,
        acknowledgement="pending",
        dismissed=False,
        repeated=False,
    ) == ClaimedDelivery(
        2,
        "claim",
        10,
        0,
        1,
        AnnouncementState(AnnouncementAck.PENDING, False, False),
    )


def test_cleanup_transitions_preserve_and_fence_release_ownership() -> None:
    pending = begin_cleanup("stage")
    owned = claim_cleanup(
        pending,
        "stage",
        token="owner",
        owner_pid=42,
        owner_boot_id="boot",
        owner_start="start",
    )

    assert owned == CleanupOwned("owner", 42, "boot", "start", True)
    assert finish_cleanup_reconciliation(owned) == CleanupOwned(
        "owner", 42, "boot", "start", False
    )
    assert abandon_cleanup_owner(owned, "owner") == CleanupPending("owner", True)
    assert isinstance(settle_cleanup(owned, "owner"), CleanupSettled)
    takeover = take_over_cleanup(
        owned,
        "owner",
        token="replacement",
        owner_pid=43,
        owner_boot_id="next-boot",
        owner_start="next-start",
    )
    assert takeover == CleanupOwned("replacement", 43, "next-boot", "next-start", True)

    for transition in (
        lambda: claim_cleanup(
            pending,
            "stale",
            owner_pid=42,
            owner_boot_id="boot",
            owner_start="start",
        ),
        lambda: abandon_cleanup_owner(owned, "stale"),
        lambda: settle_cleanup(owned, "stale"),
        lambda: take_over_cleanup(
            owned,
            "stale",
            token="replacement",
            owner_pid=43,
            owner_boot_id="next-boot",
            owner_start="next-start",
        ),
        lambda: settle_cleanup(CleanupSettled(), "owner"),
    ):
        with pytest.raises(LifecycleTransitionError, match="stale|illegal"):
            transition()


def test_delivery_claim_renew_release_and_at_least_once_reclaim() -> None:
    first = claim_delivery(_pending_delivery(), "first", 100, lease_seconds=300)
    renewed = renew_delivery(first, "first", 249, lease_seconds=300)
    released = release_delivery(renewed, "first", retry_at=254)
    second = claim_delivery(released, "second", 254, lease_seconds=300)
    delivered = acknowledge_delivery(
        second,
        "second",
        255,
        AnnouncementAck.SPOKEN,
        lease_seconds=300,
    )

    assert second.attempts == 2
    assert isinstance(delivered, Delivered)
    assert delivered.announcement.acknowledgement == AnnouncementAck.SPOKEN


@pytest.mark.parametrize("operation", ["renew", "release", "acknowledge"])
@pytest.mark.parametrize("token", ["stale", "claim"])
def test_illegal_delivery_transitions_are_rejected(operation: str, token: str) -> None:
    claimed = claim_delivery(_pending_delivery(), "claim", 100, lease_seconds=300)
    state = _pending_delivery() if token == "claim" else claimed

    with pytest.raises(LifecycleTransitionError, match="stale|illegal"):
        if operation == "renew":
            renew_delivery(state, token, 101, lease_seconds=300)
        elif operation == "release":
            release_delivery(state, token, retry_at=106)
        else:
            acknowledge_delivery(
                state,
                token,
                101,
                AnnouncementAck.SPOKEN,
                lease_seconds=300,
            )


@pytest.mark.parametrize("operation", ["renew", "acknowledge"])
def test_expired_delivery_claims_are_rejected(operation: str) -> None:
    claimed = claim_delivery(_pending_delivery(), "claim", 100, lease_seconds=300)

    with pytest.raises(LifecycleTransitionError, match="expired|stale|illegal"):
        if operation == "renew":
            renew_delivery(claimed, "claim", 400, lease_seconds=300)
        else:
            acknowledge_delivery(
                claimed,
                "claim",
                400,
                AnnouncementAck.SPOKEN,
                lease_seconds=300,
            )


@pytest.mark.parametrize(
    ("acknowledgement", "delivered"),
    [
        (AnnouncementAck.SPOKEN, True),
        (AnnouncementAck.DESKTOP, False),
        (AnnouncementAck.DEFERRED, False),
        (AnnouncementAck.DISMISSED, True),
    ],
)
def test_announcement_acknowledgement_modes_remain_distinct(
    acknowledgement: AnnouncementAck, delivered: bool
) -> None:
    claimed = claim_delivery(_pending_delivery(), "claim", 100, lease_seconds=300)
    updated = acknowledge_delivery(claimed, "claim", 101, acknowledgement)

    assert isinstance(updated, Delivered) is delivered
    assert updated.announcement.acknowledgement == acknowledgement


@pytest.mark.parametrize(
    ("current", "next_acknowledgement"),
    [
        (AnnouncementAck.DESKTOP, AnnouncementAck.SPOKEN),
        (AnnouncementAck.DEFERRED, AnnouncementAck.DESKTOP),
        (AnnouncementAck.DEFERRED, AnnouncementAck.SPOKEN),
    ],
)
def test_suppressed_announcement_can_advance_to_visible_delivery(
    current: AnnouncementAck,
    next_acknowledgement: AnnouncementAck,
) -> None:
    pending = PendingDelivery(
        2,
        0,
        1,
        AnnouncementState(current, False, False),
    )
    claimed = claim_delivery(pending, "claim", 100, lease_seconds=300)

    updated = acknowledge_delivery(
        claimed,
        "claim",
        101,
        next_acknowledgement,
    )

    assert updated.announcement.acknowledgement == next_acknowledgement
    assert isinstance(updated, Delivered) is (
        next_acknowledgement == AnnouncementAck.SPOKEN
    )


def test_announcement_dismissal_flag_must_match_acknowledgement() -> None:
    with pytest.raises(LifecycleTransitionError, match="must match"):
        AnnouncementState(AnnouncementAck.PENDING, True, False)


def test_dismissal_and_repetition_are_explicit_transitions() -> None:
    initial = AnnouncementState(AnnouncementAck.PENDING, False, False)
    dismissed = dismiss_announcement(initial)
    repeated = repeat_announcement(dismissed)

    assert dismissed == AnnouncementState(AnnouncementAck.DISMISSED, True, False)
    assert repeated == AnnouncementState(AnnouncementAck.PENDING, False, True)
    with pytest.raises(LifecycleTransitionError, match="already dismissed"):
        dismiss_announcement(dismissed)


def _prompt_identity(**changes: object) -> PromptIdentity:
    values: dict[str, object] = {
        "job_id": "job-1",
        "phase": "implementing",
        "turn": 2,
        "turn_token": "job-1-2",
        "target": "agent-1",
        "agent_session": "session-1",
        "baseline_sequence": 7,
    }
    values.update(changes)
    return PromptIdentity(**values)  # type: ignore[arg-type]


def test_legal_prompt_transitions_preserve_complete_identity() -> None:
    identity = _prompt_identity()
    planned = plan_prompt(IdlePrompt(), identity)
    submitting = begin_prompt_submission(planned, identity)

    assert accept_prompt_submission(submitting, identity) == SubmittedPrompt(identity)
    assert observe_prompt_submission(
        submitting,
        identity,
        target=identity.target,
        agent_session=identity.agent_session,
        state_sequence=8,
    ) == SubmittedPrompt(identity)
    assert observe_prompt_submission(
        submitting,
        identity,
        target=identity.target,
        agent_session="replacement",
        state_sequence=8,
    ) == AmbiguousPrompt(identity)
    assert observe_prompt_submission(
        submitting,
        identity,
        target="replacement-target",
        agent_session=identity.agent_session,
        state_sequence=8,
    ) == AmbiguousPrompt(identity)
    assert mark_prompt_ambiguous(planned, identity) == AmbiguousPrompt(identity)
    assert mark_prompt_ambiguous(submitting, identity) == AmbiguousPrompt(identity)
    assert mark_prompt_ambiguous(
        SubmittedPrompt(identity), identity
    ) == AmbiguousPrompt(identity)


@pytest.mark.parametrize(
    ("transition", "operation_factory"),
    [
        (
            lambda operation, identity: plan_prompt(operation, identity),
            lambda identity: PlannedPrompt(identity),
        ),
        (begin_prompt_submission, lambda _identity: IdlePrompt()),
        (accept_prompt_submission, lambda identity: PlannedPrompt(identity)),
        (
            lambda operation, identity: observe_prompt_submission(
                operation,
                identity,
                target=identity.target,
                agent_session=identity.agent_session,
                state_sequence=8,
            ),
            lambda identity: SubmittedPrompt(identity),
        ),
        (mark_prompt_ambiguous, lambda _identity: IdlePrompt()),
        (mark_prompt_ambiguous, lambda identity: AmbiguousPrompt(identity)),
    ],
)
def test_illegal_prompt_transitions_are_rejected(
    transition: Callable[[PromptOperation, PromptIdentity], PromptOperation],
    operation_factory: Callable[[PromptIdentity], PromptOperation],
) -> None:
    identity = _prompt_identity()
    candidate = operation_factory(identity)

    with pytest.raises(PromptOperationError):
        transition(candidate, identity)


def test_prompt_transition_rejects_stale_full_identity_fence() -> None:
    identity = _prompt_identity()
    planned = PlannedPrompt(identity)

    for field, value in (
        ("job_id", "job-2"),
        ("phase", "reviewing"),
        ("turn", 3),
        ("turn_token", "job-1-3"),
        ("target", "agent-2"),
        ("agent_session", "session-2"),
        ("baseline_sequence", 8),
    ):
        with pytest.raises(PromptOperationError, match="stale"):
            begin_prompt_submission(planned, _prompt_identity(**{field: value}))


@pytest.mark.parametrize(
    "operation",
    [
        PlannedPrompt(_prompt_identity()),
        begin_prompt_submission(PlannedPrompt(_prompt_identity()), _prompt_identity()),
        SubmittedPrompt(_prompt_identity()),
        AmbiguousPrompt(_prompt_identity()),
    ],
)
def test_legacy_prompt_adapter_round_trips_typed_states(
    operation: PlannedPrompt | SubmittingPrompt | SubmittedPrompt | AmbiguousPrompt,
) -> None:
    fields = legacy_prompt_fields(operation)

    restored = load_prompt_operation(
        state=fields["prompt_operation_state"],
        job_id=operation.identity.job_id,
        phase=fields["prompt_operation_phase"],
        turn=fields["prompt_operation_turn"] or 0,
        turn_token=operation.identity.turn_token,
        target=fields["prompt_operation_target"],
        agent_session=fields["prompt_operation_agent_session"],
        baseline_sequence=fields["prompt_baseline_sequence"],
    )

    assert restored == operation


_LEGAL_QUESTION_TRANSITIONS = {
    (QuestionState.PENDING, QuestionState.DEFERRED),
    (QuestionState.PENDING, QuestionState.ANSWERED),
    (QuestionState.PENDING, QuestionState.RESOLVED),
    (QuestionState.PENDING, QuestionState.CANCELLED),
    (QuestionState.DEFERRED, QuestionState.DEFERRED),
    (QuestionState.DEFERRED, QuestionState.ANSWERED),
    (QuestionState.DEFERRED, QuestionState.RESOLVED),
    (QuestionState.DEFERRED, QuestionState.CANCELLED),
    (QuestionState.ANSWERED, QuestionState.DISPATCHING),
    (QuestionState.ANSWERED, QuestionState.RESOLVED),
    (QuestionState.ANSWERED, QuestionState.CANCELLED),
    (QuestionState.DISPATCHING, QuestionState.DISPATCHING),
    (QuestionState.DISPATCHING, QuestionState.RESOLVED),
    (QuestionState.DISPATCHING, QuestionState.CANCELLED),
}


def _question(state: QuestionState) -> Question:
    changes: dict[str, object] = {}
    if state in {QuestionState.ANSWERED, QuestionState.DISPATCHING}:
        changes = {"answer": "Use SQLite", "answered_at": 2}
        if state == QuestionState.DISPATCHING:
            changes.update(
                dispatch_token="job-1-3",
                prompt_state=PromptOperationState.PLANNED,
            )
    return Question(
        id="question-1",
        text="Which approach?",
        kind=QuestionKind.FREE_TEXT,
        sensitivity=QuestionSensitivity.ROUTINE,
        origin=QuestionOrigin("cursor", "job-1", "job-1-2"),
        owner="agent",
        state=state,
        asked_at=1,
        **changes,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("source", "target"), sorted(_LEGAL_QUESTION_TRANSITIONS, key=lambda pair: pair)
)
def test_every_legal_question_transition(
    source: QuestionState, target: QuestionState
) -> None:
    question = _question(source)
    changes: dict[str, object] = {}
    if target == QuestionState.ANSWERED:
        changes = {"answer": "Use SQLite", "answered_at": 2}
    if target == QuestionState.DISPATCHING:
        changes = {
            "dispatch_token": "job-1-3",
            "prompt_state": PromptOperationState.PLANNED,
        }

    updated = transition_question(
        question,
        target,
        QuestionIdentity("job-1", "question-1", "job-1-2"),
        **changes,
    )

    assert updated.state == target
    assert updated.origin == question.origin
    assert updated.owner == question.owner
    assert updated.sensitivity == question.sensitivity


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (source, target)
        for source in QuestionState
        for target in QuestionState
        if (source, target) not in _LEGAL_QUESTION_TRANSITIONS
    ],
)
def test_every_illegal_question_transition_is_rejected(
    source: QuestionState, target: QuestionState
) -> None:
    with pytest.raises(QuestionError, match="illegal"):
        transition_question(
            _question(source),
            target,
            QuestionIdentity("job-1", "question-1", "job-1-2"),
        )


@pytest.mark.parametrize(
    "identity",
    [
        QuestionIdentity("stale-job", "question-1", "job-1-2"),
        QuestionIdentity("job-1", "stale-question", "job-1-2"),
        QuestionIdentity("job-1", "question-1", "stale-turn"),
    ],
)
def test_question_transition_rejects_stale_identity(identity: QuestionIdentity) -> None:
    with pytest.raises(QuestionError, match="stale"):
        transition_question(
            _question(QuestionState.PENDING), QuestionState.ANSWERED, identity
        )


def test_question_transition_cannot_replace_identity_or_policy() -> None:
    question = _question(QuestionState.PENDING)
    identity = QuestionIdentity("job-1", "question-1", "job-1-2")

    for changes in (
        {"id": "replacement"},
        {"origin": QuestionOrigin("cursor", "job-2", "job-2-1")},
        {"owner": "replacement"},
        {"sensitivity": QuestionSensitivity.SECURITY},
    ):
        with pytest.raises(QuestionError, match="cannot change"):
            transition_question(
                question,
                QuestionState.ANSWERED,
                identity,
                **changes,
            )


def test_answered_question_state_requires_answer_payload() -> None:
    with pytest.raises(QuestionError, match="requires an answer"):
        Question(
            id="question-1",
            text="Which approach?",
            kind=QuestionKind.FREE_TEXT,
            sensitivity=QuestionSensitivity.ROUTINE,
            origin=QuestionOrigin("cursor", "job-1", "job-1-2"),
            owner="agent",
            state=QuestionState.ANSWERED,
            asked_at=1,
        )
