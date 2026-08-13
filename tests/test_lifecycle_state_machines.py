from __future__ import annotations

from collections.abc import Callable

import pytest

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
