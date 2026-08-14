from __future__ import annotations

from functools import partial

import pytest

from local_voice_harness.cursor.workflow import (
    ArtifactReference,
    ParticipantAdmissionState,
    ParticipantCreation,
    ParticipantCreationState,
    ParticipantLifecycle,
    PlanApproval,
    PlanApprovalProof,
    PlanApprovalSource,
    ReviewDecision,
    ReviewState,
    WorkflowClassification,
    WorkflowParticipant,
    WorkflowPhase,
    WorkflowState,
    WorkflowTier,
    WorkflowTransitionError,
)


@pytest.mark.parametrize(
    ("source", "target", "legal"),
    [
        (source, target, target in targets or source == target)
        for source, targets in {
            WorkflowPhase.CLASSIFYING: {
                WorkflowPhase.PLANNING,
                WorkflowPhase.IMPLEMENTING,
            },
            WorkflowPhase.PLANNING: {WorkflowPhase.REVIEWING},
            WorkflowPhase.REVIEWING: {
                WorkflowPhase.REVISING,
                WorkflowPhase.IMPLEMENTING,
            },
            WorkflowPhase.REVISING: {WorkflowPhase.REVIEWING},
            WorkflowPhase.IMPLEMENTING: {
                WorkflowPhase.PLANNING,
                WorkflowPhase.FINISHED,
            },
            WorkflowPhase.FINISHED: set(),
        }.items()
        for target in WorkflowPhase
    ],
)
def test_workflow_phase_edges_are_explicit(
    source: WorkflowPhase,
    target: WorkflowPhase,
    legal: bool,
) -> None:
    classification = (
        None
        if source == WorkflowPhase.CLASSIFYING
        else WorkflowClassification(WorkflowTier.HIGH_RISK, "risk")
    )
    state = WorkflowState(source, classification)

    if legal and not (source == WorkflowPhase.CLASSIFYING and target != source):
        assert state.transition(target).phase == target
    elif source == WorkflowPhase.CLASSIFYING and target != source:
        with pytest.raises(
            WorkflowTransitionError,
            match="illegal|requires workflow_tier",
        ):
            state.transition(target)
    else:
        with pytest.raises(WorkflowTransitionError, match="illegal"):
            state.transition(target)


@pytest.mark.parametrize(
    ("source", "target", "legal"),
    [
        (WorkflowTier.SIMPLE, WorkflowTier.MEDIUM, True),
        (WorkflowTier.SIMPLE, WorkflowTier.HIGH_RISK, True),
        (WorkflowTier.MEDIUM, WorkflowTier.HIGH_RISK, True),
        (WorkflowTier.HIGH_RISK, WorkflowTier.MEDIUM, False),
        (WorkflowTier.MEDIUM, WorkflowTier.SIMPLE, False),
    ],
)
def test_workflow_tier_promotions_never_downgrade(
    source: WorkflowTier,
    target: WorkflowTier,
    legal: bool,
) -> None:
    classification = WorkflowClassification(source, "initial")
    if legal:
        assert classification.promote(target, "new evidence").tier == target
    else:
        with pytest.raises(WorkflowTransitionError, match="cannot be downgraded"):
            classification.promote(target, "less risk")


@pytest.mark.parametrize(
    ("tier", "phase"),
    [
        (WorkflowTier.SIMPLE, WorkflowPhase.IMPLEMENTING),
        (WorkflowTier.MEDIUM, WorkflowPhase.PLANNING),
        (WorkflowTier.HIGH_RISK, WorkflowPhase.PLANNING),
    ],
)
def test_classification_carries_tier_and_reason(
    tier: WorkflowTier,
    phase: WorkflowPhase,
) -> None:
    classified = WorkflowState(WorkflowPhase.CLASSIFYING).classify(
        tier,
        "classification evidence",
    )
    assert classified.phase == phase
    assert classified.classification == WorkflowClassification(
        tier,
        "classification evidence",
    )


def _reference(kind: str, round_number: int) -> ArtifactReference:
    value = f".artifacts/aaaaaaaaaaaa/{kind}-{round_number}.json"
    return ArtifactReference.parse(value, job_id="aaaaaaaaaaaa", kind=kind)


@pytest.mark.parametrize(
    ("tier", "round_number", "can_revise"),
    [
        (WorkflowTier.MEDIUM, 0, True),
        (WorkflowTier.MEDIUM, 1, False),
        (WorkflowTier.HIGH_RISK, 0, True),
        (WorkflowTier.HIGH_RISK, 1, True),
        (WorkflowTier.HIGH_RISK, 2, False),
    ],
)
def test_review_limits_and_artifact_rounds(
    tier: WorkflowTier,
    round_number: int,
    can_revise: bool,
) -> None:
    review = (
        ReviewState(tier, round_number)
        .publish_plan(_reference("plan", round_number))
        .publish_review(
            _reference("review", round_number),
            ReviewDecision.REVISE,
        )
    )
    if can_revise:
        assert review.revise().round == round_number + 1
    else:
        with pytest.raises(WorkflowTransitionError, match="limit is exhausted"):
            review.revise()

    with pytest.raises(WorkflowTransitionError, match="current review round"):
        ReviewState(tier, round_number).publish_plan(
            _reference("plan", (round_number + 1) % 3)
        )


def test_plan_approval_requires_current_question_plan_and_revision_proof() -> None:
    proof = PlanApprovalProof("gate", "session", 7, 3)
    boundary = PlanApproval().record_boundary(proof)
    awaiting = boundary.await_question(
        question_id="question",
        question_turn_token="turn",
        plan_reference=".artifacts/aaaaaaaaaaaa/plan-0.json",
        review_reference=".artifacts/aaaaaaaaaaaa/review-0.json",
        review_accepted=True,
    )

    with pytest.raises(WorkflowTransitionError, match="identity is stale"):
        awaiting.approve(
            PlanApprovalSource.EXPLICIT,
            plan_reference=".artifacts/aaaaaaaaaaaa/plan-0.json",
            review_reference=".artifacts/aaaaaaaaaaaa/review-0.json",
            review_accepted=True,
            question_id="replacement",
            question_turn_token="turn",
        )
    approved = awaiting.approve(
        PlanApprovalSource.EXPLICIT,
        plan_reference=".artifacts/aaaaaaaaaaaa/plan-0.json",
        review_reference=".artifacts/aaaaaaaaaaaa/review-0.json",
        review_accepted=True,
        question_id="question",
        question_turn_token="turn",
    )
    assert approved.count().observe().counted

    with pytest.raises(WorkflowTransitionError, match="revision"):
        PlanApprovalProof("gate", "session", 7, None)  # type: ignore[arg-type]
    with pytest.raises(WorkflowTransitionError, match="revision"):
        PlanApprovalProof("gate", "session", -1, 0)
    with pytest.raises(WorkflowTransitionError, match="cannot be rewritten"):
        boundary.record_boundary(PlanApprovalProof("replacement", "session", 7, 3))


def test_user_review_approval_requires_exhausted_high_risk_round() -> None:
    revising = (
        ReviewState(WorkflowTier.HIGH_RISK, 1)
        .publish_plan(_reference("plan", 1))
        .publish_review(_reference("review", 1), ReviewDecision.REVISE)
    )
    with pytest.raises(WorkflowTransitionError, match="exhausted"):
        revising.approve_exhausted()

    exhausted = (
        ReviewState(WorkflowTier.HIGH_RISK, 2)
        .publish_plan(_reference("plan", 2))
        .publish_review(_reference("review", 2), ReviewDecision.REVISE)
    )
    assert exhausted.approve_exhausted().approved


@pytest.mark.parametrize(
    ("source", "action", "legal"),
    [
        (ParticipantAdmissionState.WAITING, "admit", True),
        (ParticipantAdmissionState.HELD, "admit", False),
        (ParticipantAdmissionState.WAITING, "release", True),
        (ParticipantAdmissionState.HELD, "release", False),
        (ParticipantAdmissionState.WAITING, "yield", False),
        (ParticipantAdmissionState.HELD, "yield", True),
        (ParticipantAdmissionState.RELEASED, "yield", False),
    ],
)
def test_participant_admission_preserves_capacity_fence(
    source: ParticipantAdmissionState,
    action: str,
    legal: bool,
) -> None:
    lifecycle = ParticipantLifecycle(source)
    if action == "admit":
        transition = lifecycle.admit
    elif action == "yield":
        transition = lifecycle.yield_capacity
    else:
        transition = partial(lifecycle.release, cleanup_confirmed=False)
    if legal:
        transition()
    else:
        with pytest.raises(WorkflowTransitionError):
            transition()


def test_participant_creation_retains_identity_and_rejects_stale_workspace() -> None:
    planned = ParticipantCreation(
        ParticipantCreationState.PLANNED,
        WorkflowParticipant.REVIEWER,
        "reviewer-target",
        "task-reviewer",
        "workspace",
    )
    submitting = planned.begin()
    manual = submitting.require_manual()
    assert manual.target == "reviewer-target"
    assert manual.workspace_id == "workspace"

    with pytest.raises(WorkflowTransitionError, match="workspace identity is stale"):
        manual.created(pane_id="pane", workspace_id="replacement")
    with pytest.raises(WorkflowTransitionError, match="identity cannot be rewritten"):
        submitting.validate_transition(
            ParticipantCreation(
                ParticipantCreationState.SUBMITTING,
                WorkflowParticipant.REVIEWER,
                "replacement-target",
                "task-reviewer",
                "workspace",
            )
        )
    created = manual.created(pane_id="pane", workspace_id="workspace")
    assert created.pane_id == "pane"
