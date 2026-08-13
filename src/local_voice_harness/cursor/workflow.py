from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import StrEnum


class WorkflowTransitionError(ValueError):
    """A workflow value or transition is incomplete or illegal."""


class WorkflowTier(StrEnum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    HIGH_RISK = "high-risk"


class WorkflowPhase(StrEnum):
    CLASSIFYING = "classifying"
    PLANNING = "planning"
    REVIEWING = "reviewing"
    REVISING = "revising"
    IMPLEMENTING = "implementing"
    FINISHED = "finished"


class WorkflowParticipant(StrEnum):
    PLANNER = "planner"
    REVIEWER = "reviewer"
    IMPLEMENTER = "implementer"


class ReviewDecision(StrEnum):
    APPROVE = "approve"
    REVISE = "revise"


class ReviewApprovalSource(StrEnum):
    REVIEWER = "reviewer"
    USER = "user"


class PlanApprovalState(StrEnum):
    NONE = "none"
    BOUNDARY = "boundary"
    AWAITING = "awaiting"
    APPROVED = "approved"
    OBSERVED = "observed"
    REJECTED = "rejected"


class PlanApprovalSource(StrEnum):
    EXPLICIT = "explicit"
    AUTO = "auto"
    LEGACY = "legacy"


class ParticipantAdmissionState(StrEnum):
    WAITING = "waiting"
    HELD = "held"
    RELEASED = "released"


class ParticipantCreationState(StrEnum):
    NONE = "none"
    PLANNED = "planned"
    SUBMITTING = "submitting"
    CREATED = "created"
    AMBIGUOUS = "ambiguous"
    MANUAL_REQUIRED = "manual_required"


_TIER_ORDER = {
    WorkflowTier.SIMPLE: 0,
    WorkflowTier.MEDIUM: 1,
    WorkflowTier.HIGH_RISK: 2,
}
_LEGAL_PHASE_EDGES: dict[WorkflowPhase, frozenset[WorkflowPhase]] = {
    WorkflowPhase.CLASSIFYING: frozenset(
        {WorkflowPhase.PLANNING, WorkflowPhase.IMPLEMENTING}
    ),
    WorkflowPhase.PLANNING: frozenset({WorkflowPhase.REVIEWING}),
    WorkflowPhase.REVIEWING: frozenset(
        {WorkflowPhase.REVISING, WorkflowPhase.IMPLEMENTING}
    ),
    WorkflowPhase.REVISING: frozenset({WorkflowPhase.REVIEWING}),
    WorkflowPhase.IMPLEMENTING: frozenset(
        {WorkflowPhase.PLANNING, WorkflowPhase.FINISHED}
    ),
    WorkflowPhase.FINISHED: frozenset(),
}
_ARTIFACT_REFERENCE = re.compile(
    r"^\.artifacts/(?P<job>[0-9a-f]{12})/"
    r"(?P<kind>plan|review)-(?P<round>[0-2])"
    r"(?:-(?P<digest>[0-9a-f]{64}))?\.json$"
)


@dataclass(frozen=True, slots=True)
class WorkflowClassification:
    tier: WorkflowTier
    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise WorkflowTransitionError(
                "classified workflow requires workflow_classification_reason"
            )

    def promote(self, tier: WorkflowTier, reason: str) -> WorkflowClassification:
        promoted = WorkflowClassification(tier, reason)
        if _TIER_ORDER[promoted.tier] < _TIER_ORDER[self.tier]:
            raise WorkflowTransitionError("Cursor workflow tier cannot be downgraded")
        if _TIER_ORDER[promoted.tier] == _TIER_ORDER[self.tier]:
            raise WorkflowTransitionError(
                "Cursor workflow promotion must increase risk"
            )
        return promoted


@dataclass(frozen=True, slots=True)
class WorkflowState:
    phase: WorkflowPhase
    classification: WorkflowClassification | None = None

    def __post_init__(self) -> None:
        if self.phase != WorkflowPhase.CLASSIFYING and self.classification is None:
            raise WorkflowTransitionError(
                f"{self.phase.value} workflow requires workflow_tier"
            )
        if (
            self.classification is not None
            and self.classification.tier == WorkflowTier.SIMPLE
            and self.phase
            in {
                WorkflowPhase.PLANNING,
                WorkflowPhase.REVIEWING,
                WorkflowPhase.REVISING,
            }
        ):
            raise WorkflowTransitionError(
                "simple workflow cannot enter planning or review"
            )

    def classify(self, tier: WorkflowTier, reason: str) -> WorkflowState:
        if self.phase != WorkflowPhase.CLASSIFYING:
            raise WorkflowTransitionError("workflow is already classified")
        classification = WorkflowClassification(tier, reason)
        phase = (
            WorkflowPhase.IMPLEMENTING
            if tier == WorkflowTier.SIMPLE
            else WorkflowPhase.PLANNING
        )
        return WorkflowState(phase, classification)

    def promote(self, tier: WorkflowTier, reason: str) -> WorkflowState:
        if self.phase != WorkflowPhase.IMPLEMENTING or self.classification is None:
            raise WorkflowTransitionError(
                "workflow promotion requires implementation phase"
            )
        return WorkflowState(
            WorkflowPhase.PLANNING,
            self.classification.promote(tier, reason),
        )

    def transition(self, phase: WorkflowPhase) -> WorkflowState:
        if phase == self.phase:
            return self
        if phase not in _LEGAL_PHASE_EDGES[self.phase]:
            raise WorkflowTransitionError(
                "illegal Cursor workflow transition "
                f"{self.phase.value} -> {phase.value}"
            )
        return WorkflowState(phase, self.classification)


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    value: str
    job_id: str
    kind: str
    round: int

    def __post_init__(self) -> None:
        match = _ARTIFACT_REFERENCE.fullmatch(self.value)
        if (
            match is None
            or match.group("job") != self.job_id
            or match.group("kind") != self.kind
            or int(match.group("round")) != self.round
        ):
            raise WorkflowTransitionError(
                f"{self.kind}_artifact has invalid workflow artifact reference"
            )

    @classmethod
    def parse(cls, value: str, *, job_id: str, kind: str) -> ArtifactReference:
        match = _ARTIFACT_REFERENCE.fullmatch(value)
        if match is None or match.group("job") != job_id or match.group("kind") != kind:
            raise WorkflowTransitionError(
                f"{kind}_artifact has invalid workflow artifact reference"
            )
        return cls(value, job_id, kind, int(match.group("round")))


@dataclass(frozen=True, slots=True)
class ReviewState:
    tier: WorkflowTier | None
    round: int
    plan: ArtifactReference | None = None
    review: ArtifactReference | None = None
    decision: ReviewDecision | None = None
    approval_source: ReviewApprovalSource | None = None

    def __post_init__(self) -> None:
        if self.round < 0 or self.round > 2:
            raise WorkflowTransitionError("review_round must be between zero and two")
        if self.tier == WorkflowTier.MEDIUM and self.round > 1:
            raise WorkflowTransitionError("medium workflow allows only one review")
        if self.approval_source is not None and self.review is None:
            raise WorkflowTransitionError("approved review requires a review artifact")
        if self.approval_source == ReviewApprovalSource.REVIEWER:
            if self.decision != ReviewDecision.APPROVE:
                raise WorkflowTransitionError(
                    "reviewer approval requires an approving review decision"
                )
        if self.approval_source == ReviewApprovalSource.USER and (
            self.tier != WorkflowTier.HIGH_RISK
            or self.decision != ReviewDecision.REVISE
            or self.round < 2
        ):
            raise WorkflowTransitionError(
                "user approval requires an exhausted high-risk rejected review"
            )
        if self.approval_source is not None and (
            self.plan is None
            or self.plan.round != self.round
            or self.review is None
            or self.review.round != self.round
            or self.plan.job_id != self.review.job_id
        ):
            raise WorkflowTransitionError(
                "approved review artifacts must match the current review round"
            )

    @property
    def approved(self) -> bool:
        return self.approval_source is not None

    def publish_plan(self, reference: ArtifactReference) -> ReviewState:
        if reference.kind != "plan" or reference.round != self.round:
            raise WorkflowTransitionError(
                "plan artifact must match the current review round"
            )
        return replace(
            self,
            plan=reference,
            review=None,
            decision=None,
            approval_source=None,
        )

    def publish_review(
        self, reference: ArtifactReference, decision: ReviewDecision
    ) -> ReviewState:
        if self.plan is None or self.plan.round != self.round:
            raise WorkflowTransitionError("review requires the current plan artifact")
        if reference.kind != "review" or reference.round != self.round:
            raise WorkflowTransitionError(
                "review artifact must match the current review round"
            )
        source = (
            ReviewApprovalSource.REVIEWER
            if decision == ReviewDecision.APPROVE
            else None
        )
        return replace(
            self,
            review=reference,
            decision=decision,
            approval_source=source,
        )

    def approve_exhausted(self) -> ReviewState:
        return replace(self, approval_source=ReviewApprovalSource.USER)

    def revise(self) -> ReviewState:
        if self.decision != ReviewDecision.REVISE:
            raise WorkflowTransitionError("revision requires a revise decision")
        next_round = self.round + 1
        if next_round > 2 or (self.tier == WorkflowTier.MEDIUM and next_round > 1):
            raise WorkflowTransitionError("workflow review limit is exhausted")
        return replace(
            self,
            round=next_round,
            approval_source=None,
        )


@dataclass(frozen=True, slots=True)
class PlanApprovalProof:
    gate_id: str
    agent_session: str
    state_change_sequence: int
    revision: int

    def __post_init__(self) -> None:
        if (
            not self.gate_id
            or not self.agent_session
            or isinstance(self.state_change_sequence, bool)
            or not isinstance(self.state_change_sequence, int)
            or isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.state_change_sequence < 0
            or self.revision < 0
        ):
            raise WorkflowTransitionError(
                "active plan approval requires gate ID, agent session, and sequence; "
                "revision proof is also required"
            )


@dataclass(frozen=True, slots=True)
class LegacyPlanApprovalProof:
    """Sentinel proof for imported approvals whose Herdr evidence was not recorded."""

    gate_id: str
    agent_session: str
    state_change_sequence: int = -1
    revision: int = -1

    def __post_init__(self) -> None:
        if (
            not self.gate_id
            or not self.agent_session
            or self.state_change_sequence != -1
            or self.revision != -1
        ):
            raise WorkflowTransitionError("legacy plan approval proof is invalid")


ApprovalProof = PlanApprovalProof | LegacyPlanApprovalProof


@dataclass(frozen=True, slots=True)
class PlanApproval:
    state: PlanApprovalState = PlanApprovalState.NONE
    proof: ApprovalProof | None = None
    source: PlanApprovalSource | None = None
    counted: bool = False
    plan_reference: str | None = None
    review_reference: str | None = None
    review_accepted: bool = False
    question_id: str | None = None
    question_turn_token: str | None = None

    def __post_init__(self) -> None:
        active = self.state != PlanApprovalState.NONE
        if active != (self.proof is not None):
            raise WorkflowTransitionError(
                "active plan approval requires gate ID, agent session, and sequence; "
                "revision proof is also required"
            )
        if self.state == PlanApprovalState.NONE and self.source is not None:
            raise WorkflowTransitionError(
                "inactive plan approval cannot retain gate proof or source"
            )
        if isinstance(self.proof, LegacyPlanApprovalProof) and (
            self.source != PlanApprovalSource.LEGACY
        ):
            raise WorkflowTransitionError(
                "legacy plan approval proof requires legacy source"
            )
        if self.source == PlanApprovalSource.LEGACY and not isinstance(
            self.proof, LegacyPlanApprovalProof
        ):
            raise WorkflowTransitionError(
                "legacy plan approval source requires legacy proof"
            )
        if self.state == PlanApprovalState.BOUNDARY and self.source is not None:
            raise WorkflowTransitionError(
                "plan approval boundary cannot retain a source"
            )
        if self.state == PlanApprovalState.AWAITING and (
            self.source is not None
            or not self.question_id
            or not self.question_turn_token
            or not self.plan_reference
            or not self.review_reference
            or not self.review_accepted
        ):
            raise WorkflowTransitionError(
                "awaiting plan approval requires current plan and question proof"
            )
        if self.state in {
            PlanApprovalState.APPROVED,
            PlanApprovalState.OBSERVED,
        } and (
            self.source is None
            or not self.plan_reference
            or not self.review_reference
            or not self.review_accepted
        ):
            raise WorkflowTransitionError(
                "approved plan requires a current accepted plan"
            )
        if self.counted and (
            self.source != PlanApprovalSource.EXPLICIT
            or self.state
            not in {PlanApprovalState.APPROVED, PlanApprovalState.OBSERVED}
            or not self.plan_reference
            or not self.review_reference
            or not self.review_accepted
        ):
            raise WorkflowTransitionError(
                "counted plan approval requires an accepted explicit approval"
            )

    def record_boundary(self, proof: PlanApprovalProof) -> PlanApproval:
        if self.state not in {PlanApprovalState.NONE, PlanApprovalState.BOUNDARY}:
            raise WorkflowTransitionError(
                "plan approval boundary is no longer available"
            )
        if self.proof is not None and self.proof != proof:
            raise WorkflowTransitionError("plan approval proof cannot be rewritten")
        return PlanApproval(PlanApprovalState.BOUNDARY, proof)

    def await_question(
        self,
        *,
        question_id: str,
        question_turn_token: str,
        plan_reference: str,
        review_reference: str,
        review_accepted: bool,
    ) -> PlanApproval:
        if self.state != PlanApprovalState.BOUNDARY:
            raise WorkflowTransitionError(
                "plan approval question requires a current boundary"
            )
        return PlanApproval(
            PlanApprovalState.AWAITING,
            self.proof,
            plan_reference=plan_reference,
            review_reference=review_reference,
            review_accepted=review_accepted,
            question_id=question_id,
            question_turn_token=question_turn_token,
        )

    def approve(
        self,
        source: PlanApprovalSource,
        *,
        plan_reference: str,
        review_reference: str,
        review_accepted: bool,
        question_id: str | None = None,
        question_turn_token: str | None = None,
    ) -> PlanApproval:
        if self.state == PlanApprovalState.AWAITING and (
            question_id != self.question_id
            or question_turn_token != self.question_turn_token
        ):
            raise WorkflowTransitionError("plan approval question identity is stale")
        if self.state not in {
            PlanApprovalState.BOUNDARY,
            PlanApprovalState.AWAITING,
        }:
            raise WorkflowTransitionError(
                "plan approval boundary is no longer available"
            )
        if (
            not plan_reference
            or (
                self.plan_reference is not None
                and plan_reference != self.plan_reference
            )
            or (
                self.review_reference is not None
                and review_reference != self.review_reference
            )
            or not review_accepted
        ):
            raise WorkflowTransitionError(
                "plan approval requires the current accepted plan"
            )
        return PlanApproval(
            PlanApprovalState.APPROVED,
            self.proof,
            source,
            plan_reference=plan_reference,
            review_reference=review_reference,
            review_accepted=True,
        )

    def observe(self) -> PlanApproval:
        if self.state != PlanApprovalState.APPROVED:
            raise WorkflowTransitionError("only an approved plan can be observed")
        return replace(self, state=PlanApprovalState.OBSERVED)

    def count(self) -> PlanApproval:
        if self.source != PlanApprovalSource.EXPLICIT:
            raise WorkflowTransitionError("only explicit plan approval can be counted")
        return replace(self, counted=True)

    def validate_transition(self, after: PlanApproval) -> None:
        legal = {
            PlanApprovalState.NONE: {
                PlanApprovalState.NONE,
                PlanApprovalState.BOUNDARY,
            },
            PlanApprovalState.BOUNDARY: {
                PlanApprovalState.BOUNDARY,
                PlanApprovalState.AWAITING,
                PlanApprovalState.APPROVED,
                PlanApprovalState.NONE,
                PlanApprovalState.REJECTED,
            },
            PlanApprovalState.AWAITING: {
                PlanApprovalState.AWAITING,
                PlanApprovalState.APPROVED,
                PlanApprovalState.NONE,
                PlanApprovalState.REJECTED,
            },
            PlanApprovalState.APPROVED: {
                PlanApprovalState.APPROVED,
                PlanApprovalState.OBSERVED,
                PlanApprovalState.REJECTED,
            },
            PlanApprovalState.OBSERVED: {
                PlanApprovalState.OBSERVED,
                PlanApprovalState.NONE,
                PlanApprovalState.REJECTED,
            },
            PlanApprovalState.REJECTED: {PlanApprovalState.REJECTED},
        }
        if after.state not in legal[self.state]:
            raise WorkflowTransitionError(
                f"illegal plan approval transition {self.state.value} -> "
                f"{after.state.value}"
            )
        if (
            self.proof is not None
            and after.proof is not None
            and self.proof != after.proof
        ):
            raise WorkflowTransitionError("plan approval proof cannot be rewritten")
        if self.counted and not after.counted:
            raise WorkflowTransitionError("counted plan approval cannot be uncounted")
        if self.state not in {
            PlanApprovalState.NONE,
            PlanApprovalState.BOUNDARY,
        } and (
            self.plan_reference != after.plan_reference
            or self.review_reference != after.review_reference
        ):
            raise WorkflowTransitionError(
                "accepted plan approval artifacts cannot be replaced"
            )


@dataclass(frozen=True, slots=True)
class ParticipantCreation:
    state: ParticipantCreationState = ParticipantCreationState.NONE
    participant: WorkflowParticipant | None = None
    target: str | None = None
    label: str | None = None
    workspace_id: str | None = None
    pane_id: str | None = None

    def __post_init__(self) -> None:
        identity = (
            self.participant,
            self.target,
            self.label,
            self.workspace_id,
        )
        if self.state == ParticipantCreationState.NONE:
            if any(value is not None for value in (*identity, self.pane_id)):
                raise WorkflowTransitionError(
                    "inactive participant creation cannot retain identity"
                )
            return
        if not all(identity):
            raise WorkflowTransitionError(
                "participant creation requires role, target, label, and workspace"
            )
        if self.state == ParticipantCreationState.CREATED and not self.pane_id:
            raise WorkflowTransitionError(
                "created participant requires pane and workspace"
            )

    def begin(self) -> ParticipantCreation:
        if self.state != ParticipantCreationState.PLANNED:
            raise WorkflowTransitionError(
                "participant submission requires planned creation"
            )
        return replace(self, state=ParticipantCreationState.SUBMITTING)

    def created(self, *, pane_id: str, workspace_id: str) -> ParticipantCreation:
        if self.state not in {
            ParticipantCreationState.SUBMITTING,
            ParticipantCreationState.MANUAL_REQUIRED,
        }:
            raise WorkflowTransitionError(
                "participant acceptance requires submitted creation"
            )
        if workspace_id != self.workspace_id:
            raise WorkflowTransitionError("participant workspace identity is stale")
        return replace(
            self,
            state=ParticipantCreationState.CREATED,
            pane_id=pane_id,
        )

    def require_manual(self) -> ParticipantCreation:
        if self.state not in {
            ParticipantCreationState.PLANNED,
            ParticipantCreationState.SUBMITTING,
            ParticipantCreationState.AMBIGUOUS,
        }:
            raise WorkflowTransitionError(
                "manual reconciliation requires uncertain participant creation"
            )
        return replace(self, state=ParticipantCreationState.MANUAL_REQUIRED)

    def validate_transition(self, after: ParticipantCreation) -> None:
        legal = {
            ParticipantCreationState.NONE: {
                ParticipantCreationState.NONE,
                ParticipantCreationState.PLANNED,
            },
            ParticipantCreationState.PLANNED: {
                ParticipantCreationState.PLANNED,
                ParticipantCreationState.SUBMITTING,
                ParticipantCreationState.NONE,
                ParticipantCreationState.MANUAL_REQUIRED,
            },
            ParticipantCreationState.SUBMITTING: {
                ParticipantCreationState.SUBMITTING,
                ParticipantCreationState.CREATED,
                ParticipantCreationState.AMBIGUOUS,
                ParticipantCreationState.MANUAL_REQUIRED,
            },
            ParticipantCreationState.AMBIGUOUS: {
                ParticipantCreationState.AMBIGUOUS,
                ParticipantCreationState.MANUAL_REQUIRED,
            },
            ParticipantCreationState.MANUAL_REQUIRED: {
                ParticipantCreationState.MANUAL_REQUIRED,
                ParticipantCreationState.CREATED,
                ParticipantCreationState.NONE,
            },
            ParticipantCreationState.CREATED: {
                ParticipantCreationState.CREATED,
                ParticipantCreationState.NONE,
            },
        }
        if after.state not in legal[self.state]:
            raise WorkflowTransitionError(
                f"illegal participant creation transition {self.state.value} -> "
                f"{after.state.value}"
            )
        if (
            self.state != ParticipantCreationState.NONE
            and after.state != ParticipantCreationState.NONE
            and (
                self.participant,
                self.target,
                self.label,
                self.workspace_id,
            )
            != (
                after.participant,
                after.target,
                after.label,
                after.workspace_id,
            )
        ):
            raise WorkflowTransitionError(
                "participant creation identity cannot be rewritten"
            )


@dataclass(frozen=True, slots=True)
class ParticipantLifecycle:
    admission: ParticipantAdmissionState
    creation: ParticipantCreation = ParticipantCreation()

    def admit(self) -> ParticipantLifecycle:
        if self.admission != ParticipantAdmissionState.WAITING:
            raise WorkflowTransitionError(
                "participant admission requires a waiting job"
            )
        return replace(self, admission=ParticipantAdmissionState.HELD)

    def release(self, *, cleanup_confirmed: bool) -> ParticipantLifecycle:
        if self.admission == ParticipantAdmissionState.RELEASED:
            return self
        if self.admission == ParticipantAdmissionState.HELD and not cleanup_confirmed:
            raise WorkflowTransitionError(
                "held participant capacity requires confirmed terminal cleanup"
            )
        return replace(self, admission=ParticipantAdmissionState.RELEASED)
