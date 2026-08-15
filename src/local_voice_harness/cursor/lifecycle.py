from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TypeAlias, TypedDict


class LifecycleTransitionError(ValueError):
    """A terminal, cleanup, delivery, or announcement transition is illegal."""


class AnnouncementAck(StrEnum):
    PENDING = "pending"
    SPOKEN = "spoken"
    DESKTOP = "desktop"
    DEFERRED = "deferred"
    DISMISSED = "dismissed"


@dataclass(frozen=True, slots=True)
class TerminalIntent:
    status: str
    result: str
    error: str | None
    completed_at: float

    def __post_init__(self) -> None:
        if self.status not in {"completed", "failed", "cancelled"}:
            raise LifecycleTransitionError("terminal intent requires a terminal status")
        if self.status == "failed" and not self.error:
            raise LifecycleTransitionError("failed terminal intent requires an error")


@dataclass(frozen=True, slots=True)
class MaterializedTerminalOutcome:
    status: str
    result: str
    error: str | None
    completed_at: float

    def __post_init__(self) -> None:
        TerminalIntent(self.status, self.result, self.error, self.completed_at)


TerminalState: TypeAlias = TerminalIntent | MaterializedTerminalOutcome | None


def load_terminal_state(
    *,
    status: str,
    result: str | None,
    error: str | None,
    completed_at: float | None,
    intent_status: str | None,
    intent_result: str | None,
    intent_error: str | None,
    intent_completed_at: float | None,
) -> TerminalState:
    if intent_status is not None:
        if intent_result is None or intent_completed_at is None:
            raise LifecycleTransitionError(
                "terminal intent requires result and completion"
            )
        return TerminalIntent(
            intent_status, intent_result, intent_error, intent_completed_at
        )
    if status in {"completed", "failed", "cancelled"}:
        if result is None or completed_at is None:
            raise LifecycleTransitionError(
                "materialized terminal outcome requires result and completion"
            )
        return MaterializedTerminalOutcome(status, result, error, completed_at)
    return None


@dataclass(frozen=True, slots=True)
class CleanupSettled:
    pass


@dataclass(frozen=True, slots=True)
class CleanupReconciling:
    """Uncertain cleanup without a target release fence."""


@dataclass(frozen=True, slots=True)
class CleanupPending:
    token: str | None
    reconciliation_pending: bool


@dataclass(frozen=True, slots=True)
class CleanupOwned:
    token: str | None
    owner_pid: int
    owner_boot_id: str
    owner_start: str
    reconciliation_pending: bool

    def __post_init__(self) -> None:
        if self.owner_pid <= 0 or not self.owner_boot_id or not self.owner_start:
            raise LifecycleTransitionError("owned cleanup requires complete ownership")


CleanupState: TypeAlias = (
    CleanupSettled | CleanupReconciling | CleanupPending | CleanupOwned
)


class CleanupFields(TypedDict):
    target_release_pending: bool
    target_release_token: str | None
    target_release_owner_pid: int | None
    target_release_owner_boot_id: str | None
    target_release_owner_start: str | None
    cancellation_reconciliation_pending: bool


def load_cleanup_state(
    *,
    pending: bool,
    token: str | None,
    owner_pid: int | None,
    owner_boot_id: str | None,
    owner_start: str | None,
    reconciliation_pending: bool,
) -> CleanupState:
    owner = (owner_pid, owner_boot_id, owner_start)
    if not pending:
        if token or any(value is not None for value in owner):
            raise LifecycleTransitionError(
                "settled cleanup cannot retain release ownership"
            )
        return CleanupReconciling() if reconciliation_pending else CleanupSettled()
    if any(value is not None for value in owner):
        if not all(value is not None for value in owner):
            raise LifecycleTransitionError(
                "cleanup owner PID, boot ID, and process start must be paired"
            )
        assert owner_pid is not None
        assert owner_boot_id is not None
        assert owner_start is not None
        return CleanupOwned(
            token,
            owner_pid,
            owner_boot_id,
            owner_start,
            reconciliation_pending,
        )
    return CleanupPending(token, reconciliation_pending)


def begin_cleanup(token: str) -> CleanupPending:
    return CleanupPending(token, True)


def finish_cleanup_reconciliation(state: CleanupState) -> CleanupState:
    if isinstance(state, CleanupReconciling):
        return CleanupSettled()
    if isinstance(state, (CleanupPending, CleanupOwned)):
        return replace(state, reconciliation_pending=False)
    return state


def claim_cleanup(
    state: CleanupState,
    expected_token: str,
    *,
    token: str | None = None,
    owner_pid: int,
    owner_boot_id: str,
    owner_start: str,
) -> CleanupOwned:
    if (
        not isinstance(state, CleanupPending)
        or (state.token or "") != expected_token
        or not (token or expected_token)
    ):
        raise LifecycleTransitionError("cleanup claim is stale or illegal")
    return CleanupOwned(
        token or expected_token,
        owner_pid,
        owner_boot_id,
        owner_start,
        state.reconciliation_pending,
    )


def take_over_cleanup(
    state: CleanupState,
    expected_token: str,
    *,
    token: str,
    owner_pid: int,
    owner_boot_id: str,
    owner_start: str,
) -> CleanupOwned:
    if (
        not isinstance(state, CleanupOwned)
        or (state.token or "") != expected_token
        or not token
    ):
        raise LifecycleTransitionError("cleanup takeover is stale or illegal")
    return CleanupOwned(
        token,
        owner_pid,
        owner_boot_id,
        owner_start,
        state.reconciliation_pending,
    )


def abandon_cleanup_owner(state: CleanupState, token: str) -> CleanupPending:
    if not isinstance(state, CleanupOwned) or (state.token or "") != token:
        raise LifecycleTransitionError("cleanup owner release is stale or illegal")
    return CleanupPending(state.token, state.reconciliation_pending)


def settle_cleanup(state: CleanupState, token: str) -> CleanupSettled:
    if (
        not isinstance(state, (CleanupPending, CleanupOwned))
        or (state.token or "") != token
    ):
        raise LifecycleTransitionError("cleanup settlement is stale or illegal")
    return CleanupSettled()


def cleanup_fields(state: CleanupState) -> CleanupFields:
    if isinstance(state, (CleanupSettled, CleanupReconciling)):
        return {
            "target_release_pending": False,
            "target_release_token": None,
            "target_release_owner_pid": None,
            "target_release_owner_boot_id": None,
            "target_release_owner_start": None,
            "cancellation_reconciliation_pending": isinstance(
                state, CleanupReconciling
            ),
        }
    values: CleanupFields = {
        "target_release_pending": True,
        "target_release_token": state.token,
        "cancellation_reconciliation_pending": state.reconciliation_pending,
        "target_release_owner_pid": None,
        "target_release_owner_boot_id": None,
        "target_release_owner_start": None,
    }
    if isinstance(state, CleanupOwned):
        values.update(
            target_release_owner_pid=state.owner_pid,
            target_release_owner_boot_id=state.owner_boot_id,
            target_release_owner_start=state.owner_start,
        )
    return values


@dataclass(frozen=True, slots=True)
class AnnouncementState:
    acknowledgement: AnnouncementAck
    repeated: bool = False

    @property
    def dismissed(self) -> bool:
        return self.acknowledgement == AnnouncementAck.DISMISSED


def acknowledge_announcement(
    state: AnnouncementState, acknowledgement: AnnouncementAck
) -> AnnouncementState:
    if acknowledgement == AnnouncementAck.PENDING:
        raise LifecycleTransitionError("announcement acknowledgement did not advance")
    allowed = {
        AnnouncementAck.PENDING: {
            AnnouncementAck.SPOKEN,
            AnnouncementAck.DESKTOP,
            AnnouncementAck.DEFERRED,
            AnnouncementAck.DISMISSED,
        },
        AnnouncementAck.DESKTOP: {
            AnnouncementAck.SPOKEN,
            AnnouncementAck.DISMISSED,
        },
        AnnouncementAck.DEFERRED: {
            AnnouncementAck.SPOKEN,
            AnnouncementAck.DESKTOP,
            AnnouncementAck.DISMISSED,
        },
        AnnouncementAck.SPOKEN: set(),
        AnnouncementAck.DISMISSED: set(),
    }
    if acknowledgement not in allowed[state.acknowledgement]:
        raise LifecycleTransitionError("announcement acknowledgement is already final")
    return replace(state, acknowledgement=acknowledgement)


def dismiss_announcement(state: AnnouncementState) -> AnnouncementState:
    if state.acknowledgement == AnnouncementAck.DISMISSED:
        raise LifecycleTransitionError("announcement is already dismissed")
    return AnnouncementState(AnnouncementAck.DISMISSED, state.repeated)


def repeat_announcement(state: AnnouncementState) -> AnnouncementState:
    return AnnouncementState(AnnouncementAck.PENDING, True)


@dataclass(frozen=True, slots=True)
class PendingDelivery:
    generation: int
    retry_at: float
    attempts: int
    announcement: AnnouncementState


@dataclass(frozen=True, slots=True)
class ClaimedDelivery:
    generation: int
    token: str
    claimed_at: float
    retry_at: float
    attempts: int
    announcement: AnnouncementState


@dataclass(frozen=True, slots=True)
class Delivered:
    generation: int
    delivered_at: float | None
    attempts: int
    announcement: AnnouncementState


DeliveryState: TypeAlias = PendingDelivery | ClaimedDelivery | Delivered


def load_delivery_state(
    *,
    delivered: bool,
    generation: int,
    claim_token: str | None,
    claimed_at: float | None,
    retry_at: float,
    attempts: int,
    delivered_at: float | None,
    acknowledgement: str,
    repeated: bool,
    dismissed: bool | None = None,
) -> DeliveryState:
    try:
        ack = AnnouncementAck(acknowledgement)
    except ValueError as exc:
        raise LifecycleTransitionError(
            "announcement acknowledgement is invalid"
        ) from exc
    announcement = AnnouncementState(ack, repeated)
    if dismissed is not None and dismissed != announcement.dismissed:
        raise LifecycleTransitionError("dismissal state and acknowledgement must match")
    if bool(claim_token) != (claimed_at is not None):
        raise LifecycleTransitionError(
            "delivery claim token and timestamp must be paired"
        )
    if delivered:
        if claim_token:
            raise LifecycleTransitionError("delivered state cannot retain a claim")
        if ack not in {AnnouncementAck.SPOKEN, AnnouncementAck.DISMISSED}:
            raise LifecycleTransitionError(
                "delivered state requires spoken or dismissed acknowledgement"
            )
        return Delivered(generation, delivered_at, attempts, announcement)
    if ack in {AnnouncementAck.SPOKEN, AnnouncementAck.DISMISSED}:
        raise LifecycleTransitionError(
            "pending delivery cannot have a final acknowledgement"
        )
    if claim_token is not None and claimed_at is not None:
        return ClaimedDelivery(
            generation,
            claim_token,
            claimed_at,
            retry_at,
            attempts,
            announcement,
        )
    return PendingDelivery(generation, retry_at, attempts, announcement)


def prepare_delivery(state: DeliveryState) -> PendingDelivery:
    return PendingDelivery(
        state.generation + 1,
        0,
        0,
        AnnouncementState(AnnouncementAck.PENDING, state.announcement.repeated),
    )


def claim_delivery(
    state: DeliveryState, token: str, claimed_at: float, *, lease_seconds: float
) -> ClaimedDelivery:
    if isinstance(state, Delivered):
        raise LifecycleTransitionError("delivered state cannot be claimed")
    if isinstance(state, ClaimedDelivery) and (
        claimed_at - state.claimed_at < lease_seconds
    ):
        raise LifecycleTransitionError("delivery claim is still live")
    if claimed_at < state.retry_at:
        raise LifecycleTransitionError("delivery retry is not due")
    return ClaimedDelivery(
        state.generation,
        token,
        claimed_at,
        state.retry_at,
        state.attempts + 1,
        state.announcement,
    )


def renew_delivery(
    state: DeliveryState,
    token: str,
    renewed_at: float,
    *,
    lease_seconds: float,
) -> ClaimedDelivery:
    if (
        not isinstance(state, ClaimedDelivery)
        or state.token != token
        or renewed_at - state.claimed_at >= lease_seconds
    ):
        raise LifecycleTransitionError("delivery renewal is stale or illegal")
    return replace(state, claimed_at=renewed_at)


def acknowledge_delivery(
    state: DeliveryState,
    token: str,
    delivered_at: float | None,
    acknowledgement: AnnouncementAck,
    *,
    lease_seconds: float | None = None,
) -> DeliveryState:
    if not isinstance(state, ClaimedDelivery) or state.token != token:
        raise LifecycleTransitionError("delivery acknowledgement is stale or illegal")
    if (
        lease_seconds is not None
        and delivered_at is not None
        and delivered_at - state.claimed_at >= lease_seconds
    ):
        raise LifecycleTransitionError("delivery acknowledgement claim expired")
    announcement = acknowledge_announcement(state.announcement, acknowledgement)
    if acknowledgement in {AnnouncementAck.SPOKEN, AnnouncementAck.DISMISSED}:
        return Delivered(state.generation, delivered_at, state.attempts, announcement)
    return PendingDelivery(state.generation, 0, state.attempts, announcement)


def acknowledge_without_claim(
    state: DeliveryState,
    delivered_at: float | None,
    acknowledgement: AnnouncementAck = AnnouncementAck.SPOKEN,
) -> Delivered:
    if isinstance(state, Delivered):
        if state.announcement.acknowledgement == acknowledgement:
            return state
        raise LifecycleTransitionError("delivered acknowledgement cannot change")
    announcement = state.announcement
    if announcement.acknowledgement != AnnouncementAck.PENDING:
        announcement = AnnouncementState(
            AnnouncementAck.PENDING,
            announcement.repeated,
        )
    announcement = acknowledge_announcement(announcement, acknowledgement)
    return Delivered(state.generation, delivered_at, state.attempts, announcement)


def release_delivery(
    state: DeliveryState, token: str, *, retry_at: float
) -> PendingDelivery:
    if not isinstance(state, ClaimedDelivery) or state.token != token:
        raise LifecycleTransitionError("delivery release is stale or illegal")
    return PendingDelivery(
        state.generation,
        retry_at,
        state.attempts,
        state.announcement,
    )


def delivery_fields(state: DeliveryState) -> dict[str, object]:
    values: dict[str, object] = {
        "delivery_generation": state.generation,
        "delivery_attempts": state.attempts,
        "announcement_ack": state.announcement.acknowledgement.value,
        "announcement_repeated": state.announcement.repeated,
        "delivery_claim_token": None,
        "delivery_claimed_at": None,
        "delivery_retry_at": 0,
        "delivered": isinstance(state, Delivered),
    }
    if isinstance(state, ClaimedDelivery):
        values.update(
            delivery_claim_token=state.token,
            delivery_claimed_at=state.claimed_at,
            delivery_retry_at=state.retry_at,
        )
    elif isinstance(state, PendingDelivery):
        values["delivery_retry_at"] = state.retry_at
    if isinstance(state, Delivered):
        values["delivered_at"] = state.delivered_at
    return values
