"""Typed coordinator commands for atomic durable transitions.

Callers submit a command against an expected revision. The store commits the
resulting job state, reservations, one audit event, and any newly admitted
outbox effects in one SQLite transaction. Duplicate ``command_id`` delivery is
a no-op that returns the already committed job.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .model import CursorJob


class CoordinatorError(ValueError):
    """A coordinator command or effect is incomplete."""


@dataclass(frozen=True, slots=True)
class DurableEffect:
    """One named outbox effect admitted with a state transition."""

    kind: str
    idempotency_key: str
    payload: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind.strip() or not self.idempotency_key.strip():
            raise CoordinatorError("effect requires kind and idempotency_key")


@dataclass(frozen=True, slots=True)
class CoordinatorCommand:
    """Idempotent durable command submitted to the coordinator."""

    job_id: str
    expected_revision: int
    command_id: str
    kind: str

    def __post_init__(self) -> None:
        if not self.job_id.strip() or not self.command_id.strip():
            raise CoordinatorError("command requires job_id and command_id")
        if not self.kind.strip():
            raise CoordinatorError("command requires kind")
        if self.expected_revision < 0:
            raise CoordinatorError("expected revision must not be negative")


@dataclass(frozen=True, slots=True)
class CoordinatorDecision:
    """State and effects produced by a command against the current job."""

    job: CursorJob
    effects: tuple[DurableEffect, ...] = ()
    event_kind: str = "transition"
    event_payload: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reserved = {"command_kind", "effects"} & self.event_payload.keys()
        if reserved:
            names = ", ".join(sorted(reserved))
            raise CoordinatorError(f"event payload cannot override {names}")
