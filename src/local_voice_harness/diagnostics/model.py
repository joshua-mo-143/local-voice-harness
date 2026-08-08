from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class Severity(StrEnum):
    """Diagnostic outcome, ordered from most to least urgent when reported."""

    FATAL = "fatal"
    WARNING = "warning"
    UNAVAILABLE = "unavailable"
    OK = "ok"


# Lower numbers sort first (most urgent). ``OK`` results are healthy and are
# printed last in the human summary.
SEVERITY_RANK: dict[Severity, int] = {
    Severity.FATAL: 0,
    Severity.WARNING: 1,
    Severity.UNAVAILABLE: 2,
    Severity.OK: 3,
}

PROBLEM_SEVERITIES = frozenset({Severity.FATAL, Severity.WARNING, Severity.UNAVAILABLE})


@dataclass(frozen=True, slots=True)
class Repair:
    """A narrowly scoped, idempotent, confirmation-gated recovery action.

    ``action`` performs the repair and returns a human-readable message. It may
    raise; the runner reports the failure without crashing the whole command.
    """

    summary: str
    action: Callable[[], str]


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Structured result of one independent diagnostic check."""

    name: str
    category: str
    severity: Severity
    detail: str
    suggestion: str | None = None
    repair: Repair | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "category": self.category,
            "severity": self.severity.value,
            "detail": self.detail,
            "suggestion": self.suggestion,
            "repair": self.repair.summary if self.repair is not None else None,
        }
