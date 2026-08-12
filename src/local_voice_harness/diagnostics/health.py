from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum

from ..responses import AssistantResponse
from .model import CheckResult, Severity
from .runner import run_diagnostics

HEALTH_TIMEOUT_SECONDS = 15.0


class HealthState(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FATAL = "fatal"


class HealthArea(StrEnum):
    CONFIGURATION = "configuration"
    SERVICES = "services"
    AUDIO = "audio"
    DEPENDENCIES = "dependencies"
    JOBS = "jobs"
    INTEGRATIONS = "integrations"
    RUNTIME = "runtime"


class HealthRunState(StrEnum):
    COMPLETE = "complete"
    TIMEOUT = "timeout"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    fatal: int
    warning: int
    unavailable: int
    area: HealthArea | None

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.fatal, self.warning, self.unavailable)
        ):
            raise ValueError("health counts must be non-negative integers")

    @property
    def state(self) -> HealthState:
        if self.fatal:
            return HealthState.FATAL
        if self.warning or self.unavailable:
            return HealthState.DEGRADED
        return HealthState.HEALTHY


@dataclass(frozen=True, slots=True)
class HealthRun:
    state: HealthRunState
    snapshot: HealthSnapshot | None = None

    def __post_init__(self) -> None:
        if (self.state is HealthRunState.COMPLETE) != (self.snapshot is not None):
            raise ValueError("only a complete health run has a snapshot")


_CATEGORY_AREAS: dict[str, HealthArea] = {
    "configuration": HealthArea.CONFIGURATION,
    "systemd": HealthArea.SERVICES,
    "audio": HealthArea.AUDIO,
    "executables": HealthArea.DEPENDENCIES,
    "python-env": HealthArea.DEPENDENCIES,
    "models": HealthArea.DEPENDENCIES,
    "gpu": HealthArea.DEPENDENCIES,
    "jobs": HealthArea.JOBS,
    "integrations": HealthArea.INTEGRATIONS,
    "runtime": HealthArea.RUNTIME,
}
_AREA_RANK = {area: rank for rank, area in enumerate(HealthArea)}
_SAFE_AREA_MESSAGES: dict[HealthArea, str] = {
    HealthArea.CONFIGURATION: "The harness configuration needs attention.",
    HealthArea.SERVICES: "A harness service needs attention.",
    HealthArea.AUDIO: "The audio system needs attention.",
    HealthArea.DEPENDENCIES: "A required harness dependency needs attention.",
    HealthArea.JOBS: "The background job system needs attention.",
    HealthArea.INTEGRATIONS: "An enabled integration needs attention.",
    HealthArea.RUNTIME: "The harness runtime needs attention.",
}

HEALTHY_RESPONSE = "The voice harness looks healthy."
TIMEOUT_RESPONSE = (
    "The harness health check took too long, so I stopped it without making changes."
)
FAILURE_RESPONSE = (
    "I couldn't complete the harness health check. Nothing was changed or repaired."
)


def snapshot_results(results: list[CheckResult]) -> HealthSnapshot:
    """Reduce typed diagnostics to non-sensitive health metadata."""

    counts = {severity: 0 for severity in Severity}
    candidates: list[tuple[int, int, HealthArea]] = []
    severity_rank = {
        Severity.FATAL: 0,
        Severity.WARNING: 1,
        Severity.UNAVAILABLE: 2,
        Severity.OK: 3,
    }
    for result in results:
        counts[result.severity] += 1
        if result.severity is Severity.OK:
            continue
        area = _CATEGORY_AREAS.get(result.category, HealthArea.RUNTIME)
        candidates.append((severity_rank[result.severity], _AREA_RANK[area], area))
    top_area = min(candidates)[2] if candidates else None
    return HealthSnapshot(
        fatal=counts[Severity.FATAL],
        warning=counts[Severity.WARNING],
        unavailable=counts[Severity.UNAVAILABLE],
        area=top_area,
    )


def collect_snapshot() -> HealthSnapshot:
    """Run read-only checks without inspecting or invoking repair callbacks."""

    return snapshot_results(run_diagnostics())


def _snapshot_json(snapshot: HealthSnapshot) -> str:
    return json.dumps(
        {
            "fatal": snapshot.fatal,
            "warning": snapshot.warning,
            "unavailable": snapshot.unavailable,
            "area": snapshot.area.value if snapshot.area is not None else None,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _parse_snapshot(value: str) -> HealthSnapshot:
    payload = json.loads(value)
    if not isinstance(payload, dict) or set(payload) != {
        "fatal",
        "warning",
        "unavailable",
        "area",
    }:
        raise ValueError("invalid health snapshot")
    raw_area = payload["area"]
    area = None if raw_area is None else HealthArea(raw_area)
    return HealthSnapshot(
        fatal=payload["fatal"],
        warning=payload["warning"],
        unavailable=payload["unavailable"],
        area=area,
    )


def run_bounded_health(*, timeout: float = HEALTH_TIMEOUT_SECONDS) -> HealthRun:
    """Run diagnostics in a disposable child with one outer deadline."""

    if timeout <= 0:
        raise ValueError("health timeout must be positive")
    try:
        process = subprocess.run(
            [sys.executable, "-m", "local_voice_harness.diagnostics.health"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return HealthRun(HealthRunState.TIMEOUT)
    except OSError:
        return HealthRun(HealthRunState.FAILED)
    if process.returncode:
        return HealthRun(HealthRunState.FAILED)
    try:
        snapshot = _parse_snapshot(process.stdout)
    except (json.JSONDecodeError, TypeError, ValueError):
        return HealthRun(HealthRunState.FAILED)
    return HealthRun(HealthRunState.COMPLETE, snapshot)


def render_health(run: HealthRun) -> AssistantResponse:
    """Render only fixed text and aggregate counts from the safe snapshot."""

    if run.state is HealthRunState.TIMEOUT:
        return AssistantResponse.from_text(TIMEOUT_RESPONSE)
    if run.state is HealthRunState.FAILED:
        return AssistantResponse.from_text(FAILURE_RESPONSE)
    assert run.snapshot is not None
    snapshot = run.snapshot
    if snapshot.state is HealthState.HEALTHY:
        return AssistantResponse.from_text(HEALTHY_RESPONSE)
    if snapshot.state is HealthState.FATAL:
        count = snapshot.fatal
        subject = "check has" if count == 1 else "checks have"
        headline = f"The voice harness is unhealthy: {count} required {subject} failed."
    else:
        count = snapshot.warning + snapshot.unavailable
        subject = "check needs" if count == 1 else "checks need"
        headline = f"The voice harness is degraded: {count} {subject} attention."
    area = _SAFE_AREA_MESSAGES[snapshot.area] if snapshot.area is not None else ""
    return AssistantResponse.from_text(f"{headline} {area}".strip())


def self_health_response(
    *, timeout: float = HEALTH_TIMEOUT_SECONDS
) -> AssistantResponse:
    return render_health(run_bounded_health(timeout=timeout))


def main() -> None:
    """Emit only the bounded typed snapshot for the parent process."""

    print(_snapshot_json(collect_snapshot()))


if __name__ == "__main__":
    main()
