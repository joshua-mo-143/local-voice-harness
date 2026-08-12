from __future__ import annotations

import subprocess
import sys
from unittest import mock

import pytest

from local_voice_harness.diagnostics import health
from local_voice_harness.diagnostics.model import CheckResult, Repair, Severity


def _result(
    severity: Severity,
    *,
    category: str = "runtime",
    detail: str = "raw detail",
    suggestion: str | None = None,
    repair: Repair | None = None,
) -> CheckResult:
    return CheckResult(
        name="fixture",
        category=category,
        severity=severity,
        detail=detail,
        suggestion=suggestion,
        repair=repair,
    )


@pytest.mark.parametrize(
    ("results", "state", "expected"),
    [
        ([_result(Severity.OK)], health.HealthState.HEALTHY, "looks healthy"),
        (
            [_result(Severity.WARNING, category="audio")],
            health.HealthState.DEGRADED,
            "audio system needs attention",
        ),
        (
            [
                _result(Severity.WARNING, category="audio"),
                _result(Severity.FATAL, category="configuration"),
            ],
            health.HealthState.FATAL,
            "configuration needs attention",
        ),
    ],
)
def test_stable_health_states(
    results: list[CheckResult],
    state: health.HealthState,
    expected: str,
) -> None:
    snapshot = health.snapshot_results(results)
    response = health.render_health(
        health.HealthRun(health.HealthRunState.COMPLETE, snapshot)
    )

    assert snapshot.state is state
    assert expected in response.spoken_text
    assert response.display_text == response.spoken_text
    assert len(response.spoken_text) < 200


def test_snapshot_never_reads_or_invokes_sensitive_diagnostic_fields() -> None:
    action = mock.Mock(return_value="token=repair-secret")
    result = _result(
        Severity.WARNING,
        detail="/home/private command output password=detail-secret",
        suggestion="run /private/path with token=suggestion-secret",
        repair=Repair("restart /private/service", action),
    )

    snapshot = health.snapshot_results([result])
    rendered = health.render_health(
        health.HealthRun(health.HealthRunState.COMPLETE, snapshot)
    )

    action.assert_not_called()
    combined = rendered.spoken_text + rendered.display_text
    for forbidden in (
        "/home/private",
        "/private/path",
        "/private/service",
        "detail-secret",
        "suggestion-secret",
        "repair-secret",
        "command output",
        "restart",
    ):
        assert forbidden not in combined


def test_unknown_categories_are_reduced_to_a_fixed_runtime_message() -> None:
    snapshot = health.snapshot_results(
        [_result(Severity.WARNING, category="/secret/category")]
    )

    response = health.render_health(
        health.HealthRun(health.HealthRunState.COMPLETE, snapshot)
    )

    assert "/secret/category" not in response.spoken_text
    assert "harness runtime needs attention" in response.spoken_text


def test_bounded_runner_accepts_only_typed_safe_snapshot() -> None:
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout='{"area":"services","fatal":0,"unavailable":0,"warning":1}\n',
        stderr="/private/path password=secret",
    )
    with mock.patch.object(
        health.subprocess, "run", return_value=completed
    ) as run_process:
        result = health.run_bounded_health(timeout=3.5)

    assert result == health.HealthRun(
        health.HealthRunState.COMPLETE,
        health.HealthSnapshot(0, 1, 0, health.HealthArea.SERVICES),
    )
    run_process.assert_called_once_with(
        [sys.executable, "-m", "local_voice_harness.diagnostics.health"],
        capture_output=True,
        text=True,
        timeout=3.5,
        check=False,
    )


def test_outer_timeout_returns_fixed_safe_response() -> None:
    with mock.patch.object(
        health.subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired(
            cmd=["private-command", "token=secret"],
            timeout=1,
            output="/private/output",
            stderr="password=secret",
        ),
    ):
        run = health.run_bounded_health(timeout=1)

    response = health.render_health(run)
    assert run.state is health.HealthRunState.TIMEOUT
    assert response.spoken_text == health.TIMEOUT_RESPONSE
    assert "private" not in response.spoken_text
    assert "secret" not in response.spoken_text


@pytest.mark.parametrize(
    "outcome",
    [
        OSError("/private/path token=secret"),
        subprocess.CompletedProcess([], 1, "", "password=secret"),
        subprocess.CompletedProcess([], 0, '{"detail":"/private/path"}', ""),
        subprocess.CompletedProcess(
            [],
            0,
            '{"area":"/private/path","fatal":0,"unavailable":0,"warning":1}',
            "",
        ),
    ],
)
def test_failed_checks_return_fixed_safe_response(outcome: object) -> None:
    if isinstance(outcome, BaseException):
        patcher = mock.patch.object(health.subprocess, "run", side_effect=outcome)
    else:
        patcher = mock.patch.object(health.subprocess, "run", return_value=outcome)
    with patcher:
        run = health.run_bounded_health()

    response = health.render_health(run)
    assert run.state is health.HealthRunState.FAILED
    assert response.spoken_text == health.FAILURE_RESPONSE
    assert "private" not in response.spoken_text
    assert "secret" not in response.spoken_text


def test_health_contract_rejects_invalid_bounds_and_payloads() -> None:
    with pytest.raises(ValueError, match="positive"):
        health.run_bounded_health(timeout=0)
    with pytest.raises(ValueError, match="non-negative"):
        health.HealthSnapshot(-1, 0, 0, None)
    with pytest.raises(ValueError, match="snapshot"):
        health.HealthRun(health.HealthRunState.COMPLETE)
