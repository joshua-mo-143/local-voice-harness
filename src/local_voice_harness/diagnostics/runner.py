from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import TextIO

from .checks import ALL_CHECKS, Check
from .model import (
    PROBLEM_SEVERITIES,
    SEVERITY_RANK,
    CheckResult,
    Severity,
)

Confirm = Callable[[str], bool]

_SEVERITY_LABELS: dict[Severity, str] = {
    Severity.FATAL: "FATAL",
    Severity.WARNING: "WARNING",
    Severity.UNAVAILABLE: "UNAVAILABLE",
    Severity.OK: "OK",
}


def run_diagnostics(checks: Sequence[Check] = ALL_CHECKS) -> list[CheckResult]:
    """Run every check, converting any crash into a FATAL result.

    The harness must remain diagnosable even when services or sockets are down,
    so no individual check is allowed to abort the whole run.
    """

    results: list[CheckResult] = []
    for check in checks:
        try:
            produced = check()
        except Exception as exc:  # never let one check crash the doctor
            results.append(
                CheckResult(
                    name=getattr(check, "__name__", "unknown-check"),
                    category="internal",
                    severity=Severity.FATAL,
                    detail=f"diagnostic check raised {type(exc).__name__}: {exc}",
                )
            )
            continue
        results.extend(produced)
    return results


def _severity_counts(results: Iterable[CheckResult]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for result in results:
        counts[result.severity.value] += 1
    return {severity.value: counts.get(severity.value, 0) for severity in Severity}


def render_json(results: Sequence[CheckResult]) -> str:
    payload = {
        "checks": [result.to_dict() for result in results],
        "summary": _severity_counts(results),
        "healthy": not any(result.severity is Severity.FATAL for result in results),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def render_human(results: Sequence[CheckResult]) -> str:
    lines: list[str] = []
    ordered = sorted(
        results, key=lambda result: (SEVERITY_RANK[result.severity], result.name)
    )
    for severity in (Severity.FATAL, Severity.WARNING, Severity.UNAVAILABLE):
        group = [result for result in ordered if result.severity is severity]
        if not group:
            continue
        lines.append(f"{_SEVERITY_LABELS[severity]} ({len(group)}):")
        for result in group:
            lines.append(f"  [{result.name}] {result.detail}")
            if result.suggestion:
                lines.append(f"      suggested: {result.suggestion}")
            if result.repair is not None:
                lines.append(f"      repair available: {result.repair.summary}")
        lines.append("")

    counts = _severity_counts(results)
    ok = counts[Severity.OK.value]
    summary = (
        f"{counts[Severity.FATAL.value]} fatal, "
        f"{counts[Severity.WARNING.value]} warning, "
        f"{counts[Severity.UNAVAILABLE.value]} unavailable, "
        f"{ok} ok"
    )
    if counts[Severity.FATAL.value]:
        headline = "The harness is unhealthy; fix the fatal issues above."
    elif counts[Severity.WARNING.value]:
        headline = "The harness is usable but some features are degraded."
    else:
        headline = "The harness looks healthy."
    lines.append(headline)
    lines.append(f"Summary: {summary}")
    if any(
        result.repair is not None and result.severity in PROBLEM_SEVERITIES
        for result in results
    ):
        lines.append("Run `voice-harness doctor --fix` for confirmation-gated repairs.")
    return "\n".join(lines)


def _default_confirm(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N] ")
    except EOFError:
        return False
    return answer.strip().casefold() in {"y", "yes"}


def apply_repairs(
    results: Sequence[CheckResult],
    *,
    confirm: Confirm,
    out: TextIO,
) -> None:
    for result in results:
        repair = result.repair
        if repair is None or result.severity not in PROBLEM_SEVERITIES:
            continue
        if not confirm(f"Apply repair for [{result.name}]: {repair.summary}?"):
            print(f"  skipped {result.name}", file=out)
            continue
        try:
            message = repair.action()
        except Exception as exc:  # a failed repair must not crash the doctor
            print(f"  repair failed for {result.name}: {exc}", file=out)
            continue
        print(f"  repaired {result.name}: {message}", file=out)


def doctor(
    *,
    json_output: bool = False,
    fix: bool = False,
    checks: Sequence[Check] = ALL_CHECKS,
    confirm: Confirm | None = None,
    out: TextIO | None = None,
) -> int:
    """Run diagnostics and return a process exit code (1 only on fatal issues)."""

    stream = out if out is not None else sys.stdout
    results = run_diagnostics(checks)
    if json_output:
        print(render_json(results), file=stream)
    else:
        print(render_human(results), file=stream)
        if fix:
            repairable = [
                result
                for result in results
                if result.repair is not None and result.severity in PROBLEM_SEVERITIES
            ]
            if repairable:
                print(
                    "\nGuided recovery (each action needs confirmation):", file=stream
                )
                apply_repairs(
                    repairable,
                    confirm=confirm if confirm is not None else _default_confirm,
                    out=stream,
                )
            else:
                print("\nNo confirmation-gated repairs are available.", file=stream)
    return 1 if any(result.severity is Severity.FATAL for result in results) else 0
