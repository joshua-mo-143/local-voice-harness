from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path


def coverage_errors(
    report: Mapping[str, object], targets: Mapping[str, float]
) -> list[str]:
    """Return failures for missing or under-covered risk-critical modules."""

    files_value = report.get("files")
    files = files_value if isinstance(files_value, dict) else {}
    errors: list[str] = []
    for path, minimum in sorted(targets.items()):
        file_value = files.get(path)
        if not isinstance(file_value, dict):
            errors.append(f"{path}: missing from coverage report")
            continue
        summary = file_value.get("summary")
        percent = summary.get("percent_covered") if isinstance(summary, dict) else None
        if not isinstance(percent, int | float):
            errors.append(f"{path}: coverage percentage is missing")
        elif percent < minimum:
            errors.append(f"{path}: {percent:.1f}% is below {minimum:.1f}%")
    return errors


def configured_targets(pyproject: Path) -> dict[str, float]:
    metadata = tomllib.loads(pyproject.read_text())
    configured = metadata["tool"]["voice-harness"]["coverage"]["module-minimums"]
    return {str(path): float(minimum) for path, minimum in configured.items()}


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Enforce risk-specific module coverage floors"
    )
    parser.add_argument("report", nargs="?", type=Path, default=Path("coverage.json"))
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    options = parser.parse_args(arguments)

    report = json.loads(options.report.read_text())
    errors = coverage_errors(report, configured_targets(options.pyproject))
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("risk-specific module coverage targets satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
