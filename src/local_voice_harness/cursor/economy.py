"""Optional Cursor economy mode: tier-aware model routing and CI verification."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..user_config import CursorEconomySettings, load_user_config
from .model import WorkflowParticipant, WorkflowPhase, WorkflowTier

_VERIFICATION_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("uv", "run", "--locked", "ruff", "format", "--check", "."),
    ("uv", "run", "--locked", "ruff", "check", "."),
    ("uv", "run", "--locked", "pyright", "--pythonversion", "3.12"),
    ("uv", "run", "--locked", "pytest", "-q"),
)
_VERIFICATION_TIMEOUT_SECONDS = 600.0


@dataclass(frozen=True)
class VerificationOutcome:
    passed: bool
    detail: str


def cursor_economy_settings() -> CursorEconomySettings:
    """Return economy settings, falling back to defaults if config is unavailable."""

    try:
        return load_user_config().cursor.economy
    except Exception:
        return CursorEconomySettings()


def resolve_cursor_model(
    economy: CursorEconomySettings,
    *,
    participant: WorkflowParticipant,
    phase: WorkflowPhase,
    tier: WorkflowTier | None,
) -> str | None:
    """Return a Cursor ``--model`` slug when economy mode is enabled."""

    if not economy.enabled:
        return None
    models = economy.models
    if phase == WorkflowPhase.CLASSIFYING:
        return models.classifier or None
    if participant == WorkflowParticipant.REVIEWER:
        return models.reviewer or None
    if participant == WorkflowParticipant.IMPLEMENTER:
        if tier == WorkflowTier.HIGH_RISK and models.implementer_high_risk:
            return models.implementer_high_risk
        return models.implementer or None
    return models.planner or None


def verify_checkout(checkout: Path) -> VerificationOutcome:
    """Run the local CI-equivalent checks in ``checkout``."""

    if not checkout.is_dir():
        return VerificationOutcome(False, "worktree checkout is missing")
    for command in _VERIFICATION_COMMANDS:
        try:
            process = subprocess.run(
                command,
                cwd=checkout,
                capture_output=True,
                text=True,
                timeout=_VERIFICATION_TIMEOUT_SECONDS,
                check=False,
            )
        except OSError as exc:
            return VerificationOutcome(False, f"{command[0]} failed to start: {exc}")
        except subprocess.TimeoutExpired:
            return VerificationOutcome(
                False,
                f"{' '.join(command)} timed out after "
                f"{int(_VERIFICATION_TIMEOUT_SECONDS)} seconds",
            )
        if process.returncode:
            detail = _command_failure_detail(command, process.stdout, process.stderr)
            return VerificationOutcome(False, detail)
    return VerificationOutcome(True, "verification passed")


def _command_failure_detail(
    command: Sequence[str],
    stdout: str,
    stderr: str,
) -> str:
    label = " ".join(command)
    output = (stderr or stdout).strip()
    if not output:
        return f"{label} exited with a non-zero status"
    first_line = output.splitlines()[0].strip()
    if len(first_line) > 180:
        first_line = first_line[:177] + "..."
    return f"{label} failed: {first_line}"
