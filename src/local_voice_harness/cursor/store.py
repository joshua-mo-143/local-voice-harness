from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import time
import uuid
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from ..integrations.github import GitHubError, load_github_provider_state
from ..job_lifecycle import (
    FollowUpEvent,
    JobLifecycleError,
    QueuedJob,
    WorkerCallbackEvent,
    apply_follow_up,
)
from .coordinator import CoordinatorCommand, CoordinatorDecision
from .model import (
    ACTIVE_STATUSES,
    CURRENT_SCHEMA_VERSION,
    LEGACY_BOOT_ID,
    TERMINAL_STATUSES,
    CursorJob,
    JobStatus,
    JobValidationError,
    validate_reservations,
    validate_transition,
)
from .operations import WorkerOwnership
from .sqlite_store import SQLiteJobDatabase, fsync_database_directory

DELIVERED_JOB_RETENTION_SECONDS = 7 * 24 * 60 * 60
MAX_WORKFLOW_ARTIFACT_BYTES = 64 * 1024
WORKFLOW_ARTIFACT_KINDS = frozenset({"plan", "review"})
_ARTIFACT_REF = re.compile(
    r"^\.artifacts/(?P<job>[0-9a-f]{12})/"
    r"(?P<kind>plan|review)-(?P<round>[0-2])"
    r"(?:-(?P<digest>[0-9a-f]{64}))?\.json$"
)
_QUARANTINE_METADATA_NAME = re.compile(r"^(?P<job_id>[0-9a-f]{12})-.+\.metadata\.json$")
LegacyWorkerDisposition = Literal["absent", "stopped", "unsafe"]
LegacyWorkerInspector = Callable[[CursorJob], LegacyWorkerDisposition]
MAINTENANCE_FILENAME = ".maintenance"
MAINTENANCE_SCHEMA_VERSION = 1


class JobQuarantinedError(JobValidationError):
    """A malformed job was isolated from active job processing."""


class ArtifactQuarantinedError(JobValidationError):
    """A malformed workflow artifact was isolated from active processing."""


class FollowUpUnavailable(JobValidationError):
    """A completed parent job cannot be used as a follow-up source."""


class FollowUpCheckoutBusy(JobValidationError):
    """The parent's retained checkout is reserved by another active job."""


class JobMaintenanceError(JobValidationError):
    """The durable job store is fenced for maintenance."""


class ActiveTicketConflict(JobValidationError):
    """Another resource-owning job already handles this canonical ticket."""

    def __init__(self, active_job_id: str) -> None:
        self.active_job_id = active_job_id
        super().__init__(f"ticket is already active as Cursor job {active_job_id}")


class JobQuarantineWarning(UserWarning):
    """A malformed job file was moved into quarantine."""


@dataclass(frozen=True, slots=True)
class MaintenanceLease:
    token: str
    started_at: float
    owner_pid: int
    owner_boot_id: str
    owner_process_start: str

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": MAINTENANCE_SCHEMA_VERSION,
            "operation": "delete_all",
            "token": self.token,
            "started_at": self.started_at,
            "owner_pid": self.owner_pid,
            "owner_boot_id": self.owner_boot_id,
            "owner_process_start": self.owner_process_start,
        }


MaintenanceOwnerAlive = Callable[[MaintenanceLease], bool | None]


@dataclass(frozen=True, slots=True)
class QuarantineAcknowledgement:
    job_id: str
    resolved_metadata: tuple[str, ...]
    acknowledged_at: float


@dataclass(frozen=True, slots=True)
class QuarantineEvidence:
    job_id: str | None
    metadata_path: Path
    payload_path: Path | None
    quarantined_at: float | None
    quarantine_error: str
    resolved: bool
    status: str | None
    worker_pid: int | None
    worker_boot_id: str | None
    worker_process_start: str | None
    herdr_target: str | None
    worktree_path: str | None
    inspection_error: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "metadata_path": str(self.metadata_path),
            "payload_path": (
                str(self.payload_path) if self.payload_path is not None else None
            ),
            "quarantined_at": self.quarantined_at,
            "quarantine_error": self.quarantine_error,
            "resolved": self.resolved,
            "status": self.status,
            "worker_pid": self.worker_pid,
            "worker_boot_id": self.worker_boot_id,
            "worker_process_start": self.worker_process_start,
            "herdr_target": self.herdr_target,
            "worktree_path": self.worktree_path,
            "inspection_error": self.inspection_error,
        }


@contextmanager
def locked(jobs_dir: Path) -> Iterator[None]:
    jobs_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = jobs_dir / ".lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _quarantine_metadata(path: Path) -> list[Path]:
    quarantine = path.parent / ".quarantine"
    return sorted(quarantine.glob(f"{path.stem}-*.metadata.json"))


def _quarantine_resolution_path(metadata_path: Path) -> Path:
    stem = metadata_path.name.removesuffix(".metadata.json")
    return metadata_path.with_name(f"{stem}.reservation-resolution.json")


def _quarantine_metadata_resolved(metadata_path: Path) -> bool:
    resolution_path = _quarantine_resolution_path(metadata_path)
    if metadata_path.is_symlink() or resolution_path.is_symlink():
        return False
    try:
        resolution = json.loads(resolution_path.read_text())
        return (
            isinstance(resolution, dict)
            and resolution.get("metadata_sha256")
            == hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        )
    except (OSError, json.JSONDecodeError):
        return False


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _quarantine_evidence_unlocked(metadata_path: Path) -> QuarantineEvidence:
    match = _QUARANTINE_METADATA_NAME.fullmatch(metadata_path.name)
    job_id = match.group("job_id") if match is not None else None
    resolved = not metadata_path.is_symlink() and _quarantine_metadata_resolved(
        metadata_path
    )
    payload_path: Path | None = None
    quarantined_at: float | None = None
    quarantine_error = "quarantine metadata is unreadable"
    status: str | None = None
    worker_pid: int | None = None
    worker_boot_id: str | None = None
    worker_process_start: str | None = None
    herdr_target: str | None = None
    worktree_path: str | None = None
    inspection_error: str | None = None

    try:
        if metadata_path.is_symlink():
            raise OSError("quarantine metadata cannot be a symlink")
        metadata = json.loads(metadata_path.read_text())
        if not isinstance(metadata, dict):
            raise ValueError("quarantine metadata is not a JSON object")
        timestamp = metadata.get("quarantined_at")
        if isinstance(timestamp, int | float) and not isinstance(timestamp, bool):
            quarantined_at = float(timestamp)
        quarantine_error = str(metadata.get("error") or "validation failed")
        quarantined_name = metadata.get("quarantined_name")
        if not isinstance(quarantined_name, str) or not quarantined_name:
            raise ValueError("quarantine metadata has no payload name")
        if Path(quarantined_name).name != quarantined_name:
            raise ValueError("quarantine payload path is not confined")
        payload_path = metadata_path.parent / quarantined_name
        if payload_path.is_symlink():
            raise OSError("quarantine payload cannot be a symlink")
        raw = json.loads(payload_path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("quarantine payload is not a JSON object")
        harness_state = _mapping_field(raw, "harness_state")
        checkout_state = _mapping_field(raw, "checkout_state")
        status = _optional_string(raw.get("status"))
        pid = raw.get("worker_pid")
        if isinstance(pid, int) and not isinstance(pid, bool):
            worker_pid = pid
        worker_boot_id = _optional_string(raw.get("worker_boot_id"))
        worker_process_start = _optional_string(raw.get("worker_process_start"))
        herdr_target = _optional_string(
            raw.get("herdr_target")
            or raw.get("session_id")
            or harness_state.get("session_id")
        )
        worktree_path = _optional_string(
            raw.get("worktree_path") or checkout_state.get("path")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        inspection_error = str(exc) or type(exc).__name__

    return QuarantineEvidence(
        job_id=job_id,
        metadata_path=metadata_path,
        payload_path=payload_path,
        quarantined_at=quarantined_at,
        quarantine_error=quarantine_error,
        resolved=resolved,
        status=status,
        worker_pid=worker_pid,
        worker_boot_id=worker_boot_id,
        worker_process_start=worker_process_start,
        herdr_target=herdr_target,
        worktree_path=worktree_path,
        inspection_error=inspection_error,
    )


def _unresolved_quarantine_metadata(jobs_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted((jobs_dir / ".quarantine").glob("*.metadata.json"))
        if not _quarantine_metadata_resolved(path)
    ]


def _mapping_field(value: dict[str, object], field: str) -> dict[str, object]:
    nested = value.get(field)
    return dict(nested) if isinstance(nested, dict) else {}


def _quarantine_may_reserve(
    metadata_path: Path,
    *,
    reservation: Literal["target", "worktree"],
    value: str,
) -> bool:
    if _quarantine_metadata_resolved(metadata_path):
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
        if not isinstance(metadata, dict):
            return True
        quarantined_name = metadata.get("quarantined_name")
        if not isinstance(quarantined_name, str) or not quarantined_name:
            return True
        raw = json.loads((metadata_path.parent / quarantined_name).read_text())
        if not isinstance(raw, dict):
            return True
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return True
    status = str(raw.get("status") or "")
    harness_state = _mapping_field(raw, "harness_state")
    checkout_state = _mapping_field(raw, "checkout_state")
    provider_state = _mapping_field(raw, "provider_state")
    try:
        github_state = load_github_provider_state(
            _mapping_field(provider_state, "github")
        )
    except GitHubError:
        return True
    if status in {item.value for item in TERMINAL_STATUSES}:
        uncertain = any(
            str(value or "")
            in {
                "dispatching",
                "submitted",
                "ambiguous",
                "failed_observing",
                "manual_required",
            }
            for value in (
                raw.get("agent_dispatch_state")
                or harness_state.get("agent_dispatch_state"),
                raw.get("fork_operation_state")
                or github_state.get("fork_operation_state"),
                raw.get("worktree_provision_state")
                or checkout_state.get("worktree_provision_state"),
            )
        )
        fenced = bool(
            raw.get("target_release_pending")
            or harness_state.get("target_release_pending")
            or raw.get("cancellation_reconciliation_pending")
            or raw.get("manual_reconcile_operation")
            or uncertain
        )
        if reservation == "worktree":
            fenced = fenced or bool(
                raw.get("worktree_manual_inspection_required")
                or checkout_state.get("worktree_manual_inspection_required")
                or (
                    raw.get("worktree_provision_state")
                    or checkout_state.get("worktree_provision_state")
                )
                in {"quarantined", "manual_required"}
                or raw.get("pull_request_worktree_state") == "quarantined"
                or github_state.get("pull_request_worktree_state") == "quarantined"
            )
        if not fenced:
            return False
    elif status not in {item.value for item in ACTIVE_STATUSES}:
        # Treat an unknown status as reservation-bearing, but still compare a
        # usable resource identity below rather than blocking unrelated jobs.
        pass
    reserved = (
        raw.get("herdr_target") or raw.get("session_id")
        if reservation == "target"
        else raw.get("worktree_path") or checkout_state.get("path")
    )
    if reserved is None:
        return False
    if not isinstance(reserved, str):
        return True
    return reserved == value


def _validate_quarantine_reservation_unlocked(
    jobs_dir: Path,
    candidate: CursorJob,
    reservation: Literal["target", "worktree"],
) -> None:
    value = (
        candidate.herdr_target if reservation == "target" else candidate.worktree_path
    )
    if not value:
        return
    quarantine = jobs_dir / ".quarantine"
    for metadata_path in sorted(quarantine.glob("*.metadata.json")):
        if _quarantine_may_reserve(
            metadata_path,
            reservation=reservation,
            value=value,
        ):
            raise JobValidationError(
                f"{reservation} reservation {value!r} is blocked by unresolved "
                f"quarantine evidence {metadata_path.name}"
            )


def _quarantine_error(path: Path) -> JobQuarantinedError | None:
    metadata_paths = _quarantine_metadata(path)
    if not metadata_paths:
        return None
    metadata_path = metadata_paths[-1]
    try:
        metadata = json.loads(metadata_path.read_text())
        quarantined_name = str(metadata.get("quarantined_name") or "unknown")
        error = str(metadata.get("error") or "validation failed")
    except (OSError, json.JSONDecodeError):
        quarantined_name = metadata_path.name
        error = "quarantine metadata is unreadable"
    return JobQuarantinedError(
        f"{path.name}: job is quarantined as {quarantined_name}: {error}"
    )


def _parse_path(path: Path) -> CursorJob:
    try:
        raw = json.loads(path.read_text())
    except UnicodeDecodeError as exc:
        raise JobValidationError(f"{path.name}: invalid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise JobValidationError(f"{path.name}: invalid JSON") from exc
    if not isinstance(raw, dict):
        raise JobValidationError(f"{path.name}: job must be a JSON object")
    try:
        return CursorJob.from_dict(raw)
    except JobValidationError as exc:
        raise JobValidationError(f"{path.name}: {exc}") from exc


def _fsync_directory(path: Path) -> None:
    directory = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(value, output, allow_nan=False, sort_keys=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_bytes(path: Path, contents: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _maintenance_path(jobs_dir: Path) -> Path:
    return jobs_dir / MAINTENANCE_FILENAME


def _read_maintenance_unlocked(jobs_dir: Path) -> MaintenanceLease | None:
    path = _maintenance_path(jobs_dir)
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise JobMaintenanceError("job maintenance fence cannot be verified safely")
    try:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("maintenance fence must be an object")
        if (
            raw.get("schema_version") != MAINTENANCE_SCHEMA_VERSION
            or raw.get("operation") != "delete_all"
        ):
            raise ValueError("unsupported maintenance fence")
        token = raw.get("token")
        started_at = raw.get("started_at")
        owner_pid = raw.get("owner_pid")
        owner_boot_id = raw.get("owner_boot_id")
        owner_process_start = raw.get("owner_process_start")
        if (
            not isinstance(token, str)
            or not token
            or not isinstance(started_at, (int, float))
            or isinstance(started_at, bool)
            or not isinstance(owner_pid, int)
            or isinstance(owner_pid, bool)
            or owner_pid <= 0
            or not isinstance(owner_boot_id, str)
            or not owner_boot_id
            or not isinstance(owner_process_start, str)
            or not owner_process_start
        ):
            raise ValueError("invalid maintenance fence identity")
        return MaintenanceLease(
            token=token,
            started_at=float(started_at),
            owner_pid=owner_pid,
            owner_boot_id=owner_boot_id,
            owner_process_start=owner_process_start,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise JobMaintenanceError(
            "job maintenance fence cannot be verified safely"
        ) from exc


def _exclusive_bytes(path: Path, contents: bytes) -> bool:
    """Create immutable contents, accepting only an identical replay."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        temporary.chmod(0o600)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.is_symlink():
                raise JobValidationError(
                    "existing workflow artifact cannot be a symlink"
                ) from None
            try:
                with path.open("rb") as source:
                    existing = source.read(MAX_WORKFLOW_ARTIFACT_BYTES + 1)
            except OSError as exc:
                raise JobValidationError(
                    "existing workflow artifact cannot be verified"
                ) from exc
            if existing != contents:
                raise JobValidationError(
                    "workflow artifact reference already contains different content"
                ) from None
            return False
        _fsync_directory(path.parent)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_path(jobs_dir: Path, reference: str) -> Path:
    match = _ARTIFACT_REF.fullmatch(reference)
    if match is None:
        raise JobValidationError("invalid workflow artifact reference")
    path = jobs_dir / reference
    if path.parent.parent != jobs_dir / ".artifacts":
        raise JobValidationError("workflow artifact escapes artifact storage")
    for parent in (path.parent.parent, path.parent):
        if parent.is_symlink():
            raise JobValidationError("workflow artifact storage cannot be a symlink")
    return path


def _ensure_artifact_directory_unlocked(jobs_dir: Path, job_id: str) -> Path:
    artifacts = jobs_dir / ".artifacts"
    directory = artifacts / job_id
    for candidate, parent in ((artifacts, jobs_dir), (directory, artifacts)):
        if candidate.is_symlink():
            raise JobValidationError("workflow artifact storage cannot be a symlink")
        if candidate.exists():
            if not candidate.is_dir():
                raise JobValidationError(
                    "workflow artifact storage must be a directory"
                )
            continue
        candidate.mkdir(mode=0o700)
        candidate.chmod(0o700)
        _fsync_directory(parent)
    return directory


def _artifact_payload(
    job_id: str,
    kind: Literal["plan", "review"],
    round_number: int,
    text: str,
    *,
    plan_sha256: str | None = None,
) -> tuple[str, bytes]:
    encoded = text.encode()
    if not text.strip():
        raise JobValidationError("workflow artifact text must be non-empty")
    if "\x00" in text:
        raise JobValidationError("workflow artifact contains a NUL byte")
    payload: dict[str, object] = {
        "schema_version": 2,
        "job_id": job_id,
        "kind": kind,
        "round": round_number,
        "text": text,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }
    if kind == "review":
        if plan_sha256 is None:
            raise JobValidationError("workflow review requires a plan digest")
        payload["plan_sha256"] = plan_sha256
    serialized = json.dumps(
        payload, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    if len(serialized) > MAX_WORKFLOW_ARTIFACT_BYTES:
        raise JobValidationError("workflow artifact exceeds size limit")
    digest = hashlib.sha256(serialized).hexdigest()
    reference = f".artifacts/{job_id}/{kind}-{round_number}-{digest}.json"
    return reference, serialized


def _quarantine_artifact(path: Path, error: Exception) -> ArtifactQuarantinedError:
    contents = (
        os.readlink(path).encode()
        if path.is_symlink()
        else path.read_bytes()[: MAX_WORKFLOW_ARTIFACT_BYTES + 1]
    )
    digest = hashlib.sha256(contents).hexdigest()
    quarantine_root = path.parents[2] / ".quarantine"
    quarantine = quarantine_root / "artifacts"
    for candidate, parent in (
        (quarantine_root, path.parents[2]),
        (quarantine, quarantine_root),
    ):
        if candidate.is_symlink() or (candidate.exists() and not candidate.is_dir()):
            return ArtifactQuarantinedError(
                f"{path.name}: workflow artifact is invalid and quarantine "
                "storage is not path-confined"
            )
        if not candidate.exists():
            candidate.mkdir(mode=0o700)
            candidate.chmod(0o700)
            _fsync_directory(parent)
    destination = quarantine / f"{path.parent.name}-{path.stem}-{digest[:12]}.json"
    sequence = 1
    while destination.exists():
        destination = quarantine / (
            f"{path.parent.name}-{path.stem}-{digest[:12]}-{sequence}.json"
        )
        sequence += 1
    metadata_path = destination.with_suffix(".metadata.json")
    _atomic_json(
        metadata_path,
        {
            "original_reference": (f".artifacts/{path.parent.name}/{path.name}"),
            "quarantined_name": destination.name,
            "quarantined_at": time.time(),
            "sha256": digest,
            "error": str(error),
        },
    )
    os.replace(path, destination)
    destination.chmod(0o600)
    _fsync_directory(path.parent)
    _fsync_directory(quarantine)
    return ArtifactQuarantinedError(
        f"{path.name}: workflow artifact is quarantined as {destination.name}: {error}"
    )


def _parse_artifact(
    path: Path,
    *,
    expected_job_id: str,
    expected_kind: str,
    expected_round: int,
    expected_source_sha256: str | None = None,
) -> str:
    try:
        if path.is_symlink():
            raise JobValidationError("workflow artifact cannot be a symlink")
        with path.open("rb") as source:
            payload = source.read(MAX_WORKFLOW_ARTIFACT_BYTES + 1)
        if len(payload) > MAX_WORKFLOW_ARTIFACT_BYTES:
            raise JobValidationError("workflow artifact exceeds size limit")
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise JobValidationError("workflow artifact must be a JSON object")
        if raw.get("job_id") != expected_job_id:
            raise JobValidationError("workflow artifact job identity does not match")
        if raw.get("kind") != expected_kind:
            raise JobValidationError("workflow artifact kind does not match")
        if raw.get("round") != expected_round:
            raise JobValidationError("workflow artifact round does not match")
        text = raw.get("text")
        if not isinstance(text, str) or not text.strip():
            raise JobValidationError("workflow artifact text must be non-empty")
        if "\x00" in text:
            raise JobValidationError("workflow artifact contains a NUL byte")
        expected_digest = hashlib.sha256(text.encode()).hexdigest()
        if raw.get("sha256") != expected_digest:
            raise JobValidationError("workflow artifact digest does not match")
        match = _ARTIFACT_REF.fullmatch(f".artifacts/{expected_job_id}/{path.name}")
        if match is None:
            raise JobValidationError("invalid workflow artifact reference")
        reference_digest = match.group("digest")
        if reference_digest is not None and hashlib.sha256(payload).hexdigest() != (
            reference_digest
        ):
            raise JobValidationError(
                "workflow artifact reference digest does not match"
            )
        if (
            expected_kind == "review"
            and raw.get("plan_sha256") != expected_source_sha256
        ):
            raise JobValidationError(
                "workflow review does not match the current plan digest"
            )
        return text
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        JobValidationError,
    ) as exc:
        if isinstance(exc, FileNotFoundError):
            raise JobValidationError("workflow artifact is missing") from exc
        if path.exists() or path.is_symlink():
            raise _quarantine_artifact(path, exc) from exc
        raise JobValidationError(str(exc)) from exc


def _delete_artifacts_unlocked(jobs_dir: Path, job_id: str) -> None:
    parent = jobs_dir / ".artifacts"
    if parent.is_symlink():
        parent.unlink()
        return
    directory = parent / job_id
    if directory.is_symlink():
        directory.unlink()
    elif directory.is_dir():
        shutil.rmtree(directory)
    if parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()


def _quarantine(path: Path, error: JobValidationError) -> JobQuarantinedError:
    contents = path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    quarantine = path.parent / ".quarantine"
    quarantine.mkdir(mode=0o700, exist_ok=True)
    quarantine.chmod(0o700)
    stem = f"{path.stem}-{digest[:12]}"
    destination = quarantine / f"{stem}.json"
    sequence = 1
    while destination.exists():
        destination = quarantine / f"{stem}-{sequence}.json"
        sequence += 1
    metadata_path = destination.with_suffix(".metadata.json")
    metadata: dict[str, object] = {
        "original_name": path.name,
        "quarantined_name": destination.name,
        "quarantined_at": time.time(),
        "sha256": digest,
        "error": str(error),
    }
    # Publish reservation-bearing evidence before moving the malformed payload.
    # A crash in the gap therefore fails closed; a later scan can retry the move.
    _atomic_json(metadata_path, metadata)
    os.replace(path, destination)
    destination.chmod(0o600)
    _fsync_directory(path.parent)
    _fsync_directory(quarantine)
    quarantined = JobQuarantinedError(
        f"{path.name}: job is quarantined as {destination.name}: {error}"
    )
    warnings.warn(str(quarantined), JobQuarantineWarning, stacklevel=2)
    return quarantined


def _quarantine_import(
    source: Path,
    jobs_dir: Path,
    error: JobValidationError,
    *,
    remove_source: bool = True,
) -> JobQuarantinedError:
    contents = source.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    quarantine = jobs_dir / ".quarantine"
    quarantine.mkdir(mode=0o700, parents=True, exist_ok=True)
    quarantine.chmod(0o700)
    stem = f"{source.stem}-legacy-import-{digest[:12]}"
    destination = quarantine / f"{stem}.json"
    sequence = 1
    while destination.exists() and destination.read_bytes() != contents:
        destination = quarantine / f"{stem}-{sequence}.json"
        sequence += 1
    if not destination.exists():
        _atomic_bytes(destination, contents)
        destination.chmod(0o600)
    metadata_path = destination.with_suffix(".metadata.json")
    if not metadata_path.exists():
        _atomic_json(
            metadata_path,
            {
                "original_name": source.name,
                "legacy_source": str(source),
                "quarantined_name": destination.name,
                "quarantined_at": time.time(),
                "sha256": digest,
                "error": str(error),
            },
        )
    if remove_source:
        source.unlink(missing_ok=True)
        _fsync_directory(source.parent)
    quarantined = JobQuarantinedError(
        f"{source.name}: legacy import is quarantined as {destination.name}: {error}"
    )
    warnings.warn(str(quarantined), JobQuarantineWarning, stacklevel=2)
    return quarantined


def _read_model_unlocked(path: Path) -> CursorJob:
    if not path.exists():
        quarantined = _quarantine_error(path)
        if quarantined is not None:
            raise quarantined
    try:
        return _parse_path(path)
    except JobValidationError as exc:
        if not path.exists():
            quarantined = _quarantine_error(path)
            if quarantined is not None:
                raise quarantined from exc
        raise _quarantine(path, exc) from exc


def read_unlocked(path: Path) -> dict[str, object]:
    model = _read_model_unlocked(path)
    raw = json.loads(path.read_text())
    assert isinstance(raw, dict)
    persisted_version = raw.get("schema_version", 0)
    return model.to_dict(
        preserve_loaded_version=(
            isinstance(persisted_version, int)
            and not isinstance(persisted_version, bool)
            and persisted_version < CURRENT_SCHEMA_VERSION
        )
    )


def _peer_models_unlocked(path: Path) -> list[CursorJob]:
    peers: list[CursorJob] = []
    for peer_path in sorted(path.parent.glob("*.json")):
        if peer_path == path:
            continue
        try:
            peers.append(_read_model_unlocked(peer_path))
        except JobQuarantinedError:
            continue
    return peers


def _validate_candidate_reservations_unlocked(
    path: Path,
    candidate: CursorJob,
    previous: CursorJob | None,
) -> None:
    # Scan first so malformed peers become quarantine evidence before either
    # live or quarantined reservations are evaluated.
    peers = _peer_models_unlocked(path)
    candidate_reserves = _job_reserves_resources(candidate)
    previous_reserves = previous is not None and _job_reserves_resources(previous)
    if candidate_reserves and (
        not previous_reserves
        or previous is None
        or candidate.herdr_target != previous.herdr_target
    ):
        _validate_quarantine_reservation_unlocked(path.parent, candidate, "target")
    if candidate_reserves and (
        not previous_reserves
        or previous is None
        or candidate.worktree_path != previous.worktree_path
    ):
        _validate_quarantine_reservation_unlocked(path.parent, candidate, "worktree")
    validate_reservations([*peers, candidate])


def _validate_candidate_artifacts_unlocked(
    path: Path, candidate: CursorJob, previous: CursorJob | None
) -> None:
    for kind, reference in (
        ("plan", candidate.plan_artifact),
        ("review", candidate.review_artifact),
    ):
        if reference is None:
            continue
        if (
            previous is not None
            and previous.to_dict().get(f"{kind}_artifact") == reference
        ):
            # Existing references were validated when published. If later
            # corruption quarantines one, unrelated recovery/failure writes
            # must still be able to terminalize the owning job.
            continue
        match = _ARTIFACT_REF.fullmatch(reference)
        if match is None:
            raise JobValidationError("invalid workflow artifact reference")
        source_sha256 = None
        if kind == "review":
            plan_reference = candidate.plan_artifact
            if plan_reference is None:
                raise JobValidationError("workflow review has no current plan")
            plan_match = _ARTIFACT_REF.fullmatch(plan_reference)
            assert plan_match is not None
            plan_text = _parse_artifact(
                _artifact_path(path.parent, plan_reference),
                expected_job_id=candidate.id,
                expected_kind="plan",
                expected_round=int(plan_match.group("round")),
            )
            source_sha256 = hashlib.sha256(plan_text.encode()).hexdigest()
        _parse_artifact(
            _artifact_path(path.parent, reference),
            expected_job_id=candidate.id,
            expected_kind=kind,
            expected_round=int(match.group("round")),
            expected_source_sha256=source_sha256,
        )


def _job_reserves_resources(job: CursorJob) -> bool:
    return bool(
        job.status in ACTIVE_STATUSES
        or job.target_release_pending
        or job.cancellation_reconciliation_pending
        or job.has_uncertain_operation()
        or job.manual_reconcile_operation is not None
        or job.worktree_manual_inspection_required
        or job.worktree_provision_state in {"quarantined", "manual_required"}
        or job.pull_request_worktree_state == "quarantined"
    )


def _ticket_identity(job: CursorJob) -> tuple[str, ...] | None:
    if (
        not job.fork_requested
        and job.github_pull_request is None
        and job.github_repository
        and job.github_issue
    ):
        return (
            "github",
            job.github_repository.casefold(),
            str(job.github_issue),
        )
    if job.issue_key:
        return (job.issue_provider or "legacy", job.issue_key.casefold())
    return None


def _grouped_ticket_identities(job: CursorJob) -> tuple[tuple[str, ...], ...]:
    identities: list[tuple[str, ...]] = []
    for target in job.grouped_repository_targets or []:
        request = target.get("request")
        if not isinstance(request, dict):
            continue
        issue_key = request.get("issue_key")
        if isinstance(issue_key, str) and issue_key:
            identities.append(("linear", issue_key.casefold()))
    return tuple(identities)


def _active_ticket_conflict_unlocked(
    path: Path,
    candidate: CursorJob,
) -> CursorJob | None:
    identity = _ticket_identity(candidate)
    if identity is None:
        return None
    return next(
        (
            peer
            for peer in _peer_models_unlocked(path)
            if _job_reserves_resources(peer) and _ticket_identity(peer) == identity
        ),
        None,
    )


def _has_legacy_worker_claim(job: CursorJob) -> bool:
    ownership = (
        job.worker_token,
        job.worker_pid,
        job.worker_boot_id,
        job.worker_process_start,
    )
    return any(value is not None for value in ownership) and (
        not all(value is not None for value in ownership)
        or job.worker_boot_id == "legacy-unknown"
    )


def _normalize_for_durable_write(
    candidate: CursorJob,
    *,
    legacy_worker_disposition: LegacyWorkerDisposition | None = None,
) -> CursorJob:
    values = candidate.to_dict()
    legacy_unowned_active = (
        candidate.loaded_schema_version < CURRENT_SCHEMA_VERSION
        and candidate.status
        in {JobStatus.ROUTING, JobStatus.RUNNING, JobStatus.RECONCILING}
        and candidate.terminal_intent_status is None
        and not any(
            value is not None
            for value in (
                candidate.worker_token,
                candidate.worker_pid,
                candidate.worker_boot_id,
                candidate.worker_process_start,
            )
        )
    )
    if _has_legacy_worker_claim(candidate) or legacy_unowned_active:
        if candidate.loaded_schema_version >= CURRENT_SCHEMA_VERSION:
            raise JobValidationError(
                f"{candidate.status.value} job requires complete current worker ownership"
            )
        active = candidate.status in {
            JobStatus.QUEUED,
            JobStatus.ROUTING,
            JobStatus.RUNNING,
            JobStatus.RECONCILING,
        }
        safely_cleared = legacy_unowned_active or legacy_worker_disposition in {
            "absent",
            "stopped",
        }
        if active and legacy_worker_disposition is None and not legacy_unowned_active:
            # Keep an unresolved imported owner as explicit canonical evidence.
            # Cancellation/nuke must inspect this fence before releasing it.
            values.update(
                worker_token=candidate.worker_token or LEGACY_BOOT_ID,
                worker_pid=candidate.worker_pid or 1,
                worker_boot_id=LEGACY_BOOT_ID,
                worker_process_start=(candidate.worker_process_start or LEGACY_BOOT_ID),
                worker_claim_operation=(
                    values.get("worker_claim_operation") or "legacy_import"
                ),
                worker_claimed_at=values.get("worker_claimed_at")
                or candidate.created_at,
            )
        else:
            values.update(
                worker_token=None,
                worker_pid=None,
                worker_boot_id=None,
                worker_process_start=None,
                worker_claim_operation=None,
                worker_claimed_at=None,
            )
        if active and safely_cleared:
            values.update(
                status=JobStatus.QUEUED.value,
                queued_at=candidate.queued_at or candidate.created_at,
                reconcile=bool(
                    candidate.herdr_target or candidate.has_uncertain_operation()
                ),
            )
        elif active and legacy_worker_disposition is not None:
            message = (
                "Legacy Cursor worker ownership could not be verified safely; "
                "manual recovery is required."
            )
            values.update(
                status=JobStatus.FAILED.value,
                error=message,
                result=message,
                completed_at=time.time(),
                delivered=False,
            )
    if values.get("target_release_owner_boot_id") == "legacy-unknown":
        values.update(
            target_release_owner_pid=None,
            target_release_owner_boot_id=None,
            target_release_owner_start=None,
        )
    if candidate.loaded_schema_version < CURRENT_SCHEMA_VERSION:
        if (
            values.get("agent_identity_legacy_compatible")
            and values.get("agent_dispatch_state") in {"ready", "retained"}
            and not (
                values.get("agent_provider") and values.get("agent_provider_session_id")
            )
        ):
            values["agent_dispatch_state"] = "ambiguous"
        if (
            values.get("agent_dispatch_state") is None
            and values.get("herdr_target")
            and values.get("herdr_workspace_id")
            and values.get("herdr_pane_id")
        ):
            raw_owners = values.get("participant_session_owners")
            owners = list(raw_owners) if isinstance(raw_owners, list) else []
            if not any(
                isinstance(owner, dict)
                and owner.get("target") == values["herdr_target"]
                for owner in owners
            ):
                owners.append(
                    {
                        "provider": str(values.get("harness_kind") or "cursor"),
                        "session_id": str(
                            values.get("session_id") or values["herdr_target"]
                        ),
                        "target": str(values["herdr_target"]),
                        "state_sequence": 0,
                        "checkout": str(
                            values.get("worktree_path")
                            or values.get("repository")
                            or "legacy-unknown"
                        ),
                        "workspace_id": str(values["herdr_workspace_id"]),
                        "pane_id": str(values["herdr_pane_id"]),
                    }
                )
                values["participant_session_owners"] = owners
        workspace_id = values.get("worktree_workspace_id")
        root_pane_id = values.get("worktree_root_pane_id")
        if bool(workspace_id) != bool(root_pane_id):
            values.update(
                worktree_workspace_id=workspace_id or LEGACY_BOOT_ID,
                worktree_root_pane_id=root_pane_id or LEGACY_BOOT_ID,
            )
        if values.get("worktree_provision_state") in {"ready", "retained"} and (
            not workspace_id or not root_pane_id
        ):
            values["worktree_provision_state"] = "ambiguous"
    for field in (
        "migration_source_schema_version",
        "phase_prompt_active",
        "agent_identity_legacy_compatible",
    ):
        values.pop(field, None)
    values["schema_version"] = CURRENT_SCHEMA_VERSION
    if candidate._compatibility_layout:
        values.pop("harness_kind", None)
        values.pop("session_id", None)
    normalized = CursorJob.from_dict(values)
    normalized.validate_invariants(require_worker_owner=True)
    return replace(normalized, _lifecycle_event=candidate._lifecycle_event)


def write_unlocked(path: Path, job: dict[str, object]) -> None:
    previous = _read_model_unlocked(path) if path.exists() else None
    try:
        parsed = CursorJob.from_dict(job)
        candidate = _normalize_for_durable_write(parsed)
        if candidate.id != path.stem:
            raise JobValidationError("Cursor job id must match its filename")
        if previous is None:
            if candidate.revision != 0:
                raise JobValidationError("new Cursor job revision must be zero")
        else:
            validate_transition(previous, candidate)
        _validate_candidate_artifacts_unlocked(path, candidate, previous)
        _validate_candidate_reservations_unlocked(path, candidate, previous)
    except JobValidationError as exc:
        if isinstance(exc, JobQuarantinedError):
            raise
        raise JobValidationError(f"{path.name}: {exc}") from exc
    _atomic_json(path, candidate.to_record())


def _write_model_unlocked(path: Path, candidate: CursorJob) -> CursorJob:
    previous = _read_model_unlocked(path) if path.exists() else None
    try:
        candidate = _normalize_for_durable_write(candidate)
        if candidate.id != path.stem:
            raise JobValidationError("Cursor job id must match its filename")
        if previous is None:
            if candidate.revision != 0:
                raise JobValidationError("new Cursor job revision must be zero")
        else:
            validate_transition(previous, candidate)
        _validate_candidate_artifacts_unlocked(path, candidate, previous)
        _validate_candidate_reservations_unlocked(path, candidate, previous)
    except JobValidationError as exc:
        if isinstance(exc, JobQuarantinedError):
            raise
        raise JobValidationError(f"{path.name}: {exc}") from exc
    _atomic_json(path, candidate.to_record())
    return candidate


def read_all_unlocked(jobs_dir: Path) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for path in sorted(jobs_dir.glob("*.json")):
        try:
            jobs.append(read_unlocked(path))
        except JobQuarantinedError:
            continue
    return jobs


def migrate_legacy_jobs(
    legacy_dir: Path,
    jobs_dir: Path,
    *,
    inspect_worker: LegacyWorkerInspector | None = None,
) -> set[str]:
    """Import runtime-era job JSON into the durable store exactly once."""
    blocked: set[str] = set()
    if not legacy_dir.is_dir() or legacy_dir.resolve() == jobs_dir.resolve():
        return blocked
    with locked(legacy_dir), locked(jobs_dir):
        if _read_maintenance_unlocked(jobs_dir) is not None:
            return blocked
        for source in sorted(legacy_dir.glob("*.json")):
            destination = jobs_dir / source.name
            preserve_source = False
            try:
                candidate = _parse_path(source)
                preserve_source = _has_legacy_worker_claim(candidate)
                if candidate.id != source.stem:
                    raise JobValidationError("Cursor job id must match its filename")
                existing: CursorJob | None = None
                if destination.exists():
                    try:
                        existing = _read_model_unlocked(destination)
                    except JobQuarantinedError:
                        existing = None
                if existing is not None:
                    if (
                        existing.id != candidate.id
                        or existing.created_at != candidate.created_at
                    ):
                        raise JobValidationError(
                            "legacy import identity/created_at lineage conflicts "
                            "with the durable job"
                        )
                disposition: LegacyWorkerDisposition | None = None
                if _has_legacy_worker_claim(candidate):
                    disposition = (
                        inspect_worker(candidate)
                        if inspect_worker is not None
                        else "unsafe"
                    )
                    if disposition == "unsafe":
                        blocked.add(candidate.id)
                        _quarantine_import(
                            source,
                            jobs_dir,
                            JobValidationError(
                                "active legacy worker could not be stopped safely; "
                                "manual recovery is required"
                            ),
                            remove_source=False,
                        )
                        continue
                candidate = _normalize_for_durable_write(
                    candidate,
                    legacy_worker_disposition=disposition,
                )
                if existing is not None:
                    if existing.revision > candidate.revision:
                        source.unlink()
                        _fsync_directory(legacy_dir)
                        continue
                    if existing.revision == candidate.revision:
                        if existing.to_record() != candidate.to_record():
                            raise JobValidationError(
                                "legacy import conflicts with durable job at the "
                                "same revision"
                            )
                        source.unlink()
                        _fsync_directory(legacy_dir)
                        continue
                _validate_candidate_reservations_unlocked(
                    destination,
                    candidate,
                    existing,
                )
                _atomic_json(destination, candidate.to_record())
                source.unlink()
                _fsync_directory(legacy_dir)
            except JobValidationError as error:
                if source.exists():
                    _quarantine_import(
                        source,
                        jobs_dir,
                        error,
                        remove_source=not preserve_source,
                    )
    return blocked


def _must_retain(job: CursorJob) -> bool:
    values = job.to_dict()
    return bool(
        job.has_uncertain_operation()
        or job.target_release_pending
        or job.cancellation_reconciliation_pending
        or job.manual_reconcile_operation
        or values.get("worktree_manual_inspection_required")
        or values.get("worktree_provision_state") in {"quarantined", "manual_required"}
        or values.get("pull_request_worktree_state") == "quarantined"
    )


def prune_jobs(
    jobs_dir: Path,
    *,
    now: float | None = None,
    retention_seconds: float = DELIVERED_JOB_RETENTION_SECONDS,
) -> list[str]:
    """Remove only old, delivered terminal jobs without retained fences."""
    if not jobs_dir.is_dir():
        return []
    cutoff = (time.time() if now is None else now) - retention_seconds
    removed: list[str] = []
    with locked(jobs_dir):
        if _read_maintenance_unlocked(jobs_dir) is not None:
            return removed
        for path in sorted(jobs_dir.glob("*.json")):
            try:
                job = _read_model_unlocked(path)
            except JobQuarantinedError:
                continue
            completed_at = job.completed_at
            if (
                job.status in TERMINAL_STATUSES
                and job.delivered
                and completed_at is not None
                and completed_at < cutoff
                and not _must_retain(job)
            ):
                path.unlink()
                _delete_artifacts_unlocked(jobs_dir, job.id)
                removed.append(job.id)
        if removed:
            _fsync_directory(jobs_dir)
    return removed


def _validate_follow_up_source(
    parent: CursorJob, expected_completed_at: float | None
) -> None:
    """Reject a follow-up whose parent is not a safe, isolated completed job."""
    if parent.status != JobStatus.COMPLETED:
        raise FollowUpUnavailable(
            f"Cursor job {parent.id} is not completed and cannot be followed up"
        )
    if (
        expected_completed_at is not None
        and parent.completed_at != expected_completed_at
    ):
        raise FollowUpUnavailable(
            f"Cursor job {parent.id} completion identity has changed"
        )
    if not (
        parent.repository
        and parent.worktree_branch
        and parent.worktree_path
        and parent.worktree_workspace_id
        and parent.worktree_root_pane_id
    ):
        raise FollowUpUnavailable(
            f"Cursor job {parent.id} has no complete isolated workspace to reuse"
        )
    if parent.worktree_provision_state not in {"ready", "retained"}:
        raise FollowUpUnavailable(
            f"Cursor job {parent.id} worktree is not in a reusable state"
        )
    if (
        parent.has_uncertain_operation()
        or parent.manual_reconcile_operation
        or parent.target_release_pending
        or parent.cancellation_reconciliation_pending
    ):
        raise FollowUpUnavailable(f"Cursor job {parent.id} is still being reconciled")
    try:
        repository = Path(parent.repository).resolve()
        checkout = Path(parent.worktree_path).resolve()
    except OSError as exc:
        raise FollowUpUnavailable(
            f"Cursor job {parent.id} checkout path is invalid"
        ) from exc
    if checkout == repository:
        raise FollowUpUnavailable(
            f"Cursor job {parent.id} used the shared repository clone"
        )


def _validate_follow_up_event(
    parent: CursorJob,
    child: CursorJob,
    expected_parent_revision: int,
    expected_completed_at: float | None,
) -> None:
    child_lifecycle = child.lifecycle
    if not isinstance(child_lifecycle, QueuedJob):
        raise JobValidationError("follow-up child must begin queued")
    assert parent.completed_at is not None
    try:
        apply_follow_up(
            parent.lifecycle,
            FollowUpEvent(
                expected_parent_revision,
                (
                    expected_completed_at
                    if expected_completed_at is not None
                    else parent.completed_at
                ),
                child_lifecycle,
            ),
        )
    except JobLifecycleError as exc:
        raise JobValidationError(str(exc)) from exc


JobCommand = Callable[[CursorJob], CursorJob | None]
FollowUpBuilder = Callable[[CursorJob], CursorJob]
ArtifactCommand = Callable[[CursorJob, str], CursorJob]


def _worker_claim_matches(job: CursorJob, expected: WorkerOwnership | str) -> bool:
    if isinstance(expected, WorkerOwnership):
        try:
            return expected.matches(job.worker_ownership)
        except JobValidationError:
            return False
    return False


class JsonJobStore:
    """Typed transaction boundary for durable Cursor jobs."""

    def __init__(self, durable_dir: Path, legacy_dir: Path) -> None:
        self.durable_dir = durable_dir
        self.legacy_dir = legacy_dir

    def path(self, job_id: str) -> Path:
        if len(job_id) != 12 or any(
            character not in "0123456789abcdef" for character in job_id
        ):
            raise JobValidationError("invalid Cursor job ID")
        return self.durable_dir / f"{job_id}.json"

    @contextmanager
    def _locked_legacy_and_durable(self) -> Iterator[None]:
        if self.legacy_dir.resolve() == self.durable_dir.resolve():
            with locked(self.durable_dir):
                yield
            return
        with locked(self.legacy_dir), locked(self.durable_dir):
            yield

    def migrate_legacy(
        self, *, inspect_worker: LegacyWorkerInspector | None = None
    ) -> set[str]:
        return migrate_legacy_jobs(
            self.legacy_dir,
            self.durable_dir,
            inspect_worker=inspect_worker,
        )

    def get(self, job_id: str) -> CursorJob:
        with locked(self.durable_dir):
            return _read_model_unlocked(self.path(job_id))

    def get_unless_maintenance(self, job_id: str) -> CursorJob | None:
        """Read a worker snapshot only while no maintenance fence is active."""
        with locked(self.durable_dir):
            if _read_maintenance_unlocked(self.durable_dir) is not None:
                return None
            return _read_model_unlocked(self.path(job_id))

    def list(self) -> list[CursorJob]:
        if not self.durable_dir.is_dir():
            return []
        with locked(self.durable_dir):
            jobs: list[CursorJob] = []
            for path in sorted(self.durable_dir.glob("*.json")):
                try:
                    jobs.append(_read_model_unlocked(path))
                except JobQuarantinedError:
                    continue
            return jobs

    def maintenance_active(self) -> bool:
        if not self.durable_dir.is_dir():
            return False
        with locked(self.durable_dir):
            return _read_maintenance_unlocked(self.durable_dir) is not None

    def begin_maintenance(
        self,
        lease: MaintenanceLease,
        stage: JobCommand,
        *,
        owner_alive: MaintenanceOwnerAlive,
    ) -> list[CursorJob]:
        """Install a durable deletion fence and stage every job under one lock."""
        with locked(self.durable_dir):
            existing = _read_maintenance_unlocked(self.durable_dir)
            if existing is not None:
                disposition = owner_alive(existing)
                if disposition is True:
                    raise JobMaintenanceError(
                        "another Cursor job deletion is already in progress"
                    )
                if disposition is None:
                    raise JobMaintenanceError(
                        "an existing Cursor job deletion owner cannot be "
                        "verified; retry after checking the recorded process"
                    )
            unresolved_quarantine = _unresolved_quarantine_metadata(self.durable_dir)
            if unresolved_quarantine:
                raise JobMaintenanceError(
                    "Cursor jobs could not be deleted safely: quarantine evidence "
                    "requires manual inspection: "
                    + ", ".join(path.name for path in unresolved_quarantine)
                )
            _atomic_json(_maintenance_path(self.durable_dir), lease.to_record())
            staged: list[CursorJob] = []
            for path in sorted(self.durable_dir.glob("*.json")):
                try:
                    current = _read_model_unlocked(path)
                except JobQuarantinedError as exc:
                    raise JobMaintenanceError(
                        "Cursor jobs could not be deleted safely: quarantined "
                        f"record {path.stem} requires manual inspection"
                    ) from exc
                candidate = stage(current)
                if candidate is not None:
                    current = _write_model_unlocked(path, candidate)
                staged.append(current)
            return staged

    def abort_maintenance(self, token: str) -> bool:
        """Remove only the maintenance fence owned by ``token``."""
        if not self.durable_dir.is_dir():
            return False
        with locked(self.durable_dir):
            existing = _read_maintenance_unlocked(self.durable_dir)
            if existing is None or existing.token != token:
                return False
            _maintenance_path(self.durable_dir).unlink()
            _fsync_directory(self.durable_dir)
            return True

    def create(
        self,
        job: CursorJob,
        *,
        enforce_unique_ticket: bool = False,
    ) -> CursorJob:
        with locked(self.durable_dir):
            if _read_maintenance_unlocked(self.durable_dir) is not None:
                raise JobMaintenanceError(
                    "Cursor jobs are temporarily unavailable during job deletion"
                )
            path = self.path(job.id)
            if path.exists():
                raise JobValidationError(f"{path.name}: Cursor job already exists")
            if enforce_unique_ticket:
                conflict = _active_ticket_conflict_unlocked(path, job)
                if conflict is not None:
                    raise ActiveTicketConflict(conflict.id)
            created = _write_model_unlocked(path, job)
        return created

    def write_artifact(
        self,
        job_id: str,
        kind: Literal["plan", "review"],
        round_number: int,
        text: str,
        *,
        source_text: str | None = None,
    ) -> str:
        """Create an immutable artifact without publishing a job reference.

        Production worker output must use :meth:`publish_artifact`, which
        verifies the worker/turn workflow fence under the same directory lock.
        This lower-level helper exists for migration and test fixture setup.
        """
        if kind not in WORKFLOW_ARTIFACT_KINDS:
            raise JobValidationError("invalid workflow artifact kind")
        if round_number < 0 or round_number > 2:
            raise JobValidationError(
                "workflow artifact round must be between zero and two"
            )
        plan_sha256 = None
        if kind == "review":
            if source_text is None or not source_text.strip():
                raise JobValidationError(
                    "workflow review requires the reviewed plan text"
                )
            plan_sha256 = hashlib.sha256(source_text.encode()).hexdigest()
        reference, serialized = _artifact_payload(
            job_id,
            kind,
            round_number,
            text,
            plan_sha256=plan_sha256,
        )
        path = _artifact_path(self.durable_dir, reference)
        with locked(self.durable_dir):
            _read_model_unlocked(self.path(job_id))
            _ensure_artifact_directory_unlocked(self.durable_dir, job_id)
            _exclusive_bytes(path, serialized)
        return reference

    def publish_artifact(
        self,
        job_id: str,
        kind: Literal["plan", "review"],
        round_number: int,
        text: str,
        *,
        expected_worker_token: WorkerOwnership | str,
        expected_revision: int,
        expected_turn_token: str,
        expected_phase: str,
        expected_prior_reference: str | None,
        change: ArtifactCommand,
        expected_plan_reference: str | None = None,
    ) -> CursorJob | None:
        """Guard, create, and publish immutable worker output under one lock.

        Returning ``None`` means the worker snapshot is stale. No artifact
        directory or sidecar is created before every workflow fence matches.
        If a crash leaves an unreferenced sidecar after creation, an identical
        replay reuses it; different bytes can never replace it.
        """
        if kind not in WORKFLOW_ARTIFACT_KINDS:
            raise JobValidationError("invalid workflow artifact kind")
        if round_number < 0 or round_number > 2:
            raise JobValidationError(
                "workflow artifact round must be between zero and two"
            )
        with locked(self.durable_dir):
            path = self.path(job_id)
            current = _read_model_unlocked(path)
            field = "plan_artifact" if kind == "plan" else "review_artifact"
            if (
                current.terminal_intent_status is not None
                or current.revision != expected_revision
                or not _worker_claim_matches(current, expected_worker_token)
                or current.turn_token != expected_turn_token
                or current.workflow_phase.value != expected_phase
                or current.review_round != round_number
                or current.to_dict().get(field) != expected_prior_reference
            ):
                return None

            plan_sha256 = None
            if kind == "review":
                if (
                    expected_plan_reference is None
                    or current.plan_artifact != expected_plan_reference
                ):
                    return None
                plan_match = _ARTIFACT_REF.fullmatch(expected_plan_reference)
                if plan_match is None:
                    raise JobValidationError("invalid reviewed plan artifact reference")
                plan_text = _parse_artifact(
                    _artifact_path(self.durable_dir, expected_plan_reference),
                    expected_job_id=job_id,
                    expected_kind="plan",
                    expected_round=int(plan_match.group("round")),
                )
                plan_sha256 = hashlib.sha256(plan_text.encode()).hexdigest()

            reference, serialized = _artifact_payload(
                job_id,
                kind,
                round_number,
                text,
                plan_sha256=plan_sha256,
            )
            artifact_path = _artifact_path(self.durable_dir, reference)
            _ensure_artifact_directory_unlocked(self.durable_dir, job_id)
            _exclusive_bytes(artifact_path, serialized)
            candidate = change(current, reference)
            if isinstance(expected_worker_token, WorkerOwnership):
                event = WorkerCallbackEvent(
                    expected_revision,
                    candidate.lifecycle,
                    expected_worker_token,
                )
                current.validate_lifecycle_event(candidate, event)
                candidate = replace(candidate, _lifecycle_event=event)
            if candidate.to_dict().get(field) != reference:
                raise JobValidationError(
                    f"artifact publication must set {field} to the new reference"
                )
            return _write_model_unlocked(path, candidate)

    def read_artifact(
        self,
        job_id: str,
        reference: str,
        *,
        kind: Literal["plan", "review"],
    ) -> str:
        """Load and validate a workflow artifact referenced by a job."""
        path = _artifact_path(self.durable_dir, reference)
        match = _ARTIFACT_REF.fullmatch(reference)
        assert match is not None
        if match.group("job") != job_id or match.group("kind") != kind:
            raise JobValidationError("workflow artifact reference does not match job")
        with locked(self.durable_dir):
            job = _read_model_unlocked(self.path(job_id))
            field = "plan_artifact" if kind == "plan" else "review_artifact"
            if job.to_dict().get(field) != reference:
                raise JobValidationError("workflow artifact reference is stale")
            source_sha256 = None
            if kind == "review":
                plan_reference = job.plan_artifact
                if plan_reference is None:
                    raise JobValidationError("workflow review has no current plan")
                plan_match = _ARTIFACT_REF.fullmatch(plan_reference)
                assert plan_match is not None
                plan_text = _parse_artifact(
                    _artifact_path(self.durable_dir, plan_reference),
                    expected_job_id=job_id,
                    expected_kind="plan",
                    expected_round=int(plan_match.group("round")),
                )
                source_sha256 = hashlib.sha256(plan_text.encode()).hexdigest()
            return _parse_artifact(
                path,
                expected_job_id=job_id,
                expected_kind=kind,
                expected_round=int(match.group("round")),
                expected_source_sha256=source_sha256,
            )

    def update(self, job_id: str, command: JobCommand) -> CursorJob | None:
        return self._transaction(job_id, command)

    def create_follow_up(
        self,
        parent_job_id: str,
        build: FollowUpBuilder,
        *,
        expected_parent_revision: int,
        expected_completed_at: float | None = None,
    ) -> CursorJob:
        """Atomically create a child job that reuses a completed parent checkout.

        The parent is re-read under the directory lock and validated as an
        isolated, completed, reconciled job. The child is written as a fresh
        queued job whose active worktree reservation guarantees a single winner
        when several follow-ups race for the same checkout. The parent record is
        never written.
        """
        with locked(self.durable_dir):
            if _read_maintenance_unlocked(self.durable_dir) is not None:
                raise JobMaintenanceError(
                    "Cursor follow-ups are temporarily unavailable during job deletion"
                )
            try:
                parent = _read_model_unlocked(self.path(parent_job_id))
            except (FileNotFoundError, JobQuarantinedError) as exc:
                raise FollowUpUnavailable(
                    f"Cursor job {parent_job_id} is no longer available"
                ) from exc
            _validate_follow_up_source(parent, expected_completed_at)
            child = build(parent)
            _validate_follow_up_event(
                parent, child, expected_parent_revision, expected_completed_at
            )
            if child.parent_job_id != parent.id:
                raise JobValidationError(
                    "follow-up child must reference its parent job id"
                )
            if child.harness_kind != parent.harness_kind:
                raise JobValidationError(
                    "follow-up child must inherit parent harness_kind exactly"
                )
            if child.issue_provider != parent.issue_provider:
                raise JobValidationError(
                    "follow-up child must inherit parent issue_provider exactly"
                )
            for field in (
                "repository",
                "worktree_branch",
                "worktree_path",
                "worktree_workspace_id",
                "worktree_root_pane_id",
            ):
                if getattr(child, field) != getattr(parent, field):
                    raise JobValidationError(
                        f"follow-up child must inherit parent {field} exactly"
                    )
            path = self.path(child.id)
            if path.exists():
                raise JobValidationError(f"{path.name}: Cursor job already exists")
            try:
                # Force malformed peers into quarantine before checking their
                # evidence. The directory lock prevents a new peer appearing
                # between this scan and the child write.
                _peer_models_unlocked(path)
                _validate_quarantine_reservation_unlocked(
                    self.durable_dir, child, "worktree"
                )
                return _write_model_unlocked(path, child)
            except JobValidationError as exc:
                if "reserved by both" in str(
                    exc
                ) or "blocked by unresolved quarantine evidence" in str(exc):
                    raise FollowUpCheckoutBusy(
                        f"{parent.worktree_path} is busy with another Cursor job"
                    ) from exc
                raise

    def reserve_target(self, job_id: str, command: JobCommand) -> CursorJob | None:
        """Atomically apply a typed target-reservation transition."""
        return self._transaction(
            job_id, command, reservation="target", reject_maintenance=True
        )

    def reserve_worktree(self, job_id: str, command: JobCommand) -> CursorJob | None:
        """Atomically apply a typed worktree-reservation transition."""
        return self._transaction(
            job_id, command, reservation="worktree", reject_maintenance=True
        )

    def update_unless_maintenance(
        self, job_id: str, command: JobCommand
    ) -> CursorJob | None:
        """Apply an ownership or side-effect acquisition unless fenced."""
        return self._transaction(job_id, command, reject_maintenance=True)

    def _transaction(
        self,
        job_id: str,
        command: JobCommand,
        *,
        reservation: Literal["target", "worktree"] | None = None,
        reject_maintenance: bool = False,
    ) -> CursorJob | None:
        with locked(self.durable_dir):
            if (
                reject_maintenance
                and _read_maintenance_unlocked(self.durable_dir) is not None
            ):
                return None
            path = self.path(job_id)
            current = _read_model_unlocked(path)
            candidate = command(current)
            if candidate is None:
                return None
            return _write_model_unlocked(path, candidate)

    def acknowledge_quarantine_reservations(
        self,
        job_id: str,
        *,
        reason: str,
        now: float | None = None,
    ) -> QuarantineAcknowledgement:
        if not reason.strip():
            raise JobValidationError("quarantine acknowledgement reason is required")
        acknowledged_at = time.time() if now is None else now
        with locked(self.durable_dir):
            metadata_paths = _quarantine_metadata(self.path(job_id))
            if not metadata_paths:
                raise JobValidationError(
                    f"{job_id}: no quarantine evidence is available to acknowledge"
                )
            resolved: list[str] = []
            for metadata_path in metadata_paths:
                if _quarantine_metadata_resolved(metadata_path):
                    continue
                if metadata_path.is_symlink():
                    raise JobValidationError(
                        f"{metadata_path.name}: quarantine metadata cannot be a symlink"
                    )
                _atomic_json(
                    _quarantine_resolution_path(metadata_path),
                    {
                        "job_id": job_id,
                        "metadata_name": metadata_path.name,
                        "metadata_sha256": hashlib.sha256(
                            metadata_path.read_bytes()
                        ).hexdigest(),
                        "acknowledged_at": acknowledged_at,
                        "reason": reason.strip(),
                    },
                )
                resolved.append(metadata_path.name)
        return QuarantineAcknowledgement(
            job_id,
            tuple(resolved),
            acknowledged_at,
        )

    def list_quarantine_evidence(
        self, *, include_resolved: bool = False
    ) -> list[QuarantineEvidence]:
        if not self.durable_dir.is_dir():
            return []
        with locked(self.durable_dir):
            evidence = [
                _quarantine_evidence_unlocked(path)
                for path in sorted(
                    (self.durable_dir / ".quarantine").glob("*.metadata.json")
                )
            ]
        return [
            record for record in evidence if include_resolved or not record.resolved
        ]

    def prune(
        self,
        *,
        now: float | None = None,
        retention_seconds: float = DELIVERED_JOB_RETENTION_SECONDS,
    ) -> list[str]:
        return prune_jobs(
            self.durable_dir,
            now=now,
            retention_seconds=retention_seconds,
        )

    def finalize_maintenance(self, token: str) -> list[str]:
        """Delete all jobs only after their safety fences have fully drained."""
        removed: list[str] = []
        with self._locked_legacy_and_durable():
            lease = _read_maintenance_unlocked(self.durable_dir)
            if lease is None or lease.token != token:
                raise JobMaintenanceError(
                    "Cursor job deletion lease changed before finalization"
                )
            legacy_records = sorted(self.legacy_dir.glob("*.json"))
            if (
                self.legacy_dir.resolve() != self.durable_dir.resolve()
                and legacy_records
            ):
                raise JobMaintenanceError(
                    "Cursor jobs could not be deleted safely: legacy source records "
                    "remain; run job recovery and retry"
                )
            jobs: list[CursorJob] = []
            quarantined: list[str] = []
            for path in sorted(self.durable_dir.glob("*.json")):
                try:
                    jobs.append(_read_model_unlocked(path))
                except JobQuarantinedError:
                    quarantined.append(path.stem)
            blockers: list[str] = []
            unresolved_quarantine = [
                path.name for path in _unresolved_quarantine_metadata(self.durable_dir)
            ]
            if quarantined or unresolved_quarantine:
                evidence = sorted({*quarantined, *unresolved_quarantine})
                blockers.append(
                    "quarantine evidence requires manual inspection: "
                    + ", ".join(evidence)
                )
            for job in jobs:
                reasons: list[str] = []
                if job.status in ACTIVE_STATUSES:
                    reasons.append(f"status {job.status.value}")
                if any(
                    value is not None
                    for value in (
                        job.worker_token,
                        job.worker_pid,
                        job.worker_boot_id,
                        job.worker_process_start,
                    )
                ):
                    reasons.append("worker ownership remains")
                if _must_retain(job):
                    reasons.append("recovery or reservation fence remains")
                if reasons:
                    blockers.append(f"{job.id}: {', '.join(reasons)}")
            if blockers:
                raise JobMaintenanceError(
                    "Cursor jobs could not be deleted safely: " + "; ".join(blockers)
                )
            for path in sorted(self.durable_dir.glob("*.json")):
                path.unlink()
                _delete_artifacts_unlocked(self.durable_dir, path.stem)
                removed.append(path.stem)
            _maintenance_path(self.durable_dir).unlink()
            _fsync_directory(self.durable_dir)
        return removed


_SQLITE_OPEN_LOCK = threading.RLock()


class JobStore:
    """SQLite-backed implementation of the existing typed store boundary."""

    def __init__(self, durable_dir: Path, legacy_dir: Path) -> None:
        self.durable_dir = durable_dir
        self.legacy_dir = legacy_dir
        self._db = SQLiteJobDatabase(durable_dir)
        self._ready = False

    @property
    def db_path(self) -> Path:
        return self._db.path

    def path(self, job_id: str) -> Path:
        if len(job_id) != 12 or any(
            character not in "0123456789abcdef" for character in job_id
        ):
            raise JobValidationError("invalid Cursor job ID")
        # Retained for legacy import fixtures and compatibility callers. SQLite
        # is authoritative after the cutover marker is committed.
        return self.durable_dir / f"{job_id}.json"

    @contextmanager
    def _locked_legacy_and_durable(self) -> Iterator[None]:
        if self.legacy_dir.resolve() == self.durable_dir.resolve():
            with locked(self.durable_dir):
                yield
            return
        with locked(self.legacy_dir), locked(self.durable_dir):
            yield

    @staticmethod
    def _maintenance_from_row(row: sqlite3.Row | None) -> MaintenanceLease | None:
        if row is None:
            return None
        return MaintenanceLease(
            token=str(row["token"]),
            started_at=float(row["started_at"]),
            owner_pid=int(row["owner_pid"]),
            owner_boot_id=str(row["owner_boot_id"]),
            owner_process_start=str(row["owner_process_start"]),
        )

    @staticmethod
    def _read_maintenance_db(
        connection: sqlite3.Connection,
    ) -> MaintenanceLease | None:
        return JobStore._maintenance_from_row(
            connection.execute(
                "SELECT token, started_at, owner_pid, owner_boot_id, "
                "owner_process_start FROM maintenance WHERE singleton = 1"
            ).fetchone()
        )

    @staticmethod
    def _reservation_rows(job: CursorJob) -> list[tuple[str, str, str]]:
        if not _job_reserves_resources(job):
            return []
        rows: list[tuple[str, str, str]] = []
        ticket = _ticket_identity(job)
        if ticket is not None:
            rows.append(("ticket", "\x1f".join(ticket), "active ticket ownership"))
        rows.extend(
            (
                "ticket",
                "\x1f".join(identity),
                "grouped repository clarification ownership",
            )
            for identity in _grouped_ticket_identities(job)
        )
        values = job.to_dict()
        targets = {
            value
            for value in (
                job.herdr_target,
                values.get("planner_target"),
                values.get("reviewer_target"),
                values.get("implementer_target"),
                values.get("participant_creation_target"),
            )
            if isinstance(value, str) and value
        }
        targets.update(str(owner["target"]) for owner in job.participant_session_owners)
        rows.extend(
            ("target", target, "active or recovery-fenced target")
            for target in sorted(targets)
        )
        if job.worktree_path:
            rows.append(
                ("worktree", job.worktree_path, "active or recovery-fenced checkout")
            )
        return rows

    def _save(self, connection: sqlite3.Connection, candidate: CursorJob) -> CursorJob:
        try:
            self._db.save_job(
                connection,
                candidate.to_dict(),
                reservations=self._reservation_rows(candidate),
            )
        except sqlite3.IntegrityError as exc:
            message = str(exc)
            if "reservations.resource_kind, reservations.resource_key" in message:
                raise JobValidationError(
                    "resource is reserved by both this job and another Cursor job"
                ) from exc
            if "delivery_claims.claim_token" in message:
                raise JobValidationError(
                    "delivery claim token is already owned by another Cursor job"
                ) from exc
            raise JobValidationError(
                f"SQLite job constraint failed: {message}"
            ) from exc
        return candidate

    @staticmethod
    def _validate_db_quarantine_reservation(
        connection: sqlite3.Connection,
        candidate: CursorJob,
        reservation: Literal["target", "worktree"],
    ) -> None:
        value = (
            candidate.herdr_target
            if reservation == "target"
            else candidate.worktree_path
        )
        if not value:
            return
        key_column = "target_key" if reservation == "target" else "worktree_key"
        flag_column = (
            "reserves_target" if reservation == "target" else "reserves_worktree"
        )
        row = connection.execute(
            f"SELECT metadata_path FROM quarantine WHERE resolved_at IS NULL "
            f"AND (blocks_all = 1 OR ({flag_column} = 1 AND {key_column} = ?)) "
            "ORDER BY metadata_path LIMIT 1",
            (value,),
        ).fetchone()
        if row is not None:
            raise JobValidationError(
                f"{reservation} reservation {value!r} is blocked by unresolved "
                f"quarantine evidence {Path(str(row['metadata_path'])).name}"
            )

    def _import_quarantine_rows(self, connection: sqlite3.Connection) -> int:
        count = 0
        quarantine = self.durable_dir / ".quarantine"
        for metadata_path in sorted(quarantine.glob("*.metadata.json")):
            evidence = _quarantine_evidence_unlocked(metadata_path)
            digest = None
            try:
                raw = json.loads(metadata_path.read_text())
                if isinstance(raw, dict):
                    digest = _optional_string(raw.get("sha256"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                pass
            reserves_target = bool(
                evidence.herdr_target
                and _quarantine_may_reserve(
                    metadata_path,
                    reservation="target",
                    value=evidence.herdr_target,
                )
            )
            reserves_worktree = bool(
                evidence.worktree_path
                and _quarantine_may_reserve(
                    metadata_path,
                    reservation="worktree",
                    value=evidence.worktree_path,
                )
            )
            connection.execute(
                """
                INSERT INTO quarantine(
                    evidence_id, job_id, metadata_path, payload_path,
                    payload_digest, error, quarantined_at, target_key,
                    worktree_key, blocks_all, reserves_target,
                    reserves_worktree, resolved_at, resolution_reason
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(metadata_path) DO UPDATE SET
                    job_id = excluded.job_id,
                    payload_path = excluded.payload_path,
                    payload_digest = excluded.payload_digest,
                    error = excluded.error,
                    quarantined_at = excluded.quarantined_at,
                    target_key = excluded.target_key,
                    worktree_key = excluded.worktree_key,
                    blocks_all = excluded.blocks_all,
                    reserves_target = excluded.reserves_target,
                    reserves_worktree = excluded.reserves_worktree,
                    resolved_at = excluded.resolved_at
                """,
                (
                    hashlib.sha256(str(metadata_path).encode()).hexdigest(),
                    evidence.job_id,
                    str(metadata_path),
                    str(evidence.payload_path) if evidence.payload_path else None,
                    digest,
                    evidence.quarantine_error,
                    evidence.quarantined_at,
                    evidence.herdr_target,
                    evidence.worktree_path,
                    int(evidence.inspection_error is not None),
                    int(reserves_target),
                    int(reserves_worktree),
                    evidence.quarantined_at if evidence.resolved else None,
                ),
            )
            if not evidence.resolved:
                count += 1
        return count

    def _import_json_sources(self) -> set[str]:
        archived: list[Path] = []
        failed_sources: list[Path] = []
        blocked: set[str] = set()
        with self._db.transaction() as connection:
            quarantine_count = self._import_quarantine_rows(connection)
            sources = sorted(self.durable_dir.glob("*.json"))
            invalid_sources: set[Path] = set()
            # Discover every malformed peer before importing any valid record.
            # This preserves the JSON store's scan-first reservation fence.
            for source in sources:
                try:
                    candidate = _parse_path(source)
                    if candidate.id != source.stem:
                        raise JobValidationError(
                            "Cursor job id must match its filename"
                        )
                    _normalize_for_durable_write(candidate)
                except JobValidationError as error:
                    blocked.add(source.stem)
                    invalid_sources.add(source)
                    _quarantine_import(
                        source,
                        self.durable_dir,
                        error,
                        remove_source=False,
                    )
                    failed_sources.append(source)
                    quarantine_count += 1
            self._import_quarantine_rows(connection)
            for source in sources:
                if source in invalid_sources:
                    continue
                try:
                    candidate = _parse_path(source)
                    if candidate.id != source.stem:
                        raise JobValidationError(
                            "Cursor job id must match its filename"
                        )
                    candidate = _normalize_for_durable_write(candidate)
                    try:
                        current_raw = self._db.load_job(connection, candidate.id)
                    except FileNotFoundError:
                        current = None
                    else:
                        current = CursorJob.from_dict(current_raw)
                    if current is not None:
                        if (
                            current.created_at != candidate.created_at
                            or current.id != candidate.id
                        ):
                            raise JobValidationError(
                                "import identity/created_at lineage conflicts "
                                "with the durable job"
                            )
                        if current.revision > candidate.revision:
                            archived.append(source)
                            continue
                        if current.revision == candidate.revision:
                            if current.to_dict() != candidate.to_dict():
                                raise JobValidationError(
                                    "import conflicts with durable job at the same revision"
                                )
                            archived.append(source)
                            continue
                    _validate_candidate_artifacts_unlocked(source, candidate, current)
                    if current is None:
                        _validate_quarantine_reservation_unlocked(
                            self.durable_dir, candidate, "target"
                        )
                        _validate_quarantine_reservation_unlocked(
                            self.durable_dir, candidate, "worktree"
                        )
                    self._save(connection, candidate)
                    projected = CursorJob.from_dict(
                        self._db.load_job(connection, candidate.id)
                    )
                    projected.validate_invariants(require_worker_owner=True)
                    if (
                        projected.loaded_schema_version != CURRENT_SCHEMA_VERSION
                        or projected.to_dict() != candidate.to_dict()
                    ):
                        raise JobValidationError(
                            "legacy import relational projection is not a "
                            "lossless native schema-v18 job"
                        )
                    archived.append(source)
                except (JobValidationError, sqlite3.IntegrityError) as error:
                    blocked.add(source.stem)
                    if source.exists():
                        validation = (
                            error
                            if isinstance(error, JobValidationError)
                            else JobValidationError(str(error))
                        )
                        _quarantine_import(
                            source,
                            self.durable_dir,
                            validation,
                            remove_source=False,
                        )
                        failed_sources.append(source)
                        quarantine_count += 1
            self._import_quarantine_rows(connection)
            self._db.set_meta(
                connection,
                "migration_status",
                "complete_with_quarantine" if quarantine_count else "complete",
            )
            self._db.set_meta(connection, "cutover_complete", "1")
            self._db.set_meta(connection, "import_failure_count", str(quarantine_count))
        for source in archived:
            destination = source.with_suffix(source.suffix + ".imported")
            if destination.exists():
                if destination.read_bytes() == source.read_bytes():
                    source.unlink()
                else:
                    destination = source.with_name(
                        f"{source.name}.{hashlib.sha256(source.read_bytes()).hexdigest()[:12]}.imported"
                    )
                    os.replace(source, destination)
            elif source.exists():
                os.replace(source, destination)
        for source in failed_sources:
            if source.exists():
                destination = source.with_suffix(source.suffix + ".failed")
                if destination.exists():
                    destination = source.with_name(
                        f"{source.name}.{hashlib.sha256(source.read_bytes()).hexdigest()[:12]}.failed"
                    )
                os.replace(source, destination)
        if archived or failed_sources:
            _fsync_directory(self.durable_dir)
        return blocked

    def _refresh_legacy_sources(self) -> set[str]:
        if not any(self.durable_dir.glob("*.json")):
            return set()
        with locked(self.durable_dir):
            return self._import_json_sources()

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        with _SQLITE_OPEN_LOCK:
            if self._ready:
                return
            self._db.initialize(normalize_legacy=_normalize_for_durable_write)
            with self._locked_legacy_and_durable():
                with self._db.transaction() as connection:
                    complete = self._db.meta(connection, "cutover_complete") == "1"
                    if not complete:
                        lease = _read_maintenance_unlocked(self.durable_dir)
                        if lease is not None:
                            connection.execute(
                                """
                                INSERT OR REPLACE INTO maintenance(
                                    singleton, token, operation, started_at, owner_pid,
                                    owner_boot_id, owner_process_start
                                ) VALUES(1, ?, 'delete_all', ?, ?, ?, ?)
                                """,
                                (
                                    lease.token,
                                    lease.started_at,
                                    lease.owner_pid,
                                    lease.owner_boot_id,
                                    lease.owner_process_start,
                                ),
                            )
                if not complete:
                    self._import_json_sources()
                    maintenance_path = _maintenance_path(self.durable_dir)
                    if maintenance_path.exists():
                        maintenance_path.rename(
                            maintenance_path.with_suffix(".imported")
                        )
                        _fsync_directory(self.durable_dir)
            fsync_database_directory(self.durable_dir)
            self._ready = True

    def migrate_legacy(
        self, *, inspect_worker: LegacyWorkerInspector | None = None
    ) -> set[str]:
        self._ensure_ready()
        blocked = migrate_legacy_jobs(
            self.legacy_dir,
            self.durable_dir,
            inspect_worker=inspect_worker,
        )
        blocked.update(self._import_json_sources())
        return blocked

    def get(self, job_id: str) -> CursorJob:
        self._ensure_ready()
        self._refresh_legacy_sources()
        with self._db.connect() as connection:
            try:
                return CursorJob.from_dict(self._db.load_job(connection, job_id))
            except FileNotFoundError as exc:
                quarantined = _quarantine_error(self.path(job_id))
                if quarantined is not None:
                    raise quarantined from exc
                raise

    def get_unless_maintenance(self, job_id: str) -> CursorJob | None:
        self._ensure_ready()
        with self._db.transaction() as connection:
            if self._read_maintenance_db(connection) is not None:
                return None
            return CursorJob.from_dict(self._db.load_job(connection, job_id))

    def list(self) -> list[CursorJob]:
        self._ensure_ready()
        self._refresh_legacy_sources()
        with self._db.connect() as connection:
            return [CursorJob.from_dict(raw) for raw in self._db.list_jobs(connection)]

    def maintenance_active(self) -> bool:
        self._ensure_ready()
        with self._db.connect() as connection:
            return self._read_maintenance_db(connection) is not None

    def create(
        self,
        job: CursorJob,
        *,
        enforce_unique_ticket: bool = False,
    ) -> CursorJob:
        self._ensure_ready()
        self._refresh_legacy_sources()
        with self._db.transaction() as connection:
            if self._read_maintenance_db(connection) is not None:
                raise JobMaintenanceError(
                    "Cursor jobs are temporarily unavailable during job deletion"
                )
            try:
                self._db.load_job(connection, job.id)
            except FileNotFoundError:
                pass
            else:
                raise JobValidationError(f"{job.id}.json: Cursor job already exists")
            candidate = _normalize_for_durable_write(job)
            if candidate.revision != 0:
                raise JobValidationError("new Cursor job revision must be zero")
            if enforce_unique_ticket:
                identity = _ticket_identity(candidate)
                if identity is not None:
                    conflict = connection.execute(
                        "SELECT job_id FROM reservations "
                        "WHERE resource_kind = 'ticket' AND resource_key = ?",
                        ("\x1f".join(identity),),
                    ).fetchone()
                    if conflict is not None:
                        raise ActiveTicketConflict(str(conflict["job_id"]))
            _validate_candidate_artifacts_unlocked(self.path(job.id), candidate, None)
            _validate_quarantine_reservation_unlocked(
                self.durable_dir, candidate, "target"
            )
            _validate_quarantine_reservation_unlocked(
                self.durable_dir, candidate, "worktree"
            )
            self._validate_db_quarantine_reservation(connection, candidate, "target")
            self._validate_db_quarantine_reservation(connection, candidate, "worktree")
            saved = self._save(connection, candidate)
            self._record_transition(
                connection,
                CoordinatorCommand(
                    job_id=saved.id,
                    expected_revision=0,
                    command_id=f"store.create:{saved.id}",
                    kind="admit",
                ),
                CoordinatorDecision(job=saved, event_kind="admit"),
            )
            return saved

    def claim_participant_capacity(self, limit: int) -> tuple[CursorJob, ...]:
        """Atomically admit the oldest waiting jobs up to the global limit."""

        if limit <= 0:
            raise JobValidationError("participant capacity must be positive")
        self._ensure_ready()
        self._refresh_legacy_sources()
        with self._db.transaction() as connection:
            jobs = [CursorJob.from_dict(raw) for raw in self._db.list_jobs(connection)]
            free = limit - sum(
                job.participant_admission_state == "held" for job in jobs
            )
            if free <= 0:
                return ()
            waiting = sorted(
                (
                    job
                    for job in jobs
                    if job.participant_admission_state == "waiting"
                    and job.status == JobStatus.QUEUED
                ),
                key=lambda job: (job.created_at, job.id),
            )
            claimed: list[CursorJob] = []
            for job in waiting[:free]:
                candidate = job.evolve_participant(job.participant_lifecycle.admit())
                self._save(connection, candidate)
                claimed.append(candidate)
            return tuple(claimed)

    def ticket_reservation_owner(self, identity: tuple[str, ...]) -> str | None:
        self._ensure_ready()
        self._refresh_legacy_sources()
        with self._db.connect(readonly=True) as connection:
            row = connection.execute(
                "SELECT job_id FROM reservations "
                "WHERE resource_kind = 'ticket' AND resource_key = ?",
                ("\x1f".join(identity),),
            ).fetchone()
        return str(row["job_id"]) if row is not None else None

    def stage_grouped_children(
        self,
        coordinator_id: str,
        command: JobCommand,
        children: tuple[CursorJob, ...],
    ) -> CursorJob | None:
        """Atomically fence a grouped answer and admit its selected children."""
        self._ensure_ready()
        self._refresh_legacy_sources()
        with self._db.transaction() as connection:
            if self._read_maintenance_db(connection) is not None:
                raise JobMaintenanceError(
                    "Cursor jobs are temporarily unavailable during job deletion"
                )
            current = CursorJob.from_dict(self._db.load_job(connection, coordinator_id))
            candidate = command(current)
            if candidate is None:
                return None
            candidate = _normalize_for_durable_write(candidate)
            validate_transition(current, candidate)
            _validate_candidate_artifacts_unlocked(
                self.path(coordinator_id), candidate, current
            )
            self._save(connection, candidate)
            for child in children:
                normalized = _normalize_for_durable_write(child)
                if normalized.revision != 0:
                    raise JobValidationError("new grouped child revision must be zero")
                _validate_candidate_artifacts_unlocked(
                    self.path(normalized.id), normalized, None
                )
                self._save(connection, normalized)
            return candidate

    def apply(
        self,
        command: CoordinatorCommand,
        decide: Callable[[CursorJob], CoordinatorDecision | None],
        *,
        reservation: Literal["target", "worktree"] | None = None,
        reject_maintenance: bool = False,
    ) -> CursorJob | None:
        """Commit one typed command, its event, and any admitted effects atomically."""

        self._ensure_ready()
        self._refresh_legacy_sources()
        with self._db.transaction() as connection:
            return self._apply_on_connection(
                connection,
                command,
                decide,
                reservation=reservation,
                reject_maintenance=reject_maintenance,
            )

    def update(self, job_id: str, command: JobCommand) -> CursorJob | None:
        return self._transaction(job_id, command)

    def reserve_target(self, job_id: str, command: JobCommand) -> CursorJob | None:
        return self._transaction(
            job_id, command, reservation="target", reject_maintenance=True
        )

    def reserve_worktree(self, job_id: str, command: JobCommand) -> CursorJob | None:
        return self._transaction(
            job_id, command, reservation="worktree", reject_maintenance=True
        )

    def update_unless_maintenance(
        self, job_id: str, command: JobCommand
    ) -> CursorJob | None:
        return self._transaction(job_id, command, reject_maintenance=True)

    def _transaction(
        self,
        job_id: str,
        command: JobCommand,
        *,
        reservation: Literal["target", "worktree"] | None = None,
        reject_maintenance: bool = False,
    ) -> CursorJob | None:
        def decide(current: CursorJob) -> CoordinatorDecision | None:
            candidate = command(current)
            if candidate is None:
                return None
            return CoordinatorDecision(job=candidate, event_kind="store.update")

        self._ensure_ready()
        self._refresh_legacy_sources()
        with self._db.transaction() as connection:
            if reject_maintenance and self._read_maintenance_db(connection) is not None:
                return None
            current = CursorJob.from_dict(self._db.load_job(connection, job_id))
            return self._apply_on_connection(
                connection,
                CoordinatorCommand(
                    job_id=job_id,
                    expected_revision=current.revision,
                    command_id=f"store.update:{job_id}:{uuid.uuid4().hex}",
                    kind="store.update",
                ),
                decide,
                reservation=reservation,
                current=current,
            )

    def _apply_on_connection(
        self,
        connection: sqlite3.Connection,
        command: CoordinatorCommand,
        decide: Callable[[CursorJob], CoordinatorDecision | None],
        *,
        reservation: Literal["target", "worktree"] | None = None,
        reject_maintenance: bool = False,
        current: CursorJob | None = None,
    ) -> CursorJob | None:
        if reject_maintenance and self._read_maintenance_db(connection) is not None:
            return None
        existing = self._db.load_command_event(connection, command.command_id)
        if existing is not None:
            try:
                payload = json.loads(str(existing["payload_json"]))
            except (TypeError, ValueError) as exc:
                raise JobValidationError(
                    f"command {command.command_id!r} has invalid audit identity"
                ) from exc
            if (
                str(existing["job_id"]) != command.job_id
                or int(existing["revision"]) != command.expected_revision + 1
                or not isinstance(payload, dict)
                or payload.get("command_kind") != command.kind
            ):
                raise JobValidationError(
                    f"command {command.command_id!r} conflicts with a persisted "
                    "command identity"
                )
            return CursorJob.from_dict(
                self._db.load_job(connection, str(existing["job_id"]))
            )
        if current is None:
            current = CursorJob.from_dict(self._db.load_job(connection, command.job_id))
        if current.revision != command.expected_revision:
            return None
        decision = decide(current)
        if decision is None:
            return None
        candidate = _normalize_for_durable_write(decision.job)
        validate_transition(current, candidate)
        _validate_candidate_artifacts_unlocked(
            self.path(command.job_id), candidate, current
        )
        if reservation is not None:
            _validate_quarantine_reservation_unlocked(
                self.durable_dir, candidate, reservation
            )
            self._validate_db_quarantine_reservation(connection, candidate, reservation)
        decision = CoordinatorDecision(
            job=candidate,
            effects=decision.effects,
            event_kind=decision.event_kind,
            event_payload=decision.event_payload,
        )
        self._record_transition(connection, command, decision)
        return self._save(connection, candidate)

    def _record_transition(
        self,
        connection: sqlite3.Connection,
        command: CoordinatorCommand,
        decision: CoordinatorDecision,
    ) -> None:
        job = decision.job
        self._db.insert_event(
            connection,
            event_id=uuid.uuid4().hex,
            command_id=command.command_id,
            job_id=job.id,
            revision=job.revision,
            kind=decision.event_kind,
            payload={
                **dict(decision.event_payload),
                "command_kind": command.kind,
                "effects": [effect.kind for effect in decision.effects],
            },
            created_at=time.time(),
        )
        try:
            self._db.insert_outbox_effects(connection, job.id, decision.effects)
        except sqlite3.IntegrityError as exc:
            raise JobValidationError(f"SQLite outbox constraint failed: {exc}") from exc

    def write_artifact(
        self,
        job_id: str,
        kind: Literal["plan", "review"],
        round_number: int,
        text: str,
        *,
        source_text: str | None = None,
    ) -> str:
        if kind not in WORKFLOW_ARTIFACT_KINDS:
            raise JobValidationError("invalid workflow artifact kind")
        if round_number < 0 or round_number > 2:
            raise JobValidationError(
                "workflow artifact round must be between zero and two"
            )
        plan_sha256 = None
        if kind == "review":
            if source_text is None or not source_text.strip():
                raise JobValidationError(
                    "workflow review requires the reviewed plan text"
                )
            plan_sha256 = hashlib.sha256(source_text.encode()).hexdigest()
        reference, serialized = _artifact_payload(
            job_id,
            kind,
            round_number,
            text,
            plan_sha256=plan_sha256,
        )
        self._ensure_ready()
        with self._db.transaction() as connection:
            self._db.load_job(connection, job_id)
            path = _artifact_path(self.durable_dir, reference)
            _ensure_artifact_directory_unlocked(self.durable_dir, job_id)
            _exclusive_bytes(path, serialized)
        return reference

    def publish_artifact(
        self,
        job_id: str,
        kind: Literal["plan", "review"],
        round_number: int,
        text: str,
        *,
        expected_worker_token: WorkerOwnership | str,
        expected_revision: int,
        expected_turn_token: str,
        expected_phase: str,
        expected_prior_reference: str | None,
        change: ArtifactCommand,
        expected_plan_reference: str | None = None,
    ) -> CursorJob | None:
        if kind not in WORKFLOW_ARTIFACT_KINDS or not 0 <= round_number <= 2:
            raise JobValidationError("invalid workflow artifact publication")
        self._ensure_ready()
        with self._db.transaction() as connection:
            current = CursorJob.from_dict(self._db.load_job(connection, job_id))
            field = "plan_artifact" if kind == "plan" else "review_artifact"
            if (
                current.terminal_intent_status is not None
                or current.revision != expected_revision
                or not _worker_claim_matches(current, expected_worker_token)
                or current.turn_token != expected_turn_token
                or current.workflow_phase.value != expected_phase
                or current.review_round != round_number
                or current.to_dict().get(field) != expected_prior_reference
            ):
                return None
            plan_sha256 = None
            if kind == "review":
                if (
                    expected_plan_reference is None
                    or current.plan_artifact != expected_plan_reference
                ):
                    return None
                match = _ARTIFACT_REF.fullmatch(expected_plan_reference)
                if match is None:
                    raise JobValidationError("invalid reviewed plan artifact reference")
                plan_text = _parse_artifact(
                    _artifact_path(self.durable_dir, expected_plan_reference),
                    expected_job_id=job_id,
                    expected_kind="plan",
                    expected_round=int(match.group("round")),
                )
                plan_sha256 = hashlib.sha256(plan_text.encode()).hexdigest()
            reference, serialized = _artifact_payload(
                job_id,
                kind,
                round_number,
                text,
                plan_sha256=plan_sha256,
            )
            artifact_path = _artifact_path(self.durable_dir, reference)
            _ensure_artifact_directory_unlocked(self.durable_dir, job_id)
            _exclusive_bytes(artifact_path, serialized)
            candidate = change(current, reference)
            if isinstance(expected_worker_token, WorkerOwnership):
                event = WorkerCallbackEvent(
                    expected_revision,
                    candidate.lifecycle,
                    expected_worker_token,
                )
                current.validate_lifecycle_event(candidate, event)
                candidate = replace(candidate, _lifecycle_event=event)
            if candidate.to_dict().get(field) != reference:
                raise JobValidationError(
                    f"artifact publication must set {field} to the new reference"
                )
            candidate = _normalize_for_durable_write(candidate)
            validate_transition(current, candidate)
            return self._save(connection, candidate)

    def read_artifact(
        self,
        job_id: str,
        reference: str,
        *,
        kind: Literal["plan", "review"],
    ) -> str:
        match = _ARTIFACT_REF.fullmatch(reference)
        if match is None or match.group("job") != job_id or match.group("kind") != kind:
            raise JobValidationError("workflow artifact reference does not match job")
        job = self.get(job_id)
        field = "plan_artifact" if kind == "plan" else "review_artifact"
        if job.to_dict().get(field) != reference:
            raise JobValidationError("workflow artifact reference is stale")
        source_sha256 = None
        if kind == "review":
            plan_reference = job.plan_artifact
            if plan_reference is None:
                raise JobValidationError("workflow review has no current plan")
            plan_match = _ARTIFACT_REF.fullmatch(plan_reference)
            assert plan_match is not None
            plan_text = _parse_artifact(
                _artifact_path(self.durable_dir, plan_reference),
                expected_job_id=job_id,
                expected_kind="plan",
                expected_round=int(plan_match.group("round")),
            )
            source_sha256 = hashlib.sha256(plan_text.encode()).hexdigest()
        return _parse_artifact(
            _artifact_path(self.durable_dir, reference),
            expected_job_id=job_id,
            expected_kind=kind,
            expected_round=int(match.group("round")),
            expected_source_sha256=source_sha256,
        )

    def create_follow_up(
        self,
        parent_job_id: str,
        build: FollowUpBuilder,
        *,
        expected_parent_revision: int,
        expected_completed_at: float | None = None,
    ) -> CursorJob:
        self._ensure_ready()
        self._refresh_legacy_sources()
        with self._db.transaction() as connection:
            if self._read_maintenance_db(connection) is not None:
                raise JobMaintenanceError(
                    "Cursor follow-ups are temporarily unavailable during job deletion"
                )
            try:
                parent = CursorJob.from_dict(
                    self._db.load_job(connection, parent_job_id)
                )
            except FileNotFoundError as exc:
                raise FollowUpUnavailable(
                    f"Cursor job {parent_job_id} is no longer available"
                ) from exc
            _validate_follow_up_source(parent, expected_completed_at)
            child = build(parent)
            _validate_follow_up_event(
                parent, child, expected_parent_revision, expected_completed_at
            )
            if child.parent_job_id != parent.id:
                raise JobValidationError(
                    "follow-up child must reference its parent job id"
                )
            if child.harness_kind != parent.harness_kind:
                raise JobValidationError(
                    "follow-up child must inherit parent harness_kind exactly"
                )
            if child.issue_provider != parent.issue_provider:
                raise JobValidationError(
                    "follow-up child must inherit parent issue_provider exactly"
                )
            for field in (
                "repository",
                "worktree_branch",
                "worktree_path",
                "worktree_workspace_id",
                "worktree_root_pane_id",
            ):
                if getattr(child, field) != getattr(parent, field):
                    raise JobValidationError(
                        f"follow-up child must inherit parent {field} exactly"
                    )
            try:
                self._db.load_job(connection, child.id)
            except FileNotFoundError:
                pass
            else:
                raise JobValidationError(f"{child.id}.json: Cursor job already exists")
            try:
                _validate_quarantine_reservation_unlocked(
                    self.durable_dir, child, "worktree"
                )
                self._validate_db_quarantine_reservation(connection, child, "worktree")
                return self._save(connection, _normalize_for_durable_write(child))
            except JobValidationError as exc:
                if "reserved by both" in str(
                    exc
                ) or "blocked by unresolved quarantine evidence" in str(exc):
                    raise FollowUpCheckoutBusy(
                        f"{parent.worktree_path} is busy with another Cursor job"
                    ) from exc
                raise

    def acknowledge_quarantine_reservations(
        self,
        job_id: str,
        *,
        reason: str,
        now: float | None = None,
    ) -> QuarantineAcknowledgement:
        if not reason.strip():
            raise JobValidationError("quarantine acknowledgement reason is required")
        self._ensure_ready()
        acknowledged_at = time.time() if now is None else now
        with self._db.transaction() as connection:
            metadata_paths = _quarantine_metadata(self.path(job_id))
            if not metadata_paths:
                raise JobValidationError(
                    f"{job_id}: no quarantine evidence is available to acknowledge"
                )
            resolved: list[str] = []
            for metadata_path in metadata_paths:
                if _quarantine_metadata_resolved(metadata_path):
                    continue
                if metadata_path.is_symlink():
                    raise JobValidationError(
                        f"{metadata_path.name}: quarantine metadata cannot be a symlink"
                    )
                _atomic_json(
                    _quarantine_resolution_path(metadata_path),
                    {
                        "job_id": job_id,
                        "metadata_name": metadata_path.name,
                        "metadata_sha256": hashlib.sha256(
                            metadata_path.read_bytes()
                        ).hexdigest(),
                        "acknowledged_at": acknowledged_at,
                        "reason": reason.strip(),
                    },
                )
                connection.execute(
                    "UPDATE quarantine SET resolved_at = ?, resolution_reason = ? "
                    "WHERE metadata_path = ?",
                    (acknowledged_at, reason.strip(), str(metadata_path)),
                )
                resolved.append(metadata_path.name)
            return QuarantineAcknowledgement(job_id, tuple(resolved), acknowledged_at)

    def list_quarantine_evidence(
        self, *, include_resolved: bool = False
    ) -> list[QuarantineEvidence]:
        self._ensure_ready()
        evidence = [
            _quarantine_evidence_unlocked(path)
            for path in sorted(
                (self.durable_dir / ".quarantine").glob("*.metadata.json")
            )
        ]
        return [
            record for record in evidence if include_resolved or not record.resolved
        ]

    def prune(
        self,
        *,
        now: float | None = None,
        retention_seconds: float = DELIVERED_JOB_RETENTION_SECONDS,
    ) -> list[str]:
        self._ensure_ready()
        cutoff = (time.time() if now is None else now) - retention_seconds
        removed: list[str] = []
        with self._db.transaction() as connection:
            if self._read_maintenance_db(connection) is not None:
                return removed
            for raw in self._db.list_jobs(connection):
                job = CursorJob.from_dict(raw)
                if (
                    job.status in TERMINAL_STATUSES
                    and job.delivered
                    and job.completed_at is not None
                    and job.completed_at < cutoff
                    and not _must_retain(job)
                ):
                    self._db.delete_job(connection, job.id)
                    removed.append(job.id)
        for job_id in removed:
            _delete_artifacts_unlocked(self.durable_dir, job_id)
        return removed

    def begin_maintenance(
        self,
        lease: MaintenanceLease,
        stage: JobCommand,
        *,
        owner_alive: MaintenanceOwnerAlive,
    ) -> list[CursorJob]:
        self._ensure_ready()
        with self._db.transaction() as connection:
            existing = self._read_maintenance_db(connection)
            if existing is not None:
                disposition = owner_alive(existing)
                if disposition is True:
                    raise JobMaintenanceError(
                        "another Cursor job deletion is already in progress"
                    )
                if disposition is None:
                    raise JobMaintenanceError(
                        "an existing Cursor job deletion owner cannot be verified; "
                        "retry after checking the recorded process"
                    )
            unresolved = _unresolved_quarantine_metadata(self.durable_dir)
            if unresolved:
                raise JobMaintenanceError(
                    "Cursor jobs could not be deleted safely: quarantine evidence "
                    "requires manual inspection: "
                    + ", ".join(path.name for path in unresolved)
                )
            connection.execute("DELETE FROM maintenance")
            connection.execute(
                """
                INSERT INTO maintenance(
                    singleton, token, operation, started_at, owner_pid,
                    owner_boot_id, owner_process_start
                ) VALUES(1, ?, 'delete_all', ?, ?, ?, ?)
                """,
                (
                    lease.token,
                    lease.started_at,
                    lease.owner_pid,
                    lease.owner_boot_id,
                    lease.owner_process_start,
                ),
            )
            staged: list[CursorJob] = []
            for raw in self._db.list_jobs(connection):
                current = CursorJob.from_dict(raw)
                candidate = stage(current)
                if candidate is not None:
                    candidate = _normalize_for_durable_write(candidate)
                    validate_transition(current, candidate)
                    current = self._save(connection, candidate)
                staged.append(current)
            return staged

    def abort_maintenance(self, token: str) -> bool:
        self._ensure_ready()
        with self._db.transaction() as connection:
            result = connection.execute(
                "DELETE FROM maintenance WHERE singleton = 1 AND token = ?",
                (token,),
            )
            return result.rowcount == 1

    def finalize_maintenance(self, token: str) -> list[str]:
        self._ensure_ready()
        removed: list[str] = []
        with self._locked_legacy_and_durable():
            with self._db.transaction() as connection:
                lease = self._read_maintenance_db(connection)
                if lease is None or lease.token != token:
                    raise JobMaintenanceError(
                        "Cursor job deletion lease changed before finalization"
                    )
                legacy_records = sorted(self.legacy_dir.glob("*.json"))
                if (
                    self.legacy_dir.resolve() != self.durable_dir.resolve()
                    and legacy_records
                ):
                    raise JobMaintenanceError(
                        "Cursor jobs could not be deleted safely: legacy source "
                        "records remain; run job recovery and retry"
                    )
                unresolved = [
                    path.name
                    for path in _unresolved_quarantine_metadata(self.durable_dir)
                ]
                blockers: list[str] = []
                if unresolved:
                    blockers.append(
                        "quarantine evidence requires manual inspection: "
                        + ", ".join(unresolved)
                    )
                jobs = [
                    CursorJob.from_dict(raw) for raw in self._db.list_jobs(connection)
                ]
                for job in jobs:
                    reasons: list[str] = []
                    if job.status in ACTIVE_STATUSES:
                        reasons.append(f"status {job.status.value}")
                    if any(
                        value is not None
                        for value in (
                            job.worker_token,
                            job.worker_pid,
                            job.worker_boot_id,
                            job.worker_process_start,
                        )
                    ):
                        reasons.append("worker ownership remains")
                    if _must_retain(job):
                        reasons.append("recovery or reservation fence remains")
                    if reasons:
                        blockers.append(f"{job.id}: {', '.join(reasons)}")
                if blockers:
                    raise JobMaintenanceError(
                        "Cursor jobs could not be deleted safely: "
                        + "; ".join(blockers)
                    )
                for job in jobs:
                    self._db.delete_job(connection, job.id)
                    removed.append(job.id)
                connection.execute(
                    "DELETE FROM maintenance WHERE singleton = 1 AND token = ?",
                    (token,),
                )
        for job_id in removed:
            _delete_artifacts_unlocked(self.durable_dir, job_id)
        return removed


AgentJobStore = JobStore
