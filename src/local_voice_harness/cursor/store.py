from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .model import (
    ACTIVE_STATUSES,
    CURRENT_SCHEMA_VERSION,
    TERMINAL_STATUSES,
    CursorJob,
    JobStatus,
    JobValidationError,
    validate_reservations,
    validate_transition,
)

DELIVERED_JOB_RETENTION_SECONDS = 7 * 24 * 60 * 60
MAX_WORKFLOW_ARTIFACT_BYTES = 64 * 1024
WORKFLOW_ARTIFACT_KINDS = frozenset({"plan", "review"})
_ARTIFACT_REF = re.compile(
    r"^\.artifacts/(?P<job>[0-9a-f]{12})/"
    r"(?P<kind>plan|review)-(?P<round>[0-2])"
    r"(?:-(?P<digest>[0-9a-f]{64}))?\.json$"
)
LegacyWorkerDisposition = Literal["absent", "stopped", "unsafe"]
LegacyWorkerInspector = Callable[[CursorJob], LegacyWorkerDisposition]


class JobQuarantinedError(JobValidationError):
    """A malformed job was isolated from active job processing."""


class ArtifactQuarantinedError(JobValidationError):
    """A malformed workflow artifact was isolated from active processing."""


class FollowUpUnavailable(JobValidationError):
    """A completed parent job cannot be used as a follow-up source."""


class FollowUpCheckoutBusy(JobValidationError):
    """The parent's retained checkout is reserved by another active job."""


class JobQuarantineWarning(UserWarning):
    """A malformed job file was moved into quarantine."""


@dataclass(frozen=True, slots=True)
class QuarantineAcknowledgement:
    job_id: str
    resolved_metadata: tuple[str, ...]
    acknowledged_at: float


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
    try:
        resolution = json.loads(resolution_path.read_text())
        return (
            isinstance(resolution, dict)
            and resolution.get("metadata_sha256")
            == hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        )
    except (OSError, json.JSONDecodeError):
        return False


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
    github_state = _mapping_field(provider_state, "github")
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
    return _read_model_unlocked(path).to_dict(preserve_loaded_version=True)


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
        values.update(
            worker_token=None,
            worker_pid=None,
            worker_boot_id=None,
            worker_process_start=None,
        )
        if active and safely_cleared:
            values.update(
                status=JobStatus.QUEUED.value,
                queued_at=candidate.queued_at or candidate.created_at,
                reconcile=bool(
                    candidate.herdr_target or candidate.has_uncertain_operation()
                ),
            )
        elif active:
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
    values["schema_version"] = CURRENT_SCHEMA_VERSION
    if candidate._compatibility_layout:
        values.pop("harness_kind", None)
        values.pop("session_id", None)
    normalized = CursorJob.from_dict(values)
    normalized.validate_invariants(require_worker_owner=True)
    return normalized


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
    if not (parent.repository and parent.worktree_branch and parent.worktree_path):
        raise FollowUpUnavailable(
            f"Cursor job {parent.id} has no isolated worktree to reuse"
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


JobCommand = Callable[[CursorJob], CursorJob | None]
FollowUpBuilder = Callable[[CursorJob], CursorJob]
ArtifactCommand = Callable[[CursorJob, str], CursorJob]


class JobStore:
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

    def create(self, job: CursorJob) -> CursorJob:
        with locked(self.durable_dir):
            path = self.path(job.id)
            if path.exists():
                raise JobValidationError(f"{path.name}: Cursor job already exists")
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
        expected_worker_token: str,
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
                or current.worker_token != expected_worker_token
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
            try:
                parent = _read_model_unlocked(self.path(parent_job_id))
            except (FileNotFoundError, JobQuarantinedError) as exc:
                raise FollowUpUnavailable(
                    f"Cursor job {parent_job_id} is no longer available"
                ) from exc
            _validate_follow_up_source(parent, expected_completed_at)
            child = build(parent)
            if child.parent_job_id != parent.id:
                raise JobValidationError(
                    "follow-up child must reference its parent job id"
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
        return self._transaction(job_id, command, reservation="target")

    def reserve_worktree(self, job_id: str, command: JobCommand) -> CursorJob | None:
        """Atomically apply a typed worktree-reservation transition."""
        return self._transaction(job_id, command, reservation="worktree")

    def _transaction(
        self,
        job_id: str,
        command: JobCommand,
        *,
        reservation: Literal["target", "worktree"] | None = None,
    ) -> CursorJob | None:
        with locked(self.durable_dir):
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

    def delete_all(self) -> list[str]:
        """Forcefully remove every durable Cursor job file.

        Unlike :meth:`prune`, this ignores retention, delivery, and reservation
        fences: it is the bulk "nuke" primitive behind the confirmation-gated
        CLI command. Quarantined payloads under ``.quarantine`` are left in
        place so their evidence survives. Returns the ids of the removed jobs.
        """
        if not self.durable_dir.is_dir():
            return []
        removed: list[str] = []
        with locked(self.durable_dir):
            for path in sorted(self.durable_dir.glob("*.json")):
                path.unlink()
                _delete_artifacts_unlocked(self.durable_dir, path.stem)
                removed.append(path.stem)
            if removed:
                _fsync_directory(self.durable_dir)
        return removed


AgentJobStore = JobStore
