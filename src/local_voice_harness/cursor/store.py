from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import time
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .model import (
    CURRENT_SCHEMA_VERSION,
    TERMINAL_STATUSES,
    CursorJob,
    JobStatus,
    JobValidationError,
    validate_reservations,
    validate_transition,
)

DELIVERED_JOB_RETENTION_SECONDS = 7 * 24 * 60 * 60
LegacyWorkerDisposition = Literal["absent", "stopped", "unsafe"]
LegacyWorkerInspector = Callable[[CursorJob], LegacyWorkerDisposition]


class JobQuarantinedError(JobValidationError):
    """A malformed job was isolated from active job processing."""


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
    if raw.get("status") in {status.value for status in TERMINAL_STATUSES}:
        return False
    field = "herdr_target" if reservation == "target" else "worktree_path"
    reserved = raw.get(field)
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
    os.replace(path, destination)
    destination.chmod(0o600)
    _fsync_directory(path.parent)
    _fsync_directory(quarantine)
    metadata_path = destination.with_suffix(".metadata.json")
    metadata: dict[str, object] = {
        "original_name": path.name,
        "quarantined_name": destination.name,
        "quarantined_at": time.time(),
        "sha256": digest,
        "error": str(error),
    }
    _atomic_json(metadata_path, metadata)
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
        validate_reservations([*_peer_models_unlocked(path), candidate])
    except JobValidationError as exc:
        if isinstance(exc, JobQuarantinedError):
            raise
        raise JobValidationError(f"{path.name}: {exc}") from exc
    _atomic_json(path, candidate.to_dict())


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
        validate_reservations([*_peer_models_unlocked(path), candidate])
    except JobValidationError as exc:
        if isinstance(exc, JobQuarantinedError):
            raise
        raise JobValidationError(f"{path.name}: {exc}") from exc
    _atomic_json(path, candidate.to_dict())
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
                        if existing.to_dict() != candidate.to_dict():
                            raise JobValidationError(
                                "legacy import conflicts with durable job at the "
                                "same revision"
                            )
                        source.unlink()
                        _fsync_directory(legacy_dir)
                        continue
                peers = [
                    peer
                    for peer in _peer_models_unlocked(destination)
                    if peer.id != candidate.id
                ]
                validate_reservations([*peers, candidate])
                _atomic_json(destination, candidate.to_dict())
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
                removed.append(job.id)
        if removed:
            _fsync_directory(jobs_dir)
    return removed


JobCommand = Callable[[CursorJob], CursorJob | None]


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

    def update(self, job_id: str, command: JobCommand) -> CursorJob | None:
        return self._transaction(job_id, command)

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
            if reservation is not None:
                _validate_quarantine_reservation_unlocked(
                    self.durable_dir,
                    candidate,
                    reservation,
                )
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
