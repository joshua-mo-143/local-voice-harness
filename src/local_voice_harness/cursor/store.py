from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .model import (
    CURRENT_SCHEMA_VERSION,
    CursorJob,
    JobValidationError,
    validate_reservations,
    validate_transition,
)


class JobQuarantinedError(JobValidationError):
    """A malformed job was isolated from active job processing."""


class JobQuarantineWarning(UserWarning):
    """A malformed job file was moved into quarantine."""


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


def write_unlocked(path: Path, job: dict[str, object]) -> None:
    previous = _read_model_unlocked(path) if path.exists() else None
    try:
        parsed = CursorJob.from_dict(job)
        upgraded_values = parsed.to_dict()
        upgraded_values["schema_version"] = CURRENT_SCHEMA_VERSION
        candidate = CursorJob.from_dict(upgraded_values)
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


def read_all_unlocked(jobs_dir: Path) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for path in sorted(jobs_dir.glob("*.json")):
        try:
            jobs.append(read_unlocked(path))
        except JobQuarantinedError:
            continue
    return jobs
