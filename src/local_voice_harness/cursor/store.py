from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


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


def read_unlocked(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def write_unlocked(path: Path, job: dict[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(job, output, sort_keys=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def read_all_unlocked(jobs_dir: Path) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for path in jobs_dir.glob("*.json"):
        try:
            jobs.append(read_unlocked(path))
        except (OSError, json.JSONDecodeError):
            continue
    return jobs
