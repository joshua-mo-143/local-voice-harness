from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..errors import HarnessError
from .model import (
    CURRENT_SCHEMA_VERSION,
    LEGACY_BOOT_ID,
    CursorJob,
    JobStatus,
    transition,
)
from .store import JobStore

WORKER_MODULE = "local_voice_harness.cursor.worker"
CRITICAL_WORKER_OPERATIONS = frozenset(
    {"agent_start", "fork_create", "worktree_create"}
)


class WorkerCancelled(Exception):
    """The durable worker claim was cancelled or fenced by another owner."""


def process_identity(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    closing = stat.rfind(")")
    fields = stat[closing + 2 :].split()
    return fields[19] if closing >= 0 and len(fields) > 19 else None


def boot_identity() -> str | None:
    try:
        identity = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None
    return identity or None


def worker_command_matches(job: CursorJob) -> bool:
    if job.worker_pid is None:
        return False
    try:
        arguments = Path(f"/proc/{job.worker_pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    if WORKER_MODULE.encode() not in arguments or job.id.encode() not in arguments:
        return False
    if not job.worker_token:
        return True
    try:
        claim = arguments.index(b"--claim")
    except ValueError:
        return False
    return (
        claim + 1 < len(arguments) and arguments[claim + 1] == job.worker_token.encode()
    )


def legacy_worker_command_matches(job: CursorJob) -> bool:
    if job.worker_pid is None:
        return False
    try:
        arguments = Path(f"/proc/{job.worker_pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return WORKER_MODULE.encode() in arguments and job.id.encode() in arguments


def has_legacy_worker_claim(job: CursorJob) -> bool:
    ownership = (
        job.worker_token,
        job.worker_pid,
        job.worker_boot_id,
        job.worker_process_start,
    )
    return any(value is not None for value in ownership) and (
        job.loaded_schema_version < CURRENT_SCHEMA_VERSION
        or job.worker_boot_id == LEGACY_BOOT_ID
    )


def inspect_and_stop_legacy_worker(
    job: CursorJob,
    timeout: float = 2.0,
    *,
    get_process_identity: Callable[[int], str | None] = process_identity,
    command_matches: Callable[[CursorJob], bool] = legacy_worker_command_matches,
) -> Literal["absent", "stopped", "unsafe"]:
    if job.worker_pid is None:
        return "absent"
    if not job.worker_process_start:
        return "unsafe"
    if get_process_identity(job.worker_pid) != job.worker_process_start:
        return "absent"
    if not command_matches(job):
        return "unsafe"
    try:
        process_group = os.getpgid(job.worker_pid)
        if get_process_identity(
            job.worker_pid
        ) != job.worker_process_start or not command_matches(job):
            return "unsafe"
        if process_group == job.worker_pid:
            os.killpg(process_group, signal.SIGTERM)
        else:
            os.kill(job.worker_pid, signal.SIGTERM)
    except ProcessLookupError:
        return "stopped"
    except PermissionError:
        return "unsafe"
    stopped = threading.Event()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if get_process_identity(job.worker_pid) != job.worker_process_start:
            return "stopped"
        stopped.wait(min(0.05, max(0.0, deadline - time.monotonic())))
    if job.worker_operation in CRITICAL_WORKER_OPERATIONS or not command_matches(job):
        return "unsafe"
    try:
        if os.getpgid(job.worker_pid) == job.worker_pid:
            os.killpg(job.worker_pid, signal.SIGKILL)
        else:
            os.kill(job.worker_pid, signal.SIGKILL)
    except ProcessLookupError:
        return "stopped"
    except PermissionError:
        return "unsafe"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if get_process_identity(job.worker_pid) != job.worker_process_start:
            return "stopped"
        stopped.wait(min(0.05, max(0.0, deadline - time.monotonic())))
    return "unsafe"


def worker_is_alive(
    job: CursorJob,
    *,
    get_boot_identity: Callable[[], str | None] = boot_identity,
    get_process_identity: Callable[[int], str | None] = process_identity,
) -> bool:
    return bool(
        job.worker_pid
        and job.worker_process_start
        and job.worker_boot_id
        and get_boot_identity() == job.worker_boot_id
        and get_process_identity(job.worker_pid) == job.worker_process_start
    )


def signal_worker(
    job: CursorJob,
    sent_signal: signal.Signals,
    *,
    include_process_group: bool = True,
    is_alive: Callable[[CursorJob], bool] = worker_is_alive,
    command_matches: Callable[[CursorJob], bool] = worker_command_matches,
) -> bool:
    if not is_alive(job) or not command_matches(job) or job.worker_pid is None:
        return False
    try:
        process_group = os.getpgid(job.worker_pid)
        if not is_alive(job) or not command_matches(job):
            return False
        if include_process_group and process_group == job.worker_pid:
            os.killpg(process_group, sent_signal)
        else:
            os.kill(job.worker_pid, sent_signal)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return True


def stop_worker(
    job: CursorJob,
    timeout: float = 2.0,
    *,
    is_alive: Callable[[CursorJob], bool] = worker_is_alive,
    send_signal: Callable[..., bool] = signal_worker,
) -> bool:
    if not is_alive(job):
        return True
    critical = job.worker_operation in CRITICAL_WORKER_OPERATIONS
    if not send_signal(job, signal.SIGTERM, include_process_group=not critical):
        return not is_alive(job)
    stopped = threading.Event()
    deadline = time.monotonic() + timeout
    while is_alive(job) and time.monotonic() < deadline:
        stopped.wait(min(0.05, max(0.0, deadline - time.monotonic())))
    if not is_alive(job):
        return True
    if critical:
        return False
    if not send_signal(job, signal.SIGKILL):
        return not is_alive(job)
    deadline = time.monotonic() + timeout
    while is_alive(job) and time.monotonic() < deadline:
        stopped.wait(min(0.05, max(0.0, deadline - time.monotonic())))
    return not is_alive(job)


def stop_legacy_worker(
    store: JobStore,
    job_id: str,
    timeout: float = 2.0,
    *,
    is_alive: Callable[[CursorJob], bool] = worker_is_alive,
) -> bool:
    job = store.get(job_id)
    if job.worker_token or not is_alive(job) or job.worker_pid is None:
        return True
    try:
        if os.getpgid(job.worker_pid) == job.worker_pid:
            os.killpg(job.worker_pid, signal.SIGTERM)
        else:
            os.kill(job.worker_pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            current = store.get(job_id)
        except (HarnessError, OSError):
            return True
        if not is_alive(current):
            return True
        time.sleep(0.05)
    return False


@contextmanager
def cooperative_worker_signals() -> Iterator[threading.Event]:
    requested = threading.Event()
    previous: signal._HANDLER | None = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.signal(
            signal.SIGTERM, lambda _signum, _frame: requested.set()
        )
    try:
        yield requested
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


@dataclass(frozen=True, slots=True)
class WorkerContext:
    store: JobStore
    job: CursorJob
    token: str
    cancellation_requested: threading.Event

    def checkpoint(self) -> CursorJob:
        if self.cancellation_requested.is_set():
            raise WorkerCancelled
        current = self.store.get(self.job.id)
        if (
            self.cancellation_requested.is_set()
            or current.worker_token != self.token
            or current.status
            not in {
                JobStatus.ROUTING,
                JobStatus.RUNNING,
                JobStatus.RECONCILING,
            }
        ):
            raise WorkerCancelled
        return current


def begin_worker(
    store: JobStore,
    job_id: str,
    claim_token: str | None,
    *,
    pid: int | None = None,
    now: float | None = None,
    get_boot_identity: Callable[[], str | None] = boot_identity,
    get_process_identity: Callable[[int], str | None] = process_identity,
) -> tuple[CursorJob, str] | None:
    token = claim_token or uuid.uuid4().hex
    owner_pid = os.getpid() if pid is None else pid
    owner_boot = get_boot_identity()
    owner_start = get_process_identity(owner_pid)
    if not owner_boot or not owner_start:
        raise HarnessError("could not establish Cursor worker process identity")
    started_at = time.time() if now is None else now

    def begin(job: CursorJob) -> CursorJob | None:
        if job.status != JobStatus.QUEUED:
            return None
        if claim_token and job.worker_token != claim_token:
            return None
        if not claim_token and job.worker_token:
            return None
        status = JobStatus.RECONCILING if job.reconcile else JobStatus.ROUTING
        return transition(
            job,
            status,
            worker_token=token,
            worker_pid=owner_pid,
            worker_boot_id=owner_boot,
            worker_process_start=owner_start,
            attempt_started_at=started_at,
            started_at=started_at,
        )

    claimed = store.update(job_id, begin)
    return (claimed, token) if claimed is not None else None


def run_worker(
    store: JobStore,
    job_id: str,
    claim_token: str | None,
    runner: Callable[[WorkerContext], None],
) -> None:
    claimed = begin_worker(store, job_id, claim_token)
    if claimed is None:
        return
    job, token = claimed
    with cooperative_worker_signals() as requested:
        runner(WorkerContext(store, job, token, requested))


def launch_worker(
    store: JobStore,
    logs_dir: Path,
    job_id: str,
    *,
    prepare_failure: Callable[[CursorJob, str, float], CursorJob],
    get_boot_identity: Callable[[], str | None] = boot_identity,
    get_process_identity: Callable[[int], str | None] = process_identity,
) -> None:
    claim_token = uuid.uuid4().hex
    launcher_pid = os.getpid()
    launcher_boot = get_boot_identity()
    launcher_start = get_process_identity(launcher_pid)
    if not launcher_boot or not launcher_start:
        raise HarnessError("could not establish Cursor launcher process identity")
    reserved_at = time.time()

    def reserve(job: CursorJob) -> CursorJob | None:
        if job.status != JobStatus.QUEUED or job.worker_token:
            return None
        return transition(
            job,
            JobStatus.QUEUED,
            worker_token=claim_token,
            worker_pid=launcher_pid,
            worker_boot_id=launcher_boot,
            worker_process_start=launcher_start,
            attempt_started_at=reserved_at,
        )

    reserved = store.update(job_id, reserve)
    if reserved is None:
        return

    def persist_launch_failure(message: str) -> None:
        failed_at = time.time()

        def fail(job: CursorJob) -> CursorJob | None:
            if job.status != JobStatus.QUEUED or job.worker_token != claim_token:
                return None
            return prepare_failure(job, message[:500], failed_at)

        store.update(job_id, fail)

    process: subprocess.Popen[bytes]
    logs_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    log_handle = (logs_dir / f"{job_id}.log").open("ab")
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                WORKER_MODULE,
                job_id,
                "--claim",
                claim_token,
            ],
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception as exc:
        message = (str(exc) or type(exc).__name__)[:500]
        persist_launch_failure(message)
        raise
    finally:
        log_handle.close()

    def retire_unadopted_child() -> bool:
        if process.poll() is not None:
            return True
        exit_observed = False
        try:
            process.terminate()
        except (ProcessLookupError, ChildProcessError):
            exit_observed = True
        try:
            process.wait(timeout=0.5)
            return True
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except (ProcessLookupError, ChildProcessError):
                exit_observed = True
            try:
                process.wait(timeout=0.5)
                return True
            except (ProcessLookupError, ChildProcessError):
                return exit_observed or process.poll() is not None
            except subprocess.TimeoutExpired:
                return False
        except (ProcessLookupError, ChildProcessError):
            return exit_observed or process.poll() is not None

    child_start: str | None = None
    child_boot: str | None = None
    child_adopted = False
    failure_message = "Cursor worker exited before process identity handoff completed"
    try:
        for _attempt in range(50):
            child_start = get_process_identity(process.pid)
            child_boot = get_boot_identity()
            if child_start and child_boot:
                break
            if process.poll() is not None:
                break
            time.sleep(0.01)
        if child_start and child_boot:

            def hand_off(job: CursorJob) -> CursorJob | None:
                if job.status != JobStatus.QUEUED or job.worker_token != claim_token:
                    return None
                return transition(
                    job,
                    JobStatus.QUEUED,
                    worker_pid=process.pid,
                    worker_boot_id=child_boot,
                    worker_process_start=child_start,
                )

            handed_off = store.update(job_id, hand_off)
            if handed_off is not None:
                child_adopted = True
            else:
                current = store.get(job_id)
                child_adopted = (
                    current.worker_token == claim_token
                    and current.worker_pid == process.pid
                    and current.worker_boot_id == child_boot
                    and current.worker_process_start == child_start
                )
                failure_message = (
                    "Cursor worker exited before ownership handoff completed"
                )
    finally:
        if not child_adopted:
            if not retire_unadopted_child():
                raise HarnessError(
                    "could not stop Cursor worker after process identity handoff failed"
                )
            persist_launch_failure(failure_message)
    if child_adopted:
        threading.Thread(target=process.wait, daemon=True).start()
