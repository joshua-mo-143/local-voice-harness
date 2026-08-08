from __future__ import annotations

import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from ..config import CURSOR_FOREGROUND_SECONDS, JOBS_DIR
from ..errors import HarnessError
from ..integrations.github import (
    GitHubClient,
    GitHubError,
    GitHubIssue,
    GitHubOperationAmbiguous,
    GitHubRepository,
)
from ..integrations.herdr import (
    AgentSelection,
    HerdrClient,
    HerdrError,
    extract_linear_issue,
    extract_marker,
)
from .prompts import cursor_prompt
from .store import locked, read_all_unlocked, read_unlocked, write_unlocked

ACTIVE_STATUSES = {
    "queued",
    "routing",
    "running",
    "reconciling",
    "awaiting_user",
    "blocked",
}
WORKER_STATUSES = {"queued", "routing", "running", "reconciling"}
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
DELIVERABLE_STATUSES = TERMINAL_STATUSES | {"awaiting_user", "blocked"}
DELIVERY_CLAIM_SECONDS = 300.0
DELIVERY_RETRY_SECONDS = 5.0
FOREGROUND_GRACE_SECONDS = 2.0
FAILED_RECONCILE_MAX_ATTEMPTS = 3
UNCERTAIN_RECONCILE_MAX_ATTEMPTS = 6
OPERATION_RECONCILE_BASE_SECONDS = 5.0
OPERATION_RECONCILE_MAX_SECONDS = 60.0
CRITICAL_WORKER_OPERATIONS = {"agent_start", "fork_create", "worktree_create"}
DeliveryClaims = list[tuple[str, str]]
FORK_CONFIRMATIONS = {
    "yes",
    "yes please",
    "confirm",
    "confirmed",
    "do it",
    "go ahead",
    "create it",
    "create the fork",
    "fork it",
}
FORK_REJECTIONS = {
    "no",
    "no thanks",
    "do not",
    "don't",
    "cancel",
    "stop",
}


class WorkerCancelled(Exception):
    pass


class ReservationConflict(Exception):
    pass


def _agent_not_found(exc: HerdrError) -> bool:
    return exc.code in {"agent_not_found", "not_found"}


def _has_uncertain_operation(job: dict[str, object]) -> bool:
    return (
        job.get("agent_dispatch_state")
        in {"dispatching", "ambiguous", "failed_observing"}
        or job.get("fork_operation_state")
        in {"submitted", "ambiguous", "failed_observing"}
        or job.get("worktree_provision_state")
        in {"dispatching", "ambiguous", "failed_observing"}
    )


def _manual_target_fence(job: dict[str, object]) -> bool:
    return job.get("agent_dispatch_state") == "manual_required" and bool(
        job.get("herdr_target")
    )


def _manual_resource_description(job: dict[str, object], operation: str) -> str:
    if operation == "agent":
        return f"Herdr agent {job.get('herdr_target') or 'unknown'}"
    if operation == "fork":
        return f"GitHub fork {job.get('fork_operation_target') or 'unknown'}"
    return f"worktree {job.get('worktree_path') or 'unknown'}"


def _refresh_failed_reconciliation_message(job: dict[str, object]) -> None:
    if job.get("status") != "failed":
        return
    base = str(
        job.get("reconciliation_base_error") or job.get("error") or "Cursor job failed"
    )
    base = base.split("; external operation reconciliation", 1)[0]
    base = base.split("; manual reconciliation required", 1)[0]
    job["reconciliation_base_error"] = base
    operation = str(job.get("manual_reconcile_operation") or "")
    if operation:
        message = (
            f"{base}; manual reconciliation required for "
            f"{_manual_resource_description(job, operation)}"
        )[:500]
    elif _has_uncertain_operation(job):
        message = f"{base}; external operation reconciliation is pending"[:500]
    else:
        message = base[:500]
    changed = job.get("error") != message or job.get("result") != message
    job.update({"error": message, "result": message})
    if changed:
        _prepare_delivery(job)


def decide_fork_confirmation(utterance: str) -> bool | None:
    normalized = re.sub(r"[^\w\s'’]", "", utterance.casefold())
    normalized = re.sub(r"\s+", " ", normalized).strip().replace("’", "'")
    if normalized in FORK_CONFIRMATIONS:
        return True
    if normalized in FORK_REJECTIONS:
        return False
    return None


def job_path(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{12}", job_id):
        raise HarnessError("invalid Cursor job ID")
    return JOBS_DIR / f"{job_id}.json"


def read_job(job_id: str) -> dict[str, object]:
    try:
        with locked(JOBS_DIR):
            return read_unlocked(job_path(job_id))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"could not read Cursor job {job_id}") from exc


def write_job(job: dict[str, object]) -> None:
    path = job_path(str(job["id"]))
    with locked(JOBS_DIR):
        write_unlocked(path, job)


def _mutate_job(
    job_id: str,
    mutate: Callable[[dict[str, object]], bool],
) -> dict[str, object] | None:
    with locked(JOBS_DIR):
        try:
            job = read_unlocked(job_path(job_id))
        except (OSError, json.JSONDecodeError) as exc:
            raise HarnessError(f"could not read Cursor job {job_id}") from exc
        if not mutate(job):
            return None
        job["revision"] = int(str(job.get("revision") or 0)) + 1
        write_unlocked(job_path(job_id), job)
        return dict(job)


def _clear_worker(job: dict[str, object]) -> None:
    job.update(
        {
            "worker_pid": None,
            "worker_process_start": None,
            "worker_token": None,
        }
    )


def _prepare_delivery(job: dict[str, object], *, now: float | None = None) -> None:
    job.update(
        {
            "delivered": False,
            "delivery_generation": int(str(job.get("delivery_generation") or 0)) + 1,
            "delivery_claim_token": None,
            "delivery_claimed_at": None,
            "delivery_retry_at": 0,
            "delivery_attempts": 0,
            "updated_at": time.time() if now is None else now,
        }
    )


def _all_jobs() -> list[dict[str, object]]:
    if not JOBS_DIR.is_dir():
        return []
    with locked(JOBS_DIR):
        return read_all_unlocked(JOBS_DIR)


def active_jobs() -> list[dict[str, object]]:
    return [job for job in _all_jobs() if job.get("status") in ACTIVE_STATUSES]


def reserved_targets(exclude_job_id: str | None = None) -> set[str]:
    return {
        str(job["herdr_target"])
        for job in _all_jobs()
        if job.get("herdr_target")
        and job.get("id") != exclude_job_id
        and (job.get("status") in ACTIVE_STATUSES or job.get("target_release_pending"))
    }


def repository_question(repositories: list[Path], reason: str = "") -> str:
    names = ", ".join(path.name for path in repositories)
    prefix = f"{reason.strip()} " if reason.strip() else ""
    return (
        f"{prefix}Which repository should Cursor use? Available repositories are {names}."
        if names
        else f"{prefix}I could not find an available local Git repository."
    )


def resolve_job_repository(
    client: HerdrClient,
    job: dict[str, object],
    repositories: list[Path],
) -> tuple[Path | None, list[Path]]:
    hint = str(job.get("repository_hint") or "").strip() or None
    task = str(job.get("request") or "")
    utterance = str(job.get("utterance") or task)
    repository, candidates = client.resolve_repository(hint, utterance, repositories)
    context_hint = str(job.get("context_repository") or "").strip() or None
    if repository is None and not hint and context_hint:
        repository, context_candidates = client.resolve_repository(
            context_hint, "", repositories
        )
        candidates = context_candidates or candidates
    return repository, candidates


def complete_from_output(
    job: dict[str, object], *, output: str, agent_status: str
) -> None:
    token = str(job.get("turn_token") or "")
    summary = extract_marker(output, "VOICE_SUMMARY", token)
    question = extract_marker(output, "VOICE_QUESTION", token)
    summary_position = output.rfind(f"VOICE_SUMMARY[{token}]")
    question_position = output.rfind(f"VOICE_QUESTION[{token}]")
    if question and question_position > summary_position:
        job.update(
            {
                "status": "awaiting_user",
                "question": question,
                "result": question,
                "clarification_kind": "agent",
                "updated_at": time.time(),
            }
        )
        _prepare_delivery(job)
    elif summary and summary_position > question_position:
        job.update(
            {
                "status": "completed",
                "result": summary,
                "completed_at": time.time(),
            }
        )
        _prepare_delivery(job)
    else:
        job.update(
            {
                "status": "blocked",
                "result": (
                    f"Herdr agent {job.get('herdr_target') or 'Cursor'} needs attention; "
                    f"it settled as {agent_status} without a voice summary."
                ),
                "completed_at": time.time(),
            }
        )
        _prepare_delivery(job)


def read_agent_completion(
    client: HerdrClient,
    job: dict[str, object],
    *,
    wait: bool,
    checkpoint: Callable[[], None] | None = None,
) -> tuple[str, str]:
    target = str(job.get("herdr_target") or "")
    if not target:
        raise HarnessError("Cursor job has no Herdr agent")
    if checkpoint is not None:
        checkpoint()
    agent = client.get_agent(target)
    if checkpoint is not None:
        checkpoint()
    if wait and agent.get("agent_status") == "working":
        if checkpoint is not None:
            checkpoint()
        result = client.run_json(
            "agent", "wait", target, "--timeout", "900000", timeout=910
        )
        if checkpoint is not None:
            checkpoint()
        agent = dict(result.get("agent") or {})
    if checkpoint is not None:
        checkpoint()
    output = client.run_text(
        "agent", "read", target, "--source", "recent-unwrapped", "--lines", "160"
    )
    if checkpoint is not None:
        checkpoint()
    return output, str(agent.get("agent_status") or "unknown")


def _process_identity(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    closing = stat.rfind(")")
    fields = stat[closing + 2 :].split()
    return fields[19] if closing >= 0 and len(fields) > 19 else None


def _worker_command_matches(job: dict[str, object]) -> bool:
    pid = int(str(job.get("worker_pid") or 0))
    if not pid:
        return False
    try:
        arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    job_id = str(job.get("id") or "").encode()
    if b"local_voice_harness.cursor.worker" not in arguments or job_id not in arguments:
        return False
    token = str(job.get("worker_token") or "").encode()
    if not token:
        return True
    try:
        claim = arguments.index(b"--claim")
    except ValueError:
        return False
    return claim + 1 < len(arguments) and arguments[claim + 1] == token


def _worker_is_alive(job: dict[str, object]) -> bool:
    pid = int(str(job.get("worker_pid") or 0))
    expected = str(job.get("worker_process_start") or "")
    if not pid:
        return False
    if expected:
        return _process_identity(pid) == expected
    return _worker_command_matches(job)


def _signal_worker(
    job: dict[str, object],
    sent_signal: signal.Signals,
    *,
    include_process_group: bool = True,
) -> bool:
    if not _worker_is_alive(job) or not _worker_command_matches(job):
        return False
    pid = int(str(job.get("worker_pid") or 0))
    try:
        process_group = os.getpgid(pid)
        if not _worker_is_alive(job) or not _worker_command_matches(job):
            return False
        if include_process_group and process_group == pid:
            os.killpg(process_group, sent_signal)
        else:
            os.kill(pid, sent_signal)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return True


def _stop_worker(job: dict[str, object], timeout: float = 2.0) -> bool:
    if not _worker_is_alive(job):
        return True
    critical = job.get("worker_operation") in CRITICAL_WORKER_OPERATIONS
    if not _signal_worker(job, signal.SIGTERM, include_process_group=not critical):
        return not _worker_is_alive(job)
    stopped = threading.Event()
    deadline = time.monotonic() + timeout
    while _worker_is_alive(job) and time.monotonic() < deadline:
        stopped.wait(min(0.05, max(0.0, deadline - time.monotonic())))
    if not _worker_is_alive(job):
        return True
    if critical:
        return False
    if not _signal_worker(job, signal.SIGKILL):
        return not _worker_is_alive(job)
    deadline = time.monotonic() + timeout
    while _worker_is_alive(job) and time.monotonic() < deadline:
        stopped.wait(min(0.05, max(0.0, deadline - time.monotonic())))
    return not _worker_is_alive(job)


@contextmanager
def _cooperative_worker_signals() -> Iterator[threading.Event]:
    requested = threading.Event()
    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.signal(
            signal.SIGTERM, lambda _signum, _frame: requested.set()
        )
    try:
        yield requested
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


def _worker_checkpoint(
    job_id: str, token: str, requested: threading.Event
) -> dict[str, object]:
    if requested.is_set():
        raise WorkerCancelled
    with locked(JOBS_DIR):
        try:
            job = read_unlocked(job_path(job_id))
        except (OSError, json.JSONDecodeError) as exc:
            raise HarnessError(f"could not read Cursor job {job_id}") from exc
    if (
        requested.is_set()
        or job.get("worker_token") != token
        or job.get("status") not in WORKER_STATUSES
    ):
        raise WorkerCancelled
    return job


def _stop_legacy_worker(job_id: str, timeout: float = 2.0) -> bool:
    job = read_job(job_id)
    if job.get("worker_token") or not _worker_is_alive(job):
        return True
    pid = int(str(job.get("worker_pid") or 0))
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            current = read_job(job_id)
        except HarnessError:
            return True
        if not _worker_is_alive(current):
            return True
        time.sleep(0.05)
    return False


def _target_release_owner_alive(job: dict[str, object]) -> bool:
    pid = int(str(job.get("target_release_owner_pid") or 0))
    expected = str(job.get("target_release_owner_start") or "")
    return bool(pid and expected and _process_identity(pid) == expected)


def _worker_change(
    job_id: str,
    token: str,
    allowed_statuses: set[str],
    change: Callable[[dict[str, object]], None],
) -> dict[str, object] | None:
    def guarded(job: dict[str, object]) -> bool:
        if (
            job.get("worker_token") != token
            or job.get("status") not in allowed_statuses
        ):
            return False
        change(job)
        return True

    return _mutate_job(job_id, guarded)


def _begin_worker(
    job_id: str, claim_token: str | None
) -> tuple[dict[str, object], str] | None:
    token = claim_token or uuid.uuid4().hex

    def begin(job: dict[str, object]) -> bool:
        if job.get("status") != "queued":
            return False
        existing = str(job.get("worker_token") or "")
        if claim_token and existing != claim_token:
            return False
        if not claim_token and existing and _worker_is_alive(job):
            return False
        reconcile = bool(job.get("reconcile"))
        job.update(
            {
                "status": "reconciling" if reconcile else "routing",
                "worker_token": token,
                "worker_pid": os.getpid(),
                "worker_process_start": _process_identity(os.getpid()),
                "attempt_started_at": time.time(),
                "started_at": time.time(),
            }
        )
        return True

    job = _mutate_job(job_id, begin)
    return (job, token) if job is not None else None


def _worker_question(
    job_id: str,
    token: str,
    question: str,
    *,
    clarification_kind: str,
) -> None:
    def ask(job: dict[str, object]) -> None:
        job.update(
            {
                "status": "awaiting_user",
                "question": question,
                "result": question,
                "clarification_kind": clarification_kind,
            }
        )
        job.pop("reconcile", None)
        _clear_worker(job)
        _prepare_delivery(job)

    _worker_change(job_id, token, {"routing", "running"}, ask)


def _worker_complete(
    job_id: str,
    token: str,
    *,
    output: str,
    agent_status: str,
    preserve_blocked_delivery: bool = False,
) -> None:
    delivery_keys = (
        "delivered",
        "delivery_generation",
        "delivery_claim_token",
        "delivery_claimed_at",
        "delivery_retry_at",
        "delivery_attempts",
    )

    def finish(job: dict[str, object]) -> None:
        previous_delivery = {key: job.get(key) for key in delivery_keys}
        complete_from_output(job, output=output, agent_status=agent_status)
        if (
            job.get("status") in TERMINAL_STATUSES
            and job.get("pull_request_worktree_state") == "ready"
        ):
            job["pull_request_worktree_state"] = "retained"
        if preserve_blocked_delivery and job.get("status") == "blocked":
            job.update(previous_delivery)
            job["next_reconcile_at"] = time.time() + DELIVERY_RETRY_SECONDS
        else:
            job["next_reconcile_at"] = time.time() + DELIVERY_RETRY_SECONDS
        job.pop("reconcile", None)
        _clear_worker(job)

    _worker_change(
        job_id,
        token,
        {"running", "reconciling"},
        finish,
    )


def _worker_fail(job_id: str, token: str, exc: Exception) -> None:
    message = (str(exc) or type(exc).__name__)[:500]

    def fail(job: dict[str, object]) -> None:
        uncertain = _has_uncertain_operation(job)
        if uncertain:
            message_with_fence = (
                f"{message}; external operation reconciliation is pending"
            )[:500]
        else:
            message_with_fence = message
        job.update(
            {
                "status": "failed",
                "error": message_with_fence,
                "result": message_with_fence,
                "reconciliation_base_error": message,
                "completed_at": time.time(),
                "target_release_pending": uncertain,
                "target_release_token": None,
                "target_release_owner_pid": None,
                "target_release_owner_start": None,
                "cancellation_reconciliation_pending": uncertain,
            }
        )
        job.pop("reconcile", None)
        if not uncertain:
            _clear_worker(job)
        _refresh_failed_reconciliation_message(job)
        _prepare_delivery(job)

    _worker_change(job_id, token, WORKER_STATUSES, fail)


def _pull_request_branch(job: dict[str, object]) -> str:
    configured = str(job.get("worktree_branch") or "")
    if configured:
        return configured
    job_id = str(job.get("id") or "")
    if not re.fullmatch(r"[0-9a-f]{12}", job_id):
        raise HarnessError("Cursor pull-request job has an invalid ID")
    return f"voice/github-pr-{job_id}"


def _prepare_pull_request_checkout(
    job_id: str,
    token: str,
    job: dict[str, object],
    checkpoint: Callable[[], None] | None = None,
) -> dict[str, object] | None:
    if not job.get("github_pull_request"):
        return job
    if job.get("pull_request_worktree_state") in {"ready", "retained"}:
        return job
    repository = Path(str(job.get("repository") or "")).resolve()
    checkout_value = str(job.get("worktree_path") or "")
    checkout = Path(checkout_value).resolve() if checkout_value else repository
    branch = _pull_request_branch(job)
    number = int(str(job.get("github_pull_request") or 0))
    try:
        if checkout == repository:
            raise HarnessError(
                "refusing to check out a pull request in the shared repository clone"
            )
        if not checkout.is_dir() or not (checkout / ".git").exists():
            raise HarnessError("pull-request worktree is missing or invalid")
        if checkpoint is not None:
            checkpoint()
        github = GitHubClient()
        if checkpoint is None:
            checked_out_branch = github.checkout_pull_request(
                checkout, number, branch=branch
            )
        else:
            checked_out_branch = github.checkout_pull_request(
                checkout, number, branch=branch, checkpoint=checkpoint
            )
        if checkpoint is not None:
            checkpoint()
    except Exception as exc:
        message = (str(exc) or type(exc).__name__)[:500]

        def quarantine(current: dict[str, object]) -> None:
            current.update(
                {
                    "pull_request_worktree_state": "quarantined",
                    "pull_request_worktree_error": message,
                }
            )

        _worker_change(job_id, token, {"routing"}, quarantine)
        raise

    def ready(current: dict[str, object]) -> None:
        current.update(
            {
                "pull_request_branch": checked_out_branch or branch,
                "pull_request_worktree_state": "ready",
                "pull_request_worktree_error": None,
            }
        )

    return _worker_change(job_id, token, {"routing"}, ready)


def _reserve_worker_target(
    job_id: str,
    token: str,
    selection: AgentSelection,
    repository: Path,
    issue_key: str | None,
    *,
    dispatching: bool = False,
) -> dict[str, object] | None:
    target = str(selection.target)
    with locked(JOBS_DIR):
        try:
            job = read_unlocked(job_path(job_id))
        except (OSError, json.JSONDecodeError) as exc:
            raise HarnessError(f"could not read Cursor job {job_id}") from exc
        if job.get("worker_token") != token or job.get("status") != "routing":
            return None
        for other in read_all_unlocked(JOBS_DIR):
            if other.get("id") != job_id and (
                (
                    other.get("herdr_target") == target
                    and (
                        other.get("status") in ACTIVE_STATUSES
                        or other.get("target_release_pending")
                    )
                )
                or (
                    selection.worktree_path
                    and other.get("worktree_path") == selection.worktree_path
                    and (
                        other.get("status") in ACTIVE_STATUSES
                        or other.get("target_release_pending")
                        or other.get("worktree_provision_state")
                        in {"quarantined", "manual_required"}
                    )
                )
            ):
                return None
        job.update(
            {
                "repository": str(repository),
                "issue_key": issue_key,
                "herdr_target": target,
                "herdr_pane_id": selection.pane_id,
                "herdr_workspace_id": selection.workspace_id,
                "worktree_path": selection.worktree_path,
                "agent_name": selection.name,
                "agent_dispatch_state": "dispatching" if dispatching else "ready",
                "revision": int(str(job.get("revision") or 0)) + 1,
            }
        )
        if dispatching:
            job["worker_operation"] = "agent_start"
        elif job.get("worker_operation") == "agent_start":
            job["worker_operation"] = None
        write_unlocked(job_path(job_id), job)
        return dict(job)


def _settle_worker_agent(
    job_id: str,
    token: str,
    selection: AgentSelection,
) -> dict[str, object] | None:
    def settle(job: dict[str, object]) -> bool:
        if (
            job.get("worker_token") != token
            or job.get("herdr_target") != selection.target
            or job.get("agent_dispatch_state") != "dispatching"
            or job.get("status") not in {"routing", "cancelled", "failed"}
        ):
            return False
        job.update(
            {
                "herdr_pane_id": selection.pane_id,
                "herdr_workspace_id": selection.workspace_id,
                "worktree_path": selection.worktree_path,
                "agent_name": selection.name,
                "agent_dispatch_state": "ready",
                "worker_operation": None,
            }
        )
        return True

    return _mutate_job(job_id, settle)


def _failed_operation_state(exc: HerdrError) -> str:
    return (
        "ambiguous"
        if exc.code in {"operation_timeout", "operation_ambiguous"}
        else "failed_observing"
    )


def _fail_worker_agent_dispatch(job_id: str, token: str, exc: HerdrError) -> None:
    def fail(job: dict[str, object]) -> bool:
        if (
            job.get("worker_token") != token
            or job.get("agent_dispatch_state") != "dispatching"
        ):
            return False
        job.update(
            {
                "agent_dispatch_state": _failed_operation_state(exc),
                "agent_dispatch_exited": True,
                "agent_reconcile_attempts": 0,
                "agent_next_reconcile_at": time.time(),
                "worker_operation": None,
            }
        )
        return True

    _mutate_job(job_id, fail)


def _reserve_worker_worktree(
    job_id: str,
    token: str,
    repository: Path,
    branch: str,
    checkout: Path,
    *,
    state: str,
) -> dict[str, object] | None:
    if state not in {"planned", "dispatching", "ready"}:
        raise HarnessError("invalid worktree provisioning state")
    with locked(JOBS_DIR):
        try:
            job = read_unlocked(job_path(job_id))
        except (OSError, json.JSONDecodeError) as exc:
            raise HarnessError(f"could not read Cursor job {job_id}") from exc
        if job.get("worker_token") != token or job.get("status") != "routing":
            return None
        checkout_value = str(checkout.resolve())
        for other in read_all_unlocked(JOBS_DIR):
            if (
                other.get("id") != job_id
                and other.get("worktree_path") == checkout_value
                and (
                    other.get("status") in ACTIVE_STATUSES
                    or other.get("target_release_pending")
                    or other.get("worktree_provision_state")
                    in {"quarantined", "manual_required"}
                )
            ):
                return None
        job.update(
            {
                "repository": str(repository.resolve()),
                "worktree_branch": branch,
                "worktree_path": checkout_value,
                "worktree_provision_state": state,
                "worker_operation": (
                    "worktree_create" if state == "dispatching" else None
                ),
                "revision": int(str(job.get("revision") or 0)) + 1,
            }
        )
        write_unlocked(job_path(job_id), job)
        return dict(job)


def _settle_worker_worktree(
    job_id: str,
    token: str,
    checkout: Path,
    workspace_id: str | None,
    pane_id: str | None,
) -> dict[str, object] | None:
    checkout_value = str(checkout.resolve())

    def settle(job: dict[str, object]) -> bool:
        if (
            job.get("worker_token") != token
            or job.get("worktree_path") != checkout_value
            or job.get("status") not in {"routing", "cancelled", "failed"}
        ):
            return False
        job.update(
            {
                "worktree_provision_state": (
                    "retained" if job.get("status") == "cancelled" else "ready"
                ),
                "worktree_workspace_id": workspace_id,
                "worktree_root_pane_id": pane_id,
            }
        )
        if job.get("worker_operation") == "worktree_create":
            job["worker_operation"] = None
        return True

    return _mutate_job(job_id, settle)


def _fail_worker_worktree(job_id: str, token: str, exc: HerdrError) -> None:
    def fail(job: dict[str, object]) -> bool:
        if (
            job.get("worker_token") != token
            or job.get("worktree_provision_state") != "dispatching"
        ):
            return False
        job.update(
            {
                "worktree_provision_state": _failed_operation_state(exc),
                "worktree_dispatch_exited": True,
                "worktree_reconcile_attempts": 0,
                "worktree_next_reconcile_at": time.time(),
                "worker_operation": None,
            }
        )
        return True

    _mutate_job(job_id, fail)


def _begin_fork_operation(
    job_id: str,
    token: str,
    source: GitHubRepository,
    login: str,
    target: str,
) -> dict[str, object] | None:
    def begin(job: dict[str, object]) -> None:
        job.update(
            {
                "github_repository": source.name_with_owner,
                "fork_operation_source": source.name_with_owner,
                "fork_operation_source_url": source.url,
                "fork_operation_source_parent": source.parent,
                "fork_operation_source_default_branch": source.default_branch,
                "fork_operation_source_private": source.is_private,
                "fork_operation_login": login,
                "fork_operation_target": target,
                "fork_operation_state": "planned",
                "worker_operation": None,
            }
        )

    return _worker_change(job_id, token, {"routing"}, begin)


def _mark_fork_dispatching(job_id: str, token: str) -> None:
    def dispatch(job: dict[str, object]) -> None:
        job.update(
            {
                "fork_operation_state": "submitted",
                "fork_committed": True,
                "fork_committed_at": time.time(),
                "worker_operation": "fork_create",
            }
        )

    if _worker_change(job_id, token, {"routing"}, dispatch) is None:
        raise WorkerCancelled


def _settle_fork_operation(
    job_id: str,
    token: str,
    fork: GitHubRepository | None,
    *,
    ambiguous: bool = False,
    failed_observing: bool = False,
) -> dict[str, object] | None:
    def settle(job: dict[str, object]) -> bool:
        if (
            job.get("worker_token") != token
            or job.get("fork_operation_state") not in {"planned", "submitted"}
            or job.get("status") not in {"routing", "cancelled", "failed"}
        ):
            return False
        if fork is not None:
            job.update(
                {
                    "fork_repository": fork.name_with_owner,
                    "fork_operation_state": "exists",
                    "fork_exists": True,
                    "worker_operation": None,
                }
            )
        elif ambiguous or failed_observing:
            job.update(
                {
                    "fork_operation_state": (
                        "failed_observing" if failed_observing else "ambiguous"
                    ),
                    "fork_exists": None,
                    "fork_dispatch_exited": True,
                    "fork_reconcile_attempts": 0,
                    "fork_next_reconcile_at": time.time(),
                    "worker_operation": None,
                }
            )
        else:
            job.update(
                {
                    "fork_operation_state": "failed",
                    "fork_exists": False,
                    "worker_operation": None,
                }
            )
        return True

    return _mutate_job(job_id, settle)


def run_worker(job_id: str, claim_token: str | None = None) -> None:
    claimed = _begin_worker(job_id, claim_token)
    if claimed is None:
        return
    job, worker_token = claimed
    with _cooperative_worker_signals() as cancellation_requested:

        def checkpoint() -> None:
            _worker_checkpoint(job_id, worker_token, cancellation_requested)

        try:
            client = HerdrClient()
            checkpoint()
            client.ensure_server()
            checkpoint()
            if job.get("reconcile"):
                output, agent_status = read_agent_completion(
                    client, job, wait=True, checkpoint=checkpoint
                )
                _worker_complete(
                    job_id,
                    worker_token,
                    output=output,
                    agent_status=agent_status,
                    preserve_blocked_delivery=bool(job.get("delivered")),
                )
                return

            turn = int(str(job.get("turn") or 0)) + 1
            turn_token = f"{job_id}-{turn}"

            def begin_turn(current: dict[str, object]) -> None:
                current.update({"turn": turn, "turn_token": turn_token})

            updated = _worker_change(job_id, worker_token, {"routing"}, begin_turn)
            if updated is None:
                return
            job = updated
            checkpoint()
            continuation = bool(job.get("continuation"))
            target = str(job.get("herdr_target") or "")
            if target and job.get("agent_dispatch_state") == "dispatching":
                checkout_value = str(job.get("worktree_path") or "")
                pane = str(job.get("herdr_pane_id") or "")
                workspace = str(job.get("herdr_workspace_id") or "")
                repository_value = str(job.get("repository") or "")
                if checkout_value and pane and workspace and repository_value:
                    checkpoint()
                    try:
                        agent = client.get_agent(target)
                    except HerdrError as exc:
                        if not _agent_not_found(exc):

                            def defer_dispatch(
                                current: dict[str, object],
                            ) -> None:
                                current.update(
                                    {"status": "queued", "queued_at": time.time()}
                                )
                                _clear_worker(current)

                            _worker_change(
                                job_id,
                                worker_token,
                                {"routing"},
                                defer_dispatch,
                            )
                            return
                        agent = {}
                    checkpoint()
                    if agent:
                        selection = AgentSelection(
                            target=target,
                            pane_id=str(agent.get("pane_id") or pane),
                            workspace_id=str(agent.get("workspace_id") or workspace),
                            cwd=str(agent.get("cwd") or checkout_value),
                            name=str(agent.get("name") or target),
                            worktree_path=checkout_value,
                        )
                    else:
                        selection = client.start_agent(
                            Path(checkout_value),
                            str(
                                job.get("worktree_label") or Path(repository_value).name
                            ),
                            pane,
                            workspace,
                            name=target,
                            checkpoint=checkpoint,
                        )
                    updated = _settle_worker_agent(job_id, worker_token, selection)
                    if updated is None:
                        return
                    job = updated
                    target = selection.target
                    checkpoint()
                else:

                    def retry_dispatch(current: dict[str, object]) -> None:
                        current.update(
                            {
                                "herdr_target": None,
                                "herdr_pane_id": None,
                                "herdr_workspace_id": None,
                                "agent_name": None,
                                "agent_dispatch_state": None,
                            }
                        )

                    updated = _worker_change(
                        job_id, worker_token, {"routing"}, retry_dispatch
                    )
                    if updated is None:
                        return
                    job = updated
                    target = ""
            if not target:
                repository: Path | None = None
                repositories: list[Path] = []
                candidates: list[Path] = []
                hint = str(job.get("repository_hint") or "").strip() or None
                task = str(job.get("request") or "")
                utterance = str(job.get("utterance") or task)
                issue_key = str(job.get("issue_key") or "") or extract_linear_issue(
                    utterance
                )
                reason = ""
                if job.get("github_pull_request"):
                    github_repository = str(job.get("github_repository") or "").strip()
                    number = int(str(job.get("github_pull_request") or 0))
                    if not github_repository or number <= 0:
                        _worker_question(
                            job_id,
                            worker_token,
                            "Which repository's pull request should I check out? "
                            "Please say its owner and repository name.",
                            clarification_kind="github_repository",
                        )
                        return
                    checkpoint()
                    provisioned_pr = GitHubClient().provision_pull_request(
                        github_repository, number, checkpoint=checkpoint
                    )
                    checkpoint()
                    repository = provisioned_pr.checkout
                    issue_key = None

                    def record_pull_request(current: dict[str, object]) -> None:
                        current.update(
                            {
                                "github_repository": (
                                    provisioned_pr.source.name_with_owner
                                ),
                                "repository": str(provisioned_pr.checkout),
                                "pull_request_worktree_state": "provisioning",
                            }
                        )

                    updated = _worker_change(
                        job_id, worker_token, {"routing"}, record_pull_request
                    )
                    if updated is None:
                        return
                    job = updated
                elif job.get("fork_requested"):
                    github_repository = str(job.get("github_repository") or "").strip()
                    if not github_repository:
                        _worker_question(
                            job_id,
                            worker_token,
                            "Which public GitHub repository should I fork? "
                            "Please say its owner and repository name.",
                            clarification_kind="github_repository",
                        )
                        return
                    if not job.get("fork_confirmed"):
                        _worker_question(
                            job_id,
                            worker_token,
                            f"Please confirm: should I create a GitHub fork of "
                            f"{github_repository}? Say yes or no.",
                            clarification_kind="fork_confirmation",
                        )
                        return
                    github = GitHubClient()
                    checkpoint()
                    source, login, fork_target = github.prepare_public_fork(
                        github_repository
                    )
                    checkpoint()
                    updated = _begin_fork_operation(
                        job_id,
                        worker_token,
                        source,
                        login,
                        fork_target,
                    )
                    if updated is None:
                        return
                    job = updated
                    try:
                        fork = github.ensure_fork(
                            source,
                            login,
                            checkpoint=checkpoint,
                            before_submit=lambda: _mark_fork_dispatching(
                                job_id, worker_token
                            ),
                        )
                    except WorkerCancelled:
                        raise
                    except GitHubError as exc:
                        current = read_job(job_id)
                        was_submitted = (
                            current.get("fork_operation_state") == "submitted"
                        )
                        visible = (
                            github.reconcile_fork(source, fork_target)
                            if was_submitted
                            else None
                        )
                        _settle_fork_operation(
                            job_id,
                            worker_token,
                            visible,
                            ambiguous=(
                                was_submitted
                                and visible is None
                                and isinstance(exc, GitHubOperationAmbiguous)
                            ),
                            failed_observing=(
                                was_submitted
                                and visible is None
                                and not isinstance(exc, GitHubOperationAmbiguous)
                            ),
                        )
                        raise
                    updated = _settle_fork_operation(job_id, worker_token, fork)
                    if updated is None:
                        return
                    job = updated
                    checkpoint()
                    if job.get("status") != "routing":
                        raise WorkerCancelled
                    checkout = github.ensure_clone(source, fork, checkpoint=checkpoint)
                    checkpoint()
                    repository = checkout

                    def record_provisioning(current: dict[str, object]) -> None:
                        current["repository"] = str(checkout)

                    updated = _worker_change(
                        job_id, worker_token, {"routing"}, record_provisioning
                    )
                    if updated is None:
                        return
                    job = updated
                elif job.get("github_issue"):
                    github_repository = str(job.get("github_repository") or "").strip()
                    number = int(str(job.get("github_issue") or 0))
                    if not github_repository or number <= 0:
                        _worker_question(
                            job_id,
                            worker_token,
                            "Which repository's GitHub issue should I work on? "
                            "Please say owner/repository and the issue number.",
                            clarification_kind="github_repository",
                        )
                        return
                    owner, repository_name = github_repository.split("/", 1)
                    checkpoint()
                    repositories = client.repository_roots()
                    checkpoint()
                    provisioned_issue = GitHubClient().provision_issue(
                        GitHubIssue(owner, repository_name, number),
                        candidates=repositories,
                        checkpoint=checkpoint,
                    )
                    checkpoint()
                    repository = provisioned_issue.checkout
                    issue_key = None

                    def record_issue(current: dict[str, object]) -> None:
                        current.update(
                            {
                                "github_repository": (
                                    provisioned_issue.source.name_with_owner
                                ),
                                "github_issue": provisioned_issue.issue.number,
                                "github_issue_url": provisioned_issue.issue.url,
                                "repository": str(provisioned_issue.checkout),
                            }
                        )

                    updated = _worker_change(
                        job_id, worker_token, {"routing"}, record_issue
                    )
                    if updated is None:
                        return
                    job = updated
                else:
                    checkpoint()
                    repositories = client.repository_roots()
                    checkpoint()
                    repository, candidates = resolve_job_repository(
                        client, job, repositories
                    )
                    checkpoint()
                if (
                    repository is None
                    and issue_key
                    and not hint
                    and not job.get("fork_requested")
                    and not job.get("github_pull_request")
                    and not job.get("github_issue")
                ):
                    checkpoint()
                    repository, _confidence, reason = client.infer_repository(
                        issue_key,
                        repositories,
                        token=f"{job_id}-route",
                        reserved=reserved_targets(job_id),
                        checkpoint=checkpoint,
                    )
                    checkpoint()
                if repository is None:
                    checkpoint()
                    repository, rofi_reason = client.choose_or_clone_repository(
                        candidates or repositories, checkpoint=checkpoint
                    )
                    checkpoint()
                    reason = rofi_reason or reason
                if repository is None:
                    question = repository_question(
                        candidates or repositories,
                        reason
                        or (
                            "The repository could not be determined confidently."
                            if hint or issue_key
                            else ""
                        ),
                    )
                    _worker_question(
                        job_id,
                        worker_token,
                        question,
                        clarification_kind="repository",
                    )
                    return
                for _attempt in range(3):
                    reservation: dict[str, object] | None = None
                    agent_settled = False

                    def reserve_selection(
                        selection: AgentSelection, dispatching: bool
                    ) -> None:
                        nonlocal reservation, job, target
                        reservation = _reserve_worker_target(
                            job_id,
                            worker_token,
                            selection,
                            repository,
                            issue_key,
                            dispatching=dispatching,
                        )
                        if reservation is None:
                            checkpoint()
                            raise ReservationConflict
                        job = reservation
                        target = selection.target

                    def settle_selection(selection: AgentSelection) -> None:
                        nonlocal reservation, job, target, agent_settled
                        reservation = _settle_worker_agent(
                            job_id, worker_token, selection
                        )
                        if reservation is None:
                            raise ReservationConflict
                        job = reservation
                        target = selection.target
                        agent_settled = True

                    def fail_selection(exc: HerdrError) -> None:
                        _fail_worker_agent_dispatch(job_id, worker_token, exc)

                    def reserve_worktree(
                        worktree_repository: Path,
                        branch: str,
                        checkout: Path,
                        state: str,
                    ) -> None:
                        nonlocal job
                        reserved_worktree = _reserve_worker_worktree(
                            job_id,
                            worker_token,
                            worktree_repository,
                            branch,
                            checkout,
                            state=state,
                        )
                        if reserved_worktree is None:
                            checkpoint()
                            raise ReservationConflict
                        job = reserved_worktree

                    def settle_worktree(
                        checkout: Path,
                        workspace_id: str | None,
                        pane_id: str | None,
                    ) -> None:
                        nonlocal job
                        settled_worktree = _settle_worker_worktree(
                            job_id,
                            worker_token,
                            checkout,
                            workspace_id,
                            pane_id,
                        )
                        if settled_worktree is None:
                            raise ReservationConflict
                        job = settled_worktree

                    def fail_worktree(exc: HerdrError) -> None:
                        _fail_worker_worktree(job_id, worker_token, exc)

                    try:
                        checkpoint()
                        selection = client.ensure_agent(
                            repository,
                            issue_key=issue_key or None,
                            agent_hint=str(job.get("agent_hint") or "") or None,
                            reserved=reserved_targets(job_id),
                            worktree_branch=(
                                str(job.get("worktree_branch") or "") or None
                            ),
                            worktree_label=(
                                str(job.get("worktree_label") or "") or None
                            ),
                            checkpoint=checkpoint,
                            reserve=reserve_selection,
                            settle=settle_selection,
                            fail_agent=fail_selection,
                            reserve_worktree=reserve_worktree,
                            settle_worktree=settle_worktree,
                            fail_worktree=fail_worktree,
                        )
                        if not agent_settled:
                            reserve_selection(selection, False)
                        checkpoint()
                        break
                    except ReservationConflict:
                        continue
                else:
                    raise HarnessError("could not reserve a Cursor agent")

            prepared = _prepare_pull_request_checkout(
                job_id, worker_token, job, checkpoint
            )
            if prepared is None:
                return
            job = prepared
            checkpoint()

            def mark_running(current: dict[str, object]) -> None:
                current["status"] = "running"
                current.pop("continuation", None)

            if _worker_change(job_id, worker_token, {"routing"}, mark_running) is None:
                return
            checkpoint()
            outcome = client.prompt_and_wait(
                target,
                cursor_prompt(
                    str(job.get("request") or ""),
                    turn_token,
                    continuation=continuation,
                    github_issue_context=(
                        str(job.get("github_issue_context") or "") or None
                    ),
                ),
                token=turn_token,
                checkpoint=checkpoint,
            )
            checkpoint()
            _worker_complete(
                job_id,
                worker_token,
                output=outcome.output,
                agent_status=outcome.status,
            )
        except WorkerCancelled:
            return
        except Exception as exc:
            _worker_fail(job_id, worker_token, exc)


def launch_worker(job_id: str) -> None:
    process: subprocess.Popen[bytes] | None = None
    with locked(JOBS_DIR):
        try:
            job = read_unlocked(job_path(job_id))
        except (OSError, json.JSONDecodeError) as exc:
            raise HarnessError(f"could not read Cursor job {job_id}") from exc
        if job.get("status") != "queued" or job.get("worker_token"):
            return
        claim_token = uuid.uuid4().hex
        log_handle = (JOBS_DIR / f"{job_id}.log").open("ab")
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "local_voice_harness.cursor.worker",
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
            job.update(
                {
                    "status": "failed",
                    "error": message,
                    "result": message,
                    "completed_at": time.time(),
                }
            )
            _prepare_delivery(job)
            job["revision"] = int(str(job.get("revision") or 0)) + 1
            write_unlocked(job_path(job_id), job)
            raise
        finally:
            log_handle.close()
        pid = process.pid
        job.update(
            {
                "worker_token": claim_token,
                "worker_pid": pid,
                "worker_process_start": _process_identity(pid),
                "attempt_started_at": time.time(),
                "revision": int(str(job.get("revision") or 0)) + 1,
            }
        )
        write_unlocked(job_path(job_id), job)
    assert process is not None
    threading.Thread(target=process.wait, daemon=True).start()


def start_job(
    text: str,
    *,
    repository: str | None = None,
    github_repository: str | None = None,
    github_issue: int | None = None,
    github_issue_context: str | None = None,
    fork_requested: bool = False,
    github_pull_request: int | None = None,
    agent: str | None = None,
    utterance: str | None = None,
    context_repository: str | None = None,
) -> str:
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    spoken_text = utterance if utterance is not None else text
    issue_repository = (github_repository or "").strip()
    github_issue_url = (
        f"https://github.com/{issue_repository}/issues/{github_issue}"
        if issue_repository and github_issue
        else None
    )
    write_job(
        {
            "id": job_id,
            "schema_version": 4,
            "revision": 0,
            "request": text,
            "utterance": utterance,
            "trusted_utterance": spoken_text,
            "repository_hint": repository,
            "context_repository": context_repository,
            "github_repository": github_repository,
            "github_issue": github_issue,
            "github_issue_url": github_issue_url,
            "github_issue_context": github_issue_context,
            "fork_requested": fork_requested,
            "github_pull_request": github_pull_request,
            "worktree_branch": (
                f"voice/github-{job_id}"
                if fork_requested
                else (
                    f"voice/github-issue-{github_issue}"
                    if github_issue
                    else (f"voice/github-pr-{job_id}" if github_pull_request else None)
                )
            ),
            "worktree_label": (
                f"github-{job_id[:6]}"
                if fork_requested
                else (
                    f"issue-{github_issue}"
                    if github_issue
                    else (f"pr-{github_pull_request}" if github_pull_request else None)
                )
            ),
            "pull_request_worktree_state": ("pending" if github_pull_request else None),
            "agent_hint": agent,
            "issue_key": extract_linear_issue(spoken_text),
            "status": "queued",
            "delivered": False,
            "created_at": now,
            "queued_at": now,
            "foreground_until": now
            + CURSOR_FOREGROUND_SECONDS
            + FOREGROUND_GRACE_SECONDS,
        }
    )
    launch_worker(job_id)
    return job_id


def reply_job(job_id: str, text: str, *, trusted_utterance: str | None = None) -> None:
    now = time.time()
    should_launch = True

    def reply(job: dict[str, object]) -> bool:
        nonlocal should_launch
        if job.get("status") != "awaiting_user":
            return False
        if job.get("clarification_kind") == "repository":
            job.update(
                {"repository_hint": text, "herdr_target": None, "continuation": False}
            )
        elif job.get("clarification_kind") == "github_repository":
            job.update(
                {
                    "github_repository": text.strip(),
                    "herdr_target": None,
                    "continuation": False,
                }
            )
        elif job.get("clarification_kind") == "fork_confirmation":
            confirmation = decide_fork_confirmation(trusted_utterance or "")
            if confirmation is False:
                should_launch = False
                job.update(
                    {
                        "status": "completed",
                        "question": None,
                        "clarification_kind": None,
                        "result": "Okay, I did not create a GitHub fork.",
                        "completed_at": now,
                    }
                )
                _clear_worker(job)
                _prepare_delivery(job, now=now)
                return True
            if confirmation is None:
                should_launch = False
                question = "Please answer yes or no. Should I create the GitHub fork?"
                job.update({"question": question, "result": question})
                _prepare_delivery(job, now=now)
                return True
            job.update(
                {
                    "fork_confirmed": True,
                    "herdr_target": None,
                    "continuation": False,
                }
            )
        else:
            job.update({"continuation": True, "request": text})
        job.update(
            {
                "status": "queued",
                "question": None,
                "clarification_kind": None,
                "delivered": True,
                "delivery_claim_token": None,
                "delivery_claimed_at": None,
                "queued_at": now,
                "updated_at": now,
                "foreground_until": now
                + CURSOR_FOREGROUND_SECONDS
                + FOREGROUND_GRACE_SECONDS,
            }
        )
        _clear_worker(job)
        return True

    if _mutate_job(job_id, reply) is None:
        raise HarnessError(f"Cursor job {job_id} is not waiting for a reply")
    if should_launch:
        launch_worker(job_id)


def _cancel_target_and_release(
    job_id: str,
    target: str,
    release_token: str,
    *,
    worker_stopped: bool = True,
) -> None:
    interrupted = not target
    try:
        if target:
            try:
                client = HerdrClient()
                client.ensure_server()
                client.cancel_agent(target)
                interrupted = True
            except HerdrError as exc:
                interrupted = _agent_not_found(exc)
    finally:

        def release_target(job: dict[str, object]) -> bool:
            if (
                job.get("status") not in {"cancelled", "failed"}
                or job.get("target_release_token") != release_token
            ):
                return False
            if interrupted and worker_stopped and not _has_uncertain_operation(job):
                job.update(
                    {
                        "target_release_pending": False,
                        "target_release_token": None,
                        "target_release_owner_pid": None,
                        "target_release_owner_start": None,
                        "cancellation_reconciliation_pending": False,
                    }
                )
                if job.get("status") == "cancelled":
                    job["result"] = f"Cursor job {job_id} was cancelled."
                _clear_worker(job)
            else:
                job.update(
                    {
                        "target_release_owner_pid": None,
                        "target_release_owner_start": None,
                    }
                )
            return True

        _mutate_job(job_id, release_target)


def _reconciliation_due(job: dict[str, object], prefix: str, now: float) -> bool:
    return now >= float(str(job.get(f"{prefix}_next_reconcile_at") or 0))


def _record_reconciliation_observation(
    job_id: str,
    prefix: str,
    state_key: str,
    expected_states: set[str],
    *,
    now: float,
    observed_absent: bool,
) -> None:
    def observe(job: dict[str, object]) -> bool:
        state = str(job.get(state_key) or "")
        if state not in expected_states:
            return False
        attempts = int(str(job.get(f"{prefix}_reconcile_attempts") or 0)) + 1
        job[f"{prefix}_reconcile_attempts"] = attempts
        job[f"{prefix}_last_reconciled_at"] = now
        absent_observations = int(str(job.get(f"{prefix}_absent_observations") or 0))
        if observed_absent:
            absent_observations += 1
            job[f"{prefix}_absent_observations"] = absent_observations
        if (
            state == "failed_observing"
            and absent_observations >= FAILED_RECONCILE_MAX_ATTEMPTS
        ):
            job[state_key] = "confirmed_absent"
            job[f"{prefix}_confirmed_absent_at"] = now
            job[f"{prefix}_next_reconcile_at"] = None
            job["worker_operation"] = None
            job["cancellation_reconciliation_pending"] = False
            _refresh_failed_reconciliation_message(job)
        elif attempts >= UNCERTAIN_RECONCILE_MAX_ATTEMPTS:
            job[state_key] = "manual_required"
            job[f"{prefix}_next_reconcile_at"] = None
            job[f"{prefix}_automatic_reconcile_stopped_at"] = now
            job["manual_reconcile_operation"] = prefix
            job["manual_reconcile_token"] = uuid.uuid4().hex
            job["manual_reconcile_required_at"] = now
            job["worker_operation"] = None
            job["cancellation_reconciliation_pending"] = False
            _refresh_failed_reconciliation_message(job)
        else:
            job[f"{prefix}_next_reconcile_at"] = now + (
                min(
                    OPERATION_RECONCILE_MAX_SECONDS,
                    OPERATION_RECONCILE_BASE_SECONDS * (2 ** (attempts - 1)),
                )
            )
        return True

    _mutate_job(job_id, observe)


def _reconcile_uncertain_agent(job: dict[str, object], *, now: float) -> None:
    states = {"dispatching", "ambiguous", "failed_observing"}
    if job.get("agent_dispatch_state") not in states or not _reconciliation_due(
        job, "agent", now
    ):
        return
    target = str(job.get("herdr_target") or "")
    if not target:
        _record_reconciliation_observation(
            str(job["id"]),
            "agent",
            "agent_dispatch_state",
            states,
            now=now,
            observed_absent=False,
        )
        return
    try:
        client = HerdrClient()
        client.ensure_server()
        agent = client.get_agent(target)
    except HerdrError as exc:
        if _agent_not_found(exc):
            _record_reconciliation_observation(
                str(job["id"]),
                "agent",
                "agent_dispatch_state",
                states,
                now=now,
                observed_absent=True,
            )
        else:
            _record_reconciliation_observation(
                str(job["id"]),
                "agent",
                "agent_dispatch_state",
                states,
                now=now,
                observed_absent=False,
            )
        return
    if not agent:
        _record_reconciliation_observation(
            str(job["id"]),
            "agent",
            "agent_dispatch_state",
            states,
            now=now,
            observed_absent=True,
        )
        return

    def visible(current: dict[str, object]) -> bool:
        if (
            current.get("agent_dispatch_state") not in states
            or current.get("herdr_target") != target
        ):
            return False
        current.update(
            {
                "agent_dispatch_state": "ready",
                "agent_name": str(agent.get("name") or target),
                "herdr_pane_id": str(
                    agent.get("pane_id") or current.get("herdr_pane_id") or ""
                ),
                "herdr_workspace_id": str(
                    agent.get("workspace_id") or current.get("herdr_workspace_id") or ""
                ),
                "worker_operation": None,
                "agent_next_reconcile_at": None,
            }
        )
        _refresh_failed_reconciliation_message(current)
        return True

    _mutate_job(str(job["id"]), visible)


def _fork_source_from_job(job: dict[str, object]) -> GitHubRepository | None:
    name = str(job.get("fork_operation_source") or "")
    url = str(job.get("fork_operation_source_url") or "")
    if not name or not url:
        return None
    return GitHubRepository(
        name_with_owner=name,
        url=url,
        is_private=bool(job.get("fork_operation_source_private")),
        default_branch=str(job.get("fork_operation_source_default_branch") or ""),
        parent=str(job.get("fork_operation_source_parent") or "") or None,
    )


def _reconcile_uncertain_fork(job: dict[str, object], *, now: float) -> None:
    states = {"submitted", "ambiguous", "failed_observing"}
    if job.get("fork_operation_state") not in states or not _reconciliation_due(
        job, "fork", now
    ):
        return
    source = _fork_source_from_job(job)
    target = str(job.get("fork_operation_target") or "")
    if source is None or not target:
        _record_reconciliation_observation(
            str(job["id"]),
            "fork",
            "fork_operation_state",
            states,
            now=now,
            observed_absent=False,
        )
        return
    try:
        fork = GitHubClient().reconcile_fork(source, target)
    except GitHubError:
        _record_reconciliation_observation(
            str(job["id"]),
            "fork",
            "fork_operation_state",
            states,
            now=now,
            observed_absent=False,
        )
        return
    if fork is None:
        _record_reconciliation_observation(
            str(job["id"]),
            "fork",
            "fork_operation_state",
            states,
            now=now,
            observed_absent=True,
        )
        return

    def visible(current: dict[str, object]) -> bool:
        if current.get("fork_operation_state") not in states:
            return False
        current.update(
            {
                "fork_operation_state": "exists",
                "fork_exists": True,
                "fork_repository": fork.name_with_owner,
                "worker_operation": None,
                "fork_next_reconcile_at": None,
            }
        )
        _refresh_failed_reconciliation_message(current)
        return True

    _mutate_job(str(job["id"]), visible)


def _reconcile_uncertain_worktree(job: dict[str, object], *, now: float) -> None:
    states = {"dispatching", "ambiguous", "failed_observing"}
    if job.get("worktree_provision_state") not in states or not _reconciliation_due(
        job, "worktree", now
    ):
        return
    repository = str(job.get("repository") or "")
    branch = str(job.get("worktree_branch") or "")
    checkout_value = str(job.get("worktree_path") or "")
    if not repository or not branch or not checkout_value:
        _record_reconciliation_observation(
            str(job["id"]),
            "worktree",
            "worktree_provision_state",
            states,
            now=now,
            observed_absent=False,
        )
        return
    checkout = Path(checkout_value).resolve()
    try:
        client = HerdrClient()
        client.ensure_server()
        listing = client.run_json("worktree", "list", "--cwd", repository, "--json")
    except HerdrError:
        _record_reconciliation_observation(
            str(job["id"]),
            "worktree",
            "worktree_provision_state",
            states,
            now=now,
            observed_absent=False,
        )
        return
    match = next(
        (
            item
            for item in listing.get("worktrees") or []
            if item.get("branch") == branch
            and Path(str(item.get("path") or "")).resolve() == checkout
        ),
        None,
    )
    if match is None and not checkout.exists():
        _record_reconciliation_observation(
            str(job["id"]),
            "worktree",
            "worktree_provision_state",
            states,
            now=now,
            observed_absent=True,
        )
        return

    def reconciled(current: dict[str, object]) -> bool:
        if current.get("worktree_provision_state") not in states:
            return False
        if match is None:
            current["worktree_provision_state"] = "quarantined"
            current["worktree_provision_error"] = (
                "reserved worktree path exists but Herdr did not report the "
                "expected branch"
            )
            current["worktree_manual_inspection_required"] = True
            current["worker_operation"] = None
            current["cancellation_reconciliation_pending"] = False
        else:
            current.update(
                {
                    "worktree_provision_state": "retained",
                    "worktree_workspace_id": str(match.get("open_workspace_id") or "")
                    or None,
                    "worktree_provision_error": None,
                    "worker_operation": None,
                    "worktree_manual_inspection_required": False,
                    "worktree_next_reconcile_at": None,
                    "cancellation_reconciliation_pending": False,
                }
            )
        _refresh_failed_reconciliation_message(current)
        return True

    _mutate_job(str(job["id"]), reconciled)


def _reconcile_uncertain_operations(job: dict[str, object], *, now: float) -> None:
    _reconcile_uncertain_agent(job, now=now)
    current = read_job(str(job["id"]))
    _reconcile_uncertain_fork(current, now=now)
    current = read_job(str(job["id"]))
    _reconcile_uncertain_worktree(current, now=now)


def cancel_job(job_id: str) -> str:
    initial = read_job(job_id)
    if (
        initial.get("status") in WORKER_STATUSES
        and not initial.get("worker_token")
        and _worker_is_alive(initial)
        and not _stop_legacy_worker(job_id)
    ):
        raise HarnessError(f"could not safely stop legacy Cursor worker for {job_id}")
    target = ""
    worker: dict[str, object] | None = None
    release_token = uuid.uuid4().hex
    cancelled_at = time.time()

    def cancel(job: dict[str, object]) -> bool:
        nonlocal target, worker
        status = str(job.get("status") or "")
        if status == "cancelled":
            return False
        if status in {"completed", "failed"}:
            raise HarnessError(f"Cursor job {job_id} is already {status}")
        if status not in ACTIVE_STATUSES:
            raise HarnessError(f"Cursor job {job_id} cannot be cancelled")
        if job.get("fork_committed") and job.get("fork_operation_state") in {
            "submitted",
            "ambiguous",
            "failed_observing",
        }:
            raise HarnessError(
                f"Cursor job {job_id} cannot be cancelled while its committed "
                "fork submission is being reconciled"
            )
        target = str(job.get("herdr_target") or "")
        worker = dict(job)
        release_pending = bool(
            target
            or job.get("worktree_path")
            or job.get("worker_token")
            or _has_uncertain_operation(job)
        )
        reconciliation_pending = _has_uncertain_operation(job)
        job.update(
            {
                "status": "cancelled",
                "result": (
                    f"Cursor job {job_id} was cancelled; external operation "
                    "reconciliation is pending."
                    if reconciliation_pending
                    else f"Cursor job {job_id} was cancelled."
                ),
                "completed_at": cancelled_at,
                "foreground_until": cancelled_at
                + CURSOR_FOREGROUND_SECONDS
                + FOREGROUND_GRACE_SECONDS,
                "target_release_pending": release_pending,
                "target_release_token": release_token if release_pending else None,
                "target_release_owner_pid": (os.getpid() if release_pending else None),
                "target_release_owner_start": (
                    _process_identity(os.getpid()) if release_pending else None
                ),
                "cancellation_reconciliation_pending": reconciliation_pending,
            }
        )
        if (
            job.get("github_pull_request")
            and job.get("worktree_path")
            and job.get("pull_request_worktree_state") != "quarantined"
        ):
            job["pull_request_worktree_state"] = "retained"
        job.pop("reconcile", None)
        _prepare_delivery(job)
        return True

    updated = _mutate_job(job_id, cancel)
    if updated is None:
        current = read_job(job_id)
        return str(current.get("result") or f"Cursor job {job_id} was cancelled.")
    worker_stopped = True
    if worker is not None and worker.get("worker_token"):
        worker_stopped = _stop_worker(worker)

    def clear_stopped_worker(job: dict[str, object]) -> bool:
        if job.get("status") != "cancelled" or not worker_stopped:
            return False
        _clear_worker(job)
        return True

    _mutate_job(job_id, clear_stopped_worker)
    if updated.get("target_release_pending"):
        _cancel_target_and_release(
            job_id,
            target,
            release_token,
            worker_stopped=worker_stopped,
        )
    return str(updated["result"])


def job_status(job_id: str | None = None) -> str:
    if job_id:
        job = read_job(job_id)
        return f"Cursor job {job_id} is {str(job.get('status') or 'unknown').replace('_', ' ')}."
    jobs = active_jobs()
    if not jobs:
        return "There are no active Cursor jobs."
    return (
        "Active Cursor jobs: "
        + "; ".join(
            f"{job.get('id')} is {str(job.get('status')).replace('_', ' ')}"
            for job in jobs
        )
        + "."
    )


def acknowledge_worktree_quarantine(job_id: str) -> None:
    def acknowledge(job: dict[str, object]) -> bool:
        if job.get("worktree_provision_state") != "quarantined" or not job.get(
            "worktree_manual_inspection_required"
        ):
            return False
        job.update(
            {
                "worktree_provision_state": "retained",
                "worktree_manual_inspection_required": False,
                "worktree_quarantine_acknowledged_at": time.time(),
            }
        )
        return True

    if _mutate_job(job_id, acknowledge) is None:
        raise HarnessError(
            f"Cursor job {job_id} has no worktree quarantine to acknowledge"
        )


def resolve_manual_reconciliation(
    job_id: str,
    operation: str,
    token: str,
    outcome: str,
) -> dict[str, object]:
    """Record a verified outcome without performing an external side effect."""
    state_keys = {
        "agent": "agent_dispatch_state",
        "fork": "fork_operation_state",
        "worktree": "worktree_provision_state",
    }
    if operation not in state_keys:
        raise HarnessError("manual reconciliation operation is invalid")
    if outcome not in {"confirmed_absent", "materialized"}:
        raise HarnessError("manual reconciliation outcome is invalid")
    now = time.time()

    def resolve(job: dict[str, object]) -> bool:
        state_key = state_keys[operation]
        if (
            job.get("manual_reconcile_operation") != operation
            or not secrets.compare_digest(
                str(job.get("manual_reconcile_token") or ""), token
            )
            or job.get(state_key) != "manual_required"
        ):
            return False
        if outcome == "confirmed_absent":
            job[state_key] = "confirmed_absent"
            job[f"{operation}_confirmed_absent_at"] = now
        else:
            job[state_key] = "retained"
            job[f"{operation}_retained_at"] = now
            if operation == "fork":
                target = str(job.get("fork_operation_target") or "")
                if not target:
                    return False
                job["fork_exists"] = True
                job["fork_repository"] = target
            elif operation == "agent" and not job.get("herdr_target"):
                return False
            elif operation == "worktree" and not job.get("worktree_path"):
                return False
        job.update(
            {
                "manual_reconcile_operation": None,
                "manual_reconcile_token": None,
                "manual_reconcile_resolved_at": now,
                "manual_reconcile_outcome": outcome,
                "cancellation_reconciliation_pending": False,
                "target_release_pending": False,
                "target_release_token": None,
                "target_release_owner_pid": None,
                "target_release_owner_start": None,
                "worker_operation": None,
            }
        )
        _clear_worker(job)
        _refresh_failed_reconciliation_message(job)
        _prepare_delivery(job)
        return True

    resolved = _mutate_job(job_id, resolve)
    if resolved is None:
        raise HarnessError(f"Cursor job {job_id} manual reconciliation fence is stale")
    return resolved


def mark_delivered(job_id: str) -> dict[str, object]:
    def deliver(job: dict[str, object]) -> bool:
        job.update(
            {
                "delivered": True,
                "delivery_claim_token": None,
                "delivery_claimed_at": None,
                "delivery_retry_at": 0,
            }
        )
        return True

    delivered = _mutate_job(job_id, deliver)
    assert delivered is not None
    return delivered


def claim_delivery(
    job_id: str | None = None, *, foreground: bool = False
) -> dict[str, object] | None:
    if not JOBS_DIR.is_dir():
        return None
    now = time.time()
    with locked(JOBS_DIR):
        jobs = read_all_unlocked(JOBS_DIR)
        if job_id is not None:
            jobs = [job for job in jobs if job.get("id") == job_id]
        jobs.sort(
            key=lambda job: float(
                str(job.get("completed_at") or job.get("created_at") or 0)
            )
        )
        for job in jobs:
            status = str(job.get("status") or "")
            if status not in DELIVERABLE_STATUSES or job.get("delivered"):
                continue
            if not foreground and now < float(str(job.get("foreground_until") or 0)):
                continue
            if now < float(str(job.get("delivery_retry_at") or 0)):
                continue
            completed_age = now - float(str(job.get("completed_at") or now))
            if (
                not foreground
                and status not in {"awaiting_user", "blocked"}
                and completed_age < 1
            ):
                continue
            claimed_at = float(str(job.get("delivery_claimed_at") or 0))
            if job.get("delivery_claim_token") and (
                now - claimed_at < DELIVERY_CLAIM_SECONDS
            ):
                continue
            token = uuid.uuid4().hex
            job.update(
                {
                    "delivery_claim_token": token,
                    "delivery_claimed_at": now,
                    "delivery_attempts": int(str(job.get("delivery_attempts") or 0))
                    + 1,
                    "revision": int(str(job.get("revision") or 0)) + 1,
                }
            )
            write_unlocked(job_path(str(job["id"])), job)
            claimed = dict(job)
            claimed["_delivery_token"] = token
            return claimed
    return None


def acknowledge_delivery(job_id: str, token: str) -> bool:
    def acknowledge(job: dict[str, object]) -> bool:
        if job.get("delivery_claim_token") != token or job.get("delivered"):
            return False
        job.update(
            {
                "delivered": True,
                "delivery_claim_token": None,
                "delivery_claimed_at": None,
                "delivery_retry_at": 0,
                "delivered_at": time.time(),
            }
        )
        return True

    return _mutate_job(job_id, acknowledge) is not None


def release_delivery(job_id: str, token: str, *, retry: bool = True) -> bool:
    def release(job: dict[str, object]) -> bool:
        if job.get("delivery_claim_token") != token or job.get("delivered"):
            return False
        job.update(
            {
                "delivery_claim_token": None,
                "delivery_claimed_at": None,
                "delivery_retry_at": (
                    time.time() + DELIVERY_RETRY_SECONDS if retry else 0
                ),
            }
        )
        return True

    return _mutate_job(job_id, release) is not None


def acknowledge_deliveries(claims: DeliveryClaims) -> None:
    for job_id, token in claims:
        acknowledge_delivery(job_id, token)
    claims.clear()


def release_deliveries(claims: DeliveryClaims) -> None:
    for job_id, token in claims:
        release_delivery(job_id, token)
    claims.clear()


def _defer_or_acknowledge(
    job_id: str, claims: DeliveryClaims | None
) -> dict[str, object] | None:
    claimed = claim_delivery(job_id, foreground=True)
    if claimed is None:
        return None
    token = str(claimed["_delivery_token"])
    if claims is None:
        acknowledge_delivery(job_id, token)
    else:
        claims.append((job_id, token))
    return claimed


def recover_jobs() -> None:
    if not JOBS_DIR.is_dir():
        return
    now = time.time()
    for existing in _all_jobs():
        if existing.get("target_release_pending") and _has_uncertain_operation(
            existing
        ):
            _reconcile_uncertain_operations(existing, now=now)
    for existing in _all_jobs():
        if (
            existing.get("status") in WORKER_STATUSES
            and not existing.get("worker_token")
            and _worker_is_alive(existing)
        ):
            _stop_legacy_worker(str(existing["id"]))
        elif (
            existing.get("status") == "cancelled"
            and existing.get("target_release_pending")
            and existing.get("worker_token")
            and _worker_is_alive(existing)
        ):
            _stop_worker(existing)
    launch: list[str] = []
    releases: list[tuple[str, str, str]] = []
    with locked(JOBS_DIR):
        for job in read_all_unlocked(JOBS_DIR):
            status = str(job.get("status") or "")
            changed = False
            should_launch = False
            if (
                job.get("target_release_pending")
                and not _target_release_owner_alive(job)
                and not _worker_is_alive(job)
                and not _has_uncertain_operation(job)
                and not _manual_target_fence(job)
            ):
                release_token = uuid.uuid4().hex
                job.update(
                    {
                        "target_release_token": release_token,
                        "target_release_owner_pid": os.getpid(),
                        "target_release_owner_start": _process_identity(os.getpid()),
                    }
                )
                _clear_worker(job)
                releases.append(
                    (
                        str(job["id"]),
                        str(job.get("herdr_target") or ""),
                        release_token,
                    )
                )
                changed = True
            if status == "blocked":
                if (
                    job.get("delivered")
                    and job.get("herdr_target")
                    and now >= float(str(job.get("next_reconcile_at") or 0))
                ):
                    job.update(
                        {
                            "status": "queued",
                            "reconcile": True,
                            "queued_at": now,
                            "next_reconcile_at": now + DELIVERY_RETRY_SECONDS,
                        }
                    )
                    _clear_worker(job)
                    changed = True
                    should_launch = True
            elif status == "queued":
                if not job.get("worker_token") or not _worker_is_alive(job):
                    _clear_worker(job)
                    job["queued_at"] = float(str(job.get("queued_at") or now))
                    changed = True
                    should_launch = True
            elif status == "routing":
                if not _worker_is_alive(job):
                    _clear_worker(job)
                    job.update({"status": "queued", "queued_at": now})
                    job.pop("reconcile", None)
                    should_launch = True
                    changed = True
            elif status in {"running", "reconciling"}:
                if not _worker_is_alive(job):
                    _clear_worker(job)
                    if job.get("herdr_target"):
                        job.update(
                            {
                                "status": "queued",
                                "reconcile": True,
                                "queued_at": now,
                            }
                        )
                        should_launch = True
                    else:
                        message = "Cursor job was interrupted before an agent started"
                        job.update(
                            {
                                "status": "failed",
                                "error": message,
                                "result": message,
                                "completed_at": now,
                            }
                        )
                        job.pop("reconcile", None)
                        _prepare_delivery(job, now=now)
                    changed = True
            if changed:
                job["revision"] = int(str(job.get("revision") or 0)) + 1
                write_unlocked(job_path(str(job["id"])), job)
            if should_launch:
                launch.append(str(job["id"]))
    for release_job_id, target, release_token in releases:
        _cancel_target_and_release(release_job_id, target, release_token)
    for queued_job_id in launch:
        try:
            launch_worker(queued_job_id)
        except Exception:
            # launch_worker has already persisted a deliverable failure.
            continue


def pending_results() -> list[dict[str, object]]:
    recover_jobs()
    results: list[dict[str, object]] = []
    while True:
        claimed = claim_delivery()
        if claimed is None:
            break
        results.append(claimed)
    return results


def cursor_turn(
    text: str,
    session_id: str | None = None,
    *,
    repository: str | None = None,
    github_repository: str | None = None,
    github_issue: int | None = None,
    github_issue_context: str | None = None,
    fork_requested: bool = False,
    github_pull_request: int | None = None,
    agent: str | None = None,
    utterance: str | None = None,
    context_repository: str | None = None,
    action: str = "submit",
    job_id: str | None = None,
    delivery_claims: DeliveryClaims | None = None,
) -> tuple[str, str | None]:
    if action == "status":
        return job_status(job_id), session_id
    if action == "cancel":
        if not job_id:
            raise HarnessError("a Cursor job ID is required to cancel")
        result = cancel_job(job_id)
        _defer_or_acknowledge(job_id, delivery_claims)
        return result, None
    if action == "reply":
        reply_id = job_id or session_id
        if not reply_id:
            raise HarnessError("a Cursor job ID is required for a reply")
        reply_job(reply_id, text, trusted_utterance=utterance)
        job_id = reply_id
    else:
        job_id = start_job(
            text,
            repository=repository,
            github_repository=github_repository,
            github_issue=github_issue,
            github_issue_context=github_issue_context,
            fork_requested=fork_requested,
            github_pull_request=github_pull_request,
            agent=agent,
            utterance=utterance,
            context_repository=context_repository,
        )
    started = time.perf_counter()
    deadline = time.monotonic() + CURSOR_FOREGROUND_SECONDS
    while time.monotonic() < deadline:
        job = read_job(job_id)
        status = job.get("status")
        if status == "completed":
            job = _defer_or_acknowledge(job_id, delivery_claims) or job
            print(
                json.dumps(
                    {
                        "stage": "cursor",
                        "job_id": job_id,
                        "background": False,
                        "seconds": round(time.perf_counter() - started, 3),
                    }
                )
            )
            return str(job.get("result") or "").strip(), None
        if status == "awaiting_user":
            job = _defer_or_acknowledge(job_id, delivery_claims) or job
            return str(job.get("question") or job.get("result") or "").strip(), job_id
        if status == "blocked":
            job = _defer_or_acknowledge(job_id, delivery_claims) or job
            return str(job.get("result") or "Cursor needs attention in Herdr"), None
        if status == "failed":
            job = _defer_or_acknowledge(job_id, delivery_claims) or job
            raise HarnessError(str(job.get("error") or "Cursor failed"))
        if status == "cancelled":
            job = _defer_or_acknowledge(job_id, delivery_claims) or job
            return str(job.get("result") or "Cursor job was cancelled"), None
        time.sleep(0.1)
    _mutate_job(
        job_id,
        lambda job: (
            job.update({"foreground_until": 0, "updated_at": time.time()}) is None
        ),
    )
    print(
        json.dumps(
            {
                "stage": "cursor",
                "job_id": job_id,
                "background": True,
                "seconds": round(time.perf_counter() - started, 3),
            }
        )
    )
    return (
        f"Cursor is still working on job {job_id}. I will report back when it finishes.",
        None,
    )
