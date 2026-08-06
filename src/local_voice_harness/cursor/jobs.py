from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from ..config import CURSOR_FOREGROUND_SECONDS, JOBS_DIR
from ..errors import HarnessError
from ..integrations.github import GitHubClient
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
DeliveryClaims = list[tuple[str, str]]


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
    client: HerdrClient, job: dict[str, object], *, wait: bool
) -> tuple[str, str]:
    target = str(job.get("herdr_target") or "")
    if not target:
        raise HarnessError("Cursor job has no Herdr agent")
    agent = client.get_agent(target)
    if wait and agent.get("agent_status") == "working":
        result = client.run_json(
            "agent", "wait", target, "--timeout", "900000", timeout=910
        )
        agent = dict(result.get("agent") or {})
    output = client.run_text(
        "agent", "read", target, "--source", "recent-unwrapped", "--lines", "160"
    )
    return output, str(agent.get("agent_status") or "unknown")


def _process_identity(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    closing = stat.rfind(")")
    fields = stat[closing + 2 :].split()
    return fields[19] if closing >= 0 and len(fields) > 19 else None


def _worker_is_alive(job: dict[str, object]) -> bool:
    pid = int(str(job.get("worker_pid") or 0))
    expected = str(job.get("worker_process_start") or "")
    if not pid:
        return False
    if expected:
        return _process_identity(pid) == expected
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        return False
    return (
        b"local_voice_harness.cursor.worker" in command
        and str(job.get("id") or "").encode() in command
    )


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
        job.update(
            {
                "status": "failed",
                "error": message,
                "result": message,
                "completed_at": time.time(),
            }
        )
        job.pop("reconcile", None)
        _clear_worker(job)
        _prepare_delivery(job)

    _worker_change(job_id, token, WORKER_STATUSES, fail)


def _reserve_worker_target(
    job_id: str,
    token: str,
    selection: AgentSelection,
    repository: Path,
    issue_key: str | None,
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
            if (
                other.get("id") != job_id
                and other.get("herdr_target") == target
                and (
                    other.get("status") in ACTIVE_STATUSES
                    or other.get("target_release_pending")
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
                "revision": int(str(job.get("revision") or 0)) + 1,
            }
        )
        write_unlocked(job_path(job_id), job)
        return dict(job)


def run_worker(job_id: str, claim_token: str | None = None) -> None:
    claimed = _begin_worker(job_id, claim_token)
    if claimed is None:
        return
    job, token = claimed
    try:
        client = HerdrClient()
        client.ensure_server()
        if job.get("reconcile"):
            output, agent_status = read_agent_completion(client, job, wait=True)
            _worker_complete(
                job_id,
                token,
                output=output,
                agent_status=agent_status,
                preserve_blocked_delivery=bool(job.get("delivered")),
            )
            return

        turn = int(str(job.get("turn") or 0)) + 1
        token = f"{job_id}-{turn}"
        worker_token = claimed[1]

        def begin_turn(current: dict[str, object]) -> None:
            current.update({"turn": turn, "turn_token": token})

        updated = _worker_change(job_id, worker_token, {"routing"}, begin_turn)
        if updated is None:
            return
        job = updated
        continuation = bool(job.get("continuation"))
        target = str(job.get("herdr_target") or "")
        if not target:
            repository: Path | None = None
            repositories: list[Path] = []
            candidates: list[Path] = []
            hint = str(job.get("repository_hint") or "").strip() or None
            task = str(job.get("request") or "")
            issue_key = str(job.get("issue_key") or "") or extract_linear_issue(task)
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
                provisioned_pr = GitHubClient().provision_pull_request(
                    github_repository, number
                )
                repository = provisioned_pr.checkout
                issue_key = None

                def record_pull_request(current: dict[str, object]) -> None:
                    current.update(
                        {
                            "github_repository": (
                                provisioned_pr.source.name_with_owner
                            ),
                            "repository": str(provisioned_pr.checkout),
                            "pull_request_branch": provisioned_pr.branch,
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
                provisioned = GitHubClient().provision_public_fork(github_repository)
                repository = provisioned.checkout

                def record_provisioning(current: dict[str, object]) -> None:
                    current.update(
                        {
                            "github_repository": (provisioned.source.name_with_owner),
                            "fork_repository": provisioned.fork.name_with_owner,
                            "repository": str(provisioned.checkout),
                        }
                    )

                updated = _worker_change(
                    job_id, worker_token, {"routing"}, record_provisioning
                )
                if updated is None:
                    return
                job = updated
            else:
                repositories = client.repository_roots()
                repository, candidates = client.resolve_repository(
                    hint, task, repositories
                )
            if (
                repository is None
                and issue_key
                and not hint
                and not job.get("fork_requested")
                and not job.get("github_pull_request")
            ):
                repository, _confidence, reason = client.infer_repository(
                    issue_key,
                    repositories,
                    token=f"{job_id}-route",
                    reserved=reserved_targets(job_id),
                )
                if repository is None:
                    question = repository_question(repositories, reason)
                    _worker_question(
                        job_id,
                        worker_token,
                        question,
                        clarification_kind="repository",
                    )
                    return
            if repository is None:
                question = repository_question(
                    candidates or repositories,
                    "The repository could not be determined confidently."
                    if hint or issue_key
                    else "",
                )
                _worker_question(
                    job_id,
                    worker_token,
                    question,
                    clarification_kind="repository",
                )
                return
            for _attempt in range(3):
                selection = client.ensure_agent(
                    repository,
                    issue_key=issue_key or None,
                    agent_hint=str(job.get("agent_hint") or "") or None,
                    reserved=reserved_targets(job_id),
                    worktree_branch=(str(job.get("worktree_branch") or "") or None),
                    worktree_label=(str(job.get("worktree_label") or "") or None),
                )
                reserved = _reserve_worker_target(
                    job_id, worker_token, selection, repository, issue_key
                )
                if reserved is not None:
                    job = reserved
                    target = selection.target
                    break
            else:
                raise HarnessError("could not reserve a Cursor agent")

        def mark_running(current: dict[str, object]) -> None:
            current["status"] = "running"
            current.pop("continuation", None)

        if _worker_change(job_id, worker_token, {"routing"}, mark_running) is None:
            return
        outcome = client.prompt_and_wait(
            target,
            cursor_prompt(
                str(job.get("request") or ""), token, continuation=continuation
            ),
            token=token,
        )
        _worker_complete(
            job_id,
            worker_token,
            output=outcome.output,
            agent_status=outcome.status,
        )
    except Exception as exc:
        _worker_fail(job_id, claimed[1], exc)


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
    fork_requested: bool = False,
    github_pull_request: int | None = None,
    agent: str | None = None,
) -> str:
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    write_job(
        {
            "id": job_id,
            "schema_version": 2,
            "revision": 0,
            "request": text,
            "repository_hint": repository,
            "github_repository": github_repository,
            "fork_requested": fork_requested,
            "github_pull_request": github_pull_request,
            "worktree_branch": (f"voice/github-{job_id}" if fork_requested else None),
            "worktree_label": (f"github-{job_id[:6]}" if fork_requested else None),
            "agent_hint": agent,
            "issue_key": extract_linear_issue(text),
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


def reply_job(job_id: str, text: str) -> None:
    now = time.time()

    def reply(job: dict[str, object]) -> bool:
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
    launch_worker(job_id)


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
    cancelled_at = time.time()

    def cancel(job: dict[str, object]) -> bool:
        nonlocal target
        status = str(job.get("status") or "")
        if status == "cancelled":
            return False
        if status in {"completed", "failed"}:
            raise HarnessError(f"Cursor job {job_id} is already {status}")
        if status not in ACTIVE_STATUSES:
            raise HarnessError(f"Cursor job {job_id} cannot be cancelled")
        target = str(job.get("herdr_target") or "")
        job.update(
            {
                "status": "cancelled",
                "result": f"Cursor job {job_id} was cancelled.",
                "completed_at": cancelled_at,
                "foreground_until": cancelled_at
                + CURSOR_FOREGROUND_SECONDS
                + FOREGROUND_GRACE_SECONDS,
                "target_release_pending": bool(target),
                "target_release_owner_pid": os.getpid() if target else None,
                "target_release_owner_start": (
                    _process_identity(os.getpid()) if target else None
                ),
            }
        )
        job.pop("reconcile", None)
        _clear_worker(job)
        _prepare_delivery(job)
        return True

    updated = _mutate_job(job_id, cancel)
    if updated is None:
        current = read_job(job_id)
        return str(current.get("result") or f"Cursor job {job_id} was cancelled.")
    if target:
        try:
            try:
                client = HerdrClient()
                client.ensure_server()
                client.cancel_agent(target)
            except HerdrError:
                pass
        finally:

            def release_target(job: dict[str, object]) -> bool:
                if job.get("status") != "cancelled":
                    return False
                job.update(
                    {
                        "target_release_pending": False,
                        "target_release_owner_pid": None,
                        "target_release_owner_start": None,
                    }
                )
                return True

            _mutate_job(job_id, release_target)
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
    for existing in _all_jobs():
        if (
            existing.get("status") in WORKER_STATUSES
            and not existing.get("worker_token")
            and _worker_is_alive(existing)
        ):
            _stop_legacy_worker(str(existing["id"]))
    now = time.time()
    launch: list[str] = []
    with locked(JOBS_DIR):
        for job in read_all_unlocked(JOBS_DIR):
            status = str(job.get("status") or "")
            changed = False
            should_launch = False
            if job.get("target_release_pending") and not _target_release_owner_alive(
                job
            ):
                job.update(
                    {
                        "target_release_pending": False,
                        "target_release_owner_pid": None,
                        "target_release_owner_start": None,
                    }
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
    for queued_job_id in launch:
        try:
            launch_worker(queued_job_id)
        except Exception:
            # launch_worker has already persisted a deliverable failure.
            continue


def pending_results() -> list[dict[str, object]]:
    recover_jobs()
    claimed = claim_delivery()
    return [claimed] if claimed is not None else []


def cursor_turn(
    text: str,
    session_id: str | None = None,
    *,
    repository: str | None = None,
    github_repository: str | None = None,
    fork_requested: bool = False,
    github_pull_request: int | None = None,
    agent: str | None = None,
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
        reply_job(reply_id, text)
        job_id = reply_id
    else:
        if github_repository or fork_requested or github_pull_request:
            job_id = start_job(
                text,
                repository=repository,
                github_repository=github_repository,
                fork_requested=fork_requested,
                github_pull_request=github_pull_request,
                agent=agent,
            )
        else:
            job_id = start_job(text, repository=repository, agent=agent)
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
