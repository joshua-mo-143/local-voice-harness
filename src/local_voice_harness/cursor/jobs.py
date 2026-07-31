from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from ..config import CURSOR_FOREGROUND_SECONDS, JOBS_DIR
from ..errors import HarnessError
from ..integrations.herdr import (
    HerdrClient,
    HerdrError,
    extract_linear_issue,
    extract_marker,
)
from .prompts import cursor_prompt


def job_path(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{12}", job_id):
        raise HarnessError("invalid Cursor job ID")
    return JOBS_DIR / f"{job_id}.json"


def read_job(job_id: str) -> dict[str, object]:
    try:
        return json.loads(job_path(job_id).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"could not read Cursor job {job_id}") from exc


def write_job(job: dict[str, object]) -> None:
    path = job_path(str(job["id"]))
    JOBS_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(job, sort_keys=True))
    os.replace(temporary, path)


def active_jobs() -> list[dict[str, object]]:
    if not JOBS_DIR.is_dir():
        return []
    jobs = []
    for path in JOBS_DIR.glob("*.json"):
        try:
            job = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if job.get("status") in {
            "queued",
            "routing",
            "running",
            "reconciling",
            "awaiting_user",
            "blocked",
        }:
            jobs.append(job)
    return jobs


def reserved_targets(exclude_job_id: str | None = None) -> set[str]:
    return {
        str(job["herdr_target"])
        for job in active_jobs()
        if job.get("herdr_target")
        and job.get("id") != exclude_job_id
        and job.get("status") not in {"completed", "failed", "cancelled"}
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
                "delivered": False,
                "updated_at": time.time(),
            }
        )
    elif summary and summary_position > question_position:
        job.update(
            {
                "status": "completed",
                "result": summary,
                "completed_at": time.time(),
                "delivered": False,
            }
        )
    else:
        job.update(
            {
                "status": "blocked",
                "result": (
                    f"Herdr agent {job.get('herdr_target') or 'Cursor'} needs attention; "
                    f"it settled as {agent_status} without a voice summary."
                ),
                "completed_at": time.time(),
                "delivered": False,
            }
        )


def read_agent_completion(
    client: HerdrClient, job: dict[str, object], *, wait: bool
) -> None:
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
    complete_from_output(
        job, output=output, agent_status=str(agent.get("agent_status") or "unknown")
    )


def run_worker(job_id: str) -> None:
    job = read_job(job_id)
    job.update(
        {
            "status": "reconciling" if job.get("reconcile") else "routing",
            "worker_pid": os.getpid(),
            "started_at": time.time(),
        }
    )
    write_job(job)
    try:
        client = HerdrClient()
        client.ensure_server()
        if job.pop("reconcile", False):
            read_agent_completion(client, job, wait=True)
            job["worker_pid"] = None
            write_job(job)
            return

        turn = int(job.get("turn") or 0) + 1
        token = f"{job_id}-{turn}"
        job.update({"turn": turn, "turn_token": token})
        continuation = bool(job.pop("continuation", False))
        target = str(job.get("herdr_target") or "")
        if not target:
            repositories = client.repository_roots()
            hint = str(job.get("repository_hint") or "").strip() or None
            task = str(job.get("request") or "")
            repository, candidates = client.resolve_repository(hint, task, repositories)
            issue_key = str(job.get("issue_key") or "") or extract_linear_issue(task)
            if repository is None and issue_key and not hint:
                repository, _confidence, reason = client.infer_repository(
                    issue_key,
                    repositories,
                    token=f"{job_id}-route",
                    reserved=reserved_targets(job_id),
                )
                if repository is None:
                    question = repository_question(repositories, reason)
                    job.update(
                        {
                            "status": "awaiting_user",
                            "question": question,
                            "result": question,
                            "clarification_kind": "repository",
                            "delivered": False,
                            "updated_at": time.time(),
                            "worker_pid": None,
                        }
                    )
                    write_job(job)
                    return
            if repository is None:
                question = repository_question(
                    candidates or repositories,
                    "The repository could not be determined confidently."
                    if hint or issue_key
                    else "",
                )
                job.update(
                    {
                        "status": "awaiting_user",
                        "question": question,
                        "result": question,
                        "clarification_kind": "repository",
                        "delivered": False,
                        "updated_at": time.time(),
                        "worker_pid": None,
                    }
                )
                write_job(job)
                return
            selection = client.ensure_agent(
                repository,
                issue_key=issue_key or None,
                agent_hint=str(job.get("agent_hint") or "") or None,
                reserved=reserved_targets(job_id),
            )
            job.update(
                {
                    "repository": str(repository),
                    "issue_key": issue_key,
                    "herdr_target": selection.target,
                    "herdr_pane_id": selection.pane_id,
                    "herdr_workspace_id": selection.workspace_id,
                    "worktree_path": selection.worktree_path,
                    "agent_name": selection.name,
                }
            )
            target = selection.target
            write_job(job)
        job["status"] = "running"
        write_job(job)
        outcome = client.prompt_and_wait(
            target,
            cursor_prompt(str(job.get("request") or ""), token, continuation=continuation),
            token=token,
        )
        if read_job(job_id).get("status") == "cancelled":
            return
        complete_from_output(job, output=outcome.output, agent_status=outcome.status)
    except Exception as exc:
        try:
            if read_job(job_id).get("status") == "cancelled":
                return
        except HarnessError:
            pass
        job.update(
            {
                "status": "failed",
                "error": (str(exc) or type(exc).__name__)[:500],
                "result": (str(exc) or type(exc).__name__)[:500],
                "completed_at": time.time(),
                "delivered": False,
            }
        )
    job["worker_pid"] = None
    write_job(job)


def launch_worker(job_id: str) -> None:
    JOBS_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    log_handle = (JOBS_DIR / f"{job_id}.log").open("ab")
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "local_voice_harness.cursor.worker",
                job_id,
            ],
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()
    threading.Thread(target=process.wait, daemon=True).start()


def start_job(
    text: str, *, repository: str | None = None, agent: str | None = None
) -> str:
    job_id = uuid.uuid4().hex[:12]
    write_job(
        {
            "id": job_id,
            "request": text,
            "repository_hint": repository,
            "agent_hint": agent,
            "issue_key": extract_linear_issue(text),
            "status": "queued",
            "delivered": False,
            "created_at": time.time(),
        }
    )
    launch_worker(job_id)
    return job_id


def reply_job(job_id: str, text: str) -> None:
    job = read_job(job_id)
    if job.get("status") != "awaiting_user":
        raise HarnessError(f"Cursor job {job_id} is not waiting for a reply")
    if job.get("clarification_kind") == "repository":
        job.update({"repository_hint": text, "herdr_target": None, "continuation": False})
    else:
        job.update({"continuation": True, "request": text})
    job.update(
        {
            "status": "queued",
            "question": None,
            "clarification_kind": None,
            "delivered": False,
            "updated_at": time.time(),
        }
    )
    write_job(job)
    launch_worker(job_id)


def cancel_job(job_id: str) -> str:
    job = read_job(job_id)
    target = str(job.get("herdr_target") or "")
    if target and job.get("status") in {
        "running",
        "reconciling",
        "awaiting_user",
        "blocked",
    }:
        try:
            client = HerdrClient()
            client.ensure_server()
            client.cancel_agent(target)
        except HerdrError:
            pass
    job.update(
        {
            "status": "cancelled",
            "result": f"Cursor job {job_id} was cancelled.",
            "completed_at": time.time(),
            "delivered": True,
        }
    )
    write_job(job)
    return str(job["result"])


def job_status(job_id: str | None = None) -> str:
    if job_id:
        job = read_job(job_id)
        return f"Cursor job {job_id} is {str(job.get('status') or 'unknown').replace('_', ' ')}."
    jobs = active_jobs()
    if not jobs:
        return "There are no active Cursor jobs."
    return "Active Cursor jobs: " + "; ".join(
        f"{job.get('id')} is {str(job.get('status')).replace('_', ' ')}"
        for job in jobs
    ) + "."


def mark_delivered(job_id: str) -> dict[str, object]:
    job = read_job(job_id)
    job["delivered"] = True
    write_job(job)
    return job


def pending_results() -> list[dict[str, object]]:
    if not JOBS_DIR.is_dir():
        return []
    pending = []
    for path in JOBS_DIR.glob("*.json"):
        try:
            job = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        status = job.get("status")
        if (
            status == "blocked"
            and job.get("delivered")
            and job.get("herdr_target")
            and time.time() >= float(job.get("next_reconcile_at") or 0)
        ):
            job["next_reconcile_at"] = time.time() + 5
            try:
                client = HerdrClient()
                client.ensure_server()
                agent = client.get_agent(str(job["herdr_target"]))
                if agent.get("agent_status") == "working":
                    job.update(
                        {"status": "queued", "reconcile": True, "worker_pid": None}
                    )
                    write_job(job)
                    launch_worker(str(job["id"]))
                elif agent.get("agent_status") in {"idle", "done"}:
                    output = client.run_text(
                        "agent",
                        "read",
                        str(job["herdr_target"]),
                        "--source",
                        "recent-unwrapped",
                        "--lines",
                        "160",
                    )
                    complete_from_output(
                        job,
                        output=output,
                        agent_status=str(agent.get("agent_status")),
                    )
                    if job.get("status") == "blocked":
                        job["delivered"] = True
                    write_job(job)
                else:
                    write_job(job)
            except (HarnessError, HerdrError):
                write_job(job)
            status = job.get("status")
        if status in {"queued", "routing", "running", "reconciling"}:
            worker_pid = int(job.get("worker_pid") or 0)
            age = time.time() - float(job.get("created_at") or time.time())
            if status == "queued" and not worker_pid and age > 10:
                job.update(
                    {
                        "status": "failed",
                        "error": "Cursor job did not start",
                        "result": "Cursor job did not start",
                        "completed_at": time.time(),
                    }
                )
                write_job(job)
                status = "failed"
            elif worker_pid and age > 5:
                try:
                    os.kill(worker_pid, 0)
                except ProcessLookupError:
                    if job.get("herdr_target"):
                        job.update(
                            {"status": "queued", "reconcile": True, "worker_pid": None}
                        )
                        write_job(job)
                        launch_worker(str(job["id"]))
                    else:
                        job.update(
                            {
                                "status": "failed",
                                "error": "Cursor job was interrupted before an agent started",
                                "result": "Cursor job was interrupted before an agent started",
                                "completed_at": time.time(),
                            }
                        )
                        write_job(job)
                    status = job.get("status")
                except PermissionError:
                    pass
        completed_age = time.time() - float(job.get("completed_at") or time.time())
        if (
            status in {"completed", "failed", "blocked", "awaiting_user", "cancelled"}
            and not job.get("delivered")
            and (status in {"awaiting_user", "blocked"} or completed_age >= 1)
        ):
            pending.append(job)
    return sorted(
        pending,
        key=lambda job: float(job.get("completed_at") or job.get("created_at") or 0),
    )


def cursor_turn(
    text: str,
    session_id: str | None = None,
    *,
    repository: str | None = None,
    agent: str | None = None,
    action: str = "submit",
    job_id: str | None = None,
) -> tuple[str, str | None]:
    if action == "status":
        return job_status(job_id), session_id
    if action == "cancel":
        if not job_id:
            raise HarnessError("a Cursor job ID is required to cancel")
        return cancel_job(job_id), None
    if action == "reply":
        reply_id = job_id or session_id
        if not reply_id:
            raise HarnessError("a Cursor job ID is required for a reply")
        reply_job(reply_id, text)
        job_id = reply_id
    else:
        job_id = start_job(text, repository=repository, agent=agent)
    started = time.perf_counter()
    deadline = time.monotonic() + CURSOR_FOREGROUND_SECONDS
    while time.monotonic() < deadline:
        job = read_job(job_id)
        status = job.get("status")
        if status == "completed":
            mark_delivered(job_id)
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
            mark_delivered(job_id)
            return str(job.get("question") or job.get("result") or "").strip(), job_id
        if status == "blocked":
            mark_delivered(job_id)
            return str(job.get("result") or "Cursor needs attention in Herdr"), None
        if status == "failed":
            mark_delivered(job_id)
            raise HarnessError(str(job.get("error") or "Cursor failed"))
        if status == "cancelled":
            mark_delivered(job_id)
            return str(job.get("result") or "Cursor job was cancelled"), None
        time.sleep(0.1)
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
    return f"Cursor is still working on job {job_id}. I will report back when it finishes.", None
