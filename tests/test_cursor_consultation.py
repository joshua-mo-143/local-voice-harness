from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from local_voice_harness.cursor import consultation
from local_voice_harness.cursor.model import CursorJob, JobStatus
from local_voice_harness.cursor.store import JobStore
from local_voice_harness.errors import HarnessError
from local_voice_harness.integrations.herdr import AgentSelection, PromptOutcome
from local_voice_harness.questions import (
    Choice,
    Question,
    QuestionKind,
    QuestionOrigin,
    QuestionSensitivity,
)


def _store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "jobs", tmp_path / "legacy")


def _awaiting(
    store: JobStore,
    *,
    job_id: str = "aaaaaaaaaaaa",
    delivered: bool = True,
    choices: tuple[Choice, ...] = (
        Choice("safe", "Use the safe design"),
        Choice("fast", "Use the fast design"),
    ),
) -> CursorJob:
    question = Question(
        id=f"question-{job_id}",
        text="Which design should I use?",
        kind=QuestionKind.MULTIPLE_CHOICE,
        choices=choices,
        sensitivity=QuestionSensitivity.ARCHITECTURE,
        origin=QuestionOrigin("cursor", job_id, f"{job_id}-turn-1"),
        asked_at=10,
    )
    return store.create(
        CursorJob.from_dict(
            {
                "id": job_id,
                "request": "build it",
                "status": JobStatus.AWAITING_USER.value,
                "created_at": 1,
                "updated_at": 10,
                "delivered": delivered,
                "question": question.text,
                "result": question.text,
                "clarification_kind": "agent",
                "turn": 1,
                "turn_token": question.origin.turn_token,
                "repository": f"/projects/{job_id}",
                "worktree_path": f"/worktrees/{job_id}",
                "worktree_label": job_id,
                "worktree_workspace_id": f"workspace-{job_id}",
                "worktree_root_pane_id": f"root-pane-{job_id}",
                "worktree_provision_state": "ready",
                "voice_question": question.to_dict(),
            }
        )
    )


def test_pending_snapshot_requires_one_delivered_target(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _awaiting(store)

    snapshot = consultation.pending_question_snapshot(store, "aaaaaaaaaaaa")

    assert snapshot is not None
    assert snapshot.question_id == "question-aaaaaaaaaaaa"
    assert snapshot.choices[0] == Choice("safe", "Use the safe design")
    unique = consultation.pending_question_snapshot(store, None)
    assert unique is not None
    assert unique.question_id == snapshot.question_id
    assert consultation.pending_question_snapshot(store, "bbbbbbbbbbbb") is None


def test_pending_snapshot_fails_closed_for_undelivered_or_competing_jobs(
    tmp_path: Path,
) -> None:
    undelivered_store = _store(tmp_path / "undelivered")
    _awaiting(undelivered_store, delivered=False)
    assert (
        consultation.pending_question_snapshot(undelivered_store, "aaaaaaaaaaaa")
        is None
    )

    competing_store = _store(tmp_path / "competing")
    _awaiting(competing_store)
    _awaiting(competing_store, job_id="bbbbbbbbbbbb")
    assert (
        consultation.pending_question_snapshot(competing_store, "aaaaaaaaaaaa") is None
    )


def test_pending_consultation_revalidates_identity_and_preserves_question(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    original = _awaiting(store)
    before = store.get(original.id).to_dict()
    snapshot = consultation.pending_question_snapshot(store, original.id)
    assert snapshot is not None
    client = mock.Mock()
    client.workspace_for.return_value = {"workspace_id": "workspace-aaaaaaaaaaaa"}
    client.start_fresh_agent.return_value = AgentSelection(
        "consultant",
        "pane-2",
        "workspace-aaaaaaaaaaaa",
        "/worktrees/aaaaaaaaaaaa",
        "consultant",
    )
    client.prompt_and_wait.return_value = PromptOutcome(
        "done",
        "I recommend the safe design.",
        None,
        "",
    )

    answer = consultation.consult_pending_question(
        client, store, snapshot, "Which option do you recommend?"
    )

    assert answer == "I recommend the safe design."
    assert store.get(original.id).to_dict() == before
    assert client.start_fresh_agent.call_args.kwargs["mode"] == "ask"
    prompt_call = client.prompt_and_wait.call_args
    prompt = prompt_call.args[1]
    assert "Do not edit files" in prompt
    assert '"id": "safe"' in prompt
    assert "abstain" in prompt

    current = store.get(original.id)
    assert current.voice_question is not None
    changed = current.evolve(
        voice_question={
            **current.voice_question,
            "choices": [
                {"id": "safe", "label": "Use a newly changed design"},
                {"id": "fast", "label": "Use the fast design"},
            ],
        }
    )
    store.update(current.id, lambda _job: changed)
    before_submit = client.prompt_and_wait.call_args.kwargs["before_submit"]
    with pytest.raises(HarnessError, match="changed before consultation"):
        before_submit(1)


def test_consultation_allows_abstention_without_answering_question(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    original = _awaiting(store)
    snapshot = consultation.pending_question_snapshot(store, original.id)
    assert snapshot is not None
    client = mock.Mock()
    client.workspace_for.return_value = {"workspace_id": "workspace-aaaaaaaaaaaa"}
    client.start_fresh_agent.return_value = AgentSelection(
        "consultant",
        "pane-2",
        "workspace-aaaaaaaaaaaa",
        "/worktrees/aaaaaaaaaaaa",
        "consultant",
    )
    client.prompt_and_wait.return_value = PromptOutcome(
        "idle",
        "I cannot recommend one without knowing your latency preference.",
        None,
        "",
    )

    result = consultation.consult_pending_question(
        client, store, snapshot, "What do you think?"
    )

    assert result.startswith("I cannot recommend")
    assert store.get(original.id).status == JobStatus.AWAITING_USER


def test_workspace_target_prefers_explicit_focus_then_retained_completion() -> None:
    client = mock.Mock()
    client.focused_checkout.return_value = Path("/projects/focused")
    client.workspace_for.return_value = {"workspace_id": "focused-workspace"}
    completed = CursorJob.from_dict(
        {
            "id": "cccccccccccc",
            "request": "finished",
            "status": "completed",
            "created_at": 1,
            "completed_at": 2,
            "delivered": True,
            "result": "done",
            "repository": "/projects/example",
            "worktree_path": "/worktrees/example",
            "worktree_label": "retained",
            "worktree_workspace_id": "retained-workspace",
            "worktree_provision_state": "retained",
        }
    )

    focused = consultation.workspace_target(
        client,
        focused_repository="owner/focused",
        completed_job=completed,
    )

    assert focused == consultation.WorkspaceTarget(
        Path("/projects/focused"), "focused-workspace", "focused"
    )
    client.focused_checkout.assert_called_once_with()

    client.focused_checkout.return_value = None
    client.workspace_for.return_value = {"workspace_id": "retained-workspace"}
    retained = consultation.workspace_target(
        client,
        focused_repository=None,
        completed_job=completed,
    )
    assert retained == consultation.WorkspaceTarget(
        Path("/worktrees/example"), "retained-workspace", "retained"
    )


def test_workspace_target_rejects_missing_or_changed_workspace() -> None:
    client = mock.Mock()
    client.focused_checkout.return_value = None
    assert (
        consultation.workspace_target(
            client, focused_repository="owner/project", completed_job=None
        )
        is None
    )

    client.workspace_for.return_value = {"workspace_id": "different-workspace"}
    completed = CursorJob.from_dict(
        {
            "id": "cccccccccccc",
            "request": "finished",
            "status": "completed",
            "created_at": 1,
            "completed_at": 2,
            "delivered": True,
            "result": "done",
            "worktree_path": "/worktrees/example",
            "worktree_workspace_id": "expected-workspace",
            "worktree_provision_state": "retained",
        }
    )
    assert (
        consultation.workspace_target(
            client, focused_repository=None, completed_job=completed
        )
        is None
    )
