from __future__ import annotations

from collections.abc import Iterator
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


@pytest.fixture(autouse=True)
def _clear_recommendation() -> Iterator[None]:
    consultation.clear_recommendation()
    yield
    consultation.clear_recommendation()


def _awaiting(
    store: JobStore,
    *,
    job_id: str = "aaaaaaaaaaaa",
    delivered: bool = True,
    owner: str = "agent",
    kind: QuestionKind = QuestionKind.MULTIPLE_CHOICE,
    sensitivity: QuestionSensitivity = QuestionSensitivity.ARCHITECTURE,
    choices: tuple[Choice, ...] = (
        Choice("safe", "Use the safe design"),
        Choice("fast", "Use the fast design"),
    ),
) -> CursorJob:
    question = Question(
        id=f"question-{job_id}",
        text="Which design should I use?",
        kind=kind,
        choices=choices,
        sensitivity=sensitivity,
        origin=QuestionOrigin("cursor", job_id, f"{job_id}-turn-1"),
        owner=owner,
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
    assert "VOICE_RECOMMENDATION" in prompt

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


@pytest.mark.parametrize(
    ("owner", "sensitivity"),
    [
        ("fork_confirmation", QuestionSensitivity.ARCHITECTURE),
        ("agent", QuestionSensitivity.DESTRUCTIVE),
    ],
)
def test_protected_pending_question_does_not_request_a_recommendation_marker(
    tmp_path: Path,
    owner: str,
    sensitivity: QuestionSensitivity,
) -> None:
    store = _store(tmp_path)
    original = _awaiting(store, owner=owner, sensitivity=sensitivity)
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
        "I would not approve that fork.",
        None,
        "",
    )

    consultation.consult_pending_question(
        client, store, snapshot, "Should I approve this?"
    )

    prompt = client.prompt_and_wait.call_args.args[1]
    assert "VOICE_SUMMARY" in prompt
    assert "VOICE_RECOMMENDATION" not in prompt


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


def _consultant(
    summary: str,
    *,
    choice_id: str | None = "safe",
    extra_choice_id: str | None = None,
) -> mock.Mock:
    client = mock.Mock()
    client.workspace_for.return_value = {"workspace_id": "workspace-aaaaaaaaaaaa"}
    client.start_fresh_agent.return_value = AgentSelection(
        "consultant",
        "pane-2",
        "workspace-aaaaaaaaaaaa",
        "/worktrees/aaaaaaaaaaaa",
        "consultant",
    )

    def prompt_and_wait(
        _target: object,
        _prompt: str,
        token: str = "",
        **_kwargs: object,
    ) -> PromptOutcome:
        lines = [f"VOICE_SUMMARY[{token}]: {summary}"]
        if choice_id is not None:
            lines.append(f"VOICE_RECOMMENDATION[{token}]: {choice_id}")
        if extra_choice_id is not None:
            lines.append(f"VOICE_RECOMMENDATION[{token}]: {extra_choice_id}")
        return PromptOutcome("done", summary, None, "\n".join(lines))

    client.prompt_and_wait.side_effect = prompt_and_wait
    return client


def test_apply_phrases_are_explicit_and_unconditional() -> None:
    assert consultation.is_apply_recommendation_request("Use your recommendation")
    assert consultation.is_apply_recommendation_request("apply your recommendation.")
    assert not consultation.is_apply_recommendation_request("okay")
    assert not consultation.is_apply_recommendation_request("yes")
    assert not consultation.is_apply_recommendation_request("sounds good")
    assert not consultation.is_apply_recommendation_request("use your recommendation?")
    assert not consultation.is_apply_recommendation_request("use your recommendation？")
    assert not consultation.is_apply_recommendation_request("¿use your recommendation")
    assert not consultation.is_apply_recommendation_request(
        "don't use your recommendation"
    )
    assert not consultation.is_apply_recommendation_request(
        "use your recommendation if it's cheaper"
    )
    assert not consultation.is_apply_recommendation_request(
        "use your recommendation or option two"
    )


def test_delivered_recommendation_submits_only_the_stored_choice(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    original = _awaiting(store)
    snapshot = consultation.pending_question_snapshot(store, original.id)
    assert snapshot is not None
    summary = "I recommend the safe design."

    answer = consultation.consult_pending_question(
        _consultant(summary), store, snapshot, "Which option do you recommend?"
    )
    assert answer == summary
    assert consultation.applicable_choice_id(store, original.id) is None

    consultation.complete_recommendation_delivery(summary=summary)
    assert consultation.applicable_choice_id(store, original.id) == "safe"
    assert store.get(original.id).status == JobStatus.AWAITING_USER


def test_abstention_and_ambiguous_markers_are_not_applicable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    original = _awaiting(store)
    snapshot = consultation.pending_question_snapshot(store, original.id)
    assert snapshot is not None
    summary = "I cannot recommend one without your latency preference."

    consultation.consult_pending_question(
        _consultant(summary, choice_id="ABSTAIN"),
        store,
        snapshot,
        "What do you think?",
    )
    consultation.complete_recommendation_delivery(summary=summary)
    assert consultation.applicable_choice_id(store, original.id) is None

    consultation.consult_pending_question(
        _consultant(summary, choice_id="safe", extra_choice_id="safe"),
        store,
        snapshot,
        "What do you think?",
    )
    consultation.complete_recommendation_delivery(summary=summary)
    assert consultation.applicable_choice_id(store, original.id) is None

    consultation.consult_pending_question(
        _consultant(summary, choice_id="safe", extra_choice_id="fast"),
        store,
        snapshot,
        "What do you think?",
    )
    consultation.complete_recommendation_delivery(summary=summary)
    assert consultation.applicable_choice_id(store, original.id) is None


def test_undelivered_interrupted_and_mismatched_playback_fail_closed(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    original = _awaiting(store)
    snapshot = consultation.pending_question_snapshot(store, original.id)
    assert snapshot is not None
    summary = "I recommend the safe design."
    consultation.consult_pending_question(
        _consultant(summary), store, snapshot, "Which option do you recommend?"
    )
    assert consultation.applicable_choice_id(store, original.id) is None

    consultation.complete_recommendation_delivery(summary=summary, interrupted=True)
    assert consultation.applicable_choice_id(store, original.id) is None

    consultation.consult_pending_question(
        _consultant(summary), store, snapshot, "Which option do you recommend?"
    )
    consultation.complete_recommendation_delivery(summary="a different spoken result")
    assert consultation.applicable_choice_id(store, original.id) is None


def test_changed_identity_competing_and_cross_job_references_fail_closed(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    original = _awaiting(store)
    snapshot = consultation.pending_question_snapshot(store, original.id)
    assert snapshot is not None
    summary = "I recommend the safe design."
    consultation.consult_pending_question(
        _consultant(summary), store, snapshot, "Which option do you recommend?"
    )
    consultation.complete_recommendation_delivery(summary=summary)
    assert consultation.applicable_choice_id(store, original.id) == "safe"

    current = store.get(original.id)
    assert current.voice_question is not None
    original_question = current.voice_question
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
    assert consultation.applicable_choice_id(store, original.id) is None
    store.update(
        current.id,
        lambda job: job.evolve(voice_question=original_question),
    )
    assert consultation.applicable_choice_id(store, original.id) is None


def test_competing_recommendation_supersedes_and_cross_job_fails_closed(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    original = _awaiting(store)
    snapshot = consultation.pending_question_snapshot(store, original.id)
    assert snapshot is not None
    summary = "I recommend the safe design."
    consultation.consult_pending_question(
        _consultant(summary), store, snapshot, "Which option do you recommend?"
    )
    consultation.complete_recommendation_delivery(summary=summary)
    assert consultation.applicable_choice_id(store, original.id) == "safe"

    later = "I now recommend the fast design."
    consultation.consult_pending_question(
        _consultant(later, choice_id="fast"),
        store,
        snapshot,
        "Which option now?",
    )
    consultation.complete_recommendation_delivery(summary=later)
    assert consultation.applicable_choice_id(store, original.id) == "fast"
    assert consultation.applicable_choice_id(store, "bbbbbbbbbbbb") is None


def test_post_restart_and_protected_owners_cannot_apply(tmp_path: Path) -> None:
    store = _store(tmp_path)
    original = _awaiting(store)
    snapshot = consultation.pending_question_snapshot(store, original.id)
    assert snapshot is not None
    summary = "I recommend the safe design."
    consultation.consult_pending_question(
        _consultant(summary), store, snapshot, "Which option do you recommend?"
    )
    consultation.complete_recommendation_delivery(summary=summary)
    with mock.patch.object(consultation, "boot_identity", return_value="other-boot"):
        assert consultation.applicable_choice_id(store, original.id) is None

    consultation.clear_recommendation()
    with mock.patch.object(consultation, "boot_identity", return_value=None):
        consultation.consult_pending_question(
            _consultant(summary), store, snapshot, "Which option do you recommend?"
        )
        consultation.complete_recommendation_delivery(summary=summary)
        assert consultation.applicable_choice_id(store, original.id) is None

    protected = _store(tmp_path / "protected")
    job = _awaiting(protected, owner="fork_confirmation")
    protected_snapshot = consultation.pending_question_snapshot(protected, job.id)
    assert protected_snapshot is not None
    consultation.consult_pending_question(
        _consultant(summary),
        protected,
        protected_snapshot,
        "Which option do you recommend?",
    )
    consultation.complete_recommendation_delivery(summary=summary)
    assert consultation.applicable_choice_id(protected, job.id) is None

    approval = _store(tmp_path / "approval")
    approval_job = _awaiting(approval, owner="workflow_plan_approval")
    approval_snapshot = consultation.pending_question_snapshot(
        approval, approval_job.id
    )
    assert approval_snapshot is not None
    consultation.consult_pending_question(
        _consultant(summary),
        approval,
        approval_snapshot,
        "Which option do you recommend?",
    )
    consultation.complete_recommendation_delivery(summary=summary)
    assert consultation.applicable_choice_id(approval, approval_job.id) is None

    free_text = _store(tmp_path / "free")
    free = _awaiting(free_text, kind=QuestionKind.FREE_TEXT, choices=())
    free_snapshot = consultation.pending_question_snapshot(free_text, free.id)
    assert free_snapshot is not None
    consultation.consult_pending_question(
        _consultant(summary),
        free_text,
        free_snapshot,
        "What do you think?",
    )
    consultation.complete_recommendation_delivery(summary=summary)
    assert consultation.applicable_choice_id(free_text, free.id) is None

    destructive = _store(tmp_path / "destructive")
    destructive_job = _awaiting(
        destructive,
        sensitivity=QuestionSensitivity.DESTRUCTIVE,
    )
    destructive_snapshot = consultation.pending_question_snapshot(
        destructive, destructive_job.id
    )
    assert destructive_snapshot is not None
    consultation.consult_pending_question(
        _consultant(summary),
        destructive,
        destructive_snapshot,
        "Which option do you recommend?",
    )
    consultation.complete_recommendation_delivery(summary=summary)
    assert consultation.applicable_choice_id(destructive, destructive_job.id) is None
