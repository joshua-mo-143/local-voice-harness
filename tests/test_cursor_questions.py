from __future__ import annotations

import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest import mock

import pytest

from local_voice_harness import user_config
from local_voice_harness.cursor import provisioning, questions, recovery, service
from local_voice_harness.cursor.delivery import claim_delivery
from local_voice_harness.cursor.model import CursorJob, JobStatus
from local_voice_harness.cursor.service import CursorTurnRequest, cursor_turn
from local_voice_harness.cursor.store import JobStore
from local_voice_harness.errors import HarnessError
from local_voice_harness.integrations.herdr import HerdrClient, HerdrError
from local_voice_harness.questions import (
    AnswerOutcome,
    AnswerProvenance,
    Choice,
    PromptOperationState,
    Question,
    QuestionError,
    QuestionKind,
    QuestionOrigin,
    QuestionSensitivity,
    QuestionState,
    parse_question_spec,
    question_prompt,
    resolve_answer,
)


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> JobStore:
    jobs = tmp_path / "jobs"
    legacy = tmp_path / "legacy"
    monkeypatch.setattr(service, "JOBS_DIR", jobs)
    monkeypatch.setattr(service, "LEGACY_JOBS_DIR", legacy)
    return JobStore(jobs, legacy)


def _question(
    *,
    question_id: str = "question-1",
    turn_token: str = "aaaaaaaaaaaa-1",
    kind: QuestionKind = QuestionKind.FREE_TEXT,
    choices: tuple[Choice, ...] = (),
    sensitivity: QuestionSensitivity = QuestionSensitivity.ROUTINE,
) -> Question:
    return Question(
        id=question_id,
        text="Which approach should I use?",
        kind=kind,
        choices=choices,
        sensitivity=sensitivity,
        origin=QuestionOrigin("cursor", "aaaaaaaaaaaa", turn_token),
        owner="agent",
        asked_at=10,
    )


def _awaiting(store: JobStore, question: Question | None = None) -> CursorJob:
    pending = question or _question()
    return store.create(
        CursorJob.from_dict(
            {
                "id": "aaaaaaaaaaaa",
                "request": "build the feature",
                "status": JobStatus.AWAITING_USER.value,
                "created_at": 1,
                "updated_at": 10,
                "delivered": True,
                "question": pending.text,
                "result": pending.text,
                "clarification_kind": "agent",
                "turn": 1,
                "turn_token": pending.origin.turn_token,
                "herdr_target": "retained-agent",
                "voice_question": pending.to_dict(),
            }
        )
    )


def _plan_approval_awaiting(store: JobStore) -> CursorJob:
    pending = replace(
        _question(sensitivity=QuestionSensitivity.ARCHITECTURE),
        text="Approve the reviewed plan?",
        owner="workflow_plan_approval",
    )
    created = store.create(
        CursorJob.from_dict(
            {
                "id": "aaaaaaaaaaaa",
                "request": "build the reviewed feature",
                "status": JobStatus.AWAITING_USER.value,
                "created_at": 1,
                "updated_at": 10,
                "delivered": True,
                "question": pending.text,
                "result": pending.text,
                "clarification_kind": "workflow_plan_approval",
                "turn": 3,
                "turn_token": pending.origin.turn_token,
                "workflow_tier": "medium",
                "workflow_classification_reason": "reviewed change",
                "workflow_phase": "reviewing",
                "review_round": 0,
                "herdr_target": "planner",
                "planner_target": "planner",
                "active_participant": "planner",
                "voice_question": pending.to_dict(),
            }
        )
    )
    plan = "Implement the reviewed feature."
    plan_reference = store.write_artifact(created.id, "plan", 0, plan)
    review_reference = store.write_artifact(
        created.id,
        "review",
        0,
        "The plan is safe.",
        source_text=plan,
    )
    updated = store.update(
        created.id,
        lambda job: job.evolve(
            plan_artifact=plan_reference,
            review_artifact=review_reference,
            review_decision="approve",
            review_approved=True,
            review_approval_source="reviewer",
            plan_approval_state="awaiting",
            plan_approval_id="gate-id",
            plan_approval_agent_session="planner-session",
            plan_approval_state_change_sequence=7,
        ),
    )
    assert updated is not None
    return updated


def test_structured_and_legacy_question_payloads() -> None:
    structured = parse_question_spec(
        '{"version":1,"text":"Pick one","kind":"multiple_choice",'
        '"choices":[{"id":"safe","label":"Safe mode"},'
        '{"id":"fast","label":"Fast mode"}],"sensitivity":"architecture"}'
    )

    assert structured.kind == QuestionKind.MULTIPLE_CHOICE
    assert structured.sensitivity == QuestionSensitivity.ARCHITECTURE
    assert structured.choices[0] == Choice("safe", "Safe mode")
    assert parse_question_spec("What name should I use?").kind == QuestionKind.FREE_TEXT


@pytest.mark.parametrize(
    "payload",
    [
        '{"version":1,"text":"Pick","kind":"free_text","choices":[]}',
        '{"version":1,"text":"Pick","kind":"multiple_choice",'
        '"choices":[{"id":"x","label":"One"},{"id":"x","label":"Two"}],'
        '"sensitivity":"routine"}',
        '{"version":1,"text":"Pick","kind":"multiple_choice",'
        '"choices":["","Two"],"sensitivity":"routine"}',
    ],
)
def test_invalid_structured_question_is_rejected_during_parsing(payload: str) -> None:
    with pytest.raises(QuestionError):
        parse_question_spec(payload)


def test_multiple_choice_never_guesses_ambiguous_answer() -> None:
    question = _question(
        kind=QuestionKind.MULTIPLE_CHOICE,
        choices=(Choice("a", "Shared"), Choice("b", "Shared")),
    )

    assert resolve_answer(question, "shared").outcome == AnswerOutcome.AMBIGUOUS
    selected = resolve_answer(question, "second")
    assert selected.outcome == AnswerOutcome.ACCEPTED
    assert selected.answer == "b"


@pytest.mark.parametrize(
    "spoken_answer",
    [
        "option 2",
        "option two",
        "the second option",
        "I choose choice number 2 please",
        "Point option two",
    ],
)
def test_multiple_choice_accepts_explicit_spoken_option(
    spoken_answer: str,
) -> None:
    question = _question(
        kind=QuestionKind.MULTIPLE_CHOICE,
        choices=(Choice("a", "Use SQLite"), Choice("b", "Use PostgreSQL")),
    )

    selected = resolve_answer(question, spoken_answer)

    assert selected.outcome == AnswerOutcome.ACCEPTED
    assert selected.answer == "b"


def test_multiple_choice_rejects_conflicting_spoken_options() -> None:
    question = _question(
        kind=QuestionKind.MULTIPLE_CHOICE,
        choices=(Choice("a", "Use SQLite"), Choice("b", "Use PostgreSQL")),
    )

    resolution = resolve_answer(question, "option one or option two")

    assert resolution.outcome == AnswerOutcome.AMBIGUOUS


def test_multiple_choice_accepts_equivalent_acknowledgment_spelling() -> None:
    question = _question(
        kind=QuestionKind.MULTIPLE_CHOICE,
        choices=(
            Choice("send", "Successful server send completion"),
            Choice("ack", "Explicit client acknowledgment"),
        ),
    )

    selected = resolve_answer(question, "Explicit client acknowledgement")

    assert selected.outcome == AnswerOutcome.ACCEPTED
    assert selected.answer == "ack"


def test_multiple_choice_prompt_numbers_each_spoken_option() -> None:
    question = _question(
        kind=QuestionKind.MULTIPLE_CHOICE,
        choices=(Choice("a", "Use SQLite"), Choice("b", "Use PostgreSQL")),
    )

    assert question_prompt(question) == (
        "Which approach should I use? Option 1 is Use SQLite. "
        "Option 2 is Use PostgreSQL. Please choose one."
    )


@pytest.mark.parametrize(
    "sensitivity",
    [
        QuestionSensitivity.SECURITY,
        QuestionSensitivity.DESTRUCTIVE,
        QuestionSensitivity.ARCHITECTURE,
        QuestionSensitivity.PRODUCT,
    ],
)
def test_protected_question_requires_explicit_user_text(
    sensitivity: QuestionSensitivity,
) -> None:
    question = _question(sensitivity=sensitivity)
    assert resolve_answer(question, "Use the safer design").outcome == (
        AnswerOutcome.REJECTED
    )
    assert (
        resolve_answer(
            question,
            "",
            provenance=AnswerProvenance.USER_VOICE,
        ).outcome
        == AnswerOutcome.AMBIGUOUS
    )
    assert (
        resolve_answer(
            question,
            "Use the safer design",
            provenance=AnswerProvenance.USER_VOICE,
        ).outcome
        == AnswerOutcome.ACCEPTED
    )


def test_pending_question_survives_store_restart(
    store: JobStore, tmp_path: Path
) -> None:
    created = _awaiting(store)

    restarted = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    reloaded = questions.current(restarted.get(created.id))

    assert reloaded is not None
    assert reloaded.id == "question-1"
    assert reloaded.origin.turn_token == "aaaaaaaaaaaa-1"


def test_recovery_leaves_unanswered_question_pending(store: JobStore) -> None:
    original = _awaiting(store)
    launch = mock.Mock()

    recovery.recover_jobs(store, launch_worker=launch, now=100)

    launch.assert_not_called()
    recovered = store.get(original.id)
    assert recovered.status == JobStatus.AWAITING_USER
    assert questions.current(recovered) == questions.current(original)


def test_question_delivery_has_one_atomic_claim_winner(store: JobStore) -> None:
    pending = _awaiting(store)
    store.update(pending.id, lambda job: job.evolve(delivered=False))

    first = claim_delivery(store, pending.id, foreground=True)
    second = claim_delivery(store, pending.id, foreground=True)

    assert first is not None
    assert second is None


def test_answer_later_persists_without_launching(store: JobStore) -> None:
    _awaiting(store)

    with mock.patch.object(service, "launch_worker") as launch:
        result = cursor_turn(
            CursorTurnRequest(
                "answer later",
                action="reply",
                job_id="aaaaaaaaaaaa",
                expected_question_id="question-1",
                expected_question_turn="aaaaaaaaaaaa-1",
            )
        )

    launch.assert_not_called()
    assert result.session_id is None
    deferred = questions.current(store.get("aaaaaaaaaaaa"))
    assert deferred is not None
    assert deferred.state == QuestionState.DEFERRED


def test_repeat_returns_same_question_without_mutation(store: JobStore) -> None:
    original = _awaiting(store)

    with mock.patch.object(service, "launch_worker") as launch:
        result = cursor_turn(
            CursorTurnRequest(
                "repeat",
                action="reply",
                job_id=original.id,
                expected_question_id="question-1",
            )
        )

    launch.assert_not_called()
    assert result.text == "Which approach should I use?"
    assert result.session_id == original.id
    assert store.get(original.id).revision == original.revision


@pytest.mark.parametrize(
    "answer",
    ["yes", "sure", "go ahead", "lgtm", "ok then"],
)
def test_natural_plan_approval_queues_fenced_planner_prompt(
    store: JobStore,
    answer: str,
) -> None:
    original = _plan_approval_awaiting(store)

    with mock.patch.object(service, "launch_worker") as launch:
        service.reply_job(original.id, answer)

    updated = store.get(original.id)
    assert updated.status == JobStatus.QUEUED
    assert updated.plan_approval_state == "approved"
    assert updated.plan_approval_source == "explicit"
    assert updated.herdr_target == "planner"
    launch.assert_called_once_with(original.id)


def test_ambiguous_plan_approval_reprompts_without_mutation(store: JobStore) -> None:
    original = _plan_approval_awaiting(store)

    message = service.reply_job(original.id, "maybe after lunch")

    assert message is not None
    assert "answer yes" in message
    assert store.get(original.id).revision == original.revision


def test_automated_answer_cannot_approve_plan(store: JobStore) -> None:
    original = _plan_approval_awaiting(store)

    with mock.patch.object(service, "launch_worker") as launch:
        message = service.reply_job(
            original.id,
            "yes",
            answer_provenance=AnswerProvenance.AUTOMATION,
        )

    assert message is not None
    assert "direct user answer" in message
    assert store.get(original.id).revision == original.revision
    launch.assert_not_called()


@pytest.mark.parametrize("answer", ["no", "nah", "don't", "cancel"])
def test_natural_plan_rejection_cancels_and_retains_artifact_reference(
    store: JobStore,
    answer: str,
) -> None:
    original = _plan_approval_awaiting(store)

    with mock.patch.object(service, "_cancel_target_and_release") as release:
        service.reply_job(original.id, answer)

    rejected = store.get(original.id)
    assert rejected.status == JobStatus.RECONCILING
    assert rejected.terminal_intent_status == JobStatus.CANCELLED
    assert rejected.plan_approval_state == "rejected"
    assert rejected.plan_artifact == original.plan_artifact
    release.assert_called_once()


def test_third_accepted_approval_offers_auto_after_implementation(
    store: JobStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preferences_path = tmp_path / "approval.json"
    monkeypatch.setenv(
        "VOICE_HARNESS_PLAN_APPROVAL_FILE",
        str(preferences_path),
    )
    for approval_id in ("first", "second", "gate-id"):
        user_config.record_explicit_plan_approval(
            approval_id,
            path=preferences_path,
        )
    created = store.create(
        CursorJob.from_dict(
            {
                "id": "aaaaaaaaaaaa",
                "request": "build reviewed feature",
                "status": "running",
                "created_at": 1,
                "delivered": False,
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_boot_id": "boot",
                "worker_process_start": "start",
                "workflow_tier": "medium",
                "workflow_classification_reason": "reviewed",
                "workflow_phase": "reviewing",
                "review_round": 0,
                "turn": 4,
                "turn_token": "aaaaaaaaaaaa-4",
                "workflow_turn_phase": "reviewing",
                "herdr_target": "planner",
                "planner_target": "planner",
                "active_participant": "planner",
                "plan_approval_state": "boundary",
                "plan_approval_id": "gate-id",
                "plan_approval_agent_session": "planner-session",
                "plan_approval_state_change_sequence": 7,
            }
        )
    )
    plan = "Implement the reviewed feature."
    plan_reference = store.write_artifact(created.id, "plan", 0, plan)
    review_reference = store.write_artifact(
        created.id,
        "review",
        0,
        "The plan is safe.",
        source_text=plan,
    )
    job = store.update(
        created.id,
        lambda current: current.evolve(
            workflow_phase="implementing",
            workflow_turn_phase="implementing",
            plan_artifact=plan_reference,
            review_artifact=review_reference,
            review_decision="approve",
            review_approved=True,
            review_approval_source="reviewer",
            plan_approval_state="observed",
            plan_approval_source="explicit",
            plan_approval_counted=True,
        ),
    )
    assert job is not None

    provisioning._worker_complete(
        store,
        job.id,
        "worker",
        output="VOICE_SUMMARY[aaaaaaaaaaaa-4]: done",
        agent_status="idle",
    )

    offered = store.get(job.id)
    assert offered.status == JobStatus.AWAITING_USER
    assert offered.workflow_phase.value == "finished"
    pending = questions.current(offered)
    assert pending is not None
    assert pending.owner == "workflow_plan_auto_offer"

    with mock.patch.object(service, "_cancel_target_and_release") as release:
        message = service.reply_job(job.id, "yes")

    assert message is not None
    assert "enabled" in message
    preferences = user_config.load_plan_approval_preferences(preferences_path)
    assert preferences.mode == user_config.PlanApprovalMode.AUTO
    completed = store.get(job.id)
    assert completed.terminal_intent_status == JobStatus.COMPLETED
    assert completed.terminal_intent_result == "done"
    release.assert_called_once()


def test_repeat_returns_numbered_multiple_choice_question(store: JobStore) -> None:
    original = _awaiting(
        store,
        _question(
            kind=QuestionKind.MULTIPLE_CHOICE,
            choices=(Choice("a", "Use SQLite"), Choice("b", "Use PostgreSQL")),
        ),
    )

    with mock.patch.object(service, "launch_worker") as launch:
        result = cursor_turn(
            CursorTurnRequest(
                "repeat",
                action="reply",
                job_id=original.id,
                expected_question_id="question-1",
            )
        )

    launch.assert_not_called()
    assert result.text == (
        "Which approach should I use? Option 1 is Use SQLite. "
        "Option 2 is Use PostgreSQL. Please choose one."
    )
    assert store.get(original.id).revision == original.revision


def test_foreground_multiple_choice_delivery_numbers_options(store: JobStore) -> None:
    original = _awaiting(
        store,
        _question(
            kind=QuestionKind.MULTIPLE_CHOICE,
            choices=(Choice("a", "Use SQLite"), Choice("b", "Use PostgreSQL")),
        ),
    )

    result = service._await_foreground(original.id, None)

    assert result.text == (
        "Which approach should I use? Option 1 is Use SQLite. "
        "Option 2 is Use PostgreSQL. Please choose one."
    )
    assert result.session_id == original.id


def test_stale_answer_does_not_mutate_or_launch(store: JobStore) -> None:
    original = _awaiting(store)

    with mock.patch.object(service, "launch_worker") as launch:
        result = cursor_turn(
            CursorTurnRequest(
                "safe",
                action="reply",
                job_id=original.id,
                expected_question_id="old-question",
            )
        )

    launch.assert_not_called()
    assert "older question" in result.text
    assert store.get(original.id).revision == original.revision


def test_duplicate_answer_cannot_launch_twice(store: JobStore) -> None:
    _awaiting(store)

    with mock.patch.object(service, "launch_worker") as launch:
        service.reply_job("aaaaaaaaaaaa", "Use SQLite")
        with pytest.raises(HarnessError, match="not waiting"):
            service.reply_job("aaaaaaaaaaaa", "Use Postgres")

    launch.assert_called_once_with("aaaaaaaaaaaa")
    queued = store.get("aaaaaaaaaaaa")
    assert queued.request == "build the feature"
    continuation = str(queued.to_dict()["continuation_answer"])
    assert "Which approach should I use?" in continuation
    assert "Use SQLite" in continuation
    assert provisioning._prompt_request(queued) == continuation
    planned = queued.evolve(continuation=False)
    assert provisioning._prompt_request(planned) == continuation


def test_multiple_choice_continuation_includes_label_and_question(
    store: JobStore,
) -> None:
    _awaiting(
        store,
        _question(
            kind=QuestionKind.MULTIPLE_CHOICE,
            choices=(Choice("a", "Use SQLite"), Choice("b", "Use PostgreSQL")),
        ),
    )

    with mock.patch.object(service, "launch_worker"):
        service.reply_job("aaaaaaaaaaaa", "b")

    continuation = str(store.get("aaaaaaaaaaaa").continuation_answer)
    assert "Which approach should I use?" in continuation
    assert "Use PostgreSQL" in continuation
    assert "choice id: b" in continuation


def test_protected_answer_requires_explicit_request_provenance(
    store: JobStore,
) -> None:
    _awaiting(
        store,
        _question(sensitivity=QuestionSensitivity.SECURITY),
    )
    with mock.patch.object(service, "launch_worker") as launch:
        rejected = cursor_turn(
            CursorTurnRequest(
                "allow broad access",
                action="reply",
                job_id="aaaaaaaaaaaa",
                answer_provenance=AnswerProvenance.AUTOMATION,
            )
        )
    launch.assert_not_called()
    assert "direct user answer" in rejected.text
    assert store.get("aaaaaaaaaaaa").status == JobStatus.AWAITING_USER

    with mock.patch.object(service, "launch_worker") as launch:
        cursor_turn(
            CursorTurnRequest(
                "allow broad access",
                action="reply",
                job_id="aaaaaaaaaaaa",
                answer_provenance=AnswerProvenance.USER_VOICE,
            )
        )
    launch.assert_called_once()


def test_unknown_question_owner_fails_closed(store: JobStore) -> None:
    question = replace(_question(), owner="unknown-owner")
    _awaiting(store, question)

    with mock.patch.object(service, "launch_worker") as launch:
        result = cursor_turn(
            CursorTurnRequest(
                "approve",
                action="reply",
                job_id="aaaaaaaaaaaa",
                answer_provenance=AnswerProvenance.USER_VOICE,
            )
        )

    launch.assert_not_called()
    assert "cannot safely route" in result.text
    assert store.get("aaaaaaaaaaaa").status == JobStatus.AWAITING_USER


def test_exact_turn_is_fenced_before_answer_dispatch(store: JobStore) -> None:
    _awaiting(store)
    with mock.patch.object(service, "launch_worker"):
        service.reply_job("aaaaaaaaaaaa", "Use SQLite")
    queued = store.get("aaaaaaaaaaaa")

    dispatching = provisioning._begin_prompt_turn(queued, 2, "aaaaaaaaaaaa-2")
    question = questions.current(dispatching)
    assert question is not None
    assert question.state == QuestionState.DISPATCHING
    assert question.prompt_state == PromptOperationState.PLANNED
    assert question.dispatch_token == "aaaaaaaaaaaa-2"
    assert dispatching.herdr_target == "retained-agent"

    stale_values = queued.to_dict()
    envelope = dict(cast(dict[str, object], stale_values["voice_question"]))
    origin = dict(cast(dict[str, object], envelope["origin"]))
    origin["turn_token"] = "aaaaaaaaaaaa-0"
    envelope["origin"] = origin
    stale_values["voice_question"] = envelope
    stale = CursorJob.from_dict(stale_values)
    with pytest.raises(HarnessError, match="originating turn"):
        provisioning._begin_prompt_turn(stale, 2, "aaaaaaaaaaaa-2")


def _planned_dispatch(store: JobStore) -> CursorJob:
    _awaiting(store)
    with mock.patch.object(service, "launch_worker"):
        service.reply_job("aaaaaaaaaaaa", "Use SQLite")
    updated = store.update(
        "aaaaaaaaaaaa",
        lambda job: provisioning._begin_prompt_turn(job, 2, "aaaaaaaaaaaa-2"),
    )
    assert updated is not None
    return updated


@pytest.mark.parametrize("crashed_status", [JobStatus.ROUTING, JobStatus.RUNNING])
def test_planned_dispatch_recovers_same_turn_and_answer(
    store: JobStore,
    crashed_status: JobStatus,
) -> None:
    planned = _planned_dispatch(store)
    crashed = store.update(
        planned.id,
        lambda job: job.evolve(
            status=JobStatus.ROUTING,
            worker_token="dead-worker",
            worker_pid=42,
            worker_boot_id="old-boot",
            worker_process_start="old-start",
        ),
    )
    assert crashed is not None
    if crashed_status == JobStatus.RUNNING:
        crashed = store.update(
            planned.id,
            lambda job: job.evolve(status=JobStatus.RUNNING),
        )
        assert crashed is not None
    launch = mock.Mock()

    recovery.recover_jobs(
        store,
        launch_worker=launch,
        is_worker_alive=lambda _job: False,
        now=100,
    )

    launch.assert_called_once_with(planned.id)
    recovered = store.get(planned.id)
    question = questions.current(recovered)
    assert recovered.status == JobStatus.QUEUED
    assert recovered.turn == 2
    assert recovered.turn_token == "aaaaaaaaaaaa-2"
    assert recovered.continuation_answer == planned.continuation_answer
    assert question is not None
    assert question.prompt_state == PromptOperationState.PLANNED
    assert question.dispatch_token == "aaaaaaaaaaaa-2"


def test_submitted_dispatch_retries_only_after_repeated_absence(
    store: JobStore,
) -> None:
    planned = _planned_dispatch(store)
    question = questions.current(planned)
    assert question is not None
    store.update(
        planned.id,
        lambda job: job.evolve(
            status=JobStatus.ROUTING,
            worker_token="dead-worker",
            worker_pid=42,
            worker_boot_id="old-boot",
            worker_process_start="old-start",
        ),
    )
    running = store.update(
        planned.id,
        lambda job: job.evolve(
            status=JobStatus.RUNNING,
            voice_question=questions.envelope(
                question,
                QuestionState.DISPATCHING,
                prompt_state=PromptOperationState.SUBMITTED,
                prompt_baseline_seq=7,
                prompt_submitted_at=10,
            ),
        ),
    )
    assert running is not None
    herdr = mock.Mock()
    herdr.ensure_server.return_value = None
    herdr.get_agent.return_value = {
        "state_change_seq": 7,
        "agent_status": "idle",
    }
    launch = mock.Mock()

    for observed_at in (100, 106):
        recovery.recover_jobs(
            store,
            launch_worker=launch,
            herdr_factory=lambda: herdr,
            is_worker_alive=lambda _job: False,
            now=observed_at,
        )

    launch.assert_called_once_with(planned.id)
    recovered = store.get(planned.id)
    recovered_question = questions.current(recovered)
    assert recovered.status == JobStatus.QUEUED
    assert recovered.turn_token == "aaaaaaaaaaaa-2"
    assert recovered_question is not None
    assert recovered_question.prompt_state == PromptOperationState.PLANNED
    assert recovered.continuation_answer == planned.continuation_answer


def test_observed_dispatch_recovers_by_reading_without_resubmission(
    store: JobStore,
) -> None:
    planned = _planned_dispatch(store)
    question = questions.current(planned)
    assert question is not None
    store.update(
        planned.id,
        lambda job: job.evolve(
            status=JobStatus.ROUTING,
            worker_token="dead-worker",
            worker_pid=42,
            worker_boot_id="old-boot",
            worker_process_start="old-start",
        ),
    )
    store.update(
        planned.id,
        lambda job: job.evolve(
            status=JobStatus.RUNNING,
            voice_question=questions.envelope(
                question,
                QuestionState.DISPATCHING,
                prompt_state=PromptOperationState.OBSERVED,
                prompt_baseline_seq=7,
                prompt_submitted_at=10,
            ),
        ),
    )
    launch = mock.Mock()

    recovery.recover_jobs(
        store,
        launch_worker=launch,
        is_worker_alive=lambda _job: False,
        now=100,
    )

    launch.assert_called_once_with(planned.id)
    recovered = store.get(planned.id)
    assert recovered.status == JobStatus.QUEUED
    assert recovered.reconcile is True
    assert recovered.turn_token == "aaaaaaaaaaaa-2"


def test_cancellation_closes_pending_question(store: JobStore) -> None:
    _awaiting(store)

    with mock.patch.object(service, "_cancel_target_and_release"):
        service.cancel_job("aaaaaaaaaaaa")

    cancelled = questions.current(store.get("aaaaaaaaaaaa"))
    assert cancelled is not None
    assert cancelled.state == QuestionState.CANCELLED


def test_successful_cleanup_resolves_dispatched_question(store: JobStore) -> None:
    pending = replace(
        _question(turn_token="aaaaaaaaaaaa-1"),
        state=QuestionState.DISPATCHING,
        dispatch_token="aaaaaaaaaaaa-2",
        prompt_state=PromptOperationState.SUBMITTED,
        prompt_baseline_seq=1,
        prompt_submitted_at=2,
    )
    job = store.create(
        CursorJob.from_dict(
            {
                "id": "aaaaaaaaaaaa",
                "request": "build the feature",
                "status": JobStatus.RUNNING.value,
                "created_at": 1,
                "delivered": False,
                "turn": 2,
                "turn_token": "aaaaaaaaaaaa-2",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_boot_id": "boot",
                "worker_process_start": "start",
                "herdr_target": "retained-agent",
                "workflow_tier": "simple",
                "workflow_classification_reason": "localized change",
                "workflow_phase": "implementing",
                "active_participant": "implementer",
                "implementer_target": "retained-agent",
                "voice_question": pending.to_dict(),
            }
        )
    )

    provisioning._worker_complete(
        store,
        job.id,
        "worker",
        output="VOICE_SUMMARY[aaaaaaaaaaaa-2]: completed",
        agent_status="idle",
    )

    resolved = questions.current(store.get(job.id))
    assert resolved is not None
    assert resolved.state == QuestionState.RESOLVED
    assert resolved.prompt_state == PromptOperationState.RESOLVED


def test_structured_agent_question_becomes_durable_envelope() -> None:
    job = CursorJob.from_dict(
        {
            "id": "aaaaaaaaaaaa",
            "request": "build it",
            "status": "running",
            "created_at": 1,
            "delivered": False,
            "turn": 1,
            "turn_token": "aaaaaaaaaaaa-1",
            "worker_token": "worker",
            "worker_pid": 42,
            "worker_boot_id": "boot",
            "worker_process_start": "start",
        }
    )
    output = (
        'VOICE_QUESTION[aaaaaaaaaaaa-1]: {"version":1,"text":"Pick a database",'
        '"kind":"multiple_choice","choices":["SQLite","Postgres"],'
        '"sensitivity":"architecture"}'
    )

    awaiting = provisioning.complete_from_output(
        job, output=output, agent_status="idle", now=time.time()
    )

    pending = questions.current(awaiting)
    assert awaiting.status == JobStatus.AWAITING_USER
    assert pending is not None
    assert pending.kind == QuestionKind.MULTIPLE_CHOICE
    assert pending.sensitivity == QuestionSensitivity.ARCHITECTURE


def test_malformed_structured_agent_question_blocks_job() -> None:
    job = CursorJob.from_dict(
        {
            "id": "aaaaaaaaaaaa",
            "request": "build it",
            "status": "running",
            "created_at": 1,
            "delivered": False,
            "turn": 1,
            "turn_token": "aaaaaaaaaaaa-1",
            "worker_token": "worker",
            "worker_pid": 42,
            "worker_boot_id": "boot",
            "worker_process_start": "start",
        }
    )

    blocked = provisioning.complete_from_output(
        job,
        output="VOICE_QUESTION[aaaaaaaaaaaa-1]: {not-json}",
        agent_status="idle",
        now=20,
    )

    assert blocked.status == JobStatus.BLOCKED
    assert "invalid voice question" in str(blocked.result)


def test_interactive_questionnaire_is_blocked_without_terminal_automation() -> None:
    client = HerdrClient("herdr")
    with (
        mock.patch.object(
            client,
            "get_agent",
            return_value={"interactive_ready": False, "agent_status": "idle"},
        ),
        mock.patch("subprocess.Popen") as popen,
        pytest.raises(HerdrError) as raised,
    ):
        client.prompt_and_wait("agent", "prompt", token="turn")

    assert raised.value.code == "interactive_questionnaire"
    popen.assert_not_called()


def test_late_interactive_questionnaire_terminates_local_wait() -> None:
    client = HerdrClient("herdr")
    process = mock.Mock()
    process.args = ["herdr"]
    process.poll.return_value = None
    process.wait.side_effect = [
        subprocess.TimeoutExpired(["herdr"], 1),
        0,
    ]
    client.get_agent = mock.Mock(
        side_effect=[
            {
                "interactive_ready": True,
                "agent_status": "idle",
                "state_change_seq": 1,
            },
            {
                "interactive_ready": True,
                "agent_status": "working",
                "state_change_seq": 2,
            },
            {
                "interactive_ready": False,
                "agent_status": "working",
                "state_change_seq": 3,
            },
        ]
    )
    with (
        mock.patch("subprocess.Popen", return_value=process),
        mock.patch("time.sleep"),
        pytest.raises(HerdrError) as raised,
    ):
        client.prompt_and_wait("agent", "prompt", token="turn")

    assert raised.value.code == "interactive_questionnaire"
    process.kill.assert_called_once()


def test_interactive_questionnaire_error_persists_blocked_status(
    store: JobStore,
) -> None:
    running = store.create(
        CursorJob.from_dict(
            {
                "id": "aaaaaaaaaaaa",
                "request": "build it",
                "status": "running",
                "created_at": 1,
                "delivered": False,
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_boot_id": "boot",
                "worker_process_start": "start",
                "herdr_target": "agent",
                "workflow_tier": "simple",
                "workflow_classification_reason": "localized",
                "workflow_phase": "implementing",
                "active_participant": "implementer",
                "implementer_target": "agent",
                "turn": 1,
                "turn_token": "aaaaaaaaaaaa-1",
                "workflow_turn_phase": "implementing",
                "prompt_operation_state": "planned",
                "prompt_operation_phase": "implementing",
                "prompt_operation_turn": 1,
                "prompt_operation_target": "agent",
                "prompt_baseline_sequence": 7,
            }
        )
    )

    provisioning._worker_error(
        store,
        running.id,
        "worker",
        HerdrError("questionnaire", code="interactive_questionnaire"),
        prompt_may_be_active=True,
    )

    blocked = store.get(running.id)
    assert blocked.status == JobStatus.BLOCKED
    assert "manual attention" in str(blocked.result)
    store.update(blocked.id, lambda job: job.evolve(delivered=True))
    launch = mock.Mock()
    recovery.recover_jobs(store, launch_worker=launch, now=100)
    launch.assert_not_called()

    client = mock.Mock()
    client.ensure_server.return_value = None
    client.get_agent.return_value = {
        "interactive_ready": True,
        "agent_status": "idle",
    }
    recovery.recover_jobs(
        store,
        launch_worker=launch,
        herdr_factory=lambda: client,
        now=106,
    )
    launch.assert_called_once_with(running.id)
    resumed = store.get(running.id)
    assert resumed.status == JobStatus.QUEUED
    assert resumed.reconcile
    assert not resumed.interactive_questionnaire_blocked
    assert resumed.prompt_operation_state == "none"
    assert resumed.prompt_baseline_sequence is None
