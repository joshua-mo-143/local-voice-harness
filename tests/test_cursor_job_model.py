from __future__ import annotations

import unittest

from local_voice_harness.cursor.lifecycle import CleanupReconciling
from local_voice_harness.cursor.model import (
    CURRENT_SCHEMA_VERSION,
    CursorJob,
    JobStatus,
    JobValidationError,
    WorkflowPhase,
    WorkflowTier,
    legal_transitions,
    transition,
    validate_reservations,
)
from local_voice_harness.cursor.workflow import (
    ArtifactReference,
    LegacyPlanApprovalProof,
    ReviewDecision,
)
from local_voice_harness.job_lifecycle import ReconcilingJob


class FollowUpLineageTests(unittest.TestCase):
    def _child(self, **fields: object) -> CursorJob:
        value: dict[str, object] = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "id": "123456789abc",
            "parent_job_id": "aaaaaaaaaaaa",
            "revision": 0,
            "request": "review",
            "status": JobStatus.QUEUED.value,
            "created_at": 1,
            "queued_at": 1,
            "delivered": False,
            "repository": "/repo",
            "worktree_branch": "voice/feature",
            "worktree_path": "/repo-wt",
            "worktree_workspace_id": "workspace",
            "worktree_root_pane_id": "root-pane",
            "worktree_provision_state": "ready",
        }
        value.update(fields)
        return CursorJob.from_dict(value)

    def test_parent_job_id_round_trips(self) -> None:
        child = self._child()
        self.assertEqual(child.parent_job_id, "aaaaaaaaaaaa")
        self.assertEqual(child.to_dict()["parent_job_id"], "aaaaaaaaaaaa")

    def test_transition_cannot_change_parent_job_id(self) -> None:
        child = self._child()
        with self.assertRaises(JobValidationError):
            transition(child, JobStatus.QUEUED, parent_job_id="dddddddddddd")

    def test_child_cannot_substitute_inherited_checkout(self) -> None:
        child = self._child()
        with self.assertRaises(JobValidationError):
            transition(child, JobStatus.QUEUED, worktree_path="/somewhere-else")

    def test_child_cannot_remove_inherited_worktree(self) -> None:
        child = self._child()
        with self.assertRaises(JobValidationError):
            transition(child, JobStatus.QUEUED, worktree_path=None)

    def test_child_recovery_keeps_confirmed_absent_worktree_identity(self) -> None:
        child = self._child(
            worktree_provision_state="failed_observing",
            worktree_dispatch_exited=True,
        )

        updated = child.record_operation_observation(
            "worktree",
            "worktree_provision_state",
            frozenset({"failed_observing"}),
            now=10,
            observed_absent=True,
            failed_max_attempts=1,
            uncertain_max_attempts=6,
            base_seconds=5,
            max_seconds=60,
        )

        assert updated is not None
        self.assertEqual(updated.worktree_provision_state, "confirmed_absent")
        self.assertEqual(updated.worktree_path, "/repo-wt")


class CursorJobModelTests(unittest.TestCase):
    def job_for_status(self, status: JobStatus) -> CursorJob:
        value: dict[str, object] = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "id": "123456789abc",
            "revision": 0,
            "request": "do it",
            "status": status.value,
            "created_at": 1,
            "delivered": False,
        }
        if status == JobStatus.QUEUED:
            value["queued_at"] = 1
        if status in {
            JobStatus.ROUTING,
            JobStatus.RUNNING,
            JobStatus.RECONCILING,
        }:
            value.update(
                {
                    "worker_token": "claim",
                    "worker_pid": 42,
                    "worker_boot_id": "boot",
                    "worker_process_start": "start",
                    "worker_claim_operation": status.value,
                    "worker_claimed_at": 1,
                }
            )
        if status == JobStatus.AWAITING_USER:
            value.update({"question": "question", "result": "question"})
        if status == JobStatus.BLOCKED:
            value.update({"result": "blocked", "completed_at": 2})
        if status == JobStatus.COMPLETED:
            value.update({"result": "done", "completed_at": 2})
        if status == JobStatus.FAILED:
            value.update({"error": "failed", "result": "failed", "completed_at": 2})
        if status == JobStatus.CANCELLED:
            value.update({"result": "cancelled", "completed_at": 2})
        return CursorJob.from_dict(value)

    def fields_for_status(self, status: JobStatus) -> dict[str, object]:
        fields: dict[str, object] = {}
        if status == JobStatus.QUEUED:
            fields["queued_at"] = 3
        if status in {
            JobStatus.ROUTING,
            JobStatus.RUNNING,
            JobStatus.RECONCILING,
        }:
            fields.update(
                {
                    "worker_token": "next-claim",
                    "worker_pid": 43,
                    "worker_boot_id": "next-boot",
                    "worker_process_start": "next-start",
                    "worker_claim_operation": status.value,
                    "worker_claimed_at": 3,
                }
            )
        if status == JobStatus.AWAITING_USER:
            fields.update({"question": "question", "result": "question"})
        if status == JobStatus.BLOCKED:
            fields.update({"result": "blocked", "completed_at": 3})
        if status == JobStatus.COMPLETED:
            fields.update({"result": "done", "completed_at": 3})
        if status == JobStatus.FAILED:
            fields.update({"error": "failed", "result": "failed", "completed_at": 3})
        if status == JobStatus.CANCELLED:
            fields.update({"result": "cancelled", "completed_at": 3})
        return fields

    def test_unversioned_job_uses_explicit_v0_compatibility(self) -> None:
        job = CursorJob.from_dict(
            {
                "id": "123456789abc",
                "status": "completed",
                "result": "done",
                "completed_at": "2.5",
                "delivered": 0,
            }
        )

        self.assertEqual(job.loaded_schema_version, 0)
        self.assertEqual(job.status, JobStatus.COMPLETED)
        self.assertEqual(job.completed_at, 2.5)
        self.assertFalse(job.delivered)
        self.assertEqual(job.to_dict()["schema_version"], CURRENT_SCHEMA_VERSION)

    def test_current_schema_rejects_missing_status_specific_field(self) -> None:
        with self.assertRaisesRegex(
            JobValidationError, "completed job requires result"
        ):
            CursorJob.from_dict(
                {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "id": "123456789abc",
                    "revision": 0,
                    "request": "do it",
                    "status": "completed",
                    "created_at": 1,
                    "completed_at": 2,
                    "delivered": False,
                }
            )

    def test_current_schema_rejects_invalid_explicit_field_type(self) -> None:
        with self.assertRaisesRegex(
            JobValidationError, "worker_pid must be an integer"
        ):
            CursorJob.from_dict(
                {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "id": "123456789abc",
                    "revision": 0,
                    "request": "do it",
                    "status": "queued",
                    "created_at": 1,
                    "queued_at": 1,
                    "delivered": False,
                    "worker_pid": "not-a-pid",
                }
            )

    def test_v5_worker_ownership_is_migrated_with_stale_boot_fence(self) -> None:
        job = CursorJob.from_dict(
            {
                "schema_version": 5,
                "id": "123456789abc",
                "revision": 4,
                "request": "do it",
                "status": "running",
                "created_at": 1,
                "delivered": False,
                "worker_token": "claim",
                "worker_pid": 42,
                "worker_process_start": "start",
                "target_release_pending": True,
                "target_release_owner_pid": 43,
                "target_release_owner_start": "release-start",
            }
        )

        self.assertEqual(job.worker_boot_id, "legacy-unknown")
        self.assertEqual(
            job.to_dict()["target_release_owner_boot_id"], "legacy-unknown"
        )
        self.assertEqual(job.to_dict()["schema_version"], CURRENT_SCHEMA_VERSION)

    def test_operation_fence_requires_reserved_identity(self) -> None:
        with self.assertRaisesRegex(
            JobValidationError, "dispatching agent operation requires herdr_target"
        ):
            CursorJob.from_dict(
                {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "id": "123456789abc",
                    "revision": 0,
                    "request": "do it",
                    "status": "queued",
                    "created_at": 1,
                    "queued_at": 1,
                    "delivered": False,
                    "agent_dispatch_state": "dispatching",
                }
            )

    def test_manual_reconciliation_fields_form_one_fence(self) -> None:
        with self.assertRaisesRegex(
            JobValidationError, "manual reconciliation requires operation and token"
        ):
            CursorJob.from_dict(
                {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "id": "123456789abc",
                    "revision": 0,
                    "request": "do it",
                    "status": "failed",
                    "created_at": 1,
                    "completed_at": 2,
                    "result": "failed",
                    "error": "failed",
                    "delivered": False,
                    "herdr_target": "agent",
                    "agent_dispatch_state": "manual_required",
                }
            )

    def test_v6_job_migrates_to_v7_with_default_announcement_state(self) -> None:
        job = CursorJob.from_dict(
            {
                "schema_version": 6,
                "id": "123456789abc",
                "revision": 3,
                "request": "do it",
                "status": "completed",
                "created_at": 1,
                "completed_at": 2,
                "result": "done",
                "delivered": True,
            }
        )

        self.assertEqual(job.loaded_schema_version, 6)
        self.assertEqual(job.to_dict()["schema_version"], CURRENT_SCHEMA_VERSION)
        self.assertFalse(job.announcement_dismissed)
        self.assertFalse(job.announcement_repeated)
        self.assertIsNone(job.speakable_label)

    def test_v15_delivered_job_migrates_to_spoken_or_dismissed_ack(self) -> None:
        spoken = CursorJob.from_dict(
            {
                "schema_version": 15,
                "id": "123456789abc",
                "revision": 0,
                "request": "do it",
                "status": "completed",
                "created_at": 1,
                "completed_at": 2,
                "result": "done",
                "delivered": True,
            }
        )
        dismissed = CursorJob.from_dict(
            {
                "schema_version": 15,
                "id": "bbbbbbbbbbbb",
                "revision": 0,
                "request": "do it",
                "status": "completed",
                "created_at": 1,
                "completed_at": 2,
                "result": "done",
                "delivered": True,
                "announcement_dismissed": True,
            }
        )

        self.assertEqual(spoken.announcement_ack, "spoken")
        self.assertEqual(dismissed.announcement_ack, "dismissed")
        self.assertEqual(spoken.to_dict()["schema_version"], CURRENT_SCHEMA_VERSION)

    def test_v7_announcement_fields_round_trip(self) -> None:
        job = CursorJob.from_dict(
            {
                "schema_version": CURRENT_SCHEMA_VERSION,
                "id": "123456789abc",
                "revision": 0,
                "request": "do it",
                "status": "completed",
                "created_at": 1,
                "completed_at": 2,
                "result": "done",
                "delivered": True,
                "speakable_label": "issue 42",
                "announcement_dismissed": True,
                "announcement_repeated": True,
            }
        )

        self.assertEqual(job.speakable_label, "issue 42")
        self.assertTrue(job.announcement_dismissed)
        self.assertEqual(job.announcement_ack, "dismissed")
        self.assertTrue(job.announcement_repeated)
        self.assertNotIn("announcement_dismissed", job.to_dict())
        reloaded = CursorJob.from_dict(job.to_dict())
        self.assertEqual(reloaded.speakable_label, "issue 42")
        self.assertTrue(reloaded.announcement_dismissed)
        self.assertEqual(reloaded.announcement_ack, "dismissed")
        self.assertNotIn("announcement_dismissed", reloaded.to_dict())

    def test_legacy_boolean_only_dismissal_imports_to_acknowledgement(self) -> None:
        job = CursorJob.from_dict(
            {
                "schema_version": 14,
                "id": "123456789abc",
                "revision": 0,
                "request": "do it",
                "status": "completed",
                "created_at": 1,
                "completed_at": 2,
                "result": "done",
                "delivered": True,
                "announcement_dismissed": True,
            }
        )

        self.assertEqual(job.announcement_ack, "dismissed")
        self.assertTrue(job.announcement_dismissed)
        self.assertNotIn("announcement_dismissed", job.to_dict())

    def test_legacy_pending_ack_and_dismissal_import_as_dismissed(self) -> None:
        job = CursorJob.from_dict(
            {
                "schema_version": 15,
                "id": "123456789abc",
                "revision": 0,
                "request": "do it",
                "status": "completed",
                "created_at": 1,
                "completed_at": 2,
                "result": "done",
                "delivered": True,
                "announcement_ack": "pending",
                "announcement_dismissed": True,
            }
        )

        self.assertEqual(job.announcement_ack, "dismissed")
        self.assertTrue(job.announcement_dismissed)

    def test_native_dismissal_cannot_disagree_with_acknowledgement(self) -> None:
        with self.assertRaisesRegex(JobValidationError, "must match"):
            CursorJob.from_dict(
                {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "id": "123456789abc",
                    "revision": 0,
                    "request": "do it",
                    "status": "completed",
                    "created_at": 1,
                    "completed_at": 2,
                    "result": "done",
                    "delivered": True,
                    "announcement_ack": "spoken",
                    "announcement_dismissed": True,
                }
            )
        with self.assertRaisesRegex(JobValidationError, "must match"):
            CursorJob.from_dict(
                {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "id": "bbbbbbbbbbbb",
                    "revision": 0,
                    "request": "do it",
                    "status": "completed",
                    "created_at": 1,
                    "completed_at": 2,
                    "result": "done",
                    "delivered": True,
                    "announcement_ack": "dismissed",
                    "announcement_dismissed": False,
                }
            )
        with self.assertRaisesRegex(JobValidationError, "must match"):
            CursorJob.from_dict(
                {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "id": "cccccccccccc",
                    "revision": 0,
                    "request": "do it",
                    "status": "completed",
                    "created_at": 1,
                    "completed_at": 2,
                    "result": "done",
                    "delivered": True,
                    "announcement_ack": "pending",
                    "announcement_dismissed": True,
                }
            )

    def test_native_dismiss_and_repeat_do_not_emit_dismissal_mirror(self) -> None:
        job = self.job_for_status(JobStatus.COMPLETED)
        dismissed = job.dismiss_announcement(delivered_at=3)
        repeated = dismissed.repeat_announcement(now=4)

        self.assertEqual(dismissed.announcement_ack, "dismissed")
        self.assertTrue(dismissed.announcement_dismissed)
        self.assertNotIn("announcement_dismissed", dismissed.to_dict())
        self.assertEqual(repeated.announcement_ack, "pending")
        self.assertFalse(repeated.announcement_dismissed)
        self.assertNotIn("announcement_dismissed", repeated.to_dict())
        reloaded = CursorJob.from_dict(dismissed.to_dict())
        self.assertTrue(reloaded.announcement_dismissed)
        self.assertEqual(
            reloaded.announcement_ack,
            reloaded.delivery_state.announcement.acknowledgement.value,
        )

    def test_v8_job_migrates_to_finished_simple_workflow(self) -> None:
        job = CursorJob.from_dict(
            {
                "schema_version": 8,
                "id": "123456789abc",
                "revision": 3,
                "request": "do it",
                "status": "completed",
                "created_at": 1,
                "completed_at": 2,
                "result": "done",
                "delivered": True,
            }
        )

        self.assertEqual(job.workflow_tier, WorkflowTier.SIMPLE)
        self.assertEqual(job.workflow_phase, WorkflowPhase.FINISHED)
        self.assertIn("Migrated", job.workflow_classification_reason or "")

    def test_v9_approved_review_migrates_reviewer_approval_source(self) -> None:
        job = CursorJob.from_dict(
            {
                "schema_version": 9,
                "id": "123456789abc",
                "revision": 3,
                "request": "change recovery",
                "status": "running",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_boot_id": "boot",
                "worker_process_start": "start",
                "created_at": 1,
                "delivered": False,
                "workflow_tier": "high-risk",
                "workflow_classification_reason": "recovery",
                "workflow_phase": "implementing",
                "review_round": 1,
                "plan_artifact": ".artifacts/123456789abc/plan-1.json",
                "review_artifact": ".artifacts/123456789abc/review-1.json",
                "review_decision": "approve",
                "review_approved": True,
            }
        )

        self.assertEqual(job.review_approval_source, "reviewer")
        self.assertEqual(job.plan_approval_state, "observed")
        self.assertEqual(job.plan_approval_source, "legacy")
        self.assertIsInstance(job.plan_approval.proof, LegacyPlanApprovalProof)
        self.assertEqual(job.plan_approval_state_change_sequence, -1)
        self.assertEqual(job.plan_approval_revision, -1)
        self.assertEqual(job.plan_approval.plan_reference, job.plan_artifact)
        self.assertEqual(job.plan_approval.review_reference, job.review_artifact)
        self.assertEqual(
            CursorJob.from_dict(job.to_dict()).plan_approval,
            job.plan_approval,
        )
        self.assertTrue(job.review_approved)
        self.assertNotIn("review_approved", job.to_dict())

    def test_legacy_review_boolean_imports_to_approval_source(self) -> None:
        job = CursorJob.from_dict(
            {
                "schema_version": 9,
                "id": "123456789abc",
                "revision": 0,
                "request": "change recovery",
                "status": "queued",
                "created_at": 1,
                "queued_at": 1,
                "delivered": False,
                "workflow_tier": "medium",
                "workflow_classification_reason": "recovery",
                "workflow_phase": "reviewing",
                "review_round": 0,
                "plan_artifact": ".artifacts/123456789abc/plan-0.json",
                "review_artifact": ".artifacts/123456789abc/review-0.json",
                "review_decision": "approve",
                "review_approved": True,
            }
        )

        self.assertEqual(job.review_approval_source, "reviewer")
        self.assertTrue(job.review_approved)
        self.assertNotIn("review_approved", job.to_dict())

    def test_native_review_boolean_cannot_create_reviewer_provenance(self) -> None:
        with self.assertRaisesRegex(JobValidationError, "explicit approval source"):
            CursorJob.from_dict(
                {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "id": "123456789abc",
                    "revision": 0,
                    "request": "change recovery",
                    "status": "queued",
                    "created_at": 1,
                    "queued_at": 1,
                    "delivered": False,
                    "workflow_tier": "medium",
                    "workflow_classification_reason": "recovery",
                    "workflow_phase": "reviewing",
                    "review_round": 0,
                    "plan_artifact": ".artifacts/123456789abc/plan-0.json",
                    "review_artifact": ".artifacts/123456789abc/review-0.json",
                    "review_decision": "approve",
                    "review_approved": True,
                }
            )

    def test_native_review_approval_cannot_disagree_with_source(self) -> None:
        with self.assertRaisesRegex(JobValidationError, "must be paired"):
            CursorJob.from_dict(
                {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "id": "123456789abc",
                    "revision": 0,
                    "request": "change recovery",
                    "status": "queued",
                    "created_at": 1,
                    "queued_at": 1,
                    "delivered": False,
                    "workflow_tier": "medium",
                    "workflow_classification_reason": "recovery",
                    "workflow_phase": "reviewing",
                    "review_round": 0,
                    "plan_artifact": ".artifacts/123456789abc/plan-0.json",
                    "review_artifact": ".artifacts/123456789abc/review-0.json",
                    "review_decision": "approve",
                    "review_approved": True,
                    "review_approval_source": None,
                }
            )
        with self.assertRaisesRegex(JobValidationError, "must be paired"):
            CursorJob.from_dict(
                {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "id": "bbbbbbbbbbbb",
                    "revision": 0,
                    "request": "change recovery",
                    "status": "queued",
                    "created_at": 1,
                    "queued_at": 1,
                    "delivered": False,
                    "workflow_tier": "medium",
                    "workflow_classification_reason": "recovery",
                    "workflow_phase": "reviewing",
                    "review_round": 0,
                    "plan_artifact": ".artifacts/123456789abc/plan-0.json",
                    "review_artifact": ".artifacts/123456789abc/review-0.json",
                    "review_decision": "approve",
                    "review_approved": False,
                    "review_approval_source": "reviewer",
                }
            )

    def test_reviewer_approval_still_requires_decision_and_artifact(self) -> None:
        with self.assertRaisesRegex(JobValidationError, "approving review decision"):
            CursorJob.from_dict(
                {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "id": "123456789abc",
                    "revision": 0,
                    "request": "change recovery",
                    "status": "queued",
                    "created_at": 1,
                    "queued_at": 1,
                    "delivered": False,
                    "workflow_tier": "medium",
                    "workflow_classification_reason": "recovery",
                    "workflow_phase": "reviewing",
                    "review_round": 0,
                    "plan_artifact": ".artifacts/123456789abc/plan-0.json",
                    "review_artifact": ".artifacts/123456789abc/review-0.json",
                    "review_decision": "revise",
                    "review_approval_source": "reviewer",
                }
            )
        with self.assertRaisesRegex(JobValidationError, "review artifact"):
            CursorJob.from_dict(
                {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "id": "bbbbbbbbbbbb",
                    "revision": 0,
                    "request": "change recovery",
                    "status": "queued",
                    "created_at": 1,
                    "queued_at": 1,
                    "delivered": False,
                    "workflow_tier": "medium",
                    "workflow_classification_reason": "recovery",
                    "workflow_phase": "reviewing",
                    "review_round": 0,
                    "plan_artifact": ".artifacts/bbbbbbbbbbbb/plan-0.json",
                    "review_decision": "approve",
                    "review_approval_source": "reviewer",
                }
            )

    def test_user_override_still_requires_exhausted_high_risk_proof(self) -> None:
        with self.assertRaisesRegex(JobValidationError, "exhausted high-risk"):
            CursorJob.from_dict(
                {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "id": "123456789abc",
                    "revision": 0,
                    "request": "change recovery",
                    "status": "queued",
                    "created_at": 1,
                    "queued_at": 1,
                    "delivered": False,
                    "workflow_tier": "high-risk",
                    "workflow_classification_reason": "recovery",
                    "workflow_phase": "reviewing",
                    "review_round": 1,
                    "plan_artifact": ".artifacts/123456789abc/plan-1.json",
                    "review_artifact": ".artifacts/123456789abc/review-1.json",
                    "review_decision": "revise",
                    "review_approval_source": "user",
                }
            )

    def test_native_review_transition_does_not_emit_approval_boolean(self) -> None:
        job = CursorJob.from_dict(
            {
                "schema_version": CURRENT_SCHEMA_VERSION,
                "id": "123456789abc",
                "revision": 0,
                "request": "change recovery",
                "status": "queued",
                "created_at": 1,
                "queued_at": 1,
                "delivered": False,
                "workflow_tier": "medium",
                "workflow_classification_reason": "recovery",
                "workflow_phase": "reviewing",
                "review_round": 0,
                "plan_artifact": ".artifacts/123456789abc/plan-0.json",
            }
        )
        approved = job.evolve_review(
            job.review_state.publish_review(
                ArtifactReference.parse(
                    ".artifacts/123456789abc/review-0.json",
                    job_id=job.id,
                    kind="review",
                ),
                ReviewDecision.APPROVE,
            )
        )

        self.assertTrue(approved.review_approved)
        self.assertEqual(approved.review_approval_source, "reviewer")
        self.assertNotIn("review_approved", approved.to_dict())
        reloaded = CursorJob.from_dict(approved.to_dict())
        self.assertTrue(reloaded.review_approved)
        self.assertEqual(reloaded.review_state.approved, reloaded.review_approved)

    def test_v9_exhausted_review_migrates_to_explicit_clarification(self) -> None:
        job = CursorJob.from_dict(
            {
                "schema_version": 9,
                "id": "123456789abc",
                "revision": 3,
                "request": "change recovery",
                "status": "awaiting_user",
                "created_at": 1,
                "delivered": False,
                "question": "Decide how to proceed.",
                "result": "Decide how to proceed.",
                "clarification_kind": "workflow_review",
                "workflow_tier": "high-risk",
                "workflow_classification_reason": "recovery",
                "workflow_phase": "reviewing",
                "review_round": 2,
                "plan_artifact": ".artifacts/123456789abc/plan-2.json",
                "review_artifact": ".artifacts/123456789abc/review-2.json",
                "review_decision": "revise",
            }
        )

        self.assertEqual(
            job.clarification_kind,
            "workflow_review_exhausted",
        )

    def test_v9_round_two_revising_state_is_preserved(self) -> None:
        job = CursorJob.from_dict(
            {
                "schema_version": 9,
                "id": "123456789abc",
                "revision": 3,
                "request": "change recovery",
                "status": "running",
                "created_at": 1,
                "delivered": False,
                "workflow_tier": "high-risk",
                "workflow_classification_reason": "recovery",
                "workflow_phase": "revising",
                "review_round": 2,
                "plan_artifact": ".artifacts/123456789abc/plan-1.json",
                "review_artifact": ".artifacts/123456789abc/review-1.json",
                "review_decision": "revise",
            }
        )

        self.assertEqual(job.workflow_phase, WorkflowPhase.REVISING)
        self.assertEqual(job.review_round, 2)

    def test_workflow_tier_can_only_be_promoted(self) -> None:
        job = CursorJob.from_dict(
            {
                "id": "123456789abc",
                "status": "queued",
                "workflow_tier": "high-risk",
                "workflow_classification_reason": "recovery",
                "workflow_phase": "planning",
            }
        )

        with self.assertRaisesRegex(JobValidationError, "cannot be downgraded"):
            job.evolve(
                workflow_tier="medium",
                workflow_classification_reason="smaller",
            )

    def test_simple_workflow_cannot_enter_review(self) -> None:
        with self.assertRaisesRegex(
            JobValidationError, "simple workflow cannot enter planning or review"
        ):
            CursorJob.from_dict(
                {
                    "id": "123456789abc",
                    "status": "queued",
                    "workflow_tier": "simple",
                    "workflow_classification_reason": "localized",
                    "workflow_phase": "reviewing",
                }
            )

    def test_planned_workflow_cannot_implement_without_approved_review(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            JobValidationError, "cannot implement without approved"
        ):
            CursorJob.from_dict(
                {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "id": "123456789abc",
                    "revision": 0,
                    "request": "change persistence",
                    "status": "queued",
                    "created_at": 1,
                    "queued_at": 1,
                    "delivered": False,
                    "workflow_tier": "high-risk",
                    "workflow_classification_reason": "persistence",
                    "workflow_phase": "implementing",
                }
            )

    def test_current_planned_workflow_cannot_implement_without_plan_approval(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            JobValidationError,
            "without plan approval",
        ):
            CursorJob.from_dict(
                {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "id": "123456789abc",
                    "revision": 0,
                    "request": "change persistence",
                    "status": "queued",
                    "created_at": 1,
                    "queued_at": 1,
                    "delivered": False,
                    "workflow_tier": "medium",
                    "workflow_classification_reason": "persistence",
                    "workflow_phase": "implementing",
                    "plan_artifact": ".artifacts/123456789abc/plan-0.json",
                    "review_artifact": ".artifacts/123456789abc/review-0.json",
                    "review_decision": "approve",
                    "review_approved": True,
                    "review_approval_source": "reviewer",
                }
            )

    def test_plan_approval_boundary_requires_session_proof(self) -> None:
        with self.assertRaisesRegex(
            JobValidationError,
            "requires gate ID, agent session, and sequence",
        ):
            CursorJob.from_dict(
                {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "id": "123456789abc",
                    "revision": 0,
                    "request": "change persistence",
                    "status": "running",
                    "created_at": 1,
                    "delivered": False,
                    "worker_token": "worker",
                    "worker_pid": 42,
                    "worker_boot_id": "boot",
                    "worker_process_start": "start",
                    "workflow_tier": "medium",
                    "workflow_classification_reason": "persistence",
                    "workflow_phase": "planning",
                    "plan_approval_state": "boundary",
                    "plan_approval_id": "gate",
                }
            )

    def test_illegal_status_transition_is_rejected(self) -> None:
        job = CursorJob.from_dict(
            {
                "id": "123456789abc",
                "status": "completed",
                "result": "done",
                "completed_at": 2,
                "delivered": False,
            }
        )

        with self.assertRaisesRegex(
            JobValidationError, "illegal Cursor job transition completed -> running"
        ):
            transition(
                job,
                JobStatus.RUNNING,
                worker_token="claim",
                worker_pid=12,
                worker_boot_id="boot",
                worker_process_start="start",
            )

    def test_worker_owned_transition_requires_complete_claim(self) -> None:
        job = CursorJob.from_dict(
            {
                "id": "123456789abc",
                "status": "queued",
                "request": "do it",
                "delivered": False,
            }
        )

        with self.assertRaisesRegex(
            JobValidationError, "routing job requires complete worker ownership"
        ):
            transition(job, JobStatus.ROUTING, worker_token="claim")

    def test_delivery_claim_fields_must_be_paired(self) -> None:
        with self.assertRaisesRegex(
            JobValidationError, "delivery claim token and timestamp must be paired"
        ):
            CursorJob.from_dict(
                {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "id": "123456789abc",
                    "revision": 0,
                    "request": "do it",
                    "status": "queued",
                    "created_at": 1,
                    "queued_at": 1,
                    "delivered": False,
                    "delivery_claim_token": "claim",
                }
            )

    def test_duplicate_active_target_reservation_is_rejected(self) -> None:
        first = CursorJob.from_dict(
            {
                "id": "aaaaaaaaaaaa",
                "status": "running",
                "herdr_target": "agent",
                "worker_token": "one",
                "worker_pid": 1,
                "worker_boot_id": "boot",
                "worker_process_start": "start-one",
            }
        )
        second = CursorJob.from_dict(
            {
                "id": "bbbbbbbbbbbb",
                "status": "routing",
                "herdr_target": "agent",
                "worker_token": "two",
                "worker_pid": 2,
                "worker_boot_id": "boot",
                "worker_process_start": "start-two",
            }
        )

        with self.assertRaisesRegex(
            JobValidationError, "Herdr target agent is reserved by both"
        ):
            validate_reservations([first, second])

    def test_every_legal_and_illegal_status_edge_is_explicit(self) -> None:
        for source in JobStatus:
            for target in JobStatus:
                with self.subTest(source=source, target=target):
                    job = self.job_for_status(source)
                    fields = self.fields_for_status(target)
                    immutable_same_status = (
                        source
                        in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
                        and target == source
                    )
                    if immutable_same_status:
                        with self.assertRaisesRegex(
                            JobValidationError,
                            "materialized terminal outcome is immutable",
                        ):
                            transition(job, target, **fields)
                    elif target == source or target in legal_transitions(source):
                        updated = transition(job, target, **fields)
                        self.assertEqual(updated.status, target)
                        self.assertEqual(updated.revision, 1)
                    else:
                        with self.assertRaisesRegex(
                            JobValidationError,
                            f"illegal Cursor job transition {source} -> {target}",
                        ):
                            transition(job, target, **fields)

    def test_generic_evolve_cannot_infer_recovery_only_edge(self) -> None:
        queued = self.job_for_status(JobStatus.QUEUED).evolve(
            github_repository="owner/repository",
            github_issue_create_title="Recovered issue",
            github_issue_create_marker="marker",
            github_issue_create_operation_state="submitted",
        )

        with self.assertRaisesRegex(
            JobValidationError, "illegal Cursor job transition queued -> completed"
        ):
            queued.evolve(
                status=JobStatus.COMPLETED,
                result="Created issue",
                completed_at=2,
                github_issue_create_operation_state="created",
            )

    def test_current_reconciling_adapter_accepts_cleanup_reconciliation(self) -> None:
        job = self.job_for_status(JobStatus.RECONCILING).evolve(
            cancellation_reconciliation_pending=True,
            github_repository="owner/repository",
            github_issue_create_title="Issue",
            github_issue_create_marker="marker",
            github_issue_create_operation_state="submitted",
        )

        lifecycle = job.lifecycle
        self.assertEqual(job.loaded_schema_version, CURRENT_SCHEMA_VERSION)
        self.assertIsInstance(lifecycle, ReconcilingJob)
        assert isinstance(lifecycle, ReconcilingJob)
        self.assertIsInstance(lifecycle.cleanup, CleanupReconciling)
        self.assertIsNone(lifecycle.terminal_intent)

    def test_materialized_terminal_outcome_rejects_every_payload_rewrite(self) -> None:
        job = self.job_for_status(JobStatus.COMPLETED)

        for rewrite in (
            lambda current: current.evolve(result="replacement"),
            lambda current: current.evolve(error="replacement"),
            lambda current: current.evolve(completed_at=3),
        ):
            with (
                self.subTest(rewrite=rewrite),
                self.assertRaisesRegex(
                    JobValidationError,
                    "materialized terminal outcome is immutable",
                ),
            ):
                rewrite(job)

    def test_terminal_outcome_cannot_materialize_with_cleanup_fences(self) -> None:
        job = self.job_for_status(JobStatus.RECONCILING).evolve(
            terminal_intent_status="cancelled",
            terminal_intent_result="cancelled",
            terminal_intent_completed_at=2,
            target_release_pending=True,
            target_release_token="release",
            cancellation_reconciliation_pending=True,
        )

        with self.assertRaisesRegex(
            JobValidationError,
            "cannot materialize before cleanup settles",
        ):
            job.evolve(
                status=JobStatus.CANCELLED,
                result="cancelled",
                completed_at=2,
                terminal_intent_status=None,
                terminal_intent_result=None,
                terminal_intent_completed_at=None,
            )

    def test_schema_v10_prompt_boolean_migrates_fail_closed_in_place(self) -> None:
        job = CursorJob.from_dict(
            {
                "schema_version": 10,
                "id": "123456789abc",
                "revision": 0,
                "request": "test",
                "status": "running",
                "created_at": 1,
                "delivered": False,
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_boot_id": "boot",
                "worker_process_start": "start",
                "workflow_phase": "classifying",
                "turn": 2,
                "workflow_turn_phase": "classifying",
                "herdr_target": "planner",
                "planner_target": "planner",
                "active_participant": "planner",
                "phase_prompt_active": True,
            }
        )

        self.assertEqual(job.schema_version, CURRENT_SCHEMA_VERSION)
        self.assertEqual(job.prompt_operation_state, "ambiguous")
        self.assertEqual(job.prompt_operation_turn, 2)
        self.assertEqual(job.prompt_operation_target, "planner")
        self.assertEqual(job.prompt_baseline_sequence, -1)

    def test_schema_v11_defaults_deferred_plan_completion_to_false(self) -> None:
        job = CursorJob.from_dict(
            {
                "schema_version": 11,
                "id": "123456789abc",
                "revision": 0,
                "request": "test",
                "status": "queued",
                "created_at": 1,
                "queued_at": 1,
                "delivered": False,
            }
        )

        self.assertEqual(job.schema_version, CURRENT_SCHEMA_VERSION)
        self.assertFalse(job.plan_approval_completion_pending)

    def test_deferred_plan_completion_requires_durable_finished_output(self) -> None:
        with self.assertRaisesRegex(
            JobValidationError,
            "requires durable finished output",
        ):
            CursorJob.from_dict(
                {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "id": "123456789abc",
                    "revision": 0,
                    "request": "test",
                    "status": "queued",
                    "created_at": 1,
                    "queued_at": 1,
                    "delivered": False,
                    "plan_approval_completion_pending": True,
                }
            )

    def test_all_participant_targets_remain_reserved_during_cleanup(self) -> None:
        first = self.job_for_status(JobStatus.RECONCILING).evolve(
            terminal_intent_status="cancelled",
            terminal_intent_result="cancelled",
            terminal_intent_completed_at=2,
            target_release_pending=True,
            cancellation_reconciliation_pending=True,
            planner_target="planner",
        )
        second = self.job_for_status(JobStatus.QUEUED)
        second = CursorJob.from_dict(
            {
                **second.to_dict(),
                "id": "bbbbbbbbbbbb",
                "herdr_target": "planner",
            }
        )

        with self.assertRaisesRegex(JobValidationError, "planner"):
            validate_reservations([first, second])


if __name__ == "__main__":
    unittest.main()
