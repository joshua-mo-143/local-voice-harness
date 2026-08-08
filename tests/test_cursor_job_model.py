from __future__ import annotations

import unittest

from local_voice_harness.cursor.model import (
    CURRENT_SCHEMA_VERSION,
    CursorJob,
    JobStatus,
    JobValidationError,
    legal_transitions,
    transition,
    validate_reservations,
)


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
        self.assertTrue(job.announcement_repeated)
        reloaded = CursorJob.from_dict(job.to_dict())
        self.assertEqual(reloaded.speakable_label, "issue 42")
        self.assertTrue(reloaded.announcement_dismissed)

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
                    if target == source or target in legal_transitions(source):
                        updated = transition(job, target, **fields)
                        self.assertEqual(updated.status, target)
                        self.assertEqual(updated.revision, 1)
                    else:
                        with self.assertRaisesRegex(
                            JobValidationError,
                            f"illegal Cursor job transition {source} -> {target}",
                        ):
                            transition(job, target, **fields)


if __name__ == "__main__":
    unittest.main()
