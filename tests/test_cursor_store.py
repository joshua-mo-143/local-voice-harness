from __future__ import annotations

import hashlib
import json
import stat
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

from local_voice_harness.cursor.model import (
    CURRENT_SCHEMA_VERSION,
    CursorJob,
    JobStatus,
    JobValidationError,
    transition,
)
from local_voice_harness.cursor.store import (
    JobQuarantinedError,
    JobQuarantineWarning,
    JobStore,
    locked,
    migrate_legacy_jobs,
    prune_jobs,
    read_all_unlocked,
    read_unlocked,
    write_unlocked,
)


class CursorStoreIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.jobs_dir = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def job(
        self,
        job_id: str,
        *,
        status: str = "queued",
        revision: int = 0,
        **fields: object,
    ) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "id": job_id,
            "revision": revision,
            "request": "do it",
            "status": status,
            "created_at": 1,
            "delivered": False,
        }
        if status == "queued":
            value["queued_at"] = 1
        if status in {"routing", "running", "reconciling"}:
            value.update(
                {
                    "worker_token": f"claim-{job_id}",
                    "worker_pid": 42,
                    "worker_boot_id": "boot",
                    "worker_process_start": f"start-{job_id}",
                }
            )
        value.update(fields)
        return value

    def write(self, job: dict[str, object]) -> None:
        path = self.jobs_dir / f"{job['id']}.json"
        with locked(self.jobs_dir):
            write_unlocked(path, job)

    def test_legacy_read_then_write_migrates_to_current_schema(self) -> None:
        for index, version in enumerate((None, 0, 1, 2, 3, 4, 5)):
            with self.subTest(version=version):
                job_id = f"{index:012x}"
                path = self.jobs_dir / f"{job_id}.json"
                persisted_legacy: dict[str, object] = {
                    "id": job_id,
                    "status": "completed",
                    "result": "old result",
                    "completed_at": 2,
                }
                if version is not None:
                    persisted_legacy["schema_version"] = version
                path.write_text(json.dumps(persisted_legacy))

                with locked(self.jobs_dir):
                    legacy = read_unlocked(path)
                    if version in {None, 0}:
                        self.assertNotIn("schema_version", legacy)
                    else:
                        self.assertEqual(legacy["schema_version"], version)
                    legacy.update({"result": "new result", "revision": 1})
                    write_unlocked(path, legacy)

                persisted = json.loads(path.read_text())
                self.assertEqual(persisted["schema_version"], CURRENT_SCHEMA_VERSION)
                self.assertEqual(persisted["revision"], 1)
                self.assertEqual(persisted["result"], "new result")
                self.assertIn("created_at", persisted)
                self.assertIn("delivered", persisted)

    def test_new_worker_owned_job_requires_complete_ownership(self) -> None:
        job = self.job("123456789abc", status="routing")
        job.pop("worker_boot_id")

        with self.assertRaisesRegex(
            JobValidationError, "routing job requires complete worker ownership"
        ):
            self.write(job)

        self.assertFalse((self.jobs_dir / "123456789abc.json").exists())

    def test_legacy_incomplete_owner_is_fenced_before_current_schema_write(
        self,
    ) -> None:
        path = self.jobs_dir / "123456789abc.json"
        legacy = {
            "schema_version": 5,
            "id": "123456789abc",
            "revision": 0,
            "request": "legacy",
            "status": "running",
            "created_at": 1,
            "delivered": False,
            "worker_token": "claim",
            "worker_pid": 42,
            "worker_process_start": "start",
        }

        with locked(self.jobs_dir):
            write_unlocked(path, legacy)
            persisted = read_unlocked(path)

        self.assertEqual(persisted["schema_version"], CURRENT_SCHEMA_VERSION)
        self.assertEqual(persisted["status"], "failed")
        self.assertIsNone(persisted["worker_token"])
        self.assertIsNone(persisted["worker_pid"])
        self.assertIsNone(persisted["worker_boot_id"])
        self.assertIn("manual recovery", str(persisted["error"]))

    def test_legacy_runtime_import_is_idempotent_and_leaves_logs_transient(
        self,
    ) -> None:
        legacy = self.jobs_dir / "runtime"
        durable = self.jobs_dir / "state"
        legacy.mkdir()
        source = legacy / "123456789abc.json"
        source.write_text(
            json.dumps(
                {
                    "schema_version": 5,
                    "id": "123456789abc",
                    "revision": 3,
                    "request": "do it",
                    "status": "running",
                    "created_at": 1,
                    "delivered": False,
                    "worker_token": "claim",
                    "worker_pid": 42,
                    "worker_process_start": "start",
                }
            )
        )
        log = legacy / "123456789abc.log"
        log.write_text("worker output")

        migrate_legacy_jobs(legacy, durable, inspect_worker=lambda _job: "stopped")
        migrate_legacy_jobs(legacy, durable, inspect_worker=lambda _job: "stopped")

        imported = json.loads((durable / "123456789abc.json").read_text())
        self.assertEqual(imported["schema_version"], CURRENT_SCHEMA_VERSION)
        self.assertEqual(imported["status"], "queued")
        self.assertIsNone(imported["worker_boot_id"])
        self.assertIsNone(imported["worker_pid"])
        self.assertFalse(source.exists())
        self.assertEqual(log.read_text(), "worker output")

    def test_unowned_legacy_active_statuses_requeue_as_absent(self) -> None:
        legacy = self.jobs_dir / "runtime"
        durable = self.jobs_dir / "state"
        legacy.mkdir()
        inspector = mock.Mock()
        for index, status in enumerate(("routing", "running", "reconciling")):
            job_id = f"{index + 1:012x}"
            (legacy / f"{job_id}.json").write_text(
                json.dumps(
                    {
                        "schema_version": 5,
                        "id": job_id,
                        "revision": 0,
                        "request": "legacy",
                        "status": status,
                        "created_at": 1,
                        "delivered": False,
                        "herdr_target": f"agent-{index}",
                    }
                )
            )

        migrate_legacy_jobs(legacy, durable, inspect_worker=inspector)

        inspector.assert_not_called()
        for index in range(3):
            imported = json.loads((durable / f"{index + 1:012x}.json").read_text())
            self.assertEqual(imported["status"], "queued")
            self.assertTrue(imported["reconcile"])
            self.assertIsNone(imported["worker_pid"])

    def test_legacy_import_never_overwrites_higher_revision_and_quarantines_conflict(
        self,
    ) -> None:
        legacy = self.jobs_dir / "runtime"
        durable = self.jobs_dir / "state"
        legacy.mkdir()
        durable.mkdir()
        higher = self.job("aaaaaaaaaaaa", revision=5)
        (durable / "aaaaaaaaaaaa.json").write_text(json.dumps(higher))
        (legacy / "aaaaaaaaaaaa.json").write_text(
            json.dumps(self.job("aaaaaaaaaaaa", revision=4, request="stale"))
        )
        conflicting = self.job("bbbbbbbbbbbb", revision=2)
        (durable / "bbbbbbbbbbbb.json").write_text(json.dumps(conflicting))
        (legacy / "bbbbbbbbbbbb.json").write_text(
            json.dumps(self.job("bbbbbbbbbbbb", revision=2, request="conflict"))
        )

        with self.assertWarnsRegex(
            JobQuarantineWarning, "legacy import is quarantined"
        ):
            migrate_legacy_jobs(legacy, durable)

        self.assertEqual(
            json.loads((durable / "aaaaaaaaaaaa.json").read_text())["revision"], 5
        )
        self.assertFalse((legacy / "aaaaaaaaaaaa.json").exists())
        self.assertEqual(
            json.loads((durable / "bbbbbbbbbbbb.json").read_text())["request"],
            "do it",
        )
        self.assertTrue(
            list((durable / ".quarantine").glob("bbbbbbbbbbbb-legacy-import-*.json"))
        )

    def test_legacy_import_respects_new_durable_quarantine_evidence(self) -> None:
        legacy = self.jobs_dir / "runtime"
        durable = self.jobs_dir / "state"
        legacy.mkdir()
        durable.mkdir()
        malformed = self.job(
            "aaaaaaaaaaaa",
            status="running",
            worktree_path="/worktrees/shared",
        )
        malformed["parent_job_id"] = "invalid"
        (durable / "aaaaaaaaaaaa.json").write_text(json.dumps(malformed))
        (legacy / "bbbbbbbbbbbb.json").write_text(
            json.dumps(
                self.job(
                    "bbbbbbbbbbbb",
                    worktree_path="/worktrees/shared",
                )
            )
        )

        with self.assertWarns(JobQuarantineWarning):
            migrate_legacy_jobs(legacy, durable)

        self.assertFalse((durable / "bbbbbbbbbbbb.json").exists())
        self.assertTrue(
            list((durable / ".quarantine").glob("bbbbbbbbbbbb-legacy-import-*.json"))
        )

    def test_lower_revision_collision_requires_matching_created_at_lineage(
        self,
    ) -> None:
        legacy = self.jobs_dir / "runtime"
        durable = self.jobs_dir / "state"
        legacy.mkdir()
        durable.mkdir()
        durable_job = self.job("aaaaaaaaaaaa", revision=5, created_at=1)
        legacy_job = self.job("aaaaaaaaaaaa", revision=4, created_at=2)
        (durable / "aaaaaaaaaaaa.json").write_text(json.dumps(durable_job))
        (legacy / "aaaaaaaaaaaa.json").write_text(json.dumps(legacy_job))

        with self.assertWarnsRegex(
            JobQuarantineWarning, "identity/created_at lineage conflicts"
        ):
            migrate_legacy_jobs(legacy, durable)

        self.assertEqual(
            json.loads((durable / "aaaaaaaaaaaa.json").read_text())["created_at"],
            1,
        )
        self.assertTrue(
            list((durable / ".quarantine").glob("aaaaaaaaaaaa-legacy-import-*.json"))
        )

    def test_unsafe_live_legacy_worker_preserves_source_and_blocks_import(
        self,
    ) -> None:
        legacy = self.jobs_dir / "runtime"
        durable = self.jobs_dir / "state"
        legacy.mkdir()
        source = legacy / "123456789abc.json"
        source.write_text(
            json.dumps(
                {
                    "schema_version": 5,
                    "id": "123456789abc",
                    "revision": 0,
                    "request": "legacy",
                    "status": "running",
                    "created_at": 1,
                    "delivered": False,
                    "worker_pid": 42,
                    "worker_process_start": "start",
                }
            )
        )

        with self.assertWarnsRegex(JobQuarantineWarning, "could not be stopped safely"):
            blocked = migrate_legacy_jobs(
                legacy, durable, inspect_worker=lambda _job: "unsafe"
            )

        self.assertEqual(blocked, {"123456789abc"})
        self.assertTrue(source.exists())
        self.assertFalse((durable / source.name).exists())

    def test_pruning_removes_only_old_delivered_unfenced_terminal_jobs(self) -> None:
        old = 100.0
        self.write(
            self.job(
                "aaaaaaaaaaaa",
                status="completed",
                result="done",
                completed_at=old,
                delivered=True,
            )
        )
        self.write(
            self.job(
                "bbbbbbbbbbbb",
                status="completed",
                result="pending delivery",
                completed_at=old,
                delivered=False,
            )
        )
        self.write(
            self.job(
                "cccccccccccc",
                status="failed",
                result="uncertain",
                error="uncertain",
                completed_at=old,
                delivered=True,
                herdr_target="agent",
                agent_dispatch_state="ambiguous",
            )
        )
        self.write(
            self.job(
                "dddddddddddd",
                status="failed",
                result="fenced",
                error="fenced",
                completed_at=old,
                delivered=True,
                target_release_pending=True,
            )
        )
        quarantine = self.jobs_dir / ".quarantine"
        quarantine.mkdir()
        evidence = quarantine / "evidence.json"
        evidence.write_text("{}")

        removed = prune_jobs(self.jobs_dir, now=old + 8 * 24 * 60 * 60)

        self.assertEqual(removed, ["aaaaaaaaaaaa"])
        self.assertFalse((self.jobs_dir / "aaaaaaaaaaaa.json").exists())
        self.assertTrue((self.jobs_dir / "bbbbbbbbbbbb.json").exists())
        self.assertTrue((self.jobs_dir / "cccccccccccc.json").exists())
        self.assertTrue((self.jobs_dir / "dddddddddddd.json").exists())
        self.assertTrue(evidence.exists())

    def test_non_finite_numeric_values_are_rejected(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf"), "1e100000"):
            with self.subTest(value=value):
                job = self.job("123456789abc", created_at=value)
                with self.assertRaisesRegex(
                    JobValidationError, "created_at must be finite"
                ):
                    self.write(job)
        worker = self.job("123456789abc", status="routing", worker_pid=float("inf"))
        with self.assertRaisesRegex(
            JobValidationError, "worker_pid must be an integer"
        ):
            self.write(worker)

    def test_duplicate_target_and_worktree_reservations_are_rejected(self) -> None:
        self.write(
            self.job(
                "aaaaaaaaaaaa",
                status="running",
                herdr_target="shared-agent",
                worktree_path="/worktrees/shared",
            )
        )

        with self.assertRaisesRegex(
            JobValidationError, "Herdr target shared-agent is reserved by both"
        ):
            self.write(
                self.job(
                    "bbbbbbbbbbbb",
                    status="running",
                    herdr_target="shared-agent",
                    worktree_path="/worktrees/other",
                )
            )
        with self.assertRaisesRegex(
            JobValidationError, "worktree /worktrees/shared is reserved by both"
        ):
            self.write(
                self.job(
                    "cccccccccccc",
                    status="running",
                    herdr_target="other-agent",
                    worktree_path="/worktrees/shared",
                )
            )

    def test_active_quarantine_blocks_conflicting_reservations_until_acknowledged(
        self,
    ) -> None:
        store = JobStore(self.jobs_dir, self.jobs_dir / "legacy")
        quarantined_path = self.jobs_dir / "aaaaaaaaaaaa.json"
        quarantined_path.write_text(
            json.dumps(
                {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "id": "aaaaaaaaaaaa",
                    "revision": 0,
                    "request": "corrupt active",
                    "status": "running",
                    "created_at": 1,
                    "delivered": False,
                    "herdr_target": "shared-agent",
                    "worktree_path": "/worktrees/shared",
                }
            )
        )
        with self.assertWarns(JobQuarantineWarning):
            self.assertEqual(store.list(), [])
        store.create(CursorJob.from_dict(self.job("bbbbbbbbbbbb")))

        with self.assertRaisesRegex(
            JobValidationError, "blocked by unresolved quarantine evidence"
        ):
            store.reserve_target(
                "bbbbbbbbbbbb",
                lambda job: transition(
                    job, JobStatus.QUEUED, herdr_target="shared-agent"
                ),
            )
        with self.assertRaisesRegex(
            JobValidationError, "blocked by unresolved quarantine evidence"
        ):
            store.reserve_worktree(
                "bbbbbbbbbbbb",
                lambda job: transition(
                    job, JobStatus.QUEUED, worktree_path="/worktrees/shared"
                ),
            )

        acknowledgement = store.acknowledge_quarantine_reservations(
            "aaaaaaaaaaaa",
            reason="operator verified no external reservation remains",
            now=100,
        )
        reserved = store.reserve_target(
            "bbbbbbbbbbbb",
            lambda job: transition(job, JobStatus.QUEUED, herdr_target="shared-agent"),
        )

        self.assertIsNotNone(reserved)
        self.assertEqual(acknowledgement.acknowledged_at, 100)
        self.assertTrue(
            list((self.jobs_dir / ".quarantine").glob("aaaaaaaaaaaa-*.json"))
        )

    def test_newly_quarantined_peer_blocks_plain_create_reservation(self) -> None:
        store = JobStore(self.jobs_dir, self.jobs_dir / "legacy")
        malformed = self.job(
            "aaaaaaaaaaaa",
            status="running",
            herdr_target="shared-agent",
        )
        malformed["parent_job_id"] = "invalid"
        (self.jobs_dir / "aaaaaaaaaaaa.json").write_text(json.dumps(malformed))
        candidate = CursorJob.from_dict(
            self.job("bbbbbbbbbbbb", herdr_target="shared-agent")
        )

        with (
            self.assertWarns(JobQuarantineWarning),
            self.assertRaisesRegex(
                JobValidationError,
                "blocked by unresolved quarantine evidence",
            ),
        ):
            store.create(candidate)

        self.assertFalse((self.jobs_dir / "bbbbbbbbbbbb.json").exists())

    def test_terminal_uncertain_operation_keeps_live_target_reservation(self) -> None:
        self.write(
            self.job(
                "aaaaaaaaaaaa",
                status="failed",
                completed_at=2,
                error="timed out",
                result="timed out",
                herdr_target="shared-agent",
                agent_dispatch_state="ambiguous",
                cancellation_reconciliation_pending=True,
            )
        )

        with self.assertRaisesRegex(JobValidationError, "reserved by both"):
            self.write(
                self.job(
                    "bbbbbbbbbbbb",
                    herdr_target="shared-agent",
                )
            )

    def test_terminal_quarantined_pull_request_keeps_worktree_reservation(
        self,
    ) -> None:
        self.write(
            self.job(
                "aaaaaaaaaaaa",
                status="failed",
                completed_at=2,
                error="checkout failed",
                result="checkout failed",
                worktree_path="/worktrees/quarantined",
                worktree_branch="voice/github-pr-aaaaaaaaaaaa",
                pull_request_worktree_state="quarantined",
            )
        )

        with self.assertRaisesRegex(JobValidationError, "reserved by both"):
            self.write(
                self.job(
                    "bbbbbbbbbbbb",
                    worktree_path="/worktrees/quarantined",
                )
            )

    def test_quarantine_evidence_does_not_block_existing_owner_release(self) -> None:
        store = JobStore(self.jobs_dir, self.jobs_dir / "legacy")
        store.create(
            CursorJob.from_dict(self.job("aaaaaaaaaaaa", herdr_target="owned-agent"))
        )
        (self.jobs_dir / "bbbbbbbbbbbb.json").write_text("{unknown")

        with self.assertWarns(JobQuarantineWarning):
            updated = store.update(
                "aaaaaaaaaaaa",
                lambda current: transition(
                    current,
                    JobStatus.CANCELLED,
                    result="cancelled",
                    completed_at=2,
                    target_release_pending=True,
                ),
            )

        assert updated is not None
        self.assertTrue(updated.target_release_pending)

    def test_unknown_quarantine_blocks_any_new_reservation(self) -> None:
        store = JobStore(self.jobs_dir, self.jobs_dir / "legacy")
        unknown = self.jobs_dir / "aaaaaaaaaaaa.json"
        unknown.write_text("{unknown")
        with self.assertWarns(JobQuarantineWarning):
            store.list()
        store.create(CursorJob.from_dict(self.job("bbbbbbbbbbbb")))

        for reserve in (
            lambda: store.reserve_target(
                "bbbbbbbbbbbb",
                lambda job: transition(job, JobStatus.QUEUED, herdr_target="agent"),
            ),
            lambda: store.reserve_worktree(
                "bbbbbbbbbbbb",
                lambda job: transition(
                    job, JobStatus.QUEUED, worktree_path="/worktrees/new"
                ),
            ),
        ):
            with self.assertRaisesRegex(
                JobValidationError, "unresolved quarantine evidence"
            ):
                reserve()

    def test_malformed_terminal_quarantine_does_not_block_reservations(self) -> None:
        store = JobStore(self.jobs_dir, self.jobs_dir / "legacy")
        terminal = self.jobs_dir / "aaaaaaaaaaaa.json"
        terminal.write_text(
            json.dumps(
                {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "id": "aaaaaaaaaaaa",
                    "revision": 0,
                    "request": "terminal evidence",
                    "status": "completed",
                    "created_at": 1,
                    "delivered": False,
                }
            )
        )
        with self.assertWarns(JobQuarantineWarning):
            store.list()
        store.create(CursorJob.from_dict(self.job("bbbbbbbbbbbb")))

        reserved = store.reserve_target(
            "bbbbbbbbbbbb",
            lambda job: transition(job, JobStatus.QUEUED, herdr_target="agent"),
        )

        self.assertIsNotNone(reserved)

    def test_update_enforces_immutable_creation_fields_and_exact_revision(self) -> None:
        original = self.job("123456789abc")
        self.write(original)

        changed_id = self.job("bbbbbbbbbbbb", revision=1)
        with (
            locked(self.jobs_dir),
            self.assertRaisesRegex(JobValidationError, "id must match its filename"),
        ):
            write_unlocked(
                self.jobs_dir / "123456789abc.json",
                changed_id,
            )

        changed_creation = self.job("123456789abc", revision=1, created_at=2)
        with self.assertRaisesRegex(JobValidationError, "cannot change created_at"):
            self.write(changed_creation)

        skipped_revision = self.job("123456789abc", revision=2)
        with self.assertRaisesRegex(
            JobValidationError, "revision must increase by exactly one"
        ):
            self.write(skipped_revision)

    def test_same_status_cancellation_reconciliation_update_is_valid(self) -> None:
        cancelled = self.job(
            "123456789abc",
            status="cancelled",
            result="cancelled; reconciliation pending",
            completed_at=2,
            herdr_target="late-agent",
            agent_dispatch_state="ambiguous",
            target_release_pending=True,
            cancellation_reconciliation_pending=True,
        )
        self.write(cancelled)
        cancelled.update(
            {
                "revision": 1,
                "agent_dispatch_state": "ready",
            }
        )

        self.write(cancelled)

        persisted = json.loads((self.jobs_dir / "123456789abc.json").read_text())
        self.assertEqual(persisted["status"], "cancelled")
        self.assertEqual(persisted["agent_dispatch_state"], "ready")
        self.assertTrue(persisted["target_release_pending"])

    def test_malformed_peer_is_quarantined_while_valid_jobs_continue(self) -> None:
        first = self.job("aaaaaaaaaaaa")
        self.write(first)
        malformed_path = self.jobs_dir / "bbbbbbbbbbbb.json"
        malformed_path.write_text("{not json")

        with self.assertWarnsRegex(
            JobQuarantineWarning, "bbbbbbbbbbbb.json: job is quarantined"
        ):
            self.write(self.job("cccccccccccc"))

        with locked(self.jobs_dir):
            jobs = read_all_unlocked(self.jobs_dir)
            with self.assertRaisesRegex(
                JobQuarantinedError, "bbbbbbbbbbbb.json: job is quarantined"
            ):
                read_unlocked(malformed_path)

        self.assertEqual({job["id"] for job in jobs}, {"aaaaaaaaaaaa", "cccccccccccc"})
        quarantine = self.jobs_dir / ".quarantine"
        self.assertEqual(stat.S_IMODE(quarantine.stat().st_mode), 0o700)
        quarantined_jobs = [
            path
            for path in quarantine.glob("bbbbbbbbbbbb-*.json")
            if not path.name.endswith(".metadata.json")
        ]
        metadata = list(quarantine.glob("bbbbbbbbbbbb-*.metadata.json"))
        self.assertEqual(len(quarantined_jobs), 1)
        self.assertEqual(len(metadata), 1)
        self.assertEqual(stat.S_IMODE(quarantined_jobs[0].stat().st_mode), 0o600)
        evidence = json.loads(metadata[0].read_text())
        self.assertEqual(evidence["original_name"], "bbbbbbbbbbbb.json")
        self.assertIn("invalid JSON", evidence["error"])

    def test_invalid_utf8_scan_preserves_payload_and_continues_valid_jobs(
        self,
    ) -> None:
        self.write(self.job("aaaaaaaaaaaa"))
        invalid_path = self.jobs_dir / "bbbbbbbbbbbb.json"
        payload = b'{"request":"\xff\xfe"}'
        invalid_path.write_bytes(payload)

        with (
            locked(self.jobs_dir),
            self.assertWarnsRegex(
                JobQuarantineWarning, "bbbbbbbbbbbb.json:.*invalid UTF-8"
            ),
        ):
            jobs = read_all_unlocked(self.jobs_dir)

        self.assertEqual([job["id"] for job in jobs], ["aaaaaaaaaaaa"])
        quarantine = self.jobs_dir / ".quarantine"
        quarantined = [
            path
            for path in quarantine.glob("bbbbbbbbbbbb-*.json")
            if not path.name.endswith(".metadata.json")
        ]
        metadata = list(quarantine.glob("bbbbbbbbbbbb-*.metadata.json"))
        self.assertEqual(len(quarantined), 1)
        self.assertEqual(len(metadata), 1)
        self.assertEqual(quarantined[0].read_bytes(), payload)
        self.assertEqual(stat.S_IMODE(quarantine.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(quarantined[0].stat().st_mode), 0o600)
        evidence = json.loads(metadata[0].read_text())
        self.assertEqual(evidence["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(evidence["error"], "bbbbbbbbbbbb.json: invalid UTF-8")

    def test_direct_invalid_utf8_read_quarantines_once_and_reports_tombstone(
        self,
    ) -> None:
        invalid_path = self.jobs_dir / "123456789abc.json"
        invalid_path.write_bytes(b"\xff\xfe\x80")

        with (
            locked(self.jobs_dir),
            self.assertWarnsRegex(JobQuarantineWarning, "invalid UTF-8"),
            self.assertRaisesRegex(
                JobQuarantinedError, "123456789abc.json: job is quarantined"
            ),
        ):
            read_unlocked(invalid_path)

        with warnings.catch_warnings(record=True) as repeated_warnings:
            warnings.simplefilter("always")
            with (
                locked(self.jobs_dir),
                self.assertRaisesRegex(
                    JobQuarantinedError,
                    "123456789abc.json: job is quarantined.*invalid UTF-8",
                ),
            ):
                read_unlocked(invalid_path)

        quarantine = self.jobs_dir / ".quarantine"
        self.assertEqual(repeated_warnings, [])
        self.assertEqual(
            len(
                [
                    path
                    for path in quarantine.glob("123456789abc-*.json")
                    if not path.name.endswith(".metadata.json")
                ]
            ),
            1,
        )
        self.assertEqual(len(list(quarantine.glob("123456789abc-*.metadata.json"))), 1)

    def test_typed_store_returns_models_and_writes_exact_revisions(self) -> None:
        store = JobStore(self.jobs_dir, self.jobs_dir / "legacy")
        created = store.create(CursorJob.from_dict(self.job("123456789abc")))

        updated = store.update(
            created.id,
            lambda job: transition(
                job,
                JobStatus.ROUTING,
                **{
                    "worker_token": "claim",
                    "worker_pid": 42,
                    "worker_boot_id": "boot",
                    "worker_process_start": "start",
                },
            ),
        )

        assert updated is not None
        self.assertIsInstance(store.get(created.id), CursorJob)
        self.assertTrue(all(isinstance(job, CursorJob) for job in store.list()))
        self.assertEqual(updated.revision, 1)
        self.assertEqual(store.get(created.id).revision, 1)

    def test_typed_store_failed_command_does_not_write(self) -> None:
        store = JobStore(self.jobs_dir, self.jobs_dir / "legacy")
        created = store.create(CursorJob.from_dict(self.job("123456789abc")))
        before = (self.jobs_dir / "123456789abc.json").read_bytes()

        def fail(_job: CursorJob) -> CursorJob:
            raise RuntimeError("command failed")

        with self.assertRaisesRegex(RuntimeError, "command failed"):
            store.update(created.id, fail)

        self.assertEqual((self.jobs_dir / "123456789abc.json").read_bytes(), before)
        self.assertEqual(store.get(created.id).revision, 0)

    def test_typed_reservation_transaction_has_single_winner(self) -> None:
        store = JobStore(self.jobs_dir, self.jobs_dir / "legacy")
        for job_id in ("aaaaaaaaaaaa", "bbbbbbbbbbbb"):
            store.create(CursorJob.from_dict(self.job(job_id)))

        def reserve(job: CursorJob) -> CursorJob:
            return transition(job, JobStatus.QUEUED, herdr_target="shared")

        first = store.reserve_target("aaaaaaaaaaaa", reserve)
        with self.assertRaisesRegex(JobValidationError, "reserved by both"):
            store.reserve_target("bbbbbbbbbbbb", reserve)

        self.assertIsNotNone(first)
        self.assertEqual(store.get("bbbbbbbbbbbb").revision, 0)


if __name__ == "__main__":
    unittest.main()
