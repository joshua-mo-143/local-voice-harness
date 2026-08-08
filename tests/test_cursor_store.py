from __future__ import annotations

import hashlib
import json
import stat
import tempfile
import unittest
import warnings
from pathlib import Path

from local_voice_harness.cursor.model import (
    CURRENT_SCHEMA_VERSION,
    JobValidationError,
)
from local_voice_harness.cursor.store import (
    JobQuarantinedError,
    JobQuarantineWarning,
    locked,
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
        for index, version in enumerate((None, 0, 1, 2, 3, 4)):
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
        job.pop("worker_process_start")

        with self.assertRaisesRegex(
            JobValidationError, "routing job requires complete worker ownership"
        ):
            self.write(job)

        self.assertFalse((self.jobs_dir / "123456789abc.json").exists())

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


if __name__ == "__main__":
    unittest.main()
