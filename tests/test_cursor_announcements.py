from __future__ import annotations

import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

from local_voice_harness.cursor.announcements import (
    AnnouncementDisposition,
    AnnouncementKind,
    AnnouncementSnooze,
    claim_missed_announcements,
    classify,
    disposition,
    drain_background_announcements,
    in_quiet_hours,
    inspect_missed_announcements,
    render_digest,
)
from local_voice_harness.cursor.delivery import (
    acknowledge_delivery,
    announcement_drain_lock,
    claim_delivery,
)
from local_voice_harness.cursor.model import AnnouncementAck, CursorJob, JobStatus
from local_voice_harness.cursor.store import JobStore
from local_voice_harness.notifications import NotificationResult
from local_voice_harness.responses import AssistantResponse
from local_voice_harness.user_config import AnnouncementMode, AnnouncementSettings
from tests.support import join_threads


def _job(
    job_id: str,
    status: str,
    *,
    completed_at: float = 10,
    ack: str = AnnouncementAck.PENDING.value,
    delivered: bool = False,
    result: str = "done",
    question: str | None = None,
) -> CursorJob:
    values: dict[str, object] = {
        "id": job_id,
        "status": status,
        "request": f"work {job_id}",
        "speakable_label": f"job {job_id[-4:]}",
        "created_at": 1,
        "delivered": delivered,
        "announcement_ack": ack,
        "result": result,
    }
    if status in {"completed", "failed", "cancelled", "blocked"}:
        values["completed_at"] = completed_at
    if status == "failed":
        values["error"] = result
    if status == "awaiting_user":
        values["question"] = question or "Which option?"
        values["result"] = question or "Which option?"
        values["completed_at"] = completed_at
    return CursorJob.from_dict(values)


def _settings(
    mode: AnnouncementMode = AnnouncementMode.ALL,
    *,
    start: str = "",
    end: str = "",
    timezone: str = "",
) -> AnnouncementSettings:
    return AnnouncementSettings(
        mode=mode,
        quiet_hours_start=start,
        quiet_hours_end=end,
        timezone=timezone,
    )


class AnnouncementPolicyTests(unittest.TestCase):
    def test_classifies_each_deliverable_status(self) -> None:
        cases = (
            ("completed", AnnouncementKind.COMPLETION),
            ("awaiting_user", AnnouncementKind.QUESTION),
            ("failed", AnnouncementKind.FAILURE),
            ("cancelled", AnnouncementKind.CANCELLATION),
            ("blocked", AnnouncementKind.INFORMATIONAL),
        )
        for status, kind in cases:
            with self.subTest(status=status):
                self.assertEqual(classify(_job("aaaaaaaaaaaa", status)), kind)

    def test_each_mode_selects_the_documented_disposition(self) -> None:
        question = AnnouncementKind.QUESTION
        completion = AnnouncementKind.COMPLETION
        self.assertEqual(
            disposition(_settings(AnnouncementMode.ALL), completion),
            AnnouncementDisposition.SPEAK,
        )
        self.assertEqual(
            disposition(_settings(AnnouncementMode.ACTION_REQUIRED), question),
            AnnouncementDisposition.SPEAK,
        )
        self.assertEqual(
            disposition(_settings(AnnouncementMode.ACTION_REQUIRED), completion),
            AnnouncementDisposition.DEFER,
        )
        self.assertEqual(
            disposition(_settings(AnnouncementMode.DESKTOP_ONLY), question),
            AnnouncementDisposition.DESKTOP,
        )
        self.assertEqual(
            disposition(_settings(AnnouncementMode.QUIET), question),
            AnnouncementDisposition.DEFER,
        )

    def test_default_snooze_defers_completions_but_speaks_action_required(self) -> None:
        snooze = AnnouncementSnooze(until=200)
        settings = _settings(AnnouncementMode.ALL)
        self.assertEqual(
            disposition(
                settings,
                AnnouncementKind.COMPLETION,
                now=100,
                snooze=snooze,
            ),
            AnnouncementDisposition.DEFER,
        )
        self.assertEqual(
            disposition(
                settings,
                AnnouncementKind.CANCELLATION,
                now=100,
                snooze=snooze,
            ),
            AnnouncementDisposition.DEFER,
        )
        for kind in (
            AnnouncementKind.QUESTION,
            AnnouncementKind.FAILURE,
            AnnouncementKind.INFORMATIONAL,
        ):
            with self.subTest(kind=kind):
                self.assertEqual(
                    disposition(settings, kind, now=100, snooze=snooze),
                    AnnouncementDisposition.SPEAK,
                )

    def test_mute_everything_defers_action_required(self) -> None:
        snooze = AnnouncementSnooze(until=200, mute_everything=True)
        self.assertEqual(
            disposition(
                _settings(AnnouncementMode.ALL),
                AnnouncementKind.QUESTION,
                now=100,
                snooze=snooze,
            ),
            AnnouncementDisposition.DEFER,
        )
        self.assertEqual(
            disposition(
                _settings(AnnouncementMode.ALL),
                AnnouncementKind.QUESTION,
                now=201,
                snooze=snooze,
            ),
            AnnouncementDisposition.SPEAK,
        )


class QuietHoursTests(unittest.TestCase):
    def test_wraparound_window_uses_local_clock(self) -> None:
        settings = _settings(
            start="22:00",
            end="07:00",
            timezone="America/New_York",
        )
        inside = datetime(2026, 1, 15, 23, 0, tzinfo=ZoneInfo("America/New_York"))
        outside = datetime(2026, 1, 15, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        self.assertTrue(in_quiet_hours(settings, now=inside.timestamp()))
        self.assertFalse(in_quiet_hours(settings, now=outside.timestamp()))
        self.assertEqual(
            disposition(
                settings,
                AnnouncementKind.QUESTION,
                now=inside.timestamp(),
            ),
            AnnouncementDisposition.DEFER,
        )

    def test_dst_spring_forward_and_fall_back_use_civil_time(self) -> None:
        settings = _settings(
            start="01:00",
            end="03:00",
            timezone="America/New_York",
        )
        before_spring = datetime(2026, 3, 8, 1, 30, tzinfo=ZoneInfo("America/New_York"))
        after_spring = datetime(2026, 3, 8, 3, 30, tzinfo=ZoneInfo("America/New_York"))
        first_fall = datetime(
            2026, 11, 1, 1, 30, fold=0, tzinfo=ZoneInfo("America/New_York")
        )
        second_fall = datetime(
            2026, 11, 1, 1, 30, fold=1, tzinfo=ZoneInfo("America/New_York")
        )
        self.assertTrue(in_quiet_hours(settings, now=before_spring.timestamp()))
        self.assertFalse(in_quiet_hours(settings, now=after_spring.timestamp()))
        self.assertTrue(in_quiet_hours(settings, now=first_fall.timestamp()))
        self.assertTrue(in_quiet_hours(settings, now=second_fall.timestamp()))


class AnnouncementDrainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.store = JobStore(root / "jobs", root / "legacy")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _create(self, *jobs: CursorJob) -> None:
        for job in jobs:
            self.store.create(job)

    def test_quiet_mode_defers_without_marking_spoken(self) -> None:
        self._create(
            _job("aaaaaaaaaaaa", "completed", completed_at=1),
            _job("bbbbbbbbbbbb", "awaiting_user", completed_at=2),
            _job("cccccccccccc", "failed", completed_at=3, result="boom"),
        )
        result = drain_background_announcements(
            self.store,
            _settings(AnnouncementMode.QUIET),
            now=100,
            notify=mock.Mock(),
            render_job=lambda job: AssistantResponse.from_text(job.id),
        )

        self.assertEqual(result.speak, ())
        self.assertEqual(len(result.deferred), 3)
        for job_id in ("aaaaaaaaaaaa", "bbbbbbbbbbbb", "cccccccccccc"):
            job = self.store.get(job_id)
            self.assertFalse(job.delivered)
            self.assertEqual(job.announcement_ack, AnnouncementAck.DEFERRED.value)
        question = self.store.get("bbbbbbbbbbbb")
        self.assertEqual(question.status, JobStatus.AWAITING_USER)
        self.assertEqual(question.question, "Which option?")

    def test_desktop_only_records_desktop_ack_not_spoken(self) -> None:
        self._create(_job("aaaaaaaaaaaa", "completed", completed_at=1))
        notify = mock.Mock(return_value=NotificationResult.SUCCEEDED)
        result = drain_background_announcements(
            self.store,
            _settings(AnnouncementMode.DESKTOP_ONLY),
            now=100,
            notify=notify,
            render_job=lambda job: AssistantResponse(
                spoken_text="spoken",
                display_text=f"details for {job.id}",
            ),
        )

        self.assertEqual(result.speak, ())
        self.assertEqual(len(result.desktop), 1)
        notify.assert_called_once()
        self.assertIn("details for aaaaaaaaaaaa", notify.call_args.args[0])
        job = self.store.get("aaaaaaaaaaaa")
        self.assertFalse(job.delivered)
        self.assertEqual(job.announcement_ack, AnnouncementAck.DESKTOP.value)

    def test_failed_desktop_notification_retries_without_changing_durable_ack(
        self,
    ) -> None:
        self._create(_job("aaaaaaaaaaaa", "completed", completed_at=1))
        notify = mock.Mock(
            side_effect=[
                NotificationResult.FAILED,
                NotificationResult.SUCCEEDED,
            ]
        )

        failed = drain_background_announcements(
            self.store,
            _settings(AnnouncementMode.DESKTOP_ONLY),
            now=100,
            notify=notify,
            render_job=lambda job: AssistantResponse.from_text(job.id),
        )
        persisted = JobStore(self.store.durable_dir, self.store.legacy_dir)
        pending = persisted.get("aaaaaaaaaaaa")
        self.assertEqual(failed.desktop, ())
        self.assertEqual(
            [claim.job.id for claim in failed.desktop_failed], [pending.id]
        )
        self.assertEqual(pending.announcement_ack, AnnouncementAck.PENDING.value)
        self.assertFalse(pending.delivered)
        self.assertIsNone(pending.delivery_claim_token)

        too_soon = drain_background_announcements(
            persisted,
            _settings(AnnouncementMode.DESKTOP_ONLY),
            now=104,
            notify=notify,
            render_job=lambda job: AssistantResponse.from_text(job.id),
        )
        self.assertEqual(too_soon.desktop, ())
        self.assertEqual(notify.call_count, 1)

        succeeded = drain_background_announcements(
            persisted,
            _settings(AnnouncementMode.DESKTOP_ONLY),
            now=105,
            notify=notify,
            render_job=lambda job: AssistantResponse.from_text(job.id),
        )
        self.assertEqual(len(succeeded.desktop), 1)
        acknowledged = persisted.get("aaaaaaaaaaaa")
        self.assertEqual(acknowledged.announcement_ack, AnnouncementAck.DESKTOP.value)
        self.assertFalse(acknowledged.delivered)

        repeated = drain_background_announcements(
            persisted,
            _settings(AnnouncementMode.DESKTOP_ONLY),
            now=110,
            notify=notify,
            render_job=lambda job: AssistantResponse.from_text(job.id),
        )
        self.assertEqual(repeated.desktop, ())
        self.assertEqual(notify.call_count, 2)

    def test_action_required_speaks_questions_and_defers_completions(self) -> None:
        self._create(
            _job("aaaaaaaaaaaa", "completed", completed_at=1),
            _job("bbbbbbbbbbbb", "awaiting_user", completed_at=2),
        )
        result = drain_background_announcements(
            self.store,
            _settings(AnnouncementMode.ACTION_REQUIRED),
            now=100,
            notify=mock.Mock(),
            render_job=lambda job: AssistantResponse.from_text(job.id),
        )

        self.assertEqual([claim.job.id for claim in result.speak], ["bbbbbbbbbbbb"])
        self.assertEqual([claim.job.id for claim in result.deferred], ["aaaaaaaaaaaa"])
        self.assertEqual(
            self.store.get("aaaaaaaaaaaa").announcement_ack,
            AnnouncementAck.DEFERRED.value,
        )
        self.assertFalse(self.store.get("bbbbbbbbbbbb").delivered)

    def test_all_mode_batches_concurrent_completions_in_order(self) -> None:
        self._create(
            _job("aaaaaaaaaaaa", "completed", completed_at=2),
            _job("bbbbbbbbbbbb", "completed", completed_at=1),
            _job("cccccccccccc", "cancelled", completed_at=3),
        )
        result = drain_background_announcements(
            self.store,
            _settings(AnnouncementMode.ALL),
            now=100,
            notify=mock.Mock(),
            render_job=lambda job: AssistantResponse.from_text(job.id),
        )

        self.assertEqual(
            [claim.job.id for claim in result.speak],
            ["bbbbbbbbbbbb", "aaaaaaaaaaaa", "cccccccccccc"],
        )

    def test_policy_change_from_quiet_to_all_does_not_drop_or_duplicate(self) -> None:
        self._create(_job("aaaaaaaaaaaa", "completed", completed_at=1))
        first = drain_background_announcements(
            self.store,
            _settings(AnnouncementMode.QUIET),
            now=100,
            notify=mock.Mock(),
            render_job=lambda job: AssistantResponse.from_text("quiet"),
        )
        self.assertEqual(len(first.deferred), 1)
        second = drain_background_announcements(
            self.store,
            _settings(AnnouncementMode.QUIET),
            now=101,
            notify=mock.Mock(),
            render_job=lambda job: AssistantResponse.from_text("quiet"),
        )
        self.assertEqual(second.deferred, ())
        third = drain_background_announcements(
            self.store,
            _settings(AnnouncementMode.ALL),
            now=102,
            notify=mock.Mock(),
            render_job=lambda job: AssistantResponse.from_text("all"),
        )
        self.assertEqual([claim.job.id for claim in third.speak], ["aaaaaaaaaaaa"])
        self.assertTrue(
            acknowledge_delivery(
                self.store,
                third.speak[0].job.id,
                third.speak[0].token,
                now=103,
            )
        )
        spoken = self.store.get("aaaaaaaaaaaa")
        self.assertTrue(spoken.delivered)
        self.assertEqual(spoken.announcement_ack, AnnouncementAck.SPOKEN.value)
        fourth = drain_background_announcements(
            self.store,
            _settings(AnnouncementMode.ALL),
            now=104,
            notify=mock.Mock(),
            render_job=lambda job: AssistantResponse.from_text("all"),
        )
        self.assertEqual(fourth.speak, ())

    def test_restart_preserves_deferred_results(self) -> None:
        self._create(_job("aaaaaaaaaaaa", "awaiting_user", completed_at=1))
        drain_background_announcements(
            self.store,
            _settings(AnnouncementMode.QUIET),
            now=100,
            notify=mock.Mock(),
            render_job=lambda job: AssistantResponse.from_text("q"),
        )
        reloaded = JobStore(self.store.durable_dir, self.store.legacy_dir)
        job = reloaded.get("aaaaaaaaaaaa")
        self.assertEqual(job.announcement_ack, AnnouncementAck.DEFERRED.value)
        self.assertEqual(job.status, JobStatus.AWAITING_USER)
        self.assertFalse(job.delivered)

    def test_missed_digest_is_bounded_and_keeps_details_on_display(self) -> None:
        jobs = [
            _job(
                f"{index:012x}", "completed", completed_at=float(index), ack="deferred"
            )
            for index in range(1, 7)
        ]
        self._create(*jobs)
        claimed = claim_missed_announcements(self.store, now=100)
        self.assertEqual(len(claimed), 6)
        digest = render_digest(
            [claim.job for claim in claimed],
            missed=True,
            render_job=lambda job: AssistantResponse(
                spoken_text="hidden",
                display_text=f"detail {job.id}",
            ),
        )
        self.assertIn("You missed 6 background updates", digest.spoken_text)
        self.assertIn("2 more", digest.spoken_text)
        self.assertNotIn("hidden", digest.spoken_text)
        for job in jobs:
            self.assertIn(f"detail {job.id}", digest.display_text)

    def test_cli_inspect_does_not_claim_or_mark_spoken(self) -> None:
        self._create(_job("aaaaaaaaaaaa", "completed", completed_at=1, ack="deferred"))
        jobs = inspect_missed_announcements(self.store)
        self.assertEqual([job.id for job in jobs], ["aaaaaaaaaaaa"])
        self.assertIsNotNone(claim_delivery(self.store, "aaaaaaaaaaaa", now=100))
        persisted = self.store.get("aaaaaaaaaaaa")
        self.assertFalse(persisted.delivered)
        self.assertEqual(persisted.announcement_ack, AnnouncementAck.DEFERRED.value)

    def test_default_snooze_defers_completions_and_keeps_digest(self) -> None:
        self._create(
            _job("aaaaaaaaaaaa", "completed", completed_at=1),
            _job("bbbbbbbbbbbb", "awaiting_user", completed_at=2),
        )
        result = drain_background_announcements(
            self.store,
            _settings(AnnouncementMode.ALL),
            now=100,
            snooze=AnnouncementSnooze(until=200),
            notify=mock.Mock(),
            render_job=lambda job: AssistantResponse.from_text(job.id),
        )
        self.assertEqual([claim.job.id for claim in result.speak], ["bbbbbbbbbbbb"])
        self.assertEqual([claim.job.id for claim in result.deferred], ["aaaaaaaaaaaa"])
        completed = self.store.get("aaaaaaaaaaaa")
        self.assertEqual(completed.announcement_ack, AnnouncementAck.DEFERRED.value)
        self.assertFalse(completed.delivered)
        missed = inspect_missed_announcements(self.store)
        self.assertEqual([job.id for job in missed], ["aaaaaaaaaaaa"])

    def test_concurrent_drainers_have_a_single_winner(self) -> None:
        self._create(_job("aaaaaaaaaaaa", "completed", completed_at=1))
        barrier = threading.Barrier(2)
        started = threading.Event()
        released = threading.Event()
        winners: list[bool] = []

        def race() -> None:
            barrier.wait()
            with announcement_drain_lock(self.store) as acquired:
                winners.append(acquired)
                if acquired:
                    started.set()
                    released.wait(timeout=1)
                else:
                    started.wait(timeout=1)
                    released.set()

        threads = [threading.Thread(target=race) for _ in range(2)]
        for thread in threads:
            thread.start()
        join_threads(threads)
        self.assertEqual(sorted(winners), [False, True])
