from __future__ import annotations

import unittest

from local_voice_harness.cursor import inbox
from local_voice_harness.cursor.model import CursorJob, JobStatus


def _job(job_id: str, **overrides: object) -> CursorJob:
    values: dict[str, object] = {
        "id": job_id,
        "status": "running",
        "request": "do the thing",
        "created_at": 1,
        "delivered": False,
        "worker_token": "claim",
        "worker_pid": 42,
        "worker_boot_id": "boot",
        "worker_process_start": "start",
    }
    values.update(overrides)
    status = str(values["status"])
    if status == "queued":
        values.setdefault("queued_at", 1)
    if status in {"completed", "cancelled", "blocked"}:
        values.setdefault("result", "done")
        values.setdefault("completed_at", 2)
    if status == "failed":
        values.setdefault("result", "boom")
        values.setdefault("error", "boom")
        values.setdefault("completed_at", 2)
    if status == "awaiting_user":
        values.setdefault("question", "which repo?")
        values.setdefault("result", "which repo?")
    if status in {"queued", "completed", "cancelled", "failed", "blocked"}:
        for field in (
            "worker_token",
            "worker_pid",
            "worker_boot_id",
            "worker_process_start",
        ):
            values.pop(field, None)
    return CursorJob.from_dict(values)


class SpeakableLabelTests(unittest.TestCase):
    def test_issue_key_is_preferred_and_repository_prefixed(self) -> None:
        label = inbox.build_speakable_label(
            "fix the login bug",
            issue_key="APP-43",
            github_repository="acme/widgets",
        )
        self.assertEqual(label, "widgets APP-43")

    def test_github_issue_and_pull_request_are_spoken(self) -> None:
        self.assertEqual(
            inbox.build_speakable_label("do it", github_issue=42),
            "issue 42",
        )
        self.assertEqual(
            inbox.build_speakable_label("do it", github_pull_request=7),
            "pull request 7",
        )

    def test_falls_back_to_leading_request_words(self) -> None:
        label = inbox.build_speakable_label(
            "please refactor the authentication module carefully and quickly now"
        )
        self.assertEqual(label, "please refactor the authentication module carefully")

    def test_stored_label_is_used_when_present(self) -> None:
        job = _job("aaaaaaaaaaaa", speakable_label="venice fix")
        self.assertEqual(inbox.speakable_label_for(job), "venice fix")

    def test_legacy_job_without_label_derives_one(self) -> None:
        job = _job("bbbbbbbbbbbb", request="update the readme")
        self.assertEqual(inbox.speakable_label_for(job), "update the readme")


class ReferenceResolutionTests(unittest.TestCase):
    def test_short_id_uniquely_selects_a_job(self) -> None:
        jobs = [_job("aaaa11112222"), _job("bbbb33334444")]
        resolution = inbox.resolve_reference(jobs, "cancel job aaaa")
        assert resolution.unique is not None
        self.assertEqual(resolution.unique.id, "aaaa11112222")

    def test_strong_number_match_beats_shared_label_words(self) -> None:
        jobs = [
            _job("aaaaaaaaaaaa", github_issue=42, speakable_label="issue 42"),
            _job("bbbbbbbbbbbb", github_issue=43, speakable_label="issue 43"),
        ]
        resolution = inbox.resolve_reference(jobs, "check on issue 42")
        assert resolution.unique is not None
        self.assertEqual(resolution.unique.id, "aaaaaaaaaaaa")

    def test_issue_key_matches_even_when_tokenized(self) -> None:
        jobs = [_job("aaaaaaaaaaaa", issue_key="APP-43", speakable_label="APP-43")]
        resolution = inbox.resolve_reference(jobs, "what about app 43")
        assert resolution.unique is not None
        self.assertEqual(resolution.unique.id, "aaaaaaaaaaaa")

    def test_shared_label_words_are_ambiguous(self) -> None:
        jobs = [
            _job("aaaaaaaaaaaa", speakable_label="issue 42"),
            _job("bbbbbbbbbbbb", speakable_label="issue 43"),
        ]
        resolution = inbox.resolve_reference(jobs, "the issue job")
        self.assertTrue(resolution.ambiguous)
        self.assertEqual(len(resolution.matches), 2)

    def test_no_tokens_and_no_match_return_empty(self) -> None:
        jobs = [_job("aaaaaaaaaaaa", speakable_label="venice fix")]
        self.assertEqual(inbox.resolve_reference(jobs, "   ").matches, ())
        self.assertEqual(inbox.resolve_reference(jobs, "the widget report").matches, ())


class InboxSummaryTests(unittest.TestCase):
    def test_empty_inbox_is_reported(self) -> None:
        self.assertEqual(inbox.describe_inbox([]), "You have no Cursor jobs.")

    def test_summary_groups_by_category(self) -> None:
        jobs = [
            _job("aaaaaaaaaaaa", status="running", speakable_label="issue 42"),
            _job(
                "bbbbbbbbbbbb", status="awaiting_user", speakable_label="api refactor"
            ),
            _job("cccccccccccc", status="completed", speakable_label="bug fix"),
        ]
        summary = inbox.describe_inbox(jobs)
        self.assertIn("You have 3 Cursor jobs.", summary)
        self.assertIn("waiting for you: api refactor", summary)
        self.assertIn("in progress: issue 42", summary)
        self.assertIn("recently finished: bug fix", summary)

    def test_large_group_is_truncated(self) -> None:
        jobs = [
            _job(f"{index:012x}", status="running", speakable_label=f"job {index}")
            for index in range(6)
        ]
        summary = inbox.describe_inbox(jobs)
        self.assertIn("2 more", summary)

    def test_clarify_lists_options_with_short_ids(self) -> None:
        summaries = inbox.summarize_all(
            [
                _job("aaaaaaaaaaaa", speakable_label="issue 42"),
                _job("bbbbbbbbbbbb", speakable_label="issue 43"),
            ]
        )
        prompt = inbox.clarify(summaries, "cancel")
        self.assertIn("issue 42 (aaaa)", prompt)
        self.assertIn("issue 43 (bbbb)", prompt)
        self.assertIn("cancel", prompt)

    def test_summary_detail_reflects_status(self) -> None:
        awaiting = inbox.summarize(_job("aaaaaaaaaaaa", status="awaiting_user"))
        self.assertEqual(awaiting.category, "awaiting_user")
        self.assertEqual(awaiting.detail, "which repo?")
        self.assertEqual(awaiting.status, JobStatus.AWAITING_USER)


if __name__ == "__main__":
    unittest.main()
