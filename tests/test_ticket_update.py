from __future__ import annotations

import json
import unittest
from unittest import mock

from local_voice_harness.errors import HarnessError
from local_voice_harness.ticket_snapshot import TicketSnapshot
from local_voice_harness.ticket_targets import extract_ticket_targets
from local_voice_harness.ticket_update import (
    MISSING_TICKET_IDENTITY,
    admit_ticket_update,
    draft_ticket_update,
    update_turn_arguments,
    wants_ticket_update_context,
)


class TicketUpdateAdmissionTests(unittest.TestCase):
    def test_asks_which_ticket_when_identity_is_missing(self) -> None:
        admission = admit_ticket_update(
            "update this ticket title",
            extract_ticket_targets("update this ticket title"),
            focused_issue=None,
        )

        self.assertIsNotNone(admission)
        assert admission is not None
        self.assertIsNone(admission.ticket)
        self.assertEqual(admission.missing_identity_response, MISSING_TICKET_IDENTITY)

    def test_binds_focused_or_spoken_identity(self) -> None:
        focused = admit_ticket_update(
            "update this ticket title",
            extract_ticket_targets("update this ticket title"),
            focused_issue="owner/repo#12",
        )
        spoken = admit_ticket_update(
            "change the title of API-79",
            extract_ticket_targets("change the title of API-79"),
            focused_issue=None,
        )

        self.assertIsNotNone(focused)
        assert focused is not None
        assert focused.ticket is not None
        self.assertEqual(focused.ticket.canonical, "owner/repo#12")
        self.assertIsNotNone(spoken)
        assert spoken is not None
        assert spoken.ticket is not None
        self.assertEqual(spoken.ticket.canonical, "API-79")

    def test_spoken_identity_wins_over_focused_ticket(self) -> None:
        admission = admit_ticket_update(
            "update the body of API-79",
            extract_ticket_targets("update the body of API-79"),
            focused_issue="owner/repo#12",
        )

        self.assertIsNotNone(admission)
        assert admission is not None
        assert admission.ticket is not None
        self.assertEqual(admission.ticket.canonical, "API-79")

    def test_batch_or_unscoped_identity_is_missing(self) -> None:
        batch = admit_ticket_update(
            "update API-79 and API-80 titles",
            extract_ticket_targets("update API-79 and API-80 titles"),
            focused_issue=None,
        )

        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertIsNone(batch.ticket)

    def test_blocked_verbs_are_not_updates(self) -> None:
        for utterance in (
            "implement this ticket",
            "close this ticket",
            "split this ticket",
            "merge these tickets",
            "create an issue",
            "review this ticket",
            "summarize API-79",
            "adversarially review this ticket",
        ):
            with self.subTest(utterance=utterance):
                self.assertIsNone(
                    admit_ticket_update(
                        utterance,
                        extract_ticket_targets(utterance),
                        focused_issue="owner/repo#12",
                    )
                )

    def test_update_context_requires_ticket_or_title_body(self) -> None:
        self.assertTrue(wants_ticket_update_context("update this ticket"))
        self.assertTrue(wants_ticket_update_context("change the title"))
        self.assertFalse(wants_ticket_update_context("update the code"))
        self.assertFalse(wants_ticket_update_context("close this ticket"))

    def test_dispatch_uses_validated_canonical_identity(self) -> None:
        github = admit_ticket_update(
            "update owner/repo#12 title",
            extract_ticket_targets("update owner/repo#12 title"),
            focused_issue=None,
        )
        linear = admit_ticket_update(
            "update API-79 description",
            extract_ticket_targets("update API-79 description"),
            focused_issue=None,
        )
        assert github is not None and github.ticket is not None
        assert linear is not None and linear.ticket is not None

        github_dispatch = update_turn_arguments(github.ticket)
        linear_dispatch = update_turn_arguments(linear.ticket)

        self.assertEqual(github_dispatch.github_repository, "owner/repo")
        self.assertEqual(github_dispatch.github_issue, 12)
        self.assertTrue(github_dispatch.github_issue_update_requested)
        self.assertFalse(github_dispatch.linear_ticket_update_requested)
        self.assertEqual(linear_dispatch.issue_key, "API-79")
        self.assertTrue(linear_dispatch.linear_ticket_update_requested)
        self.assertFalse(linear_dispatch.github_issue_update_requested)


class TicketUpdateDraftTests(unittest.TestCase):
    snapshot = TicketSnapshot(
        "github",
        "owner/repo#12",
        "https://github.com/owner/repo/issues/12",
        "Current title",
        "Current body",
        "2026-08-15T10:00:00Z",
        "https://github.com/owner/repo/issues/12",
        "OPEN",
    )

    def test_generates_a_bounded_structured_draft(self) -> None:
        transport = mock.Mock()
        transport.chat_completion.return_value = {
            "tool_calls": [
                {
                    "function": {
                        "name": "draft_ticket_update",
                        "arguments": json.dumps(
                            {
                                "title": "  Fix   launcher startup  ",
                                "body": "The launcher fails after reboot.",
                            }
                        ),
                    }
                }
            ]
        }
        with mock.patch(
            "local_voice_harness.ticket_update.LlmTransport.from_settings",
            return_value=transport,
        ):
            draft = draft_ticket_update(
                "Rewrite the title of owner/repo#12 to mention startup",
                self.snapshot,
            )

        self.assertEqual(draft.title, "Fix launcher startup")
        self.assertEqual(draft.body, "The launcher fails after reboot.")
        request = transport.chat_completion.call_args.args[0]
        self.assertEqual(request.tool_choice["function"]["name"], "draft_ticket_update")
        self.assertEqual(request.temperature, 0)

    def test_rejects_malformed_or_oversized_draft(self) -> None:
        transport = mock.Mock()
        transport.chat_completion.return_value = {
            "tool_calls": [
                {
                    "function": {
                        "name": "draft_ticket_update",
                        "arguments": json.dumps(
                            {"title": "x" * 201, "body": "Details"}
                        ),
                    }
                }
            ]
        }
        with (
            mock.patch(
                "local_voice_harness.ticket_update.LlmTransport.from_settings",
                return_value=transport,
            ),
            self.assertRaisesRegex(HarnessError, "title is too long"),
        ):
            draft_ticket_update("Update the ticket", self.snapshot)

    def test_preserves_fields_not_requested_by_the_user(self) -> None:
        transport = mock.Mock()
        transport.chat_completion.return_value = {
            "tool_calls": [
                {
                    "function": {
                        "name": "draft_ticket_update",
                        "arguments": json.dumps({"title": "New title", "body": None}),
                    }
                }
            ]
        }
        with mock.patch(
            "local_voice_harness.ticket_update.LlmTransport.from_settings",
            return_value=transport,
        ):
            draft = draft_ticket_update("Change only the title", self.snapshot)

        self.assertEqual(draft.title, "New title")
        self.assertEqual(draft.body, "Current body")
        self.assertTrue(draft.title_changed)
        self.assertFalse(draft.body_changed)


if __name__ == "__main__":
    unittest.main()
