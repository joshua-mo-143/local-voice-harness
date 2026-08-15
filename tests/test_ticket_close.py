from __future__ import annotations

import unittest

from local_voice_harness.ticket_close import (
    MISSING_TICKET_IDENTITY,
    admit_ticket_close,
    close_turn_arguments,
    wants_ticket_close_context,
)
from local_voice_harness.ticket_targets import extract_ticket_targets


class TicketCloseAdmissionTests(unittest.TestCase):
    def test_asks_which_ticket_when_identity_is_missing(self) -> None:
        admission = admit_ticket_close(
            "close this ticket",
            extract_ticket_targets("close this ticket"),
            focused_issue=None,
        )

        self.assertIsNotNone(admission)
        assert admission is not None
        self.assertIsNone(admission.ticket)
        self.assertEqual(admission.missing_identity_response, MISSING_TICKET_IDENTITY)

    def test_binds_focused_or_spoken_identity(self) -> None:
        focused = admit_ticket_close(
            "close this ticket",
            extract_ticket_targets("close this ticket"),
            focused_issue="owner/repo#12",
        )
        spoken = admit_ticket_close(
            "close API-79",
            extract_ticket_targets("close API-79"),
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
        admission = admit_ticket_close(
            "close API-79",
            extract_ticket_targets("close API-79"),
            focused_issue="owner/repo#12",
        )

        self.assertIsNotNone(admission)
        assert admission is not None
        assert admission.ticket is not None
        self.assertEqual(admission.ticket.canonical, "API-79")

    def test_batch_or_unscoped_identity_is_missing(self) -> None:
        batch = admit_ticket_close(
            "close API-79 and API-80",
            extract_ticket_targets("close API-79 and API-80"),
            focused_issue=None,
        )

        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertIsNone(batch.ticket)

    def test_blocked_verbs_are_not_closes(self) -> None:
        for utterance in (
            "implement this ticket",
            "update this ticket",
            "split this ticket",
            "merge these tickets",
            "create an issue",
            "review this ticket",
            "summarize API-79",
            "adversarially review this ticket",
        ):
            with self.subTest(utterance=utterance):
                self.assertIsNone(
                    admit_ticket_close(
                        utterance,
                        extract_ticket_targets(utterance),
                        focused_issue="owner/repo#12",
                    )
                )

    def test_close_context_requires_ticket_noun(self) -> None:
        self.assertTrue(wants_ticket_close_context("close this ticket"))
        self.assertFalse(wants_ticket_close_context("close the laptop"))
        self.assertFalse(wants_ticket_close_context("update this ticket"))

    def test_dispatch_uses_validated_canonical_identity(self) -> None:
        github = admit_ticket_close(
            "close owner/repo#12",
            extract_ticket_targets("close owner/repo#12"),
            focused_issue=None,
        )
        linear = admit_ticket_close(
            "close API-79",
            extract_ticket_targets("close API-79"),
            focused_issue=None,
        )
        assert github is not None and github.ticket is not None
        assert linear is not None and linear.ticket is not None

        github_dispatch = close_turn_arguments(github.ticket)
        linear_dispatch = close_turn_arguments(linear.ticket)

        self.assertEqual(github_dispatch.github_repository, "owner/repo")
        self.assertEqual(github_dispatch.github_issue, 12)
        self.assertTrue(github_dispatch.github_issue_close_requested)
        self.assertFalse(github_dispatch.linear_ticket_close_requested)
        self.assertEqual(linear_dispatch.issue_key, "API-79")
        self.assertTrue(linear_dispatch.linear_ticket_close_requested)
        self.assertFalse(linear_dispatch.github_issue_close_requested)


if __name__ == "__main__":
    unittest.main()
