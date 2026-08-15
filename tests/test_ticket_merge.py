from __future__ import annotations

import unittest
from unittest import mock

from local_voice_harness.ticket_merge import (
    MISSING_TICKET_IDENTITY,
    MergeClosingTicket,
    TicketMergeDraft,
    admit_ticket_merge,
    decode_merge_closing,
    draft_ticket_merge,
    encode_merge_closing,
    merge_preview,
    merge_result_message,
    merge_turn_arguments,
    spoken_merge_confirmation,
    wants_ticket_merge_context,
)
from local_voice_harness.ticket_snapshot import TicketSnapshot
from local_voice_harness.ticket_targets import extract_ticket_targets


class TicketMergeAdmissionTests(unittest.TestCase):
    def test_asks_which_tickets_when_identity_is_missing(self) -> None:
        admission = admit_ticket_merge(
            "merge these tickets",
            extract_ticket_targets("merge these tickets"),
            focused_issue=None,
        )

        self.assertIsNotNone(admission)
        assert admission is not None
        self.assertIsNone(admission.survivor)
        self.assertEqual(admission.missing_identity_response, MISSING_TICKET_IDENTITY)

    def test_requires_at_least_two_tickets(self) -> None:
        admission = admit_ticket_merge(
            "merge this ticket",
            extract_ticket_targets("merge this ticket"),
            focused_issue="owner/repo#12",
        )

        self.assertIsNotNone(admission)
        assert admission is not None
        self.assertIsNone(admission.survivor)

    def test_binds_spoken_tickets_and_first_as_survivor(self) -> None:
        admission = admit_ticket_merge(
            "merge API-79 and API-80",
            extract_ticket_targets("merge API-79 and API-80"),
            focused_issue=None,
        )

        self.assertIsNotNone(admission)
        assert admission is not None
        assert admission.survivor is not None
        self.assertEqual(admission.survivor.canonical, "API-79")
        self.assertEqual(
            [ticket.canonical for ticket in admission.tickets],
            ["API-79", "API-80"],
        )

    def test_explicit_into_selects_survivor(self) -> None:
        admission = admit_ticket_merge(
            "merge API-79 and API-80 into API-80",
            extract_ticket_targets("merge API-79 and API-80 into API-80"),
            focused_issue=None,
        )

        self.assertIsNotNone(admission)
        assert admission is not None
        assert admission.survivor is not None
        self.assertEqual(admission.survivor.canonical, "API-80")

    def test_focused_plus_spoken_ticket_is_enough(self) -> None:
        admission = admit_ticket_merge(
            "merge this ticket and API-80",
            extract_ticket_targets("merge this ticket and API-80"),
            focused_issue="API-79",
        )

        self.assertIsNotNone(admission)
        assert admission is not None
        assert admission.survivor is not None
        self.assertEqual(admission.survivor.canonical, "API-79")
        self.assertEqual(
            [ticket.canonical for ticket in admission.tickets],
            ["API-79", "API-80"],
        )

    def test_mixed_sources_are_missing(self) -> None:
        admission = admit_ticket_merge(
            "merge owner/repo#12 and API-79",
            extract_ticket_targets("merge owner/repo#12 and API-79"),
            focused_issue=None,
        )

        self.assertIsNotNone(admission)
        assert admission is not None
        self.assertIsNone(admission.survivor)

    def test_blocked_verbs_are_not_merges(self) -> None:
        for utterance in (
            "implement these tickets",
            "split these tickets",
            "review these tickets",
            "summarize API-79 and API-80",
            "merge the pull request",
            "work on these tickets",
        ):
            with self.subTest(utterance=utterance):
                self.assertIsNone(
                    admit_ticket_merge(
                        utterance,
                        extract_ticket_targets(utterance),
                        focused_issue="owner/repo#12",
                    )
                )

    def test_merge_context_requires_ticket_noun(self) -> None:
        self.assertTrue(wants_ticket_merge_context("merge these tickets"))
        self.assertFalse(wants_ticket_merge_context("merge the laptop"))
        self.assertFalse(wants_ticket_merge_context("close these tickets"))

    def test_dispatch_uses_validated_canonical_set(self) -> None:
        github = admit_ticket_merge(
            "merge owner/repo#12 and owner/repo#13",
            extract_ticket_targets("merge owner/repo#12 and owner/repo#13"),
            focused_issue=None,
        )
        linear = admit_ticket_merge(
            "merge API-79 and API-80",
            extract_ticket_targets("merge API-79 and API-80"),
            focused_issue=None,
        )
        assert github is not None and github.survivor is not None
        assert linear is not None and linear.survivor is not None

        github_dispatch = merge_turn_arguments(github.survivor, github.tickets)
        linear_dispatch = merge_turn_arguments(linear.survivor, linear.tickets)

        self.assertEqual(github_dispatch.github_repository, "owner/repo")
        self.assertEqual(github_dispatch.github_issue, 12)
        self.assertTrue(github_dispatch.github_issue_merge_requested)
        self.assertEqual(github_dispatch.ticket_merge_survivor, "owner/repo#12")
        self.assertIn("owner/repo#13", github_dispatch.ticket_merge_closing or "")
        self.assertEqual(linear_dispatch.issue_key, "API-79")
        self.assertTrue(linear_dispatch.linear_ticket_merge_requested)
        self.assertEqual(linear_dispatch.ticket_merge_survivor, "API-79")


class TicketMergeDraftTests(unittest.TestCase):
    def test_spoken_confirmation_names_survivor_and_closing(self) -> None:
        self.assertEqual(
            spoken_merge_confirmation("API-79", ("API-80", "API-81")),
            "Update API-79 and close API-80 and API-81?",
        )
        self.assertEqual(
            spoken_merge_confirmation("owner/repo#12", ("owner/repo#13",)),
            "Update owner/repo#12 and close owner/repo#13?",
        )

    def test_preview_displays_survivor_closing_and_body(self) -> None:
        preview = merge_preview(
            "API-79",
            TicketMergeDraft("Combined auth", "Handle login and invoices."),
            (MergeClosingTicket("API-80", "a" * 32),),
        )

        self.assertIn("Update API-79 and close API-80?", preview)
        self.assertIn("Survivor: API-79", preview)
        self.assertIn("Title: Combined auth", preview)
        self.assertIn("Handle login and invoices.", preview)
        self.assertIn("- API-80", preview)

    def test_encode_decode_round_trip_and_result_message(self) -> None:
        closing = (
            MergeClosingTicket("API-80", "a" * 32, state="created"),
            MergeClosingTicket("API-81", "b" * 32, state="ambiguous"),
        )
        decoded = decode_merge_closing(encode_merge_closing(closing))

        self.assertEqual(decoded, closing)
        self.assertEqual(
            merge_result_message("API-79", closing, survivor_state="created"),
            "Updated API-79. Closed API-80. Close outcome requires manual "
            "verification for: API-81.",
        )
        self.assertEqual(
            merge_result_message("API-79", closing, survivor_state=None),
            "API-79 was not updated. Closed API-80. Close outcome requires manual "
            "verification for: API-81.",
        )

    def test_draft_uses_forced_tool_call(self) -> None:
        with mock.patch(
            "local_voice_harness.ticket_merge.LlmTransport.from_settings"
        ) as transport_factory:
            transport = mock.Mock()
            transport.chat_completion.return_value = {
                "tool_calls": [
                    {
                        "function": {
                            "name": "draft_ticket_merge",
                            "arguments": (
                                '{"title":"Combined auth",'
                                '"body":"Handle login and invoices."}'
                            ),
                        }
                    }
                ]
            }
            transport_factory.return_value = transport
            draft = draft_ticket_merge(
                "merge API-79 and API-80",
                "API-79",
                (
                    TicketSnapshot(
                        "linear",
                        "API-79",
                        "issue-79",
                        "Auth",
                        "Handle login.",
                        "revision-79",
                        "https://linear.app/acme/issue/API-79/auth",
                        "In Progress",
                    ),
                    TicketSnapshot(
                        "linear",
                        "API-80",
                        "issue-80",
                        "Billing",
                        "Handle invoices.",
                        "revision-80",
                        "https://linear.app/acme/issue/API-80/billing",
                        "In Progress",
                    ),
                ),
            )

        self.assertEqual(draft.title, "Combined auth")
        self.assertEqual(draft.body, "Handle login and invoices.")
        request = transport.chat_completion.call_args.args[0]
        self.assertIn("Handle login.", request.messages[1]["content"])
        self.assertIn("Handle invoices.", request.messages[1]["content"])
