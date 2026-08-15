from __future__ import annotations

import json
import unittest
from unittest import mock

from local_voice_harness.errors import HarnessError
from local_voice_harness.ticket_snapshot import TicketSnapshot
from local_voice_harness.ticket_split import (
    MISSING_TICKET_IDENTITY,
    SplitChild,
    TicketSplitDraft,
    admit_ticket_split,
    assign_split_markers,
    decode_split_children,
    draft_ticket_split,
    encode_split_children,
    linear_team_key,
    split_parent_identity,
    split_preview,
    split_result_message,
    split_turn_arguments,
    spoken_split_confirmation,
    wants_ticket_split_context,
)
from local_voice_harness.ticket_targets import extract_ticket_targets


class TicketSplitAdmissionTests(unittest.TestCase):
    def test_asks_which_ticket_when_identity_is_missing(self) -> None:
        admission = admit_ticket_split(
            "split this ticket into two issues",
            extract_ticket_targets("split this ticket into two issues"),
            focused_issue=None,
        )

        self.assertIsNotNone(admission)
        assert admission is not None
        self.assertIsNone(admission.ticket)
        self.assertEqual(admission.missing_identity_response, MISSING_TICKET_IDENTITY)

    def test_binds_focused_or_spoken_identity(self) -> None:
        focused = admit_ticket_split(
            "split this ticket into two issues",
            extract_ticket_targets("split this ticket into two issues"),
            focused_issue="owner/repo#12",
        )
        spoken = admit_ticket_split(
            "split API-79 into two tickets",
            extract_ticket_targets("split API-79 into two tickets"),
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
        admission = admit_ticket_split(
            "split API-79 into two tickets",
            extract_ticket_targets("split API-79 into two tickets"),
            focused_issue="owner/repo#12",
        )

        self.assertIsNotNone(admission)
        assert admission is not None
        assert admission.ticket is not None
        self.assertEqual(admission.ticket.canonical, "API-79")

    def test_batch_or_unscoped_identity_is_missing(self) -> None:
        batch = admit_ticket_split(
            "split API-79 and API-80",
            extract_ticket_targets("split API-79 and API-80"),
            focused_issue=None,
        )

        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertIsNone(batch.ticket)

    def test_blocked_verbs_are_not_splits(self) -> None:
        for utterance in (
            "implement this ticket",
            "merge these tickets",
            "review this ticket",
            "summarize API-79",
            "adversarially review this ticket",
            "work on this ticket",
            "fix this ticket",
        ):
            with self.subTest(utterance=utterance):
                self.assertIsNone(
                    admit_ticket_split(
                        utterance,
                        extract_ticket_targets(utterance),
                        focused_issue="owner/repo#12",
                    )
                )

    def test_split_context_requires_ticket_noun_or_identity(self) -> None:
        self.assertTrue(wants_ticket_split_context("split this ticket"))
        self.assertFalse(wants_ticket_split_context("split the laptop"))
        self.assertFalse(wants_ticket_split_context("close this ticket"))

    def test_dispatch_uses_validated_canonical_identity(self) -> None:
        github = admit_ticket_split(
            "split owner/repo#12 into two issues",
            extract_ticket_targets("split owner/repo#12 into two issues"),
            focused_issue=None,
        )
        linear = admit_ticket_split(
            "split API-79 into two tickets",
            extract_ticket_targets("split API-79 into two tickets"),
            focused_issue=None,
        )
        assert github is not None and github.ticket is not None
        assert linear is not None and linear.ticket is not None

        github_dispatch = split_turn_arguments(github.ticket)
        linear_dispatch = split_turn_arguments(linear.ticket)

        self.assertEqual(github_dispatch.github_repository, "owner/repo")
        self.assertEqual(github_dispatch.github_issue, 12)
        self.assertTrue(github_dispatch.github_issue_split_requested)
        self.assertFalse(github_dispatch.linear_ticket_split_requested)
        self.assertEqual(linear_dispatch.issue_key, "API-79")
        self.assertTrue(linear_dispatch.linear_ticket_split_requested)
        self.assertFalse(linear_dispatch.github_issue_split_requested)


class TicketSplitDraftTests(unittest.TestCase):
    snapshot = TicketSnapshot(
        "linear",
        "API-79",
        "linear-issue-id",
        "Current parent",
        "Current parent body",
        "revision-1",
        "https://linear.app/acme/issue/API-79/current-parent",
        "In Progress",
    )

    def test_spoken_confirmation_names_count_and_parent(self) -> None:
        self.assertEqual(
            spoken_split_confirmation("API-79", 3, "close"),
            "Create 3 issues and close API-79?",
        )
        self.assertEqual(
            spoken_split_confirmation("owner/repo#12", 1, "update"),
            "Create 1 issue and update owner/repo#12?",
        )
        self.assertEqual(
            spoken_split_confirmation("API-79", 2, "none"),
            "Create 2 issues from API-79?",
        )

    def test_preview_displays_the_exact_child_set(self) -> None:
        draft = TicketSplitDraft(
            (
                SplitChild("Auth", "Handle login.", "a" * 32),
                SplitChild("Billing", "Handle invoices.", "b" * 32),
            ),
            "close",
        )
        preview = split_preview("API-79", draft)

        self.assertIn("Create 2 issues and close API-79?", preview)
        self.assertIn("Child 1 title: Auth", preview)
        self.assertIn("Handle login.", preview)
        self.assertIn("Child 2 title: Billing", preview)
        self.assertIn("Parent API-79 will be closed.", preview)

    def test_encode_decode_round_trip_and_result_message(self) -> None:
        children = (
            SplitChild(
                "Auth",
                "Handle login.",
                "a" * 32,
                state="created",
                created_ref="owner/repo#21",
            ),
            SplitChild("Billing", "Handle invoices.", "b" * 32, state="ambiguous"),
        )
        decoded = decode_split_children(encode_split_children(children))

        self.assertEqual(decoded, children)
        self.assertEqual(
            split_result_message(
                "owner/repo#12",
                decoded,
                parent_action="close",
                parent_state="planned",
            ),
            "Created child tickets owner/repo#21. Creation outcome requires manual "
            "verification for: Billing. owner/repo#12 was not closed.",
        )

    def test_assign_split_markers_replaces_placeholders(self) -> None:
        draft = TicketSplitDraft(
            (SplitChild("Auth", "Handle login.", "0" * 32),),
            "none",
        )

        assigned = assign_split_markers(draft)

        self.assertNotEqual(assigned.children[0].marker, "0" * 32)
        self.assertRegex(assigned.children[0].marker, r"^[0-9a-f]{32}$")

    def test_linear_team_key_and_parent_identity(self) -> None:
        self.assertEqual(linear_team_key("API-79"), "API")
        self.assertEqual(
            split_parent_identity(
                github_repository="owner/repo",
                github_issue=12,
                issue_key=None,
            ),
            "owner/repo#12",
        )

    def test_draft_ticket_split_uses_forced_tool(self) -> None:
        transport = mock.Mock()
        transport.chat_completion.return_value = {
            "tool_calls": [
                {
                    "function": {
                        "name": "draft_ticket_split",
                        "arguments": json.dumps(
                            {
                                "children": [
                                    {"title": "Auth", "body": "Handle login."},
                                ],
                                "parent_action": "close",
                            }
                        ),
                    }
                }
            ]
        }
        with mock.patch(
            "local_voice_harness.ticket_split.LlmTransport.from_settings",
            return_value=transport,
        ):
            draft = draft_ticket_split("split API-79", self.snapshot)

        self.assertEqual(draft.parent_action, "close")
        self.assertEqual(draft.children[0].title, "Auth")
        self.assertEqual(draft.children[0].body, "Handle login.")
        request = transport.chat_completion.call_args.args[0]
        self.assertIn("Current parent body", request.messages[1]["content"])

    def test_rejects_duplicate_normalized_child_titles(self) -> None:
        transport = mock.Mock()
        transport.chat_completion.return_value = {
            "tool_calls": [
                {
                    "function": {
                        "name": "draft_ticket_split",
                        "arguments": json.dumps(
                            {
                                "children": [
                                    {"title": "Auth flow", "body": "First."},
                                    {"title": "  AUTH   FLOW ", "body": "Second."},
                                ],
                                "parent_action": "none",
                            }
                        ),
                    }
                }
            ]
        }
        with (
            mock.patch(
                "local_voice_harness.ticket_split.LlmTransport.from_settings",
                return_value=transport,
            ),
            self.assertRaisesRegex(HarnessError, "duplicate child titles"),
        ):
            draft_ticket_split("split API-79", self.snapshot)


if __name__ == "__main__":
    unittest.main()
