from __future__ import annotations

import json
import unittest
from unittest import mock

from local_voice_harness.errors import HarnessError
from local_voice_harness.linear_ticket_creation import (
    draft_linear_ticket,
    team_from_utterance,
)


class LinearTicketDraftTests(unittest.TestCase):
    def test_extracts_explicit_team_from_trusted_utterance(self) -> None:
        self.assertEqual(
            team_from_utterance("Create a Linear ticket in team API about startup"),
            "API",
        )
        self.assertIsNone(team_from_utterance("Create a Linear ticket in this team"))

    def test_generates_a_bounded_structured_draft(self) -> None:
        transport = mock.Mock()
        transport.chat_completion.return_value = {
            "tool_calls": [
                {
                    "function": {
                        "name": "draft_linear_ticket",
                        "arguments": json.dumps(
                            {
                                "title": "  Fix   launcher startup  ",
                                "description": "The launcher fails after reboot.",
                            }
                        ),
                    }
                }
            ]
        }
        with mock.patch(
            "local_voice_harness.linear_ticket_creation.LlmTransport.from_settings",
            return_value=transport,
        ):
            draft = draft_linear_ticket(
                "Create a ticket about the launcher failing after reboot",
                "api",
            )

        self.assertEqual(draft.team, "API")
        self.assertEqual(draft.title, "Fix launcher startup")
        self.assertEqual(draft.description, "The launcher fails after reboot.")
        request = transport.chat_completion.call_args.args[0]
        self.assertEqual(request.tool_choice["function"]["name"], "draft_linear_ticket")
        self.assertEqual(request.temperature, 0)

    def test_rejects_oversized_draft(self) -> None:
        transport = mock.Mock()
        transport.chat_completion.return_value = {
            "tool_calls": [
                {
                    "function": {
                        "name": "draft_linear_ticket",
                        "arguments": json.dumps(
                            {"title": "x" * 256, "description": "Details"}
                        ),
                    }
                }
            ]
        }
        with (
            mock.patch(
                "local_voice_harness.linear_ticket_creation.LlmTransport.from_settings",
                return_value=transport,
            ),
            self.assertRaisesRegex(HarnessError, "title is too long"),
        ):
            draft_linear_ticket("Create a ticket", "API")


if __name__ == "__main__":
    unittest.main()
