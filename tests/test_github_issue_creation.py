from __future__ import annotations

import json
import unittest
from unittest import mock

from local_voice_harness.errors import HarnessError
from local_voice_harness.github_issue_creation import (
    draft_github_issue,
    issue_draft_from_trusted_brief,
    repository_from_utterance,
)


class GitHubIssueDraftTests(unittest.TestCase):
    def test_extracts_explicit_repository_from_trusted_utterance(self) -> None:
        self.assertEqual(
            repository_from_utterance(
                "Create an issue in example/project about the broken launcher"
            ),
            "example/project",
        )
        self.assertIsNone(repository_from_utterance("Create an issue in this repo"))

    def test_generates_a_bounded_structured_draft(self) -> None:
        transport = mock.Mock()
        transport.chat_completion.return_value = {
            "tool_calls": [
                {
                    "function": {
                        "name": "draft_github_issue",
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
            "local_voice_harness.github_issue_creation.LlmTransport.from_settings",
            return_value=transport,
        ):
            draft = draft_github_issue(
                "Create an issue about the launcher failing after reboot",
                "example/project",
            )

        self.assertEqual(draft.repository, "example/project")
        self.assertEqual(draft.title, "Fix launcher startup")
        self.assertEqual(draft.body, "The launcher fails after reboot.")
        request = transport.chat_completion.call_args.args[0]
        self.assertEqual(request.tool_choice["function"]["name"], "draft_github_issue")
        self.assertEqual(request.temperature, 0)

    def test_rejects_malformed_or_oversized_draft(self) -> None:
        transport = mock.Mock()
        transport.chat_completion.return_value = {
            "tool_calls": [
                {
                    "function": {
                        "name": "draft_github_issue",
                        "arguments": json.dumps(
                            {"title": "x" * 201, "body": "Details"}
                        ),
                    }
                }
            ]
        }
        with (
            mock.patch(
                "local_voice_harness.github_issue_creation.LlmTransport.from_settings",
                return_value=transport,
            ),
            self.assertRaisesRegex(HarnessError, "title is too long"),
        ):
            draft_github_issue("Create an issue", "example/project")

    def test_trusted_brief_becomes_title_and_body(self) -> None:
        draft = issue_draft_from_trusted_brief(
            "create me a SaaS called widgets",
            "alice/widgets",
        )
        self.assertEqual(draft.repository, "alice/widgets")
        self.assertEqual(draft.title, "create me a SaaS called widgets")
        self.assertEqual(draft.body, "create me a SaaS called widgets")

    def test_trusted_brief_truncates_oversized_title(self) -> None:
        brief = "word " * 80
        draft = issue_draft_from_trusted_brief(brief, "alice/widgets")
        self.assertLessEqual(len(draft.title), 200)
        self.assertTrue(draft.title.endswith("…"))
        self.assertEqual(draft.body, brief.strip())


if __name__ == "__main__":
    unittest.main()
