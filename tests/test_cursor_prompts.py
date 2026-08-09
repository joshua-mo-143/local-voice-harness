from __future__ import annotations

import unittest

from local_voice_harness.cursor.prompts import cursor_prompt


class CursorPromptTests(unittest.TestCase):
    def test_removes_redundant_cursor_delegation_prefix(self) -> None:
        requests = (
            "ask Cursor to fix API-98",
            "Use curser: fix API-98",
            "call cursa, to fix API-98",
        )
        for request in requests:
            with self.subTest(request=request):
                prompt = cursor_prompt(request, "token")
                self.assertIn("User request: fix API-98", prompt)
                self.assertNotIn(f"User request: {request}", prompt)

    def test_preserves_ordinary_request_and_appended_context(self) -> None:
        request = (
            "To fix API-98, inspect this\n\nFocused browser context:\nIssue details"
        )
        prompt = cursor_prompt(request, "token")
        self.assertIn(f"User request: {request}", prompt)

    def test_declares_prepared_workspace_and_continuation_contract(self) -> None:
        prompt = cursor_prompt("Use Cursor to continue", "token", continuation=True)
        self.assertIn("Continue the existing task using this clarification.", prompt)
        self.assertIn(
            "Do not create or switch Git worktrees, Herdr workspaces, tabs, or panes.",
            prompt,
        )
        self.assertIn("VOICE_QUESTION[token]", prompt)
        self.assertIn("VOICE_SUMMARY[token]", prompt)
        self.assertIn("run them only in the foreground", prompt)
        self.assertIn("Never leave background work running", prompt)

    def test_github_issue_context_is_read_only_and_preserved(self) -> None:
        prompt = cursor_prompt(
            "fix the reported bug",
            "token",
            github_issue_context="Repository: example/project\nIssue: #42",
        )
        self.assertIn("Issue: #42", prompt)
        self.assertIn("read it with gh", prompt)
        self.assertIn("Do not comment on, edit, label, assign, close", prompt)

    def test_issue_completion_summary_uses_exact_concise_wording(self) -> None:
        prompt = cursor_prompt(
            "fix API-98",
            "token",
            issue_reference="API-98",
        )

        self.assertIn(
            'the summary must be exactly "I\'ve finished working on API-98"',
            prompt,
        )
        self.assertIn("a plain-text summary of at most 20 words", prompt)


if __name__ == "__main__":
    unittest.main()
