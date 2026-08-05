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


if __name__ == "__main__":
    unittest.main()
