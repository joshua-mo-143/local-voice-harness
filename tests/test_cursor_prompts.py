from __future__ import annotations

import unittest

from local_voice_harness.cursor.prompts import (
    classification_prompt,
    cursor_prompt,
    review_prompt,
)


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

    def test_classification_contract_is_read_only_and_has_hard_risk_triggers(
        self,
    ) -> None:
        prompt = classification_prompt("change storage", "turn")

        self.assertIn("Do not edit files or implement anything", prompt)
        self.assertIn("persistence", prompt)
        self.assertIn("WORKFLOW_TIER[turn]", prompt)
        self.assertIn("WORKFLOW_REASON[turn]", prompt)

    def test_review_contract_is_independent_and_approval_gated(self) -> None:
        prompt = review_prompt(
            "change storage",
            "Preserve atomic replacement.",
            "turn",
            tier="high-risk",
            github_issue_context="Issue requires crash recovery.",
            classification_reason="Persistence changes force high-risk.",
        )

        self.assertIn("fresh read-only reviewer", prompt)
        self.assertIn("Approve only if implementation may safely start", prompt)
        self.assertIn("WORKFLOW_REVIEW_DECISION[turn]", prompt)
        self.assertIn("Issue requires crash recovery.", prompt)
        self.assertIn("Persistence changes force high-risk.", prompt)


if __name__ == "__main__":
    unittest.main()
