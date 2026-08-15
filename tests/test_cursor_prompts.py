from __future__ import annotations

import unittest

from local_voice_harness.cursor.prompts import (
    MAX_PROMPT_CHARS,
    SECTION_LIMITS,
    PromptSizeError,
    bounded_prompt_payload,
    classification_prompt,
    continuation_prompt,
    cursor_prompt,
    plan_approval_prompt,
    planning_prompt,
    review_prompt,
    revision_prompt,
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
        self.assertIn(
            "Do not modify external systems, commit, push, open a pull request",
            prompt,
        )

    def test_implementation_and_follow_up_prompts_still_forbid_git_writes(
        self,
    ) -> None:
        forbidden = "Do not modify external systems, commit, push, open a pull request"
        self.assertIn(forbidden, cursor_prompt("implement the change", "token"))
        self.assertIn(
            forbidden,
            cursor_prompt("review the changes", "token", continuation=True),
        )
        self.assertIn(
            forbidden,
            planning_prompt("plan the change", "token", tier="simple"),
        )

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

    def test_plan_approval_prompt_binds_fresh_implementation_token(
        self,
    ) -> None:
        prompt = plan_approval_prompt(
            "change storage",
            "turn",
            plan="Preserve atomic replacement.",
        )

        self.assertTrue(prompt.startswith("Implement the following user request"))
        self.assertNotIn("lgtm", prompt)
        self.assertIn("Implement only from this approved plan", prompt)
        self.assertIn("VOICE_SUMMARY[turn]", prompt)
        self.assertIn("WORKFLOW_PROMOTE[turn]", prompt)

    def test_linear_instructions_are_only_added_as_a_contribution(self) -> None:
        without = cursor_prompt("fix API-98", "token", issue_reference="API-98")
        with_linear = cursor_prompt(
            "fix API-98",
            "token",
            issue_reference="API-98",
            integration_instructions=(
                "Use configured Linear MCP tools only to read it.",
            ),
        )

        self.assertNotIn("Linear MCP", without)
        self.assertIn("Linear MCP", with_linear)

    def test_integration_instructions_reach_every_read_only_phase(self) -> None:
        instructions = ("Use configured Linear MCP tools only to read it.",)
        prompts = (
            classification_prompt(
                "fix API-98",
                "token",
                integration_instructions=instructions,
            ),
            planning_prompt(
                "fix API-98",
                "token",
                tier="medium",
                integration_instructions=instructions,
            ),
            review_prompt(
                "fix API-98",
                "plan",
                "token",
                tier="medium",
                integration_instructions=instructions,
            ),
            revision_prompt(
                "fix API-98",
                "plan",
                "review",
                "token",
                integration_instructions=instructions,
            ),
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt.splitlines()[0]):
                self.assertIn("Trusted integration instructions", prompt)
                self.assertIn("Linear MCP", prompt)

    def test_same_session_continuation_contains_only_delta_and_markers(self) -> None:
        prompt = continuation_prompt(
            "planning",
            "Question: Which format?\nUser answered: Keep JSON",
            "turn",
        )

        self.assertIn("Keep JSON", prompt)
        self.assertIn("WORKFLOW_PLAN[turn]", prompt)
        self.assertIn("VOICE_QUESTION[turn]", prompt)
        self.assertNotIn("User request:", prompt)
        self.assertNotIn("Approved-plan candidate:", prompt)

    def test_prompt_manifest_is_redacted_and_records_section_hashes(self) -> None:
        payload = bounded_prompt_payload(
            "bounded prompt",
            phase="planning",
            session_identity="session-1",
            full_rehydration=True,
            sections={"request": "secret request", "plan": "safe plan"},
        )

        self.assertEqual(payload.manifest["phase"], "planning")
        self.assertTrue(payload.manifest["full_rehydration"])
        self.assertNotIn("secret request", repr(payload.manifest))
        sections = payload.manifest["sections"]
        assert isinstance(sections, dict)
        self.assertEqual(sections["request"]["chars"], len("secret request"))
        self.assertEqual(len(sections["request"]["sha256"]), 64)

    def test_required_oversized_context_fails_closed(self) -> None:
        with self.assertRaises(PromptSizeError):
            bounded_prompt_payload(
                "x" * (MAX_PROMPT_CHARS + 1),
                phase="reviewing",
                session_identity="session-1",
                full_rehydration=True,
                sections={"issue_context": "issue"},
            )

        with self.assertRaisesRegex(PromptSizeError, "issue_context"):
            bounded_prompt_payload(
                "small envelope",
                phase="reviewing",
                session_identity="session-1",
                full_rehydration=True,
                sections={"issue_context": "x" * (SECTION_LIMITS["issue_context"] + 1)},
            )


if __name__ == "__main__":
    unittest.main()
