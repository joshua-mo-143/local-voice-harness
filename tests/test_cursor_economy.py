from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness.cursor.economy import (
    resolve_cursor_model,
    verify_checkout,
)
from local_voice_harness.cursor.model import (
    WorkflowParticipant,
    WorkflowPhase,
    WorkflowTier,
)
from local_voice_harness.cursor.prompts import implementation_prompt
from local_voice_harness.user_config import (
    CursorEconomySettings,
    CursorModelSettings,
)


class CursorEconomyTests(unittest.TestCase):
    def test_resolve_cursor_model_is_disabled_by_default(self) -> None:
        economy = CursorEconomySettings()
        self.assertIsNone(
            resolve_cursor_model(
                economy,
                participant=WorkflowParticipant.IMPLEMENTER,
                phase=WorkflowPhase.IMPLEMENTING,
                tier=WorkflowTier.SIMPLE,
            )
        )

    def test_resolve_cursor_model_routes_simple_implementer(self) -> None:
        economy = CursorEconomySettings(
            enabled=True,
            models=CursorModelSettings(implementer="gpt-5.6-luna-medium"),
        )
        self.assertEqual(
            resolve_cursor_model(
                economy,
                participant=WorkflowParticipant.IMPLEMENTER,
                phase=WorkflowPhase.IMPLEMENTING,
                tier=WorkflowTier.SIMPLE,
            ),
            "gpt-5.6-luna-medium",
        )

    def test_resolve_cursor_model_uses_high_risk_override(self) -> None:
        economy = CursorEconomySettings(
            enabled=True,
            models=CursorModelSettings(
                implementer="gpt-5.6-luna-medium",
                implementer_high_risk="composer-1.5",
            ),
        )
        self.assertEqual(
            resolve_cursor_model(
                economy,
                participant=WorkflowParticipant.IMPLEMENTER,
                phase=WorkflowPhase.IMPLEMENTING,
                tier=WorkflowTier.HIGH_RISK,
            ),
            "composer-1.5",
        )

    def test_implementation_prompt_includes_economy_policy_when_requested(self) -> None:
        prompt = implementation_prompt("fix it", "token-1", economy_simple=True)
        self.assertIn("Economy mode:", prompt)
        self.assertIn("WORKFLOW_PROMOTE", prompt)

    def test_verify_checkout_reports_missing_directory(self) -> None:
        outcome = verify_checkout(Path("/tmp/does-not-exist-for-voice-harness"))
        self.assertFalse(outcome.passed)
        self.assertIn("missing", outcome.detail.casefold())

    def test_verify_checkout_runs_configured_commands(self) -> None:
        with mock.patch(
            "local_voice_harness.cursor.economy.subprocess.run",
            return_value=mock.Mock(returncode=0, stdout="", stderr=""),
        ) as run:
            outcome = verify_checkout(Path("/tmp"))

        self.assertTrue(outcome.passed)
        self.assertEqual(run.call_count, 4)
