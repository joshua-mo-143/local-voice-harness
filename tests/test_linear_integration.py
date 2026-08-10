from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness.cursor import service
from local_voice_harness.errors import HarnessError
from local_voice_harness.integrations import linear, registry
from local_voice_harness.user_config import IntegrationSettings

DISABLED = IntegrationSettings(linear_enabled=False)
ENABLED = IntegrationSettings(linear_enabled=True)


class LinearEnablementTests(unittest.TestCase):
    def test_disabled_integration_contributes_nothing(self) -> None:
        self.assertIsNone(registry.extract_issue_reference("work on API-42", DISABLED))
        self.assertEqual(registry.prompt_instructions("API-42", DISABLED), ())
        self.assertEqual(registry.capability_statuses(DISABLED), ())

    def test_enabled_integration_extracts_and_contributes_prompt(self) -> None:
        self.assertEqual(
            registry.extract_issue_reference("work on API 42", ENABLED), "API-42"
        )
        self.assertEqual(registry.resolve_issue_reference("api-42", ENABLED), "API-42")
        instructions = registry.prompt_instructions("API-42", ENABLED)
        self.assertTrue(
            any("Linear MCP" in instruction for instruction in instructions)
        )
        self.assertTrue(
            any(
                "untrusted external data" in instruction for instruction in instructions
            )
        )

    def test_enabled_integration_recognizes_url_with_external_provenance(self) -> None:
        fragment = registry.capture_context(
            "https://linear.app/acme/issue/API-42/fix-it", ENABLED
        )
        self.assertIsNotNone(fragment)
        assert fragment is not None
        self.assertEqual(fragment.source, "linear")
        self.assertEqual(fragment.issue_reference, "API-42")
        self.assertIn("untrusted external identifier", fragment.text)

    def test_disabled_explicit_issue_key_is_not_persisted(self) -> None:
        with (
            mock.patch.object(registry, "_integration_settings", return_value=DISABLED),
            mock.patch.object(service, "_job_store") as store,
            mock.patch.object(service, "launch_worker"),
        ):
            service.start_job("work on this ticket", issue_key="ENG-123")

        created = store.return_value.create.call_args.args[0]
        self.assertIsNone(created.issue_key)
        self.assertNotEqual(created.speakable_label, "ENG-123")

    def test_missing_capability_rejects_job_before_persistence(self) -> None:
        with (
            mock.patch.object(registry, "_integration_settings", return_value=ENABLED),
            mock.patch.object(
                linear.LinearIntegration,
                "capability_status",
                return_value=linear.CapabilityStatus(
                    False, "cursor-mcp unavailable", "configure Cursor MCP"
                ),
            ),
            mock.patch.object(service, "_job_store") as store,
            mock.patch.object(service, "launch_worker") as launch,
        ):
            with self.assertRaisesRegex(
                HarnessError, "cursor-mcp unavailable.*configure Cursor MCP"
            ):
                service.start_job("work on API-42")

        store.return_value.create.assert_not_called()
        launch.assert_not_called()


class LinearCapabilityTests(unittest.TestCase):
    def test_cursor_mcp_and_linear_server_are_required(self) -> None:
        integration = linear.LinearIntegration()
        with mock.patch.object(linear.shutil, "which", return_value=None):
            self.assertFalse(integration.capability_status().available)

        process = subprocess.CompletedProcess([], 1, "", "unsupported")
        with (
            mock.patch.object(linear.shutil, "which", return_value="/usr/bin/agent"),
            mock.patch.object(linear, "_run_mcp_list", return_value=process),
        ):
            status = integration.capability_status()
        self.assertFalse(status.available)
        self.assertIn("cursor-mcp", status.detail)

        process = subprocess.CompletedProcess([], 0, "github: ready", "")
        with (
            mock.patch.object(linear.shutil, "which", return_value="/usr/bin/agent"),
            mock.patch.object(linear, "_run_mcp_list", return_value=process),
        ):
            status = integration.capability_status()
        self.assertFalse(status.available)
        self.assertIn("not configured", status.detail)

        for healthy in ("ready", "connected"):
            with self.subTest(status=healthy):
                process = subprocess.CompletedProcess([], 0, f"linear: {healthy}", "")
                with (
                    mock.patch.object(
                        linear.shutil, "which", return_value="/usr/bin/agent"
                    ),
                    mock.patch.object(linear, "_run_mcp_list", return_value=process),
                ):
                    self.assertTrue(integration.capability_status().available)

        unavailable = {
            "requires_authentication": "requires authentication",
            "disabled": "disabled",
            "disconnected": "disconnected",
            "Error: Connection failed": "unavailable",
        }
        for server_status, expected in unavailable.items():
            with self.subTest(status=server_status):
                process = subprocess.CompletedProcess(
                    [], 0, f"linear: {server_status}", ""
                )
                with (
                    mock.patch.object(
                        linear.shutil, "which", return_value="/usr/bin/agent"
                    ),
                    mock.patch.object(linear, "_run_mcp_list", return_value=process),
                ):
                    status = integration.capability_status()
                self.assertFalse(status.available)
                self.assertIn(expected, status.detail)


class LinearRoutingTests(unittest.TestCase):
    def test_enabled_connector_owns_router_prompt(self) -> None:
        repository = Path("/repos/api")
        router = mock.Mock(target="router")
        outcome = mock.Mock(
            output=(
                "ROUTE_REPO[token]: api\n"
                "ROUTE_CONFIDENCE[token]: high\n"
                "ROUTE_REASON[token]: matching service"
            )
        )
        client = mock.Mock()
        client.ensure_router.return_value = router
        client.prompt_and_wait.return_value = outcome
        client.resolve_repository.return_value = (repository, [repository])

        routed = registry.route_issue_repository(
            client,
            "API-42",
            [repository],
            token="token",
            reserved=set(),
            integrations=ENABLED,
        )

        self.assertEqual(routed, (repository, "high", "matching service"))
        prompt = client.prompt_and_wait.call_args.args[1]
        self.assertIn("Linear issue API-42", prompt)
        self.assertIn("untrusted external data", prompt)

    def test_disabled_connector_does_not_start_router(self) -> None:
        client = mock.Mock()
        routed = registry.route_issue_repository(
            client,
            "API-42",
            [],
            token="token",
            reserved=set(),
            integrations=DISABLED,
        )
        self.assertIsNone(routed)
        client.ensure_router.assert_not_called()


if __name__ == "__main__":
    unittest.main()
