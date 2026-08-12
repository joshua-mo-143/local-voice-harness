from __future__ import annotations

import multiprocessing
import subprocess
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import mock

from local_voice_harness.cursor import service
from local_voice_harness.errors import HarnessError
from local_voice_harness.integrations import linear, registry
from local_voice_harness.integrations.herdr import HerdrClient
from local_voice_harness.integrations.herdr.cursor_auth import cursor_project_id
from local_voice_harness.user_config import IntegrationSettings, default_user_config

DISABLED = IntegrationSettings(linear_enabled=False)
ENABLED = IntegrationSettings(linear_enabled=True)


def _hold_router_lock(path: str, entered: Any, release: Any) -> None:
    linear.LINEAR_ROUTER_LOCK = Path(path)
    with linear._router_owner():
        entered.set()
        release.wait(5)


def _observe_router_lock(path: str, acquired: Any) -> None:
    linear.LINEAR_ROUTER_LOCK = Path(path)
    with linear._router_owner():
        acquired.set()


def _authenticated_source(root: Path) -> tuple[Path, Path]:
    source = root / "authenticated"
    source.mkdir()
    projects = root / "projects"
    auth = projects / cursor_project_id(source.resolve()) / "mcp-auth.json"
    auth.parent.mkdir(parents=True)
    auth.write_text("{}")
    auth.chmod(0o600)
    return source, projects


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

    def test_issue_extraction_ignores_github_targets_with_linear_like_names(
        self,
    ) -> None:
        for target in (
            "https://github.com/joshua-mo-143/local-voice-harness/issues/229",
            "joshua-mo-143/local-voice-harness#229",
            "example/eng-42#7",
        ):
            with self.subTest(target=target):
                self.assertIsNone(registry.extract_issue_reference(target, ENABLED))

    def test_enabled_integration_recognizes_url_with_external_provenance(self) -> None:
        fragment = registry.capture_context(
            "https://linear.app/acme/issue/API-42/fix-it", ENABLED
        )
        self.assertIsNotNone(fragment)
        assert fragment is not None
        self.assertEqual(fragment.source, "linear")
        self.assertEqual(fragment.issue_reference, "API-42")
        self.assertIn("untrusted external identifier", fragment.text)

    def test_team_issue_list_exposes_scope_without_an_issue_reference(self) -> None:
        fragment = registry.capture_context(
            "https://linear.app/acme/team/eng/active", ENABLED
        )

        self.assertIsNotNone(fragment)
        assert fragment is not None
        self.assertEqual(fragment.source, "linear")
        self.assertEqual(fragment.issue_scope, "ENG")
        self.assertIsNone(fragment.issue_reference)
        self.assertIn("identifiers must come from the user's request", fragment.text)

    def test_disabled_explicit_issue_key_is_rejected_before_persistence(self) -> None:
        with (
            mock.patch.object(service, "_job_store") as store,
            mock.patch.object(service, "launch_worker"),
            self.assertRaisesRegex(HarnessError, "provider is unavailable"),
        ):
            service.start_job(
                "work on this ticket",
                issue_key="ENG-123",
                integrations=registry.build_integration_registry(
                    replace(default_user_config(), integrations=DISABLED)
                ),
            )

        store.return_value.create.assert_not_called()

    def test_missing_capability_rejects_job_before_persistence(self) -> None:
        with (
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
                service.start_job(
                    "work on API-42",
                    integrations=registry.build_integration_registry(
                        replace(default_user_config(), integrations=ENABLED)
                    ),
                )

        store.return_value.create.assert_not_called()
        launch.assert_not_called()


class LinearCapabilityTests(unittest.TestCase):
    def test_cursor_mcp_and_linear_server_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, projects = _authenticated_source(Path(temporary))
            integration = linear.LinearIntegration(
                cursor_mcp_auth_source=source,
                cursor_projects_root=projects,
            )
            with mock.patch.object(linear.shutil, "which", return_value=None):
                self.assertFalse(integration.capability_status().available)

            process = subprocess.CompletedProcess([], 1, "", "unsupported")
            with (
                mock.patch.object(
                    linear.shutil, "which", return_value="/usr/bin/agent"
                ),
                mock.patch.object(linear, "_run_mcp_list", return_value=process),
            ):
                status = integration.capability_status()
            self.assertFalse(status.available)
            self.assertIn("cursor-mcp", status.detail)

            process = subprocess.CompletedProcess([], 0, "github: ready", "")
            with (
                mock.patch.object(
                    linear.shutil, "which", return_value="/usr/bin/agent"
                ),
                mock.patch.object(linear, "_run_mcp_list", return_value=process),
            ):
                status = integration.capability_status()
            self.assertFalse(status.available)
            self.assertIn("not configured", status.detail)

            for healthy in ("ready", "connected"):
                with self.subTest(status=healthy):
                    process = subprocess.CompletedProcess(
                        [], 0, f"linear: {healthy}", ""
                    )
                    with (
                        mock.patch.object(
                            linear.shutil, "which", return_value="/usr/bin/agent"
                        ),
                        mock.patch.object(
                            linear, "_run_mcp_list", return_value=process
                        ),
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
                        mock.patch.object(
                            linear, "_run_mcp_list", return_value=process
                        ),
                    ):
                        status = integration.capability_status()
                    self.assertFalse(status.available)
                    self.assertIn(expected, status.detail)

    def test_authenticated_preflight_launches_agent_with_mcp_approval(self) -> None:
        process = subprocess.CompletedProcess([], 0, "linear: ready", "")
        spawned = {
            "name": "voice-router",
            "pane_id": "pane",
            "workspace_id": "workspace",
            "cwd": "/repositories",
            "agent_session": "session",
            "interactive_ready": True,
        }
        client = HerdrClient("herdr")

        with tempfile.TemporaryDirectory() as temporary:
            source, projects = _authenticated_source(Path(temporary))
            with (
                mock.patch.object(
                    linear.shutil, "which", return_value="/usr/bin/agent"
                ),
                mock.patch.object(linear, "_run_mcp_list", return_value=process),
                mock.patch.object(
                    client, "run_json", return_value={"agent": spawned}
                ) as run,
            ):
                self.assertTrue(
                    linear.LinearIntegration(
                        cursor_mcp_auth_source=source,
                        cursor_projects_root=projects,
                    )
                    .capability_status()
                    .available
                )
                client.start_agent(
                    Path("/repositories"),
                    "router",
                    "pane",
                    "workspace",
                    name="voice-router",
                )

        self.assertEqual(
            run.call_args.args[-3:],
            ("--", "--trust", "--approve-mcps"),
        )


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

        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(
                linear, "LINEAR_ROUTER_LOCK", Path(temporary) / "router.lock"
            ):
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

    def test_router_auth_failure_refuses_repository_fallback(self) -> None:
        client = mock.Mock()
        client.ensure_router.return_value = mock.Mock(target="router")
        client.prompt_and_wait.return_value = mock.Mock(
            output=(
                "JOS-17 was unavailable through Linear MCP; generic repository "
                "selected as fallback.\n"
                "ROUTE_REPO[token]: unrelated\n"
                "ROUTE_CONFIDENCE[token]: high\n"
                "ROUTE_REASON[token]: fallback"
            )
        )

        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(
                linear, "LINEAR_ROUTER_LOCK", Path(temporary) / "router.lock"
            ),
            self.assertRaisesRegex(
                HarnessError, "refusing unrelated repository fallback"
            ),
        ):
            registry.route_issue_repository(
                client,
                "API-42",
                [Path("/repos/api"), Path("/repos/unrelated")],
                token="token",
                reserved=set(),
                integrations=ENABLED,
            )

        client.resolve_repository.assert_not_called()
        client.choose_or_clone_repository.assert_not_called()

    def test_router_ownership_is_serialized_across_processes(self) -> None:
        context = multiprocessing.get_context("spawn")
        entered = context.Event()
        release = context.Event()
        acquired = context.Event()
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = str(Path(temporary) / "router.lock")
            owner = context.Process(
                target=_hold_router_lock,
                args=(lock_path, entered, release),
            )
            waiter = context.Process(
                target=_observe_router_lock,
                args=(lock_path, acquired),
            )
            owner.start()
            self.assertTrue(entered.wait(5))
            waiter.start()
            time.sleep(0.2)
            self.assertFalse(acquired.is_set())
            release.set()
            self.assertTrue(acquired.wait(5))
            owner.join(5)
            waiter.join(5)
            if owner.is_alive():
                owner.terminate()
            if waiter.is_alive():
                waiter.terminate()
            self.assertEqual(owner.exitcode, 0)
            self.assertEqual(waiter.exitcode, 0)


if __name__ == "__main__":
    unittest.main()
