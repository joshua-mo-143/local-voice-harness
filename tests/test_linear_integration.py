from __future__ import annotations

import json
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
from local_voice_harness.integrations.herdr import HerdrClient, HerdrError
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

    def test_ticket_creation_preflights_capability_before_persistence(self) -> None:
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
            self.assertRaisesRegex(
                HarnessError, "cursor-mcp unavailable.*configure Cursor MCP"
            ),
        ):
            service.start_job(
                "create a Linear ticket",
                linear_team="API",
                linear_ticket_create_requested=True,
                integrations=registry.build_integration_registry(
                    replace(default_user_config(), integrations=ENABLED)
                ),
            )

        store.return_value.create.assert_not_called()
        launch.assert_not_called()

    def test_ticket_creation_does_not_adopt_referenced_existing_ticket(self) -> None:
        with (
            mock.patch.object(
                linear.LinearIntegration,
                "capability_status",
                return_value=linear.CapabilityStatus(True, "ready"),
            ),
            mock.patch.object(service, "_job_store") as store,
            mock.patch.object(service, "_dispatch_waiting_jobs"),
        ):
            service.start_job(
                "create a Linear ticket in team API about API-42 failing",
                linear_team="API",
                linear_ticket_create_requested=True,
                integrations=registry.build_integration_registry(
                    replace(default_user_config(), integrations=ENABLED)
                ),
            )

        created = store.return_value.create.call_args.args[0]
        self.assertEqual(created.issue_provider, "linear")
        self.assertIsNone(created.issue_key)


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
    def test_enabled_connector_returns_advisory_evidence_without_authority(
        self,
    ) -> None:
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

        self.assertEqual(
            routed,
            (None, "high", "matching service Suggested repository: api."),
        )
        prompt = client.prompt_and_wait.call_args.args[1]
        self.assertIn("Linear issue API-42", prompt)
        self.assertIn("untrusted external data", prompt)
        self.assertIn("advisory and cannot authorize a checkout", prompt)
        client.resolve_repository.assert_not_called()

    def test_injected_repository_and_confidence_never_authorize_checkout(
        self,
    ) -> None:
        repositories = [Path("/repos/api"), Path("/repos/payments")]
        injected_channels = ("title", "description", "comment", "link")

        for channel in injected_channels:
            with (
                self.subTest(channel=channel),
                tempfile.TemporaryDirectory() as temporary,
            ):
                client = mock.Mock()
                client.ensure_router.return_value = mock.Mock(target="router")
                client.prompt_and_wait.return_value = mock.Mock(
                    output=(
                        "ROUTE_REPO[token]: payments\n"
                        "ROUTE_CONFIDENCE[token]: high\n"
                        f"ROUTE_REASON[token]: injected through Linear {channel}"
                    )
                )
                client.resolve_repository.return_value = (
                    repositories[1],
                    [repositories[1]],
                )
                with mock.patch.object(
                    linear, "LINEAR_ROUTER_LOCK", Path(temporary) / "router.lock"
                ):
                    routed = registry.route_issue_repository(
                        client,
                        "API-42",
                        repositories,
                        token="token",
                        reserved=set(),
                        integrations=ENABLED,
                    )

                self.assertIsNotNone(routed)
                assert routed is not None
                self.assertIsNone(routed[0])
                self.assertEqual(routed[1], "high")
                client.resolve_repository.assert_not_called()

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


class LinearTicketCreationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.integration = linear.LinearIntegration()
        self.plan = self.integration.plan_ticket_creation(
            "team-id-api",
            "api",
            "Fix startup",
            "The launcher fails after reboot.",
            correlation_marker="a" * 32,
        )

    def test_ticket_snapshot_fetches_identity_checked_content(self) -> None:
        client = mock.Mock()
        client.ensure_router.return_value = mock.Mock(target="router")

        def prompt(
            _target: str,
            _text: str,
            *,
            token: str,
            **_kwargs: object,
        ) -> mock.Mock:
            payload = {
                "identifier": "API-42",
                "id": "linear-issue-id",
                "title": "Expected title",
                "description": "Expected body",
                "url": "https://linear.app/acme/issue/API-42/expected-title",
                "updatedAt": "2026-08-15T10:00:00Z",
                "state": "In Progress",
            }
            return mock.Mock(
                output=f"VOICE_LINEAR_SNAPSHOT[{token}]: {json.dumps(payload)}"
            )

        client.prompt_and_wait.side_effect = prompt
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(
                linear, "LINEAR_ROUTER_LOCK", Path(temporary) / "router.lock"
            ),
            mock.patch.object(self.integration, "require_capabilities"),
        ):
            snapshot = self.integration.ticket_snapshot(client, "api-42")

        self.assertEqual(snapshot.identity, "API-42")
        self.assertEqual(snapshot.body, "Expected body")
        self.assertIn("read-only", client.prompt_and_wait.call_args.args[1])

    def test_ticket_snapshot_rejects_null_or_non_string_url(self) -> None:
        for invalid_url in (None, 42):
            with self.subTest(invalid_url=invalid_url):
                client = mock.Mock()
                client.ensure_router.return_value = mock.Mock(target="router")

                def prompt(
                    _target: str,
                    _text: str,
                    *,
                    token: str,
                    invalid_url: object = invalid_url,
                    **_kwargs: object,
                ) -> mock.Mock:
                    payload = {
                        "identifier": "API-42",
                        "id": "linear-issue-id",
                        "title": "Expected title",
                        "description": "Expected body",
                        "url": invalid_url,
                        "updatedAt": "2026-08-15T10:00:00Z",
                        "state": "In Progress",
                    }
                    return mock.Mock(
                        output=(
                            f"VOICE_LINEAR_SNAPSHOT[{token}]: {json.dumps(payload)}"
                        )
                    )

                client.prompt_and_wait.side_effect = prompt
                with (
                    tempfile.TemporaryDirectory() as temporary,
                    mock.patch.object(
                        linear,
                        "LINEAR_ROUTER_LOCK",
                        Path(temporary) / "router.lock",
                    ),
                    mock.patch.object(self.integration, "require_capabilities"),
                    self.assertRaisesRegex(
                        linear.LinearError,
                        "invalid ticket snapshot",
                    ),
                ):
                    self.integration.ticket_snapshot(client, "api-42")

    def test_submit_requires_confirmation_and_returns_validated_identity(self) -> None:
        client = mock.Mock()
        client.ensure_router.return_value = mock.Mock(target="voice-router")
        client.get_agent.return_value = {
            "agent_session": "router-session",
            "state_change_seq": 7,
        }

        def prompt(*_args: object, **kwargs: object) -> object:
            before_submit = kwargs["before_submit"]
            assert callable(before_submit)
            before_submit(7)
            accepted = kwargs["accepted"]
            assert callable(accepted)
            accepted()
            token = str(kwargs["token"])
            return mock.Mock(
                output=(
                    f"VOICE_LINEAR_IDENTIFIER[{token}]: API-42\n"
                    f"VOICE_LINEAR_URL[{token}]: "
                    "https://linear.app/acme/issue/API-42/fix-startup"
                )
            )

        client.prompt_and_wait.side_effect = prompt
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(
                linear,
                "LINEAR_ROUTER_LOCK",
                Path(temporary) / "router.lock",
            ),
            mock.patch.object(self.integration, "require_capabilities"),
        ):
            result = self.integration.submit_ticket_creation(
                client,
                self.plan,
                confirmed=True,
            )

        self.assertEqual(result.issue.identifier, "API-42")
        submitted = client.prompt_and_wait.call_args.args[1]
        self.assertIn("voice-harness-linear-ticket:" + "a" * 32, submitted)
        self.assertIn("Team key: API", submitted)
        self.assertIn("Immutable team ID: team-id-api", submitted)
        self.assertEqual(
            client.prompt_and_wait.call_args.kwargs["expected_agent_session"],
            "router-session",
        )

        with self.assertRaisesRegex(linear.LinearError, "explicit confirmation"):
            self.integration.submit_ticket_creation(
                client,
                self.plan,
                confirmed=False,
            )

    def test_accepted_prompt_failure_is_ambiguous(self) -> None:
        client = mock.Mock()
        client.ensure_router.return_value = mock.Mock(target="voice-router")
        client.get_agent.return_value = {
            "agent_session": "router-session",
            "state_change_seq": 7,
        }

        def prompt(*_args: object, **kwargs: object) -> object:
            before_submit = kwargs["before_submit"]
            assert callable(before_submit)
            before_submit(7)
            accepted = kwargs["accepted"]
            assert callable(accepted)
            accepted()
            raise HerdrError("timeout", code="operation_ambiguous")

        client.prompt_and_wait.side_effect = prompt
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(
                linear,
                "LINEAR_ROUTER_LOCK",
                Path(temporary) / "router.lock",
            ),
            mock.patch.object(self.integration, "require_capabilities"),
            self.assertRaises(linear.LinearOperationAmbiguous),
        ):
            self.integration.submit_ticket_creation(
                client,
                self.plan,
                confirmed=True,
            )

    def test_observe_uses_read_only_marker_search(self) -> None:
        client = mock.Mock()
        client.ensure_router.return_value = mock.Mock(target="voice-router")

        def prompt(*_args: object, **kwargs: object) -> object:
            token = str(kwargs["token"])
            return mock.Mock(
                output=(
                    f"VOICE_LINEAR_STATUS[{token}]: found\n"
                    f"VOICE_LINEAR_IDENTIFIER[{token}]: API-42\n"
                    f"VOICE_LINEAR_URL[{token}]: "
                    "https://linear.app/acme/issue/API-42/fix-startup"
                )
            )

        client.prompt_and_wait.side_effect = prompt
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(
                linear,
                "LINEAR_ROUTER_LOCK",
                Path(temporary) / "router.lock",
            ),
            mock.patch.object(self.integration, "require_capabilities"),
        ):
            result = self.integration.observe_ticket_creation(client, self.plan)

        assert result is not None
        self.assertEqual(result.issue.identifier, "API-42")
        prompt_text = client.prompt_and_wait.call_args.args[1]
        self.assertIn("read-only", prompt_text)
        self.assertIn("Do not create or modify anything", prompt_text)
        self.assertIn("list_issues", prompt_text)
        self.assertIn("<!-- voice-harness-linear-ticket:", prompt_text)
        self.assertNotIn("Search team ", prompt_text)

    def test_resolves_exact_team_identity_read_only(self) -> None:
        client = mock.Mock()
        client.ensure_router.return_value = mock.Mock(target="voice-router")

        def prompt(*_args: object, **kwargs: object) -> object:
            token = str(kwargs["token"])
            return mock.Mock(
                output=(
                    f"VOICE_LINEAR_STATUS[{token}]: found\n"
                    f"VOICE_LINEAR_TEAM_ID[{token}]: team-id-api\n"
                    f"VOICE_LINEAR_TEAM_KEY[{token}]: API\n"
                    f"VOICE_LINEAR_TEAM_NAME[{token}]: Application Platform"
                )
            )

        client.prompt_and_wait.side_effect = prompt
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(
                linear,
                "LINEAR_ROUTER_LOCK",
                Path(temporary) / "router.lock",
            ),
            mock.patch.object(self.integration, "require_capabilities"),
        ):
            team = self.integration.resolve_team(client, "api")

        self.assertEqual(team.id, "team-id-api")
        self.assertEqual(team.key, "API")
        self.assertIn("read-only", client.prompt_and_wait.call_args.args[1])

    def test_missing_team_is_reported_as_not_found(self) -> None:
        client = mock.Mock()
        client.ensure_router.return_value = mock.Mock(target="voice-router")

        def prompt(*_args: object, **kwargs: object) -> object:
            token = str(kwargs["token"])
            return mock.Mock(output=f"VOICE_LINEAR_STATUS[{token}]: not_found\n")

        client.prompt_and_wait.side_effect = prompt
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(
                linear,
                "LINEAR_ROUTER_LOCK",
                Path(temporary) / "router.lock",
            ),
            mock.patch.object(self.integration, "require_capabilities"),
        ):
            with self.assertRaises(linear.LinearError) as raised:
                self.integration.resolve_team(client, "api")

        self.assertEqual(str(raised.exception), "I couldn't find Linear team API.")


class LinearIssueResolveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.integration = linear.LinearIntegration()

    def _resolve(
        self,
        status: str,
        *,
        identifier: str = "ENG-1",
        url: str = "https://linear.app/acme/issue/ENG-1/title",
        herdr_error: HerdrError | None = None,
        raw_output: str | None = None,
        reference: str = "eng-1",
    ) -> linear.LinearIssue:
        client = mock.Mock()
        client.ensure_router.return_value = mock.Mock(target="voice-router")

        def prompt(*_args: object, **kwargs: object) -> object:
            if herdr_error is not None:
                raise herdr_error
            token = str(kwargs["token"])
            if raw_output is not None:
                return mock.Mock(output=raw_output)
            lines = [f"VOICE_LINEAR_STATUS[{token}]: {status}"]
            if identifier is not None:
                lines.append(f"VOICE_LINEAR_IDENTIFIER[{token}]: {identifier}")
            if url is not None:
                lines.append(f"VOICE_LINEAR_URL[{token}]: {url}")
            return mock.Mock(output="\n".join(lines))

        client.prompt_and_wait.side_effect = prompt
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(
                linear,
                "LINEAR_ROUTER_LOCK",
                Path(temporary) / "router.lock",
            ),
            mock.patch.object(self.integration, "require_capabilities"),
        ):
            return self.integration.resolve_issue(client, reference)

    def test_parse_rejects_malformed_and_nonpositive_keys(self) -> None:
        with self.assertRaises(linear.LinearIssueLookupError) as malformed:
            linear.parse_linear_issue_reference("ENG-")
        self.assertEqual(
            malformed.exception.reason,
            linear.LinearIssueLookupReason.MALFORMED,
        )
        with self.assertRaises(linear.LinearIssueLookupError) as nonpositive:
            linear.parse_linear_issue_reference("ENG-0")
        self.assertEqual(
            nonpositive.exception.reason,
            linear.LinearIssueLookupReason.NONPOSITIVE,
        )
        self.assertEqual(
            linear.parse_linear_issue_reference("eng-1").identifier, "ENG-1"
        )

    def test_resolves_canonical_identity_read_only(self) -> None:
        issue = self._resolve("found")

        self.assertEqual(issue.identifier, "ENG-1")

    def test_missing_issue_is_not_found(self) -> None:
        with self.assertRaises(linear.LinearIssueLookupError) as raised:
            self._resolve("not_found", identifier="", url="")

        self.assertEqual(
            raised.exception.reason,
            linear.LinearIssueLookupReason.NOT_FOUND_OR_INACCESSIBLE,
        )

    def test_inaccessible_issue_is_rejected(self) -> None:
        with self.assertRaises(linear.LinearIssueLookupError) as raised:
            self._resolve("inaccessible", identifier="", url="")

        self.assertEqual(
            raised.exception.reason,
            linear.LinearIssueLookupReason.NOT_FOUND_OR_INACCESSIBLE,
        )

    def test_unauthorized_issue_is_rejected(self) -> None:
        with self.assertRaises(linear.LinearIssueLookupError) as raised:
            self._resolve("unauthorized", identifier="", url="")

        self.assertEqual(
            raised.exception.reason,
            linear.LinearIssueLookupReason.UNAUTHORIZED,
        )

    def test_transient_provider_failure_fails_closed(self) -> None:
        with self.assertRaises(linear.LinearIssueLookupError) as raised:
            self._resolve(
                "found",
                herdr_error=HerdrError("timeout", code="timeout"),
            )

        self.assertEqual(
            raised.exception.reason,
            linear.LinearIssueLookupReason.TRANSIENT,
        )

    def test_thrown_authentication_failure_is_unauthorized(self) -> None:
        for error in (
            HerdrError(
                "Linear MCP requires authentication",
                code="mcp_authentication_required",
            ),
            HerdrError("MCP request rejected", code="permission_denied"),
            HerdrError("not authorized to call the Linear tool"),
        ):
            with self.subTest(error=error):
                with self.assertRaises(linear.LinearIssueLookupError) as raised:
                    self._resolve("found", herdr_error=error)

                self.assertEqual(
                    raised.exception.reason,
                    linear.LinearIssueLookupReason.UNAUTHORIZED,
                )

    def test_unstructured_mcp_authorization_failure_is_unauthorized(self) -> None:
        for output in (
            "Linear MCP server requires authentication before tools can be used.",
            "Tool call failed: permission denied.",
            "You are not authorized to access this Linear workspace.",
        ):
            with self.subTest(output=output):
                with self.assertRaises(linear.LinearIssueLookupError) as raised:
                    self._resolve("unknown", raw_output=output)

                self.assertEqual(
                    raised.exception.reason,
                    linear.LinearIssueLookupReason.UNAUTHORIZED,
                )

    def test_identity_mismatch_fails_closed(self) -> None:
        with self.assertRaises(linear.LinearIssueLookupError) as raised:
            self._resolve(
                "found",
                identifier="ENG-2",
                url="https://linear.app/acme/issue/ENG-2/other",
            )

        self.assertEqual(
            raised.exception.reason,
            linear.LinearIssueLookupReason.UNKNOWN,
        )

    def test_resolve_prompt_is_read_only(self) -> None:
        client = mock.Mock()
        client.ensure_router.return_value = mock.Mock(target="voice-router")

        def prompt(*_args: object, **kwargs: object) -> object:
            token = str(kwargs["token"])
            return mock.Mock(
                output=(
                    f"VOICE_LINEAR_STATUS[{token}]: found\n"
                    f"VOICE_LINEAR_IDENTIFIER[{token}]: ENG-1\n"
                    f"VOICE_LINEAR_URL[{token}]: "
                    "https://linear.app/acme/issue/ENG-1/title"
                )
            )

        client.prompt_and_wait.side_effect = prompt
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(
                linear,
                "LINEAR_ROUTER_LOCK",
                Path(temporary) / "router.lock",
            ),
            mock.patch.object(self.integration, "require_capabilities"),
        ):
            self.integration.resolve_issue(client, "ENG-1")

        prompt_text = client.prompt_and_wait.call_args.args[1]
        self.assertIn("read-only", prompt_text)
        self.assertIn("Do not create or modify anything", prompt_text)


class LinearTicketUpdateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.integration = linear.LinearIntegration()
        self.plan = self.integration.plan_ticket_update(
            "issue-id-api-79",
            "API-79",
            "Fix startup",
            "The launcher fails after reboot.",
            correlation_marker="a" * 32,
        )

    def test_resolves_one_configured_terminal_state_deterministically(self) -> None:
        client = mock.Mock()
        client.ensure_router.return_value = mock.Mock(target="voice-router")

        def prompt(*_args: object, **kwargs: object) -> object:
            token = str(kwargs["token"])
            payload = {
                "identifier": "API-79",
                "states": [
                    {"id": "state-canceled", "name": "Canceled", "type": "canceled"},
                    {"id": "state-released", "name": "Released", "type": "completed"},
                    {"id": "state-done", "name": "Done", "type": "completed"},
                ],
            }
            return mock.Mock(
                output=(f"VOICE_LINEAR_TERMINAL_STATES[{token}]: {json.dumps(payload)}")
            )

        client.prompt_and_wait.side_effect = prompt
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(
                linear,
                "LINEAR_ROUTER_LOCK",
                Path(temporary) / "router.lock",
            ),
            mock.patch.object(self.integration, "require_capabilities"),
        ):
            state = self.integration.resolve_terminal_state(client, "api-79")

        self.assertEqual(
            state, linear.LinearWorkflowState("state-done", "Done", "completed")
        )
        self.assertIn("read-only", client.prompt_and_wait.call_args.args[1])

    def test_submit_requires_confirmation_and_returns_validated_identity(self) -> None:
        client = mock.Mock()
        client.ensure_router.return_value = mock.Mock(target="voice-router")
        client.get_agent.return_value = {
            "agent_session": "router-session",
            "state_change_seq": 7,
        }

        def prompt(*_args: object, **kwargs: object) -> object:
            before_submit = kwargs["before_submit"]
            assert callable(before_submit)
            before_submit(7)
            accepted = kwargs["accepted"]
            assert callable(accepted)
            accepted()
            token = str(kwargs["token"])
            return mock.Mock(
                output=(
                    f"VOICE_LINEAR_IDENTIFIER[{token}]: API-79\n"
                    f"VOICE_LINEAR_URL[{token}]: "
                    "https://linear.app/acme/issue/API-79/fix-startup"
                )
            )

        client.prompt_and_wait.side_effect = prompt
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(
                linear,
                "LINEAR_ROUTER_LOCK",
                Path(temporary) / "router.lock",
            ),
            mock.patch.object(self.integration, "require_capabilities"),
        ):
            result = self.integration.submit_ticket_update(
                client,
                self.plan,
                confirmed=True,
            )

        self.assertEqual(result.issue.identifier, "API-79")
        submitted = client.prompt_and_wait.call_args.args[1]
        self.assertIn("Identifier: API-79", submitted)
        self.assertIn("Immutable issue ID: issue-id-api-79", submitted)
        self.assertIn("Update title: yes", submitted)
        self.assertIn("Update description: yes", submitted)
        self.assertIn("Do not create a new issue", submitted)
        self.assertEqual(
            client.prompt_and_wait.call_args.kwargs["expected_agent_session"],
            "router-session",
        )

        with self.assertRaisesRegex(linear.LinearError, "explicit confirmation"):
            self.integration.submit_ticket_update(
                client,
                self.plan,
                confirmed=False,
            )

    def test_accepted_prompt_failure_is_ambiguous(self) -> None:
        client = mock.Mock()
        client.ensure_router.return_value = mock.Mock(target="voice-router")
        client.get_agent.return_value = {
            "agent_session": "router-session",
            "state_change_seq": 7,
        }

        def prompt(*_args: object, **kwargs: object) -> object:
            before_submit = kwargs["before_submit"]
            assert callable(before_submit)
            before_submit(7)
            accepted = kwargs["accepted"]
            assert callable(accepted)
            accepted()
            raise HerdrError("timeout", code="operation_ambiguous")

        client.prompt_and_wait.side_effect = prompt
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(
                linear,
                "LINEAR_ROUTER_LOCK",
                Path(temporary) / "router.lock",
            ),
            mock.patch.object(self.integration, "require_capabilities"),
            self.assertRaises(linear.LinearOperationAmbiguous),
        ):
            self.integration.submit_ticket_update(
                client,
                self.plan,
                confirmed=True,
            )

    def test_observe_uses_read_only_exact_snapshot(self) -> None:
        client = mock.Mock()
        client.ensure_router.return_value = mock.Mock(target="voice-router")

        def prompt(*_args: object, **kwargs: object) -> object:
            token = str(kwargs["token"])
            return mock.Mock(
                output=(
                    f"VOICE_LINEAR_STATUS[{token}]: found\n"
                    f"VOICE_LINEAR_IDENTIFIER[{token}]: API-79\n"
                    f"VOICE_LINEAR_URL[{token}]: "
                    "https://linear.app/acme/issue/API-79/fix-startup"
                )
            )

        client.prompt_and_wait.side_effect = prompt
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(
                linear,
                "LINEAR_ROUTER_LOCK",
                Path(temporary) / "router.lock",
            ),
            mock.patch.object(self.integration, "require_capabilities"),
        ):
            result = self.integration.observe_ticket_update(client, self.plan)

        assert result is not None
        self.assertEqual(result.issue.identifier, "API-79")
        prompt_text = client.prompt_and_wait.call_args.args[1]
        self.assertIn("read-only", prompt_text)
        self.assertIn("Do not create or modify anything", prompt_text)
        self.assertIn("title and description exactly equal", prompt_text)

    def test_ticket_update_preflights_capability_before_persistence(self) -> None:
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
            self.assertRaisesRegex(
                HarnessError, "cursor-mcp unavailable.*configure Cursor MCP"
            ),
        ):
            service.start_job(
                "update Linear ticket API-79",
                issue_key="API-79",
                linear_ticket_update_requested=True,
                integrations=registry.build_integration_registry(
                    replace(default_user_config(), integrations=ENABLED)
                ),
            )

        store.return_value.create.assert_not_called()
        launch.assert_not_called()


class LinearTicketCloseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.integration = linear.LinearIntegration()
        self.plan = self.integration.plan_ticket_close(
            "issue-id-api-79",
            "API-79",
            "state-done",
            "Done",
            correlation_marker="a" * 32,
        )

    def test_submit_requires_confirmation_and_returns_validated_identity(self) -> None:
        client = mock.Mock()
        client.ensure_router.return_value = mock.Mock(target="voice-router")
        client.get_agent.return_value = {
            "agent_session": "router-session",
            "state_change_seq": 7,
        }

        def prompt(*_args: object, **kwargs: object) -> object:
            before_submit = kwargs["before_submit"]
            assert callable(before_submit)
            before_submit(7)
            accepted = kwargs["accepted"]
            assert callable(accepted)
            accepted()
            token = str(kwargs["token"])
            return mock.Mock(
                output=(
                    f"VOICE_LINEAR_IDENTIFIER[{token}]: API-79\n"
                    f"VOICE_LINEAR_URL[{token}]: "
                    "https://linear.app/acme/issue/API-79/fix-startup"
                )
            )

        client.prompt_and_wait.side_effect = prompt
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(
                linear,
                "LINEAR_ROUTER_LOCK",
                Path(temporary) / "router.lock",
            ),
            mock.patch.object(self.integration, "require_capabilities"),
        ):
            result = self.integration.submit_ticket_close(
                client,
                self.plan,
                confirmed=True,
            )

        self.assertEqual(result.issue.identifier, "API-79")
        submitted = client.prompt_and_wait.call_args.args[1]
        self.assertIn("voice-harness-linear-ticket:" + "a" * 32, submitted)
        self.assertIn("Identifier: API-79", submitted)
        self.assertIn("Immutable issue ID: issue-id-api-79", submitted)
        self.assertIn("configured workflow state named Done", submitted)
        self.assertIn("immutable state ID is state-done", submitted)
        self.assertIn("Do not create a new issue", submitted)
        self.assertEqual(
            client.prompt_and_wait.call_args.kwargs["expected_agent_session"],
            "router-session",
        )

        with self.assertRaisesRegex(linear.LinearError, "explicit confirmation"):
            self.integration.submit_ticket_close(
                client,
                self.plan,
                confirmed=False,
            )

    def test_accepted_prompt_failure_is_ambiguous(self) -> None:
        client = mock.Mock()
        client.ensure_router.return_value = mock.Mock(target="voice-router")
        client.get_agent.return_value = {
            "agent_session": "router-session",
            "state_change_seq": 7,
        }

        def prompt(*_args: object, **kwargs: object) -> object:
            before_submit = kwargs["before_submit"]
            assert callable(before_submit)
            before_submit(7)
            accepted = kwargs["accepted"]
            assert callable(accepted)
            accepted()
            raise HerdrError("timeout", code="operation_ambiguous")

        client.prompt_and_wait.side_effect = prompt
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(
                linear,
                "LINEAR_ROUTER_LOCK",
                Path(temporary) / "router.lock",
            ),
            mock.patch.object(self.integration, "require_capabilities"),
            self.assertRaises(linear.LinearOperationAmbiguous),
        ):
            self.integration.submit_ticket_close(
                client,
                self.plan,
                confirmed=True,
            )

    def test_observe_uses_read_only_marker_search(self) -> None:
        client = mock.Mock()
        client.ensure_router.return_value = mock.Mock(target="voice-router")

        def prompt(*_args: object, **kwargs: object) -> object:
            token = str(kwargs["token"])
            return mock.Mock(
                output=(
                    f"VOICE_LINEAR_STATUS[{token}]: found\n"
                    f"VOICE_LINEAR_IDENTIFIER[{token}]: API-79\n"
                    f"VOICE_LINEAR_URL[{token}]: "
                    "https://linear.app/acme/issue/API-79/fix-startup"
                )
            )

        client.prompt_and_wait.side_effect = prompt
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(
                linear,
                "LINEAR_ROUTER_LOCK",
                Path(temporary) / "router.lock",
            ),
            mock.patch.object(self.integration, "require_capabilities"),
        ):
            result = self.integration.observe_ticket_close(client, self.plan)

        assert result is not None
        self.assertEqual(result.issue.identifier, "API-79")
        prompt_text = client.prompt_and_wait.call_args.args[1]
        self.assertIn("read-only", prompt_text)
        self.assertIn("Do not create or modify anything", prompt_text)
        self.assertIn("<!-- voice-harness-linear-ticket:", prompt_text)

    def test_ticket_close_preflights_capability_before_persistence(self) -> None:
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
            self.assertRaisesRegex(
                HarnessError, "cursor-mcp unavailable.*configure Cursor MCP"
            ),
        ):
            service.start_job(
                "close Linear ticket API-79",
                issue_key="API-79",
                linear_ticket_close_requested=True,
                integrations=registry.build_integration_registry(
                    replace(default_user_config(), integrations=ENABLED)
                ),
            )

        store.return_value.create.assert_not_called()
        launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
