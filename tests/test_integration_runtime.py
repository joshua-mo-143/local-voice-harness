from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from local_voice_harness.cursor import service, worker
from local_voice_harness.errors import HarnessError
from local_voice_harness.integrations.registry import (
    IntegrationRegistry,
    build_integration_registry,
    issue_provider_identity,
    resolve_issue_reference,
)
from local_voice_harness.user_config import default_user_config


class IntegrationRuntimeTests(unittest.TestCase):
    def test_cursor_turn_uses_registry_process_snapshot_for_runtime_timing(
        self,
    ) -> None:
        config = default_user_config(Path("/home/example"))
        snapshot = replace(
            config,
            platform=replace(config.platform, cursor_foreground_seconds=2.5),
        )
        registry = build_integration_registry(snapshot)
        with (
            mock.patch.object(service, "start_job", return_value="job-1") as start,
            mock.patch.object(
                service,
                "read_job",
                return_value=mock.Mock(participant_admission_state="held"),
            ),
            mock.patch.object(
                service,
                "_await_foreground",
                return_value=service.CursorTurnResult("started", None),
            ) as foreground,
        ):
            result = service.cursor_turn("implement it", integrations=registry)

        self.assertEqual(result.text, "started")
        self.assertEqual(start.call_args.kwargs["foreground_seconds"], 2.5)
        foreground.assert_called_once_with("job-1", None, timeout=2.5)

    def test_explicit_provider_identity_prevents_config_rerouting(self) -> None:
        class Provider:
            def __init__(self, name: str) -> None:
                self.name = name

            def owns_issue_reference(self, reference: str) -> bool:
                return reference.casefold().startswith("eng-")

            def canonicalize_issue_reference(self, reference: str) -> str:
                return reference.upper()

        config = default_user_config(Path("/home/example"))
        enabled = replace(config.integrations, linear_enabled=True)
        startup = IntegrationRegistry(
            enabled,
            (("linear_enabled", lambda: Provider("linear")),),
            mock.Mock(),
            mock.Mock(),
        )
        changed = IntegrationRegistry(
            enabled,
            (("linear_enabled", lambda: Provider("replacement")),),
            mock.Mock(),
            mock.Mock(),
        )

        canonical = resolve_issue_reference("eng-42", startup)
        self.assertEqual(canonical, "ENG-42")
        self.assertEqual(issue_provider_identity(canonical or "", startup), "linear")
        with self.assertRaisesRegex(HarnessError, "linear.*unavailable"):
            resolve_issue_reference(canonical, changed, provider="linear")

    def test_registry_retains_its_startup_snapshot(self) -> None:
        config = default_user_config(Path("/home/example"))
        startup = replace(
            config,
            platform=replace(
                config.platform,
                github_root=Path("/startup/github"),
                herdr_bin=Path("/startup/herdr"),
                git_bin=Path("/startup/git"),
            ),
        )
        restarted = replace(
            config,
            platform=replace(
                config.platform,
                github_root=Path("/restart/github"),
                herdr_bin=Path("/restart/herdr"),
                git_bin=Path("/restart/git"),
            ),
        )

        running_registry = build_integration_registry(startup)
        restarted_registry = build_integration_registry(restarted)

        self.assertEqual(
            running_registry.github_client().clone_root, Path("/startup/github")
        )
        self.assertEqual(
            running_registry.github_client().local_git.git_executable,
            "/startup/git",
        )
        self.assertEqual(running_registry.herdr_client().executable, "/startup/herdr")
        self.assertEqual(
            restarted_registry.github_client().clone_root, Path("/restart/github")
        )
        self.assertEqual(
            restarted_registry.github_client().local_git.git_executable,
            "/restart/git",
        )
        self.assertEqual(restarted_registry.herdr_client().executable, "/restart/herdr")

    def test_worker_builds_clients_from_one_loaded_snapshot(self) -> None:
        config = default_user_config(Path("/home/worker"))
        context = object()
        with (
            mock.patch.object(sys, "argv", ["worker", "job-1", "--claim", "token"]),
            mock.patch.object(
                worker, "load_user_config", return_value=config
            ) as load_config,
            mock.patch.object(worker, "run_worker") as run_worker,
            mock.patch.object(worker, "run_claimed_worker") as run_claimed,
        ):
            worker.main()
            runner = run_worker.call_args.args[3]
            runner(context)

        load_config.assert_called_once_with()
        factories = run_claimed.call_args.args[1]
        self.assertIs(factories.integrations.settings, config.integrations)
        self.assertEqual(
            factories.github().clone_root,
            config.platform.github_root,
        )
        self.assertEqual(
            factories.herdr().workspace.worktree_root,
            config.platform.herdr_worktree_root,
        )

    def test_recovery_and_release_use_injected_client_factories(self) -> None:
        registry = build_integration_registry(
            default_user_config(Path("/home/foreground"))
        )
        store = object()
        with (
            mock.patch.object(service, "_job_store", return_value=store),
            mock.patch.object(service, "_dispatch_waiting_jobs"),
            mock.patch.object(service.recovery, "recover_jobs") as recover,
            mock.patch.object(service.recovery, "cancel_target_and_release") as release,
        ):
            service.recover_jobs(integrations=registry)
            service._cancel_target_and_release(
                "job-1",
                "agent-1",
                "release-1",
                integrations=registry,
            )

        self.assertIs(recover.call_args.kwargs["herdr_factory"], registry.herdr_client)
        self.assertIs(
            recover.call_args.kwargs["github_factory"], registry.github_client
        )
        self.assertIs(release.call_args.kwargs["herdr_factory"], registry.herdr_client)


if __name__ == "__main__":
    unittest.main()
