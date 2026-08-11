from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness.integrations import herdr, linear


class HerdrIntegrationTests(unittest.TestCase):
    def test_normalizes_spoken_linear_issue_key(self) -> None:
        self.assertEqual(linear.extract_linear_issue("work on API 77"), "API-77")
        self.assertEqual(linear.extract_linear_issue("work on api - 78"), "API-78")
        self.assertEqual(linear.extract_linear_issue("work on API-79"), "API-79")

    def test_extracts_latest_multiline_marker(self) -> None:
        output = """
VOICE_SUMMARY[token]: old
more old text
VOICE_QUESTION[token]: Which repository
should I use?
"""
        self.assertEqual(
            herdr.extract_marker(output, "VOICE_QUESTION", "token"),
            "Which repository should I use?",
        )

    def test_ignores_prompt_marker_placeholders(self) -> None:
        output = """
VOICE_QUESTION[token]: <one concise question>
or:
WORKFLOW_PLAN[token]: <bounded multiline implementation plan>
"""
        self.assertIsNone(herdr.extract_marker(output, "VOICE_QUESTION", "token"))
        self.assertIsNone(herdr.extract_marker(output, "WORKFLOW_PLAN", "token"))

    def test_repository_resolution_requires_unique_valid_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            alpha = root / "alpha-app"
            alphabet = root / "alphabet"
            alpha.mkdir()
            alphabet.mkdir()
            client = herdr.HerdrClient("herdr")
            repository, matches = client.resolve_repository(
                "alpha", "task", [alpha, alphabet]
            )
            self.assertIsNone(repository)
            self.assertEqual(matches, [alpha, alphabet])
            repository, matches = client.resolve_repository(
                "alpha-app", "task", [alpha, alphabet]
            )
            self.assertEqual(repository, alpha)
            self.assertEqual(matches, [alpha])

    def test_live_agents_accepts_detected_idle_cursor_without_readiness(self) -> None:
        client = herdr.HerdrClient("herdr")
        agents = [
            {
                "agent": "cursor",
                "agent_status": "idle",
                "cwd": "/repo",
                "pane_id": "w1:p1",
            },
            {
                "agent": "cursor",
                "agent_status": "idle",
                "cwd": "/repo",
                "interactive_ready": False,
                "pane_id": "w1:p2",
            },
        ]
        with mock.patch.object(client, "list_agents", return_value=agents):
            live = client.live_agents()
        self.assertEqual([agent["pane_id"] for agent in live], ["w1:p1"])

    def test_router_uses_stable_single_owner_name(self) -> None:
        client = herdr.HerdrClient("herdr")
        selection = mock.Mock()
        with (
            mock.patch.object(client, "find_agent", return_value=None),
            mock.patch.object(client, "list_workspaces", return_value=[]),
            mock.patch.object(client, "new_pane", return_value=("pane", "workspace")),
            mock.patch.object(
                client, "start_agent", return_value=selection
            ) as start_agent,
        ):
            self.assertIs(client.ensure_router(set()), selection)

        start_agent.assert_called_once_with(
            herdr.HOME_ROOT,
            "router",
            "pane",
            "workspace",
            name="voice-router",
            checkpoint=None,
        )

    def test_existing_detected_agent_avoids_new_tab(self) -> None:
        client = herdr.HerdrClient("herdr")
        repository = Path("/repo")
        checkout = Path("/checkout")
        agent = {
            "agent": "cursor",
            "agent_status": "idle",
            "cwd": str(checkout),
            "pane_id": "w1:p1",
            "workspace_id": "w1",
        }
        listing = {
            "worktrees": [
                {
                    "branch": "voice/api-98",
                    "path": str(checkout),
                    "open_workspace_id": "w1",
                }
            ]
        }
        with (
            mock.patch.object(client, "run_json", return_value=listing),
            mock.patch.object(client, "list_agents", return_value=[agent]),
            mock.patch.object(client, "new_pane") as new_pane,
            mock.patch.object(client, "start_agent") as start_agent,
        ):
            selection = client.ensure_agent(
                repository,
                issue_key="API-98",
                agent_hint=None,
                reserved=set(),
            )
        self.assertEqual(selection.target, "w1:p1")
        new_pane.assert_not_called()
        start_agent.assert_not_called()

    def test_extracts_safe_repository_name_from_git_url(self) -> None:
        self.assertEqual(
            herdr.repository_name_from_url(
                "https://github.com/example/example-project.git"
            ),
            "example-project",
        )
        self.assertEqual(
            herdr.repository_name_from_url("git@github.com:example/project.git"),
            "project",
        )
        self.assertIsNone(herdr.repository_name_from_url("file:///tmp/project"))
        self.assertIsNone(
            herdr.repository_name_from_url("https://github.com/example/.hidden.git")
        )

    def test_confirmed_rofi_url_is_cloned_under_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = herdr.HerdrClient("herdr")

            def clone(
                command: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                destination = Path(command[-1])
                destination.mkdir()
                (destination / ".git").mkdir()
                return subprocess.CompletedProcess(command, 0, "", "")

            with (
                mock.patch(
                    "local_voice_harness.integrations.herdr.repository.HOME_ROOT",
                    root,
                ),
                mock.patch(
                    "local_voice_harness.integrations.herdr.types.HOME_ROOT",
                    root,
                ),
                mock.patch(
                    "local_voice_harness.integrations.herdr.repository.choose_repository",
                    return_value="https://github.com/example/project.git",
                ),
                mock.patch(
                    "local_voice_harness.integrations.herdr.repository.confirm_clone",
                    return_value=True,
                ),
                mock.patch(
                    "local_voice_harness.local_git.run_command",
                    side_effect=clone,
                ) as run,
            ):
                repository, reason = client.choose_or_clone_repository([])

            self.assertEqual(repository, root / "project")
            self.assertEqual(reason, "")
            command = run.call_args.args[0]
            self.assertEqual(
                command[:4],
                ["git", "clone", "--", "https://github.com/example/project.git"],
            )
            self.assertTrue(Path(command[-1]).name.startswith(".project.clone-"))

    def test_clone_timeout_reports_ambiguous_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = herdr.HerdrClient("herdr")
            with (
                mock.patch(
                    "local_voice_harness.integrations.herdr.repository.HOME_ROOT",
                    root,
                ),
                mock.patch(
                    "local_voice_harness.local_git.run_command",
                    side_effect=subprocess.TimeoutExpired(["git", "clone"], 300),
                ),
                self.assertRaisesRegex(
                    herdr.HerdrError,
                    "timed out; the clone outcome is ambiguous",
                ) as raised,
            ):
                client.repository.clone_repository(
                    "https://github.com/example/project.git"
                )

            self.assertEqual(raised.exception.code, "repository_clone_ambiguous")

    def test_cancelled_rofi_clone_returns_spoken_fallback_reason(self) -> None:
        client = herdr.HerdrClient("herdr")
        with (
            mock.patch(
                "local_voice_harness.integrations.herdr.repository.choose_repository",
                return_value="https://github.com/example/project.git",
            ),
            mock.patch(
                "local_voice_harness.integrations.herdr.repository.confirm_clone",
                return_value=False,
            ),
        ):
            repository, reason = client.choose_or_clone_repository([])

        self.assertIsNone(repository)
        self.assertEqual(reason, "Repository cloning was cancelled.")

    def test_server_start_falls_back_to_transient_unit(self) -> None:
        client = herdr.HerdrClient("herdr")
        states = iter([False, False, True])
        completed = subprocess.CompletedProcess([], 0, "", "")
        with (
            mock.patch.object(
                client.transport, "is_running", side_effect=lambda: next(states)
            ),
            mock.patch("subprocess.run", return_value=completed) as run,
            mock.patch("time.sleep"),
        ):
            client.ensure_server(timeout=1)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertTrue(any(command[0] == "systemd-run" for command in commands))

    def test_agent_name_stays_within_herdr_limit_for_long_label(self) -> None:
        client = herdr.HerdrClient("herdr")
        agent = {
            "name": "voice-issue-22-aaaaaaaaaa",
            "pane_id": "pane",
            "workspace_id": "workspace",
            "cwd": "/tmp/worktree",
        }
        with (
            mock.patch.object(client, "run_json", return_value={"agent": agent}) as run,
            mock.patch.object(
                herdr.workspace.uuid,
                "uuid4",
                return_value=mock.Mock(hex="a" * 32),
            ),
        ):
            client.start_agent(
                Path("/tmp/worktree"),
                "github-joshua-mo-143-local-voice-harness-22",
                "pane",
                "workspace",
            )

        name = run.call_args.args[2]
        self.assertLessEqual(len(name), 32)
        self.assertRegex(name, r"^[a-z][a-z0-9_-]{0,31}$")
        self.assertTrue(name.endswith("-aaaaaaaaaa"))

    def test_start_agent_passes_explicit_cursor_read_only_mode(self) -> None:
        client = herdr.HerdrClient("herdr")
        agent = {
            "name": "reviewer",
            "pane_id": "pane",
            "workspace_id": "workspace",
            "cwd": "/tmp/worktree",
        }
        with mock.patch.object(
            client, "run_json", return_value={"agent": agent}
        ) as run:
            client.start_agent(
                Path("/tmp/worktree"),
                "review",
                "pane",
                "workspace",
                name="reviewer",
                mode="ask",
            )

        self.assertEqual(
            run.call_args.args[-4:],
            ("--", "--trust", "--mode", "ask"),
        )

    def test_non_linear_task_creates_requested_worktree(self) -> None:
        client = herdr.HerdrClient("herdr")
        selection = herdr.AgentSelection(
            "agent", "pane", "workspace", "/tmp/worktree", "agent", "/tmp/worktree"
        )
        with (
            mock.patch.object(
                client,
                "run_json",
                side_effect=[
                    {"worktrees": []},
                    {
                        "worktree": {"path": "/tmp/worktree"},
                        "workspace": {"workspace_id": "workspace"},
                        "root_pane": {"pane_id": "pane"},
                    },
                ],
            ) as run_json,
            mock.patch.object(client, "find_agent", return_value=None),
            mock.patch.object(client, "workspace_for", return_value=None),
            mock.patch.object(
                client,
                "planned_worktree_path",
                return_value=Path("/tmp/worktree"),
            ),
            mock.patch.object(
                client, "start_agent", return_value=selection
            ) as start_agent,
        ):
            result = client.ensure_agent(
                Path("/tmp/repository"),
                issue_key=None,
                agent_hint=None,
                reserved=set(),
                worktree_branch="voice/github-123456",
                worktree_label="github-123456",
            )

        self.assertEqual(result, selection)
        self.assertEqual(
            run_json.call_args_list[1].args[:9],
            (
                "worktree",
                "create",
                "--cwd",
                "/tmp/repository",
                "--branch",
                "voice/github-123456",
                "--path",
                "/tmp/worktree",
                "--label",
            ),
        )
        start_agent.assert_called_once_with(
            Path("/tmp/worktree"),
            "github-123456",
            "pane",
            "workspace",
            name=mock.ANY,
            mode=None,
            checkpoint=None,
        )

    def test_worktree_and_agent_are_reserved_before_dispatch(self) -> None:
        client = herdr.HerdrClient("herdr")
        checkout = Path("/tmp/worktree")
        selection = herdr.AgentSelection(
            "planned-agent",
            "pane",
            "workspace",
            str(checkout),
            "planned-agent",
            str(checkout),
        )
        events: list[str] = []

        def reserve_worktree(
            _repository: Path, _branch: str, _checkout: Path, state: str
        ) -> None:
            events.append(f"worktree:{state}")

        def settle_worktree(
            _checkout: Path, _workspace: str | None, _pane: str | None
        ) -> None:
            events.append("worktree:settled")

        def reserve_agent(_selection: herdr.AgentSelection, dispatching: bool) -> None:
            events.append(f"agent:reserved:{dispatching}")

        def start_agent(*_args: object, **_kwargs: object) -> herdr.AgentSelection:
            events.append("agent:started")
            return selection

        with (
            mock.patch.object(
                client,
                "run_json",
                side_effect=[
                    {"worktrees": []},
                    {
                        "worktree": {"path": str(checkout)},
                        "workspace": {"workspace_id": "workspace"},
                        "root_pane": {"pane_id": "pane"},
                    },
                ],
            ),
            mock.patch.object(client, "find_agent", return_value=None),
            mock.patch.object(client, "workspace_for", return_value=None),
            mock.patch.object(client, "planned_worktree_path", return_value=checkout),
            mock.patch.object(client, "start_agent", side_effect=start_agent),
        ):
            client.ensure_agent(
                Path("/tmp/repository"),
                issue_key=None,
                agent_hint=None,
                reserved=set(),
                worktree_branch="voice/task",
                reserve=reserve_agent,
                settle=lambda _selection: events.append("agent:settled"),
                reserve_worktree=reserve_worktree,
                settle_worktree=settle_worktree,
            )

        self.assertEqual(
            events,
            [
                "worktree:planned",
                "worktree:dispatching",
                "worktree:settled",
                "agent:reserved:True",
                "agent:started",
                "agent:settled",
            ],
        )

    def test_worktree_post_checkpoint_prevents_pane_and_agent_creation(self) -> None:
        client = herdr.HerdrClient("herdr")
        checkpoints = 0

        def checkpoint() -> None:
            nonlocal checkpoints
            checkpoints += 1
            if checkpoints == 4:
                raise RuntimeError("cancelled after worktree creation")

        with (
            mock.patch.object(
                client,
                "run_json",
                side_effect=[
                    {"worktrees": []},
                    {
                        "worktree": {"path": "/tmp/worktree"},
                        "workspace": {"workspace_id": "workspace"},
                        "root_pane": {"pane_id": "pane"},
                    },
                ],
            ),
            mock.patch.object(client, "find_agent") as find_agent,
            mock.patch.object(client, "new_pane") as new_pane,
            mock.patch.object(client, "start_agent") as start_agent,
            mock.patch.object(
                client,
                "planned_worktree_path",
                return_value=Path("/tmp/worktree"),
            ),
            self.assertRaisesRegex(RuntimeError, "after worktree creation"),
        ):
            client.ensure_agent(
                Path("/tmp/repository"),
                issue_key=None,
                agent_hint=None,
                reserved=set(),
                worktree_branch="voice/github-123456",
                checkpoint=checkpoint,
            )

        find_agent.assert_not_called()
        new_pane.assert_not_called()
        start_agent.assert_not_called()

    def test_cancel_agent_waits_until_agent_is_settled(self) -> None:
        client = herdr.HerdrClient("herdr")
        with mock.patch.object(
            client,
            "run_json",
            side_effect=[{}, {"agent": {"agent_status": "idle"}}],
        ) as run_json:
            client.cancel_agent("agent")

        self.assertEqual(
            run_json.call_args_list,
            [
                mock.call("agent", "send-keys", "agent", "ctrl+c"),
                mock.call(
                    "agent",
                    "wait",
                    "agent",
                    "--timeout",
                    "5000",
                    timeout=10,
                ),
            ],
        )

    def test_completion_wait_survives_idle_then_working_resumption(self) -> None:
        client = herdr.HerdrClient("herdr")
        marker = "VOICE_SUMMARY[token]: finished"
        with (
            mock.patch.object(
                client,
                "get_agent",
                side_effect=[
                    {"agent_status": "idle", "state_change_seq": 1},
                    {"agent_status": "working", "state_change_seq": 2},
                    {"agent_status": "idle", "state_change_seq": 3},
                    {"agent_status": "idle", "state_change_seq": 3},
                ],
            ),
            mock.patch.object(client, "run_text", return_value=marker),
            mock.patch(
                "local_voice_harness.integrations.herdr.transport.time.monotonic",
                side_effect=[0, 0, 1, 2, 3],
            ),
            mock.patch("local_voice_harness.integrations.herdr.transport.time.sleep"),
        ):
            outcome = client.wait_for_stable_completion(
                "agent",
                token="token",
                inactivity_timeout=10,
                max_runtime=20,
                quiet_period=1,
            )

        self.assertEqual(outcome.status, "idle")
        self.assertEqual(outcome.summary, "finished")

    def test_active_plan_marker_returns_fenced_build_boundary(self) -> None:
        client = herdr.HerdrClient("herdr")
        agent = {
            "agent_status": "working",
            "state_change_seq": 7,
            "revision": 11,
            "agent_session": {"id": "session", "generation": 2},
            "interactive_ready": True,
        }
        with (
            mock.patch.object(client, "get_agent", return_value=agent),
            mock.patch.object(
                client,
                "run_text",
                return_value="WORKFLOW_PLAN[token]: implement safely",
            ),
            mock.patch(
                "local_voice_harness.integrations.herdr.transport.time.monotonic",
                side_effect=[0, 0, 1],
            ),
            mock.patch("local_voice_harness.integrations.herdr.transport.time.sleep"),
        ):
            outcome = client.wait_for_stable_completion(
                "agent",
                token="token",
                active_marker="WORKFLOW_PLAN",
                quiet_period=1,
            )

        self.assertEqual(outcome.status, "working")
        self.assertEqual(outcome.boundary_marker, "WORKFLOW_PLAN")
        self.assertEqual(outcome.state_change_sequence, 7)
        self.assertEqual(outcome.revision, 11)
        self.assertEqual(
            outcome.agent_session,
            '{"generation":2,"id":"session"}',
        )

    def test_active_plan_marker_never_bypasses_interactive_questionnaire(self) -> None:
        client = herdr.HerdrClient("herdr")
        with (
            mock.patch.object(
                client,
                "get_agent",
                return_value={
                    "agent_status": "working",
                    "state_change_seq": 7,
                    "agent_session": "session",
                    "interactive_ready": False,
                },
            ),
            mock.patch.object(
                client,
                "run_text",
                return_value="WORKFLOW_PLAN[token]: implement safely",
            ),
            mock.patch(
                "local_voice_harness.integrations.herdr.transport.time.monotonic",
                side_effect=[0, 0],
            ),
            self.assertRaises(herdr.HerdrError) as raised,
        ):
            client.wait_for_stable_completion(
                "agent",
                token="token",
                active_marker="WORKFLOW_PLAN",
            )

        self.assertEqual(raised.exception.code, "interactive_questionnaire")

    def test_expected_plan_boundary_can_be_captured_without_mode_switch(self) -> None:
        client = herdr.HerdrClient("herdr")
        agent = {
            "agent_status": "blocked",
            "state_change_seq": 7,
            "revision": 11,
            "agent_session": {"id": "session", "generation": 2},
            "interactive_ready": False,
        }
        with (
            mock.patch.object(client, "get_agent", return_value=agent),
            mock.patch.object(
                client,
                "run_text",
                return_value="WORKFLOW_PLAN[token]: implement safely",
            ),
            mock.patch(
                "local_voice_harness.integrations.herdr.transport.time.monotonic",
                side_effect=[0, 0, 1],
            ),
            mock.patch("local_voice_harness.integrations.herdr.transport.time.sleep"),
        ):
            outcome = client.wait_for_stable_completion(
                "agent",
                token="token",
                active_marker="WORKFLOW_PLAN",
                allow_interactive_plan_boundary=True,
                quiet_period=1,
            )

        self.assertEqual(outcome.boundary_marker, "WORKFLOW_PLAN")
        self.assertEqual(outcome.status, "blocked")
        self.assertEqual(outcome.state_change_sequence, 7)
        self.assertEqual(outcome.revision, 11)

    def test_generic_questionnaire_stays_blocked_at_expected_plan_boundary(
        self,
    ) -> None:
        client = herdr.HerdrClient("herdr")
        with (
            mock.patch.object(
                client,
                "get_agent",
                return_value={
                    "agent_status": "blocked",
                    "state_change_seq": 7,
                    "agent_session": "session",
                    "interactive_ready": False,
                },
            ),
            mock.patch.object(client, "run_text", return_value="Choose an option"),
            self.assertRaises(herdr.HerdrError) as raised,
        ):
            client.wait_for_stable_completion(
                "agent",
                token="token",
                active_marker="WORKFLOW_PLAN",
                allow_interactive_plan_boundary=True,
            )

        self.assertEqual(raised.exception.code, "interactive_questionnaire")

    def test_prompt_wait_detaches_from_expected_native_plan_boundary(self) -> None:
        client = herdr.HerdrClient("herdr")
        process = mock.Mock()
        process.poll.return_value = None
        process.communicate.return_value = ("", "")
        outcome = herdr.PromptOutcome(
            "blocked",
            None,
            None,
            "WORKFLOW_PLAN[token]: implement safely",
            boundary_marker="WORKFLOW_PLAN",
            agent_session="planner-session",
            state_change_sequence=2,
            revision=3,
        )
        client.get_agent = mock.Mock(
            side_effect=[
                {
                    "interactive_ready": True,
                    "agent_status": "idle",
                    "state_change_seq": 1,
                    "agent_session": "planner-session",
                },
                {
                    "interactive_ready": False,
                    "agent_status": "blocked",
                    "state_change_seq": 2,
                    "agent_session": "planner-session",
                },
                {
                    "interactive_ready": False,
                    "agent_status": "blocked",
                    "state_change_seq": 2,
                    "agent_session": "planner-session",
                },
            ]
        )
        client.run_text = mock.Mock(
            return_value="WORKFLOW_PLAN[token]: implement safely"
        )
        accepted = mock.Mock()
        with (
            mock.patch("subprocess.Popen", return_value=process),
            mock.patch("time.sleep"),
            mock.patch.object(
                client.session,
                "wait_for_stable_completion",
                return_value=outcome,
            ) as wait,
        ):
            captured = client.prompt_and_wait(
                "planner",
                "create a plan",
                token="token",
                baseline_sequence=1,
                expected_agent_session="planner-session",
                accepted=accepted,
                active_marker="WORKFLOW_PLAN",
                allow_interactive_plan_boundary=True,
            )

        self.assertEqual(captured, outcome)
        accepted.assert_called_once_with()
        process.kill.assert_called_once_with()
        wait.assert_called_once_with(
            "planner",
            token="token",
            inactivity_timeout=mock.ANY,
            max_runtime=mock.ANY,
            started_at=mock.ANY,
            checkpoint=None,
            expected_agent_session="planner-session",
            active_marker="WORKFLOW_PLAN",
            allow_interactive_plan_boundary=True,
        )

    def test_prompt_wait_timeout_continues_with_stable_observer(self) -> None:
        client = herdr.HerdrClient("herdr")
        expected = herdr.PromptOutcome(
            status="idle",
            summary="finished",
            question=None,
            output="VOICE_SUMMARY[token]: finished",
        )
        errors = (
            ("timeout", "timed out waiting for agent status"),
            ("operation_timeout", "still working"),
        )
        for code, message in errors:
            with self.subTest(code=code):
                process = mock.Mock()
                process.returncode = 1
                process.communicate.return_value = (
                    f'{{"error":{{"code":"{code}","message":"{message}"}}}}',
                    "",
                )
                with (
                    mock.patch.object(
                        client,
                        "get_agent",
                        side_effect=[
                            {"agent_status": "idle", "state_change_seq": 1},
                            {"agent_status": "working", "state_change_seq": 2},
                        ],
                    ),
                    mock.patch.object(
                        client.session,
                        "wait_for_stable_completion",
                        return_value=expected,
                    ) as wait_for_stable_completion,
                    mock.patch(
                        "local_voice_harness.integrations.herdr.transport.subprocess.Popen",
                        return_value=process,
                    ),
                    mock.patch(
                        "local_voice_harness.integrations.herdr.transport.time.sleep"
                    ),
                ):
                    outcome = client.prompt_and_wait("agent", "do work", token="token")

                self.assertEqual(outcome, expected)
                wait_for_stable_completion.assert_called_once()

    def test_successful_prompt_requires_sequence_proof_before_acceptance(self) -> None:
        client = herdr.HerdrClient("herdr")
        process = mock.Mock()
        process.returncode = 0
        process.poll.return_value = 0
        process.communicate.return_value = ('{"result":{}}', "")
        agent = {
            "agent_status": "blocked",
            "state_change_seq": 7,
            "agent_session": "session",
            "interactive_ready": True,
        }
        accepted = mock.Mock()
        with (
            mock.patch.object(client, "get_agent", side_effect=[agent, agent, agent]),
            mock.patch.object(client, "wait_for_stable_completion") as wait,
            mock.patch(
                "local_voice_harness.integrations.herdr.transport.subprocess.Popen",
                return_value=process,
            ),
            mock.patch("local_voice_harness.integrations.herdr.transport.time.sleep"),
            self.assertRaisesRegex(
                herdr.HerdrError, "did not accept the prompt"
            ) as raised,
        ):
            client.prompt_and_wait(
                "agent",
                "approve",
                token="token",
                accepted=accepted,
                expected_agent_session="session",
                allow_enter_fallback=False,
            )

        self.assertEqual(raised.exception.code, "agent_prompt_stalled")
        accepted.assert_not_called()
        wait.assert_not_called()

    def test_prompt_rejects_replaced_expected_agent_session(self) -> None:
        client = herdr.HerdrClient("herdr")
        with (
            mock.patch.object(
                client,
                "get_agent",
                return_value={
                    "agent_status": "blocked",
                    "state_change_seq": 7,
                    "agent_session": "replacement",
                    "interactive_ready": True,
                },
            ),
            self.assertRaisesRegex(
                herdr.HerdrError, "no longer has the expected session"
            ) as raised,
        ):
            client.prompt_and_wait(
                "agent",
                "approve",
                token="token",
                expected_agent_session="original",
            )

        self.assertEqual(raised.exception.code, "agent_session_changed")

    def test_completion_wait_times_out_after_inactivity(self) -> None:
        client = herdr.HerdrClient("herdr")
        with (
            mock.patch.object(
                client,
                "get_agent",
                return_value={"agent_status": "working", "state_change_seq": 1},
            ),
            mock.patch.object(client, "run_text", return_value="still working"),
            mock.patch(
                "local_voice_harness.integrations.herdr.transport.time.monotonic",
                side_effect=[0, 0, 1, 2],
            ),
            mock.patch("local_voice_harness.integrations.herdr.transport.time.sleep"),
            self.assertRaisesRegex(herdr.HerdrError, "inactivity timeout") as raised,
        ):
            client.wait_for_stable_completion(
                "agent",
                token="token",
                inactivity_timeout=2,
                max_runtime=20,
            )

        self.assertEqual(raised.exception.code, "agent_stalled")

    def test_completion_wait_enforces_maximum_runtime_despite_activity(self) -> None:
        client = herdr.HerdrClient("herdr")
        with (
            mock.patch.object(
                client,
                "get_agent",
                side_effect=[
                    {"agent_status": "working", "state_change_seq": 1},
                    {"agent_status": "working", "state_change_seq": 2},
                    {"agent_status": "working", "state_change_seq": 3},
                ],
            ),
            mock.patch.object(
                client,
                "run_text",
                side_effect=["one", "two", "three"],
            ),
            mock.patch(
                "local_voice_harness.integrations.herdr.transport.time.monotonic",
                side_effect=[0, 0, 1, 2],
            ),
            mock.patch("local_voice_harness.integrations.herdr.transport.time.sleep"),
            self.assertRaisesRegex(herdr.HerdrError, "maximum runtime") as raised,
        ):
            client.wait_for_stable_completion(
                "agent",
                token="token",
                inactivity_timeout=10,
                max_runtime=2,
            )

        self.assertEqual(raised.exception.code, "agent_stalled")

    def test_completion_wait_rejects_replaced_agent_session(self) -> None:
        client = herdr.HerdrClient("herdr")
        with (
            mock.patch.object(
                client,
                "get_agent",
                side_effect=[
                    {
                        "agent_status": "working",
                        "state_change_seq": 1,
                        "agent_session": "first",
                    },
                    {
                        "agent_status": "idle",
                        "state_change_seq": 2,
                        "agent_session": "second",
                    },
                ],
            ),
            mock.patch.object(client, "run_text", return_value="working"),
            mock.patch(
                "local_voice_harness.integrations.herdr.transport.time.monotonic",
                side_effect=[0, 0, 1],
            ),
            mock.patch("local_voice_harness.integrations.herdr.transport.time.sleep"),
            self.assertRaisesRegex(herdr.HerdrError, "changed sessions") as raised,
        ):
            client.wait_for_stable_completion("agent", token="token")

        self.assertEqual(raised.exception.code, "agent_session_changed")

    def test_cancel_agent_rejects_still_working_agent(self) -> None:
        client = herdr.HerdrClient("herdr")
        with (
            mock.patch.object(
                client,
                "run_json",
                side_effect=[{}, {"agent": {"agent_status": "working"}}],
            ),
            self.assertRaisesRegex(herdr.HerdrError, "did not stop"),
        ):
            client.cancel_agent("agent")


if __name__ == "__main__":
    unittest.main()
