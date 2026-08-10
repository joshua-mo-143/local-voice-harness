from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness import cli
from local_voice_harness.cursor.service import CursorTurnRequest, CursorTurnResult
from local_voice_harness.cursor.store import QuarantineEvidence
from local_voice_harness.user_config import PlanApprovalMode, PlanApprovalPreferences


class JobsCliTests(unittest.TestCase):
    def _dispatch(self, argv: list[str]) -> mock.Mock:
        args = cli.parser().parse_args(argv)
        with mock.patch.object(
            cli, "cursor_turn", return_value=CursorTurnResult("ok", None)
        ) as cursor_turn:
            cli.dispatch(args)
        return cursor_turn

    def test_list_maps_to_list_action(self) -> None:
        cursor_turn = self._dispatch(["jobs", "list"])
        cursor_turn.assert_called_once_with(
            CursorTurnRequest("", action="list"),
        )

    def test_status_without_reference_summarizes(self) -> None:
        cursor_turn = self._dispatch(["jobs", "status"])
        cursor_turn.assert_called_once_with(
            CursorTurnRequest("", action="status", reference=""),
        )

    def test_cancel_joins_reference_words(self) -> None:
        cursor_turn = self._dispatch(["jobs", "cancel", "the", "venice", "fix"])
        cursor_turn.assert_called_once_with(
            CursorTurnRequest(
                "the venice fix", action="cancel", reference="the venice fix"
            ),
        )

    def test_dismiss_and_repeat_map_to_actions(self) -> None:
        for action in ("dismiss", "repeat"):
            with self.subTest(action=action):
                cursor_turn = self._dispatch(["jobs", action, "bug", "fix"])
                cursor_turn.assert_called_once_with(
                    CursorTurnRequest("bug fix", action=action, reference="bug fix"),
                )

    def test_reply_targets_explicit_job(self) -> None:
        cursor_turn = self._dispatch(
            ["jobs", "reply", "--job", "aaaaaaaaaaaa", "yes", "please"]
        )
        cursor_turn.assert_called_once_with(
            CursorTurnRequest(
                "yes please",
                action="reply",
                job_id="aaaaaaaaaaaa",
                reference="yes please",
            ),
        )

    def test_reply_without_job_resolves_by_reference(self) -> None:
        cursor_turn = self._dispatch(["jobs", "reply", "use", "the", "api", "repo"])
        cursor_turn.assert_called_once_with(
            CursorTurnRequest(
                "use the api repo",
                action="reply",
                job_id=None,
                reference="use the api repo",
            ),
        )


class PlanApprovalCliTests(unittest.TestCase):
    def test_status_reports_mode_and_explicit_count(self) -> None:
        args = cli.parser().parse_args(["plan-approval", "status"])
        preferences = PlanApprovalPreferences(
            mode=PlanApprovalMode.AUTO,
            explicit_approval_ids=("one", "two", "three"),
            offer_completed=True,
        )
        with (
            mock.patch.object(
                cli,
                "load_plan_approval_preferences",
                return_value=preferences,
            ),
            mock.patch("builtins.print") as output,
        ):
            cli.dispatch(args)

        self.assertIn("auto (3/3", output.call_args.args[0])

    def test_ask_disables_automatic_approval(self) -> None:
        args = cli.parser().parse_args(["plan-approval", "ask"])
        preferences = PlanApprovalPreferences(mode=PlanApprovalMode.ASK)
        with (
            mock.patch.object(
                cli,
                "set_plan_approval_mode",
                return_value=preferences,
            ) as set_mode,
            mock.patch("builtins.print"),
        ):
            cli.dispatch(args)

        set_mode.assert_called_once_with(PlanApprovalMode.ASK)


class JobsNukeCliTests(unittest.TestCase):
    def _dispatch_nuke(
        self, argv: list[str], *, total: int, answer: str = "delete"
    ) -> tuple[mock.Mock, mock.Mock]:
        args = cli.parser().parse_args(argv)
        with (
            mock.patch.object(cli, "count_jobs", return_value=total),
            mock.patch.object(
                cli, "nuke_jobs", return_value="Deleted all jobs."
            ) as nuke_jobs,
            mock.patch.object(cli, "list_quarantine_evidence", return_value=[]),
            mock.patch.object(cli, "input", return_value=answer, create=True) as prompt,
            mock.patch("builtins.print"),
        ):
            cli.dispatch(args)
        return nuke_jobs, prompt

    def test_force_deletes_without_prompting(self) -> None:
        nuke_jobs, prompt = self._dispatch_nuke(
            ["jobs", "nuke", "--force"], total=3, answer="delete"
        )
        nuke_jobs.assert_called_once_with()
        prompt.assert_not_called()

    def test_confirmation_word_deletes(self) -> None:
        nuke_jobs, prompt = self._dispatch_nuke(
            ["jobs", "nuke"], total=2, answer="delete"
        )
        prompt.assert_called_once()
        nuke_jobs.assert_called_once_with()

    def test_other_answer_aborts(self) -> None:
        nuke_jobs, prompt = self._dispatch_nuke(["jobs", "nuke"], total=2, answer="no")
        prompt.assert_called_once()
        nuke_jobs.assert_not_called()

    def test_no_jobs_skips_prompt_and_delete(self) -> None:
        nuke_jobs, prompt = self._dispatch_nuke(["jobs", "nuke"], total=0)
        prompt.assert_not_called()
        nuke_jobs.assert_not_called()

    def test_no_live_jobs_reports_unresolved_quarantine(self) -> None:
        args = cli.parser().parse_args(["jobs", "nuke", "--force"])
        with (
            mock.patch.object(cli, "count_jobs", return_value=0),
            mock.patch.object(
                cli, "list_quarantine_evidence", return_value=[mock.sentinel.evidence]
            ),
            mock.patch.object(cli, "nuke_jobs") as nuke_jobs,
            mock.patch("builtins.print") as output,
        ):
            cli.dispatch(args)

        nuke_jobs.assert_not_called()
        self.assertIn("jobs quarantine list", output.call_args.args[0])


class JobsQuarantineCliTests(unittest.TestCase):
    def evidence(self, *, resolved: bool = False) -> QuarantineEvidence:
        return QuarantineEvidence(
            job_id="aaaaaaaaaaaa",
            metadata_path=Path("/state/jobs/.quarantine/job.metadata.json"),
            payload_path=Path("/state/jobs/.quarantine/job.json"),
            quarantined_at=10,
            quarantine_error="unsupported schema",
            resolved=resolved,
            status="running",
            worker_pid=42,
            worker_boot_id="boot",
            worker_process_start="start",
            herdr_target="held-agent",
            worktree_path="/worktrees/held",
            inspection_error=None,
        )

    def test_list_displays_unresolved_reconciliation_details(self) -> None:
        args = cli.parser().parse_args(["jobs", "quarantine", "list"])
        with (
            mock.patch.object(
                cli, "list_quarantine_evidence", return_value=[self.evidence()]
            ) as listing,
            mock.patch("builtins.print") as output,
        ):
            cli.dispatch(args)

        listing.assert_called_once_with(include_resolved=False)
        displayed = "\n".join(str(call.args[0]) for call in output.call_args_list)
        self.assertIn("aaaaaaaaaaaa: unresolved", displayed)
        self.assertIn("pid=42", displayed)
        self.assertIn("held-agent", displayed)
        self.assertIn("/worktrees/held", displayed)

    def test_list_json_includes_resolved_evidence_when_requested(self) -> None:
        args = cli.parser().parse_args(
            ["jobs", "quarantine", "list", "--all", "--json"]
        )
        with (
            mock.patch.object(
                cli,
                "list_quarantine_evidence",
                return_value=[self.evidence(resolved=True)],
            ) as listing,
            mock.patch("builtins.print") as output,
        ):
            cli.dispatch(args)

        listing.assert_called_once_with(include_resolved=True)
        payload = json.loads(output.call_args.args[0])
        self.assertEqual(payload[0]["job_id"], "aaaaaaaaaaaa")
        self.assertTrue(payload[0]["resolved"])

    def test_acknowledge_requires_exact_confirmation(self) -> None:
        args = cli.parser().parse_args(
            [
                "jobs",
                "quarantine",
                "acknowledge",
                "aaaaaaaaaaaa",
                "--reason",
                "verified absent",
            ]
        )
        with (
            mock.patch.object(
                cli, "list_quarantine_evidence", return_value=[self.evidence()]
            ),
            mock.patch.object(
                cli, "acknowledge_quarantine_reservations"
            ) as acknowledge,
            mock.patch.object(cli, "input", return_value="no", create=True) as prompt,
            mock.patch("builtins.print"),
        ):
            cli.dispatch(args)

        prompt.assert_called_once()
        acknowledge.assert_not_called()

    def test_force_acknowledges_with_reason_without_prompting(self) -> None:
        args = cli.parser().parse_args(
            [
                "jobs",
                "quarantine",
                "acknowledge",
                "aaaaaaaaaaaa",
                "--reason",
                "verified absent",
                "--force",
            ]
        )
        with (
            mock.patch.object(
                cli, "list_quarantine_evidence", return_value=[self.evidence()]
            ) as listing,
            mock.patch.object(
                cli,
                "acknowledge_quarantine_reservations",
                return_value="acknowledged",
            ) as acknowledge,
            mock.patch.object(cli, "input", create=True) as prompt,
            mock.patch("builtins.print"),
        ):
            cli.dispatch(args)

        listing.assert_called_once_with(include_resolved=True)
        acknowledge.assert_called_once_with("aaaaaaaaaaaa", reason="verified absent")
        prompt.assert_not_called()


class ListenCliTests(unittest.TestCase):
    def test_listen_asks_the_wake_daemon_to_start_a_conversation(self) -> None:
        args = cli.parser().parse_args(["listen"])
        with (
            mock.patch(
                "local_voice_harness.wake.daemon.request_listen"
            ) as request_listen,
            mock.patch("builtins.print"),
        ):
            cli.dispatch(args)

        request_listen.assert_called_once_with()


class CredentialsCliTests(unittest.TestCase):
    def test_set_prompts_without_accepting_key_as_argument(self) -> None:
        args = cli.parser().parse_args(["credentials", "set"])
        with (
            mock.patch.object(cli.getpass, "getpass", return_value="venice-secret"),
            mock.patch.object(cli, "store_venice_api_key") as store,
            mock.patch("builtins.print"),
        ):
            cli.dispatch(args)

        store.assert_called_once_with("venice-secret")

    def test_status_and_delete_dispatch_to_secret_service(self) -> None:
        for action, function_name in (
            ("status", "get_venice_api_key"),
            ("delete", "delete_venice_api_key"),
        ):
            with (
                self.subTest(action=action),
                mock.patch.object(cli, function_name) as function,
                mock.patch("builtins.print"),
            ):
                cli.dispatch(cli.parser().parse_args(["credentials", action]))
                function.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
