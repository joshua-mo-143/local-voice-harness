from __future__ import annotations

import unittest
from unittest import mock

from local_voice_harness import cli
from local_voice_harness.cursor.service import CursorTurnRequest, CursorTurnResult


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
