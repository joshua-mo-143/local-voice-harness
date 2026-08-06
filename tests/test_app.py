from __future__ import annotations

import unittest
from unittest import mock

from local_voice_harness import app
from local_voice_harness.browser_context import GitHubContext


class ForegroundDeliveryTests(unittest.TestCase):
    def test_cursor_result_is_acknowledged_after_playback(self) -> None:
        events: list[str] = []

        def cursor_turn(
            _text: str, *, delivery_claims: list[tuple[str, str]]
        ) -> tuple[str, None]:
            delivery_claims.append(("123456789abc", "claim"))
            return "done", None

        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "enrich_request", side_effect=lambda text: text),
            mock.patch.object(app, "cursor_turn", side_effect=cursor_turn),
            mock.patch.object(
                app,
                "stream_and_play",
                side_effect=lambda _text: events.append("played"),
            ),
            mock.patch.object(
                app,
                "acknowledge_deliveries",
                side_effect=lambda _claims: events.append("acknowledged"),
            ),
            mock.patch.object(app, "release_deliveries"),
        ):
            app.respond("Use Cursor to inspect this repository")

        self.assertEqual(events, ["played", "acknowledged"])

    def test_playback_failure_releases_cursor_result(self) -> None:
        def cursor_turn(
            _text: str, *, delivery_claims: list[tuple[str, str]]
        ) -> tuple[str, None]:
            delivery_claims.append(("123456789abc", "claim"))
            return "done", None

        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "enrich_request", side_effect=lambda text: text),
            mock.patch.object(app, "cursor_turn", side_effect=cursor_turn),
            mock.patch.object(
                app,
                "stream_and_play",
                side_effect=RuntimeError("playback failed"),
            ),
            mock.patch.object(app, "acknowledge_deliveries") as acknowledge,
            mock.patch.object(app, "release_deliveries") as release,
            self.assertRaisesRegex(RuntimeError, "playback failed"),
        ):
            app.respond("Use Cursor to inspect this repository")

        acknowledge.assert_not_called()
        release.assert_called_once_with([("123456789abc", "claim")])


class AppContextTests(unittest.TestCase):
    def test_manual_cursor_request_includes_focused_context(self) -> None:
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app, "enrich_request", return_value="ask Cursor to fix this\n\ncontext"
            ) as enrich,
            mock.patch.object(
                app, "cursor_turn", return_value=("done", None)
            ) as cursor_turn,
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("ask Cursor to fix this")

        enrich.assert_called_once_with("ask Cursor to fix this")
        cursor_turn.assert_called_once_with(
            "ask Cursor to fix this\n\ncontext", delivery_claims=mock.ANY
        )

    def test_manual_conversation_includes_focused_context(self) -> None:
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app, "enrich_request", return_value="summarize this\n\ncontext"
            ),
            mock.patch.object(app, "qwen_response", return_value="summary") as qwen,
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("summarize this")

        qwen.assert_called_once_with(
            "summarize this\n\ncontext", delivery_claims=mock.ANY
        )

    def test_explicit_fork_passes_validated_focused_repository(self) -> None:
        request = GitHubContext(
            "fork this repo and add Venice\n\ncontext",
            github_repository="source/project",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "enrich_request", return_value=request),
            mock.patch.object(app, "qwen_response", return_value="started") as qwen,
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("fork this repo and add Venice")

        qwen.assert_called_once_with(
            request,
            github_repository="source/project",
            fork_requested=True,
            github_pull_request=None,
            delivery_claims=mock.ANY,
        )


if __name__ == "__main__":
    unittest.main()
