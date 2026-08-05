from __future__ import annotations

import unittest
from unittest import mock

from local_voice_harness import app


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
        cursor_turn.assert_called_once_with("ask Cursor to fix this\n\ncontext")

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

        qwen.assert_called_once_with("summarize this\n\ncontext")


if __name__ == "__main__":
    unittest.main()
