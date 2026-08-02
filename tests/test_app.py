from __future__ import annotations

import unittest
from unittest import mock

from local_voice_harness import app


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
            mock.patch.object(app, "cursor_turn", side_effect=cursor_turn),
            mock.patch.object(
                app,
                "synthesize_and_play",
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
            mock.patch.object(app, "cursor_turn", side_effect=cursor_turn),
            mock.patch.object(
                app,
                "synthesize_and_play",
                side_effect=RuntimeError("playback failed"),
            ),
            mock.patch.object(app, "acknowledge_deliveries") as acknowledge,
            mock.patch.object(app, "release_deliveries") as release,
            self.assertRaisesRegex(RuntimeError, "playback failed"),
        ):
            app.respond("Use Cursor to inspect this repository")

        acknowledge.assert_not_called()
        release.assert_called_once_with([("123456789abc", "claim")])


if __name__ == "__main__":
    unittest.main()
