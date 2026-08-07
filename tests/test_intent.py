from __future__ import annotations

import io
import json
import unittest
from unittest import mock

from local_voice_harness import intent
from local_voice_harness.browser_context import RequestContext
from local_voice_harness.config import CURSOR_PATTERN


def _response(route: str, confidence: str = "high") -> io.BytesIO:
    return io.BytesIO(
        json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "route_intent",
                                        "arguments": json.dumps(
                                            {
                                                "intent": route,
                                                "confidence": confidence,
                                            }
                                        ),
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        ).encode()
    )


class IntentRouterTests(unittest.TestCase):
    def test_sends_bounded_context_to_forced_router_tool(self) -> None:
        context = RequestContext(
            text="work on this\n\nuntrusted issue body",
            focused_repository="example/project",
            focused_issue="example/project#42",
        )
        with mock.patch.object(
            intent.urllib.request,
            "urlopen",
            return_value=_response("cursor_submit"),
        ) as urlopen:
            route = intent.route_intent("work on this", context)

        self.assertEqual(route.intent, intent.Intent.CURSOR_SUBMIT)
        self.assertTrue(route.actionable)
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        router_input = json.loads(payload["messages"][1]["content"])
        self.assertEqual(router_input["utterance"], "work on this")
        self.assertEqual(router_input["focused_repository"], "example/project")
        self.assertNotIn("untrusted issue body", request.data.decode())
        self.assertEqual(
            payload["tool_choice"]["function"]["name"],
            "route_intent",
        )
        self.assertEqual(payload["temperature"], 0)

    def test_only_high_confidence_non_conversation_routes_are_actionable(self) -> None:
        context = RequestContext("hello")
        cases = [
            ("cursor_reply", "high", True),
            ("cursor_status", "medium", False),
            ("conversation", "high", False),
            ("uncertain", "high", False),
        ]
        for route_name, confidence, actionable in cases:
            with (
                self.subTest(route=route_name, confidence=confidence),
                mock.patch.object(
                    intent.urllib.request,
                    "urlopen",
                    return_value=_response(route_name, confidence),
                ),
            ):
                route = intent.route_intent("request", context)
                self.assertEqual(route.actionable, actionable)

    def test_malformed_or_failed_router_falls_back_safely(self) -> None:
        responses: list[object] = [
            io.BytesIO(b'{"choices": []}'),
            _response("not-an-intent"),
            OSError("offline"),
        ]
        for response in responses:
            side_effect = response if isinstance(response, Exception) else None
            return_value = None if side_effect else response
            with (
                self.subTest(response=response),
                mock.patch.object(
                    intent.urllib.request,
                    "urlopen",
                    return_value=return_value,
                    side_effect=side_effect,
                ),
            ):
                route = intent.route_intent("request", RequestContext("request"))
                self.assertEqual(route, intent.FALLBACK_ROUTE)


class ForkIntentTests(unittest.TestCase):
    def test_accepts_only_unambiguous_fork_requests(self) -> None:
        for utterance in (
            "fork this repo and add Venice",
            "please fork owner/project",
            "ask Cursor to fork this repository",
            "I want you to fork this repo",
            "I would like you to fork this repo",
        ):
            with self.subTest(utterance=utterance):
                self.assertEqual(
                    intent.decide_fork_intent(utterance),
                    intent.ForkIntent.AFFIRMATIVE,
                )

    def test_rejects_negative_quoted_hypothetical_and_informational_uses(self) -> None:
        for utterance in (
            "do not fork this repository",
            'the ticket says "fork this repo"',
            "the ticket says please fork this repo",
            "if we fork this repo, what happens",
            "is this already forked",
            "could you fork this repository",
            "this issue discusses fork behavior",
        ):
            with self.subTest(utterance=utterance):
                self.assertEqual(
                    intent.decide_fork_intent(utterance),
                    intent.ForkIntent.NON_AFFIRMATIVE,
                )

    def test_unrelated_request_has_no_fork_intent(self) -> None:
        self.assertEqual(
            intent.decide_fork_intent("work on this repository"),
            intent.ForkIntent.NONE,
        )


class CursorPatternTests(unittest.TestCase):
    def test_matches_explicit_cursor_delegation(self) -> None:
        for utterance in (
            "use cursor to fix this",
            "ask Cursor to inspect",
            "call curser: refactor",
        ):
            with self.subTest(utterance=utterance):
                self.assertIsNotNone(CURSOR_PATTERN.search(utterance))

    def test_does_not_match_implicit_work_requests(self) -> None:
        for utterance in (
            "work on this",
            "summarize this",
            "fix the bug in auth",
        ):
            with self.subTest(utterance=utterance):
                self.assertIsNone(CURSOR_PATTERN.search(utterance))


if __name__ == "__main__":
    unittest.main()
