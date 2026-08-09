from __future__ import annotations

import io
import json
import unittest
from dataclasses import replace
from unittest import mock

from local_voice_harness import intent
from local_voice_harness.browser_context import RequestContext
from local_voice_harness.config import CURSOR_PATTERN, load_backend_settings
from local_voice_harness.credentials import CredentialError


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

    def test_local_uses_configured_endpoint_model_and_timeout(self) -> None:
        settings = replace(
            load_backend_settings({}),
            llm_provider="local",
            llm_model="local-router",
            llm_endpoint="http://localhost:9000/v1/chat/completions",
            llm_timeout=11,
        )
        with (
            mock.patch.object(intent, "load_backend_settings", return_value=settings),
            mock.patch.object(intent, "get_venice_api_key") as get_key,
            mock.patch.object(
                intent.urllib.request,
                "urlopen",
                return_value=_response("conversation"),
            ) as urlopen,
        ):
            route = intent.route_intent("hello", RequestContext("hello"))

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(route.intent, intent.Intent.CONVERSATION)
        self.assertEqual(request.full_url, settings.llm_endpoint)
        self.assertEqual(payload["model"], settings.llm_model)
        self.assertEqual(urlopen.call_args.kwargs["timeout"], settings.llm_timeout)
        self.assertIsNone(request.get_header("Authorization"))
        get_key.assert_not_called()

    def test_venice_uses_configured_endpoint_model_timeout_and_credentials(
        self,
    ) -> None:
        settings = replace(
            load_backend_settings({}),
            llm_provider="venice",
            llm_model="venice-router",
            llm_endpoint="https://example.test/v1/chat/completions",
            llm_timeout=17,
        )
        with (
            mock.patch.object(intent, "load_backend_settings", return_value=settings),
            mock.patch.object(
                intent, "get_venice_api_key", return_value="venice-secret"
            ) as get_key,
            mock.patch.object(
                intent.urllib.request,
                "urlopen",
                return_value=_response("cursor_submit"),
            ) as urlopen,
        ):
            route = intent.route_intent("work on this", RequestContext("work on this"))

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(route.intent, intent.Intent.CURSOR_SUBMIT)
        self.assertEqual(request.full_url, settings.llm_endpoint)
        self.assertEqual(payload["model"], settings.llm_model)
        self.assertEqual(urlopen.call_args.kwargs["timeout"], settings.llm_timeout)
        self.assertEqual(request.get_header("Authorization"), "Bearer venice-secret")
        get_key.assert_called_once_with()

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

    def test_missing_venice_credentials_fall_back_safely(self) -> None:
        settings = replace(load_backend_settings({}), llm_provider="venice")
        with (
            mock.patch.object(intent, "load_backend_settings", return_value=settings),
            mock.patch.object(
                intent,
                "get_venice_api_key",
                side_effect=CredentialError("missing"),
            ),
            mock.patch.object(intent.urllib.request, "urlopen") as urlopen,
        ):
            route = intent.route_intent("request", RequestContext("request"))

        self.assertEqual(route, intent.FALLBACK_ROUTE)
        urlopen.assert_not_called()

    def test_venice_router_uses_configured_endpoint_and_authorization(self) -> None:
        settings = mock.Mock(
            llm_provider="venice",
            llm_model="venice-model",
            llm_endpoint="https://api.venice.example/chat",
            llm_timeout=17,
        )
        with (
            mock.patch.object(intent, "load_backend_settings", return_value=settings),
            mock.patch.object(
                intent, "get_venice_api_key", return_value="secret-token"
            ),
            mock.patch.object(
                intent.urllib.request,
                "urlopen",
                return_value=_response("conversation"),
            ) as urlopen,
        ):
            intent.route_intent("hello", RequestContext("hello"))

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, settings.llm_endpoint)
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-token")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 17)
        self.assertEqual(json.loads(request.data)["model"], "venice-model")


class InboxIntentTests(unittest.TestCase):
    def test_route_tool_exposes_inbox_intents(self) -> None:
        enum = intent.ROUTE_TOOL["function"]["parameters"]["properties"]["intent"][
            "enum"
        ]
        self.assertIn("cursor_list", enum)
        self.assertIn("cursor_dismiss", enum)
        self.assertIn("cursor_repeat", enum)

    def test_inbox_intents_are_actionable_at_high_confidence(self) -> None:
        for name in ("cursor_list", "cursor_dismiss", "cursor_repeat"):
            with (
                self.subTest(intent=name),
                mock.patch.object(
                    intent.urllib.request,
                    "urlopen",
                    return_value=_response(name),
                ),
            ):
                route = intent.route_intent("request", RequestContext("request"))
                self.assertTrue(route.actionable)
                self.assertEqual(route.intent, intent.Intent(name))


class FollowUpIntentTests(unittest.TestCase):
    def test_route_tool_exposes_follow_up_intents(self) -> None:
        enum = intent.ROUTE_TOOL["function"]["parameters"]["properties"]["intent"][
            "enum"
        ]
        self.assertIn("cursor_followup", enum)
        self.assertIn("cursor_pr_unsupported", enum)

    def test_recent_completion_flag_is_forwarded(self) -> None:
        with mock.patch.object(
            intent.urllib.request,
            "urlopen",
            return_value=_response("cursor_followup"),
        ) as urlopen:
            route = intent.route_intent(
                "review the changes",
                RequestContext("review the changes"),
                recent_completion=True,
            )

        self.assertEqual(route.intent, intent.Intent.CURSOR_FOLLOWUP)
        self.assertTrue(route.actionable)
        payload = json.loads(urlopen.call_args.args[0].data)
        router_input = json.loads(payload["messages"][1]["content"])
        self.assertTrue(router_input["recent_completed_job"])

    def test_recent_completion_defaults_to_false(self) -> None:
        with mock.patch.object(
            intent.urllib.request,
            "urlopen",
            return_value=_response("conversation"),
        ) as urlopen:
            intent.route_intent("hello", RequestContext("hello"))

        payload = json.loads(urlopen.call_args.args[0].data)
        router_input = json.loads(payload["messages"][1]["content"])
        self.assertFalse(router_input["recent_completed_job"])

    def test_pending_clarification_suppresses_completed_context(self) -> None:
        with mock.patch.object(
            intent.urllib.request,
            "urlopen",
            return_value=_response("cursor_reply"),
        ) as urlopen:
            route = intent.route_intent(
                "use the api repository",
                RequestContext("use the api repository"),
                cursor_session="aaaaaaaaaaaa",
                pending_question="Which repository should I use?",
                clarification_kind="repository",
                recent_completion=True,
            )

        self.assertEqual(route.intent, intent.Intent.CURSOR_REPLY)
        payload = json.loads(urlopen.call_args.args[0].data)
        router_input = json.loads(payload["messages"][1]["content"])
        self.assertFalse(router_input["recent_completed_job"])
        self.assertEqual(
            router_input["pending_cursor_question"],
            "Which repository should I use?",
        )
        self.assertEqual(router_input["clarification_kind"], "repository")

    def test_pr_unsupported_is_actionable(self) -> None:
        with mock.patch.object(
            intent.urllib.request,
            "urlopen",
            return_value=_response("cursor_pr_unsupported"),
        ):
            route = intent.route_intent("open a pull request", RequestContext("open"))
        self.assertTrue(route.actionable)
        self.assertEqual(route.intent, intent.Intent.CURSOR_PR_UNSUPPORTED)


class EndConversationIntentTests(unittest.TestCase):
    def test_route_tool_exposes_end_conversation(self) -> None:
        enum = intent.ROUTE_TOOL["function"]["parameters"]["properties"]["intent"][
            "enum"
        ]
        self.assertIn("end_conversation", enum)

    def test_end_conversation_is_actionable_at_high_confidence(self) -> None:
        with mock.patch.object(
            intent.urllib.request,
            "urlopen",
            return_value=_response("end_conversation"),
        ):
            route = intent.route_intent("thanks, that's all", RequestContext("thanks"))

        self.assertEqual(route.intent, intent.Intent.END_CONVERSATION)
        self.assertTrue(route.actionable)

    def test_end_conversation_needs_high_confidence(self) -> None:
        with mock.patch.object(
            intent.urllib.request,
            "urlopen",
            return_value=_response("end_conversation", "medium"),
        ):
            route = intent.route_intent("maybe done", RequestContext("maybe done"))

        self.assertFalse(route.actionable)


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
