from __future__ import annotations

import io
import json
import unittest
import urllib.error
from contextlib import redirect_stdout
from dataclasses import replace
from email.message import Message
from unittest import mock

from local_voice_harness import llm
from local_voice_harness.config import load_backend_settings
from local_voice_harness.cursor.service import CursorTurnRequest
from local_voice_harness.errors import HarnessError


def _response(message: dict[str, object]) -> io.BytesIO:
    return io.BytesIO(json.dumps({"choices": [{"message": message}]}).encode())


def _stream_response(*deltas: dict[str, object]) -> io.BytesIO:
    lines = [
        f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n" for delta in deltas
    ]
    lines.append("data: [DONE]\n\n")
    return io.BytesIO("".join(lines).encode())


class QwenClientTests(unittest.TestCase):
    def test_cursor_followup_uses_only_one_leading_system_message(self) -> None:
        with (
            mock.patch.object(
                llm.urllib.request,
                "urlopen",
                return_value=_response({"content": "Continuing now."}),
            ) as urlopen,
            redirect_stdout(io.StringIO()),
        ):
            answer, session = llm.qwen_turn(
                "Continue please.",
                [{"role": "assistant", "content": "The job needs authentication."}],
                "123456789abc",
            )

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        system_messages = [
            message for message in payload["messages"] if message["role"] == "system"
        ]
        self.assertEqual(len(system_messages), 1)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertIn("awaiting the user's reply", system_messages[0]["content"])
        self.assertEqual((answer, session), ("Continuing now.", "123456789abc"))

    def test_allow_tools_false_omits_tools_and_returns_content(self) -> None:
        with (
            mock.patch.object(
                llm.urllib.request,
                "urlopen",
                return_value=_response({"content": "Just chatting."}),
            ) as urlopen,
            redirect_stdout(io.StringIO()),
        ):
            answer, session = llm.qwen_turn("hello", allow_tools=False)

        self.assertEqual((answer, session), ("Just chatting.", None))
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)

    def test_allow_tools_false_rejects_returned_tool_calls(self) -> None:
        message = {
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "cursor_turn",
                        "arguments": json.dumps({"request": "change the code"}),
                    },
                }
            ],
        }
        with (
            mock.patch.object(
                llm.urllib.request,
                "urlopen",
                return_value=_response(message),
            ),
            mock.patch.object(llm, "cursor_turn") as cursor,
            redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(HarnessError, "tools are disabled"),
        ):
            llm.qwen_turn("hello", allow_tools=False)

        cursor.assert_not_called()

    def test_allow_tools_true_includes_tools(self) -> None:
        with (
            mock.patch.object(
                llm.urllib.request,
                "urlopen",
                return_value=_response({"content": "answer"}),
            ) as urlopen,
            redirect_stdout(io.StringIO()),
        ):
            llm.qwen_turn("hello", allow_tools=True)

        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertIn("tools", payload)
        self.assertEqual(payload["tool_choice"], "auto")

    def test_sends_expected_chat_payload_and_limits_history(self) -> None:
        history = [
            {"role": "user", "content": f"message {index}"} for index in range(10)
        ]
        with (
            mock.patch.object(
                llm.urllib.request,
                "urlopen",
                return_value=_response({"content": "  concise answer  "}),
            ) as urlopen,
            redirect_stdout(io.StringIO()),
        ):
            answer, session = llm.qwen_turn("hello", history)

        self.assertEqual(answer, "concise answer")
        self.assertIsNone(session)
        request = urlopen.call_args.args[0]
        settings = load_backend_settings()
        self.assertEqual(request.full_url, settings.llm_endpoint)
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 60)
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], settings.llm_model)
        system_prompt = payload["messages"][0]["content"]
        self.assertIn(
            'respond only with "I\'ve finished working on <identifier>"',
            system_prompt,
        )
        self.assertIn("acknowledge it in one brief sentence", system_prompt)
        self.assertEqual(payload["messages"][1], history[2])
        self.assertEqual(payload["messages"][-1], {"role": "user", "content": "hello"})
        self.assertEqual(payload["tools"], llm.QWEN_TOOLS)
        self.assertFalse(payload["stream"])

    def test_executes_cursor_tool_and_returns_followup_answer(self) -> None:
        tool_call = {
            "id": "call-1",
            "function": {
                "name": "cursor",
                "arguments": json.dumps(
                    {"task": "fix it", "repository": "example", "action": "submit"}
                ),
            },
        }
        with (
            mock.patch.object(
                llm.urllib.request,
                "urlopen",
                side_effect=[
                    _response({"content": None, "tool_calls": [tool_call]}),
                    _response({"content": "job started"}),
                ],
            ) as urlopen,
            mock.patch.object(
                llm, "cursor_turn", return_value=("accepted", "job-123")
            ) as cursor_turn,
            mock.patch.object(llm, "notify") as notify,
            redirect_stdout(io.StringIO()),
        ):
            answer, session = llm.qwen_turn("please fix it")

        self.assertEqual((answer, session), ("job started", "job-123"))
        notify.assert_called_once_with("Cursor is working…")
        cursor_turn.assert_called_once_with(
            CursorTurnRequest(
                "fix it",
                None,
                repository="example",
                agent=None,
                utterance=None,
                action="submit",
                job_id=None,
            ),
            delivery_claims=None,
        )
        second_request = urlopen.call_args_list[1].args[0]
        second_payload = json.loads(second_request.data)
        self.assertEqual(
            second_payload["messages"][-1],
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "cursor",
                "content": "accepted",
            },
        )

    def test_retries_without_replaying_malformed_tool_arguments(self) -> None:
        malformed_call = {
            "id": "call-bad",
            "type": "function",
            "function": {
                "name": "cursor",
                "arguments": '{"task":"on this issue please."',
            },
        }
        with (
            mock.patch.object(
                llm.urllib.request,
                "urlopen",
                side_effect=[
                    _response({"content": None, "tool_calls": [malformed_call]}),
                    _response({"content": "Please try that request again."}),
                ],
            ) as urlopen,
            mock.patch.object(llm, "cursor_turn") as cursor_turn,
            redirect_stdout(io.StringIO()),
        ):
            answer, session = llm.qwen_turn("on this issue please.")

        self.assertEqual((answer, session), ("Please try that request again.", None))
        cursor_turn.assert_not_called()
        second_request = urlopen.call_args_list[1].args[0]
        second_payload = json.loads(second_request.data)
        self.assertFalse(
            any("tool_calls" in message for message in second_payload["messages"])
        )

    def test_returns_recovery_after_repeated_malformed_tool_arguments(self) -> None:
        malformed_message = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call-bad",
                    "type": "function",
                    "function": {"name": "cursor", "arguments": '{"task":'},
                }
            ],
        }
        with (
            mock.patch.object(
                llm.urllib.request,
                "urlopen",
                side_effect=[
                    _response(malformed_message),
                    _response(malformed_message),
                ],
            ) as urlopen,
            mock.patch.object(llm, "cursor_turn") as cursor_turn,
            redirect_stdout(io.StringIO()),
        ):
            answer, session = llm.qwen_turn("on this issue please.")

        self.assertEqual((answer, session), (llm.MALFORMED_TOOL_CALL_RECOVERY, None))
        self.assertEqual(urlopen.call_count, 2)
        cursor_turn.assert_not_called()

    def test_focused_repository_and_explicit_fork_are_forwarded(self) -> None:
        tool_call = {
            "id": "call-1",
            "function": {
                "name": "cursor",
                "arguments": json.dumps(
                    {
                        "task": "add Venice",
                        "github_repository": "hallucinated/wrong",
                    }
                ),
            },
        }
        with (
            mock.patch.object(
                llm.urllib.request,
                "urlopen",
                side_effect=[
                    _response({"content": None, "tool_calls": [tool_call]}),
                    _response({"content": "started"}),
                ],
            ),
            mock.patch.object(
                llm, "cursor_turn", return_value=("accepted", None)
            ) as cursor_turn,
            mock.patch.object(llm, "notify"),
            redirect_stdout(io.StringIO()),
        ):
            llm.qwen_turn(
                "fork this repo and add Venice",
                github_repository="source/project",
                fork_requested=True,
            )

        cursor_turn.assert_called_once_with(
            CursorTurnRequest(
                "add Venice",
                None,
                repository=None,
                github_repository="source/project",
                github_issue=None,
                github_issue_context=None,
                fork_requested=True,
                github_pull_request=None,
                agent=None,
                utterance=None,
                action="submit",
                job_id=None,
            ),
            delivery_claims=None,
        )

    def test_rejects_empty_and_malformed_responses(self) -> None:
        responses = [
            _response({"content": ""}),
            io.BytesIO(b"[]"),
            io.BytesIO(b'{"choices": []}'),
            io.BytesIO(b"not json"),
            _response({"content": "ignored", "tool_calls": ["not-an-object"]}),
        ]
        for response in responses:
            with (
                self.subTest(response=response.getvalue()),
                mock.patch.object(llm.urllib.request, "urlopen", return_value=response),
                redirect_stdout(io.StringIO()),
                self.assertRaises(HarnessError),
            ):
                llm.qwen_turn("hello")

    def test_wraps_http_transport_errors(self) -> None:
        with (
            mock.patch.object(
                llm.urllib.request,
                "urlopen",
                side_effect=urllib.error.URLError("connection refused"),
            ),
            self.assertRaisesRegex(HarnessError, "LLM request failed"),
        ):
            llm.qwen_turn("hello")

    def test_includes_http_error_response_detail(self) -> None:
        error = urllib.error.HTTPError(
            "https://api.venice.ai/api/v1/chat/completions",
            400,
            "Bad Request",
            Message(),
            io.BytesIO(b'{"error":"invalid tool_call_id"}'),
        )
        with (
            mock.patch.object(
                llm.urllib.request,
                "urlopen",
                side_effect=error,
            ),
            self.assertRaisesRegex(HarnessError, "invalid tool_call_id"),
        ):
            llm.qwen_turn("hello")

    def test_venice_uses_configured_endpoint_model_and_bearer_token(self) -> None:
        settings = replace(
            load_backend_settings({}),
            llm_provider="venice",
            llm_model="venice-uncensored",
            llm_endpoint="https://api.venice.ai/api/v1/chat/completions",
            llm_timeout=17,
        )
        with (
            mock.patch.object(llm, "load_backend_settings", return_value=settings),
            mock.patch.object(
                llm, "get_venice_api_key", return_value="venice-secret"
            ) as get_key,
            mock.patch.object(
                llm.urllib.request,
                "urlopen",
                return_value=_stream_response(
                    {"content": "hello "},
                    {"content": "there."},
                ),
            ) as urlopen,
            redirect_stdout(io.StringIO()),
        ):
            chunks: list[str] = []
            answer, _ = llm.qwen_turn("hello", on_text_chunk=chunks.append)

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(answer, "hello there.")
        self.assertEqual(chunks, ["hello there."])
        self.assertEqual(request.full_url, settings.llm_endpoint)
        self.assertEqual(request.get_header("Authorization"), "Bearer venice-secret")
        self.assertEqual(request.get_header("Accept"), "text/event-stream")
        self.assertEqual(payload["model"], settings.llm_model)
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["reasoning"], {"enabled": False})
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 17)
        get_key.assert_called_once_with()

    def test_venice_aggregates_streamed_tool_call_fragments(self) -> None:
        settings = replace(
            load_backend_settings({}),
            llm_provider="venice",
            llm_model="zai-org-glm-5-2",
        )
        with (
            mock.patch.object(llm, "load_backend_settings", return_value=settings),
            mock.patch.object(llm, "get_venice_api_key", return_value="secret"),
            mock.patch.object(
                llm.urllib.request,
                "urlopen",
                side_effect=[
                    _stream_response(
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "cursor",
                                        "arguments": '{"task":"fix',
                                    },
                                }
                            ]
                        },
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": ' it","action":"submit"}'
                                    },
                                }
                            ]
                        },
                    ),
                    _stream_response({"content": "Started."}),
                ],
            ),
            mock.patch.object(
                llm, "cursor_turn", return_value=("accepted", "job-123")
            ) as cursor_turn,
            mock.patch.object(llm, "notify"),
            redirect_stdout(io.StringIO()),
        ):
            chunks: list[str] = []
            answer, session = llm.qwen_turn("fix it", on_text_chunk=chunks.append)

        self.assertEqual((answer, session), ("Started.", "job-123"))
        self.assertEqual(chunks, ["Started."])
        cursor_turn.assert_called_once()

    def test_logs_payload_aggregated_response_and_tool_exchange(self) -> None:
        settings = replace(
            load_backend_settings({}),
            llm_provider="venice",
            llm_model="zai-org-glm-5-2",
        )
        tool_arguments = '{"task":"fix it","action":"submit"}'
        output = io.StringIO()
        with (
            mock.patch.object(llm, "load_backend_settings", return_value=settings),
            mock.patch.object(llm, "get_venice_api_key", return_value="secret"),
            mock.patch.object(
                llm.urllib.request,
                "urlopen",
                side_effect=[
                    _stream_response(
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "function": {
                                        "name": "cursor",
                                        "arguments": tool_arguments,
                                    },
                                }
                            ]
                        }
                    ),
                    _stream_response({"content": "Started."}),
                ],
            ),
            mock.patch.object(llm, "cursor_turn", return_value=("accepted", "job-123")),
            mock.patch.object(llm, "notify"),
            redirect_stdout(output),
        ):
            llm.qwen_turn("fix it")

        records = [json.loads(line) for line in output.getvalue().splitlines()]
        requests = [record for record in records if record.get("event") == "request"]
        responses = [
            record for record in records if record.get("event") == "aggregated_response"
        ]
        calls = [record for record in records if record.get("event") == "tool_call"]
        results = [record for record in records if record.get("event") == "tool_result"]

        self.assertEqual(len(requests), 2)
        self.assertEqual(
            json.loads(requests[0]["payload"])["messages"][-1]["content"], "fix it"
        )
        self.assertNotIn("secret", requests[0]["payload"])
        self.assertEqual(responses[0]["response"]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(responses[1]["response"], {"content": "Started."})
        self.assertEqual(calls[0]["arguments"], tool_arguments)
        self.assertEqual(results[0]["result"], "accepted")

    def test_venice_accepts_mixed_text_and_tool_stream_without_speaking(self) -> None:
        response = _stream_response(
            {"content": "Speaking already. "},
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call-1",
                        "function": {"name": "cursor", "arguments": "{}"},
                    }
                ]
            },
        )
        chunks: list[str] = []

        message = llm._streamed_message(response, chunks.append)

        self.assertEqual(message["content"], "Speaking already. ")
        tool_calls = message["tool_calls"]
        self.assertIsInstance(tool_calls, list)
        assert isinstance(tool_calls, list)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(chunks, [])

    def test_venice_retries_truncated_streamed_tool_arguments(self) -> None:
        settings = replace(
            load_backend_settings({}),
            llm_provider="venice",
            llm_model="zai-org-glm-5-2",
        )
        with (
            mock.patch.object(llm, "load_backend_settings", return_value=settings),
            mock.patch.object(llm, "get_venice_api_key", return_value="secret"),
            mock.patch.object(
                llm.urllib.request,
                "urlopen",
                side_effect=[
                    _stream_response(
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-truncated",
                                    "type": "function",
                                    "function": {
                                        "name": "cursor",
                                        "arguments": '{"task":"on this issue',
                                    },
                                }
                            ]
                        }
                    ),
                    _stream_response({"content": "Please try that request again."}),
                ],
            ) as urlopen,
            mock.patch.object(llm, "cursor_turn") as cursor_turn,
            redirect_stdout(io.StringIO()),
        ):
            answer, session = llm.qwen_turn("on this issue please.")

        self.assertEqual((answer, session), ("Please try that request again.", None))
        cursor_turn.assert_not_called()
        first_payload = json.loads(urlopen.call_args_list[0].args[0].data)
        second_payload = json.loads(urlopen.call_args_list[1].args[0].data)
        self.assertEqual(first_payload["max_tokens"], llm.MAX_COMPLETION_TOKENS)
        self.assertGreater(first_payload["max_tokens"], 128)
        self.assertFalse(
            any("tool_calls" in message for message in second_payload["messages"])
        )

    def test_venice_surfaces_stream_errors_and_empty_streams(self) -> None:
        error = io.BytesIO(b'data: {"error": "model unavailable"}\n\n')
        with self.assertRaisesRegex(HarnessError, "model unavailable"):
            llm._streamed_message(error, None)

        malformed = io.BytesIO(b"data: []\n\n")
        with self.assertRaisesRegex(HarnessError, "malformed streaming event"):
            llm._streamed_message(malformed, None)

        empty = io.BytesIO(b"data: [DONE]\n\n")
        with self.assertRaisesRegex(HarnessError, "empty streaming response"):
            llm._streamed_message(empty, None)

    def test_venice_rejects_malformed_streamed_tool_index(self) -> None:
        response = _stream_response(
            {
                "tool_calls": [
                    {
                        "index": "not-an-index",
                        "function": {"name": "cursor", "arguments": "{}"},
                    }
                ]
            }
        )

        with self.assertRaisesRegex(HarnessError, "malformed streaming tool calls"):
            llm._streamed_message(response, None)


if __name__ == "__main__":
    unittest.main()
