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
