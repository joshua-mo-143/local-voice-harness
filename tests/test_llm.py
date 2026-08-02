from __future__ import annotations

import io
import json
import unittest
import urllib.error
from contextlib import redirect_stdout
from unittest import mock

from local_voice_harness import llm
from local_voice_harness.errors import HarnessError


def _response(message: dict[str, object]) -> io.BytesIO:
    return io.BytesIO(json.dumps({"choices": [{"message": message}]}).encode())


class QwenClientTests(unittest.TestCase):
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
        self.assertEqual(request.full_url, llm.LLM_CHAT)
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 60)
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "qwen3.5-4b")
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
            "fix it",
            None,
            repository="example",
            agent=None,
            action="submit",
            job_id=None,
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

    def test_rejects_empty_and_malformed_responses(self) -> None:
        responses = [
            _response({"content": ""}),
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
            self.assertRaisesRegex(HarnessError, "Qwen request failed"),
        ):
            llm.qwen_turn("hello")


if __name__ == "__main__":
    unittest.main()
