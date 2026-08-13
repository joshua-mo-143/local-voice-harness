from __future__ import annotations

import io
import json
import unittest
import urllib.error
from collections.abc import Iterator
from dataclasses import replace
from email.message import Message
from unittest import mock

from local_voice_harness.config import load_backend_settings
from local_voice_harness.credentials import CredentialError
from local_voice_harness.errors import HarnessError
from local_voice_harness.llm_transport import (
    ChatCompletionRequest,
    LlmTransport,
    LlmTransportConfig,
    streamed_message,
)


def _response(message: dict[str, object]) -> io.BytesIO:
    return io.BytesIO(json.dumps({"choices": [{"message": message}]}).encode())


def _stream_response(*deltas: dict[str, object]) -> io.BytesIO:
    lines = [
        f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n" for delta in deltas
    ]
    lines.append("data: [DONE]\n\n")
    return io.BytesIO("".join(lines).encode())


class LlmTransportContractTests(unittest.TestCase):
    def _transport(
        self,
        *,
        provider: str = "local",
        api_key: str | None = None,
        timeout: float = 60,
    ) -> LlmTransport:
        settings = replace(
            load_backend_settings({}),
            llm_provider=provider,
            llm_model="test-model",
            llm_endpoint="http://localhost:9000/v1/chat/completions",
            llm_timeout=timeout,
        )
        return LlmTransport(
            LlmTransportConfig(
                provider=settings.llm_provider,
                model=settings.llm_model,
                endpoint=settings.llm_endpoint,
                timeout=settings.llm_timeout,
                api_key=api_key,
            )
        )

    def test_local_uses_configured_endpoint_model_and_timeout(self) -> None:
        transport = self._transport(timeout=11)
        with mock.patch(
            "local_voice_harness.llm_transport.urllib.request.urlopen",
            return_value=_response({"content": "hello"}),
        ) as urlopen:
            message = transport.chat_completion(
                ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])
            )

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(message, {"content": "hello"})
        self.assertEqual(request.full_url, transport.config.endpoint)
        self.assertEqual(payload["model"], "test-model")
        self.assertFalse(payload["stream"])
        self.assertNotIn("reasoning", payload)
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 11)

    def test_venice_uses_bearer_token_streaming_and_reasoning(self) -> None:
        transport = self._transport(
            provider="venice", api_key="venice-secret", timeout=17
        )
        with mock.patch(
            "local_voice_harness.llm_transport.pooled_urlopen",
            return_value=_stream_response({"content": "hello"}),
        ) as urlopen:
            message = transport.chat_completion(
                ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])
            )

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(message, {"content": "hello"})
        self.assertEqual(request.get_header("Authorization"), "Bearer venice-secret")
        self.assertEqual(request.get_header("Accept"), "text/event-stream")
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["reasoning"], {"enabled": False})
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 17)

    def test_from_settings_resolves_venice_credentials(self) -> None:
        settings = replace(load_backend_settings({}), llm_provider="venice")
        with mock.patch(
            "local_voice_harness.llm_transport.get_venice_api_key",
            return_value="resolved-key",
        ) as get_key:
            transport = LlmTransport.from_settings(settings)

        self.assertEqual(transport.config.api_key, "resolved-key")
        get_key.assert_called_once_with()

    def test_from_settings_skips_credentials_for_local_provider(self) -> None:
        settings = replace(load_backend_settings({}), llm_provider="local")
        with mock.patch(
            "local_voice_harness.llm_transport.get_venice_api_key"
        ) as get_key:
            transport = LlmTransport.from_settings(settings)

        self.assertIsNone(transport.config.api_key)
        get_key.assert_not_called()

    def test_malformed_json_response_raises(self) -> None:
        transport = self._transport()
        with (
            mock.patch(
                "local_voice_harness.llm_transport.urllib.request.urlopen",
                return_value=io.BytesIO(b"not json"),
            ),
            self.assertRaisesRegex(HarnessError, "LLM request failed"),
        ):
            transport.chat_completion(
                ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])
            )

    def test_malformed_response_shape_raises(self) -> None:
        transport = self._transport()
        cases = [
            io.BytesIO(b"[]"),
            io.BytesIO(b'{"choices": []}'),
        ]
        for response in cases:
            with (
                self.subTest(response=response.getvalue()),
                mock.patch(
                    "local_voice_harness.llm_transport.urllib.request.urlopen",
                    return_value=response,
                ),
                self.assertRaises(HarnessError),
            ):
                transport.chat_completion(
                    ChatCompletionRequest(
                        messages=[{"role": "user", "content": "hello"}]
                    )
                )

    def test_http_failures_include_redacted_body_without_authorization(self) -> None:
        transport = self._transport(provider="venice", api_key="secret-token")
        error = urllib.error.HTTPError(
            transport.config.endpoint,
            401,
            "Unauthorized",
            Message(),
            io.BytesIO(b'{"error":"Bearer secret-token is invalid"}'),
        )
        with (
            mock.patch(
                "local_voice_harness.llm_transport.pooled_urlopen",
                side_effect=error,
            ),
            self.assertRaisesRegex(HarnessError, "HTTP 401") as ctx,
        ):
            transport.chat_completion(
                ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])
            )

        self.assertNotIn("secret-token", str(ctx.exception))
        assert error.fp is not None
        self.assertTrue(error.fp.closed)

    def test_transport_errors_are_wrapped(self) -> None:
        transport = self._transport()
        with (
            mock.patch(
                "local_voice_harness.llm_transport.urllib.request.urlopen",
                side_effect=urllib.error.URLError("connection refused"),
            ),
            self.assertRaisesRegex(HarnessError, "LLM request failed"),
        ):
            transport.chat_completion(
                ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])
            )

    def test_timeouts_are_wrapped(self) -> None:
        transport = self._transport(timeout=3)
        with (
            mock.patch(
                "local_voice_harness.llm_transport.urllib.request.urlopen",
                side_effect=TimeoutError("timed out"),
            ),
            self.assertRaisesRegex(HarnessError, "timed out after 3s"),
        ):
            transport.chat_completion(
                ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])
            )

    def test_socket_timeouts_are_wrapped(self) -> None:
        transport = self._transport(timeout=5)
        with (
            mock.patch(
                "local_voice_harness.llm_transport.urllib.request.urlopen",
                side_effect=TimeoutError("timed out"),
            ),
            self.assertRaisesRegex(HarnessError, "LLM request failed"),
        ):
            transport.chat_completion(
                ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])
            )

    def test_malformed_sse_and_empty_streams_raise(self) -> None:
        error = io.BytesIO(b'data: {"error": "model unavailable"}\n\n')
        with self.assertRaisesRegex(HarnessError, "model unavailable"):
            streamed_message(error, None)

        malformed = io.BytesIO(b"data: []\n\n")
        with self.assertRaisesRegex(HarnessError, "malformed streaming event"):
            streamed_message(malformed, None)

        empty = io.BytesIO(b"data: [DONE]\n\n")
        with self.assertRaisesRegex(HarnessError, "empty streaming response"):
            streamed_message(empty, None)

    def test_first_sentence_emits_before_stream_is_exhausted(self) -> None:
        events = [
            'data: {"choices":[{"delta":{"content":"First sentence. "}}]}\n',
            'data: {"choices":[{"delta":{"content":"Second sentence. "}}]}\n',
            'data: {"choices":[{"delta":{"content":"Trailing fragment"}}]}\n',
            "data: [DONE]\n",
        ]
        remaining_at_first: list[int] = []
        chunks: list[str] = []

        def on_chunk(text: str) -> None:
            chunks.append(text)
            if len(chunks) == 1:
                remaining_at_first.append(len(events))

        def stream() -> Iterator[str]:
            while events:
                yield events.pop(0)

        message = streamed_message(stream(), on_chunk, emit_text_early=True)

        self.assertEqual(
            chunks,
            ["First sentence.", "Second sentence.", "Trailing fragment"],
        )
        self.assertGreater(remaining_at_first[0], 0)
        self.assertEqual(
            message["content"],
            "First sentence. Second sentence. Trailing fragment",
        )

    def test_callback_can_stop_consuming_stream_after_barge_in(self) -> None:
        events = [
            'data: {"choices":[{"delta":{"content":"First sentence. "}}]}\n',
            'data: {"choices":[{"delta":{"content":"Second sentence. "}}]}\n',
            'data: {"choices":[{"delta":{"content":"Never consumed."}}]}\n',
        ]
        chunks: list[str] = []

        def on_chunk(text: str) -> bool:
            chunks.append(text)
            return False

        def stream() -> Iterator[str]:
            while events:
                yield events.pop(0)

        message = streamed_message(stream(), on_chunk, emit_text_early=True)

        self.assertEqual(chunks, ["First sentence."])
        self.assertEqual(message["content"], "First sentence. Second sentence. ")
        self.assertEqual(len(events), 1)

    def test_cancellation_is_checked_for_every_sse_event(self) -> None:
        events = [
            'data: {"choices":[{"delta":{"content":"A long"}}]}\n',
            'data: {"choices":[{"delta":{"content":" unfinished sentence"}}]}\n',
            'data: {"choices":[{"delta":{"content":" keeps going"}}]}\n',
        ]
        checks = iter((False, True))

        def stream() -> Iterator[str]:
            while events:
                yield events.pop(0)

        message = streamed_message(
            stream(),
            mock.Mock(side_effect=AssertionError("no sentence should emit")),
            emit_text_early=True,
            should_cancel=lambda: next(checks),
        )

        self.assertEqual(message["content"], "A long")
        self.assertEqual(len(events), 1)

    def test_tool_capable_stream_defers_speech_until_completion(self) -> None:
        events = [
            'data: {"choices":[{"delta":{"content":"First sentence. "}}]}\n',
            'data: {"choices":[{"delta":{"content":"Second sentence. "}}]}\n',
            "data: [DONE]\n",
        ]
        chunks: list[str] = []
        remaining_at_callback: list[int] = []

        def on_chunk(text: str) -> None:
            chunks.append(text)
            remaining_at_callback.append(len(events))

        def stream() -> Iterator[str]:
            while events:
                yield events.pop(0)

        message = streamed_message(stream(), on_chunk)

        self.assertEqual(chunks, ["First sentence. Second sentence."])
        self.assertEqual(remaining_at_callback, [0])
        self.assertEqual(message["content"], "First sentence. Second sentence. ")

    def test_tool_call_streams_do_not_emit_held_sentences(self) -> None:
        chunks: list[str] = []
        message = streamed_message(
            _stream_response(
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
            ),
            chunks.append,
        )

        self.assertEqual(message["content"], "Speaking already. ")
        self.assertEqual(chunks, [])
        tool_calls = message["tool_calls"]
        self.assertIsInstance(tool_calls, list)
        assert isinstance(tool_calls, list)
        self.assertEqual(len(tool_calls), 1)

    def test_streamed_tool_call_fragments_are_aggregated(self) -> None:
        transport = self._transport(provider="venice", api_key="secret")
        with mock.patch(
            "local_voice_harness.llm_transport.pooled_urlopen",
            return_value=_stream_response(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "cursor", "arguments": '{"task":"fix'},
                        }
                    ]
                },
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": ' it"}'},
                        }
                    ]
                },
            ),
        ):
            message = transport.chat_completion(
                ChatCompletionRequest(messages=[{"role": "user", "content": "fix it"}])
            )

        tool_calls = message["tool_calls"]
        self.assertIsInstance(tool_calls, list)
        assert isinstance(tool_calls, list)
        self.assertEqual(
            tool_calls[0]["function"]["arguments"],
            '{"task":"fix it"}',
        )

    def test_truncated_streamed_tool_arguments_surface_as_malformed(self) -> None:
        transport = self._transport(provider="venice", api_key="secret")
        with mock.patch(
            "local_voice_harness.llm_transport.pooled_urlopen",
            return_value=_stream_response(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-truncated",
                            "function": {
                                "name": "cursor",
                                "arguments": '{"task":"on this issue',
                            },
                        }
                    ]
                }
            ),
        ):
            message = transport.chat_completion(
                ChatCompletionRequest(
                    messages=[{"role": "user", "content": "on this issue please."}]
                )
            )

        tool_calls = message["tool_calls"]
        self.assertIsInstance(tool_calls, list)
        assert isinstance(tool_calls, list)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(str(tool_calls[0]["function"]["arguments"]))

    def test_request_payload_never_includes_authorization(self) -> None:
        transport = self._transport(provider="venice", api_key="secret")
        output = io.StringIO()
        with (
            mock.patch(
                "local_voice_harness.llm_transport.pooled_urlopen",
                return_value=_stream_response({"content": "hello"}),
            ),
            mock.patch("sys.stdout", output),
        ):
            transport.chat_completion(
                ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}]),
                telemetry_round=1,
            )

        records = [json.loads(line) for line in output.getvalue().splitlines()]
        request_record = next(
            record for record in records if record.get("event") == "request"
        )
        self.assertNotIn("secret", request_record["payload"])
        self.assertNotIn("Authorization", request_record["payload"])

    def test_credential_errors_propagate_from_from_settings(self) -> None:
        settings = replace(load_backend_settings({}), llm_provider="venice")
        with (
            mock.patch(
                "local_voice_harness.llm_transport.get_venice_api_key",
                side_effect=CredentialError("missing"),
            ),
            self.assertRaises(CredentialError),
        ):
            LlmTransport.from_settings(settings)

    def test_explicit_stream_override(self) -> None:
        transport = self._transport(provider="local")
        with mock.patch(
            "local_voice_harness.llm_transport.urllib.request.urlopen",
            return_value=_stream_response({"content": "hello"}),
        ) as urlopen:
            transport.chat_completion(
                ChatCompletionRequest(
                    messages=[{"role": "user", "content": "hello"}],
                    stream=True,
                )
            )

        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertTrue(payload["stream"])
        self.assertIsNone(urlopen.call_args.args[0].get_header("Accept"))


if __name__ == "__main__":
    unittest.main()
