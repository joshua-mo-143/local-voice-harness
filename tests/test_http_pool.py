from __future__ import annotations

import http.client
import json
import unittest
import urllib.request
from email.message import Message
from unittest import mock

from local_voice_harness import http_pool
from local_voice_harness.llm_transport import (
    ChatCompletionRequest,
    LlmTransport,
    LlmTransportConfig,
)


def _ok_headers(content_type: str = "application/json") -> Message:
    headers = Message()
    headers["Content-Type"] = content_type
    return headers


def _response(
    *,
    body: bytes = b"{}",
    lines: list[bytes] | None = None,
    status: int = 200,
    will_close: bool = False,
) -> mock.Mock:
    response = mock.Mock()
    response.status = status
    response.reason = "OK" if status == 200 else "Error"
    response.headers = _ok_headers()
    response.will_close = will_close
    response.isclosed.return_value = False
    response.read.return_value = body
    if lines is None:
        response.readline.return_value = b""
    else:
        response.readline.side_effect = lines
    return response


class HttpPoolTests(unittest.TestCase):
    def setUp(self) -> None:
        http_pool.clear()
        self.addCleanup(http_pool.clear)

    def test_consecutive_https_calls_reuse_one_connection(self) -> None:
        created: list[mock.Mock] = []

        def https_connection(
            host: str,
            port: int | None = None,
            timeout: object = None,
            **_kwargs: object,
        ) -> mock.Mock:
            connection = mock.Mock()
            created.append(connection)
            connection.host = host
            connection.port = port
            connection.getresponse.return_value = _response(body=b'{"ok":true}')
            return connection

        with mock.patch("http.client.HTTPSConnection", side_effect=https_connection):
            request = urllib.request.Request(
                "https://api.venice.ai/api/v1/chat/completions",
                data=b"{}",
                headers={
                    "Authorization": "Bearer secret",
                    "Content-Type": "application/json",
                },
            )
            with http_pool.urlopen(request, timeout=5) as first:
                first.read()
            with http_pool.urlopen(request, timeout=5) as second:
                second.read()

        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].request.call_count, 2)
        first_headers = created[0].request.call_args_list[0].args[3]
        self.assertEqual(first_headers["Connection"], "keep-alive")
        self.assertEqual(first_headers["Authorization"], "Bearer secret")

    def test_same_process_requests_to_same_host_share_connection(self) -> None:
        created: list[mock.Mock] = []

        def https_connection(
            host: str,
            port: int | None = None,
            timeout: object = None,
            **_kwargs: object,
        ) -> mock.Mock:
            connection = mock.Mock()
            created.append(connection)
            connection.getresponse.side_effect = [
                _response(body=b'{"choices":[{"message":{"content":"hi"}}]}'),
                _response(body=b"RIFF", status=200),
            ]
            return connection

        with mock.patch("http.client.HTTPSConnection", side_effect=https_connection):
            llm = urllib.request.Request(
                "https://api.venice.ai/api/v1/chat/completions",
                data=b"{}",
                headers={"Content-Type": "application/json"},
            )
            tts = urllib.request.Request(
                "https://api.venice.ai/api/v1/audio/speech",
                data=b"{}",
                headers={"Content-Type": "application/json"},
            )
            with http_pool.urlopen(llm, timeout=5) as first:
                first.read()
            with http_pool.urlopen(tts, timeout=11) as second:
                second.read()

        self.assertEqual(len(created), 1)
        selectors = [call.args[1] for call in created[0].request.call_args_list]
        self.assertEqual(
            selectors,
            ["/api/v1/chat/completions", "/api/v1/audio/speech"],
        )

    def test_streamed_sse_yields_lines_without_buffering_the_body(self) -> None:
        lines = [
            b'data: {"choices":[{"delta":{"content":"Hi."}}]}\n',
            b"data: [DONE]\n",
            b"",
        ]
        response = _response(lines=lines, body=b"unused")
        connection = mock.Mock()
        connection.getresponse.return_value = response

        with mock.patch("http.client.HTTPSConnection", return_value=connection):
            request = urllib.request.Request(
                "https://api.venice.ai/api/v1/chat/completions",
                data=b"{}",
                headers={"Accept": "text/event-stream"},
            )
            with http_pool.urlopen(request, timeout=5) as stream:
                received = [line for line in stream]
                response.read.assert_not_called()

        self.assertEqual(received, lines[:2])
        response.read.assert_not_called()
        response.close.assert_called_once()

    def test_llm_transport_reuses_the_process_pool(self) -> None:
        created: list[mock.Mock] = []
        payload = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

        def https_connection(
            host: str,
            port: int | None = None,
            timeout: object = None,
            **_kwargs: object,
        ) -> mock.Mock:
            connection = mock.Mock()
            created.append(connection)
            connection.getresponse.return_value = _response(body=payload)
            return connection

        transport = LlmTransport(
            LlmTransportConfig(
                provider="venice",
                model="test-model",
                endpoint="https://api.venice.ai/api/v1/chat/completions",
                timeout=9,
                api_key="venice-secret",
            )
        )
        with mock.patch("http.client.HTTPSConnection", side_effect=https_connection):
            transport.chat_completion(
                ChatCompletionRequest(
                    messages=[{"role": "user", "content": "one"}],
                    stream=False,
                )
            )
            transport.chat_completion(
                ChatCompletionRequest(
                    messages=[{"role": "user", "content": "two"}],
                    stream=False,
                )
            )

        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].request.call_count, 2)

    def test_completed_sse_response_returns_connection_to_pool(self) -> None:
        created: list[mock.Mock] = []

        def https_connection(
            host: str,
            port: int | None = None,
            timeout: object = None,
            **_kwargs: object,
        ) -> mock.Mock:
            connection = mock.Mock()
            connection.sock = None
            created.append(connection)
            connection.getresponse.side_effect = [
                _response(
                    lines=[
                        b'data: {"choices":[{"delta":{"content":"one"}}]}\n',
                        b"data: [DONE]\n",
                        b"",
                    ]
                ),
                _response(
                    lines=[
                        b'data: {"choices":[{"delta":{"content":"two"}}]}\n',
                        b"data: [DONE]\n",
                        b"",
                    ]
                ),
            ]
            return connection

        transport = LlmTransport(
            LlmTransportConfig(
                provider="venice",
                model="test-model",
                endpoint="https://api.venice.ai/api/v1/chat/completions",
                timeout=9,
                api_key="venice-secret",
            )
        )
        with mock.patch("http.client.HTTPSConnection", side_effect=https_connection):
            first = transport.chat_completion(
                ChatCompletionRequest(messages=[], stream=True)
            )
            second = transport.chat_completion(
                ChatCompletionRequest(messages=[], stream=True)
            )

        self.assertEqual(first["content"], "one")
        self.assertEqual(second["content"], "two")
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].request.call_count, 2)

    def test_https_proxy_uses_tunnel_without_forwarding_credentials(self) -> None:
        connection = mock.Mock()
        connection.sock = None
        connection.getresponse.return_value = _response()
        request = urllib.request.Request(
            "https://api.venice.ai/api/v1/chat/completions",
            data=b"{}",
            headers={"Proxy-Authorization": "Basic secret"},
        )
        request.set_proxy("proxy.example:8443", "https")

        with mock.patch(
            "http.client.HTTPSConnection", return_value=connection
        ) as https_connection:
            with http_pool.PooledHTTPSHandler().https_open(request) as response:
                response.read()

        https_connection.assert_called_once_with("proxy.example:8443", timeout=None)
        connection.set_tunnel.assert_called_once_with(
            "api.venice.ai",
            headers={"Proxy-Authorization": "Basic secret"},
        )
        sent_headers = connection.request.call_args.args[3]
        self.assertNotIn("Proxy-Authorization", sent_headers)

    def test_reused_post_is_not_replayed_after_disconnect(self) -> None:
        connection = mock.Mock()
        connection.sock = None
        connection.request.side_effect = http.client.RemoteDisconnected("closed")
        key = ("https", "api.venice.ai", "")
        http_pool._release(key, connection)
        request = urllib.request.Request(
            "https://api.venice.ai/api/v1/chat/completions",
            data=b"{}",
        )

        with self.assertRaises(http.client.RemoteDisconnected):
            http_pool.PooledHTTPSHandler().https_open(request)

        connection.request.assert_called_once()
        connection.close.assert_called_once()

    def test_partial_response_close_discards_connection_without_draining(self) -> None:
        response = _response(body=b"x" * 20)
        connection = mock.Mock()
        connection.sock = None
        connection.getresponse.return_value = response
        request = urllib.request.Request("https://api.venice.ai/stream")

        with mock.patch("http.client.HTTPSConnection", return_value=connection):
            opened = http_pool.PooledHTTPSHandler().https_open(request)
            opened.read(10)
            opened.close()

        connection.close.assert_called_once()
        response.read.assert_called_once_with(10)

    def test_reused_connection_updates_socket_timeout(self) -> None:
        connection = mock.Mock()
        socket = mock.Mock()
        connection.sock = socket
        key = ("https", "api.venice.ai", "")
        http_pool._release(key, connection)

        checked_out, reused = http_pool._checkout(
            key,
            http.client.HTTPSConnection,
            3,
        )

        self.assertTrue(reused)
        self.assertIs(checked_out, connection)
        self.assertEqual(connection.timeout, 3)
        socket.settimeout.assert_called_once_with(3)
