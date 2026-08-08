from __future__ import annotations

import socket
import time
import unittest

from local_voice_harness.ipc import unix_request
from tests.support import UnixSocketServer, receive_all


class UnixRequestTransportTests(unittest.TestCase):
    """Exercise transport behavior only; fake handlers do not validate protocols."""

    def test_transports_arbitrary_request_and_response_bytes_without_rewriting(
        self,
    ) -> None:
        def handle(connection: socket.socket) -> None:
            request = receive_all(connection)
            connection.sendall(b"\x00response:" + request)

        with UnixSocketServer(handle) as server:
            response = unix_request(server.path, b"\xffrequest\x00", timeout=1)

        self.assertEqual(response, b"\x00response:\xffrequest\x00")

    def test_large_fragmented_response_is_not_truncated(self) -> None:
        response = b"x" * (2 * 64 * 1024 + 17)

        def handle(connection: socket.socket) -> None:
            self.assertEqual(receive_all(connection), b"request\n")
            for offset in range(0, len(response), 997):
                connection.sendall(response[offset : offset + 997])

        with UnixSocketServer(handle) as server:
            actual = unix_request(server.path, b"request\n", timeout=2)

        self.assertEqual(actual, response)

    def test_slow_response_obeys_client_timeout(self) -> None:
        def handle(connection: socket.socket) -> None:
            receive_all(connection)
            time.sleep(0.1)

        with UnixSocketServer(handle) as server:
            with self.assertRaises(TimeoutError):
                unix_request(server.path, b"request\n", timeout=0.02)

    def test_eof_returns_bytes_received_before_disconnect(self) -> None:
        def handle(connection: socket.socket) -> None:
            receive_all(connection)
            connection.sendall(b"partial")

        with UnixSocketServer(handle) as server:
            response = unix_request(server.path, b"request\n", timeout=1)

        self.assertEqual(response, b"partial")


if __name__ == "__main__":
    unittest.main()
