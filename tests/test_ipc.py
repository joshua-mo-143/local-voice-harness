from __future__ import annotations

import socket
import tempfile
import threading
import unittest
from pathlib import Path

from local_voice_harness.ipc import socket_ready, unix_request


def _server(path: Path) -> socket.socket:
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)
    return server


class UnixSocketTests(unittest.TestCase):
    def test_request_sends_all_bytes_and_combines_response_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "service.sock"
            server = _server(path)
            received: list[bytes] = []

            def serve() -> None:
                connection, _ = server.accept()
                with connection:
                    while chunk := connection.recv(16):
                        received.append(chunk)
                    connection.sendall(b"first")
                    connection.sendall(b"-second")

            thread = threading.Thread(target=serve)
            thread.start()
            try:
                response = unix_request(path, b"request body", timeout=1)
            finally:
                thread.join(timeout=1)
                server.close()

        self.assertFalse(thread.is_alive())
        self.assertEqual(b"".join(received), b"request body")
        self.assertEqual(response, b"first-second")

    def test_socket_ready_probes_a_listening_socket(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "service.sock"
            server = _server(path)

            def serve() -> None:
                connection, _ = server.accept()
                with connection:
                    connection.recv(1)

            thread = threading.Thread(target=serve)
            thread.start()
            try:
                self.assertTrue(socket_ready(path))
            finally:
                thread.join(timeout=1)
                server.close()

        self.assertFalse(thread.is_alive())

    def test_socket_ready_rejects_missing_regular_and_stale_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertFalse(socket_ready(root / "missing.sock"))

            regular = root / "regular"
            regular.write_text("not a socket")
            self.assertFalse(socket_ready(regular))

            stale = root / "stale.sock"
            server = _server(stale)
            server.close()
            self.assertFalse(socket_ready(stale))


if __name__ == "__main__":
    unittest.main()
