from __future__ import annotations

import multiprocessing
import socket
import time
import unittest

from tests.support import UnixSocketServer, run_concurrently

CONTEXT = multiprocessing.get_context("fork")


class TestSupportTests(unittest.TestCase):
    def test_concurrent_outcomes_preserve_call_order_and_errors(self) -> None:
        def fail() -> int:
            raise ValueError("expected")

        outcomes = run_concurrently([lambda: 1, fail, lambda: 3])

        self.assertEqual(outcomes.values, (1, None, 3))
        self.assertIsNone(outcomes.errors[0])
        self.assertIsInstance(outcomes.errors[1], ValueError)
        self.assertIsNone(outcomes.errors[2])

    def test_uncooperative_concurrent_call_is_killed_and_reaped(self) -> None:
        children_before = {child.pid for child in multiprocessing.active_children()}

        def hang() -> None:
            while True:
                time.sleep(1)

        started = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "did not finish"):
            run_concurrently([hang], timeout=0.05)

        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(
            {child.pid for child in multiprocessing.active_children()},
            children_before,
        )

    def test_uncooperative_unix_handler_is_killed_and_reaped(self) -> None:
        children_before = {child.pid for child in multiprocessing.active_children()}
        handler_started = CONTEXT.Event()
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        def handle(connection: socket.socket) -> None:
            handler_started.set()
            while True:
                time.sleep(1)

        try:
            started = time.monotonic()
            with UnixSocketServer(handle) as server:
                client.connect(str(server.path))
                self.assertTrue(handler_started.wait(timeout=1))
            self.assertLess(time.monotonic() - started, 1)
            self.assertEqual(
                {child.pid for child in multiprocessing.active_children()},
                children_before,
            )
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
