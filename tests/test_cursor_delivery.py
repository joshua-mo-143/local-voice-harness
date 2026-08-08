from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from local_voice_harness.cursor.delivery import (
    DeliveryClaim,
    acknowledge_delivery,
    claim_delivery,
    release_delivery,
)
from local_voice_harness.cursor.model import CursorJob
from local_voice_harness.cursor.store import JobStore


class CursorDeliveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = JobStore(self.root / "jobs", self.root / "legacy")
        self.store.create(
            CursorJob.from_dict(
                {
                    "id": "123456789abc",
                    "status": "completed",
                    "request": "test",
                    "result": "done",
                    "created_at": 1,
                    "completed_at": 1,
                    "delivered": False,
                }
            )
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_claim_is_typed_and_synthetic_token_is_not_persisted(self) -> None:
        claim = claim_delivery(self.store, now=100)

        self.assertIsInstance(claim, DeliveryClaim)
        assert claim is not None
        self.assertIsInstance(claim.job, CursorJob)
        persisted = json.loads((self.root / "jobs" / "123456789abc.json").read_text())
        self.assertNotIn("_delivery_token", persisted)
        self.assertEqual(persisted["delivery_claim_token"], claim.token)
        self.assertEqual(persisted["revision"], 1)

    def test_atomic_claim_has_single_winner_and_exact_revisions(self) -> None:
        barrier = threading.Barrier(2)
        claims: list[DeliveryClaim | None] = []

        def claim() -> None:
            barrier.wait()
            claims.append(claim_delivery(self.store, now=100))

        threads = [threading.Thread(target=claim) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        winner = next(item for item in claims if item is not None)
        self.assertEqual(sum(item is not None for item in claims), 1)
        self.assertEqual(self.store.get(winner.job.id).revision, 1)
        self.assertTrue(
            acknowledge_delivery(self.store, winner.job.id, winner.token, now=101)
        )
        self.assertEqual(self.store.get(winner.job.id).revision, 2)

    def test_stale_ack_does_not_write_and_release_preserves_at_least_once(self) -> None:
        claim = claim_delivery(self.store, now=100)
        assert claim is not None

        self.assertFalse(
            acknowledge_delivery(self.store, claim.job.id, "stale", now=101)
        )
        self.assertEqual(self.store.get(claim.job.id).revision, 1)
        self.assertTrue(
            release_delivery(self.store, claim.job.id, claim.token, now=101)
        )
        released = self.store.get(claim.job.id)
        self.assertFalse(released.delivered)
        self.assertEqual(released.revision, 2)
        self.assertIsNone(claim_delivery(self.store, now=105))
        self.assertIsNotNone(claim_delivery(self.store, now=106))


if __name__ == "__main__":
    unittest.main()
