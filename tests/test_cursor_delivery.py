from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from local_voice_harness.cursor.delivery import (
    DELIVERY_CLAIM_SECONDS,
    DeliveryClaim,
    acknowledge_deferred_delivery,
    acknowledge_delivery,
    acknowledge_desktop_delivery,
    claim_delivery,
    pending_deliveries,
    release_delivery,
    renew_delivery,
)
from local_voice_harness.cursor.model import CursorJob
from local_voice_harness.cursor.store import JobStore
from tests.support import join_threads


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
        persisted = self.store.get("123456789abc").to_dict()
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
        join_threads(threads)

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

    def test_pending_deliveries_claims_only_the_oldest_window(self) -> None:
        self.store.create(
            CursorJob.from_dict(
                {
                    "id": "bbbbbbbbbbbb",
                    "status": "completed",
                    "request": "second",
                    "result": "second",
                    "created_at": 2,
                    "completed_at": 2,
                    "delivered": False,
                }
            )
        )

        claims = pending_deliveries(self.store)

        self.assertEqual([claim.job.id for claim in claims], ["123456789abc"])
        self.assertIsNone(self.store.get("bbbbbbbbbbbb").delivery_claim_token)

    def test_renewal_extends_the_lease_without_incrementing_attempts(self) -> None:
        claim = claim_delivery(self.store, now=100)
        assert claim is not None

        self.assertTrue(renew_delivery(self.store, claim.job.id, claim.token, now=249))

        renewed = self.store.get(claim.job.id)
        self.assertEqual(renewed.delivery_claimed_at, 249)
        self.assertEqual(renewed.delivery_attempts, 1)
        self.assertEqual(renewed.revision, 2)
        self.assertTrue(
            acknowledge_delivery(
                self.store,
                claim.job.id,
                claim.token,
                now=249 + DELIVERY_CLAIM_SECONDS - 1,
            )
        )

    def test_expired_lease_cannot_be_renewed_or_acknowledged(self) -> None:
        claim = claim_delivery(self.store, now=100)
        assert claim is not None
        expires_at = 100 + DELIVERY_CLAIM_SECONDS

        self.assertFalse(
            renew_delivery(self.store, claim.job.id, claim.token, now=expires_at)
        )
        self.assertFalse(
            acknowledge_delivery(self.store, claim.job.id, claim.token, now=expires_at)
        )
        self.assertEqual(self.store.get(claim.job.id).revision, 1)

    def test_expired_lease_rejects_desktop_and_deferred_acknowledgements(
        self,
    ) -> None:
        callbacks = (
            acknowledge_desktop_delivery,
            acknowledge_deferred_delivery,
        )
        for callback in callbacks:
            with self.subTest(callback=callback.__name__):
                temporary = tempfile.TemporaryDirectory()
                self.addCleanup(temporary.cleanup)
                root = Path(temporary.name)
                store = JobStore(root / "jobs", root / "legacy")
                store.create(
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
                claim = claim_delivery(store, now=100)
                assert claim is not None

                self.assertFalse(
                    callback(
                        store,
                        claim.job.id,
                        claim.token,
                        now=100 + DELIVERY_CLAIM_SECONDS,
                    )
                )
                self.assertEqual(store.get(claim.job.id).revision, 1)

    def test_reclaim_replaces_token_and_fences_old_acknowledgement(self) -> None:
        original = claim_delivery(self.store, now=100)
        assert original is not None
        replacement = claim_delivery(
            self.store,
            now=100 + DELIVERY_CLAIM_SECONDS,
        )
        assert replacement is not None

        self.assertNotEqual(original.token, replacement.token)
        revision = self.store.get(original.job.id).revision
        self.assertFalse(
            acknowledge_delivery(
                self.store,
                original.job.id,
                original.token,
                now=401,
            )
        )
        self.assertEqual(self.store.get(original.job.id).revision, revision)
        self.assertTrue(
            acknowledge_delivery(
                self.store,
                replacement.job.id,
                replacement.token,
                now=401,
            )
        )


if __name__ == "__main__":
    unittest.main()
