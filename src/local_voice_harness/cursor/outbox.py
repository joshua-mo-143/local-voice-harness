"""Leased execution of durable outbox effects outside state transactions.

Handlers run only after the claim transaction commits. They report an
observation keyed by effect ID and idempotency key; they must not write
canonical job rows. A crash before the submit fence is replayable. A crash
after that fence becomes ``OutcomeUnknown`` unless the handler never marks
dispatch (replay-safe operations).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping

from .coordinator import (
    OUTBOX_RENEW_SECONDS,
    EffectObservation,
    OutboxLease,
    OutboxLeaseLost,
)
from .store import JobStore

EffectHandler = Callable[[OutboxLease, Callable[[], None]], EffectObservation]


class _LeaseRenewer:
    def __init__(self, store: JobStore, lease: OutboxLease) -> None:
        self._store = store
        self._lease = lease
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"outbox-renew-{lease.effect_id[:8]}",
            daemon=True,
        )

    def __enter__(self) -> _LeaseRenewer:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.wait(OUTBOX_RENEW_SECONDS):
            if not self._store.renew_outbox_lease(self._lease):
                return


def _bind_mark(store: JobStore, lease: OutboxLease) -> Callable[[], None]:
    def mark_dispatched() -> None:
        if not store.mark_outbox_dispatched(lease):
            raise OutboxLeaseLost(f"outbox lease lost for effect {lease.effect_id}")

    return mark_dispatched


def drain_outbox(
    store: JobStore,
    handlers: Mapping[str, EffectHandler],
    *,
    now: float | None = None,
    limit: int = 8,
) -> int:
    """Claim due effects, run handlers, and report observations."""

    kinds = tuple(handlers)
    if not kinds:
        return 0
    processed = 0
    seen: set[str] = set()
    while processed < limit:
        lease = store.claim_outbox(kinds, now=now, exclude=tuple(seen))
        if lease is None:
            break
        seen.add(lease.effect_id)
        handler = handlers[lease.kind]
        mark_dispatched = _bind_mark(store, lease)
        try:
            with _LeaseRenewer(store, lease):
                observation = handler(lease, mark_dispatched)
        except Exception as exc:
            error = str(exc)
            if store.outbox_dispatched(lease) or not store.release_outbox_lease(
                lease, error=error, now=now
            ):
                store.observe_outbox(
                    lease,
                    EffectObservation(
                        outcome="OutcomeUnknown",
                        detail={"error": error},
                    ),
                    now=now,
                )
            processed += 1
            continue
        store.observe_outbox(lease, observation, now=now)
        processed += 1
    return processed


def recover_outbox(
    store: JobStore,
    *,
    handlers: Mapping[str, EffectHandler] | None = None,
    now: float | None = None,
) -> int:
    """Reap expired leases, then drain any registered handlers."""

    store.reap_expired_outbox_leases(now=now)
    return drain_outbox(store, handlers or {}, now=now)
