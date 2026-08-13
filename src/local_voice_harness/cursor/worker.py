from __future__ import annotations

import argparse

from ..config import JOBS_DIR, LEGACY_JOBS_DIR
from ..integrations.registry import IntegrationRegistry, build_integration_registry
from ..user_config import load_user_config
from .provisioning import ClientFactories, run_claimed_worker
from .store import JobStore
from .worker_lifecycle import run_worker


def dispatch_waiting_jobs(registry: IntegrationRegistry) -> None:
    """Fill capacity released by this detached worker."""

    from .service import _dispatch_waiting_jobs

    _dispatch_waiting_jobs(integrations=registry)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one persisted Cursor job")
    parser.add_argument("job_id")
    parser.add_argument("--claim")
    parser.add_argument("--claim-operation")
    parser.add_argument("--claim-time", type=float)
    args = parser.parse_args()
    registry = build_integration_registry(load_user_config())
    factories = ClientFactories(
        herdr=registry.herdr_client,
        github=registry.github_client,
        integrations=registry,
    )
    try:
        run_worker(
            JobStore(JOBS_DIR, LEGACY_JOBS_DIR),
            args.job_id,
            args.claim,
            lambda context: run_claimed_worker(context, factories),
            claim_operation=args.claim_operation,
            claim_time=args.claim_time,
        )
    finally:
        dispatch_waiting_jobs(registry)


if __name__ == "__main__":
    main()
