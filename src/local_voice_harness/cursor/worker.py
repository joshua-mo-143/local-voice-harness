from __future__ import annotations

import argparse

from ..config import JOBS_DIR, LEGACY_JOBS_DIR
from .provisioning import run_claimed_worker
from .store import JobStore
from .worker_lifecycle import run_worker


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one persisted Cursor job")
    parser.add_argument("job_id")
    parser.add_argument("--claim")
    args = parser.parse_args()
    run_worker(
        JobStore(JOBS_DIR, LEGACY_JOBS_DIR),
        args.job_id,
        args.claim,
        run_claimed_worker,
    )


if __name__ == "__main__":
    main()
