from __future__ import annotations

import argparse

from .jobs import run_worker


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one persisted Cursor job")
    parser.add_argument("job_id")
    parser.add_argument("--claim")
    args = parser.parse_args()
    run_worker(args.job_id, args.claim)


if __name__ == "__main__":
    main()
