"""Agent-neutral worker entry point.

The current registry contains the Cursor harness; future harness kinds can be
dispatched here without changing the durable worker contract.
"""

from ..cursor.worker import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
