from __future__ import annotations

import sys
from collections.abc import Sequence

COMMANDS = ("begin", "toggle", "end", "cancel", "transcribe", "vad")


def _usage() -> str:
    return f"usage: voice-harness-dictate {{{','.join(COMMANDS)}}}"


def _parse_command(arguments: Sequence[str]) -> str:
    if len(arguments) == 1 and arguments[0] in {"-h", "--help"}:
        print(_usage())
        raise SystemExit(0)
    if len(arguments) != 1 or arguments[0] not in COMMANDS:
        print(_usage(), file=sys.stderr)
        raise SystemExit(2)
    return arguments[0]


def _run(command: str) -> None:
    from .dictation import run
    from .user_config import load_user_config

    run(command, load_user_config())


def main(arguments: Sequence[str] | None = None) -> None:
    command = _parse_command(sys.argv[1:] if arguments is None else arguments)
    try:
        _run(command)
    except Exception as exc:
        from .diagnostic_safety import COMMAND_FAILURE, log_diagnostic
        from .notifications import notify

        log_diagnostic(
            "dictation_cli", "command_failed", f"{type(exc).__name__}: {exc}"
        )
        print(f"voice-harness-dictate: {COMMAND_FAILURE}", file=sys.stderr)
        notify(COMMAND_FAILURE, error=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
