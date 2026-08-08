from __future__ import annotations

import argparse
from pathlib import Path

from .app import respond, status
from .dictation import run as run_dictation
from .notifications import notify
from .recording import (
    cancel_recording,
    handoff_recording,
    retry_generation,
    start_recording,
    stop_recording,
)
from .service_manager import (
    audit_services,
    install_services,
    logs,
    restart_services,
    start_services,
    stop_services,
    uninstall_services,
)
from .service_manager import (
    status as service_status,
)
from .stt.client import transcribe


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Local voice agent harness")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("begin")
    commands.add_parser("end")
    commands.add_parser("cancel")
    transcribe_audio = commands.add_parser("transcribe")
    transcribe_audio.add_argument("--generation", type=Path)
    dictate = commands.add_parser(
        "dictate", help="record and type transcription into the focused window"
    )
    dictate_commands = dictate.add_subparsers(dest="dictation_command", required=True)
    for command in ("begin", "end", "toggle", "transcribe", "cancel"):
        dictate_commands.add_parser(command)
    text = commands.add_parser("text")
    text.add_argument("text", nargs="+")
    commands.add_parser("status")

    services = commands.add_parser("services")
    service_commands = services.add_subparsers(dest="service_command", required=True)
    install = service_commands.add_parser("install")
    install.add_argument("--force", action="store_true")
    install.add_argument("--replace-dictation", action="store_true")
    service_commands.add_parser("start")
    stop = service_commands.add_parser("stop")
    stop.add_argument("--include-herdr", action="store_true")
    restart = service_commands.add_parser("restart")
    restart.add_argument("--include-herdr", action="store_true")
    service_commands.add_parser("status")
    service_commands.add_parser(
        "audit", help="read and validate effective installed units and drop-ins"
    )
    service_logs = service_commands.add_parser("logs")
    service_logs.add_argument("-f", "--follow", action="store_true")
    service_logs.add_argument("-n", "--lines", type=int, default=100)
    uninstall = service_commands.add_parser("uninstall")
    uninstall.add_argument("--include-herdr", action="store_true")

    return root


def dispatch(args: argparse.Namespace) -> None:
    if args.command == "begin":
        start_recording()
    elif args.command == "end":
        audio_path = stop_recording()
        respond(transcribe(audio_path))
    elif args.command == "cancel":
        cancel_recording()
    elif args.command == "transcribe":
        audio_path = (
            retry_generation(args.generation)
            if args.generation is not None
            else handoff_recording()
        )
        respond(transcribe(audio_path))
    elif args.command == "dictate":
        run_dictation(args.dictation_command)
    elif args.command == "text":
        respond(" ".join(args.text))
    elif args.command == "status":
        status()
    elif args.service_command == "install":
        install_services(force=args.force, replace_dictation=args.replace_dictation)
    elif args.service_command == "start":
        start_services()
    elif args.service_command == "stop":
        stop_services(include_herdr=args.include_herdr)
    elif args.service_command == "restart":
        restart_services(include_herdr=args.include_herdr)
    elif args.service_command == "status":
        service_status()
    elif args.service_command == "audit":
        raise SystemExit(audit_services())
    elif args.service_command == "logs":
        logs(follow=args.follow, lines=max(args.lines, 1))
    else:
        uninstall_services(include_herdr=args.include_herdr)


def main() -> None:
    try:
        dispatch(parser().parse_args())
    except Exception as exc:
        message = str(exc) or type(exc).__name__
        print(f"voice-harness: {message}", file=__import__("sys").stderr)
        notify(message, error=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
