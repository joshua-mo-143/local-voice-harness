from __future__ import annotations

import argparse
from pathlib import Path

from .app import respond, status
from .cursor.service import CursorTurnRequest, cursor_turn
from .diagnostics import doctor
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
from .vocabulary import (
    add_alias,
    add_replacement,
    export_entries,
    import_entries,
    list_entries,
    remove_alias,
    remove_replacement,
)


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

    jobs = commands.add_parser("jobs", help="manage background Cursor jobs")
    job_commands = jobs.add_subparsers(dest="jobs_command", required=True)
    job_commands.add_parser("list", help="summarize the Cursor job inbox")
    job_status = job_commands.add_parser("status", help="report a job's status")
    job_status.add_argument("reference", nargs="*")
    for name, help_text in (
        ("cancel", "cancel a job"),
        ("dismiss", "dismiss a job announcement"),
        ("repeat", "repeat a job announcement"),
    ):
        job_action = job_commands.add_parser(name, help=help_text)
        job_action.add_argument("reference", nargs="+")
    job_reply = job_commands.add_parser("reply", help="answer a job clarification")
    job_reply.add_argument("--job", "-j", help="target job id")
    job_reply.add_argument("message", nargs="+")

    doctor_command = commands.add_parser(
        "doctor",
        help="diagnose harness health and suggest safe recovery steps",
    )
    doctor_command.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit machine-readable diagnostics instead of the human summary",
    )
    doctor_command.add_argument(
        "--fix",
        action="store_true",
        help="offer confirmation-gated repairs for issues that support them",
    )

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

    _add_vocabulary_parser(commands)

    return root


def run_job_command(args: argparse.Namespace) -> None:
    if args.jobs_command == "list":
        request = CursorTurnRequest("", action="list")
    elif args.jobs_command == "reply":
        message = " ".join(args.message)
        request = CursorTurnRequest(
            message,
            action="reply",
            job_id=args.job,
            reference=message,
        )
    else:
        reference = " ".join(args.reference)
        request = CursorTurnRequest(
            reference,
            action=args.jobs_command,
            reference=reference,
        )
    print(cursor_turn(request).text)


def _add_vocabulary_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    vocabulary = commands.add_parser(
        "vocabulary", help="manage local STT corrections and entity aliases"
    )
    vocabulary_commands = vocabulary.add_subparsers(
        dest="vocabulary_command", required=True
    )

    listing = vocabulary_commands.add_parser("list", help="show stored entries")
    listing.add_argument("--kind", choices=("replacement", "alias"))

    add = vocabulary_commands.add_parser("add", help="add or update an entry")
    add_kinds = add.add_subparsers(dest="vocabulary_kind", required=True)
    add_replacement_parser = add_kinds.add_parser(
        "replacement", help="add an STT text correction"
    )
    add_replacement_parser.add_argument("spoken")
    add_replacement_parser.add_argument("written")
    add_replacement_parser.add_argument("--force", action="store_true")
    add_alias_parser = add_kinds.add_parser(
        "alias", help="map a spoken phrase to owner/repo or owner/repo#number"
    )
    add_alias_parser.add_argument("phrase")
    add_alias_parser.add_argument("target")
    add_alias_parser.add_argument("--force", action="store_true")

    remove = vocabulary_commands.add_parser("remove", help="delete an entry")
    remove_kinds = remove.add_subparsers(dest="vocabulary_kind", required=True)
    remove_replacement_parser = remove_kinds.add_parser("replacement")
    remove_replacement_parser.add_argument("spoken")
    remove_alias_parser = remove_kinds.add_parser("alias")
    remove_alias_parser.add_argument("phrase")

    export = vocabulary_commands.add_parser(
        "export", help="print or back up the store as JSON"
    )
    export.add_argument("--output", type=Path)

    importer = vocabulary_commands.add_parser(
        "import", help="merge or replace the store from a JSON backup"
    )
    importer.add_argument("path", type=Path)
    importer.add_argument("--replace", action="store_true")


def _dispatch_vocabulary(args: argparse.Namespace) -> None:
    if args.vocabulary_command == "list":
        list_entries(args.kind)
    elif args.vocabulary_command == "add" and args.vocabulary_kind == "replacement":
        add_replacement(args.spoken, args.written, force=args.force)
    elif args.vocabulary_command == "add":
        add_alias(args.phrase, args.target, force=args.force)
    elif args.vocabulary_command == "remove" and args.vocabulary_kind == "replacement":
        remove_replacement(args.spoken)
    elif args.vocabulary_command == "remove":
        remove_alias(args.phrase)
    elif args.vocabulary_command == "export":
        export_entries(args.output)
    else:
        import_entries(args.path, replace=args.replace)


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
    elif args.command == "jobs":
        run_job_command(args)
    elif args.command == "doctor":
        raise SystemExit(doctor(json_output=args.json_output, fix=args.fix))
    elif args.command == "vocabulary":
        _dispatch_vocabulary(args)
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
