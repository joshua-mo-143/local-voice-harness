from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path

from .agents.service import AgentTurnRequest as CursorTurnRequest
from .agents.service import (
    acknowledge_quarantine_reservations,
    count_jobs,
    list_quarantine_evidence,
    nuke_jobs,
)
from .agents.service import agent_turn as cursor_turn
from .agents.store import QuarantineEvidence
from .app import respond, status
from .browser_context import RequestContext, request_context
from .config import CURSOR_PATTERN, REPLAY_DIR
from .config_management import (
    commit_config_change,
    format_restart_notice,
    list_integrations,
    reset_config,
    run_integration_doctor,
    run_setup,
    set_integration_enabled,
    show_config,
)
from .credentials import (
    delete_venice_api_key,
    get_venice_api_key,
    store_venice_api_key,
)
from .diagnostic_safety import COMMAND_FAILURE, log_diagnostic
from .diagnostics import doctor
from .dictation import run as run_dictation
from .integrations.registry import build_integration_registry
from .intent import Intent, IntentRoute, route_intent
from .notifications import notify
from .recording import (
    cancel_recording,
    handoff_recording,
    retry_generation,
    start_recording,
    stop_recording,
)
from .replay import (
    capture_bundle,
    default_bundle_path,
    load_bundle,
    manifest_summary,
    run_replay,
    save_bundle,
)
from .responses import AssistantResponse, as_assistant_response
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
from .user_config import (
    PlanApprovalMode,
    load_plan_approval_preferences,
    load_user_config,
    set_plan_approval_mode,
)
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
    commands.add_parser(
        "listen", help="start a conversation now without saying the wake word"
    )
    transcribe_audio = commands.add_parser("transcribe")
    transcribe_audio.add_argument("--generation", type=Path)
    dictate = commands.add_parser(
        "dictate", help="record and type transcription into the focused window"
    )
    dictate_commands = dictate.add_subparsers(dest="dictation_command", required=True)
    for command in ("begin", "end", "toggle", "vad", "transcribe", "cancel"):
        dictate_commands.add_parser(command)
    text = commands.add_parser("text")
    text.add_argument("text", nargs="+")
    commands.add_parser("status")
    setup = commands.add_parser(
        "setup",
        help="interactive first-run configuration for providers, integrations, and audio",
    )
    setup.add_argument(
        "--defaults",
        action="store_true",
        help="write recommended defaults without prompting",
    )

    config = commands.add_parser(
        "config", help="inspect or update unified configuration"
    )
    config_commands = config.add_subparsers(dest="config_command", required=True)
    config_show = config_commands.add_parser("show", help="print stored configuration")
    config_show.add_argument("key", nargs="?", help="optional dotted configuration key")
    config_show.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit machine-readable output",
    )
    config_set = config_commands.add_parser(
        "set", help="validate and persist one setting"
    )
    config_set.add_argument("key", help="dotted configuration key")
    config_set.add_argument("value", help="new value")
    config_reset = config_commands.add_parser(
        "reset", help="restore defaults for one section or the whole file"
    )
    config_reset.add_argument(
        "--section",
        choices=("providers", "integrations", "compute", "audio", "platform"),
        help="reset only one section",
    )

    integrations = commands.add_parser(
        "integrations", help="manage optional context and routing integrations"
    )
    integration_commands = integrations.add_subparsers(
        dest="integration_command", required=True
    )
    integration_list = integration_commands.add_parser(
        "list", help="show which integrations are enabled"
    )
    integration_list.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit machine-readable output",
    )
    integration_enable = integration_commands.add_parser(
        "enable", help="enable one integration"
    )
    integration_enable.add_argument("name")
    integration_disable = integration_commands.add_parser(
        "disable", help="disable one integration"
    )
    integration_disable.add_argument("name")
    integration_doctor = integration_commands.add_parser(
        "doctor", help="inspect enabled integrations only"
    )
    integration_doctor.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit machine-readable diagnostics",
    )

    plan_approval = commands.add_parser(
        "plan-approval",
        help="inspect or disable automatic Cursor plan approval",
    )
    plan_approval.add_argument(
        "plan_approval_command",
        choices=("status", "ask"),
        help="show the current mode or require explicit approval",
    )

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
    job_quarantine = job_commands.add_parser(
        "quarantine", help="inspect and resolve quarantined job records"
    )
    quarantine_commands = job_quarantine.add_subparsers(
        dest="quarantine_command", required=True
    )
    quarantine_list = quarantine_commands.add_parser(
        "list", help="show unresolved quarantine evidence"
    )
    quarantine_list.add_argument(
        "--all",
        action="store_true",
        help="include previously acknowledged evidence",
    )
    quarantine_list.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit machine-readable evidence",
    )
    quarantine_acknowledge = quarantine_commands.add_parser(
        "acknowledge",
        help="release reservations after manually inspecting one job's evidence",
    )
    quarantine_acknowledge.add_argument(
        "job_id", help="12-character quarantined job ID"
    )
    quarantine_acknowledge.add_argument(
        "--reason",
        required=True,
        help="operator verification recorded in the resolution tombstone",
    )
    quarantine_acknowledge.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="acknowledge without the confirmation prompt",
    )
    job_nuke = job_commands.add_parser(
        "nuke", help="permanently delete ALL Cursor jobs"
    )
    job_nuke.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="delete without the confirmation prompt",
    )

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

    credentials = commands.add_parser(
        "credentials", help="manage credentials in the desktop Secret Service"
    )
    credential_commands = credentials.add_subparsers(
        dest="credential_command", required=True
    )
    credential_commands.add_parser("set", help="securely store the Venice API key")
    credential_commands.add_parser(
        "status", help="check whether the Venice API key is stored"
    )
    credential_commands.add_parser(
        "delete", help="delete the Venice API key from Secret Service"
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
    _add_replay_parser(commands)

    return root


def run_job_command(args: argparse.Namespace) -> None:
    if args.jobs_command == "quarantine":
        _run_job_quarantine(args)
        return
    if args.jobs_command == "nuke":
        _run_job_nuke(force=args.force)
        return
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
    response = as_assistant_response(cursor_turn(request).text)
    print(response.display_text)


def _print_quarantine_evidence(record: QuarantineEvidence) -> None:
    identity = record.job_id or f"unknown ({record.metadata_path.name})"
    state = "acknowledged" if record.resolved else "unresolved"
    print(f"{identity}: {state}")
    print(f"  metadata: {record.metadata_path}")
    if record.payload_path is not None:
        print(f"  payload: {record.payload_path}")
    print(f"  quarantine error: {record.quarantine_error}")
    if record.status is not None:
        print(f"  recorded status: {record.status}")
    if record.worker_pid is not None:
        print(
            "  recorded worker: "
            f"pid={record.worker_pid} "
            f"boot={record.worker_boot_id or 'unknown'} "
            f"start={record.worker_process_start or 'unknown'}"
        )
    if record.herdr_target is not None:
        print(f"  recorded Herdr target: {record.herdr_target}")
    if record.worktree_path is not None:
        print(f"  recorded worktree: {record.worktree_path}")
    if record.inspection_error is not None:
        print(f"  inspection error: {record.inspection_error}")


def _run_job_quarantine(args: argparse.Namespace) -> None:
    if args.quarantine_command == "list":
        evidence = list_quarantine_evidence(include_resolved=args.all)
        if args.json_output:
            print(
                json.dumps(
                    [record.to_dict() for record in evidence],
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        if not evidence:
            qualifier = "" if args.all else " unresolved"
            print(f"There are no{qualifier} quarantined job records.")
            return
        for index, record in enumerate(evidence):
            if index:
                print()
            _print_quarantine_evidence(record)
        return

    evidence = [
        record
        for record in list_quarantine_evidence(include_resolved=True)
        if record.job_id == args.job_id
    ]
    unresolved = [record for record in evidence if not record.resolved]
    if unresolved and not args.force:
        print(
            "WARNING: acknowledging quarantine evidence releases any target and "
            "worktree reservation fences recorded for this job."
        )
        for record in unresolved:
            _print_quarantine_evidence(record)
        confirmation = input("Type 'acknowledge' to confirm: ").strip().casefold()
        if confirmation != "acknowledge":
            print("Aborted. Quarantine reservations were not acknowledged.")
            return
    print(
        acknowledge_quarantine_reservations(
            args.job_id,
            reason=args.reason,
        )
    )


def _run_job_nuke(*, force: bool) -> None:
    total = count_jobs()
    if total == 0:
        unresolved = list_quarantine_evidence()
        if unresolved:
            print(
                "There are no live Cursor jobs to delete, but unresolved quarantine "
                "evidence remains. Inspect it with "
                "'voice-harness jobs quarantine list'."
            )
            return
        print("There are no Cursor jobs to delete.")
        return
    if not force:
        noun = "job" if total == 1 else "jobs"
        print(
            f"WARNING: this permanently deletes all {total} Cursor {noun}, "
            "including any that are still running. This cannot be undone."
        )
        confirmation = input("Type 'delete' to confirm: ").strip().casefold()
        if confirmation != "delete":
            print("Aborted. No Cursor jobs were deleted.")
            return
    print(nuke_jobs())


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


def _add_replay_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    replay = commands.add_parser(
        "replay", help="capture and run side-effect-free semantic voice replays"
    )
    replay_commands = replay.add_subparsers(dest="replay_command", required=True)

    capture = replay_commands.add_parser(
        "capture", help="capture bounded semantic inputs and decisions"
    )
    capture.add_argument("text", nargs="+", help="raw transcript to capture")
    capture.add_argument("--output", type=Path)
    capture.add_argument(
        "--without-context",
        action="store_true",
        help="record an explicit empty context decision",
    )
    capture.add_argument("--spoken-response")
    capture.add_argument("--display-response")
    capture.add_argument(
        "--intent",
        choices=sorted({intent.value for intent in Intent}),
        help="inject an observed router decision instead of calling the router",
    )
    capture.add_argument(
        "--confidence",
        choices=("high", "medium", "low"),
        help="confidence paired with --intent",
    )

    for name in ("run", "inspect"):
        command = replay_commands.add_parser(name)
        command.add_argument("path", type=Path)

    export = replay_commands.add_parser(
        "export", help="review a summary before copying a portable bundle"
    )
    export.add_argument("path", type=Path)
    export.add_argument("output", type=Path)

    promote = replay_commands.add_parser(
        "promote", help="manually review and copy a bundle as a test fixture"
    )
    promote.add_argument("path", type=Path)
    promote.add_argument("output", type=Path)


def _configured_request_context(text: str) -> RequestContext:
    settings = load_user_config()
    return request_context(
        text,
        platform=settings.platform,
        integrations=build_integration_registry(settings),
    )


def _capture_replay(args: argparse.Namespace) -> None:
    from .stt.server import transcript_replacements
    from .transcript import normalize_transcript

    raw = " ".join(args.text).strip()
    replacements = transcript_replacements()
    normalized = normalize_transcript(raw, replacements)
    context = (
        RequestContext(normalized)
        if args.without_context
        else _configured_request_context(normalized)
    )
    if (args.intent is None) != (args.confidence is None):
        raise ValueError("--intent and --confidence must be provided together")
    if args.intent is not None:
        route = IntentRoute(Intent(args.intent), args.confidence)
    elif CURSOR_PATTERN.search(normalized):
        route = IntentRoute(Intent.AGENT_SUBMIT, "high")
    else:
        route = route_intent(normalized, context)
    if args.display_response is not None and args.spoken_response is None:
        raise ValueError("--display-response requires --spoken-response")
    response = (
        AssistantResponse(
            args.spoken_response,
            args.display_response or args.spoken_response,
        )
        if args.spoken_response is not None
        else None
    )
    bundle = capture_bundle(
        raw,
        replacements=replacements,
        context=context,
        route=route,
        response=response,
    )
    output = args.output or default_bundle_path(REPLAY_DIR)
    print(manifest_summary(bundle))
    save_bundle(bundle, output)
    print(f"Captured replay: {output}")


def _copy_replay_after_review(
    source: Path,
    output: Path,
    *,
    confirmation: str,
    purpose: str,
    show_contents: bool = False,
) -> None:
    bundle = load_bundle(source)
    print(manifest_summary(bundle))
    if show_contents:
        print("Complete bounded bundle for manual content review:")
        print(
            json.dumps(bundle.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
        )
    answer = input(
        f"Type '{confirmation}' after reviewing the displayed replay: "
    ).strip()
    if answer != confirmation:
        print(f"Aborted. Replay was not {purpose}.")
        return
    save_bundle(bundle, output)
    print(f"Replay {purpose}: {output}")


def _dispatch_replay(args: argparse.Namespace) -> None:
    if args.replay_command == "capture":
        _capture_replay(args)
        return
    if args.replay_command == "export":
        _copy_replay_after_review(
            args.path, args.output, confirmation="export", purpose="exported"
        )
        return
    if args.replay_command == "promote":
        _copy_replay_after_review(
            args.path,
            args.output,
            confirmation="reviewed",
            purpose="promoted",
            show_contents=True,
        )
        return
    bundle = load_bundle(args.path)
    print(manifest_summary(bundle))
    if args.replay_command == "inspect":
        return
    response = run_replay(bundle)
    print("Verified deterministic stages: transcript normalization, ticket extraction.")
    print("Injected recorded decisions: context selection, intent routing.")
    if response is not None:
        print("Injected recorded response rendering; no TTS was invoked.")
        print(f"Display: {response.display_text}")
        print(f"Speech: {response.spoken_text}")


def dispatch(args: argparse.Namespace) -> None:
    if args.command == "begin":
        start_recording(load_user_config().audio)
    elif args.command == "end":
        user_config = load_user_config()
        audio_path = stop_recording()
        respond(transcribe(audio_path), user_config=user_config)
    elif args.command == "cancel":
        cancel_recording()
    elif args.command == "listen":
        from .wake.daemon import request_listen

        request_listen()
        print("listening")
    elif args.command == "transcribe":
        user_config = load_user_config()
        audio_path = (
            retry_generation(args.generation)
            if args.generation is not None
            else handoff_recording()
        )
        respond(transcribe(audio_path), user_config=user_config)
    elif args.command == "dictate":
        run_dictation(args.dictation_command, load_user_config())
    elif args.command == "text":
        user_config = load_user_config()
        respond(" ".join(args.text), user_config=user_config)
    elif args.command == "status":
        status()
    elif args.command == "setup":
        run_setup(defaults_only=args.defaults)
    elif args.command == "config":
        if args.config_command == "show":
            print(show_config(key=args.key, json_output=args.json_output))
        elif args.config_command == "set":
            result = commit_config_change({args.key: args.value})
            print(format_restart_notice(result.restart_services))
        else:
            result = reset_config(section=args.section)
            print(format_restart_notice(result.restart_services))
    elif args.command == "integrations":
        if args.integration_command == "list":
            print(list_integrations(json_output=args.json_output))
        elif args.integration_command == "enable":
            result = set_integration_enabled(args.name, enabled=True)
            print(format_restart_notice(result.restart_services))
        elif args.integration_command == "disable":
            result = set_integration_enabled(args.name, enabled=False)
            print(format_restart_notice(result.restart_services))
        else:
            exit_code, output = run_integration_doctor(json_output=args.json_output)
            print(output)
            if exit_code:
                raise SystemExit(exit_code)
    elif args.command == "plan-approval":
        preferences = (
            set_plan_approval_mode(PlanApprovalMode.ASK)
            if args.plan_approval_command == "ask"
            else load_plan_approval_preferences()
        )
        print(
            f"Cursor plan approval: {preferences.mode.value} "
            f"({preferences.explicit_approval_count}/3 explicit approvals)"
        )
    elif args.command == "jobs":
        run_job_command(args)
    elif args.command == "doctor":
        raise SystemExit(doctor(json_output=args.json_output, fix=args.fix))
    elif args.command == "credentials":
        if args.credential_command == "set":
            key = getpass.getpass(
                "Venice API key (input hidden; paste then press Enter): "
            )
            print("Storing Venice API key…", flush=True)
            store_venice_api_key(key)
            print("Venice API key stored in the desktop Secret Service")
        elif args.credential_command == "status":
            get_venice_api_key()
            print("Venice API key is stored in the desktop Secret Service")
        else:
            delete_venice_api_key()
            print("Venice API key deleted from the desktop Secret Service")
    elif args.command == "vocabulary":
        _dispatch_vocabulary(args)
    elif args.command == "replay":
        _dispatch_replay(args)
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
        log_diagnostic("cli", "command_failed", f"{type(exc).__name__}: {exc}")
        print(f"voice-harness: {COMMAND_FAILURE}", file=__import__("sys").stderr)
        notify(COMMAND_FAILURE, error=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
