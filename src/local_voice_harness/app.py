from __future__ import annotations

import json

from .agents.delivery import (
    AgentDeliveryClaims as DeliveryClaims,
)
from .agents.delivery import (
    acknowledge_deliveries as acknowledge_claims,
)
from .agents.delivery import (
    release_deliveries as release_claims,
)
from .agents.model import AgentJob, JobStatus
from .agents.service import AgentTurnRequest as CursorTurnRequest
from .agents.service import agent_turn as cursor_turn
from .agents.store import AgentJobStore as JobStore
from .browser_context import RequestContext, request_context
from .components import component_usage, llm_ready, start_components
from .config import (
    CURSOR_PATTERN,
    JOBS_DIR,
    LEGACY_JOBS_DIR,
    PID_PATH,
    PROJECT_ROOT,
    STT_SOCKET,
    TTS_SOCKET,
)
from .cursor import consultation as cursor_consultation
from .cursor import provisioning as cursor_provisioning
from .cursor import questions as cursor_questions
from .diagnostics.health import self_health_response
from .diagnostics.help import harness_help_response
from .errors import HarnessError, SpeechDeliveryError
from .github_issue_creation import repository_from_utterance
from .integrations.github import resolve_pull_request_merge_identity
from .integrations.registry import (
    IntegrationRegistry,
    build_integration_registry,
    ticket_snapshot,
)
from .intent import (
    NON_ACTIONABLE_SUBMIT_RESPONSE,
    ForkIntent,
    Intent,
    IntentRoute,
    decide_fork_intent,
    is_grouped_repository_mapping,
    route_intent,
)
from .ipc import socket_ready
from .linear_ticket_creation import team_from_utterance
from .llm import qwen_response
from .questions import AnswerProvenance
from .responses import AssistantResponse, as_assistant_response
from .self_management import (
    UNSUPPORTED_INSPECTION_RESPONSE,
    inspect_config_utterance,
)
from .speech import SpeechRenderer
from .ticket_close import (
    admit_ticket_close,
    close_turn_arguments,
    wants_ticket_close_context,
)
from .ticket_targets import MISSING_ISSUE_SCOPE_RESPONSE, extract_ticket_targets
from .ticket_update import (
    admit_ticket_update,
    update_turn_arguments,
    wants_ticket_update_context,
)
from .tts.client import stream_and_play
from .user_config import UserConfig, load_user_config
from .vocabulary import parse_spoken_alias_request, resolve_aliases

CURSOR_STORE = JobStore(JOBS_DIR, LEGACY_JOBS_DIR)

CURSOR_MANAGEMENT_ACTIONS = {
    Intent.AGENT_LIST: "list",
    Intent.AGENT_STATUS: "status",
    Intent.AGENT_CANCEL: "cancel",
    Intent.AGENT_DISMISS: "dismiss",
    Intent.AGENT_REPEAT: "repeat",
    Intent.ANNOUNCEMENT_DIGEST: "missed",
}


def _context_for_route(
    text: str,
    route: IntentRoute,
    *,
    settings: UserConfig,
    integrations: IntegrationRegistry,
) -> RequestContext:
    """Capture external context only when the routed action can consume it."""
    if (
        route.intent == Intent.CONVERSATION
        or (
            route.actionable
            and route.intent
            in {
                Intent.AGENT_SUBMIT,
                Intent.GITHUB_ISSUE_CREATE,
                Intent.GITHUB_PR_MERGE,
                Intent.GITHUB_REPO_CREATE,
                Intent.GITHUB_ORG_REPO_CREATE,
                Intent.LINEAR_TICKET_CREATE,
                Intent.GITHUB_ISSUE_UPDATE,
                Intent.LINEAR_TICKET_UPDATE,
                Intent.GITHUB_ISSUE_CLOSE,
                Intent.LINEAR_TICKET_CLOSE,
                Intent.WORKSPACE_CONSULTATION,
            }
        )
        or cursor_consultation.wants_ticket_consultation_context(text)
        or wants_ticket_update_context(text)
        or wants_ticket_close_context(text)
    ):
        return request_context(
            text,
            platform=settings.platform,
            integrations=integrations,
        )
    return RequestContext(text)


def _pending_grouped_repository_question() -> tuple[str, str, str] | None:
    matches: list[tuple[str, str, str]] = []
    for job in CURSOR_STORE.list():
        if job.status != JobStatus.AWAITING_USER:
            continue
        question = cursor_questions.current(job)
        if question is None or question.owner != "grouped_repository":
            continue
        matches.append((job.id, question.id, question.origin.turn_token))
    return matches[0] if len(matches) == 1 else None


def _pending_repository_question() -> tuple[str, str, str] | None:
    matches: list[tuple[str, str, str]] = []
    for job in CURSOR_STORE.list():
        if job.status != JobStatus.AWAITING_USER:
            continue
        question = cursor_questions.current(job)
        if question is None or question.owner != "repository":
            continue
        matches.append((job.id, question.id, question.origin.turn_token))
    return matches[0] if len(matches) == 1 else None


def _single_pending_job() -> AgentJob | None:
    pending = [
        job for job in CURSOR_STORE.list() if job.status == JobStatus.AWAITING_USER
    ]
    return pending[0] if len(pending) == 1 else None


def _recent_completed_pr_parent() -> AgentJob | None:
    matches = [
        job
        for job in CURSOR_STORE.list()
        if job.status == JobStatus.COMPLETED
        and bool(job.worktree_path or job.repository)
    ]
    if not matches:
        return None
    return max(matches, key=lambda job: float(job.completed_at or 0))


def acknowledge_deliveries(claims: DeliveryClaims) -> None:
    acknowledge_claims(CURSOR_STORE, claims)


def release_deliveries(claims: DeliveryClaims) -> None:
    release_claims(CURSOR_STORE, claims)


def _acknowledge_consultation(
    speech_renderer: SpeechRenderer,
    settings: UserConfig,
    utterance: str,
) -> None:
    acknowledgement = cursor_consultation.acknowledgement(utterance)
    print(f"Assistant: {acknowledgement.display_text}")
    stream_and_play(
        speech_renderer.render(acknowledgement.spoken_text),
        settings=settings.audio,
    )


def respond(text: str, *, user_config: UserConfig | None = None) -> None:
    """Handle one foreground request from one immutable startup snapshot."""
    settings = user_config if user_config is not None else load_user_config()
    speech_renderer = SpeechRenderer.from_local_config(local_checkout=PROJECT_ROOT)
    integrations = build_integration_registry(settings)
    text = text.strip()
    if not text:
        raise HarnessError("request text is empty")
    trusted_utterance = text
    text = resolve_aliases(text)
    delivery_claims: DeliveryClaims = []
    with component_usage():
        try:
            start_components(settings.providers)
            print(f"You: {text}")
            recommendation_playback = False
            routing_context = RequestContext(text)
            grouped_reply = (
                _pending_grouped_repository_question()
                if is_grouped_repository_mapping(text)
                else None
            )
            repository_reply = (
                _pending_repository_question()
                if cursor_provisioning.is_repository_list_request(text)
                else None
            )
            pending = cursor_consultation.pending_question_snapshot(CURSOR_STORE, None)
            pending_job = _single_pending_job() if pending is None else None
            if grouped_reply is not None or repository_reply is not None:
                route = IntentRoute(Intent.AGENT_REPLY, "high")
            elif parse_spoken_alias_request(text) is not None:
                route = IntentRoute(Intent.VOCABULARY_ALIAS_ADD, "high")
            elif CURSOR_PATTERN.search(text):
                route = IntentRoute(Intent.AGENT_SUBMIT, "high")
            else:
                route = route_intent(
                    text,
                    routing_context,
                    cursor_session=(
                        pending.job_id
                        if pending is not None
                        else pending_job.id
                        if pending_job is not None
                        else None
                    ),
                    pending_question=(
                        pending.text
                        if pending is not None
                        else (
                            str(pending_job.question or "")
                            if pending_job is not None
                            else None
                        )
                    ),
                    clarification_kind=(
                        pending.owner
                        if pending is not None
                        else (
                            pending_job.clarification_kind
                            if pending_job is not None
                            else None
                        )
                    ),
                    settings=settings.providers,
                )
            context = _context_for_route(
                text,
                route,
                settings=settings,
                integrations=integrations,
            )
            fork_requested = decide_fork_intent(text) == ForkIntent.AFFIRMATIVE
            github_arguments = (
                {
                    "github_repository": context.github_repository,
                    "github_issue": context.github_issue,
                    "github_issue_context": context.github_issue_context,
                    "fork_requested": fork_requested,
                    "github_pull_request": context.github_pull_request,
                }
                if context.github_repository
                or context.github_issue
                or fork_requested
                or context.github_pull_request
                else {}
            )
            extraction = extract_ticket_targets(
                text,
                scope_source=context.issue_scope_source,
                scope=context.issue_scope,
            )
            missing_ticket_scope = extraction.has_unresolved_scope and route.intent in {
                Intent.AGENT_SUBMIT,
                Intent.UNCERTAIN,
            }
            ticket_admission = cursor_consultation.admit_ticket_consultation(
                text,
                extraction,
                focused_issue=context.focused_issue,
            )
            update_admission = admit_ticket_update(
                text,
                extraction,
                focused_issue=context.focused_issue,
            )
            close_admission = admit_ticket_close(
                text,
                extraction,
                focused_issue=context.focused_issue,
            )
            if cursor_consultation.is_apply_recommendation_request(text):
                choice_id = (
                    cursor_consultation.applicable_choice_id(
                        CURSOR_STORE, pending.job_id
                    )
                    if pending is not None
                    else None
                )
                if pending is None or choice_id is None:
                    response = cursor_consultation.RECOMMENDATION_UNAVAILABLE
                else:
                    response = cursor_turn(
                        CursorTurnRequest(
                            choice_id,
                            action="reply",
                            utterance=choice_id,
                            job_id=pending.job_id,
                            expected_question_id=pending.question_id,
                            expected_question_turn=pending.turn_token,
                            answer_provenance=AnswerProvenance.USER_TEXT,
                        ),
                        delivery_claims=delivery_claims,
                        integrations=integrations,
                    )[0]
            elif ticket_admission is not None:
                if ticket_admission.ticket is None:
                    response = ticket_admission.missing_identity_response
                else:
                    try:
                        client = integrations.herdr_client()
                        target = cursor_consultation.workspace_target(
                            client,
                            focused_repository=context.focused_repository,
                            completed_job=None,
                        )
                        if target is None:
                            response = cursor_consultation.NO_WORKSPACE
                        else:
                            _acknowledge_consultation(speech_renderer, settings, text)
                            assert ticket_admission.ticket.canonical is not None
                            assert ticket_admission.ticket.source is not None
                            snapshot = ticket_snapshot(
                                ticket_admission.ticket.canonical,
                                integrations,
                                provider=ticket_admission.ticket.source,
                                client=client,
                            )
                            response = cursor_consultation.consult_ticket(
                                client,
                                target,
                                text,
                                snapshot=snapshot,
                                kind=ticket_admission.kind,
                                adversarial=ticket_admission.adversarial,
                            )
                    except Exception:  # noqa: BLE001 - consultation fails closed
                        response = cursor_consultation.CONSULTATION_FAILED
            elif update_admission is not None:
                if update_admission.ticket is None:
                    response = update_admission.missing_identity_response
                elif not route.actionable:
                    response = (
                        "I did not update a ticket because the request was unclear. "
                        "Please name the ticket and the title or body change."
                    )
                else:
                    dispatch = update_turn_arguments(update_admission.ticket)
                    response = cursor_turn(
                        CursorTurnRequest(
                            context.text,
                            utterance=text,
                            github_repository=dispatch.github_repository,
                            github_issue=dispatch.github_issue,
                            github_issue_update_requested=(
                                dispatch.github_issue_update_requested
                            ),
                            issue_key=dispatch.issue_key,
                            linear_ticket_update_requested=(
                                dispatch.linear_ticket_update_requested
                            ),
                        ),
                        delivery_claims=delivery_claims,
                        integrations=integrations,
                    )[0]
            elif close_admission is not None:
                if close_admission.ticket is None:
                    response = close_admission.missing_identity_response
                elif not route.actionable:
                    response = (
                        "I did not close a ticket because the request was unclear. "
                        "Please name the ticket to close."
                    )
                else:
                    dispatch = close_turn_arguments(close_admission.ticket)
                    response = cursor_turn(
                        CursorTurnRequest(
                            context.text,
                            utterance=text,
                            github_repository=dispatch.github_repository,
                            github_issue=dispatch.github_issue,
                            github_issue_close_requested=(
                                dispatch.github_issue_close_requested
                            ),
                            issue_key=dispatch.issue_key,
                            linear_ticket_close_requested=(
                                dispatch.linear_ticket_close_requested
                            ),
                        ),
                        delivery_claims=delivery_claims,
                        integrations=integrations,
                    )[0]
            elif route.actionable and route.intent == Intent.GITHUB_PR_MERGE:
                identity = resolve_pull_request_merge_identity(
                    utterance=text,
                    focused_repository=context.github_repository,
                    focused_number=context.github_pull_request,
                )
                response = cursor_turn(
                    CursorTurnRequest(
                        context.text,
                        utterance=text,
                        github_repository=(
                            identity.repository if identity is not None else None
                        ),
                        github_pr_merge_requested=True,
                        github_pr_merge_number=(
                            identity.number if identity is not None else None
                        ),
                    ),
                    delivery_claims=delivery_claims,
                    integrations=integrations,
                )[0]
            elif route.actionable and route.intent == Intent.GITHUB_PR_CREATE:
                parent = _recent_completed_pr_parent()
                if parent is None:
                    response = (
                        "I don't have a recent completed job checkout to open a "
                        "pull request from."
                    )
                else:
                    response = cursor_turn(
                        CursorTurnRequest(
                            context.text,
                            utterance=text,
                            action="follow_up",
                            job_id=parent.id,
                            expected_parent_revision=parent.revision,
                            expected_completed_at=parent.completed_at,
                            github_pr_create_requested=True,
                        ),
                        delivery_claims=delivery_claims,
                        integrations=integrations,
                    )[0]
            elif route.actionable and route.intent == Intent.GITHUB_ISSUE_CREATE:
                response = cursor_turn(
                    CursorTurnRequest(
                        context.text,
                        utterance=trusted_utterance,
                        github_repository=(
                            context.github_repository or repository_from_utterance(text)
                        ),
                        github_issue_create_requested=True,
                    ),
                    delivery_claims=delivery_claims,
                    integrations=integrations,
                )[0]
            elif route.actionable and route.intent == Intent.GITHUB_REPO_CREATE:
                response = cursor_turn(
                    CursorTurnRequest(
                        context.text,
                        utterance=text,
                        github_repository=repository_from_utterance(text),
                        github_repo_create_requested=True,
                    ),
                    delivery_claims=delivery_claims,
                    integrations=integrations,
                )[0]
            elif route.actionable and route.intent == Intent.GITHUB_ORG_REPO_CREATE:
                response = cursor_turn(
                    CursorTurnRequest(
                        context.text,
                        utterance=text,
                        github_repository=repository_from_utterance(text),
                        github_repo_create_requested=True,
                        github_repo_create_org_requested=True,
                    ),
                    delivery_claims=delivery_claims,
                    integrations=integrations,
                )[0]
            elif route.actionable and route.intent == Intent.LINEAR_TICKET_CREATE:
                response = cursor_turn(
                    CursorTurnRequest(
                        context.text,
                        utterance=trusted_utterance,
                        linear_team=(
                            context.issue_scope
                            if context.issue_scope_source == "linear"
                            else team_from_utterance(text)
                        ),
                        linear_ticket_create_requested=True,
                    ),
                    delivery_claims=delivery_claims,
                    integrations=integrations,
                )[0]
            elif route.intent == Intent.HARNESS_CONFIG_INSPECT:
                response = (
                    inspect_config_utterance(text, settings)
                    if route.actionable
                    else AssistantResponse.from_text(UNSUPPORTED_INSPECTION_RESPONSE)
                )
            elif route.intent == Intent.HARNESS_CONFIG_CHANGE:
                response = AssistantResponse.from_text(
                    "Configuration changes require a wake conversation so I can read "
                    "back the exact change and receive confirmation. I didn't write "
                    "anything."
                )
            elif route.intent == Intent.VOCABULARY_ALIAS_ADD:
                response = AssistantResponse.from_text(
                    "Adding a spoken alias requires a wake conversation so I can read "
                    "back the phrase and target and receive confirmation. I didn't "
                    "write anything."
                )
            elif route.actionable and route.intent == Intent.SELF_HEALTH:
                response = self_health_response()
            elif route.actionable and route.intent == Intent.HARNESS_HELP:
                response = harness_help_response()
            elif missing_ticket_scope:
                response = MISSING_ISSUE_SCOPE_RESPONSE
            elif route.actionable and route.intent == Intent.QUESTION_CONSULTATION:
                if pending is None:
                    response = cursor_consultation.NO_PENDING_QUESTION
                else:
                    try:
                        client = integrations.herdr_client()
                        _acknowledge_consultation(
                            speech_renderer,
                            settings,
                            trusted_utterance,
                        )
                        response = cursor_consultation.consult_pending_question(
                            client,
                            CURSOR_STORE,
                            pending,
                            context.text,
                        )
                        recommendation_playback = True
                    except Exception as exc:  # noqa: BLE001 - consultation fails closed
                        response = (
                            str(exc)
                            if isinstance(exc, HarnessError)
                            and str(exc) == cursor_consultation.STALE_PENDING_QUESTION
                            else cursor_consultation.CONSULTATION_FAILED
                        )
            elif route.actionable and route.intent == Intent.WORKSPACE_CONSULTATION:
                try:
                    client = integrations.herdr_client()
                    target = cursor_consultation.workspace_target(
                        client,
                        focused_repository=context.focused_repository,
                        completed_job=None,
                    )
                    if target is None:
                        response = cursor_consultation.NO_WORKSPACE
                    else:
                        _acknowledge_consultation(
                            speech_renderer,
                            settings,
                            trusted_utterance,
                        )
                        response = cursor_consultation.consult(
                            client, target, context.text
                        )
                except Exception:  # noqa: BLE001 - consultation fails closed
                    response = cursor_consultation.CONSULTATION_FAILED
            elif route.actionable and route.intent == Intent.AGENT_SUBMIT:
                response = cursor_turn(
                    CursorTurnRequest(
                        context.text,
                        utterance=trusted_utterance,
                        context_repository=context.focused_repository,
                        issue_key=context.external_issue_reference,
                        issue_scope=context.issue_scope,
                        issue_scope_source=context.issue_scope_source,
                        **github_arguments,
                    ),
                    delivery_claims=delivery_claims,
                    integrations=integrations,
                )[0]
            elif route.intent == Intent.AGENT_SUBMIT:
                response = NON_ACTIONABLE_SUBMIT_RESPONSE
            elif route.intent == Intent.GITHUB_ISSUE_CREATE:
                response = (
                    "I did not create an issue because the request was unclear. "
                    "Please name the repository and issue."
                )
            elif route.intent in {
                Intent.GITHUB_REPO_CREATE,
                Intent.GITHUB_ORG_REPO_CREATE,
            }:
                response = (
                    "I did not create a repository because the request was unclear. "
                    "Please name the repository."
                )
            elif route.intent == Intent.LINEAR_TICKET_CREATE:
                response = (
                    "I did not create a Linear ticket because the request was unclear. "
                    "Please name the Linear team and ticket."
                )
            elif route.intent in {
                Intent.GITHUB_ISSUE_UPDATE,
                Intent.LINEAR_TICKET_UPDATE,
            }:
                response = (
                    "I did not update a ticket because the request was unclear. "
                    "Please name the ticket and the title or body change."
                )
            elif route.intent in {
                Intent.GITHUB_ISSUE_CLOSE,
                Intent.LINEAR_TICKET_CLOSE,
            }:
                response = (
                    "I did not close a ticket because the request was unclear. "
                    "Please name the ticket to close."
                )
            elif route.actionable and route.intent in CURSOR_MANAGEMENT_ACTIONS:
                response = cursor_turn(
                    CursorTurnRequest(
                        context.text,
                        action=CURSOR_MANAGEMENT_ACTIONS[route.intent],
                        reference=text,
                    ),
                    delivery_claims=delivery_claims,
                    integrations=integrations,
                )[0]
            elif route.intent == Intent.GITHUB_PR_MERGE:
                response = (
                    "I did not merge a pull request because the request was unclear."
                )
            elif route.intent == Intent.GITHUB_PR_CREATE:
                response = (
                    "I did not open a pull request because the request was unclear."
                )
            elif route.actionable and route.intent == Intent.AGENT_REPLY:
                reply_target = grouped_reply or repository_reply
                response = cursor_turn(
                    CursorTurnRequest(
                        context.text,
                        action="reply",
                        reference=text,
                        utterance=trusted_utterance,
                        job_id=reply_target[0] if reply_target is not None else None,
                        expected_question_id=(
                            reply_target[1] if reply_target is not None else None
                        ),
                        expected_question_turn=(
                            reply_target[2] if reply_target is not None else None
                        ),
                        answer_provenance=AnswerProvenance.USER_TEXT,
                    ),
                    delivery_claims=delivery_claims,
                    integrations=integrations,
                )[0]
            else:
                response = qwen_response(
                    context.text,
                    **github_arguments,
                    trusted_utterance=trusted_utterance,
                    delivery_claims=delivery_claims,
                    allow_tools=False,
                    settings=settings.providers,
                )
            rendered_response = as_assistant_response(response)
            print(f"Assistant: {rendered_response.display_text}")
            try:
                playback = stream_and_play(
                    speech_renderer.render(rendered_response.spoken_text),
                    settings=settings.audio,
                )
            except Exception as exc:
                raise SpeechDeliveryError(f"speech delivery failed: {exc}") from exc
            if recommendation_playback:
                interrupted = isinstance(playback, dict) and bool(
                    playback.get("interrupted")
                )
                cursor_consultation.complete_recommendation_delivery(
                    summary=rendered_response.spoken_text,
                    interrupted=interrupted,
                )
            acknowledge_deliveries(delivery_claims)
        except Exception:
            release_deliveries(delivery_claims)
            raise


def status() -> None:
    print(
        json.dumps(
            {
                "stt_ready": socket_ready(STT_SOCKET),
                "llm_ready": llm_ready(),
                "tts_ready": socket_ready(TTS_SOCKET),
                "recording": PID_PATH.exists(),
            }
        )
    )
