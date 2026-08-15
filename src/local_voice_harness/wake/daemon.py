from __future__ import annotations

import collections
import contextlib
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
import wave
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .. import recorder
from ..agents.model import AgentJob as CursorJob
from ..agents.model import JobStatus
from ..agents.service import AgentTurnRequest as CursorTurnRequest
from ..agents.service import agent_turn as cursor_turn
from ..agents.service import recover_jobs
from ..browser_context import (
    RequestContext,
    focused_browser_url,
    focused_herdr_github_context,
    request_context,
)
from ..components import start_components, stop_components
from ..config import (
    DICTATION_PID_PATH,
    DICTATION_RECORDER_LOG,
    DICTATION_STATE_DIR,
    DICTATION_WAV_PATH,
    JOBS_DIR,
    LEGACY_JOBS_DIR,
    PID_PATH,
    PROJECT_ROOT,
    RECORDER_LOG,
    RECORDING_LOCK,
    STATE_DIR,
    WAKE_LOCK,
    WAKE_PID_PATH,
    WAV_PATH,
)
from ..config_activation import (
    ActivationDecision,
    ActivationDelivery,
    ActivationDeliveryKind,
    ActivationStateError,
    ActivationStatus,
    ActivationStore,
    launch_activation_worker,
    publish_service_snapshot,
    render_activation_delivery,
    resolve_activation_decision,
)
from ..config_management import StaleConfigChangeError, apply_config_values
from ..critical_targets import (
    CriticalTarget,
    ReadbackCandidate,
    ReadbackReply,
    TargetSelection,
    identified_target_response,
    new_candidate,
    readback_response,
    resolve_readback,
    select_submit_target,
)
from ..cursor import announcements as announcement_policy
from ..cursor import consultation as cursor_consultation
from ..cursor import inbox as cursor_inbox
from ..cursor import provisioning as cursor_provisioning
from ..cursor import questions as cursor_questions
from ..cursor import service as cursor_service
from ..cursor.delivery import (
    DELIVERY_RENEW_SECONDS,
    DELIVERY_WINDOW,
    DeliveryClaim,
    DeliveryClaims,
)
from ..cursor.delivery import (
    acknowledge_deliveries as acknowledge_claims,
)
from ..cursor.delivery import (
    acknowledge_delivery as acknowledge_claim,
)
from ..cursor.delivery import (
    release_deliveries as release_claims,
)
from ..cursor.delivery import (
    release_delivery as release_claim,
)
from ..cursor.delivery import (
    renew_delivery as renew_claim,
)
from ..cursor.store import JobStore
from ..diagnostic_safety import (
    DAEMON_FAILURE,
    PLAYBACK_FAILURE,
    RECORDING_FAILURE,
    SPEECH_DELIVERY_FAILURE,
    VOICE_REQUEST_FAILURE,
    redact_diagnostic,
)
from ..diagnostics.health import self_health_response
from ..diagnostics.help import harness_help_response
from ..errors import HarnessError, NoSpeechError, SpeechDeliveryError
from ..github_issue_creation import repository_from_utterance
from ..integrations.github import (
    GitHubPullRequest,
    github_pull_request_from_url,
    resolve_pull_request_merge_identity,
)
from ..integrations.registry import (
    IntegrationRegistry,
    build_integration_registry,
    capture_context,
    ticket_snapshot,
)
from ..intent import (
    NON_ACTIONABLE_SUBMIT_RESPONSE,
    ForkIntent,
    Intent,
    IntentRoute,
    decide_fork_intent,
    is_grouped_repository_mapping,
    needs_intent_router,
    route_intent,
)
from ..linear_ticket_creation import team_from_utterance
from ..llm import qwen_turn
from ..notifications import notify
from ..process import ProcessHandle, process_identity
from ..questions import (
    AnswerOutcome,
    AnswerProvenance,
    Question,
    question_control,
    resolve_answer,
)
from ..responses import AssistantResponse, ResponseLike, as_assistant_response
from ..self_management import (
    UNSUPPORTED_INSPECTION_RESPONSE,
    ConfigChangeRequest,
    ConfirmationDecision,
    PendingConfigChange,
    commit_pending_change,
    inspect_config_utterance,
    prepare_config_change,
    render_change_committed,
    render_change_preparation,
    resolve_confirmation,
)
from ..speech import SpeechRenderer, StreamingSpeechRenderer
from ..stt.client import (
    RetainedTranscript,
    recover_retained_transcripts,
)
from ..stt.client import (
    transcribe_retained as transcribe,
)
from ..ticket_close import (
    admit_ticket_close,
    close_turn_arguments,
    wants_ticket_close_context,
)
from ..ticket_merge import (
    admit_ticket_merge,
    merge_turn_arguments,
    wants_ticket_merge_context,
)
from ..ticket_split import (
    admit_ticket_split,
    split_turn_arguments,
    wants_ticket_split_context,
)
from ..ticket_targets import (
    MISSING_ISSUE_SCOPE_RESPONSE,
    TicketExtraction,
    extract_ticket_targets,
)
from ..ticket_update import (
    admit_ticket_update,
    update_turn_arguments,
    wants_ticket_update_context,
)
from ..tts.queue import PlaybackQueue, PlaybackRequest
from ..user_config import AnnouncementSettings, UserConfig, load_user_config
from ..vad import FRAME_BYTES, FRAME_MS, SAMPLE_RATE, SpeechDetector
from ..vocabulary import (
    PendingSpokenAlias,
    SpokenAliasPreparation,
    SpokenAliasStatus,
    commit_spoken_alias,
    parse_spoken_alias_request,
    prepare_spoken_alias,
    render_spoken_alias_committed,
    render_spoken_alias_preparation,
)
from ..vocabulary import (
    load as load_vocabulary,
)

RECORDING_PATHS = recorder.RecorderPaths(
    STATE_DIR, WAV_PATH, PID_PATH, RECORDER_LOG, RECORDING_LOCK
)
DICTATION_RECORDING_PATHS = recorder.RecorderPaths(
    DICTATION_STATE_DIR,
    DICTATION_WAV_PATH,
    DICTATION_PID_PATH,
    DICTATION_RECORDER_LOG,
    RECORDING_LOCK,
)
CAPTURE_PATHS = (RECORDING_PATHS, DICTATION_RECORDING_PATHS)
CURSOR_STORE = JobStore(JOBS_DIR, LEGACY_JOBS_DIR)
END_SILENCE_MS = 720
MAX_UTTERANCE_SECONDS = 120
CONVERSATION_TIMEOUT_SECONDS = 60
HOLD_EXTENSION_SECONDS = 120
MAX_HOLD_EXTENSIONS = 2
PRE_ROLL_FRAMES = 25
MICROPHONE_START_ATTEMPTS = 30
MICROPHONE_RETRY_SECONDS = 1
RETAINED_RECOVERY_RETRY_SECONDS = 5.0
PLAYBACK_ECHO_WINDOW_SECONDS = 8.0
RECENT_PLAYBACK_LIMIT = 8
TARGET_RESOLUTION_CONTEXT_RESPONSE = AssistantResponse.from_text(
    "I still can't verify the requested issue. Open its exact issue page and say "
    '"I\'ve opened it," or repeat the request with a fully qualified reference.'
)
TARGET_RESOLUTION_CONTINUATION_PATTERN = re.compile(
    r"\s*(?:(?:okay|ok|yeah|yes|yep|right|well|um|uh)[,\s]+)*"
    r"(?:(?:i(?:'ve| have)?\s+)?opened\s+(?:it|this|the issue)|"
    r"(?:this|that)\s+issue|this one)"
    r"(?:\s*[,;]?\s+(?:now|please|for you|as requested|like you asked|um|uh))*"
    r"\s*[,!?\.]?\s*",
    re.IGNORECASE,
)
SIDE_EFFECTING_INTENTS = frozenset(
    {
        Intent.AGENT_SUBMIT,
        Intent.AGENT_REPLY,
        Intent.AGENT_FOLLOWUP,
        Intent.AGENT_CANCEL,
        Intent.AGENT_DISMISS,
        Intent.AGENT_REPEAT,
        Intent.ANNOUNCEMENT_DIGEST,
        Intent.GITHUB_ISSUE_CREATE,
        Intent.GITHUB_PR_CREATE,
        Intent.GITHUB_PR_MERGE,
        Intent.GITHUB_REPO_CREATE,
        Intent.GITHUB_ORG_REPO_CREATE,
        Intent.GITHUB_ISSUE_UPDATE,
        Intent.GITHUB_ISSUE_CLOSE,
        Intent.LINEAR_TICKET_CREATE,
        Intent.LINEAR_TICKET_UPDATE,
        Intent.LINEAR_TICKET_CLOSE,
        Intent.QUESTION_CONSULTATION,
        Intent.WORKSPACE_CONSULTATION,
    }
)


@dataclass(frozen=True, slots=True)
class PendingTargetReadback:
    candidate: ReadbackCandidate
    request: CursorTurnRequest


@dataclass(frozen=True, slots=True)
class PendingTargetResolution:
    trusted_utterance: str
    route: IntentRoute
    created_at: float

    @property
    def expires_at(self) -> float:
        return self.created_at + CONVERSATION_TIMEOUT_SECONDS


@dataclass(frozen=True, slots=True)
class PendingQuestionRetarget:
    candidate_ids: tuple[str, ...]
    created_at: float

    @property
    def expires_at(self) -> float:
        return self.created_at + CONVERSATION_TIMEOUT_SECONDS


def _is_target_resolution_continuation(text: str) -> bool:
    return TARGET_RESOLUTION_CONTINUATION_PATTERN.fullmatch(text) is not None


def _has_exact_target_resolution(
    extraction: TicketExtraction,
    context: RequestContext,
) -> bool:
    if (
        context.focused_issue is None
        or extraction.requested_count != 1
        or extraction.has_unresolved_scope
    ):
        return False
    selection = select_submit_target(extraction, context)
    return bool(
        selection is not None
        and selection.target.canonical.casefold() == context.focused_issue.casefold()
    )


def _critical_target_request(target: CriticalTarget) -> CursorTurnRequest:
    """Build a dispatch request containing only the confirmed canonical identity."""

    text = f"Work on {target.canonical}."
    if target.provider == "github":
        return CursorTurnRequest(
            text,
            utterance=text,
            context_repository=target.repository,
            github_repository=target.repository,
            github_issue=int(target.ticket),
            issue_scope=target.repository,
            issue_scope_source="github",
        )
    return CursorTurnRequest(
        text,
        utterance=text,
        issue_key=target.canonical,
        issue_scope=target.repository,
        issue_scope_source="linear",
    )


def acknowledge_delivery(job_id: str, token: str) -> bool:
    return acknowledge_claim(CURSOR_STORE, job_id, token)


def release_delivery(job_id: str, token: str) -> bool:
    return release_claim(CURSOR_STORE, job_id, token)


def renew_delivery(job_id: str, token: str) -> bool:
    return renew_claim(CURSOR_STORE, job_id, token)


def acknowledge_deliveries(claims: DeliveryClaims) -> list[DeliveryClaim]:
    return acknowledge_claims(CURSOR_STORE, claims)


def release_deliveries(claims: DeliveryClaims) -> None:
    release_claims(CURSOR_STORE, claims)


def drain_pending_announcements(
    settings: AnnouncementSettings,
    *,
    integrations: IntegrationRegistry | None = None,
    snooze: announcement_policy.AnnouncementSnooze | None = None,
) -> announcement_policy.DrainResult:
    recover_jobs(integrations=integrations)
    try:
        result = announcement_policy.drain_background_announcements(
            CURSOR_STORE,
            settings,
            snooze=snooze,
        )
    except Exception as exc:  # noqa: BLE001 - notification failures cannot stop wake
        log(f"announcement drain failed: {type(exc).__name__}: {exc}")
        return announcement_policy.DrainResult()
    for claim in result.desktop_failed:
        log(f"desktop notification failed for job {claim.job.id}; delivery will retry")
    return result


class _DeliveryLeaseGuard:
    def __init__(self, requests: list[PlaybackRequest]) -> None:
        self._claims = {
            (request.job_id, request.delivery_token)
            for request in requests
            if request.job_id and request.delivery_token
        }
        for request in requests:
            for job_id, token, _status in request.extra_claims:
                self._claims.add((job_id, token))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error: HarnessError | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self._claims or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="voice-delivery-lease",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(DELIVERY_RENEW_SECONDS):
            with self._lock:
                for job_id, token in self._claims:
                    try:
                        renewed = renew_delivery(job_id, token)
                    except Exception as exc:
                        self._error = HarnessError(
                            "delivery lease renewal failed for "
                            f"{job_id}: {type(exc).__name__}: {exc}"
                        )
                        self._stop.set()
                        return
                    if not renewed:
                        self._error = HarnessError(
                            f"delivery lease lost for job {job_id}"
                        )
                        self._stop.set()
                        return
                if not self._claims:
                    return

    def maintain(self) -> None:
        with self._lock:
            if self._error is not None:
                raise self._error

    def complete(self, request: PlaybackRequest) -> None:
        with self._lock:
            if request.job_id and request.delivery_token:
                self._claims.discard((request.job_id, request.delivery_token))
            for job_id, token, _status in request.extra_claims:
                self._claims.discard((job_id, token))
            if not self._claims:
                self._stop.set()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is None:
            return
        self._thread.join(timeout=2)
        if self._thread.is_alive():
            raise HarnessError("delivery lease worker did not stop")


WAKE_NAME = r"(?:jarvis|travis|service|jarvus|jervis)"
WAKE_PREFIX = re.compile(rf"^\s*hey[,\s]+{WAKE_NAME}\b[\s,;:!?.-]*", re.IGNORECASE)
WAKE_ANYWHERE = re.compile(rf"\bhey[,\s]+{WAKE_NAME}\b[\s,;:!?.-]*", re.IGNORECASE)
SPOKEN_WAKE_PATTERN = re.compile(rf"\bhey[,\s]+{WAKE_NAME}\b", re.IGNORECASE)


def strip_wake_prefix(text: str) -> tuple[str, bool]:
    """Remove a leading wake phrase, tolerating Parakeet mis-transcriptions."""

    match = WAKE_PREFIX.match(text)
    if match is not None:
        return text[match.end() :].strip(), True
    match = WAKE_ANYWHERE.search(text)
    if match is not None:
        return (text[: match.start()] + text[match.end() :]).strip(), True
    return text.strip(), False


CLOSE_PATTERN = re.compile(
    r"\b(?:goodbye|stop listening|go to sleep|end conversation)\b", re.IGNORECASE
)
STOP_TALKING_PATTERN = re.compile(r"\b(?:stop talking|shut up)\b", re.IGNORECASE)
TRANSCRIPT_REPLAY_PATTERN = re.compile(
    r"^\s*(?:"
    r"what\s+did\s+you\s+hear|"
    r"what\s+did\s+i\s+(?:just\s+)?say|"
    r"(?:please\s+)?(?:repeat|replay|read\s+back)\s+(?:the\s+)?"
    r"(?:last\s+)?(?:transcript|utterance)|"
    r"play\s+(?:that|it|what\s+i\s+said)\s+back"
    r")\s*[?.!]?\s*$",
    re.IGNORECASE,
)
TRANSCRIPT_CORRECTION_PATTERN = re.compile(
    r"^\s*(?:"
    r"i\s+said\s+|"
    r"that(?:'s|\s+is)\s+not\s+what\s+i\s+said(?:[,.]?\s*(?:i\s+said\s+))?"
    r")(.+?)\s*$",
    re.IGNORECASE,
)
TRANSCRIPT_CORRECTION_BARE_PATTERN = re.compile(
    r"^\s*that(?:'s|\s+is)\s+not\s+what\s+i\s+said\s*[!.]?\s*$",
    re.IGNORECASE,
)
INSPECT_CONTEXT_PATTERN = re.compile(
    r"^\s*(?:"
    r"what\s+are\s+you\s+looking\s+at|"
    r"what(?:'s|\s+is)\s+(?:the\s+)?focused\s+(?:tab|context|source|app)|"
    r"what\s+context\s+are\s+you\s+using"
    r")\s*[?.!]?\s*$",
    re.IGNORECASE,
)
OMIT_CONTEXT_PATTERN = re.compile(
    r"^\s*(?:"
    r"don['’]?t\s+use\s+that|"
    r"ignore\s+(?:the\s+)?focused\s+(?:tab|app|context|window|source)|"
    r"stop\s+using\s+(?:the\s+)?focused\s+(?:tab|context|app)"
    r")\s*[?.!]?\s*$",
    re.IGNORECASE,
)
HOLD_PATTERN = re.compile(
    r"^\s*(?:"
    r"hold\s+on|"
    r"give\s+me\s+a\s+(?:minute|moment|sec(?:ond)?s?)|"
    r"wait\s+(?:a\s+(?:minute|moment|second)|please)|"
    r"hang\s+on"
    r")\s*[!.]?\s*$",
    re.IGNORECASE,
)
SNOOZE_PATTERN = re.compile(
    r"^\s*(?:"
    r"(?:please\s+)?(?:snooze|mute)"
    r"(?:\s+(?P<target>announcements|background(?:\s+announcements)?|everything))?"
    r"(?:\s+for\s+(?P<minutes>\d+)\s*(?:minute|minutes|min))?"
    r"|be\s+quiet\s+for\s+a\s+(?:bit|while|minute)"
    r"|don['’]?t\s+talk\s+at\s+all"
    r")\s*[!.]?\s*$",
    re.IGNORECASE,
)
CLEAR_SNOOZE_PATTERN = re.compile(
    r"^\s*(?:"
    r"you\s+can\s+talk\s+again|"
    r"stop\s+snooz(?:e|ing)|"
    r"unmute(?:\s+announcements)?"
    r")\s*[!.]?\s*$",
    re.IGNORECASE,
)
RETARGET_PATTERN = re.compile(
    r"^\s*(?:"
    r"(?:let(?:'s| us)\s+)?(?:talk|switch)\s+(?:to|about)\s+(?:the\s+)?(?P<ref>.+?)|"
    r"what\s+was\s+the\s+(?P<qref>.+?)\s+question"
    r")\s*[?.!]?\s*$",
    re.IGNORECASE,
)
RESUME_PATTERN = re.compile(
    r"^\s*(?:"
    r"(?:please\s+)?(?:resume|continue)"
    r"(?P<with>\s+with)?"
    r"(?:\s+(?:the\s+)?(?P<ref>.+?))?|"
    r"pick\s+(?:this|it|(?:the\s+)?(?P<pickref>.+?))\s+back\s+up|"
    r"where\s+were\s+we(?:\s+(?:on|with|about)\s+(?:the\s+)?(?P<whereref>.+?))?|"
    r"where\s+was\s+I"
    r")\s*[?.!]?\s*$",
    re.IGNORECASE,
)
PENDING_SUBMIT_PATTERN = re.compile(
    r"\b(?:work\s+on|fix|change|update|implement|add|remove|run|review|inspect|"
    r"start|create|build|refactor|test)\b",
    re.IGNORECASE,
)
FILLER_WORDS = frozenset(
    {
        "ah",
        "and",
        "but",
        "erm",
        "hmm",
        "i",
        "like",
        "mean",
        "mm",
        "okay",
        "ok",
        "so",
        "uh",
        "um",
        "well",
        "yeah",
        "you",
        "know",
    }
)
END_CONVERSATION_RESPONSE = "Okay, I'll be here if you need me."
RECENT_DETAILS_UNAVAILABLE = (
    "I no longer have details for that recent announcement. "
    "Ask for the job by name or ID."
)
MISSING_TRANSCRIPT_RESPONSE = (
    "I don't have a recent transcript to replay, so I won't invent one."
)
DISPATCHED_TRANSCRIPT_RESPONSE = (
    "I already started that request, so I can't replace what I heard."
)
BARE_TRANSCRIPT_CORRECTION_RESPONSE = (
    "I heard a correction but no replacement. Say I said, then the request."
)
NO_FOCUSED_CONTEXT_RESPONSE = "I'm not looking at a focused source."
OMIT_FOCUSED_CONTEXT_RESPONSE = (
    "Okay, I won't use the focused tab or app for the rest of this conversation."
)
HOLD_ACCEPTED_RESPONSE = "Okay, I'll keep listening."
HOLD_EXHAUSTED_RESPONSE = (
    "I'm already holding. I'll keep listening until this window ends."
)
HOLD_INACTIVE_RESPONSE = "I can only hold while I'm already listening for a follow-up."
DEFAULT_SNOOZE_SECONDS = 30 * 60
SNOOZE_STARTED_RESPONSE = (
    "Okay, I'll hold ordinary background announcements for 30 minutes."
)


def snooze_started_response(minutes: int | None) -> str:
    if minutes is None:
        return SNOOZE_STARTED_RESPONSE
    label = "1 minute" if minutes == 1 else f"{minutes} minutes"
    return f"Okay, I'll hold ordinary background announcements for {label}."


SNOOZE_MUTE_ALL_RESPONSE = (
    "Okay, I'll mute background announcements, including questions and failures."
)
SNOOZE_CLEARED_RESPONSE = "Okay, I can announce background updates again."
SNOOZE_INACTIVE_RESPONSE = "I wasn't snoozing background announcements."
RETARGET_NOT_FOUND_RESPONSE = "I couldn't find a job matching that."
RETARGET_INACTIVE_RESPONSE = (
    "I can only switch questions while this conversation has a live pending question."
)
RESUME_NONE_RESPONSE = "There's no Cursor job waiting for a reply."


@dataclass
class BargeIn:
    initial: list[bytes]
    woke: bool


@dataclass
class CompletedFollowup:
    """A bounded, one-shot reference to the last announced completed job.

    ``expires_at`` is a ``time.monotonic()`` deadline, so it is intentionally
    volatile and cannot survive a restart.
    """

    job_id: str
    parent_revision: int
    completed_at: float | None
    expires_at: float
    display_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class FocusedIdentity:
    """Speakable identity from RequestContext, excluding untrusted bodies."""

    repository: str | None = None
    issue: str | None = None
    pull_request: str | None = None
    app_class: str | None = None
    source_kinds: tuple[str, ...] = ()

    def spoken(self) -> str:
        parts: list[str] = []
        if self.repository:
            parts.append(f"repository {self.repository}")
        if self.issue:
            parts.append(f"issue {self.issue}")
        if self.pull_request:
            parts.append(f"pull request {self.pull_request}")
        if self.app_class:
            parts.append(f"app {self.app_class}")
        if self.source_kinds:
            parts.append("source kinds " + ", ".join(self.source_kinds))
        if not parts:
            return NO_FOCUSED_CONTEXT_RESPONSE
        return "I'm looking at " + "; ".join(parts) + "."


@dataclass(frozen=True, slots=True)
class LastTranscript:
    """Volatile trusted utterance for the current wake conversation.

    ``expires_at`` is a ``time.monotonic()`` deadline, so the slot cannot
    survive a restart. ``dispatched`` becomes true only after a job or write
    has started for this utterance.
    """

    utterance: str
    dispatched: bool
    expires_at: float


@dataclass(frozen=True, slots=True)
class RecentPlayback:
    text: str
    expires_at: float


@dataclass(frozen=True, slots=True)
class PendingQuestionSnapshot:
    job_id: str
    text: str
    owner: str
    question_id: str
    turn_token: str
    question: Question | None = None


def log(message: str) -> None:
    print(
        f"[voice-harness-wake] {redact_diagnostic(message)}",
        file=sys.stderr,
        flush=True,
    )


def _display_fingerprint(display_text: str) -> str:
    return hashlib.sha256(display_text.encode("utf-8")).hexdigest()


def _transcript_words(text: str) -> tuple[str, ...]:
    return tuple(
        re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text.casefold(), flags=re.UNICODE)
    )


def _matches_playback_prefix(transcript: str, playback: str) -> bool:
    captured_words = _transcript_words(transcript)
    played_words = _transcript_words(playback)
    if not captured_words or len(captured_words) > len(played_words):
        return False
    if captured_words == played_words:
        return True
    return (
        len(captured_words) >= 3
        and captured_words == played_words[: len(captured_words)]
    )


def _is_filler_speech(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.casefold())
    return bool(words) and all(word in FILLER_WORDS for word in words)


class WakeConversationDaemon:
    def __init__(
        self,
        user_config: UserConfig,
        *,
        config_activation_store: ActivationStore | None = None,
    ) -> None:
        import numpy as np
        import openwakeword
        from openwakeword.model import Model

        self.user_config = user_config
        self.audio = user_config.audio
        self.platform = user_config.platform
        self.announcements = user_config.announcements
        self.providers = user_config.providers
        self.speech_renderer = SpeechRenderer.from_local_config(
            local_checkout=PROJECT_ROOT
        )
        self.integrations = build_integration_registry(user_config)
        self.np: Any = np
        module_path = openwakeword.__file__
        if module_path is None:
            raise HarnessError("Could not locate OpenWakeWord package resources")
        model_path = (
            Path(module_path).parent / "resources" / "models" / "hey_jarvis_v0.1.onnx"
        )
        self.wake_model: Any = Model(
            wakeword_models=[str(model_path)],
            inference_framework="onnx",
            vad_threshold=0.0,
        )
        self.wake_key = next(iter(self.wake_model.models))
        self.speech_detector = SpeechDetector(minimum_rms=self.audio.min_speech_rms)
        self.pre_roll: collections.deque[bytes] = collections.deque(
            maxlen=PRE_ROLL_FRAMES
        )
        self.history: list[dict[str, str]] = []
        self.cursor_session: str | None = None
        self.completed_followup: CompletedFollowup | None = None
        self.conversation_created_pull_request: GitHubPullRequest | None = None
        self.recent_playback: collections.deque[RecentPlayback] = collections.deque(
            maxlen=RECENT_PLAYBACK_LIMIT
        )
        self.pending_target_readback: PendingTargetReadback | None = None
        self.pending_target_resolution: PendingTargetResolution | None = None
        self.pending_question_retarget: PendingQuestionRetarget | None = None
        self.pending_config_change: PendingConfigChange | None = None
        self.pending_spoken_alias: PendingSpokenAlias | None = None
        self.config_activation_store = config_activation_store or ActivationStore()
        self.config_activation_delivery: ActivationDelivery | None = None
        self.launched_config_activations: dict[str, subprocess.Popen[bytes]] = {}
        self.config_activation_dispatch_attempts: dict[str, int] = {}
        self.retained_recovery_required = False
        self.retained_recovery_retry_at = 0.0
        self.conversation_deadline = 0.0
        self.awaiting_followup = False
        self.last_ordinary_reply: str | None = None
        self.last_transcript: LastTranscript | None = None
        self.omit_focused_context = False
        self.last_focused_identity: FocusedIdentity | None = None
        self.hold_extensions = 0
        self.announcement_snooze: announcement_policy.AnnouncementSnooze | None = None
        self.last_wake = 0.0
        self.force_listen = threading.Event()
        self.running = True
        self.microphone: subprocess.Popen[bytes] | None = None
        self.microphone_paused = False
        self.activation_thread: threading.Thread | None = None
        self.activation_error: Exception | None = None
        self.component_lock = threading.Lock()
        self.playback_queue = PlaybackQueue(self.audio)

    def is_speech(self, frame: bytes) -> bool:
        return self.speech_detector.is_speech(frame)

    def read_frame(self) -> bytes:
        if self.microphone is None or self.microphone.stdout is None:
            raise HarnessError("microphone process is unavailable")
        data = bytearray()
        while len(data) < FRAME_BYTES:
            chunk = self.microphone.stdout.read(FRAME_BYTES - len(data))
            if not chunk:
                detail = (
                    self.microphone.stderr.read().decode(errors="replace").strip()
                    if self.microphone.stderr is not None
                    else ""
                )
                raise HarnessError(f"microphone stream ended: {detail}")
            data.extend(chunk)
        return bytes(data)

    def start_microphone(self) -> None:
        command = ["pw-record", "--raw"]
        if self.audio.source:
            command.extend(("--target", self.audio.source))
        command.extend(("--channels=1", "--rate=16000", "--format=s16", "-"))
        for attempt in range(1, MICROPHONE_START_ATTEMPTS + 1):
            self.microphone = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            time.sleep(0.2)
            if self.microphone.poll() is None:
                log(
                    "listening for Hey Jarvis on "
                    f"{self.audio.source or 'PipeWire default source'}"
                )
                return

            detail = (
                self.microphone.stderr.read().decode(errors="replace").strip()
                if self.microphone.stderr
                else ""
            )
            if (
                "no target node available" not in detail.lower()
                or attempt == MICROPHONE_START_ATTEMPTS
            ):
                raise HarnessError(
                    f"pw-record failed: {detail or self.microphone.returncode}"
                )
            log(
                f"microphone is not ready; retrying in "
                f"{MICROPHONE_RETRY_SECONDS}s ({attempt}/{MICROPHONE_START_ATTEMPTS})"
            )
            time.sleep(MICROPHONE_RETRY_SECONDS)

    def pause_microphone(self) -> None:
        if (
            not self.microphone_paused
            and self.microphone is not None
            and self.microphone.poll() is None
        ):
            os.kill(self.microphone.pid, signal.SIGSTOP)
            self.microphone_paused = True

    def resume_microphone(self) -> None:
        if not self.microphone_paused:
            return
        if self.microphone is None or self.microphone.poll() is not None:
            self.microphone_paused = False
            return
        os.kill(self.microphone.pid, signal.SIGCONT)
        self.microphone_paused = False
        for _ in range(4):
            self.read_frame()
        self.pre_roll.clear()

    def record_utterance(
        self,
        initial: list[bytes],
        *,
        wait_for_fresh_speech: bool = False,
    ) -> Path:
        def write_audio(path: Path) -> None:
            frames = list(initial)
            has_speech = (
                False
                if wait_for_fresh_speech
                else any(self.is_speech(frame) for frame in frames)
            )
            silence_ms = 0
            captured_frames = 0
            while self.running:
                frame = self.read_frame()
                frames.append(frame)
                captured_frames += 1
                if self.is_speech(frame):
                    has_speech = True
                    silence_ms = 0
                elif has_speech:
                    silence_ms += FRAME_MS
                duration = captured_frames * FRAME_MS / 1000
                if has_speech and silence_ms >= END_SILENCE_MS:
                    break
                if duration >= MAX_UTTERANCE_SECONDS:
                    break

            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(SAMPLE_RATE)
                output.writeframes(b"".join(frames))

        return recorder.write_audio_generation(
            RECORDING_PATHS,
            write_audio,
            conflicts=(DICTATION_RECORDING_PATHS,),
        )

    def record_utterance_safely(
        self,
        initial: list[bytes],
        *,
        wait_for_fresh_speech: bool = False,
    ) -> Path | None:
        try:
            if recorder.any_recording_active(CAPTURE_PATHS):
                message = "wake activation deferred while another recording is active"
                log(message)
                notify(message)
                return None
            return self.record_utterance(
                initial,
                wait_for_fresh_speech=wait_for_fresh_speech,
            )
        except HarnessError as exc:
            message = f"wake recording suppressed: {exc}"
            log(message)
            notify(RECORDING_FAILURE, error=True)
            return None

    def begin_activation(self) -> None:
        if self.activation_thread is not None and self.activation_thread.is_alive():
            return
        self.activation_error = None

        def activate() -> None:
            try:
                with self.component_lock:
                    start_components(self.providers)
                    if self.providers.llm_provider == "local":
                        qwen_turn(
                            "Reply with only OK. Do not call a tool.",
                            allow_tools=False,
                            settings=self.providers,
                        )
                    log("LLM tool graph and TTS backend are warm")
            except Exception as exc:
                self.activation_error = exc

        self.activation_thread = threading.Thread(
            target=activate, name="voice-harness-activation", daemon=True
        )
        self.activation_thread.start()

    def ensure_components(self) -> None:
        if self.activation_thread is None:
            with self.component_lock:
                start_components(self.providers)
            return
        self.activation_thread.join(timeout=60)
        if self.activation_thread.is_alive():
            raise HarnessError("model activation did not finish within 60 seconds")
        if self.activation_error is not None:
            raise HarnessError(f"model activation failed: {self.activation_error}")

    def stop_components_when_idle(self) -> None:
        with self.component_lock:
            stop_components()

    def close_conversation(self, reason: str) -> None:
        log(f"conversation closed: {reason}")
        self.history.clear()
        self.cursor_session = None
        self.completed_followup = None
        self.conversation_created_pull_request = None
        self.recent_playback.clear()
        self.pending_target_readback = None
        self.pending_target_resolution = None
        self.pending_question_retarget = None
        self.pending_config_change = None
        self.pending_spoken_alias = None
        self.conversation_deadline = 0.0
        self.awaiting_followup = False
        self.last_ordinary_reply = None
        self.last_transcript = None
        self.omit_focused_context = False
        self.last_focused_identity = None
        self.hold_extensions = 0
        self.pause_microphone()
        try:
            self.stop_components_when_idle()
        finally:
            self.resume_microphone()
        notify("Conversation closed")

    def close_pending_capture(self, reason: str) -> None:
        """Stop ambient follow-up capture without discarding the durable question."""
        log(f"pending-question capture closed: {reason}")
        self.history.clear()
        self.completed_followup = None
        self.pending_question_retarget = None
        self.pending_config_change = None
        self.pending_spoken_alias = None
        self.conversation_deadline = 0.0
        self.awaiting_followup = False
        self.hold_extensions = 0
        self.pause_microphone()
        try:
            self.stop_components_when_idle()
        finally:
            self.resume_microphone()

    def end_conversation(self) -> BargeIn | None:
        """Speak a brief farewell, then close unless the user barges in."""
        response = as_assistant_response(END_CONVERSATION_RESPONSE)
        print(f"Assistant: {response.display_text}", flush=True)
        _playback, interruption = self.play_response(response)
        if interruption is not None:
            return interruption
        self.close_conversation("assistant ended the conversation")
        return None

    def _acknowledge_consultation(self, utterance: str) -> BargeIn | None:
        response = cursor_consultation.acknowledgement(utterance)
        print(f"Assistant: {response.display_text}", flush=True)
        _playback, interruption = self.play_response(response)
        return interruption

    def wait_for_playback_quiet(self) -> None:
        if self.microphone is None or self.microphone.poll() is not None:
            self.pre_roll.clear()
            return
        quiet = 0
        max_frames = max(
            self.audio.playback_quiet_frames,
            int(self.audio.playback_quiet_timeout_seconds * 1000 / FRAME_MS),
        )
        for _ in range(max_frames):
            frame = self.read_frame()
            quiet = quiet + 1 if not self.is_speech(frame) else 0
            if quiet >= self.audio.playback_quiet_frames:
                break
        self.pre_roll.clear()

    def _remember_recent_playback(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.recent_playback.append(
            RecentPlayback(
                text=text,
                expires_at=time.monotonic() + PLAYBACK_ECHO_WINDOW_SECONDS,
            )
        )

    def _active_recent_playback(self) -> tuple[RecentPlayback, ...]:
        now = time.monotonic()
        while self.recent_playback and self.recent_playback[0].expires_at <= now:
            self.recent_playback.popleft()
        return tuple(self.recent_playback)

    @staticmethod
    def _is_playback_echo(
        transcript: str, recent_playback: tuple[RecentPlayback, ...]
    ) -> bool:
        return any(
            _matches_playback_prefix(transcript, playback.text)
            for playback in recent_playback
        )

    def _build_interrupt_checker(
        self, response: str
    ) -> tuple[Callable[[], bool], Callable[[], BargeIn | None]]:
        self.resume_microphone()
        self.pre_roll.clear()
        self.wake_model.reset()
        speech_streak = 0
        interruption: BargeIn | None = None
        wake_barge_enabled = SPOKEN_WAKE_PATTERN.search(response) is None
        if self.audio.barge_in_mode == "wake" and not wake_barge_enabled:
            log(
                "wake barge-in suppressed because the response contains the wake phrase"
            )

        def should_interrupt() -> bool:
            nonlocal speech_streak, interruption
            frame = self.read_frame()
            self.pre_roll.append(frame)
            if self.audio.barge_in_mode == "off":
                return False
            if self.audio.barge_in_mode == "vad":
                speech_streak = speech_streak + 1 if self.is_speech(frame) else 0
                detected = speech_streak >= self.audio.barge_in_speech_frames
                woke = False
            else:
                samples = self.np.frombuffer(frame, dtype="<i2")
                score = float(self.wake_model.predict(samples).get(self.wake_key, 0.0))
                detected = wake_barge_enabled and score >= self.audio.wake_threshold
                woke = True
            if detected:
                interruption = BargeIn(initial=list(self.pre_roll), woke=woke)
                self.pre_roll.clear()
                log(f"barge-in detected ({self.audio.barge_in_mode})")
                return True
            return False

        def interruption_result() -> BargeIn | None:
            return interruption

        return should_interrupt, interruption_result

    def _drain_playback_queue(
        self,
        response: str,
        *,
        on_poll: Callable[[], None] | None = None,
        on_played: Callable[[dict[str, object], bool, PlaybackRequest], None]
        | None = None,
    ) -> tuple[list[tuple[dict[str, object], bool, PlaybackRequest]], BargeIn | None]:
        should_interrupt, interruption_result = self._build_interrupt_checker(response)
        try:
            if on_poll is None:
                batch = self.playback_queue.drain(
                    should_interrupt=should_interrupt,
                    on_played=on_played,
                )
            else:
                batch = self.playback_queue.drain(
                    should_interrupt=should_interrupt,
                    on_poll=on_poll,
                    on_played=on_played,
                )
        finally:
            self.wake_model.reset()
        interruption = interruption_result()
        for playback, _interrupted, _request in batch:
            self._remember_recent_playback(str(playback.get("played_text") or ""))
        if interruption is None and batch and not batch[-1][1]:
            self.wait_for_playback_quiet()
        return batch, interruption

    def _render_speech(self, text: str) -> str:
        renderer = getattr(self, "speech_renderer", None)
        return (
            renderer.render(text)
            if renderer is not None
            else SpeechRenderer().render(text)
        )

    def play_response(
        self, response: ResponseLike
    ) -> tuple[dict[str, object], BargeIn | None]:
        spoken_text = self._render_speech(as_assistant_response(response).spoken_text)
        self.playback_queue.enqueue(PlaybackRequest(text=spoken_text))
        finished: set[int] = set()

        def finish_job(
            playback: dict[str, object],
            interrupted: bool,
            request: PlaybackRequest,
        ) -> None:
            if request.job_id:
                self._finish_job_playback(request, playback, interrupted=interrupted)
            finished.add(id(request))

        batch, interruption = self._drain_playback_queue(
            spoken_text,
            on_played=finish_job,
        )
        for playback, interrupted, request in batch:
            if id(request) not in finished:
                finish_job(playback, interrupted, request)
        if not batch:
            return {}, interruption
        return batch[-1][0], interruption

    def play_streamed_response(
        self,
        generate: Callable[
            [Callable[[str], bool], Callable[[], bool]],
            tuple[str, str | None],
        ],
    ) -> tuple[str, str | None, dict[str, object], BargeIn | None]:
        condition = threading.Condition()
        finished = False
        stopped = False
        chunks: list[str] = []
        played: list[dict[str, object]] = []
        playback_errors: list[BaseException] = []
        interruption: BargeIn | None = None
        first_streamed_request = True
        stream_renderer = StreamingSpeechRenderer(
            getattr(self, "speech_renderer", SpeechRenderer())
        )

        def enqueue_streamed_speech(spoken_text: str) -> None:
            nonlocal first_streamed_request
            self.playback_queue.enqueue(
                PlaybackRequest(
                    text=spoken_text,
                    skip_first_speed=first_streamed_request,
                    preflight_speed=first_streamed_request,
                )
            )
            first_streamed_request = False

        def on_text_chunk(text: str) -> bool:
            nonlocal stopped
            if not text.strip():
                return True
            with condition:
                if stopped:
                    return False
                chunks.append(text.strip())
                for spoken_text in stream_renderer.feed(text):
                    enqueue_streamed_speech(spoken_text)
                condition.notify()
                return True

        def flush_text_chunks() -> None:
            with condition:
                if stopped:
                    return
                for spoken_text in stream_renderer.flush():
                    enqueue_streamed_speech(spoken_text)
                condition.notify()

        def should_cancel_generation() -> bool:
            with condition:
                return stopped

        def player() -> None:
            nonlocal stopped, interruption
            try:
                while True:
                    with condition:
                        condition.wait_for(
                            lambda: len(self.playback_queue) > 0 or finished
                        )
                        if len(self.playback_queue) == 0 and finished:
                            return
                        response_so_far = " ".join(chunks)
                    batch, current_interruption = self._drain_playback_queue(
                        response_so_far
                    )
                    played.extend(result for result, _interrupted, _request in batch)
                    if current_interruption is not None:
                        interruption = current_interruption
                        with condition:
                            stopped = True
                            self.playback_queue.clear()
                        return
            except BaseException as exc:
                playback_errors.append(exc)
                with condition:
                    stopped = True
                    self.playback_queue.clear()

        playback_thread = threading.Thread(
            target=player,
            name="voice-streamed-playback",
            daemon=True,
        )
        playback_thread.start()
        try:
            response, next_cursor_session = generate(
                on_text_chunk,
                should_cancel_generation,
            )
            flush_text_chunks()
        finally:
            with condition:
                finished = True
                condition.notify_all()
            playback_thread.join()
        if playback_errors:
            error = playback_errors[0]
            raise SpeechDeliveryError(f"speech delivery failed: {error}") from error
        if not chunks and interruption is None:
            try:
                playback, interruption = self.play_response(response)
            except Exception as exc:
                raise SpeechDeliveryError(f"speech delivery failed: {exc}") from exc
            return (
                response,
                next_cursor_session,
                playback,
                interruption,
            )
        played_text = " ".join(
            str(result.get("played_text") or "").strip() for result in played
        ).strip()
        playback = {
            "ok": True,
            "stage": "tts",
            "interrupted": interruption is not None,
            "played_text": played_text,
        }
        return response, next_cursor_session, playback, interruption

    def _job_response(self, job: CursorJob) -> AssistantResponse:
        return cursor_service.render_job_announcement(job)

    def _enqueue_job_announcement(self, claim: DeliveryClaim) -> None:
        self._enqueue_announcement_batch((claim,))

    def _enqueue_announcement_batch(self, claims: tuple[DeliveryClaim, ...]) -> None:
        if not claims:
            return
        if len(claims) == 1:
            job = claims[0].job
            response = self._job_response(job)
        else:
            response = announcement_policy.render_digest(
                [claim.job for claim in claims],
                render_job=cursor_service.render_job_announcement,
            )
        log(f"job announcement queued: {response.display_text}")
        print(f"Assistant: {response.display_text}", flush=True)
        primary = claims[0]
        extra = tuple(
            (claim.job.id, claim.token, claim.job.status.value) for claim in claims[1:]
        )
        self.playback_queue.enqueue(
            PlaybackRequest(
                text=self._render_speech(response.spoken_text),
                job_id=primary.job.id,
                delivery_token=primary.token,
                job_status=primary.job.status.value,
                job_completed_at=primary.job.completed_at,
                display_fingerprint=_display_fingerprint(response.display_text),
                extra_claims=extra,
            )
        )

    def _enable_post_job_conversation(
        self,
        *,
        job_id: str,
        job_status: str,
        played_text: str,
        job_completed_at: float | None = None,
        display_fingerprint: str | None = None,
    ) -> None:
        if played_text:
            self.history.append({"role": "assistant", "content": played_text})
            self.history = self.history[-8:]
        if job_status == "awaiting_user":
            self.cursor_session = job_id
        elif job_status == "completed":
            # A job that was awaiting clarification and then completes must give
            # up the clarification slot and take the completed slot atomically.
            if self.cursor_session == job_id:
                self.cursor_session = None
            self._remember_completed_job(
                job_id,
                expected_completed_at=job_completed_at,
                display_fingerprint=display_fingerprint,
            )
        self.conversation_deadline = time.monotonic() + CONVERSATION_TIMEOUT_SECONDS
        self.awaiting_followup = True
        notify("Listening for a follow-up…")

    def _remember_completed_job(
        self,
        job_id: str,
        *,
        expected_completed_at: float | None = None,
        display_fingerprint: str | None = None,
    ) -> None:
        """Install the most recently played completed job as follow-up context.

        Called only after a successful, uninterrupted announcement was
        acknowledged, so the reference reflects work the user just heard about.
        The last successfully announced completion wins.
        """
        if not self.platform.cursor_followup_enabled:
            return
        try:
            job = CURSOR_STORE.get(job_id)
        except HarnessError:
            return
        except Exception as exc:  # noqa: BLE001 - never let context tracking crash a turn
            log(f"follow-up context skipped for {job_id}: {type(exc).__name__}: {exc}")
            return
        if (
            expected_completed_at is not None
            and job.completed_at != expected_completed_at
        ):
            return
        self.completed_followup = CompletedFollowup(
            job_id=job_id,
            parent_revision=job.revision,
            completed_at=job.completed_at,
            expires_at=(
                time.monotonic() + self.platform.cursor_followup_window_seconds
            ),
            display_fingerprint=display_fingerprint,
        )
        created_url = job.github_pr_created_url
        created = (
            github_pull_request_from_url(created_url)
            if isinstance(created_url, str)
            else None
        )
        if created is not None and job.github_pr_created_number == created.number:
            self.conversation_created_pull_request = created
        log(f"follow-up context retained for completed job {job_id}")

    def _active_completed_followup(self) -> CompletedFollowup | None:
        """Return the live completed reference, clearing it once it has expired."""
        followup = self.completed_followup
        if followup is None:
            return None
        if time.monotonic() >= followup.expires_at:
            self.completed_followup = None
            return None
        return followup

    def _active_last_transcript(self) -> LastTranscript | None:
        """Return the live last-transcript slot, clearing it once it has expired."""
        slot = self.last_transcript
        if slot is None:
            return None
        if time.monotonic() >= slot.expires_at:
            self.last_transcript = None
            return None
        return slot

    def _remember_last_transcript(self, utterance: str) -> None:
        text = utterance.strip()
        if not text:
            return
        self.last_transcript = LastTranscript(
            utterance=text,
            dispatched=False,
            expires_at=time.monotonic() + CONVERSATION_TIMEOUT_SECONDS,
        )

    def _refresh_last_transcript_deadline(self) -> None:
        slot = self.last_transcript
        if slot is None:
            return
        default_deadline = time.monotonic() + CONVERSATION_TIMEOUT_SECONDS
        self.last_transcript = LastTranscript(
            utterance=slot.utterance,
            dispatched=slot.dispatched,
            expires_at=max(slot.expires_at, default_deadline),
        )

    def _mark_last_transcript_dispatched(self) -> None:
        slot = self._active_last_transcript()
        if slot is None:
            return
        self.last_transcript = LastTranscript(
            utterance=slot.utterance,
            dispatched=True,
            expires_at=slot.expires_at,
        )

    def _dispatch_cursor_turn(self, request: CursorTurnRequest, **kwargs):
        existing_on_started = request.on_job_started

        def mutation_started() -> None:
            if existing_on_started is not None:
                existing_on_started()
            self._mark_last_transcript_dispatched()

        guarded_request = replace(request, on_job_started=mutation_started)
        result = cursor_turn(guarded_request, **kwargs)
        response, session = result[0], result[1]
        if getattr(result, "mutated", False) or request.action in {
            "cancel",
            "dismiss",
            "repeat",
        }:
            # These synchronous operations return only after their durable update.
            self._mark_last_transcript_dispatched()
        return response, session

    def _focused_identity_from_context(
        self, context: RequestContext
    ) -> FocusedIdentity:
        issue = context.focused_issue or context.external_issue_reference
        if issue is None and context.github_issue is not None:
            repository = context.github_repository or context.focused_repository
            issue = (
                f"{repository}#{context.github_issue}"
                if repository
                else str(context.github_issue)
            )
        pull_request = None
        if context.github_pull_request is not None:
            repository = context.github_repository or context.focused_repository
            pull_request = (
                f"{repository}#{context.github_pull_request}"
                if repository
                else str(context.github_pull_request)
            )
        return FocusedIdentity(
            repository=context.focused_repository or context.github_repository,
            issue=issue,
            pull_request=pull_request,
            app_class=context.focused_app_class,
            source_kinds=context.focused_app_sources,
        )

    def _bind_awaiting_job(self, job_id: str, *, missing: str) -> str:
        """Load a fresh snapshot and bind the session without answering."""
        try:
            job = CURSOR_STORE.get(job_id)
        except Exception as exc:  # noqa: BLE001 - resume/retarget must fail closed
            log(f"awaiting job unavailable for {job_id}: {type(exc).__name__}: {exc}")
            return missing
        if job.status != JobStatus.AWAITING_USER:
            return cursor_service.job_status(job.id)
        question = cursor_questions.current(job)
        if question is None:
            return cursor_service.job_status(job.id)
        self.cursor_session = job.id
        return question.text

    def _retarget_named_question(self, reference: str) -> str:
        """Switch the live session to a named awaiting job without answering."""
        if (
            not self.conversation_deadline
            or time.monotonic() >= self.conversation_deadline
            or self._pending_cursor_question() is None
        ):
            self.pending_question_retarget = None
            return RETARGET_INACTIVE_RESPONSE
        jobs = CURSOR_STORE.list()
        resolution = cursor_inbox.resolve_reference(jobs, reference)
        if resolution.ambiguous:
            self.pending_question_retarget = PendingQuestionRetarget(
                tuple(match.id for match in resolution.matches),
                time.monotonic(),
            )
            return cursor_inbox.clarify(list(resolution.matches), "talk about")
        if resolution.unique is None:
            self.pending_question_retarget = None
            return RETARGET_NOT_FOUND_RESPONSE
        self.pending_question_retarget = None
        return self._bind_awaiting_job(
            resolution.unique.id, missing=RETARGET_NOT_FOUND_RESPONSE
        )

    def _resume_awaiting_question(self, reference: str | None) -> str:
        """Replay a durable awaiting question without submitting an answer."""
        if reference:
            jobs = CURSOR_STORE.list()
            resolution = cursor_inbox.resolve_reference(jobs, reference)
            if resolution.ambiguous:
                return cursor_inbox.clarify(list(resolution.matches), "resume")
            if resolution.unique is None:
                return RETARGET_NOT_FOUND_RESPONSE
            return self._bind_awaiting_job(
                resolution.unique.id, missing=RETARGET_NOT_FOUND_RESPONSE
            )
        awaiting = [
            job
            for job in CURSOR_STORE.list()
            if job.status == JobStatus.AWAITING_USER
            and cursor_questions.current(job) is not None
        ]
        if not awaiting:
            return RESUME_NONE_RESPONSE
        if len(awaiting) > 1:
            return cursor_inbox.clarify(
                [cursor_inbox.summarize(job) for job in awaiting],
                "resume",
            )
        return self._bind_awaiting_job(
            awaiting[0].id, missing=RETARGET_NOT_FOUND_RESPONSE
        )

    def _resolve_pending_question_retarget(self, reference: str) -> str | None:
        pending = self.pending_question_retarget
        if pending is None:
            return None
        if time.monotonic() >= pending.expires_at:
            self.pending_question_retarget = None
            return None
        if (
            not self.conversation_deadline
            or time.monotonic() >= self.conversation_deadline
            or self._pending_cursor_question() is None
        ):
            self.pending_question_retarget = None
            return RETARGET_INACTIVE_RESPONSE

        live_jobs = []
        for job_id in pending.candidate_ids:
            try:
                job = CURSOR_STORE.get(job_id)
            except Exception:  # noqa: BLE001 - stale candidates fail closed
                continue
            if (
                job.status == JobStatus.AWAITING_USER
                and cursor_questions.current(job) is not None
            ):
                live_jobs.append(job)
        if not live_jobs:
            self.pending_question_retarget = None
            return RETARGET_NOT_FOUND_RESPONSE

        resolution = cursor_inbox.resolve_reference(live_jobs, reference)
        if resolution.unique is not None:
            self.pending_question_retarget = None
            return self._retarget_named_question(resolution.unique.id)

        summaries = (
            list(resolution.matches)
            if resolution.ambiguous
            else [cursor_inbox.summarize(job) for job in live_jobs]
        )
        self.pending_question_retarget = PendingQuestionRetarget(
            tuple(summary.id for summary in summaries),
            time.monotonic(),
        )
        return cursor_inbox.clarify(summaries, "talk about")

    def _active_announcement_snooze(
        self, now: float | None = None
    ) -> announcement_policy.AnnouncementSnooze | None:
        snooze = self.announcement_snooze
        if snooze is None:
            return None
        current = time.time() if now is None else now
        if not snooze.active(current):
            self.announcement_snooze = None
            return None
        return snooze

    def _followup_listening_armed(self) -> bool:
        return bool(
            self.awaiting_followup
            or (
                self.conversation_deadline
                and time.monotonic() < self.conversation_deadline
            )
        )

    def _extend_conversation_hold(self) -> bool:
        if self.hold_extensions >= MAX_HOLD_EXTENSIONS:
            return False
        self.hold_extensions += 1
        now = time.monotonic()
        base = self.conversation_deadline if self.conversation_deadline > now else now
        self.conversation_deadline = base + HOLD_EXTENSION_SECONDS
        pending = self.pending_target_readback
        if pending is not None:
            self.pending_target_readback = PendingTargetReadback(
                replace(
                    pending.candidate,
                    created_at=pending.candidate.created_at + HOLD_EXTENSION_SECONDS,
                ),
                pending.request,
            )
        resolution = self.pending_target_resolution
        if resolution is not None:
            self.pending_target_resolution = replace(
                resolution,
                created_at=resolution.created_at + HOLD_EXTENSION_SECONDS,
            )
        retarget = self.pending_question_retarget
        if retarget is not None:
            self.pending_question_retarget = replace(
                retarget,
                created_at=retarget.created_at + HOLD_EXTENSION_SECONDS,
            )
        slot = self.last_transcript
        if slot is not None:
            self.last_transcript = LastTranscript(
                utterance=slot.utterance,
                dispatched=slot.dispatched,
                expires_at=slot.expires_at + HOLD_EXTENSION_SECONDS,
            )
        return True

    def _capture_request_context(self, text: str) -> RequestContext:
        context = request_context(
            text,
            platform=self.platform,
            integrations=self.integrations,
            include_focused_context=not self.omit_focused_context,
        )
        self.last_focused_identity = (
            None
            if self.omit_focused_context
            else self._focused_identity_from_context(context)
        )
        return context

    def _speak_control_notice(self, spoken: str) -> BargeIn | None:
        self.ensure_components()
        response = AssistantResponse.from_text(spoken)
        print(f"Assistant: {response.display_text}", flush=True)
        _playback, interruption = self.play_response(response)
        if interruption is not None:
            return interruption
        self.awaiting_followup = True
        default_deadline = time.monotonic() + CONVERSATION_TIMEOUT_SECONDS
        self.conversation_deadline = max(
            self.conversation_deadline,
            default_deadline,
        )
        self._refresh_last_transcript_deadline()
        notify("Listening for a follow-up…")
        return None

    def _recent_completion_details(
        self,
        followup: CompletedFollowup,
    ) -> AssistantResponse:
        if self.completed_followup is not followup:
            return AssistantResponse.from_text(RECENT_DETAILS_UNAVAILABLE)
        try:
            job = CURSOR_STORE.get(followup.job_id)
        except Exception as exc:  # noqa: BLE001 - retrieval must fail closed
            log(
                "recent completion details unavailable for "
                f"{followup.job_id}: {type(exc).__name__}: {exc}"
            )
            self.completed_followup = None
            return AssistantResponse.from_text(RECENT_DETAILS_UNAVAILABLE)
        if (
            job.status != JobStatus.COMPLETED
            or job.completed_at != followup.completed_at
            or followup.display_fingerprint is None
        ):
            self.completed_followup = None
            return AssistantResponse.from_text(RECENT_DETAILS_UNAVAILABLE)
        rendered = cursor_service.render_job_announcement(job)
        if _display_fingerprint(rendered.display_text) != followup.display_fingerprint:
            self.completed_followup = None
            return AssistantResponse.from_text(RECENT_DETAILS_UNAVAILABLE)
        self.completed_followup = None
        return AssistantResponse(
            spoken_text="I've displayed the details from that completed Cursor job.",
            display_text=rendered.display_text,
        )

    def _has_announceable_jobs(self) -> bool:
        if self._active_completed_followup() is not None:
            return True
        return any(
            job.status in cursor_service.ANNOUNCEABLE_STATUSES and not job.delivered
            for job in CURSOR_STORE.list()
        )

    def _pending_cursor_question(self) -> PendingQuestionSnapshot | None:
        """Load one immutable question snapshot for routing and answer fencing."""
        if self.cursor_session is None:
            return None
        try:
            job = CURSOR_STORE.get(self.cursor_session)
        except Exception as exc:  # noqa: BLE001 - routing must fail closed
            log(
                "clarification context unavailable for "
                f"{self.cursor_session}: {type(exc).__name__}: {exc}"
            )
            return None
        if job.status != JobStatus.AWAITING_USER:
            return None
        question = cursor_questions.current(job)
        if question is None:
            return None
        return PendingQuestionSnapshot(
            job_id=job.id,
            text=question.text,
            owner=question.owner,
            question_id=question.id,
            turn_token=question.origin.turn_token,
            question=question,
        )

    def _finish_job_playback(
        self,
        request: PlaybackRequest,
        playback: dict[str, object],
        *,
        interrupted: bool,
    ) -> None:
        tokened: list[tuple[str, str, str]] = []
        if request.job_id and request.delivery_token:
            tokened.append(
                (request.job_id, request.delivery_token, str(request.job_status or ""))
            )
        tokened.extend(request.extra_claims)
        if interrupted:
            for job_id, delivery_token, _status in tokened:
                release_delivery(job_id, delivery_token)
            return
        acknowledged_all = True
        for job_id, delivery_token, _status in tokened:
            if not acknowledge_delivery(job_id, delivery_token):
                acknowledged_all = False
                release_delivery(job_id, delivery_token)
        if tokened and not acknowledged_all:
            return
        played_text = str(playback.get("played_text") or "").strip() or request.text
        if not request.extra_claims:
            if request.job_id:
                self._enable_post_job_conversation(
                    job_id=request.job_id,
                    job_status=str(request.job_status or ""),
                    played_text=played_text,
                    job_completed_at=request.job_completed_at,
                    display_fingerprint=request.display_fingerprint,
                )
            return
        if played_text:
            self.history.append({"role": "assistant", "content": played_text})
            self.history = self.history[-8:]
        awaiting = [
            job_id for job_id, _token, status in tokened if status == "awaiting_user"
        ]
        if len(awaiting) == 1:
            self.cursor_session = awaiting[0]
        self.conversation_deadline = time.monotonic() + CONVERSATION_TIMEOUT_SECONDS
        self.awaiting_followup = True
        notify("Listening for a follow-up…")

    def _play_pending_announcements(self) -> BargeIn | None:
        if len(self.playback_queue) == 0:
            return None
        self.pause_microphone()
        batch: list[tuple[dict[str, object], bool, PlaybackRequest]] = []
        interruption: BargeIn | None = None
        finished: set[tuple[str, str] | int] = set()
        with self.playback_queue._lock:
            pending_requests = [request for request, _ in self.playback_queue._items]
        lease_guard = _DeliveryLeaseGuard(pending_requests)

        def request_key(request: PlaybackRequest) -> tuple[str, str] | int:
            if request.job_id and request.delivery_token:
                return request.job_id, request.delivery_token
            return id(request)

        def release_unfinished(request: PlaybackRequest) -> None:
            if request_key(request) in finished:
                return
            if request.delivery_token and request.job_id:
                release_delivery(request.job_id, request.delivery_token)
            for job_id, token, _status in request.extra_claims:
                release_delivery(job_id, token)

        def finish_job(
            playback: dict[str, object],
            interrupted: bool,
            request: PlaybackRequest,
        ) -> None:
            lease_guard.complete(request)
            if request.job_id:
                self._finish_job_playback(request, playback, interrupted=interrupted)
            finished.add(request_key(request))

        try:
            lease_guard.start()
            with self.component_lock:
                start_components(self.providers)
                lease_guard.maintain()
                self.playback_queue.start_prefetch(limit=DELIVERY_WINDOW)
            batch, interruption = self._drain_playback_queue(
                self.playback_queue.queued_text(),
                on_poll=lease_guard.maintain,
                on_played=finish_job,
            )
            for playback, interrupted, request in batch:
                if request_key(request) not in finished:
                    finish_job(playback, interrupted, request)
            if interruption is not None:
                # Barge-in must not leave later announcements ahead of the
                # user's response. Release their durable claims so they can be
                # announced again only after the interrupted turn completes.
                self.playback_queue.clear()
                for request in pending_requests:
                    release_unfinished(request)
            return interruption
        except Exception as exc:
            self.playback_queue.clear()
            for request in pending_requests:
                release_unfinished(request)
            log(f"queued playback failed: {type(exc).__name__}: {exc}")
            notify(PLAYBACK_FAILURE, error=True)
            return None
        finally:
            try:
                lease_guard.stop()
            finally:
                if not self.conversation_deadline and interruption is None:
                    self.stop_components_when_idle()
                self.resume_microphone()
                self.wake_model.reset()

    def continue_after_barge_in(self, interruption: BargeIn | None) -> None:
        while interruption is not None and self.running:
            audio_path = self.record_utterance_safely(
                interruption.initial,
                wait_for_fresh_speech=interruption.woke,
            )
            if audio_path is None:
                return
            interruption = self.process_utterance(audio_path, woke=interruption.woke)

    def _resolve_pending_config_confirmation(
        self,
        text: str,
        *,
        blocked: bool,
        before_mutation: Callable[[], None] | None = None,
        after_mutation: Callable[[], None] | None = None,
    ) -> tuple[
        AssistantResponse | None, ConfirmationDecision, PendingConfigChange | None
    ]:
        pending = self.pending_config_change
        if pending is None or blocked:
            return None, ConfirmationDecision.AMBIGUOUS, pending
        decision = resolve_confirmation(text)
        if decision == ConfirmationDecision.AMBIGUOUS:
            return None, decision, pending
        self.pending_config_change = None
        if decision == ConfirmationDecision.CANCEL:
            return (
                AssistantResponse.from_text(
                    "Okay, I cancelled that configuration change. Nothing was written."
                ),
                decision,
                pending,
            )
        try:
            if before_mutation is not None:
                before_mutation()
            result = commit_pending_change(pending)
            self._mark_last_transcript_dispatched()
        except StaleConfigChangeError:
            if after_mutation is not None:
                after_mutation()
            response = AssistantResponse.from_text(
                "The stored value changed before confirmation, so I didn't write "
                "anything. Please start the change again."
            )
        except Exception as exc:  # noqa: BLE001 - report a bounded write failure
            log(f"confirmed configuration write failed: {type(exc).__name__}: {exc}")
            response = AssistantResponse.from_text(
                "I couldn't save that configuration change, so I didn't write "
                "anything. The running configuration snapshot is unchanged; "
                "please start the change again."
            )
        else:
            activation_error: ActivationStateError | None = None
            offer = None
            try:
                expected_config = apply_config_values(
                    self.user_config,
                    {pending.setting.value: pending.raw_value},
                )
                offer = self.config_activation_store.create_offer(
                    pending,
                    result,
                    expected_config=expected_config,
                )
            except ActivationStateError as exc:
                activation_error = exc
            finally:
                if after_mutation is not None:
                    after_mutation()
            if activation_error is not None:
                log(f"activation offer persistence failed: {activation_error}")
                response = AssistantResponse.from_text(
                    f"Saved {pending.setting.value}, but I could not durably track "
                    "activation. The running snapshot is unchanged and no service "
                    "was restarted."
                )
            else:
                if offer is None:
                    response = render_change_committed(pending, result)
                else:
                    delivery = ActivationDelivery(
                        offer,
                        ActivationDeliveryKind.OFFER,
                    )
                    self.config_activation_delivery = delivery
                    response = render_activation_delivery(delivery)
        return response, decision, pending

    def _focused_alias_identity(
        self,
    ) -> tuple[str | None, str | None, str | None]:
        """Return focused fragment source and references, never from STT."""

        try:
            url = focused_browser_url()
            fragment = (
                capture_context(url, self.integrations) if url is not None else None
            )
        except Exception:
            fragment = None
        if fragment is None:
            try:
                fragment = focused_herdr_github_context(self.integrations)
            except Exception:
                fragment = None
        if fragment is None:
            return None, None, None
        source = fragment.source.strip() if fragment.source else None
        return source, fragment.repository_reference, fragment.issue_reference

    def _just_used_alias_identity(
        self,
    ) -> tuple[str | None, str | None, str | None]:
        """Return fragment identity from the live just-used completed job."""

        followup = self._active_completed_followup()
        if followup is None:
            return None, None, None
        try:
            job = CURSOR_STORE.get(followup.job_id)
        except Exception:
            return None, None, None
        if (
            job.status != JobStatus.COMPLETED
            or job.completed_at != followup.completed_at
        ):
            return None, None, None
        if job.issue_key:
            source = job.issue_provider.strip() if job.issue_provider else None
            return source, None, job.issue_key
        if job.github_repository or job.github_issue:
            repository = job.github_repository
            issue = (
                f"{job.github_repository}#{job.github_issue}"
                if job.github_repository and job.github_issue
                else None
            )
            return "github", repository, issue
        return None, None, None

    def _trusted_alias_identity(self) -> tuple[str | None, str | None, str | None]:
        """Prefer focused fragment identity; fall back to just-used when absent."""

        source, repository, issue = self._focused_alias_identity()
        if source or repository or issue:
            return source, repository, issue
        return self._just_used_alias_identity()

    def _resolve_pending_spoken_alias_confirmation(
        self,
        text: str,
        *,
        blocked: bool,
        before_mutation: Callable[[], None] | None = None,
        after_mutation: Callable[[], None] | None = None,
    ) -> tuple[
        AssistantResponse | None, ConfirmationDecision, PendingSpokenAlias | None
    ]:
        pending = self.pending_spoken_alias
        if pending is None or blocked:
            return None, ConfirmationDecision.AMBIGUOUS, pending
        decision = resolve_confirmation(text)
        if decision == ConfirmationDecision.AMBIGUOUS:
            return None, decision, pending
        self.pending_spoken_alias = None
        if decision == ConfirmationDecision.CANCEL:
            return (
                AssistantResponse.from_text(
                    "Okay, I cancelled that alias. Nothing was written."
                ),
                decision,
                pending,
            )
        if pending.existing_target is not None and not pending.replace:
            replacement = PendingSpokenAlias(
                trusted_utterance=pending.trusted_utterance,
                phrase=pending.phrase,
                target=pending.target,
                kind=pending.kind,
                source=pending.source,
                existing_target=pending.existing_target,
                replace=True,
            )
            self.pending_spoken_alias = replacement
            return (
                render_spoken_alias_preparation(
                    SpokenAliasPreparation(SpokenAliasStatus.READY, replacement)
                ),
                decision,
                pending,
            )
        if pending.replace:
            current = load_vocabulary().alias_for(pending.phrase)
            current_target = None if current is None else current.target
            if current_target != pending.existing_target:
                if current_target == pending.target:
                    return (
                        render_spoken_alias_preparation(
                            SpokenAliasPreparation(SpokenAliasStatus.NO_CHANGE)
                        ),
                        decision,
                        pending,
                    )
                refreshed = PendingSpokenAlias(
                    trusted_utterance=pending.trusted_utterance,
                    phrase=pending.phrase,
                    target=pending.target,
                    kind=pending.kind,
                    existing_target=current_target,
                    replace=current_target is not None,
                )
                self.pending_spoken_alias = refreshed
                return (
                    render_spoken_alias_preparation(
                        SpokenAliasPreparation(SpokenAliasStatus.READY, refreshed)
                    ),
                    decision,
                    pending,
                )
        try:
            if before_mutation is not None:
                before_mutation()
            commit_spoken_alias(
                pending,
                force=pending.replace,
                integrations=self.integrations,
            )
        except Exception as exc:  # noqa: BLE001 - report a bounded write failure
            if after_mutation is not None:
                after_mutation()
            log(f"confirmed alias write failed: {type(exc).__name__}: {exc}")
            return (
                AssistantResponse.from_text(
                    "I couldn't save that alias, so I didn't write anything."
                ),
                decision,
                pending,
            )
        if after_mutation is not None:
            after_mutation()
        return render_spoken_alias_committed(pending), decision, pending

    def _resolve_activation_confirmation(
        self,
        text: str,
        *,
        before_mutation: Callable[[], None] | None = None,
        after_mutation: Callable[[], None] | None = None,
    ) -> AssistantResponse | None:
        record = self.config_activation_store.current()
        if (
            record is None
            or record.status != ActivationStatus.OFFERED
            or not record.offer_delivered
        ):
            return None
        decision = resolve_activation_decision(text)
        if decision == ActivationDecision.ACTIVATE:
            if before_mutation is not None:
                before_mutation()
            record = self.config_activation_store.accept(record.id)
            if after_mutation is not None:
                after_mutation()
            delivery = ActivationDelivery(
                record,
                ActivationDeliveryKind.PRE_RESTART,
            )
            self.config_activation_delivery = delivery
            return render_activation_delivery(delivery)
        if decision == ActivationDecision.DECLINE:
            if before_mutation is not None:
                before_mutation()
            record = self.config_activation_store.decline(record.id)
            if after_mutation is not None:
                after_mutation()
            delivery = ActivationDelivery(record, ActivationDeliveryKind.RESULT)
            self.config_activation_delivery = delivery
            return render_activation_delivery(delivery)
        if (
            resolve_confirmation(text) == ConfirmationDecision.CONFIRM
            or question_control(text) == AnswerOutcome.REPEAT
        ):
            return AssistantResponse.from_text(
                "This activation needs a separate explicit confirmation. "
                "Say activate now to restart the affected service, or not now "
                "to leave it unchanged."
            )
        return None

    def _dispatch_ready_config_activation(self, record_id: str) -> None:
        if record_id in self.launched_config_activations:
            return
        attempts = self.config_activation_dispatch_attempts.get(record_id, 0) + 1
        self.config_activation_dispatch_attempts[record_id] = attempts
        try:
            process = launch_activation_worker(record_id)
        except OSError as exc:
            current = self.config_activation_store.current()
            if (
                current is not None
                and current.id == record_id
                and current.setting_key == "audio.voice"
                and current.status
                in {
                    ActivationStatus.READY,
                    ActivationStatus.VALIDATING,
                    ActivationStatus.ROLLING_BACK,
                }
            ):
                if attempts >= 2 and current.status in {
                    ActivationStatus.READY,
                    ActivationStatus.VALIDATING,
                }:
                    self.config_activation_store.begin_rollback(
                        record_id,
                        "The isolated activation worker could not start after "
                        f"{attempts} attempts: {exc}",
                    )
                    return
                log(
                    "could not relaunch voice activation reconciliation worker: "
                    f"{type(exc).__name__}: {exc}"
                )
                return
            if (
                current is not None
                and current.id == record_id
                and current.setting_key == "audio.voice"
                and current.status == ActivationStatus.RESTARTING
            ):
                self.config_activation_store.begin_rollback(
                    record_id,
                    "The isolated activation worker could not be relaunched after "
                    f"voice restart state was recorded: {exc}",
                )
                return
            self.config_activation_store.fail_worker(
                record_id,
                f"Could not launch the isolated activation worker: {exc}",
            )
            return
        self.launched_config_activations[record_id] = process

    def _complete_config_activation_delivery(
        self,
        delivery: ActivationDelivery,
    ) -> None:
        if delivery.kind == ActivationDeliveryKind.OFFER:
            self.config_activation_store.mark_offer_delivered(delivery.record.id)
        elif delivery.kind == ActivationDeliveryKind.PRE_RESTART:
            ready = self.config_activation_store.mark_pre_restart_delivered(
                delivery.record.id
            )
            self._dispatch_ready_config_activation(ready.id)
        else:
            self.config_activation_store.acknowledge(delivery.record.id)
        if self.config_activation_delivery == delivery:
            self.config_activation_delivery = None

    def _recover_config_activation(
        self,
        *,
        actions_allowed: bool = True,
    ) -> BargeIn | None:
        for record_id, process in tuple(self.launched_config_activations.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            del self.launched_config_activations[record_id]
            current = self.config_activation_store.current()
            if (
                current is not None
                and current.id == record_id
                and current.status
                in {
                    ActivationStatus.READY,
                    ActivationStatus.VALIDATING,
                    ActivationStatus.RESTARTING,
                    ActivationStatus.ROLLING_BACK,
                }
                and self.config_activation_dispatch_attempts.get(record_id, 0) >= 2
            ):
                detail = (
                    "The isolated activation worker could not reconcile the durable "
                    f"request (status {return_code})."
                )
                if current.setting_key == "audio.voice":
                    if current.status != ActivationStatus.ROLLING_BACK:
                        self.config_activation_store.begin_rollback(record_id, detail)
                else:
                    self.config_activation_store.fail_worker(record_id, detail)
        record = self.config_activation_store.current()
        dispatchable_statuses = (
            {
                ActivationStatus.READY,
                ActivationStatus.VALIDATING,
                ActivationStatus.RESTARTING,
                ActivationStatus.ROLLING_BACK,
            }
            if actions_allowed
            else {
                ActivationStatus.VALIDATING,
                ActivationStatus.ROLLING_BACK,
            }
        )
        if record is not None and record.status in dispatchable_statuses:
            self._dispatch_ready_config_activation(record.id)
        if not actions_allowed:
            return None
        delivery = self.config_activation_store.next_delivery()
        if delivery is None:
            return None
        self.ensure_components()
        response = render_activation_delivery(delivery)
        print(f"Assistant: {response.display_text}", flush=True)
        _playback, interruption = self.play_response(response)
        if interruption is None:
            self._complete_config_activation_delivery(delivery)
        return interruption

    def _resolve_pending_target_readback(
        self,
        text: str,
        pending: PendingTargetReadback,
        context: RequestContext,
    ) -> tuple[AssistantResponse | None, CursorTurnRequest | None]:
        resolution = resolve_readback(
            pending.candidate,
            text,
            context,
        )
        self.pending_target_readback = None
        if resolution.reply == ReadbackReply.AFFIRMATIVE:
            return None, pending.request
        if resolution.reply == ReadbackReply.CORRECTION:
            assert resolution.replacement is not None
            replacement = new_candidate(
                TargetSelection(
                    resolution.replacement,
                    pending.candidate.context_binding,
                ),
                origin_turn=uuid.uuid4().hex,
            )
            self.pending_target_readback = PendingTargetReadback(
                replacement,
                _critical_target_request(resolution.replacement),
            )
            return readback_response(replacement), None
        if resolution.reply == ReadbackReply.NEGATIVE:
            return AssistantResponse.from_text("Okay, I didn't start that work."), None
        if resolution.reply == ReadbackReply.EXPIRED:
            return (
                AssistantResponse.from_text(
                    "That confirmation expired because the target context changed. "
                    "Please repeat the request."
                ),
                None,
            )
        return None, None

    def process_utterance(  # pyright: ignore[reportGeneralTypeIssues]
        self,
        audio_path: Path | None,
        *,
        woke: bool,
        retained: RetainedTranscript | None = None,
    ) -> BargeIn | None:
        had_active_conversation = bool(self.conversation_deadline)
        delivery_claims: DeliveryClaims = []
        transcript_delivery = retained
        delivery_ambiguous = retained is not None and retained.state == "ambiguous"
        delivery_terminal = retained is not None and retained.state == "terminal"
        preserve_delivery = False
        turn_failed = False
        next_cursor_session = self.cursor_session

        def fence_side_effect() -> None:
            nonlocal delivery_ambiguous, preserve_delivery
            if transcript_delivery is None or delivery_ambiguous:
                return
            preserve_delivery = True
            try:
                transcript_delivery.mark_ambiguous()
            except Exception:
                self.retained_recovery_required = True
                log(
                    "voice turn replay fence failed; retained evidence will be "
                    f"retried in-process: {transcript_delivery.delivery_id}"
                )
                raise
            delivery_ambiguous = True
            self.retained_recovery_required = True

        def terminalize_non_side_effect() -> None:
            nonlocal delivery_terminal, preserve_delivery
            if transcript_delivery is None or delivery_terminal:
                return
            preserve_delivery = True
            try:
                transcript_delivery.mark_terminal()
            except Exception:
                self.retained_recovery_required = True
                self.retained_recovery_retry_at = (
                    time.monotonic() + RETAINED_RECOVERY_RETRY_SECONDS
                )
                log(
                    "voice turn terminalization failed before safe completion; "
                    f"retained evidence will be retried: "
                    f"{transcript_delivery.delivery_id}"
                )
                raise
            delivery_terminal = True
            preserve_delivery = False

        self.config_activation_delivery = None
        recent_playback = self._active_recent_playback() if not woke else ()
        self.pause_microphone()
        try:
            try:
                if transcript_delivery is None:
                    assert audio_path is not None
                    transcription = transcribe(audio_path, woke=woke)
                    if isinstance(transcription, RetainedTranscript):
                        transcript_delivery = transcription
                        text = transcription.text
                    else:  # compatibility for injected transcription test doubles
                        text = str(transcription)
                else:
                    text = transcript_delivery.text
                if transcript_delivery is not None and transcript_delivery.state in {
                    "ambiguous",
                    "uncertain",
                }:
                    preserve_delivery = True
                    turn_failed = True
                    self.retained_recovery_required = True
                    self.retained_recovery_retry_at = (
                        time.monotonic() + RETAINED_RECOVERY_RETRY_SECONDS
                    )
                    log(
                        "voice turn retention acknowledgment requires reconciliation: "
                        f"{transcript_delivery.delivery_id} "
                        f"({transcript_delivery.state})"
                    )
                    notify(
                        "I retained that voice request because its handoff was "
                        "interrupted. I will not run it until recovery confirms "
                        "whether it is safe.",
                        error=True,
                    )
                    return None
                if text.startswith("__DICTATION_ERROR__:"):
                    terminalize_non_side_effect()
                    raise HarnessError(text.removeprefix("__DICTATION_ERROR__:"))
                if not text:
                    terminalize_non_side_effect()
                    raise NoSpeechError("STT did not recognize any speech")
            except NoSpeechError:
                pending_target_resolution = getattr(
                    self, "pending_target_resolution", None
                )
                if (
                    getattr(self, "pending_target_readback", None) is None
                    and pending_target_resolution is None
                    and getattr(self, "pending_config_change", None) is None
                    and getattr(self, "pending_spoken_alias", None) is None
                ):
                    raise
                terminalize_non_side_effect()
                log("confirmation reply contained no recognizable speech")
                self.awaiting_followup = True
                self.conversation_deadline = (
                    time.monotonic() + CONVERSATION_TIMEOUT_SECONDS
                )
                if pending_target_resolution is not None:
                    self.conversation_deadline = min(
                        self.conversation_deadline,
                        pending_target_resolution.expires_at,
                    )
                if getattr(self, "pending_target_readback", None) is not None:
                    notify(
                        "I didn't catch that. Please repeat yes, no, or the correction."
                    )
                elif pending_target_resolution is not None:
                    notify(TARGET_RESOLUTION_CONTEXT_RESPONSE.spoken_text)
                else:
                    notify("I didn't catch that. Please repeat yes or no.")
                return None
            if woke:
                text, found_wake = strip_wake_prefix(text)
                if not found_wake:
                    if text:
                        log(
                            "wake prefix absent in STT; trusting OpenWakeWord: "
                            f"{text!r}"
                        )
                    else:
                        terminalize_non_side_effect()
                        log(f"rejected wake candidate: {text!r}")
                        self.stop_components_when_idle()
                        return None
            if self._is_playback_echo(text, recent_playback):
                terminalize_non_side_effect()
                log("rejected follow-up matching recent local playback")
                self.awaiting_followup = True
                return None
            if not text:
                terminalize_non_side_effect()
                log("wake phrase contained no request; waiting for follow-up")
                self.awaiting_followup = True
                self.conversation_deadline = (
                    time.monotonic() + CONVERSATION_TIMEOUT_SECONDS
                )
                return None
            self.awaiting_followup = False
            log(f"user: {text}")
            if CLOSE_PATTERN.search(text):
                terminalize_non_side_effect()
                self.close_conversation("spoken command")
                return None
            pending_target_resolution = getattr(self, "pending_target_resolution", None)
            if (
                pending_target_resolution is not None
                and time.monotonic() >= pending_target_resolution.expires_at
            ):
                self.pending_target_resolution = None
                pending_target_resolution = None
            if STOP_TALKING_PATTERN.search(text):
                terminalize_non_side_effect()
                self.playback_queue.clear()
                self.awaiting_followup = True
                self.conversation_deadline = (
                    time.monotonic() + CONVERSATION_TIMEOUT_SECONDS
                )
                notify("Listening for a follow-up…")
                return None
            if TRANSCRIPT_REPLAY_PATTERN.search(text):
                terminalize_non_side_effect()
                slot = self._active_last_transcript()
                if slot is None:
                    return self._speak_control_notice(MISSING_TRANSCRIPT_RESPONSE)
                return self._speak_control_notice(f"I heard: {slot.utterance}")
            if TRANSCRIPT_CORRECTION_BARE_PATTERN.search(text):
                terminalize_non_side_effect()
                if self._active_last_transcript() is None:
                    return self._speak_control_notice(MISSING_TRANSCRIPT_RESPONSE)
                return self._speak_control_notice(BARE_TRANSCRIPT_CORRECTION_RESPONSE)
            correction = TRANSCRIPT_CORRECTION_PATTERN.search(text)
            if correction is not None:
                replacement = correction.group(1).strip()
                slot = self._active_last_transcript()
                if slot is None or not replacement:
                    terminalize_non_side_effect()
                    return self._speak_control_notice(MISSING_TRANSCRIPT_RESPONSE)
                if slot.dispatched:
                    terminalize_non_side_effect()
                    return self._speak_control_notice(DISPATCHED_TRANSCRIPT_RESPONSE)
                # Abandon confirmations that belonged to the replaced utterance.
                # The replacement re-enters the same routing path and must still
                # satisfy ticket readback, create, fork, clone, and config checks.
                self.pending_target_readback = None
                self.pending_target_resolution = None
                self.pending_config_change = None
                text = replacement
            if INSPECT_CONTEXT_PATTERN.search(text):
                terminalize_non_side_effect()
                if self.omit_focused_context:
                    return self._speak_control_notice(NO_FOCUSED_CONTEXT_RESPONSE)
                captured = self._capture_request_context(text)
                identity = self._focused_identity_from_context(captured)
                self.last_focused_identity = identity
                return self._speak_control_notice(identity.spoken())
            if OMIT_CONTEXT_PATTERN.search(text):
                terminalize_non_side_effect()
                self.omit_focused_context = True
                self.last_focused_identity = None
                return self._speak_control_notice(OMIT_FOCUSED_CONTEXT_RESPONSE)
            if HOLD_PATTERN.search(text):
                terminalize_non_side_effect()
                if not self._followup_listening_armed():
                    return self._speak_control_notice(HOLD_INACTIVE_RESPONSE)
                spoken = (
                    HOLD_ACCEPTED_RESPONSE
                    if self._extend_conversation_hold()
                    else HOLD_EXHAUSTED_RESPONSE
                )
                self.ensure_components()
                response = AssistantResponse.from_text(spoken)
                print(f"Assistant: {response.display_text}", flush=True)
                _playback, interruption = self.play_response(response)
                if interruption is not None:
                    return interruption
                self.awaiting_followup = True
                notify("Listening for a follow-up…")
                return None
            if CLEAR_SNOOZE_PATTERN.search(text):
                terminalize_non_side_effect()
                had_snooze = self._active_announcement_snooze() is not None
                self.announcement_snooze = None
                return self._speak_control_notice(
                    SNOOZE_CLEARED_RESPONSE if had_snooze else SNOOZE_INACTIVE_RESPONSE
                )
            snooze_match = SNOOZE_PATTERN.search(text)
            if snooze_match is not None:
                terminalize_non_side_effect()
                minutes = snooze_match.group("minutes")
                parsed_minutes = int(minutes) if minutes else None
                duration = (
                    parsed_minutes * 60
                    if parsed_minutes is not None
                    else DEFAULT_SNOOZE_SECONDS
                )
                target = (snooze_match.group("target") or "").casefold()
                mute_everything = (
                    target == "everything"
                    or "don" in text.casefold()
                    and "talk" in text.casefold()
                )
                self.announcement_snooze = announcement_policy.AnnouncementSnooze(
                    until=time.time() + duration,
                    mute_everything=mute_everything,
                )
                return self._speak_control_notice(
                    SNOOZE_MUTE_ALL_RESPONSE
                    if mute_everything
                    else snooze_started_response(parsed_minutes)
                )
            pending_retarget_response = self._resolve_pending_question_retarget(text)
            if pending_retarget_response is not None:
                terminalize_non_side_effect()
                return self._speak_control_notice(pending_retarget_response)
            retarget = RETARGET_PATTERN.search(text)
            if retarget is not None:
                reference = (
                    retarget.group("ref") or retarget.group("qref") or ""
                ).strip()
                if reference:
                    terminalize_non_side_effect()
                    self.pending_target_readback = None
                    self.pending_target_resolution = None
                    self.pending_config_change = None
                    return self._speak_control_notice(
                        self._retarget_named_question(reference)
                    )
            resume = RESUME_PATTERN.search(text)
            if resume is not None:
                reference = (
                    resume.group("ref")
                    or resume.group("pickref")
                    or resume.group("whereref")
                    or ""
                ).strip()
                with_prefix = bool(resume.group("with"))
                handle_resume = True
                if reference and re.search(
                    r"\b(?:work\s+on|fix|change|update|implement|add|remove|"
                    r"run|review|inspect|start|create|build|refactor|test)",
                    reference,
                    re.IGNORECASE,
                ):
                    handle_resume = False
                elif (
                    with_prefix
                    and reference
                    and self._pending_cursor_question() is not None
                ):
                    # "Continue with X" while a question is live is an answer,
                    # even when X also matches another awaiting job.
                    handle_resume = False
                elif with_prefix and reference:
                    resolution = cursor_inbox.resolve_reference(
                        CURSOR_STORE.list(), reference
                    )
                    if resolution.unique is None and not resolution.ambiguous:
                        handle_resume = False
                if handle_resume:
                    terminalize_non_side_effect()
                    self.pending_target_readback = None
                    self.pending_target_resolution = None
                    self.pending_config_change = None
                    return self._speak_control_notice(
                        self._resume_awaiting_question(reference or None)
                    )
            pending_resolution = getattr(self, "pending_target_resolution", None)
            resolution_active = (
                pending_resolution is not None
                and time.monotonic() < pending_resolution.expires_at
            )
            activation_record = self.config_activation_store.current()
            activation_offer_pending = (
                activation_record is not None
                and activation_record.status == ActivationStatus.OFFERED
                and activation_record.offer_delivered
            )
            if (
                question_control(text) == AnswerOutcome.REPEAT
                and self._pending_cursor_question() is None
                and not self._has_announceable_jobs()
                and self.pending_config_change is None
                and getattr(self, "pending_spoken_alias", None) is None
                and getattr(self, "pending_target_readback", None) is None
                and not resolution_active
                and not activation_offer_pending
            ):
                terminalize_non_side_effect()
                spoken = (
                    self.last_ordinary_reply
                    if self.last_ordinary_reply
                    else "I don't have a reply to repeat."
                )
                self.ensure_components()
                response = AssistantResponse.from_text(spoken)
                print(f"Assistant: {response.display_text}", flush=True)
                _playback, interruption = self.play_response(response)
                if interruption is not None:
                    return interruption
                self.awaiting_followup = True
                self.conversation_deadline = (
                    time.monotonic() + CONVERSATION_TIMEOUT_SECONDS
                )
                notify("Listening for a follow-up…")
                return None
            pending_target_resolution = getattr(self, "pending_target_resolution", None)
            if (
                pending_target_resolution is not None
                and time.monotonic() >= pending_target_resolution.expires_at
            ):
                self.pending_target_resolution = None
                pending_target_resolution = None
            resuming_target_resolution = bool(
                pending_target_resolution is not None
                and _is_target_resolution_continuation(text)
            )
            self.ensure_components()
            print(f"You: {text}", flush=True)
            if resuming_target_resolution:
                assert pending_target_resolution is not None
                text = pending_target_resolution.trusted_utterance
            next_cursor_session = self.cursor_session
            next_history = list(self.history)
            remember_response = False
            ordinary_reply = False
            streamed_playback = False
            recommendation_playback = False
            playback: dict[str, object] = {}
            interruption: BargeIn | None = None
            readback_result: AssistantResponse | None = None
            confirmed_request: CursorTurnRequest | None = None
            pending_readback = getattr(self, "pending_target_readback", None)
            routing_context = RequestContext(text)
            context = (
                self._capture_request_context(text)
                if pending_readback is not None or resuming_target_resolution
                else routing_context
            )
            if (
                pending_readback is not None
                and question_control(text) == AnswerOutcome.REPEAT
            ):
                readback_result = readback_response(pending_readback.candidate)
            elif pending_readback is not None:
                (
                    readback_result,
                    confirmed_request,
                ) = self._resolve_pending_target_readback(
                    text,
                    pending_readback,
                    context,
                )
            (
                config_response,
                config_decision,
                pending_config,
            ) = self._resolve_pending_config_confirmation(
                text,
                blocked=pending_readback is not None,
                before_mutation=fence_side_effect,
                after_mutation=terminalize_non_side_effect,
            )
            (
                alias_response,
                alias_decision,
                pending_alias,
            ) = self._resolve_pending_spoken_alias_confirmation(
                text,
                blocked=pending_readback is not None or pending_config is not None,
                before_mutation=fence_side_effect,
                after_mutation=terminalize_non_side_effect,
            )
            activation_response = (
                self._resolve_activation_confirmation(
                    text,
                    before_mutation=fence_side_effect,
                    after_mutation=terminalize_non_side_effect,
                )
                if pending_readback is None
                and pending_config is None
                and config_response is None
                and pending_alias is None
                and alias_response is None
                else None
            )
            active_completed = self._active_completed_followup()
            pending = (
                None if resuming_target_resolution else self._pending_cursor_question()
            )
            if (
                readback_result is not None
                or confirmed_request is not None
                or config_response is not None
                or alias_response is not None
                or activation_response is not None
            ):
                route = IntentRoute(Intent.UNCERTAIN, "low")
            elif resuming_target_resolution:
                assert pending_target_resolution is not None
                route = pending_target_resolution.route
            elif parse_spoken_alias_request(text) is not None:
                route = IntentRoute(Intent.VOCABULARY_ALIAS_ADD, "high")
            elif (
                self.providers.llm_provider == "venice"
                and pending is None
                and active_completed is None
                and not needs_intent_router(text)
            ):
                route = IntentRoute(Intent.CONVERSATION, "high")
            else:
                route = route_intent(
                    text,
                    routing_context,
                    cursor_session=pending.job_id if pending is not None else None,
                    pending_question=pending.text if pending is not None else None,
                    clarification_kind=pending.owner if pending is not None else None,
                    recent_completion=active_completed is not None,
                    settings=self.providers,
                )
                apply_recommendation = (
                    pending_config is None
                    and cursor_consultation.is_apply_recommendation_request(text)
                )
                if apply_recommendation and route.intent != Intent.HARNESS_HELP:
                    route = IntentRoute(Intent.AGENT_REPLY, "high")
                if pending is not None and pending_config is None:
                    deterministic_answer = (
                        resolve_answer(
                            pending.question,
                            text,
                            provenance=AnswerProvenance.USER_VOICE,
                        ).outcome
                        if pending.question is not None
                        else None
                    )
                    resolved_as_answer = deterministic_answer in {
                        AnswerOutcome.REPEAT,
                        AnswerOutcome.DEFERRED,
                    } or (
                        deterministic_answer == AnswerOutcome.ACCEPTED
                        and pending.question is not None
                        and bool(pending.question.choices)
                    )
                    grouped_repository_answer = (
                        pending.owner == "grouped_repository"
                        and is_grouped_repository_mapping(text)
                    )
                    repository_list_answer = (
                        pending.owner == "repository"
                        and cursor_provisioning.is_repository_list_request(text)
                    )
                    if route.intent != Intent.HARNESS_HELP and (
                        resolved_as_answer
                        or apply_recommendation
                        or grouped_repository_answer
                        or repository_list_answer
                        or question_control(text) is not None
                    ):
                        route = IntentRoute(Intent.AGENT_REPLY, "high")
                    invalid_pending_reply = (
                        not resolved_as_answer
                        and not apply_recommendation
                        and route.intent == Intent.AGENT_REPLY
                        and (
                            (
                                pending.question is not None
                                and bool(pending.question.choices)
                            )
                            or _is_filler_speech(text)
                        )
                    )
                    implicit_submit = (
                        route.intent == Intent.AGENT_SUBMIT
                        and PENDING_SUBMIT_PATTERN.search(text) is None
                    )
                    if not woke and (
                        not route.actionable or invalid_pending_reply or implicit_submit
                    ):
                        # Follow-up VAD can capture nearby conversation. A pending
                        # structured question makes conversational fallback unsafe:
                        # close silently and leave the durable question untouched.
                        terminalize_non_side_effect()
                        self.close_pending_capture("non-actionable speech")
                        return None
            answering_existing = (
                readback_result is not None
                or confirmed_request is not None
                or config_response is not None
                or activation_response is not None
                or (pending is not None and route.intent == Intent.AGENT_REPLY)
            )
            if not answering_existing:
                self._remember_last_transcript(text)
            if confirmed_request is not None or (
                route.actionable and route.intent in SIDE_EFFECTING_INTENTS
            ):
                fence_side_effect()
            else:
                terminalize_non_side_effect()
            if (
                pending_target_resolution is not None
                and not resuming_target_resolution
                and route.actionable
            ):
                self.pending_target_resolution = None
            if (
                pending_config is not None
                and config_decision == ConfirmationDecision.AMBIGUOUS
                and config_response is None
            ):
                if route.intent == Intent.HARNESS_CONFIG_CHANGE:
                    self.pending_config_change = None
                elif (
                    route.actionable and question_control(text) != AnswerOutcome.REPEAT
                ):
                    self.pending_config_change = None
                else:
                    config_response = AssistantResponse.from_text(
                        "Please say yes to confirm that configuration change or no to "
                        "cancel it."
                    )
                    route = IntentRoute(Intent.UNCERTAIN, "low")
            if (
                pending_alias is not None
                and alias_decision == ConfirmationDecision.AMBIGUOUS
                and alias_response is None
            ):
                if route.intent == Intent.VOCABULARY_ALIAS_ADD:
                    self.pending_spoken_alias = None
                elif (
                    route.actionable and question_control(text) != AnswerOutcome.REPEAT
                ):
                    self.pending_spoken_alias = None
                else:
                    alias_response = AssistantResponse.from_text(
                        "Please say yes to confirm that alias or no to cancel it."
                    )
                    route = IntentRoute(Intent.UNCERTAIN, "low")
            if (
                pending_readback is None
                and not resuming_target_resolution
                and (
                    route.intent == Intent.CONVERSATION
                    or (
                        route.actionable
                        and route.intent
                        in {
                            Intent.AGENT_SUBMIT,
                            Intent.GITHUB_ISSUE_CREATE,
                            Intent.GITHUB_PR_CREATE,
                            Intent.GITHUB_PR_MERGE,
                            Intent.GITHUB_REPO_CREATE,
                            Intent.GITHUB_ORG_REPO_CREATE,
                            Intent.GITHUB_ISSUE_UPDATE,
                            Intent.GITHUB_ISSUE_CLOSE,
                            Intent.GITHUB_ISSUE_SPLIT,
                            Intent.LINEAR_TICKET_CREATE,
                            Intent.LINEAR_TICKET_UPDATE,
                            Intent.LINEAR_TICKET_CLOSE,
                            Intent.LINEAR_TICKET_SPLIT,
                            Intent.GITHUB_ISSUE_MERGE,
                            Intent.LINEAR_TICKET_MERGE,
                            Intent.WORKSPACE_CONSULTATION,
                        }
                    )
                    or cursor_consultation.wants_ticket_consultation_context(text)
                    or wants_ticket_update_context(text)
                    or wants_ticket_close_context(text)
                    or wants_ticket_split_context(text)
                    or wants_ticket_merge_context(text)
                )
            ):
                context = self._capture_request_context(text)
            if route.actionable and route.intent == Intent.END_CONVERSATION:
                return self.end_conversation()
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
            split_admission = admit_ticket_split(
                text,
                extraction,
                focused_issue=context.focused_issue,
            )
            merge_admission = admit_ticket_merge(
                text,
                extraction,
                focused_issue=context.focused_issue,
            )
            invalid_target_resolution = (
                resuming_target_resolution
                and not _has_exact_target_resolution(extraction, context)
            )
            if config_response is not None:
                response = config_response
            elif alias_response is not None:
                response = alias_response
            elif activation_response is not None:
                response = activation_response
            elif readback_result is not None:
                response = readback_result
            elif confirmed_request is not None:
                self.completed_followup = None
                response, next_cursor_session = self._dispatch_cursor_turn(
                    confirmed_request,
                    delivery_claims=delivery_claims,
                    integrations=self.integrations,
                )
            elif route.actionable and route.intent == Intent.GITHUB_PR_MERGE:
                created = self.conversation_created_pull_request
                identity = resolve_pull_request_merge_identity(
                    utterance=text,
                    focused_repository=context.github_repository,
                    focused_number=context.github_pull_request,
                    conversation_repository=(
                        created.name_with_owner if created is not None else None
                    ),
                    conversation_number=(
                        created.number if created is not None else None
                    ),
                )
                self.completed_followup = None
                response, next_cursor_session = cursor_turn(
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
                    integrations=self.integrations,
                )
            elif ticket_admission is not None:
                if ticket_admission.ticket is None:
                    response = ticket_admission.missing_identity_response
                    ordinary_reply = True
                else:
                    completed_job = None
                    if active_completed is not None:
                        try:
                            completed_job = CURSOR_STORE.get(active_completed.job_id)
                        except Exception:  # noqa: BLE001 - selection must fail closed
                            completed_job = None
                    try:
                        client = self.integrations.herdr_client()
                        target = cursor_consultation.workspace_target(
                            client,
                            focused_repository=context.focused_repository,
                            completed_job=completed_job,
                        )
                        if target is None:
                            response = cursor_consultation.NO_WORKSPACE
                        else:
                            interruption = self._acknowledge_consultation(text)
                            if interruption is not None:
                                return interruption
                            assert ticket_admission.ticket.canonical is not None
                            assert ticket_admission.ticket.source is not None
                            snapshot = ticket_snapshot(
                                ticket_admission.ticket.canonical,
                                self.integrations,
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
                        ordinary_reply = True
                    except Exception:  # noqa: BLE001 - consultation fails closed
                        response = cursor_consultation.CONSULTATION_FAILED
                        ordinary_reply = True
            elif update_admission is not None:
                if update_admission.ticket is None:
                    response = update_admission.missing_identity_response
                    ordinary_reply = True
                elif not route.actionable:
                    response = (
                        "I did not update a ticket because the request was unclear. "
                        "Please name the ticket and the title or body change."
                    )
                    ordinary_reply = True
                else:
                    self.completed_followup = None
                    dispatch = update_turn_arguments(update_admission.ticket)
                    response, next_cursor_session = cursor_turn(
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
                        integrations=self.integrations,
                    )
            elif close_admission is not None:
                if close_admission.ticket is None:
                    response = close_admission.missing_identity_response
                    ordinary_reply = True
                elif not route.actionable:
                    response = (
                        "I did not close a ticket because the request was unclear. "
                        "Please name the ticket to close."
                    )
                    ordinary_reply = True
                else:
                    self.completed_followup = None
                    dispatch = close_turn_arguments(close_admission.ticket)
                    response, next_cursor_session = cursor_turn(
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
                        integrations=self.integrations,
                    )
            elif split_admission is not None:
                if split_admission.ticket is None:
                    response = split_admission.missing_identity_response
                    ordinary_reply = True
                elif not route.actionable:
                    response = (
                        "I did not split a ticket because the request was unclear. "
                        "Please name the ticket to split."
                    )
                    ordinary_reply = True
                else:
                    self.completed_followup = None
                    dispatch = split_turn_arguments(split_admission.ticket)
                    response, next_cursor_session = cursor_turn(
                        CursorTurnRequest(
                            context.text,
                            utterance=text,
                            github_repository=dispatch.github_repository,
                            github_issue=dispatch.github_issue,
                            github_issue_split_requested=(
                                dispatch.github_issue_split_requested
                            ),
                            issue_key=dispatch.issue_key,
                            linear_ticket_split_requested=(
                                dispatch.linear_ticket_split_requested
                            ),
                        ),
                        delivery_claims=delivery_claims,
                        integrations=self.integrations,
                    )
            elif merge_admission is not None:
                if merge_admission.survivor is None:
                    response = merge_admission.missing_identity_response
                    ordinary_reply = True
                elif not route.actionable:
                    response = (
                        "I did not merge tickets because the request was unclear. "
                        "Please name at least two tickets to merge."
                    )
                    ordinary_reply = True
                else:
                    self.completed_followup = None
                    dispatch = merge_turn_arguments(
                        merge_admission.survivor, merge_admission.tickets
                    )
                    response, next_cursor_session = cursor_turn(
                        CursorTurnRequest(
                            context.text,
                            utterance=text,
                            github_repository=dispatch.github_repository,
                            github_issue=dispatch.github_issue,
                            github_issue_merge_requested=(
                                dispatch.github_issue_merge_requested
                            ),
                            issue_key=dispatch.issue_key,
                            linear_ticket_merge_requested=(
                                dispatch.linear_ticket_merge_requested
                            ),
                            ticket_merge_survivor=dispatch.ticket_merge_survivor,
                            ticket_merge_closing=dispatch.ticket_merge_closing,
                        ),
                        delivery_claims=delivery_claims,
                        integrations=self.integrations,
                    )
            elif route.actionable and route.intent == Intent.GITHUB_ISSUE_CREATE:
                self.completed_followup = None
                response, next_cursor_session = self._dispatch_cursor_turn(
                    CursorTurnRequest(
                        context.text,
                        utterance=text,
                        github_repository=(
                            context.github_repository or repository_from_utterance(text)
                        ),
                        github_issue_create_requested=True,
                    ),
                    delivery_claims=delivery_claims,
                    integrations=self.integrations,
                )
            elif route.actionable and route.intent == Intent.GITHUB_REPO_CREATE:
                self.completed_followup = None
                response, next_cursor_session = cursor_turn(
                    CursorTurnRequest(
                        context.text,
                        utterance=text,
                        github_repository=repository_from_utterance(text),
                        github_repo_create_requested=True,
                    ),
                    delivery_claims=delivery_claims,
                    integrations=self.integrations,
                )
            elif route.actionable and route.intent == Intent.GITHUB_ORG_REPO_CREATE:
                self.completed_followup = None
                response, next_cursor_session = cursor_turn(
                    CursorTurnRequest(
                        context.text,
                        utterance=text,
                        github_repository=repository_from_utterance(text),
                        github_repo_create_requested=True,
                        github_repo_create_org_requested=True,
                    ),
                    delivery_claims=delivery_claims,
                    integrations=self.integrations,
                )
            elif route.actionable and route.intent == Intent.LINEAR_TICKET_CREATE:
                self.completed_followup = None
                response, next_cursor_session = self._dispatch_cursor_turn(
                    CursorTurnRequest(
                        context.text,
                        utterance=text,
                        linear_team=(
                            context.issue_scope
                            if context.issue_scope_source == "linear"
                            else team_from_utterance(text)
                        ),
                        linear_ticket_create_requested=True,
                    ),
                    delivery_claims=delivery_claims,
                    integrations=self.integrations,
                )
            elif route.intent == Intent.HARNESS_CONFIG_INSPECT:
                response = (
                    inspect_config_utterance(text, self.user_config)
                    if route.actionable
                    else AssistantResponse.from_text(UNSUPPORTED_INSPECTION_RESPONSE)
                )
            elif route.intent == Intent.HARNESS_CONFIG_CHANGE:
                if route.actionable:
                    preparation = prepare_config_change(
                        ConfigChangeRequest(text, route.raw_value),
                        self.user_config,
                    )
                    self.pending_config_change = preparation.pending
                    self.pending_spoken_alias = None
                    response = render_change_preparation(preparation)
                else:
                    self.pending_config_change = None
                    response = AssistantResponse.from_text(
                        "I couldn't identify a safe configuration change, so I didn't "
                        "write anything."
                    )
            elif route.intent == Intent.VOCABULARY_ALIAS_ADD:
                if route.actionable:
                    source, focused_repository, focused_issue = (
                        self._trusted_alias_identity()
                    )
                    preparation = prepare_spoken_alias(
                        text,
                        focused_repository=focused_repository,
                        focused_issue=focused_issue,
                        source=source,
                        integrations=self.integrations,
                    )
                    self.pending_spoken_alias = preparation.pending
                    self.pending_config_change = None
                    response = render_spoken_alias_preparation(preparation)
                else:
                    self.pending_spoken_alias = None
                    response = AssistantResponse.from_text(
                        "I couldn't identify a safe repository alias, so I didn't "
                        "write anything."
                    )
            elif route.actionable and route.intent == Intent.SELF_HEALTH:
                response = self_health_response()
                ordinary_reply = True
            elif route.actionable and route.intent == Intent.HARNESS_HELP:
                response = harness_help_response()
                ordinary_reply = True
            elif invalid_target_resolution:
                response = TARGET_RESOLUTION_CONTEXT_RESPONSE
            elif missing_ticket_scope:
                if (
                    not resuming_target_resolution
                    and route.actionable
                    and route.intent == Intent.AGENT_SUBMIT
                    and extraction.requested_count == 1
                ):
                    self.pending_target_resolution = PendingTargetResolution(
                        text,
                        route,
                        time.monotonic(),
                    )
                response = MISSING_ISSUE_SCOPE_RESPONSE
            elif cursor_consultation.is_apply_recommendation_request(text):
                snapshot = cursor_consultation.pending_question_snapshot(
                    CURSOR_STORE,
                    pending.job_id if pending is not None else None,
                )
                choice_id = (
                    cursor_consultation.applicable_choice_id(
                        CURSOR_STORE, snapshot.job_id
                    )
                    if snapshot is not None
                    else None
                )
                if snapshot is None or choice_id is None:
                    response = cursor_consultation.RECOMMENDATION_UNAVAILABLE
                else:
                    response, next_cursor_session = self._dispatch_cursor_turn(
                        CursorTurnRequest(
                            choice_id,
                            snapshot.job_id,
                            utterance=choice_id,
                            action="reply",
                            job_id=snapshot.job_id,
                            expected_question_id=snapshot.question_id,
                            expected_question_turn=snapshot.turn_token,
                            answer_provenance=AnswerProvenance.USER_VOICE,
                        ),
                        delivery_claims=delivery_claims,
                        integrations=self.integrations,
                    )
            elif route.actionable and route.intent == Intent.QUESTION_CONSULTATION:
                snapshot = cursor_consultation.pending_question_snapshot(
                    CURSOR_STORE,
                    pending.job_id if pending is not None else None,
                )
                if snapshot is None:
                    response = cursor_consultation.NO_PENDING_QUESTION
                    ordinary_reply = True
                else:
                    try:
                        client = self.integrations.herdr_client()
                        interruption = self._acknowledge_consultation(text)
                        if interruption is not None:
                            return interruption
                        response = cursor_consultation.consult_pending_question(
                            client,
                            CURSOR_STORE,
                            snapshot,
                            context.text,
                        )
                        recommendation_playback = True
                        ordinary_reply = True
                    except Exception as exc:  # noqa: BLE001 - consultation fails closed
                        response = (
                            str(exc)
                            if isinstance(exc, HarnessError)
                            and str(exc) == cursor_consultation.STALE_PENDING_QUESTION
                            else cursor_consultation.CONSULTATION_FAILED
                        )
                        ordinary_reply = True
            elif route.actionable and route.intent == Intent.WORKSPACE_CONSULTATION:
                completed_job = None
                if active_completed is not None:
                    try:
                        completed_job = CURSOR_STORE.get(active_completed.job_id)
                    except Exception:  # noqa: BLE001 - selection must fail closed
                        completed_job = None
                try:
                    client = self.integrations.herdr_client()
                    target = cursor_consultation.workspace_target(
                        client,
                        focused_repository=context.focused_repository,
                        completed_job=completed_job,
                    )
                    if target is None:
                        response = cursor_consultation.NO_WORKSPACE
                    else:
                        interruption = self._acknowledge_consultation(text)
                        if interruption is not None:
                            return interruption
                        response = cursor_consultation.consult(
                            client, target, context.text
                        )
                    ordinary_reply = True
                except Exception:  # noqa: BLE001 - consultation fails closed
                    response = cursor_consultation.CONSULTATION_FAILED
                    ordinary_reply = True
            elif route.actionable and route.intent == Intent.AGENT_LIST:
                listed = cursor_turn(
                    CursorTurnRequest(
                        "",
                        self.cursor_session,
                        action="list",
                    ),
                    delivery_claims=delivery_claims,
                    integrations=self.integrations,
                )
                response, next_cursor_session = listed[0], listed[1]
            elif route.actionable and route.intent == Intent.ANNOUNCEMENT_DIGEST:
                missed = cursor_turn(
                    CursorTurnRequest(
                        "",
                        self.cursor_session,
                        action="missed",
                    ),
                    delivery_claims=delivery_claims,
                    integrations=self.integrations,
                )
                response, next_cursor_session = missed[0], missed[1]
            elif route.actionable and route.intent == Intent.AGENT_CANCEL:
                response, next_cursor_session = self._dispatch_cursor_turn(
                    CursorTurnRequest(
                        text,
                        self.cursor_session,
                        action="cancel",
                        job_id=self.cursor_session,
                        reference=text,
                    ),
                    delivery_claims=delivery_claims,
                    integrations=self.integrations,
                )
            elif route.actionable and route.intent == Intent.AGENT_STATUS:
                statused = cursor_turn(
                    CursorTurnRequest(
                        text,
                        self.cursor_session,
                        action="status",
                        job_id=self.cursor_session,
                        reference=text,
                    ),
                    delivery_claims=delivery_claims,
                    integrations=self.integrations,
                )
                response, next_cursor_session = statused[0], statused[1]
            elif route.actionable and route.intent in {
                Intent.AGENT_DISMISS,
                Intent.AGENT_REPEAT,
            }:
                action = (
                    "reply"
                    if route.intent == Intent.AGENT_REPEAT and pending is not None
                    else (
                        "dismiss" if route.intent == Intent.AGENT_DISMISS else "repeat"
                    )
                )
                response, next_cursor_session = self._dispatch_cursor_turn(
                    CursorTurnRequest(
                        text,
                        self.cursor_session,
                        utterance=text if action == "reply" else None,
                        action=action,
                        job_id=self.cursor_session,
                        reference=text,
                        expected_question_id=(
                            pending.question_id
                            if action == "reply" and pending is not None
                            else None
                        ),
                        expected_question_turn=(
                            pending.turn_token
                            if action == "reply" and pending is not None
                            else None
                        ),
                        answer_provenance=(
                            AnswerProvenance.USER_VOICE
                            if action == "reply"
                            else AnswerProvenance.USER_TEXT
                        ),
                    ),
                    delivery_claims=delivery_claims,
                    integrations=self.integrations,
                )
            elif (
                route.actionable
                and route.intent == Intent.AGENT_REPLY
                and pending is not None
            ):
                response, next_cursor_session = self._dispatch_cursor_turn(
                    CursorTurnRequest(
                        context.text,
                        pending.job_id,
                        utterance=text,
                        action="reply",
                        job_id=pending.job_id,
                        expected_question_id=(
                            pending.question_id if pending is not None else None
                        ),
                        expected_question_turn=(
                            pending.turn_token if pending is not None else None
                        ),
                        answer_provenance=AnswerProvenance.USER_VOICE,
                    ),
                    delivery_claims=delivery_claims,
                    integrations=self.integrations,
                )
            elif route.actionable and route.intent == Intent.AGENT_SUBMIT:
                # Explicit new work invalidates any retained completed-job slot.
                self.completed_followup = None
                selection = select_submit_target(extraction, context)
                if (
                    selection is not None
                    and selection.readback_required
                    and not fork_requested
                ):
                    candidate = new_candidate(
                        selection,
                        origin_turn=uuid.uuid4().hex,
                    )
                    self.pending_target_readback = PendingTargetReadback(
                        candidate,
                        _critical_target_request(candidate.target),
                    )
                    if resuming_target_resolution:
                        self.pending_target_resolution = None
                    response = readback_response(candidate)
                else:
                    response, next_cursor_session = self._dispatch_cursor_turn(
                        CursorTurnRequest(
                            context.text,
                            utterance=text,
                            context_repository=context.focused_repository,
                            issue_key=context.external_issue_reference,
                            issue_scope=context.issue_scope,
                            issue_scope_source=context.issue_scope_source,
                            **github_arguments,
                        ),
                        delivery_claims=delivery_claims,
                        integrations=self.integrations,
                    )
                    if (
                        selection is not None
                        and not selection.readback_required
                        and not fork_requested
                    ):
                        response = identified_target_response(
                            selection.target,
                            response,
                        )
                    if resuming_target_resolution:
                        self.pending_target_resolution = None
            elif route.intent == Intent.AGENT_SUBMIT:
                response = NON_ACTIONABLE_SUBMIT_RESPONSE
            elif route.intent == Intent.GITHUB_PR_MERGE:
                response = (
                    "I did not merge a pull request because the request was unclear."
                )
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
            elif route.actionable and route.intent == Intent.GITHUB_PR_CREATE:
                current_completed = self._active_completed_followup()
                if (
                    self.cursor_session is not None
                    or current_completed is None
                    or current_completed is not active_completed
                ):
                    response = (
                        "I don't have a recent completed job checkout to open a "
                        "pull request from."
                    )
                else:
                    log(
                        "pull-request create dispatched for completed job "
                        f"{current_completed.job_id}"
                    )

                    def consume_pr_create_followup() -> None:
                        if self.completed_followup is current_completed:
                            self.completed_followup = None

                    response, next_cursor_session = cursor_turn(
                        CursorTurnRequest(
                            context.text,
                            utterance=text,
                            action="follow_up",
                            job_id=current_completed.job_id,
                            expected_parent_revision=current_completed.parent_revision,
                            expected_completed_at=current_completed.completed_at,
                            github_pr_create_requested=True,
                            on_follow_up_started=consume_pr_create_followup,
                        ),
                        delivery_claims=delivery_claims,
                        integrations=self.integrations,
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
            elif route.intent == Intent.GITHUB_PR_CREATE:
                response = (
                    "I did not open a pull request because the request was unclear."
                )
            elif route.intent == Intent.AGENT_DETAILS:
                current_completed = self._active_completed_followup()
                if (
                    self.cursor_session is not None
                    or current_completed is None
                    or current_completed is not active_completed
                ):
                    response = RECENT_DETAILS_UNAVAILABLE
                else:
                    response = self._recent_completion_details(current_completed)
            elif route.actionable and route.intent == Intent.AGENT_FOLLOWUP:
                current_completed = self._active_completed_followup()
                if (
                    self.cursor_session is not None
                    or current_completed is None
                    or current_completed is not active_completed
                ):
                    response = (
                        "I don't have a recent completed Cursor job to follow up on."
                    )
                else:
                    log(
                        "follow-up dispatched for completed job "
                        f"{current_completed.job_id}"
                    )

                    def consume_completed_followup() -> None:
                        # Consume only after the child is durably created. A busy
                        # checkout is retryable while this context remains live.
                        if self.completed_followup is current_completed:
                            self.completed_followup = None

                    response, next_cursor_session = self._dispatch_cursor_turn(
                        CursorTurnRequest(
                            context.text,
                            utterance=text,
                            action="follow_up",
                            job_id=current_completed.job_id,
                            expected_parent_revision=current_completed.parent_revision,
                            expected_completed_at=current_completed.completed_at,
                            on_follow_up_started=consume_completed_followup,
                        ),
                        delivery_claims=delivery_claims,
                        integrations=self.integrations,
                    )
            else:
                # The authoritative router handles every mutating action above.
                # Conversation fallback is always tool-free.
                if self.providers.llm_provider == "venice":
                    (
                        response,
                        next_cursor_session,
                        playback,
                        interruption,
                    ) = self.play_streamed_response(
                        lambda on_text_chunk, should_cancel: qwen_turn(
                            context.text,
                            self.history,
                            self.cursor_session,
                            **github_arguments,
                            trusted_utterance=text,
                            delivery_claims=delivery_claims,
                            on_text_chunk=on_text_chunk,
                            should_cancel=should_cancel,
                            allow_tools=False,
                            settings=self.providers,
                        )
                    )
                    streamed_playback = True
                else:
                    response, next_cursor_session = qwen_turn(
                        context.text,
                        self.history,
                        self.cursor_session,
                        **github_arguments,
                        trusted_utterance=text,
                        delivery_claims=delivery_claims,
                        allow_tools=False,
                        settings=self.providers,
                    )
                remember_response = True
                ordinary_reply = True
            rendered_response = as_assistant_response(response)
            if ordinary_reply and rendered_response.spoken_text:
                self.last_ordinary_reply = rendered_response.spoken_text
            print(f"Assistant: {rendered_response.display_text}", flush=True)
            cursor_session_before_playback = self.cursor_session
            if not streamed_playback:
                try:
                    playback, interruption = self.play_response(rendered_response)
                except Exception as exc:
                    raise SpeechDeliveryError(f"speech delivery failed: {exc}") from exc
            if remember_response:
                played_text = (
                    str(playback.get("played_text") or "").strip()
                    if playback.get("interrupted")
                    else rendered_response.spoken_text
                )
                next_history.append({"role": "user", "content": text})
                if played_text:
                    next_history.append({"role": "assistant", "content": played_text})
                next_history = next_history[-8:]
            if recommendation_playback:
                cursor_consultation.complete_recommendation_delivery(
                    summary=rendered_response.spoken_text,
                    interrupted=interruption is not None,
                )
            if interruption is not None:
                release_deliveries(delivery_claims)
                self.history = next_history
                return interruption
            if self.config_activation_delivery is not None:
                self._complete_config_activation_delivery(
                    self.config_activation_delivery
                )
            acknowledged = acknowledge_deliveries(delivery_claims)
            completed_claims = [
                claim
                for claim in acknowledged
                if claim.job.status == JobStatus.COMPLETED
            ]
            if len(completed_claims) == 1:
                completed = completed_claims[0].job
                self._remember_completed_job(
                    completed.id,
                    expected_completed_at=completed.completed_at,
                    display_fingerprint=_display_fingerprint(
                        rendered_response.display_text
                    ),
                )
            if self.cursor_session == cursor_session_before_playback:
                self.cursor_session = next_cursor_session
            self.history = next_history
            default_deadline = time.monotonic() + CONVERSATION_TIMEOUT_SECONDS
            self.conversation_deadline = max(
                self.conversation_deadline,
                default_deadline,
            )
            self._refresh_last_transcript_deadline()
            unresolved_target = getattr(self, "pending_target_resolution", None)
            if unresolved_target is not None:
                self.conversation_deadline = min(
                    self.conversation_deadline,
                    unresolved_target.expires_at,
                )
            self.awaiting_followup = True
            notify("Listening for a follow-up…")
        except SpeechDeliveryError as exc:
            turn_failed = True
            release_deliveries(delivery_claims)
            self.cursor_session = next_cursor_session
            log(f"speech delivery failed: {type(exc).__name__}: {exc}")
            notify(SPEECH_DELIVERY_FAILURE, error=True)
            self.awaiting_followup = False
            if not had_active_conversation:
                self.conversation_deadline = 0.0
                self.stop_components_when_idle()
        except NoSpeechError as exc:
            turn_failed = True
            release_deliveries(delivery_claims)
            pending_confirmation = (
                getattr(self, "pending_target_readback", None) is not None
                or getattr(self, "pending_target_resolution", None) is not None
                or getattr(self, "pending_config_change", None) is not None
            )
            if (
                had_active_conversation
                and not pending_confirmation
                and self._pending_cursor_question() is not None
            ):
                log(
                    "no recognizable speech after announced question; "
                    "closing ambient capture"
                )
                self.close_pending_capture(
                    "no recognizable speech after announced question"
                )
            elif had_active_conversation:
                log(
                    "follow-up contained no recognizable speech; listening remains armed"
                )
                self.awaiting_followup = True
            else:
                log(f"turn failed: NoSpeechError: {exc}")
                notify(VOICE_REQUEST_FAILURE, error=True)
                self.awaiting_followup = False
                self.history.clear()
                self.cursor_session = None
                self.conversation_deadline = 0.0
                self.stop_components_when_idle()
        except Exception as exc:
            turn_failed = True
            release_deliveries(delivery_claims)
            log(f"turn failed: {type(exc).__name__}: {exc}")
            notify(VOICE_REQUEST_FAILURE, error=True)
            self.awaiting_followup = False
            if not had_active_conversation:
                self.history.clear()
                self.cursor_session = None
                self.conversation_deadline = 0.0
                self.stop_components_when_idle()
        finally:
            if transcript_delivery is not None and not (
                turn_failed and (delivery_ambiguous or preserve_delivery)
            ):
                try:
                    if not delivery_terminal:
                        transcript_delivery.mark_terminal()
                except Exception as exc:  # noqa: BLE001 - retained evidence survives
                    self.retained_recovery_required = True
                    self.retained_recovery_retry_at = (
                        time.monotonic() + RETAINED_RECOVERY_RETRY_SECONDS
                    )
                    log(
                        "could not durably terminalize STT delivery "
                        f"{transcript_delivery.delivery_id}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                else:
                    try:
                        transcript_delivery.release()
                    except Exception as exc:  # noqa: BLE001 - terminal is durable
                        self.retained_recovery_required = True
                        self.retained_recovery_retry_at = (
                            time.monotonic() + RETAINED_RECOVERY_RETRY_SECONDS
                        )
                        log(
                            "could not clean up terminal STT delivery "
                            f"{transcript_delivery.delivery_id}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    else:
                        self.retained_recovery_required = False
                        self.retained_recovery_retry_at = 0.0
            self.resume_microphone()
            self.wake_model.reset()
        return None

    def _recover_retained_utterances(self) -> None:
        for delivery in recover_retained_transcripts():
            if delivery.state == "terminal":
                try:
                    delivery.release()
                except Exception as exc:  # noqa: BLE001 - retry ordered cleanup
                    self.retained_recovery_required = True
                    self.retained_recovery_retry_at = (
                        time.monotonic() + RETAINED_RECOVERY_RETRY_SECONDS
                    )
                    log(
                        "could not clean up terminal retained voice delivery "
                        f"{delivery.delivery_id}: {type(exc).__name__}: {exc}"
                    )
                    break
                continue
            if delivery.state == "ambiguous":
                self.retained_recovery_required = True
                self.retained_recovery_retry_at = (
                    time.monotonic() + RETAINED_RECOVERY_RETRY_SECONDS
                )
                log(
                    "retained voice turn requires reconciliation before retry: "
                    f"{delivery.delivery_id}"
                )
                notify(
                    "A previous voice request may have started before interruption. "
                    "I retained it for reconciliation and will not run it again.",
                    error=True,
                )
                break
            self.process_utterance(
                None,
                woke=delivery.woke,
                retained=delivery,
            )
            if getattr(self, "retained_recovery_required", False):
                log(
                    "retained voice recovery paused before later deliveries "
                    "until reconciliation succeeds"
                )
                break

    def _retry_retained_recovery(self) -> bool:
        if not getattr(self, "retained_recovery_required", False):
            return True
        now = time.monotonic()
        if now < getattr(self, "retained_recovery_retry_at", 0.0):
            return False
        self.retained_recovery_required = False
        try:
            self._recover_retained_utterances()
        except Exception as exc:
            self.retained_recovery_required = True
            self.retained_recovery_retry_at = now + RETAINED_RECOVERY_RETRY_SECONDS
            log(
                f"in-process retained turn recovery failed: {type(exc).__name__}: {exc}"
            )
            notify(
                "A retained voice request still needs recovery. I will keep "
                "actions paused and retry.",
                error=True,
            )
        return not self.retained_recovery_required

    def run(self) -> None:
        recover_jobs(integrations=self.integrations)
        self.retained_recovery_required = True
        self._retry_retained_recovery()
        self.start_microphone()
        speech_streak = 0
        while self.running:
            actions_allowed = self._retry_retained_recovery()
            activation_interruption = self._recover_config_activation(
                actions_allowed=actions_allowed
            )
            if actions_allowed:
                if activation_interruption is not None:
                    self.continue_after_barge_in(activation_interruption)
                    speech_streak = 0
                    continue
                batch = drain_pending_announcements(
                    self.announcements,
                    integrations=self.integrations,
                    snooze=self._active_announcement_snooze(),
                )
                if batch.speak:
                    self._enqueue_announcement_batch(batch.speak)
                if len(self.playback_queue) > 0:
                    self.continue_after_barge_in(self._play_pending_announcements())
                    speech_streak = 0
                    continue
            frame = self.read_frame()
            now = time.monotonic()
            self.pre_roll.append(frame)
            if self.conversation_deadline and now >= self.conversation_deadline:
                self.close_conversation("inactivity")
                speech_streak = 0
                continue
            if not actions_allowed:
                if self.force_listen.is_set():
                    self.force_listen.clear()
                    notify(
                        "Voice actions are paused while a retained request is "
                        "reconciled.",
                        error=True,
                    )
                samples = self.np.frombuffer(frame, dtype="<i2")
                score = float(self.wake_model.predict(samples).get(self.wake_key, 0.0))
                if score >= self.audio.wake_threshold and now - self.last_wake >= 2.0:
                    self.last_wake = now
                    log(f"wake detected while retained recovery is paused: {score:.3f}")
                    notify(
                        "Wake detected, but voice actions are paused while a "
                        "retained request is reconciled.",
                        error=True,
                    )
                speech_streak = 0
                continue
            if self.force_listen.is_set():
                self.force_listen.clear()
                log("force-listen requested")
                initial = list(self.pre_roll)
                self.pre_roll.clear()
                speech_streak = 0
                audio_path = self.record_utterance_safely(initial)
                if audio_path is None:
                    continue
                notify("Listening…")
                self.begin_activation()
                self.continue_after_barge_in(
                    self.process_utterance(audio_path, woke=False)
                )
                continue
            if self.awaiting_followup:
                speech_streak = speech_streak + 1 if self.is_speech(frame) else 0
                if speech_streak >= 5:
                    log("follow-up speech detected")
                    initial = list(self.pre_roll)
                    self.pre_roll.clear()
                    speech_streak = 0
                    audio_path = self.record_utterance_safely(initial)
                    if audio_path is None:
                        speech_streak = 0
                        continue
                    self.continue_after_barge_in(
                        self.process_utterance(audio_path, woke=False)
                    )
                continue
            samples = self.np.frombuffer(frame, dtype="<i2")
            score = float(self.wake_model.predict(samples).get(self.wake_key, 0.0))
            if score >= self.audio.wake_threshold and now - self.last_wake >= 2.0:
                self.last_wake = now
                log(f"wake detected: score={score:.3f}")
                initial = list(self.pre_roll)
                self.pre_roll.clear()
                audio_path = self.record_utterance_safely(
                    initial,
                    wait_for_fresh_speech=True,
                )
                if audio_path is None:
                    speech_streak = 0
                    continue
                notify("Wake detected — listening…")
                self.begin_activation()
                self.continue_after_barge_in(
                    self.process_utterance(audio_path, woke=True)
                )

    def stop(self) -> None:
        self.running = False
        self.pending_target_resolution = None
        self.pending_question_retarget = None
        self.pending_config_change = None
        self.pending_spoken_alias = None
        if self.microphone is not None and self.microphone.poll() is None:
            if self.microphone_paused:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(self.microphone.pid, signal.SIGCONT)
                self.microphone_paused = False
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.microphone.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.microphone.wait(timeout=2)


def _read_wake_state() -> tuple[int, str] | None:
    try:
        raw = WAKE_PID_PATH.read_text().strip()
    except OSError:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    pid = value.get("pid")
    start = value.get("process_start")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    if not isinstance(start, str) or not start:
        return None
    return pid, start


def request_listen() -> None:
    """Ask a running wake daemon to start a conversation without the wake word."""
    state = _read_wake_state()
    if state is None:
        raise HarnessError("wake daemon is not running")
    pid, start = state
    handle = ProcessHandle.open(pid, expected_start=start)
    if handle is None:
        raise HarnessError("wake daemon is not running")
    try:
        handle.send_signal(signal.SIGUSR1)
    except ProcessLookupError as exc:
        raise HarnessError("wake daemon is not running") from exc
    finally:
        handle.close()


def _acquire_wake_singleton() -> int:
    WAKE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(WAKE_LOCK, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise HarnessError("wake daemon is already running") from exc
    return descriptor


def _write_pidfile() -> None:
    identity = process_identity(os.getpid())
    if identity is None:
        raise HarnessError("could not establish wake daemon process identity")
    WAKE_PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = WAKE_PID_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"pid": os.getpid(), "process_start": identity}, separators=(",", ":")
        )
        + "\n"
    )
    temporary.chmod(0o600)
    os.replace(temporary, WAKE_PID_PATH)


def _remove_pidfile() -> None:
    with contextlib.suppress(OSError):
        WAKE_PID_PATH.unlink()


def main() -> None:
    if "--check" in sys.argv[1:]:
        print("voice-harness-wake: ok")
        return
    singleton = _acquire_wake_singleton()
    daemon = WakeConversationDaemon(load_user_config())

    def handle_signal(_signum: int, _frame: object) -> None:
        daemon.stop()

    def handle_listen(_signum: int, _frame: object) -> None:
        daemon.force_listen.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGUSR1, handle_listen)
    try:
        _write_pidfile()
        identity = process_identity(os.getpid())
        if identity is None:
            raise HarnessError("could not publish wake service configuration snapshot")
        publish_service_snapshot(
            "voice-harness-wake.service",
            daemon.user_config,
            pid=os.getpid(),
            process_start=identity,
        )
        daemon.run()
    except Exception as exc:
        if daemon.running:
            log(f"fatal: {type(exc).__name__}: {exc}")
            notify(DAEMON_FAILURE, error=True)
            raise
    finally:
        daemon.stop()
        _remove_pidfile()
        fcntl.flock(singleton, fcntl.LOCK_UN)
        os.close(singleton)


if __name__ == "__main__":
    main()
