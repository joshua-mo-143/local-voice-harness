from __future__ import annotations

import json

from .browser_context import request_context
from .components import component_usage, llm_ready, start_components
from .config import (
    CURSOR_PATTERN,
    JOBS_DIR,
    LEGACY_JOBS_DIR,
    PID_PATH,
    STT_SOCKET,
    TTS_SOCKET,
)
from .cursor.delivery import (
    DeliveryClaims,
)
from .cursor.delivery import (
    acknowledge_deliveries as acknowledge_claims,
)
from .cursor.delivery import (
    release_deliveries as release_claims,
)
from .cursor.service import CursorTurnRequest, cursor_turn
from .cursor.store import JobStore
from .errors import HarnessError
from .intent import ForkIntent, Intent, IntentRoute, decide_fork_intent, route_intent
from .ipc import socket_ready
from .llm import qwen_response
from .tts.client import stream_and_play
from .vocabulary import resolve_aliases

CURSOR_STORE = JobStore(JOBS_DIR, LEGACY_JOBS_DIR)


def acknowledge_deliveries(claims: DeliveryClaims) -> None:
    acknowledge_claims(CURSOR_STORE, claims)


def release_deliveries(claims: DeliveryClaims) -> None:
    release_claims(CURSOR_STORE, claims)


def respond(text: str) -> None:
    text = text.strip()
    if not text:
        raise HarnessError("request text is empty")
    text = resolve_aliases(text)
    delivery_claims: DeliveryClaims = []
    with component_usage():
        try:
            start_components()
            print(f"You: {text}")
            context = request_context(text)
            if CURSOR_PATTERN.search(text):
                route = IntentRoute(Intent.CURSOR_SUBMIT, "high")
            else:
                route = route_intent(text, context)
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
            if route.actionable and route.intent == Intent.CURSOR_SUBMIT:
                response = cursor_turn(
                    CursorTurnRequest(
                        context.text,
                        utterance=text,
                        context_repository=context.focused_repository,
                        **github_arguments,
                    ),
                    delivery_claims=delivery_claims,
                )[0]
            else:
                response = qwen_response(
                    context.text,
                    **github_arguments,
                    trusted_utterance=text,
                    delivery_claims=delivery_claims,
                )
            print(f"Assistant: {response}")
            stream_and_play(response)
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
