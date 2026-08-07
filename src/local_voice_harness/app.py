from __future__ import annotations

import json

from .browser_context import request_context
from .components import component_usage, llm_ready, start_components
from .config import CURSOR_PATTERN, PID_PATH, STT_SOCKET, TTS_SOCKET
from .cursor.jobs import (
    DeliveryClaims,
    acknowledge_deliveries,
    cursor_turn,
    release_deliveries,
)
from .errors import HarnessError
from .intent import ForkIntent, Intent, IntentRoute, decide_fork_intent, route_intent
from .ipc import socket_ready
from .llm import qwen_response
from .tts.client import stream_and_play


def respond(text: str) -> None:
    text = text.strip()
    if not text:
        raise HarnessError("request text is empty")
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
                    context.text,
                    utterance=text,
                    context_repository=context.focused_repository,
                    **github_arguments,
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
