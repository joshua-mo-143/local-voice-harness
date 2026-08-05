from __future__ import annotations

import json

from .browser_context import request_context
from .components import component_usage, llm_ready, start_components
from .config import PID_PATH, STT_SOCKET, TTS_SOCKET
from .cursor.jobs import (
    DeliveryClaims,
    acknowledge_deliveries,
    cursor_turn,
    release_deliveries,
)
from .errors import HarnessError
from .intent import Intent, route_intent
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
            route = route_intent(text, context)
            response = (
                cursor_turn(
                    context.text,
                    utterance=text,
                    context_repository=context.focused_repository,
                    delivery_claims=delivery_claims,
                )[0]
                if route.actionable and route.intent == Intent.CURSOR_SUBMIT
                else qwen_response(context.text, delivery_claims=delivery_claims)
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
