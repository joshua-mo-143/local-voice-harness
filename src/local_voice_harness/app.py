from __future__ import annotations

import json

from .components import llm_ready, start_components
from .config import CURSOR_PATTERN, PID_PATH, STT_SOCKET, TTS_SOCKET
from .cursor.jobs import (
    DeliveryClaims,
    acknowledge_deliveries,
    cursor_turn,
    release_deliveries,
)
from .errors import HarnessError
from .ipc import socket_ready
from .llm import qwen_response
from .tts.client import synthesize_and_play


def respond(text: str) -> None:
    text = text.strip()
    if not text:
        raise HarnessError("request text is empty")
    delivery_claims: DeliveryClaims = []
    try:
        start_components()
        print(f"You: {text}")
        response = (
            cursor_turn(text, delivery_claims=delivery_claims)[0]
            if CURSOR_PATTERN.match(text)
            else qwen_response(text, delivery_claims=delivery_claims)
        )
        print(f"Assistant: {response}")
        synthesize_and_play(response)
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
