from __future__ import annotations

import json

from .browser_context import enrich_request
from .components import llm_ready, start_components
from .config import CURSOR_PATTERN, PID_PATH, STT_SOCKET, TTS_SOCKET
from .cursor.jobs import cursor_turn
from .errors import HarnessError
from .ipc import socket_ready
from .llm import qwen_response
from .tts.client import synthesize_and_play


def respond(text: str) -> None:
    text = text.strip()
    if not text:
        raise HarnessError("request text is empty")
    start_components()
    print(f"You: {text}")
    request = enrich_request(text)
    response = (
        cursor_turn(request)[0]
        if CURSOR_PATTERN.match(text)
        else qwen_response(request)
    )
    print(f"Assistant: {response}")
    synthesize_and_play(response)


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
