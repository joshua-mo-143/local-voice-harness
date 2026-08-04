from __future__ import annotations

import os
import re
import socket
import sys
import threading
from pathlib import Path

RUNTIME = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
SOCKET_PATH = RUNTIME / "dictation.sock"
MODEL_NAME = os.environ.get("DICTATION_MODEL", "large-v3")
COMPUTE_TYPE = os.environ.get("DICTATION_COMPUTE", "float16")
PROMPT = os.environ.get(
    "DICTATION_PROMPT",
    "Technical software engineering dictation that may mention Cursor, Herdr, "
    "code, files, functions, terminals, and command-line tools.",
)
REPLACEMENTS = {
    source.strip(): target.strip()
    for entry in os.environ.get(
        "DICTATION_REPLACEMENTS", "herder:herdr;cursa:Cursor;curser:Cursor"
    ).split(";")
    if entry.strip() and ":" in entry
    for source, target in [entry.split(":", 1)]
    if source.strip()
}
LOCK = threading.Lock()


def log(message: str) -> None:
    print(f"[dictation] {message}", file=sys.stderr, flush=True)


def normalize(text: str) -> str:
    for source, target in REPLACEMENTS.items():
        text = re.sub(
            rf"(?<!\w){re.escape(source)}(?!\w)",
            lambda _match, replacement=target: replacement,
            text,
            flags=re.IGNORECASE,
        )
    return text


def main() -> None:
    from faster_whisper import WhisperModel

    log(f"loading faster-whisper {MODEL_NAME} on CUDA ({COMPUTE_TYPE})")
    model = WhisperModel(MODEL_NAME, device="cuda", compute_type=COMPUTE_TYPE)
    log("model ready")
    SOCKET_PATH.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(SOCKET_PATH))
        os.chmod(SOCKET_PATH, 0o600)
        server.listen(4)
        log(f"listening on {SOCKET_PATH}")
        while True:
            connection, _ = server.accept()
            try:
                request = bytearray()
                while not request.endswith(b"\n"):
                    chunk = connection.recv(4096)
                    if not chunk:
                        break
                    request.extend(chunk)
                audio_path = request.decode(errors="replace").strip()
                if not audio_path:
                    continue
                try:
                    with LOCK:
                        segments, _info = model.transcribe(
                            audio_path,
                            language="en",
                            beam_size=5,
                            vad_filter=True,
                            condition_on_previous_text=False,
                            initial_prompt=PROMPT,
                        )
                        text = normalize(
                            "".join(segment.text for segment in segments).strip()
                        )
                except Exception as exc:
                    log(f"transcription failed: {exc}")
                    text = f"__DICTATION_ERROR__:{type(exc).__name__}: {exc}"
                connection.sendall(text.encode())
            finally:
                connection.close()
    finally:
        server.close()
        SOCKET_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
