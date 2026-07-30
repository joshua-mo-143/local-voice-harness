from __future__ import annotations

import json
import os
import socketserver
import sys
import threading
import time
from pathlib import Path

RUNTIME = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
SOCKET_PATH = RUNTIME / "voice-harness-tts.sock"
OUTPUT_ROOT = RUNTIME / "voice-harness"
OUTPUT_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
LOCK = threading.Lock()


def log(message: str) -> None:
    print(f"[voice-harness-tts] {message}", file=sys.stderr, flush=True)


class RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            import soundfile as sf
            import torch

            line = self.rfile.readline(64 * 1024)
            if not line:
                self.wfile.write(b'{"ok":true,"ready":true}\n')
                return
            request = json.loads(line)
            text = str(request.get("text", "")).strip()
            if not text:
                raise ValueError("text must not be empty")
            if len(text) > 2_000:
                raise ValueError("text exceeds 2000 characters")
            output = Path(
                str(request.get("output", OUTPUT_ROOT / "reply.wav"))
            ).resolve()
            if OUTPUT_ROOT.resolve() not in output.parents:
                raise ValueError("output must be inside the runtime voice-harness directory")
            voice_value = str(request.get("voice", "")).strip()
            voice = Path(voice_value).expanduser().resolve() if voice_value else None
            if voice is not None and not voice.is_file():
                raise FileNotFoundError(f"reference voice not found: {voice}")
            started = time.perf_counter()
            with LOCK:
                waveform = MODEL.generate(
                    text,
                    audio_prompt_path=str(voice) if voice else None,
                    temperature=0.8,
                    top_p=0.95,
                )
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            audio = waveform.detach().cpu().float().squeeze().numpy()
            sf.write(str(output), audio, MODEL.sr)
            duration = len(audio) / MODEL.sr
            response = {
                "ok": True,
                "output": str(output),
                "sample_rate": MODEL.sr,
                "audio_seconds": round(duration, 3),
                "generation_seconds": round(elapsed, 3),
                "realtime_factor": round(elapsed / duration, 3),
            }
        except Exception as exc:
            log(f"request failed: {type(exc).__name__}: {exc}")
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        self.wfile.write((json.dumps(response) + "\n").encode())


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def main() -> None:
    global MODEL
    if "--check" in sys.argv[1:]:
        print("voice-harness-tts: ok")
        return
    import torch
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    log("loading Chatterbox Turbo on CUDA")
    MODEL = ChatterboxTurboTTS.from_pretrained(device="cuda")
    torch.cuda.synchronize()
    log("model ready")
    SOCKET_PATH.unlink(missing_ok=True)
    try:
        with Server(str(SOCKET_PATH), RequestHandler) as server:
            os.chmod(SOCKET_PATH, 0o600)
            log(f"listening on {SOCKET_PATH}")
            server.serve_forever()
    finally:
        SOCKET_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
