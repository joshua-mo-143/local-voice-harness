# Opt-in hardware smoke checks

These checks are manual and intentionally excluded from CI. They start local services,
load GPU models, use the microphone and speakers, and may start a Herdr-managed Cursor
agent. Run only the section whose side effects you intend, on the target workstation.

Before starting, install the project and services as described in the README. Open a
second terminal for logs and keep `voice-harness services stop` available for cleanup.

## PipeWire capture and playback

Entry point:

```fish
wpctl status
pw-record --version
pw-play --version
voice-harness begin
# Speak a short sentence.
voice-harness end
```

Checklist:

- The configured source appears in `wpctl status`.
- `begin` starts capture without an immediate PipeWire error.
- `end` stops capture and reaches transcription.
- A spoken response is audible on the intended sink, without overlap or truncation.
- `journalctl --user -u voice-harness-wake.service -n 50` has no device errors.

If `end` or transcription fails, cancel any surviving recording before retrying.
Then stop services started by the turn:

```fish
voice-harness cancel
voice-harness services stop
```

## CUDA model loading

Entry point:

```fish
nvidia-smi
llama-server --list-devices
systemctl --user start dictation.service voice-harness-llm.service voice-harness-tts.service
systemctl --user is-active dictation.service voice-harness-llm.service voice-harness-tts.service
curl --fail http://127.0.0.1:8090/health
journalctl --user -u dictation.service -u voice-harness-llm.service -u voice-harness-tts.service -n 100
```

Checklist:

- `llama-server` lists the CUDA device configured in the LLM unit.
- All three services remain active after their models load.
- The llama.cpp health endpoint succeeds.
- Logs identify CUDA providers/devices and contain no CPU fallback, missing model, or
  out-of-memory error.
- `nvidia-smi` shows the expected model processes and plausible VRAM use.

Stop every service started by this isolated check:

```fish
systemctl --user stop voice-harness-tts.service voice-harness-llm.service dictation.service
```

## Herdr and Cursor handoff

This check starts or reuses a Cursor agent but should not ask it to modify files.

Entry point:

```fish
herdr status server
herdr agent list
voice-harness text "Use Cursor to inspect this repository and report its current branch without changing anything."
herdr agent list
```

Checklist:

- Herdr reports a running server.
- The request selects the intended checkout.
- Exactly one suitable agent handles the request.
- The result returns to the harness and no repository files change.
- `voice-harness status` shows no abandoned running job.

Stop any voice services activated while delivering the result. Herdr and its retained
agent are intentionally left running for inspection:

```fish
voice-harness services stop
```

## End-to-end voice turn

Start from stopped on-demand models so this check includes service activation:

```fish
voice-harness services start
voice-harness services status
```

Say “Hey Jarvis, what time is it?” and wait for playback.

Checklist:

- Wake detection activates once.
- Capture stops after the utterance and produces a non-empty transcription.
- Dictation, LLM, and TTS become ready without manual intervention.
- The response begins playing and the complete turn is audible.
- A second ordinary turn succeeds with warm models.
- Saying the wake phrase during playback cancels that playback and begins a new turn.
- Service logs contain no traceback, stale socket, orphan process, or repeated restart.

Clean up:

```fish
voice-harness services stop
voice-harness services status
```
