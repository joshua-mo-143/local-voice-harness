# Configuration

Only the variables marked “wake drop-in” may be added to
`voice-harness-wake.service` with `systemctl --user edit`. The installed-unit audit
rejects other extra variables and every `EnvironmentFile` on every shipped service.
Dictation backend selectors belong in `~/.config/dictation/backend.env`, which the
launcher parses through its separate allowlist.

## AI backends

LLM and TTS providers are selected independently in
`~/.config/voice-harness/backends.toml`. Both default to `local`; use `venice` for
either hosted backend:

```toml
[llm]
provider = "venice"
model = "venice-uncensored"

[tts]
provider = "venice"
model = "tts-kokoro"
voice = "af_sky"
speed = 1.0
```

The optional `endpoint` and positive `timeout` keys may be set in either section.
TTS `speed` accepts values from `0.25` to `4.0`. Environment overrides are available
as `VOICE_HARNESS_LLM_PROVIDER`, `VOICE_HARNESS_LLM_MODEL`,
`VOICE_HARNESS_LLM_ENDPOINT`, `VOICE_HARNESS_LLM_TIMEOUT`, and their corresponding
`VOICE_HARNESS_TTS_*` forms, including `VOICE`, `SPEED`, `ENDPOINT`, and `TIMEOUT`.

Store the shared Venice API key with `voice-harness credentials set`; file-based
credentials are intentionally unsupported. Restart the wake and TTS services after
changing providers, then run `voice-harness doctor` to verify the configuration and
credential.

## Runtime variables

| Variable | Purpose | Default | Configuration channel |
| --- | --- | --- | --- |
| `VOICE_HARNESS_SOURCE` | PipeWire microphone source | Development-machine source | Wake drop-in |
| `VOICE_HARNESS_VOICE` | Absolute Chatterbox reference WAV path | Built-in voice | Wake drop-in |
| `VOICE_HARNESS_WAKE_THRESHOLD` | OpenWakeWord activation threshold (`0`–`1`) | `0.55` | Wake drop-in |
| `VOICE_HARNESS_MIN_SPEECH_RMS` | Non-negative speech energy gate | `1100` | Wake drop-in |
| `VOICE_HARNESS_BARGE_IN_MODE` | Playback interruption (`wake`, `vad`, or `off`) | `wake` | Wake drop-in |
| `VOICE_HARNESS_BARGE_IN_SPEECH_FRAMES` | Positive count of consecutive 80 ms speech frames | `5` | Wake drop-in |
| `VOICE_HARNESS_PLAYBACK_QUIET_FRAMES` | Positive count of quiet 80 ms frames after playback | `4` | Wake drop-in |
| `VOICE_HARNESS_PLAYBACK_QUIET_TIMEOUT_SECONDS` | Non-negative post-playback echo-drain timeout | `2` | Wake drop-in |
| `VOICE_HARNESS_PLAYBACK_LATENCY` | Non-negative `pw-play` duration ending in `us`, `ms`, or `s` | `100ms` | Wake drop-in |
| `VOICE_HARNESS_CURSOR_FOREGROUND_SECONDS` | Non-negative time before a Cursor job backgrounds | `5` | Wake drop-in |
| `VOICE_HARNESS_HERDR_BIN` | Absolute Herdr executable path | `~/.local/bin/herdr` | Wake drop-in |
| `VOICE_HARNESS_PROJECT_ROOT` | Absolute allowed root for inferred repositories | Home directory | Wake drop-in |
| `VOICE_HARNESS_GITHUB_ROOT` | Absolute fork-clone root inside the project root | `~/src` | Wake drop-in |
| `VOICE_HARNESS_FOCUSED_APP_CONTEXT` | Enable focused editor/terminal context capture | `1` | Wake drop-in |
| `VOICE_HARNESS_FOCUSED_APP_DENY` | Comma-separated denied focused window classes | Password managers, RuneLite | Wake drop-in |
| `VOICE_HARNESS_FOCUSED_APP_MAX_CHARS` | Positive combined focused-app context character cap | `12000` | Wake drop-in |
| `DICTATION_INJECT` | Focused-window insertion mode (`auto`, `paste`, `type`, or `stdout`) | `auto` | Wake drop-in |
| `DICTATION_REPLACEMENTS` | Semicolon-separated STT corrections | Cursor/Herdr defaults | Wake drop-in |
| `DICTATION_VAD_END_SILENCE_MS` | Positive silence duration that finishes VAD dictation | `900` | Calling environment |
| `DICTATION_VAD_START_SPEECH_FRAMES` | Consecutive 80 ms speech frames required to start an utterance | `3` | Calling environment |
| `DICTATION_VAD_MAX_SECONDS` | Positive maximum duration of each VAD utterance | `120` | Calling environment |
| `DICTATION_VAD_MIN_SPEECH_RMS` | Non-negative VAD speech energy gate | `1100` | Calling environment |
| `DICTATION_BACKEND` | Dictation engine (`parakeet` or `whisper`) | `parakeet` | `backend.env` |
| `DICTATION_MODEL` | Backend model | `nemo-parakeet-tdt-0.6b-v2` | `backend.env` |
| `DICTATION_QUANTIZATION` | Parakeet ONNX quantization (`none` disables it) | `int8` | `backend.env` |
| `DICTATION_COMPUTE` | faster-whisper compute type | `float16` | `backend.env` |
| `DICTATION_LANGUAGE` | Spoken language to transcribe (`en`, `zh`, `english`, `chinese`, or `auto`) | `auto` | `backend.env` |

`VOICE_HARNESS_PROJECT_ROOT` can narrow repository discovery to another directory.
`VOICE_HARNESS_GITHUB_ROOT` must resolve inside it. Forks are cloned to
`<github-root>/<source-owner>/<repository>`; an existing checkout is reused only when
its `origin` identifies the expected fork. The source repository is configured as the
`upstream` remote. Herdr and repository path overrides grant the wake process access
to the selected executable and trees; review those trusted local paths before
restarting and auditing the service.

`voice-harness dictate vad` reads its `DICTATION_VAD_*` settings from the process
that enables it. The listener waits indefinitely for speech, transcribes after the
configured silence, and then rearms. Another invocation while it is active disables
the listener. Configure any desktop keybind or wrapper outside the harness.

Transcription and routing accuracy for repository names, issue references, and
recurring speech-recognition mistakes can be improved with a local, user-owned
vocabulary store edited through `voice-harness vocabulary` commands. See
[Vocabulary and entity aliases](vocabulary.md).
