# Configuration

`config.toml` owns user choices for shipped services. The units set only operational
runtime values such as private sockets and cache paths; they do not set provider,
model, device, audio, integration, or platform choices. The installed-unit audit
rejects user-choice `Environment=` drop-ins and every `EnvironmentFile`.
`backends.toml` and `~/.config/dictation/backend.env` remain supported only as
legacy inputs to the unified resolver.

## Unified configuration (`config.toml`)

`~/.config/voice-harness/config.toml` provides one validated, typed model for
user-facing defaults across six sections: `providers`, `integrations`,
`compute`, `audio`, `dictation`, and `platform`. The file is optional. When it is absent, or a
key is omitted, built-in defaults apply, so existing installations that rely only
on `backends.toml` and environment variables keep working unchanged.

Values are resolved with a fixed precedence, from lowest to highest:

1. built-in defaults
2. `config.toml`
3. legacy `backends.toml` provider values and `backend.env` dictation selectors
4. environment variables

In other words, an environment override always wins, and a legacy `backends.toml`
provider setting still takes precedence over the same value in `config.toml`.
Most fields reuse the `VOICE_HARNESS_*` and `DICTATION_*` names documented in the
tables below (for example `VOICE_HARNESS_WAKE_THRESHOLD` overrides
`[audio] wake_threshold`). The sections that previously had no environment knob
add these overrides: `VOICE_HARNESS_INTEGRATION_GITHUB`,
`VOICE_HARNESS_INTEGRATION_ZENDESK`, and `VOICE_HARNESS_INTEGRATION_LINEAR` for
`[integrations]`, and `VOICE_HARNESS_CUDA_DEVICE` for `[compute] cuda_device`.

```toml
[providers.llm]
provider = "venice"

[providers.tts]
provider = "venice"

[integrations]
github = true
zendesk = false
linear = false

[compute]
cuda_device = "CUDA0"
dictation_backend = "parakeet"
dictation_language = "auto"

[audio]
wake_threshold = 0.55
barge_in_mode = "wake"
playback_latency = "100ms"

[dictation]
inject = "auto"
prompt = "Technical software engineering dictation."
replacements = "herder:herdr;cursa:Cursor"
vad_end_silence_ms = 900

[platform]
project_root = "/home/example"
github_root = "/home/example/src"
herdr_worktree_root = "/home/example/.herdr/worktrees"
gh_bin = "gh"
git_bin = "git"
herdr_bin = "/home/example/.local/bin/herdr"
github_timeout_seconds = 30
herdr_timeout_seconds = 30
focused_app_context = true
cursor_followup = true
cursor_agent_inactivity_seconds = 900
cursor_agent_max_runtime_seconds = 3600
agent_job_start_concurrency = 3
```

GitHub is enabled by default for compatibility. Setting `github = false` prevents
the GitHub provider from parsing focused URLs or spoken issue references, calling
`gh` for context, or emitting repository, issue, and pull-request metadata.
Fresh installations default the optional `zendesk` and `linear` integrations to
disabled. While disabled, their URLs are not recognized and they contribute no
context, routing, agent instructions, or diagnostics. Existing installations can
preserve prior behavior by enabling the corresponding key or
`VOICE_HARNESS_INTEGRATION_*` override.

The initial Linear connector requires the `cursor-mcp` harness capability and an
enabled Linear MCP server. Enable it with `linear = true`, then run
`agent mcp login linear && agent mcp enable linear`. `voice-harness doctor`
reports a fatal, actionable configuration result when Linear is enabled without
that capability, and a Linear job is rejected before it is persisted or dispatched.
Invalid values (an unknown provider, an out-of-range TTS speed or wake threshold,
an unknown section or key, a malformed playback latency, and so on) raise an
actionable error rather than being silently ignored. Writes are atomic: the file
is written to a temporary sibling and renamed into place with owner-only (`0600`)
permissions.

Credentials are never read from or written to `config.toml`; a Venice API key in
any section is rejected. Store it with `voice-harness credentials set` instead, as
described under [AI backends](#ai-backends).

Malformed, invalid-encoding, and unreadable `config.toml`, `backends.toml`, or
legacy `backend.env` files fail with an error that identifies the source path (and
line for environment assignments). Runtime capability discovery does not replace
such failures with defaults, because that could silently enable or disable an
integration. Failures from an individual provider while capturing optional context
remain isolated from configuration loading failures.

## Runtime lifecycle

Runtime configuration uses a process-start snapshot. Each migrated composition
root resolves `UserConfig` once, then injects its immutable typed sections into
the process's consumers. Editing `config.toml`, a documented legacy input, or an
environment override does not mutate an already running process. LLM routing and
conversation calls, STT startup, TTS serving, playback, dictation injection, audio
capture, VAD, the integration registry, and configured GitHub and Herdr client
factories all use the injected snapshot. Detached workers resolve one snapshot at
worker startup; durable provider identity across later restarts is handled
separately. Restart every
affected foreground process, service, or detached worker to apply the change;
the configuration commands report affected active services.

`voice-harness doctor` likewise resolves one snapshot before running any check and
injects its configured GitHub and Herdr clients. An invalid configuration is reported
once as the direct fatal cause; checks that do not depend on configuration may still
run. Each service-management command also resolves one snapshot, including
`services restart`, which reuses it across the stop/start sequence.

There is no general configuration hot reload. The vocabulary store is the
documented exception: the dictation service reads it for each transcription, so
vocabulary edits apply without a restart. Legacy backend files are read only by
the resolver; provider and speech runtime consumers do not parse them
independently.

The lifecycle policy for every `config.toml` setting is:

| Settings | Snapshot owner | Apply changes by |
| --- | --- | --- |
| `providers.llm.provider`, `model`, `endpoint`, `timeout` | Wake process and local LLM service | Restart `voice-harness-wake.service` and `voice-harness-llm.service`; the local LLM service is stopped when the selected provider is remote |
| `providers.tts.provider`, `model`, `voice`, `speed`, `endpoint`, `timeout` | Wake process and TTS service | Restart `voice-harness-wake.service` and `voice-harness-tts.service` |
| `integrations.github`, `zendesk`, `linear` | Wake process integration registry | Restart `voice-harness-wake.service` |
| `compute.cuda_device` | Local LLM service | Restart `voice-harness-llm.service` |
| `compute.dictation_backend`, `dictation_model`, `dictation_quantization`, `dictation_compute`, `dictation_language` | Dictation service | Restart `dictation.service` |
| Every `audio.*` setting | Wake process | Restart `voice-harness-wake.service` |
| `dictation.prompt`, `dictation.replacements` | Dictation service | Restart `dictation.service` |
| `dictation.source`, `dictation.inject`, `dictation.vad_end_silence_ms`, `dictation.vad_max_seconds`, `dictation.vad_min_speech_rms`, `dictation.vad_start_speech_frames` | Each foreground dictation command | Start the next command; an already-running VAD command keeps its startup snapshot |
| Every `platform.*` setting | Wake process and each detached worker | Restart `voice-harness-wake.service`; already-admitted workers keep their snapshot, while newly started or restarted workers resolve the new value |

The same lifecycle applies regardless of whether the winning value came from
`config.toml`, a supported legacy file, or an environment override. Changing an
environment variable outside a running process has no effect until that process
is restarted. `voice-harness config set` reports the active systemd services from
this matrix; command-scoped settings intentionally produce no service restart
notice.

## Configuration commands

Use the CLI to inspect or persist unified settings in `config.toml`. Values are
validated before every write and persisted atomically with owner-only permissions.
These commands never create or update legacy `backends.toml` or `backend.env`
files. Existing legacy values continue to participate in resolution until those
files are removed.

```console
voice-harness setup
voice-harness setup --defaults
voice-harness setup --profile showcase
voice-harness config show
voice-harness config show audio.wake_threshold
voice-harness config set providers.llm.provider venice
voice-harness config set integrations.linear true
voice-harness config reset --section audio
voice-harness integrations list
voice-harness integrations enable linear
voice-harness integrations disable zendesk
voice-harness integrations doctor
```

`setup` walks through provider, integration, audio, and compute choices that are
supported on the current machine. Use `--defaults` for a non-interactive first
write with Venice LLM/TTS and local Parakeet dictation. The equivalent
`--profile showcase` also migrates an existing `backends.toml` into the unified
configuration and retains the original as `backends.toml.migrated` so legacy
provider values cannot override the selected profile. `config set` accepts dotted keys such as
`audio.wake_threshold`, `compute.dictation_backend`, and
`platform.cursor_followup`. After a change, the CLI reports which installed
services are currently running and need a restart, for example
`voice-harness-wake.service` or `dictation.service`.

`integrations doctor` inspects only enabled integrations. Linear uses the MCP
capability check; GitHub reports `gh` authentication when GitHub is enabled;
Zendesk confirms that browser capture is active.

Common examples:

```console
# Hosted LLM/TTS with local dictation
voice-harness config set providers.llm.provider venice
voice-harness config set providers.tts.provider venice
voice-harness credentials set

# Tighter wake sensitivity and faster playback drain
voice-harness config set audio.wake_threshold 0.65
voice-harness config set audio.playback_quiet_timeout_seconds 1.5

# Enable optional ticket context providers
voice-harness integrations enable linear
voice-harness integrations doctor
```

## AI backends

LLM and TTS providers are selected independently in `[providers.llm]` and
`[providers.tts]` in `config.toml`. Both default to `venice`; select `local` for
offline inference. Existing values in
`~/.config/voice-harness/backends.toml` continue to override `config.toml` until
that legacy file is removed:

```toml
[llm]
provider = "venice"
model = "venice-uncensored"

[tts]
provider = "venice"
model = "tts-kokoro"
voice = "af_sky"
speed = 1.25
```

The optional `endpoint` and positive `timeout` keys may be set in either section.
Venice TTS defaults to pitch-preserving `1.25` speed, applied locally through
FFmpeg so the result is consistent across models. Local TTS defaults to `1.0`.
TTS `speed` accepts values from `0.25` to `4.0`. Environment overrides are
available as `VOICE_HARNESS_LLM_PROVIDER`, `VOICE_HARNESS_LLM_MODEL`,
`VOICE_HARNESS_LLM_ENDPOINT`, `VOICE_HARNESS_LLM_TIMEOUT`, and their corresponding
`VOICE_HARNESS_TTS_*` forms, including `VOICE`, `SPEED`, `ENDPOINT`, and `TIMEOUT`.

Store the shared Venice API key with `voice-harness credentials set`; file-based
credentials are intentionally unsupported. Restart the wake and TTS services after
changing providers, then run `voice-harness doctor` to verify the configuration and
credential.

## Cursor plan approval

Reviewed medium- and high-risk Cursor plans default to `ask` mode. At the Plan Mode
Build boundary the harness speaks a yes-or-no question. Natural affirmative answers
such as "yes", "sure", "go ahead", "lgtm", and "ok then" approve the current
fenced plan; "no", "nah", "don't", and "cancel" cancel the job while retaining its
plan artifact. Ambiguous replies leave the same question pending.

Only explicit approvals accepted by the Herdr prompt API count toward the learning
threshold. After three distinct accepted approvals, the third implementation
finishes before the harness offers to automatically approve future ordinary reviewed
plans. Accepting writes `auto`; declining leaves `ask`. Automatic mode never answers
reviewer objections, unresolved product or architecture questions, security or
destructive confirmations, interactive questionnaires, or tool permission prompts.
High-risk workflows remain in `ask` regardless of the learned mode, and deterministic
hard-risk evidence also keeps the Plan Mode gate in `ask`.

The learned mode, capped approval ledger, and one-time offer identity live in
`~/.config/voice-harness/plan-approval.json`. Updates use a sibling file lock,
temporary-file replacement, directory sync, and owner-only permissions. The file is
separate from `config.toml` because the harness updates it from accepted user
decisions. A temporary read or write failure preserves the completed Cursor result
and retries preference reconciliation locally.

Inspect the mode and threshold count, or return to explicit approval, with:

```console
voice-harness plan-approval status
voice-harness plan-approval ask
```

## Runtime variables

| Variable | Purpose | Default | Configuration channel |
| --- | --- | --- | --- |
| `VOICE_HARNESS_SOURCE` | PipeWire microphone source | Development-machine source | Process environment override |
| `VOICE_HARNESS_VOICE` | Absolute Chatterbox reference WAV path | Built-in voice | Process environment override |
| `VOICE_HARNESS_WAKE_THRESHOLD` | OpenWakeWord activation threshold (`0`–`1`) | `0.55` | Process environment override |
| `VOICE_HARNESS_MIN_SPEECH_RMS` | Non-negative speech energy gate | `1100` | Process environment override |
| `VOICE_HARNESS_BARGE_IN_MODE` | Playback interruption (`wake`, `vad`, or `off`) | `wake` | Process environment override |
| `VOICE_HARNESS_BARGE_IN_SPEECH_FRAMES` | Positive count of consecutive 80 ms speech frames | `5` | Process environment override |
| `VOICE_HARNESS_PLAYBACK_QUIET_FRAMES` | Positive count of quiet 80 ms frames after playback | `4` | Process environment override |
| `VOICE_HARNESS_PLAYBACK_QUIET_TIMEOUT_SECONDS` | Non-negative post-playback echo-drain timeout | `2` | Process environment override |
| `VOICE_HARNESS_PLAYBACK_LATENCY` | Non-negative `pw-play` duration ending in `us`, `ms`, or `s` | `100ms` | Process environment override |
| `VOICE_HARNESS_CURSOR_FOREGROUND_SECONDS` | Non-negative time before a Cursor job backgrounds | `5` | Process environment override |
| `VOICE_HARNESS_CURSOR_AGENT_INACTIVITY_SECONDS` | Positive time without observable Cursor progress before cancellation | `900` | Process environment override |
| `VOICE_HARNESS_CURSOR_AGENT_MAX_RUNTIME_SECONDS` | Positive absolute runtime limit for one Cursor turn | `3600` | Process environment override |
| `VOICE_HARNESS_AGENT_JOB_START_CONCURRENCY` | Positive maximum number of concurrent durable job starts in a multi-ticket request | `3` | Process environment override |
| `VOICE_HARNESS_CURSOR_FOLLOWUP` | Enable completed-job follow-up context (kill switch) | `1` | Process environment override |
| `VOICE_HARNESS_CURSOR_FOLLOWUP_WINDOW_SECONDS` | Finite, non-negative absolute lifetime of the retained completed-job reference | `60` | Process environment override |
| `VOICE_HARNESS_HERDR_BIN` | Absolute Herdr executable path | `~/.local/bin/herdr` | Process environment override |
| `VOICE_HARNESS_HERDR_WORKTREE_ROOT` | Root for Herdr-created worktrees | `~/.herdr/worktrees` | Process environment override |
| `VOICE_HARNESS_GH_BIN` | GitHub CLI executable | `gh` | Process environment override |
| `VOICE_HARNESS_GIT_BIN` | Git executable used for GitHub checkouts | `git` | Process environment override |
| `VOICE_HARNESS_GITHUB_TIMEOUT_SECONDS` | Positive default for GitHub commands without an operation-specific timeout | `30` | Process environment override |
| `VOICE_HARNESS_HERDR_TIMEOUT_SECONDS` | Positive Herdr command and startup timeout | `30` | Process environment override |
| `VOICE_HARNESS_PROJECT_ROOT` | Absolute allowed root for inferred repositories | Home directory | Process environment override |
| `VOICE_HARNESS_GITHUB_ROOT` | Absolute fork-clone root inside the project root | `~/src` | Process environment override |
| `VOICE_HARNESS_FOCUSED_APP_CONTEXT` | Enable focused editor/terminal context capture | `1` | Process environment override |
| `VOICE_HARNESS_FOCUSED_APP_DENY` | Comma-separated denied focused window classes | Password managers, RuneLite | Process environment override |
| `VOICE_HARNESS_FOCUSED_APP_MAX_CHARS` | Positive combined focused-app context character cap | `12000` | Process environment override |
| `DICTATION_SOURCE` | Dictation PipeWire source, independently overriding `audio.source` | `audio.source` | Environment override |
| `DICTATION_INJECT` | Focused-window insertion mode (`auto`, `paste`, `type`, or `stdout`) | `auto` | Process environment override |
| `DICTATION_PROMPT` | Initial Whisper transcription prompt | Technical dictation prompt | Environment override |
| `DICTATION_REPLACEMENTS` | Semicolon-separated STT corrections | Cursor/Herdr defaults | Process environment override |
| `DICTATION_VAD_END_SILENCE_MS` | Positive silence duration that finishes VAD dictation | `900` | Environment override |
| `DICTATION_VAD_START_SPEECH_FRAMES` | Consecutive 80 ms speech frames required to start an utterance | `3` | Environment override |
| `DICTATION_VAD_MAX_SECONDS` | Positive maximum duration of each VAD utterance | `120` | Environment override |
| `DICTATION_VAD_MIN_SPEECH_RMS` | Non-negative VAD speech energy gate | `1100` | Environment override |
| `DICTATION_BACKEND` | Dictation engine (`parakeet` or `whisper`) | `parakeet` | Environment override; legacy `backend.env` input |
| `DICTATION_MODEL` | Backend model | `nemo-parakeet-tdt-0.6b-v2` | Environment override; legacy `backend.env` input |
| `DICTATION_QUANTIZATION` | Parakeet ONNX quantization (`none` disables it) | `int8` | Environment override; legacy `backend.env` input |
| `DICTATION_COMPUTE` | faster-whisper compute type | `float16` | Environment override; legacy `backend.env` input |
| `DICTATION_LANGUAGE` | Spoken language to transcribe (`en`, `zh`, `english`, `chinese`, or `auto`) | `auto` | Environment override; legacy `backend.env` input |

`VOICE_HARNESS_PROJECT_ROOT` can narrow repository discovery to another directory.
`VOICE_HARNESS_GITHUB_ROOT` must resolve inside it. Forks are cloned to
`<github-root>/<source-owner>/<repository>`; an existing checkout is reused only when
its `origin` identifies the expected fork. The source repository is configured as the
`upstream` remote. Herdr and repository path overrides grant the wake process access
to the selected executable and trees; review those trusted local paths before
restarting and auditing the service.

After the harness announces a completed Cursor job, it retains a one-shot reference
to that job for `VOICE_HARNESS_CURSOR_FOLLOWUP_WINDOW_SECONDS`. Within that window a
referential request such as "review the changes" or "run the tests" starts a child
job that reuses the completed job's exact retained checkout instead of creating a
fresh workspace. The reference is volatile: it is installed only after the
completion response plays and its delivery is acknowledged, it is consumed only
after a child job is durably created (so a busy checkout remains retryable until
expiry), it is cleared by explicit new work or by ending the conversation, and it
never survives a restart. Awaiting-clarification replies always take precedence,
and opening a pull request remains unsupported. Set
`VOICE_HARNESS_CURSOR_FOLLOWUP=0` to disable the feature entirely without affecting
clarification replies or fresh submissions.

An explicit multi-ticket request starts one ordinary durable job per unique,
validated target. Bare GitHub numbers require a focused repository issue-list page;
bare Linear numbers require a focused Linear team page. Full GitHub references and
full Linear keys need no browser scope. All unique targets are preflighted before
any starts, then valid jobs start with at most
`VOICE_HARNESS_AGENT_JOB_START_CONCURRENCY` concurrent admissions. The response
reports each target as accepted, rejected, or start-failed in request order.
Fan-out is best-effort and intentionally not crash-atomic: one child can fail after
another has started, and there is no durable batch record to roll children back.

Intent routing uses the configured LLM provider and endpoint, including Venice when
selected. The router is authoritative for workspace mutations: conversation fallback
does not receive Cursor tools, and invalid, unavailable, or low-confidence routing
cannot start a job. A follow-up reuses a settled exact-checkout agent or the completed
parent's retained Herdr workspace and root pane; if neither identity is safely
available, the child fails closed without opening another pane.

`voice-harness dictate vad` reads its `DICTATION_VAD_*` settings from the process
that enables it. The listener waits indefinitely for speech, transcribes after the
configured silence, and then rearms. Another invocation while it is active disables
the listener. Configure any desktop keybind or wrapper outside the harness.

Transcription and routing accuracy for repository names, issue references, and
recurring speech-recognition mistakes can be improved with a local, user-owned
vocabulary store edited through `voice-harness vocabulary` commands. See
[Vocabulary and entity aliases](vocabulary.md).
