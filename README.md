# Local Voice Agent Harness

A voice agent harness that runs locally for wake-word detection, speech
recognition, conversational routing, and speech synthesis. Software-engineering
requests are delegated to interactive Cursor agents managed by
[Herdr](https://herdr.dev).

This is a personal, hardware-specific project rather than a polished cross-platform
installer. The documented setup reproduces the configuration on which it was tested.

## Note from maintainer
Hi! (this note is entirely human-written)

Pretty much all of this code is AI generated. While I am dogfooding this harness (so any bugs are likely to be found and squashed by me), using this software will primarily be at your own risk.

To add: I also am not looking at the code - as long as it works, I will likely keep using it (and fixing bugs as I find them). Be prepared to see LLM-made horrors beyond comprehension should you explore the codebase.


## Architecture

```text
PipeWire microphone
  -> OpenWakeWord ("Hey Jarvis")
  -> Parakeet TDT 0.6B v2 via ONNX Runtime (CUDA)
  -> Qwen3.5-4B Q4_K_M via llama.cpp (CUDA)
       -> focused intent classification
       -> ordinary conversational response
       -> Herdr-managed Cursor agent, GitHub CLI, and Linear MCP
  -> Chatterbox Turbo (CUDA)
  -> PipeWire playback
```

The always-on wake daemon verifies OpenWakeWord candidates with the configured
dictation backend to reject false activations. A request that takes longer than five
seconds becomes a persisted background job, and its completion or clarification
question is spoken later. Job transitions are serialized across the daemon and
detached workers; abandoned jobs are recovered at daemon startup and during normal
polling. Spoken background results use at-least-once delivery: playback is
acknowledged only after it succeeds, so a crash at that boundary may repeat a result
but will not silently lose it.

Spoken responses use chunk-level streaming. Chatterbox still generates a complete
waveform for each short sentence or clause, but the next chunk is synthesized while
the current chunk is sent through one low-latency PipeWire playback stream. This is
not native sample streaming from the model. Playback sessions are serialized across
processes so manual commands and daemon announcements cannot overlap.

Cursor routing works as follows:

1. Ask a focused Qwen pass to classify conversation, new work, clarification replies,
   status, and cancellation without rewriting the user's request.
2. Prefer an idle Cursor agent already running in the requested checkout.
3. For a Linear issue without a repository name, ask a dedicated routing agent to
   inspect the ticket through Linear MCP and infer the repository.
4. For a focused or explicitly spoken GitHub issue, validate it through `gh`, reuse
   an exact matching local checkout or clone its repository below the GitHub root,
   and preserve bounded issue context with the job.
5. If no repository can be resolved, open Rofi to select a local repository or paste
   a Git URL; cloning requires a second confirmation.
6. When the user unambiguously asks to fork, ask for a yes-or-no confirmation, then
   validate the focused public GitHub repository, create or reuse the authenticated
   user's fork, and clone it below the configured GitHub root.
7. When a GitHub pull request is focused, clone or reuse its repository below the
   configured GitHub root, create a job-unique `voice/github-pr-<job-id>` worktree,
   and run `gh pr checkout` only inside that reserved worktree.
8. Create or reuse a `voice/<issue-key>` worktree for Linear work, a stable
   `voice/github-issue-<number>` worktree for GitHub issue work, or a unique
   `voice/github-<job-id>` worktree for a GitHub fork task.
9. Start a new Cursor agent through Herdr when no suitable agent exists.
10. Reserve that agent and checkout until it finishes, is blocked, or is cancelled.

The harness never automatically commits, pushes, opens pull requests, modifies Linear,
or deletes generated worktrees. Fork creation is the only supported GitHub write and
is performed only after an unambiguous spoken request and a separate affirmative
confirmation. Checking out a focused pull request only reads from GitHub and writes to
its isolated local worktree. PR worktrees are reused only by recovery or continuation
of the same job. Completed and cancelled worktrees are retained for inspection, while
an invalid or partially prepared checkout is marked quarantined and is never dispatched.

### Runtime privacy and durability

Microphone recordings, recorder ownership files, logs, and service sockets are
transient session data under `$XDG_RUNTIME_DIR`. The bundled STT service accepts only
strictly named UUID generations beneath the two harness recording directories.
Stopping capture atomically moves the writable WAV to its immutable generation while
the recorder lock is still held; wake-mode recording performs the same handoff. A
later capture only replaces the writable path. After acquiring the model slot, STT
atomically moves that generation to a unique private processing path and removes only
the claimed file after the attempt. Cancellation removes writable audio after
recorder termination is confirmed. Recorder ownership includes the Linux process
start identity as well as its PID; it is not durable across login sessions.

Only one in-process GPU transcription runs at a time. A second fully framed request
receives a structured `server_busy` error immediately instead of waiting behind a
possibly hung model call, without moving or deleting its retryable generation. The
client retries that same immutable generation with bounded backoff for an overall
120-second request window. If STT remains busy, the error prints a safe
`voice-harness transcribe --generation <path>` retry command and leaves the file in
place. The accepted call remains synchronous; Python cannot safely force-cancel a
hung native GPU call, so service supervision must restart the dictation process to
recover that case. Wake capture is suppressed without stopping the daemon while a
manual or focused-dictation recorder owns the shared recording lock. Manual and
focused-dictation starts inspect every configured recorder owner atomically under
that lock, so different capture modes cannot run concurrently.

Cursor job JSON, its lock, and quarantine evidence are durable under the absolute
`$STATE_DIRECTORY/jobs` supplied by systemd. Outside the service they use
`$XDG_STATE_HOME/voice-harness/jobs`, falling back to
`~/.local/state/voice-harness/jobs`. `STATE_DIRECTORY` is service-owned and must
not be set in user environment overrides. Detached worker logs remain private,
session-only files under `$XDG_RUNTIME_DIR/voice-harness/jobs`. On first recovery,
legacy runtime job JSON is imported under both legacy and durable locks; conflicting
same-revision imports are preserved in the durable quarantine instead of replacing
state. Linux boot identity is part of worker and target-release ownership, so a
reused PID after reboot cannot inherit a stale claim. Recovery retains active,
undelivered, uncertain, fenced, manual-review, and quarantined records. It prunes
only delivered terminal jobs whose completion is more than seven days old and never
automatically deletes quarantine evidence.
Unresolved quarantine evidence conservatively fences conflicting target and
worktree reservations. Operators may explicitly release that fence through the
typed `JobStore.acknowledge_quarantine_reservations()` API, which writes a
hash-bound resolution tombstone while preserving the quarantined payload and
metadata.

## Compute requirements

Tested configuration:

- Linux x86-64 with systemd user services and PipeWire.
- NVIDIA GeForce RTX 5070 Ti Laptop GPU with 12 GB VRAM.
- 32 GB system RAM and swap.
- CUDA-capable NVIDIA driver plus a CUDA-enabled llama.cpp build.
- Python 3.11 for the management, wake, TTS, and bundled dictation environments.

Practical requirements for the included model choices:

| Resource | Minimum | Recommended |
| --- | ---: | ---: |
| GPU VRAM | 10 GB | 12 GB |
| System RAM | 16 GB | 32 GB |
| Free disk | 20 GB | 25+ GB |
| CPU | Modern 4-core x86-64 | 8+ cores |

With all models warm, expect roughly 8-10 GB of GPU memory. The main disk consumers
are:

- Qwen3.5-4B Q4_K_M GGUF: approximately 2.5 GB.
- Chatterbox Turbo cache: 3.8 GB.
- Parakeet TDT 0.6B v2 ONNX cache: less than 1 GB.
- Current Python environments: approximately 13 GB combined.

The tested configuration uses CUDA for Parakeet and Chatterbox. Parakeet can fall
back to ONNX Runtime's CPU provider, but CPU dictation has substantially higher
latency. The optional faster-whisper backend and Chatterbox are configured for CUDA.

## External prerequisites

Install these before setting up Python environments:

- PipeWire tools (`pw-record` and `pw-play`).
- `libnotify`/`notify-send`.
- Git, curl, the GitHub CLI (`gh`), and systemd user services.
- `xdotool` and `xclip` for X11 focused-window automation, or `wtype` and
  `wl-clipboard` for Hyprland/Sway focused-window automation.
- Rofi for repository selection and pasteable clone-URL prompts.
- [uv](https://docs.astral.sh/uv/) for reproducible Python versions/environments.
- A recent [llama.cpp](https://github.com/ggml-org/llama.cpp) build with CUDA and
  `llama-server`. The server runs with `--jinja` so it uses the model's native chat
  template, which llama.cpp requires for Qwen3.5 tool calling.
- The [Cursor CLI](https://cursor.com/docs/cli/installation).
- [Herdr](https://herdr.dev).
- A working NVIDIA driver.

On Arch/CachyOS, the base packages are approximately:

```bash
paru -S --needed pipewire libnotify git curl github-cli xdotool xclip \
  wl-clipboard wtype uv libsndfile
```

On Arch/CachyOS, install the CUDA-enabled llama.cpp AUR package (it conflicts with
`llama.cpp-vulkan` and other non-CUDA variants):

```bash
paru -S --needed cuda llama.cpp-cuda
```

Verify the required commands:

```bash
pw-record --version
pw-play --version
llama-server --version
llama-server --list-devices   # expect CUDA0: NVIDIA GeForce ...
nvidia-smi
```

Authenticate the GitHub CLI to let focused issue pages include repository details and
to create forks explicitly requested through the harness:

```bash
gh auth login
gh auth status
```

## Installation

The supplied systemd units assume the repository is cloned to
`$HOME/local-voice-harness`.

### 1. Clone and install the management/wake package

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL> "$HOME/local-voice-harness"
cd "$HOME/local-voice-harness"
uv sync --python 3.11 --extra wake --no-dev
mkdir -p "$HOME/.local/bin"
ln -sfn "$HOME/local-voice-harness/.venv/bin/voice-harness" \
  "$HOME/.local/bin/voice-harness"
```

This installs the package, OpenWakeWord dependencies, and all console entry points
in `.venv`. Ensure `$HOME/.local/bin` is on `PATH`.
The `wake`, `dictation`, and `tts` extras are intentionally installed into separate
environments because their CUDA and NumPy constraints are not all compatible.

OpenWakeWord includes the `hey_jarvis_v0.1.onnx` model used by the daemon.

### 2. Create the bundled dictation environment

```fish
env UV_PROJECT_ENVIRONMENT=.venv-dictation \
  uv sync --python 3.11 --extra dictation --no-dev
```

The default backend is Parakeet TDT 0.6B v2. Its first start downloads
`nemo-parakeet-tdt-0.6b-v2` from Hugging Face.

To use the supported faster-whisper backend instead, install its separate extra and
select it in the service environment file:

```fish
env UV_PROJECT_ENVIRONMENT=.venv-dictation \
  uv sync --python 3.11 --extra dictation-whisper --no-dev
mkdir -p "$HOME/.config/dictation"
printf '%s\n' \
  'DICTATION_BACKEND=whisper' \
  'DICTATION_MODEL=large-v3-turbo' \
  'DICTATION_COMPUTE=float16' \
  >"$HOME/.config/dictation/backend.env"
```

Install either `dictation` for Parakeet or `dictation-whisper` for faster-whisper;
the two extras are alternative backend environments, not a requirement to install
both.

The launcher reads `backend.env` itself and accepts only backend, model, language,
compute, and quantization selectors. Socket, CUDA/Hugging Face cache, temporary,
home, and XDG path variables are service-owned and cannot be overridden by that
file.

### 3. Create the Chatterbox environment

The existing service uses `$HOME/chatterbox-audition/.venv`:

```bash
mkdir -p "$HOME/chatterbox-audition"
UV_PROJECT_ENVIRONMENT="$HOME/chatterbox-audition/.venv" \
  uv sync --python 3.11 --extra tts --no-dev
```

The tested TTS environment used `chatterbox-tts 0.1.7` and CUDA-enabled
`torch 2.10.0`. The project-level uv overrides preserve those `torch` and
`torchaudio` versions despite Chatterbox's older metadata. If those wheels do not
support a future GPU/driver combination, update `[tool.uv].override-dependencies`
using the [PyTorch selector](https://pytorch.org/get-started/locally/).

Download Chatterbox before enabling its offline systemd service:

```bash
HF_HUB_OFFLINE=0 "$HOME/chatterbox-audition/.venv/bin/python" - <<'PY'
from chatterbox.tts_turbo import ChatterboxTurboTTS
ChatterboxTurboTTS.from_pretrained(device="cuda")
print("Chatterbox Turbo cached")
PY
```

### 4. Download Qwen

Install the Hugging Face CLI and download the expected filename:

```bash
uv tool install huggingface_hub
mkdir -p models
hf download \
  unsloth/Qwen3.5-4B-GGUF \
  Qwen3.5-4B-Q4_K_M.gguf \
  --local-dir models
```

Confirm the model exists at:

```text
~/local-voice-harness/models/Qwen3.5-4B-Q4_K_M.gguf
```

List llama.cpp devices:

```bash
llama-server --list-devices
```

Edit `systemd/user/voice-harness-llm.service` if the NVIDIA device is not `CUDA0`, or if
`llama-server` is installed somewhere other than `/usr/sbin/llama-server`.

### 5. Install Cursor and Herdr

```bash
curl https://cursor.com/install -fsS | bash
agent login

curl -fsSL https://herdr.dev/install.sh | sh

agent --version
herdr --version
```

The harness starts `herdr server` automatically as the transient user service
`voice-harness-herdr.service`. Running the Herdr TUI later attaches to that server.

For Linear support, configure the server in `~/.cursor/mcp.json`, then authenticate
and approve it:

```bash
agent mcp login linear
agent mcp enable linear
agent mcp list
agent mcp list-tools linear
```

OAuth is reused from the same local Cursor user profile. If Linear is unavailable,
ordinary repository tasks still work.

### 6. Configure audio

Find the PipeWire microphone:

```bash
wpctl status
```

The source currently defaults to the microphone from the original development
machine. Override it with a systemd drop-in:

```bash
systemctl --user edit voice-harness-wake.service
```

```ini
[Service]
Environment=VOICE_HARNESS_SOURCE=<PIPEWIRE_SOURCE_NAME>
```

Optional Chatterbox voice cloning accepts a reference WAV:

```ini
[Service]
Environment=VOICE_HARNESS_VOICE=/absolute/path/to/reference.wav
```

The default playback interruption mode is `wake`: saying “Hey Jarvis” while the
assistant is speaking stops queued audio and starts a new wake-prefixed request. The
wake detector is reset at playback boundaries, the microphone must become quiet
before ordinary follow-up VAD is re-armed, and wake interruption is temporarily
suppressed if the assistant's own response contains the wake phrase.

Natural speech barge-in is available with
`VOICE_HARNESS_BARGE_IN_MODE=vad`, but it should only be used with a PipeWire
echo-cancelled source. A physical microphone will usually classify speaker output as
speech and interrupt every response. PipeWire's PulseAudio compatibility layer can
create a session-scoped WebRTC echo-cancel source for testing:

```bash
pactl load-module module-echo-cancel \
  aec_method=webrtc \
  source_name=voice_harness_aec \
  sink_name=voice_harness_aec_sink
wpctl status
```

Set `VOICE_HARNESS_SOURCE=voice_harness_aec` and
`VOICE_HARNESS_BARGE_IN_MODE=vad` in the wake service drop-in. The virtual source
must use the same physical capture/playback devices as the harness; make this module
persistent through the machine's PipeWire/WirePlumber configuration after validating
it interactively. Use `VOICE_HARNESS_BARGE_IN_MODE=off` if no acoustic interruption
is wanted. The streaming client also exposes `StreamingPlayback.cancel()` for an
explicit stop control in local integrations.

### 7. Install and enable services

Use the packaged CLI to install the units, reload systemd, and enable the two
always-on services:

```bash
voice-harness services install
voice-harness services audit
voice-harness services start
```

Use `services install --force` only when intentionally replacing conflicting files
from an older installation. An existing, separately managed `dictation.service` is
preserved by default and is not covered by the shipped hardening policy. After
reviewing and migrating its customizations, adopt the hardened unit with:

```fish
voice-harness services install --force --replace-dictation
voice-harness services audit
```

`--replace-dictation` intentionally replaces the standalone unit. A failed audit
means the effective installed unit or a drop-in still differs from the shipped
policy; do not treat installation as complete.

Qwen and Chatterbox are intentionally not enabled at login. The wake daemon starts
them on demand and stops them when a conversation closes. It also stops them after a
failed turn when no earlier conversation remains active. Manual text turns hold a
cross-process usage lease so daemon cleanup cannot stop their models mid-response.

### Existing standalone dictation installations

The package includes a dictation backend for fresh installs, but cleanup does not
replace an active standalone service such as `~/.local/share/dictation`. The service
installer reports that it preserved the existing unit. The wake daemon continues to
use the same `$XDG_RUNTIME_DIR/dictation.sock`, so both backends are protocol
compatible. Preservation also means the standalone unit may retain unrestricted
environment-file or sandbox behavior. It is not hardened merely because the other
voice-harness units were updated.

## Repository layout

```text
pyproject.toml     Package metadata, extras, and console scripts
uv.lock            Reproducible uv resolution
src/local_voice_harness/
  cursor/          Persisted Cursor jobs, prompts, and worker entry point
  integrations/    Herdr client
  wake/            Wake-word conversation daemon
  stt/             Dictation client, launcher, and server
  tts/             Chatterbox client and server
systemd/user/       Source unit templates
tests/              Package, routing, job, service, and entry-point tests
models/             Ignored local model weights
```

## Verification

```bash
voice-harness status
voice-harness services status
voice-harness services audit
voice-harness services logs -f
```

The installed-unit audit is read-only: it uses `systemctl --user cat/show` to inspect
effective units and drop-ins, resource values, runtime state, and restart counts. It
does not reload, start, stop, or install services. Developers can invoke the same
check as `python -m local_voice_harness.service_units --audit-installed`.

Test without a microphone:

```bash
voice-harness text "Why is the sky blue?"
voice-harness text "Use Cursor to summarize the api-docs repository."
```

### Development quality checks

The default development environment contains only the package and quality tools;
it does not install the CUDA, audio, wake-word, or model extras:

```fish
set python_version 3.12
uv sync --python $python_version
uv run ruff format --check .
uv run ruff check .
uv run pyright --pythonversion $python_version
uv run python -m local_voice_harness.service_units --require-systemd-analyze
uv run pytest
uv run coverage json -o coverage.json
uv run python -m local_voice_harness.coverage_gate coverage.json
```

Use `uv run ruff format .` to apply formatting. Pytest includes branch coverage and
enforces the initial 40% project threshold. The global threshold remains a baseline;
rounded, coarse floors for risk-critical orchestration modules catch substantial
coverage regressions without implying that measured coverage can never decrease. The
service-unit check requires source/package parity and model-default consistency. When
`systemd-analyze` is installed, it verifies the units in an isolated root containing
only allowlisted external dependencies and safe stubs for configured executables, so
clean CI runners do not need model environments installed. CI selects, asserts, and
type-checks against each supported Python interpreter (3.11 and 3.12) without starting
services or downloading models. This checks shipped templates only; CI does not prove
the effective policy of deployed units or local drop-ins. Run `services audit` after
installation.

The [service hardening and local trust policy](docs/service-hardening.md) records each
unit's retained capabilities, resource-limit rationale, unauthenticated loopback LLM
policy, residual risks, and the required real-host smoke checklist.

GPU, audio, Herdr, and complete voice-turn checks are opt-in only. See the
[hardware smoke checklist](docs/hardware-smoke.md) for their entry points and cleanup
steps; CI never runs them.

The [issue #28 test plan](docs/issue-28-test-plan.md) records the protocol,
process-lifecycle, cancellation, delivery, and recovery regressions that must be
integrated after the active #24/#25 runtime changes merge. This foundation does not
claim those production paths are covered.

## Usage

Wake mode:

```text
Hey Jarvis, what time is it?
Hey Jarvis, ask Cursor to summarize the api-docs repository.
Hey Jarvis, ask Cursor to work on Linear issue API-79.
Hey Jarvis, ask Cursor to work on owner/repository#42.
Hey Jarvis, summarize this issue.  # with a GitHub issue focused in Firefox
Hey Jarvis, summarize this ticket. # with a Zendesk ticket focused in Firefox
Hey Jarvis, fork this repo and add Venice.  # with a public GitHub repo focused
Hey Jarvis, ask Cursor to check out this PR and make sure it works.  # with a PR focused
Hey Jarvis, what is the status of that Cursor job?
Hey Jarvis, cancel that Cursor job.
```

Push-to-talk/manual commands:

```bash
voice-harness begin
voice-harness end
voice-harness cancel
voice-harness transcribe
voice-harness dictate toggle
voice-harness text "Use Cursor to inspect this repository."
voice-harness status
```

`begin` records audio. `end` stops capture, transcribes, routes the request, speaks
the response, and prints stage timings. `dictate toggle` records on the first
invocation, then transcribes and inserts text into the focused window on the next
invocation without starting the conversational models. Automatic injection uses
native Herdr delivery for Cursor panes, simulated typing for terminals, and
clipboard paste for other graphical applications. Dictation is blocked while
RuneLite is focused because generated input may violate Jagex's rules.

Playback starts after the first sentence/clause is synthesized instead of waiting for
the complete response. In the default configuration, say “Hey Jarvis” during
playback to interrupt and immediately ask another question. Chatterbox cannot cancel
an active `generate()` call, so server-side cancellation may take up to one short
chunk; PipeWire playback and already queued chunks stop immediately.

On X11, Hyprland, and Sway, each new conversational request checks whether Firefox
is focused. The harness briefly selects and copies the address bar, restores the
previous clipboard, and dismisses the address bar without navigating. A focused
GitHub page contributes its URL; a focused issue page also contributes title, state,
body, labels, and recent comments fetched through the authenticated `gh` CLI. A
spoken `owner/repository#number` reference fetches the same bounded issue context
without requiring Firefox to be focused. GitHub access remains required; issue
metadata is persisted only as part of the active job and is not an offline cache. A
focused pull request page adds the same details plus its draft state, source and
target branches, and change summary, and lets a Cursor request check the branch out
locally. A focused `https://<tenant>.zendesk.com/agent/tickets/<number>` page
contributes its URL, tenant, ticket number, and bounded rendered page text copied
from the authenticated browser session; no Zendesk API credentials are required. Only
text currently loaded and selectable in the page is available, so collapsed or
unloaded comments may be absent. Page content is treated as untrusted input. Missing
tools, unsupported Wayland compositors, focus changes during capture, and browser or
GitHub errors simply omit some or all browser context without failing the voice
request.

For example, bind Super+D in i3:

```text
bindsym $mod+d exec --no-startup-id /home/joshuam/.local/bin/voice-harness dictate toggle
```

The equivalent Sway binding is:

```text
bindsym $mod+d exec /home/joshuam/.local/bin/voice-harness dictate toggle
```

For Hyprland:

```text
bind = SUPER, D, exec, /home/joshuam/.local/bin/voice-harness dictate toggle
```

Native Wayland automation is supported on Hyprland and Sway. It uses `hyprctl` or
`swaymsg` to identify the focused window, `wl-copy`/`wl-paste` for the clipboard,
and `wtype` for keyboard input. GNOME, KDE Plasma, and other compositors are not
currently supported; set `DICTATION_INJECT=stdout` if focused-window insertion is
not required there.

The wake service needs the compositor environment to collect Firefox context.
Import it into the systemd user manager from compositor startup. For Sway:

```text
exec_always systemctl --user import-environment WAYLAND_DISPLAY XDG_CURRENT_DESKTOP SWAYSOCK
```

For Hyprland:

```text
exec-once = systemctl --user import-environment WAYLAND_DISPLAY XDG_CURRENT_DESKTOP HYPRLAND_INSTANCE_SIGNATURE
```

Restart `voice-harness-wake.service` after adding the import. Dictation commands
launched directly by compositor keybindings already inherit the required
environment.

## Service management

The Python CLI is the user-facing service manager; systemd remains responsible for
process supervision, dependencies, restarts, and journal logs:

```bash
voice-harness services install
voice-harness services start
voice-harness services stop
voice-harness services restart
voice-harness services status
voice-harness services audit
voice-harness services logs
voice-harness services logs --follow
voice-harness services uninstall
```

`start` launches dictation and the wake listener. Qwen and Chatterbox remain
on-demand. `stop` shuts down the wake listener, model servers, and dictation in a
safe order but leaves Herdr and active Cursor agents running.

Stopping or uninstalling Herdr requires explicit confirmation through the option:

```bash
voice-harness services stop --include-herdr
voice-harness services uninstall --include-herdr
```

The small process launchers are Python. The `.service` files remain declarative
systemd units rather than implementing a second process supervisor.

## Configuration

Only the variables marked “wake drop-in” may be added to
`voice-harness-wake.service` with `systemctl --user edit`. The installed-unit audit
rejects other extra variables and every `EnvironmentFile` on every shipped service.
Dictation backend selectors belong in `~/.config/dictation/backend.env`, which the
launcher parses through its separate allowlist.

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
| `DICTATION_INJECT` | Focused-window insertion mode (`auto`, `paste`, `type`, or `stdout`) | `auto` | Wake drop-in |
| `DICTATION_REPLACEMENTS` | Semicolon-separated STT corrections | Cursor/Herdr defaults | Wake drop-in |
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

## Personal vocabulary and entity aliases

A local, user-owned store improves transcription and routing for repository names,
issue references, developer tools, acronyms, and recurring speech-recognition
mistakes. It is edited only through explicit `voice-harness vocabulary` commands; the
harness never silently learns a correction.

Storage format and location:

- A single JSON document (schema `version` 1) at
  `$XDG_CONFIG_HOME/voice-harness/vocabulary.json` (default
  `~/.config/voice-harness/vocabulary.json`). JSON keeps the harness free of runtime
  dependencies because it round-trips with only the standard library. The file is
  written privately (`0o600`) with sorted keys so backups and diffs are stable.
- Two entry kinds are stored. A **replacement** rewrites recognized text (`spoken`)
  to a corrected form (`written`). An **alias** maps a spoken `phrase` to a canonical
  entity `target`: an `owner/repo` repository (`kind` `repository`) or an
  `owner/repo#number` issue (`kind` `issue`). The kind is inferred from the target.

Normalization:

- Replacement sources and alias phrases are trimmed, have internal whitespace
  collapsed to single spaces, and are compared case-insensitively. Alias phrases are
  stored case-folded; written and target values keep their case.
- Matching against transcribed text is case-insensitive, whitespace-flexible, and only
  fires on whole phrases (never inside a longer word).

Precedence and conflict behavior:

- STT corrections apply in order of user vocabulary first, then
  `DICTATION_REPLACEMENTS`, then built-in defaults. A user replacement overrides any
  static entry with the same spoken source.
- When resolving aliases, longer phrases match before shorter ones so the most
  specific alias wins.
- Each spoken source and each alias phrase maps to exactly one value. `add` rejects a
  conflicting key that already resolves to a different value unless `--force` is given,
  and a stored file containing the same key twice with different values is rejected as
  ambiguous on load.

Aliases are resolved as a deterministic pre-pass on the trusted utterance before the
intent router and repository/issue detection run, so `owner/repo` and
`owner/repo#number` references become available to existing routing. Vocabulary
content never leaves the machine; it only appears externally when a resolved alias or
corrected transcription becomes part of an agent prompt the user explicitly requested.

```bash
voice-harness vocabulary list
voice-harness vocabulary list --kind alias
voice-harness vocabulary add replacement "herder" "herdr"
voice-harness vocabulary add alias "the harness repo" "joshua-mo-143/local-voice-harness"
voice-harness vocabulary add alias "harness bug" "joshua-mo-143/local-voice-harness#35"
voice-harness vocabulary remove replacement "herder"
voice-harness vocabulary remove alias "the harness repo"
voice-harness vocabulary export --output vocabulary-backup.json
voice-harness vocabulary import vocabulary-backup.json          # merge
voice-harness vocabulary import vocabulary-backup.json --replace # overwrite
```

`export` without `--output` prints the JSON document to stdout for inspection or
piping to a backup. `import` merges by default (incoming entries win on conflict) or
replaces the whole store with `--replace`. Deleting an entry uses `remove`; deleting
the store entirely is a matter of removing the JSON file. The dictation service reads
the file on each transcription, so edits take effect without restarting it.

## Performance observed

Measured with all models warm on the RTX 5070 Ti Laptop GPU. Dictation figures use
the earlier Whisper large-v3 backend; Parakeet TDT 0.6B v2 measurements are pending:

- Whisper large-v3: approximately 0.58 seconds for a short request.
- Qwen response: 0.22–0.53 seconds; first CUDA request TTFT pending re-measurement
  (Vulkan baseline was approximately 5 seconds).
- Chatterbox: 0.53 seconds for 2.72 seconds of audio; longer replies now begin
  playing after their first sentence/clause is ready.
- Cursor delegation: task-dependent and normally handled as a background job.

## Security notes

- Voice transcription can be wrong. Review all Cursor changes before committing.
- Herdr agents are started with workspace trust but not Cursor `--force`.
- Ticket and MCP content is treated as untrusted input and inferred paths are
  validated against local Git repositories.
- Focused GitHub issue and pull request content is read through `gh`, and rendered
  Zendesk ticket content is copied from the browser session; all are bounded before
  prompting and treated as untrusted external data.
- Repository cloning requires explicit Rofi confirmation, accepts only HTTPS or SSH
  Git URLs, and places the checkout beneath the configured project root.
- Merely focusing a GitHub page cannot create a fork. The original spoken request must
  unambiguously ask for one, and the user must separately confirm before the validated
  public repository is forked.
- Checking out a focused pull request clones or reuses its repository below the GitHub
  root and runs `gh pr checkout` only in a job-unique, reserved worktree. Recovery
  retries that same worktree; failed preparation quarantines it.
- Jobs never automatically commit, push, open pull requests, or remove worktrees.
- Runtime job metadata and conversational audio live under
  `$XDG_RUNTIME_DIR/voice-harness`; focused dictation audio lives under
  `$XDG_RUNTIME_DIR/dictation`.
- The unauthenticated llama.cpp API is bound to `127.0.0.1` for this trusted
  single-user workstation. Same-account processes are trusted; loopback is not a
  per-UID boundary on a mutually untrusted multi-user host.
- Shipped services use service-specific systemd hardening and bounded resources; see
  the [hardening policy](docs/service-hardening.md) for deliberate exceptions and
  host checks.

## Troubleshooting

Wake listener will not start:

```bash
voice-harness services status
voice-harness services logs
```

CUDA library or model errors:

```bash
journalctl --user -u dictation.service -u voice-harness-tts.service -n 100
nvidia-smi
```

If dictation reports a missing Python module, ensure the installed extra matches
`DICTATION_BACKEND`: use `dictation` for Parakeet or `dictation-whisper` for
faster-whisper, then restart `dictation.service`. Unknown backend names are rejected
at startup.

Herdr/Cursor failures:

```bash
herdr status server
herdr agent list
agent status
agent mcp list
```

Wrong llama.cpp GPU:

```bash
llama-server --list-devices
systemctl --user edit voice-harness-llm.service
```

After changing shipped units and intentionally adopting the bundled dictation unit:

```fish
voice-harness services install --force --replace-dictation
voice-harness services audit
voice-harness services restart
```
