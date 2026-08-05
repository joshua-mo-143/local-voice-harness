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
  -> faster-whisper large-v3 (CUDA)
  -> Qwen3.5-4B Q4_K_M via llama.cpp (Vulkan)
       -> ordinary conversational response
       -> Herdr-managed Cursor agent and Linear MCP
  -> Chatterbox Turbo (CUDA)
  -> PipeWire playback
```

The always-on wake daemon verifies OpenWakeWord candidates with Whisper to reject
false activations. A request that takes longer than five seconds becomes a persisted
background job, and its completion or clarification question is spoken later. Job
transitions are serialized across the daemon and detached workers; abandoned jobs
are recovered at daemon startup and during normal polling. Spoken background results
use at-least-once delivery: playback is acknowledged only after it succeeds, so a
crash at that boundary may repeat a result but will not silently lose it.

Spoken responses use chunk-level streaming. Chatterbox still generates a complete
waveform for each short sentence or clause, but the next chunk is synthesized while
the current chunk is sent through one low-latency PipeWire playback stream. This is
not native sample streaming from the model. Playback sessions are serialized across
processes so manual commands and daemon announcements cannot overlap.

Cursor routing works as follows:

1. Prefer an idle Cursor agent already running in the requested checkout.
2. For a Linear issue without a repository name, ask a dedicated routing agent to
   inspect the ticket through Linear MCP and infer the repository.
3. Create or reuse a `voice/<issue-key>` Git worktree for Linear implementation work.
4. Start a new Cursor agent through Herdr when no suitable agent exists.
5. Reserve that agent until it finishes, is blocked, or the job is cancelled.

The harness never automatically commits, pushes, opens pull requests, modifies
Linear, or deletes generated worktrees.

## Compute requirements

Tested configuration:

- Linux x86-64 with systemd user services and PipeWire.
- NVIDIA GeForce RTX 5070 Ti Laptop GPU with 12 GB VRAM.
- 32 GB system RAM and swap.
- CUDA-capable NVIDIA driver plus a Vulkan-capable llama.cpp build.
- Python 3.11 for the management, wake, TTS, and bundled dictation environments.

Practical requirements for the included model choices:

| Resource | Minimum | Recommended |
| --- | ---: | ---: |
| GPU VRAM | 12 GB | 16 GB |
| System RAM | 16 GB | 32 GB |
| Free disk | 25 GB | 30+ GB |
| CPU | Modern 4-core x86-64 | 8+ cores |

With all models warm, the tested machine used approximately 10.4 GB of GPU memory.
The main disk consumers are:

- Qwen3.5-4B Q4_K_M GGUF: 2.6 GB.
- Chatterbox Turbo cache: 3.8 GB.
- faster-whisper large-v3 cache: 2.9 GB.
- Current Python environments: approximately 13 GB combined.

The current implementation requires CUDA for Whisper and Chatterbox. CPU-only use
would require code and service changes and would have substantially higher latency.

## External prerequisites

Install these before setting up Python environments:

- PipeWire tools (`pw-record` and `pw-play`).
- `libnotify`/`notify-send`.
- Git, curl, the GitHub CLI (`gh`), and systemd user services.
- `xdotool` and `xclip` for X11 focused-window automation.
- [uv](https://docs.astral.sh/uv/) for reproducible Python versions/environments.
- A recent [llama.cpp](https://github.com/ggml-org/llama.cpp) build with Vulkan and
  `llama-server`.
- The [Cursor CLI](https://cursor.com/docs/cli/installation).
- [Herdr](https://herdr.dev).
- A working NVIDIA driver.

On Arch/CachyOS, the base packages are approximately:

```bash
paru -S --needed pipewire libnotify git curl github-cli xdotool xclip uv libsndfile
```

Package names for llama.cpp and NVIDIA drivers vary. Verify the required commands:

```bash
pw-record --version
pw-play --version
llama-server --version
nvidia-smi
```

Authenticate the GitHub CLI to let focused issue pages include private-repository
details:

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

```bash
UV_PROJECT_ENVIRONMENT=.venv-dictation \
  uv sync --python 3.11 --extra dictation --no-dev
```

The first dictation start downloads faster-whisper large-v3 from Hugging Face.

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
  jc-builds/Qwen3.5-4B-Q4_K_M-GGUF \
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

Edit `systemd/user/voice-harness-llm.service` if the NVIDIA device is not `Vulkan1`, or if
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
voice-harness services start
```

Use `services install --force` only when intentionally replacing conflicting files
from an older installation. An existing, separately managed `dictation.service` is
preserved by default; pass `--replace-dictation` only when intentionally migrating
to the bundled backend.

Qwen and Chatterbox are intentionally not enabled at login. The wake daemon starts
them on demand and stops them when a conversation closes. It also stops them after a
failed turn when no earlier conversation remains active. Manual text turns hold a
cross-process usage lease so daemon cleanup cannot stop their models mid-response.

### Existing standalone dictation installations

The package includes a dictation backend for fresh installs, but cleanup does not
replace an active standalone service such as `~/.local/share/dictation`. The service
installer reports that it preserved the existing unit. The wake daemon continues to
use the same `$XDG_RUNTIME_DIR/dictation.sock`, so both backends are protocol
compatible.

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
voice-harness services logs -f
```

Test without a microphone:

```bash
voice-harness text "Why is the sky blue?"
voice-harness text "Use Cursor to summarize the api-docs repository."
```

### Development quality checks

The default development environment contains only the package and quality tools;
it does not install the CUDA, audio, wake-word, or model extras:

```bash
uv sync --python 3.12
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

Use `uv run ruff format .` to apply formatting. Pytest includes branch coverage and
enforces the initial 40% project threshold. CI runs the same checks on every
supported Python version (3.11 and 3.12) without starting services or downloading
models.

## Usage

Wake mode:

```text
Hey Jarvis, what time is it?
Hey Jarvis, ask Cursor to summarize the api-docs repository.
Hey Jarvis, ask Cursor to work on Linear issue API-79.
Hey Jarvis, summarize this issue.  # with a GitHub issue focused in Firefox
Hey Jarvis, summarize this ticket. # with a Zendesk ticket focused in Firefox
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

On X11, each new conversational request checks whether Firefox is focused. The
harness briefly selects and copies the address bar, restores the previous clipboard,
and dismisses the address bar without navigating. A focused GitHub page contributes
its URL; a focused issue page also contributes title, state, body, labels, and recent
comments fetched through the authenticated `gh` CLI. A focused
`https://<tenant>.zendesk.com/agent/tickets/<number>` page contributes its URL,
tenant, ticket number, and bounded rendered page text copied from the authenticated
browser session; no Zendesk API credentials are required. Only text currently loaded
and selectable in the page is available, so collapsed or unloaded comments may be
absent. Page content is treated as untrusted input. Missing tools, unsupported
sessions such as native Wayland, focus changes during capture, and browser or GitHub
errors simply omit some or all browser context without failing the voice request.

For example, bind Super+D in i3:

```text
bindsym $mod+d exec --no-startup-id /home/joshuam/.local/bin/voice-harness dictate toggle
```

## Service management

The Python CLI is the user-facing service manager; systemd remains responsible for
process supervision, dependencies, restarts, and journal logs:

```bash
voice-harness services install
voice-harness services start
voice-harness services stop
voice-harness services restart
voice-harness services status
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

Environment variables can be added to systemd drop-ins:

| Variable | Purpose | Default |
| --- | --- | --- |
| `VOICE_HARNESS_SOURCE` | PipeWire microphone source | Development-machine source |
| `VOICE_HARNESS_VOICE` | Chatterbox reference WAV | Built-in voice |
| `VOICE_HARNESS_WAKE_THRESHOLD` | OpenWakeWord activation threshold | `0.55` |
| `VOICE_HARNESS_MIN_SPEECH_RMS` | Speech energy gate | `1100` |
| `VOICE_HARNESS_BARGE_IN_MODE` | Playback interruption (`wake`, `vad`, or `off`) | `wake` |
| `VOICE_HARNESS_BARGE_IN_SPEECH_FRAMES` | Consecutive 80 ms speech frames for VAD barge-in | `5` |
| `VOICE_HARNESS_PLAYBACK_QUIET_FRAMES` | Quiet 80 ms frames required after playback | `4` |
| `VOICE_HARNESS_PLAYBACK_QUIET_TIMEOUT_SECONDS` | Maximum post-playback echo drain | `2` |
| `VOICE_HARNESS_PLAYBACK_LATENCY` | `pw-play` raw-stream target latency | `100ms` |
| `VOICE_HARNESS_CURSOR_FOREGROUND_SECONDS` | Time before a Cursor job backgrounds | `5` |
| `VOICE_HARNESS_HERDR_BIN` | Herdr executable | `~/.local/bin/herdr` |
| `VOICE_HARNESS_PROJECT_ROOT` | Allowed root for inferred repositories | Home directory |
| `DICTATION_MODEL` | faster-whisper model | `large-v3` |
| `DICTATION_COMPUTE` | faster-whisper compute type | `float16` |
| `DICTATION_INJECT` | Focused-window insertion mode (`auto`, `paste`, `type`, or `stdout`) | `auto` |
| `DICTATION_REPLACEMENTS` | Semicolon-separated STT corrections | Cursor/Herdr defaults |

`VOICE_HARNESS_PROJECT_ROOT` can narrow repository discovery to another directory.

## Performance observed

Measured with all models warm on the RTX 5070 Ti Laptop GPU:

- Whisper large-v3: approximately 0.58 seconds for a short request.
- Qwen response: 0.22–0.53 seconds; first Vulkan request approximately 5 seconds.
- Chatterbox: 0.53 seconds for 2.72 seconds of audio; longer replies now begin
  playing after their first sentence/clause is ready.
- Cursor delegation: task-dependent and normally handled as a background job.

## Security notes

- Voice transcription can be wrong. Review all Cursor changes before committing.
- Herdr agents are started with workspace trust but not Cursor `--force`.
- Ticket and MCP content is treated as untrusted input and inferred paths are
  validated against local Git repositories.
- Focused GitHub issue and rendered Zendesk ticket content is bounded before
  prompting and treated as untrusted external data.
- Jobs never automatically commit, push, open pull requests, or remove worktrees.
- Runtime job metadata and audio live under `$XDG_RUNTIME_DIR/voice-harness`.

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

After changing a unit:

```bash
voice-harness services install --force
voice-harness services restart
```
