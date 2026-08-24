# Manual installation

The [Quick start](../README.md#quick-start) path runs `scripts/install.sh`, which
supports Arch, Ubuntu/Debian, and Fedora. It installs Venice LLM/TTS plus the
packages required by the audio stack and dictation compute profile, then pauses
for interactive logins when needed. Showcase installs skip CUDA packages on every
supported family. Re-running the installer is idempotent.

The installer discovers the checkout, `~/.local/bin`, and the systemd user
directory from its own path and XDG locations. If the checkout is not
`$HOME/local-voice-harness`, it creates that symlink so shipped user units resolve.
The standard-library plan resolver requires Python 3.11 or newer. Before resolving
packages, the installer installs the distro `python3` package with `paru`, `apt-get`,
or `dnf` when necessary, verifies its version, and stops with manual guidance if the
distro package is still too old.

## Compute requirements

Venice is the only supported LLM and TTS provider. Wake detection and dictation
are always local. Cursor, GitHub, Linear, and Venice are external services when
enabled.

The tested hosted configuration is:

- Linux x86-64 with systemd user services and PipeWire.
- Python 3.11 for the management, wake, TTS proxy, and bundled dictation
  environments.
- An NVIDIA GPU only when using the `local-cuda` dictation profile.

Practical requirements:

| Resource | Minimum | Recommended |
| --- | ---: | ---: |
| GPU VRAM | none for showcase | 4 GB for CUDA dictation |
| System RAM | 8 GB | 16 GB |
| Free disk | 10 GB | 15+ GB |
| CPU | Modern 4-core x86-64 | 8+ cores |

The main disk consumers are:

- Parakeet TDT 0.6B v2 ONNX cache: less than 1 GB.
- Current Python environments: several gigabytes combined.

Dictation can run on CPU or CUDA. CPU dictation has substantially higher latency.
The optional faster-whisper backend supports the same `auto`, `cpu`, and `cuda`
device selection. The showcase installer profile is the GPU-free path:

```bash
PROFILE=showcase ./scripts/install.sh
./scripts/install.sh --profile showcase
```

It installs no CUDA or NVIDIA packages and uses the CPU `dictation` extra.
`PROFILE=local-cuda` keeps Venice LLM/TTS and installs CUDA packages for
dictation only.

## External prerequisites

Install these before setting up Python environments:

- PipeWire tools (`pw-record` and `pw-play`).
- FFmpeg for pitch-preserving Venice TTS speed adjustment.
- `libnotify`/`notify-send`.
- Git, curl, the GitHub CLI (`gh`), and systemd user services.
- `xdotool` and `xclip` for X11 focused-window automation, or `wtype` and
  `wl-clipboard` for Hyprland/Sway focused-window automation.
- Rofi for repository selection and pasteable clone-URL prompts.
- [uv](https://docs.astral.sh/uv/) for reproducible Python versions/environments.
- The [Cursor CLI](https://cursor.com/docs/cli/installation).
- [Herdr](https://herdr.dev).
- A working NVIDIA driver when using CUDA dictation.

On Arch/CachyOS, the base packages are approximately:

```bash
paru -S --needed pipewire libnotify git curl github-cli xdotool xclip \
  wl-clipboard wtype uv libsndfile ffmpeg
```

On Ubuntu/Debian, the installer maps those names to `apt-get` packages such as
`libnotify-bin`, `gh`, `libsecret-tools`, and `gnome-keyring`, and bootstraps
`uv` when the distro does not package it. On Fedora it uses `dnf` with the
same mapping (`gh`, `libsecret`, `gnome-keyring`). Showcase installs do not
install CUDA packages on any family. An explicit CUDA dictation plan on
Ubuntu/Debian or Fedora fails before package installation because those adapters
cannot currently satisfy `cuda`; use the supported Arch adapter or the showcase
profile.

On Arch/CachyOS, CUDA dictation needs the distro CUDA package:

```bash
paru -S --needed cuda
```

Verify the required commands:

```bash
pw-record --version
pw-play --version
ffmpeg -version
```

For CUDA dictation, also confirm `nvidia-smi`.

Authenticate the GitHub CLI to let focused issue pages include repository details and
to create forks explicitly requested through the harness:

```bash
gh auth login
gh auth status
```

## 1. Clone and install the management/wake package

```fish
git clone <YOUR_GITHUB_REPOSITORY_URL> "$HOME/local-voice-harness"
cd "$HOME/local-voice-harness"
scripts/sync-wake.sh
mkdir -p "$HOME/.local/bin"
ln -sfn "$HOME/local-voice-harness/.venv/bin/voice-harness" \
  "$HOME/.local/bin/voice-harness"
```

This installs the package, OpenWakeWord dependencies, and all console entry points
in `.venv`. Ensure `$HOME/.local/bin` is on `PATH`.
The `wake`, `dictation`, and `tts` extras are intentionally installed into separate
environments because their CUDA and NumPy constraints are not all compatible.

The wake sync downloads `hey_jarvis_v0.1.onnx` when it is missing and leaves an
existing model untouched.

## 2. Create the bundled dictation environment

```bash
env UV_PROJECT_ENVIRONMENT=.venv-dictation \
  uv sync --python 3.11 --extra dictation --no-dev
voice-harness config set compute.dictation_device cpu
```

This CPU profile installs `onnxruntime`, not `onnxruntime-gpu`, and resolves no
NVIDIA Python packages. The default backend is Parakeet TDT 0.6B v2. Its first
start downloads `nemo-parakeet-tdt-0.6b-v2` from Hugging Face.

For CUDA-enabled Parakeet, select the separate profile and device:

```bash
env UV_PROJECT_ENVIRONMENT=.venv-dictation \
  uv sync --python 3.11 --extra dictation-cuda --no-dev
voice-harness config set compute.dictation_device cuda
```

Use `auto` with either profile to prefer CUDA when its runtime provider is
available and otherwise fall back compatibly to CPU. Explicit `cuda` fails clearly
at service startup when the provider or device is unavailable.

To use the supported faster-whisper backend instead, install its separate extra and
select it in the unified configuration:

```bash
env UV_PROJECT_ENVIRONMENT=.venv-dictation \
  uv sync --python 3.11 --extra dictation-whisper --no-dev
voice-harness config set compute.dictation_backend whisper
voice-harness config set compute.dictation_model large-v3-turbo
voice-harness config set compute.dictation_device cpu
```

Install `dictation` (CPU) or `dictation-cuda` for Parakeet, or
`dictation-whisper` for faster-whisper. These are alternative backend environments,
not a requirement to install more than one. On CPU, faster-whisper maps configured
`float16` and `int8_float16` compute types to `int8`; CUDA keeps the configured
compute type.

Existing `~/.config/dictation/backend.env` files remain supported as
higher-precedence legacy resolver inputs. The allowlist accepts only backend,
device, model, language, compute, and quantization selectors. Socket, CUDA/Hugging Face
cache, temporary, home, and XDG path variables are service-owned and cannot be
overridden by that file. New configuration commands never create or update it.

## 3. Configure Venice LLM and TTS

Venice is the only supported LLM and TTS provider. Local Qwen and Chatterbox
are not a supported product path.

Venice credentials are stored through libsecret in the desktop Secret Service, not
in a file, command argument, environment variable, or systemd unit. For a manual
installation:

```bash
paru -S --needed libsecret oo7
voice-harness credentials set
voice-harness config set providers.llm.provider venice
voice-harness config set providers.tts.provider venice
```

`credentials set` prompts without echo and sends the key to libsecret's
`secret-tool` over standard input. Use
`voice-harness credentials status` to check it without printing the key, or
`voice-harness credentials delete` to remove it. The desktop keyring may prompt to
unlock when the session collection is locked.

Venice chat uses its OpenAI-compatible function-calling API. Select an LLM model
whose Models API entry reports `supportsFunctionCalling`. Venice TTS voice IDs are
case-sensitive and model-specific; confirm the pair in the
[Venice TTS model catalog](https://docs.venice.ai/models/text-to-speech).
See [Configuration](configuration.md#ai-backends) for model, voice, endpoint, timeout,
and speed options.

## 4. Install Cursor and Herdr

```bash
curl https://cursor.com/install -fsS | bash
agent login

curl -fsSL https://herdr.dev/install.sh | sh

agent --version
herdr --version
```

The harness starts `herdr server` automatically as the transient user service
`voice-harness-herdr.service`. Running the Herdr TUI later attaches to that server.

For Linear support, first configure the server in `~/.cursor/mcp.json`. Cursor stores
MCP OAuth state per workspace, so authenticate one dedicated, trusted checkout whose
MCP access is suitable for every harness-managed ticket workspace:

```bash
SOURCE_CHECKOUT=/absolute/path/to/trusted-checkout
cd "$SOURCE_CHECKOUT"
agent mcp login linear
agent mcp enable linear
agent mcp list
agent mcp list-tools linear
chmod 600 "$HOME/.cursor/projects/$(printf '%s' "$SOURCE_CHECKOUT" | sed -E 's/[^[:alnum:]]+/-/g; s/^-|-$//g')/mcp-auth.json"
voice-harness config set platform.cursor_mcp_auth_source "$SOURCE_CHECKOUT"
voice-harness integrations enable linear
voice-harness integrations doctor
```

The configured source path must be absolute and its `mcp-auth.json` must be a regular
file owned by the harness user with mode `0600`. The harness links that opaque auth
file into trusted router, planner, reviewer, and implementer workspaces without
reading or logging token content. See [Configuration](configuration.md) for the
normalization and collision rules. If Linear is unavailable, ordinary repository
tasks still work.

## 5. Configure audio

Capture and playback default to PipeWire's system-selected source and sink.
To pin a microphone, pick from the devices the harness enumerates:

```console
voice-harness config set audio.source
```

That writes `audio.source` and `dictation.source`, so the next
`voice-harness-dictate` process uses the same capture node. Interactive
`voice-harness setup` offers the same picker. Store an explicit sink only when
you need to override playback:

```console
voice-harness config set audio.sink '<PIPEWIRE_SINK_NAME>'
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

Set the source and mode through unified configuration:

```console
voice-harness config set audio.source voice_harness_aec
voice-harness config set audio.barge_in_mode vad
```

The virtual source must use the same physical capture/playback devices as the
harness; make this module persistent through the machine's PipeWire/WirePlumber
configuration after validating it interactively. Select `off` for
`audio.barge_in_mode` if no acoustic interruption is wanted. The streaming client
also exposes `StreamingPlayback.cancel()` for an explicit stop control in local
integrations.

## 6. Install and enable services

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

```bash
voice-harness services install --force --replace-dictation
voice-harness services audit
```

`--replace-dictation` intentionally replaces the standalone unit. A failed audit
means the effective installed unit or a drop-in still differs from the shipped
policy; do not treat installation as complete.

The shipped units do not embed user-selected providers, models, CUDA devices,
dictation selectors, or wake/integration settings. The dictation launcher
resolves those values from the same typed `config.toml` model at process start.
Keep service drop-ins limited to operational policy; the audit rejects
user-choice `Environment=` overrides and all `EnvironmentFile=` directives.

## Existing standalone dictation installations

The package includes a dictation backend for fresh installs, but cleanup does not
replace an active standalone service such as `~/.local/share/dictation`. The service
installer reports that it preserved the existing unit. The wake daemon continues to
use the same `$XDG_RUNTIME_DIR/dictation.sock`, so both backends are protocol
compatible: the client detects a legacy server and retries once with the path-only
request. That fallback cannot provide the bundled versioned protocol's explicit
transcript acknowledgment, crash recovery, or guarantee that failed delivery retains
audio. Preservation also means the standalone unit may retain unrestricted
environment-file or sandbox behavior. It is not hardened merely because the other
voice-harness units were updated.
