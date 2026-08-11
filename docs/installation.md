# Manual installation

The [Quick install](../README.md#installation) path runs `scripts/install.sh`,
which performs every deterministic step here and pauses only for the interactive
logins you have not already completed. This document is the manual, step-by-step
breakdown for partial setups, troubleshooting, or non-Arch systems.

The supplied systemd units assume the repository is cloned to
`$HOME/local-voice-harness`.

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

The tested configuration uses CUDA for Parakeet and Chatterbox, but dictation can
run independently on CPU. CPU dictation has substantially higher latency. The
optional faster-whisper backend supports the same `auto`, `cpu`, and `cuda` device
selection.

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
- A recent [llama.cpp](https://github.com/ggml-org/llama.cpp) build with CUDA and
  `llama-server`. The server runs with `--jinja` so it uses the model's native chat
  template, which llama.cpp requires for Qwen3.5 tool calling.
- The [Cursor CLI](https://cursor.com/docs/cli/installation).
- [Herdr](https://herdr.dev).
- A working NVIDIA driver.

On Arch/CachyOS, the base packages are approximately:

```bash
paru -S --needed pipewire libnotify git curl github-cli xdotool xclip \
  wl-clipboard wtype uv libsndfile ffmpeg
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
ffmpeg -version
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

## 1. Clone and install the management/wake package

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

## 2. Create the bundled dictation environment

```fish
env UV_PROJECT_ENVIRONMENT=.venv-dictation \
  uv sync --python 3.11 --extra dictation --no-dev
voice-harness config set compute.dictation_device cpu
```

This CPU profile installs `onnxruntime`, not `onnxruntime-gpu`, and resolves no
NVIDIA Python packages. The default backend is Parakeet TDT 0.6B v2. Its first
start downloads `nemo-parakeet-tdt-0.6b-v2` from Hugging Face.

For CUDA-enabled Parakeet, select the separate profile and device:

```fish
env UV_PROJECT_ENVIRONMENT=.venv-dictation \
  uv sync --python 3.11 --extra dictation-cuda --no-dev
voice-harness config set compute.dictation_device cuda
```

Use `auto` with either profile to prefer CUDA when its runtime provider is
available and otherwise fall back compatibly to CPU. Explicit `cuda` fails clearly
at service startup when the provider or device is unavailable.

To use the supported faster-whisper backend instead, install its separate extra and
select it in the unified configuration:

```fish
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

## 3. Create the Chatterbox environment

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

## Optional Venice LLM and TTS backends

The LLM and TTS providers can be selected independently. The installer prompts for
each provider and skips the corresponding local environment or model download when
Venice is selected. Its choices can also be supplied non-interactively:

```fish
env LLM_PROVIDER=venice TTS_PROVIDER=venice ./scripts/install.sh
```

Venice credentials are stored through libsecret in the desktop Secret Service, not
in a file, command argument, environment variable, or systemd unit. For a manual
installation:

```fish
paru -S --needed libsecret oo7
voice-harness credentials set
mkdir -p "$HOME/.config/voice-harness"
printf '%s\n' \
  '[llm]' \
  'provider = "venice"' \
  '' \
  '[tts]' \
  'provider = "venice"' \
  >"$HOME/.config/voice-harness/backends.toml"
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

## 4. Download Qwen

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

## 5. Install Cursor and Herdr

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

## 6. Configure audio

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

## 7. Install and enable services

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

The shipped units do not embed user-selected providers, models, CUDA devices,
dictation selectors, or wake/integration settings. The dictation and local-LLM
launchers resolve those values from the same typed `config.toml` model at process
start. Keep service drop-ins limited to operational policy; the audit rejects
user-choice `Environment=` overrides and all `EnvironmentFile=` directives.

Qwen and Chatterbox are intentionally not enabled at login. The wake daemon starts
them on demand and stops them when a conversation closes. It also stops them after a
failed turn when no earlier conversation remains active. Manual text turns hold a
cross-process usage lease so daemon cleanup cannot stop their models mid-response.

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
