#!/usr/bin/env bash
# One-shot installer for the Local Voice Agent Harness on Arch, Ubuntu, and Fedora.
#
# Runs every deterministic setup step (system packages, uv environments, model
# downloads, service install) and only pauses for interactive logins that are
# not already authenticated. Safe to re-run: each step is idempotent.
#
# Overridable via environment:
#   PYTHON_VERSION      Python for the harness envs (default 3.11)
#   CHATTERBOX_DIR      TTS env location (default $HOME/chatterbox-audition)
#   PROFILE             Installation profile: showcase or local-cuda
#   LLM_PROVIDER        LLM backend: local or venice (prompts when unset)
#   TTS_PROVIDER        TTS backend: local or venice (prompts when unset)
#   LLM_DEVICE          Local LLM compute: auto, cpu, or cuda
#   TTS_DEVICE          Local TTS compute: auto, cpu, or cuda
#   DICTATION_DEVICE    Dictation compute: auto, cpu, or cuda
#   SKIP_SYSTEM_PACKAGES=1   Skip distro package installation
#   SKIP_MODELS=1            Skip Qwen/Chatterbox downloads
#   SKIP_AUTH=1              Skip the interactive gh/cursor/linear logins
#
# Showcase (Venice LLM/TTS + CPU dictation) installs no CUDA or NVIDIA
# packages and omits local LLM/TTS extras, models, and the local LLM service:
#   PROFILE=showcase ./scripts/install.sh
#   ./scripts/install.sh --profile showcase
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CHATTERBOX_DIR="${CHATTERBOX_DIR:-$HOME/chatterbox-audition}"
QWEN_REPO="unsloth/Qwen3.5-4B-GGUF"
QWEN_FILE="Qwen3.5-4B-Q4_K_M.gguf"
PROFILE="${PROFILE:-}"

# The script lives in the checkout, so derive the project root from its path
# rather than cloning again.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Make freshly installed user binaries (agent, herdr, voice-harness) visible to
# the rest of this run.
export PATH="$HOME/.local/bin:$PATH"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
step() { printf '\n\033[1;34m==>\033[0m \033[1m%s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '\033[1;33m warn:\033[0m %s\n' "$*" >&2; }
have() { command -v "$1" >/dev/null 2>&1; }

select_provider() {
  local label="$1" selected="$2"
  while true; do
    if [[ -z "$selected" ]]; then
      if [[ -t 0 ]]; then
        read -r -p "$label provider (local/venice) [local]: " selected
      fi
      selected="${selected:-local}"
    fi
    selected="${selected,,}"
    case "$selected" in
      local|venice)
        printf '%s\n' "$selected"
        return
        ;;
      *)
        warn "$label provider must be 'local' or 'venice'."
        if [[ ! -t 0 ]]; then
          return 1
        fi
        selected=""
        ;;
    esac
  done
}

require() {
  local missing=0 c
  for c in "$@"; do
    if ! have "$c"; then
      warn "required command not found: $c"
      missing=1
    fi
  done
  return "$missing"
}

installer_python() {
  if have python3; then
    printf '%s\n' python3
  elif have python; then
    printf '%s\n' python
  else
    warn "python3 is required to resolve installation profiles."
    return 1
  fi
}

resolve_install_plan() {
  local python
  python="$(installer_python)" || return 1
  PYTHONPATH="$PROJECT_DIR/src" "$python" -m local_voice_harness.install_profile "$@"
}

cuda_capability_args() {
  if have nvidia-smi && timeout 10 nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' "--cuda-available"
  fi
}

resolve_distro_plan() {
  local python
  python="$(installer_python)" || return 1
  PYTHONPATH="$PROJECT_DIR/src" "$python" -m local_voice_harness.install_distro "$@"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      if [[ $# -lt 2 ]]; then
        warn "--profile requires showcase or local-cuda."
        exit 1
      fi
      PROFILE="$2"
      shift 2
      ;;
    --profile=*)
      PROFILE="${1#*=}"
      shift
      ;;
    *)
      warn "unknown installer argument: $1"
      exit 1
      ;;
  esac
done

cd "$PROJECT_DIR"

bold "Local Voice Agent Harness installer"
info "Project directory: $PROJECT_DIR"

step "Selecting AI providers"
if [[ "${PROFILE,,}" == "showcase" ]]; then
  LLM_PROVIDER="venice"
  TTS_PROVIDER="venice"
  info "Showcase profile selected: Venice LLM/TTS with CPU dictation"
else
  LLM_PROVIDER="$(select_provider "LLM" "${LLM_PROVIDER:-}")"
  TTS_PROVIDER="$(select_provider "TTS" "${TTS_PROVIDER:-}")"
fi
mapfile -t CUDA_CAPABILITY_ARGS < <(cuda_capability_args)
eval "$(resolve_install_plan --format env --profile "$PROFILE" --llm "$LLM_PROVIDER" --tts "$TTS_PROVIDER" --llm-device "${LLM_DEVICE:-}" --tts-device "${TTS_DEVICE:-}" --dictation-device "${DICTATION_DEVICE:-}" "${CUDA_CAPABILITY_ARGS[@]}")"
eval "$(resolve_distro_plan --format env --checkout "$PROJECT_DIR" --os-release "${OS_RELEASE_PATH:-/etc/os-release}" --profile "$PROFILE" --llm "$LLM_PROVIDER" --tts "$TTS_PROVIDER" --llm-device "${LLM_DEVICE:-}" --tts-device "${TTS_DEVICE:-}" --dictation-device "${DICTATION_DEVICE:-}" --chatterbox-dir "$CHATTERBOX_DIR" "${CUDA_CAPABILITY_ARGS[@]}")"
LLM_PROVIDER="$INSTALL_LLM_PROVIDER"
TTS_PROVIDER="$INSTALL_TTS_PROVIDER"
CHATTERBOX_DIR="$INSTALL_CHATTERBOX_DIR"
info "Installation profile: $INSTALL_PROFILE"
info "Distro family: $INSTALL_DISTRO_FAMILY ($INSTALL_PACKAGE_MANAGER)"
info "LLM provider: $LLM_PROVIDER"
info "TTS provider: $TTS_PROVIDER"
info "Dictation extra: $INSTALL_DICTATION_EXTRA ($INSTALL_DICTATION_DEVICE)"
info "Checkout: $INSTALL_CHECKOUT"
info "User services: $INSTALL_SYSTEMD_USER_DIR"
if [[ "$INSTALL_CHECKOUT" != "$HOME/local-voice-harness" && ! -e "$HOME/local-voice-harness" ]]; then
  ln -sfn "$INSTALL_CHECKOUT" "$HOME/local-voice-harness"
  info "Linked $HOME/local-voice-harness -> $INSTALL_CHECKOUT for shipped user units"
fi

# --- 1. System packages -----------------------------------------------------
install_distro_packages() {
  local command="$1"
  shift
  if [[ $# -eq 0 ]]; then
    return 0
  fi
  # shellcheck disable=SC2086
  $command "$@"
}

if [[ "${SKIP_SYSTEM_PACKAGES:-0}" == 1 ]]; then
  step "Skipping system packages (SKIP_SYSTEM_PACKAGES=1)"
else
  PACKAGE_COMMAND="$INSTALL_PACKAGE_COMMAND"
  if [[ "$INSTALL_DISTRO_FAMILY" == "arch" ]] && ! have paru; then
    if have pacman; then
      PACKAGE_COMMAND="sudo pacman -S --needed --noconfirm"
    else
      warn "paru or pacman is required on Arch."
      exit 1
    fi
  fi
  if [[ "$INSTALL_DISTRO_FAMILY" == "debian" ]] && ! have apt-get; then
    warn "apt-get is required on Ubuntu/Debian."
    exit 1
  fi
  if [[ "$INSTALL_DISTRO_FAMILY" == "fedora" ]] && ! have dnf; then
    warn "dnf is required on Fedora."
    exit 1
  fi
  step "Installing base system packages ($INSTALL_PACKAGE_MANAGER)"
  # shellcheck disable=SC2086
  install_distro_packages $PACKAGE_COMMAND $INSTALL_DISTRO_PACKAGES
  if [[ -n "${INSTALL_DISTRO_CUDA_PACKAGES}" ]]; then
    step "Installing CUDA packages"
    # shellcheck disable=SC2086
    install_distro_packages $PACKAGE_COMMAND $INSTALL_DISTRO_CUDA_PACKAGES
  else
    info "Skipping CUDA and NVIDIA packages ($INSTALL_PROFILE profile)"
  fi
  if [[ -n "${INSTALL_SKIPPED_PACKAGES}" ]]; then
    info "Distro packages not mapped and left for manual install: $INSTALL_SKIPPED_PACKAGES"
  fi
  if [[ "$INSTALL_UV_BOOTSTRAP" == 1 ]] && ! have uv; then
    step "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

require uv git curl || {
  warn "Install the missing commands above, then re-run."
  exit 1
}

# --- 2. Management / wake environment ---------------------------------------
step "Syncing management/wake environment (.venv)"
"$PROJECT_DIR/scripts/sync-wake.sh"
mkdir -p "$INSTALL_USER_BIN"
ln -sfn "$PROJECT_DIR/.venv/bin/voice-harness" "$INSTALL_VOICE_HARNESS"
info "Linked voice-harness into $INSTALL_USER_BIN"
case ":$PATH:" in
  *":$INSTALL_USER_BIN:"*) : ;;
  *) warn "Add $INSTALL_USER_BIN to your PATH to use the voice-harness command." ;;
esac

step "Writing backend configuration"
if [[ "$INSTALL_PROFILE" == "showcase" ]]; then
  voice-harness setup --profile showcase
else
  mkdir -p "$INSTALL_CONFIG_DIR"
  cat >"$INSTALL_CONFIG_DIR/backends.toml" <<EOF
[llm]
provider = "$LLM_PROVIDER"

[tts]
provider = "$TTS_PROVIDER"
EOF
  voice-harness config set compute.llm_device "$INSTALL_LLM_DEVICE"
  voice-harness config set compute.tts_device "$INSTALL_TTS_DEVICE"
  voice-harness config set compute.dictation_device "$INSTALL_DICTATION_DEVICE"
fi

if [[ "$LLM_PROVIDER" == venice || "$TTS_PROVIDER" == venice ]]; then
  step "Configuring Venice credentials"
  if voice-harness credentials status >/dev/null 2>&1; then
    info "Venice API key is already stored"
  else
    voice-harness credentials set
  fi
fi

# --- 3. Dictation environment (Parakeet by default) -------------------------
step "Syncing bundled dictation environment (.venv-dictation)"
UV_PROJECT_ENVIRONMENT=.venv-dictation \
  uv sync --python "$PYTHON_VERSION" --extra "$INSTALL_DICTATION_EXTRA" --no-dev

# --- 4. TTS environment ------------------------------------------------------
if [[ "$INSTALL_TTS_EXTRA" == 1 ]]; then
  step "Syncing local TTS environment"
  mkdir -p "$CHATTERBOX_DIR"
  UV_PROJECT_ENVIRONMENT="$CHATTERBOX_DIR/.venv" \
    uv sync --python "$PYTHON_VERSION" --extra tts --no-dev
else
  step "Skipping local TTS extra and Chatterbox environment ($INSTALL_PROFILE profile)"
  mkdir -p "$CHATTERBOX_DIR"
  UV_PROJECT_ENVIRONMENT="$CHATTERBOX_DIR/.venv" \
    uv sync --python "$PYTHON_VERSION" --no-dev
fi

# --- 5. Models --------------------------------------------------------------
if [[ "${SKIP_MODELS:-0}" == 1 ]]; then
  step "Skipping model downloads (SKIP_MODELS=1)"
else
  if [[ "$INSTALL_DOWNLOAD_CHATTERBOX" == 1 ]]; then
    step "Caching Chatterbox Turbo weights"
    TTS_CACHE_DEVICE="$INSTALL_TTS_DEVICE"
    HF_HUB_OFFLINE=0 TTS_CACHE_DEVICE="$TTS_CACHE_DEVICE" \
      "$CHATTERBOX_DIR/.venv/bin/python" - <<'PY'
import os
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
requested = os.environ["TTS_CACHE_DEVICE"]
device = (
    "cuda" if requested == "auto" and torch.cuda.is_available()
    else "cpu" if requested == "auto"
    else requested
)
ChatterboxTurboTTS.from_pretrained(device=device)
print("Chatterbox Turbo cached")
PY
  else
    info "Skipping Chatterbox weights (local TTS extra omitted)"
  fi

  if [[ "$INSTALL_DOWNLOAD_QWEN" == 1 ]]; then
    step "Downloading Qwen GGUF"
    mkdir -p "$PROJECT_DIR/models"
    if [[ -f "$PROJECT_DIR/models/$QWEN_FILE" ]]; then
      info "Already present: models/$QWEN_FILE"
    else
      if ! have hf; then
        uv tool install huggingface_hub
      fi
      hf download "$QWEN_REPO" "$QWEN_FILE" --local-dir "$PROJECT_DIR/models"
    fi
  else
    info "Skipping Qwen download (local LLM omitted)"
  fi
fi

# --- 6. Cursor CLI and Herdr ------------------------------------------------
if have agent; then
  info "Cursor CLI already installed ($(agent --version 2>/dev/null | head -n1))"
else
  step "Installing Cursor CLI"
  curl https://cursor.com/install -fsS | bash
fi

if have herdr; then
  info "Herdr already installed ($(herdr --version 2>/dev/null | head -n1))"
else
  step "Installing Herdr"
  curl -fsSL https://herdr.dev/install.sh | sh
fi

# --- 7. Interactive authentication (only when not already logged in) --------
if [[ "${SKIP_AUTH:-0}" == 1 ]]; then
  step "Skipping interactive logins (SKIP_AUTH=1)"
else
  step "Authentication"

  if have agent; then
    if agent status >/dev/null 2>&1; then
      info "Cursor CLI already authenticated"
    else
      info "Logging in to Cursor (interactive)"
      agent login
    fi
  else
    warn "Cursor CLI unavailable; skipping agent login."
  fi

  if have gh; then
    if gh auth status >/dev/null 2>&1; then
      info "GitHub CLI already authenticated"
    else
      info "Logging in to GitHub (interactive)"
      gh auth login
    fi
  else
    warn "gh unavailable; skipping GitHub login."
  fi

  # Linear is optional; ordinary repository tasks work without it.
  if have agent; then
    if agent mcp list 2>/dev/null |
      grep -Eiq '^[[:space:]]*linear:[[:space:]]*(ready|connected)[[:space:]]*$'; then
      info "Linear MCP already configured"
    else
      info "Configuring Linear MCP (interactive; optional)"
      if agent mcp login linear; then
        agent mcp enable linear || warn "Could not enable Linear MCP."
      else
        warn "Linear MCP login skipped/failed; continuing without it."
      fi
    fi
  fi
fi

# --- 8. Install and start services ------------------------------------------
step "Installing and starting systemd user services"
if have voice-harness; then
  voice-harness services install
  voice-harness services audit
  voice-harness services start
else
  warn "voice-harness not on PATH; run the following once ~/.local/bin is on PATH:"
  warn "  voice-harness services install && voice-harness services audit && voice-harness services start"
fi

# --- Done -------------------------------------------------------------------
step "Install complete"
bold "Remaining hardware-specific steps (see README):"
info "1. Set your PipeWire mic source:"
info "     voice-harness config set audio.source <PIPEWIRE_SOURCE_NAME>  # find with: wpctl status"
if [[ "$INSTALL_LLM_SERVICE" == 1 ]]; then
  info "2. If your NVIDIA device is not CUDA0:"
  info "     voice-harness config set compute.cuda_device <LLAMA_CPP_DEVICE>"
  info "3. Verify: voice-harness doctor && voice-harness services audit"
else
  info "2. Store a Venice API key if needed: voice-harness credentials set"
  info "3. Verify: voice-harness doctor && voice-harness services audit"
fi
