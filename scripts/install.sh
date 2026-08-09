#!/usr/bin/env bash
# One-shot installer for the Local Voice Agent Harness on Arch/CachyOS.
#
# Runs every deterministic setup step (system packages, uv environments, model
# downloads, service install) and only pauses for interactive logins that are
# not already authenticated. Safe to re-run: each step is idempotent.
#
# Overridable via environment:
#   PYTHON_VERSION      Python for the harness envs (default 3.11)
#   CHATTERBOX_DIR      TTS env location (default $HOME/chatterbox-audition)
#   LLM_PROVIDER        LLM backend: local or venice (prompts when unset)
#   TTS_PROVIDER        TTS backend: local or venice (prompts when unset)
#   SKIP_SYSTEM_PACKAGES=1   Skip the paru package steps
#   SKIP_MODELS=1            Skip Qwen/Chatterbox downloads
#   SKIP_AUTH=1              Skip the interactive gh/cursor/linear logins
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CHATTERBOX_DIR="${CHATTERBOX_DIR:-$HOME/chatterbox-audition}"
QWEN_REPO="unsloth/Qwen3.5-4B-GGUF"
QWEN_FILE="Qwen3.5-4B-Q4_K_M.gguf"

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

cd "$PROJECT_DIR"

bold "Local Voice Agent Harness installer"
info "Project directory: $PROJECT_DIR"
if [[ "$PROJECT_DIR" != "$HOME/local-voice-harness" ]]; then
  warn "Shipped systemd units assume \$HOME/local-voice-harness."
  warn "Running from $PROJECT_DIR; adjust units or clone there if services fail."
fi

step "Selecting AI providers"
LLM_PROVIDER="$(select_provider "LLM" "${LLM_PROVIDER:-}")"
TTS_PROVIDER="$(select_provider "TTS" "${TTS_PROVIDER:-}")"
info "LLM provider: $LLM_PROVIDER"
info "TTS provider: $TTS_PROVIDER"

# --- 1. System packages -----------------------------------------------------
if [[ "${SKIP_SYSTEM_PACKAGES:-0}" == 1 ]]; then
  step "Skipping system packages (SKIP_SYSTEM_PACKAGES=1)"
elif have paru; then
  step "Installing base system packages"
  paru -S --needed pipewire libnotify git curl github-cli xdotool xclip \
    wl-clipboard wtype uv libsndfile ffmpeg
  if [[ "$LLM_PROVIDER" == venice || "$TTS_PROVIDER" == venice ]]; then
    step "Installing Venice credential support"
    paru -S --needed libsecret oo7
  fi

  step "Installing CUDA and CUDA-enabled llama.cpp"
  paru -S --needed cuda llama.cpp-cuda
else
  warn "paru not found; skipping system packages."
  warn "Install prerequisites manually (see README) or set SKIP_SYSTEM_PACKAGES=1."
fi

require uv git curl || {
  warn "Install the missing commands above, then re-run."
  exit 1
}

# --- 2. Management / wake environment ---------------------------------------
step "Syncing management/wake environment (.venv)"
uv sync --python "$PYTHON_VERSION" --extra wake --no-dev
mkdir -p "$HOME/.local/bin"
ln -sfn "$PROJECT_DIR/.venv/bin/voice-harness" "$HOME/.local/bin/voice-harness"
info "Linked voice-harness into ~/.local/bin"
case ":$PATH:" in
  *":$HOME/.local/bin:"*) : ;;
  *) warn "Add \$HOME/.local/bin to your PATH to use the voice-harness command." ;;
esac

step "Writing backend configuration"
mkdir -p "$HOME/.config/voice-harness"
cat >"$HOME/.config/voice-harness/backends.toml" <<EOF
[llm]
provider = "$LLM_PROVIDER"

[tts]
provider = "$TTS_PROVIDER"
EOF

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
  uv sync --python "$PYTHON_VERSION" --extra dictation --no-dev

# --- 4. TTS environment ------------------------------------------------------
step "Syncing TTS environment"
mkdir -p "$CHATTERBOX_DIR"
if [[ "$TTS_PROVIDER" == local ]]; then
  UV_PROJECT_ENVIRONMENT="$CHATTERBOX_DIR/.venv" \
    uv sync --python "$PYTHON_VERSION" --extra tts --no-dev
else
  UV_PROJECT_ENVIRONMENT="$CHATTERBOX_DIR/.venv" \
    uv sync --python "$PYTHON_VERSION" --no-dev
fi

# --- 5. Models --------------------------------------------------------------
if [[ "${SKIP_MODELS:-0}" == 1 ]]; then
  step "Skipping model downloads (SKIP_MODELS=1)"
else
  if [[ "$TTS_PROVIDER" == local ]]; then
    step "Caching Chatterbox Turbo weights"
    HF_HUB_OFFLINE=0 "$CHATTERBOX_DIR/.venv/bin/python" - <<'PY'
from chatterbox.tts_turbo import ChatterboxTurboTTS
ChatterboxTurboTTS.from_pretrained(device="cuda")
print("Chatterbox Turbo cached")
PY
  else
    info "Skipping Chatterbox weights (Venice TTS selected)"
  fi

  if [[ "$LLM_PROVIDER" == local ]]; then
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
    info "Skipping Qwen download (Venice LLM selected)"
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
info "     systemctl --user edit voice-harness-wake.service"
info "     [Service]"
info "     Environment=VOICE_HARNESS_SOURCE=<PIPEWIRE_SOURCE_NAME>   # find with: wpctl status"
info "2. If your NVIDIA device is not CUDA0, edit systemd/user/voice-harness-llm.service."
info "3. Verify: voice-harness status && voice-harness services audit"
