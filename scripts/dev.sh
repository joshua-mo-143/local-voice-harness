#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/dev.sh text <request>
  scripts/dev.sh wake
  scripts/dev.sh setup [--defaults]
  scripts/dev.sh config <show|set|reset> ...
  scripts/dev.sh integrations <list|enable|disable|doctor> ...
EOF
}

command="${1:-}"
case "$command" in
  text)
    shift
    if (($# == 0)); then
      usage
      exit 2
    fi
    ;;
  wake)
    shift
    if (($# != 0)); then
      usage
      exit 2
    fi
    ;;
  setup|config|integrations)
    shift
    ;;
  *)
    usage
    exit 2
    ;;
esac

if [[ -z "${GH_CONFIG_DIR:-}" ]]; then
  export GH_CONFIG_DIR="${XDG_CONFIG_HOME:-"$HOME/.config"}/gh"
fi
export XDG_CONFIG_HOME="$PROJECT_DIR/.dev/config"
export XDG_STATE_HOME="$PROJECT_DIR/.dev/state"
mkdir -p -- "$XDG_CONFIG_HOME" "$XDG_STATE_HOME"

if [[ "$command" == "text" ]]; then
  exec uv run --project "$PROJECT_DIR" voice-harness text "$@"
fi

if [[ "$command" == "setup" || "$command" == "config" || "$command" == "integrations" ]]; then
  exec uv run --project "$PROJECT_DIR" voice-harness "$command" "$@"
fi

if systemctl --user is-active --quiet voice-harness-wake.service; then
  cat >&2 <<'EOF'
scripts/dev.sh: voice-harness-wake.service is active.
Stop it manually before running the development wake listener:
  systemctl --user stop voice-harness-wake.service
Restore the installed listener after development:
  systemctl --user start voice-harness-wake.service
No services were changed.
EOF
  exit 1
else
  systemctl_status=$?
fi

if ((systemctl_status != 3 && systemctl_status != 4)); then
  cat >&2 <<'EOF'
scripts/dev.sh: unable to determine whether voice-harness-wake.service is active.
Check the systemd user session and retry. No services were changed.
EOF
  exit 1
fi

exec uv run --project "$PROJECT_DIR" --extra wake voice-harness-wake
