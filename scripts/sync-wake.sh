#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
WAKE_ENVIRONMENT="$PROJECT_DIR/.venv"

export UV_PROJECT_ENVIRONMENT="$WAKE_ENVIRONMENT"
uv sync \
  --project "$PROJECT_DIR" \
  --python "$PYTHON_VERSION" \
  --extra wake \
  --no-dev
"$WAKE_ENVIRONMENT/bin/python" -m local_voice_harness.wake.models
