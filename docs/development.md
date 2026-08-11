# Development

## Run the current checkout

The repository-local launcher runs the branch through `uv` while redirecting its
configuration and durable state to ignored paths in the checkout:

```fish
scripts/dev.sh text "What is two plus two?"
scripts/dev.sh setup --defaults
scripts/dev.sh config show audio.wake_threshold
scripts/dev.sh integrations list
```

The launcher sets `XDG_CONFIG_HOME` to `.dev/config` and `XDG_STATE_HOME` to
`.dev/state`. For example, branch-specific backend configuration belongs at
`.dev/config/voice-harness/backends.toml`, and durable jobs are stored below
`.dev/state/voice-harness/`. Existing `VOICE_HARNESS_*` values are inherited, so
temporary overrides remain available:

```fish
env VOICE_HARNESS_LLM_PROVIDER=venice scripts/dev.sh text "Summarize my open work"
```

GitHub CLI authentication remains shared with the normal user profile. Before
isolating `XDG_CONFIG_HOME`, the launcher sets `GH_CONFIG_DIR` to the directory
that `gh` would normally use (`$XDG_CONFIG_HOME/gh` or `$HOME/.config/gh`).
An existing `GH_CONFIG_DIR` override is preserved. Run `gh auth login` outside
the launcher if that profile is not authenticated.

To run the checkout's wake daemon in the foreground:

```fish
systemctl --user stop voice-harness-wake.service
scripts/dev.sh wake
# Press Ctrl-C when finished.
systemctl --user start voice-harness-wake.service
```

Before starting, `scripts/dev.sh wake` checks
`voice-harness-wake.service`. It refuses to run when that unit is active and prints
the stop and restore commands; it also refuses to run if the service state cannot
be checked safely. The launcher never stops, restarts, edits, or installs a service.
Its command surface is intentionally limited to `text`, `wake`, and the branch-local
configuration helpers (`setup`, `config`, and `integrations`), so it does not expose
service or credential management.

This is branch testing isolation, not a complete runtime profile:

- `XDG_RUNTIME_DIR` is preserved. Runtime sockets, recording paths, locks, and
  other transient resources remain shared.
- The installed STT, TTS, and LLM services and sockets are reused, and those
  services still use their installed configuration. This workflow is mainly for
  application, routing, integration, and wake-daemon development.
- Wake and audio settings are loaded when the process starts. Hot reload is out
  of scope; restart the foreground process after changing an override.
- Only one wake listener should own the microphone and shared runtime. Stop the
  installed wake unit yourself before using the foreground listener, then restore
  it afterward.
- Desktop Secret Service credentials and external integrations are not isolated.
  The launcher can read credentials as the application normally does, but offers
  no command that changes them.
- A full profile manager, parallel systemd units, socket namespacing, dynamic
  ports, service cloning, and profile CRUD are explicitly out of scope.

## Quality checks

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

The [service hardening and local trust policy](service-hardening.md) records each
unit's retained capabilities, resource-limit rationale, unauthenticated loopback LLM
policy, residual risks, and the required real-host smoke checklist.

GPU, audio, Herdr, and complete voice-turn checks are opt-in only. See the
[hardware smoke checklist](hardware-smoke.md) for their entry points and cleanup
steps; CI never runs them.

## Performance observed

Measured with all models warm on the RTX 5070 Ti Laptop GPU. Dictation figures use
the earlier Whisper large-v3 backend; Parakeet TDT 0.6B v2 measurements are pending:

- Whisper large-v3: approximately 0.58 seconds for a short request.
- Qwen response: 0.22–0.53 seconds; first CUDA request TTFT pending re-measurement
  (Vulkan baseline was approximately 5 seconds).
- Chatterbox: 0.53 seconds for 2.72 seconds of audio; longer replies now begin
  playing after their first sentence/clause is ready.
- Cursor delegation: task-dependent and normally handled as a background job.
