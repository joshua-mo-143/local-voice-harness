# Development

## Run the current checkout

The repository-local launcher runs the branch through `uv` while redirecting its
configuration and durable state to ignored paths in the checkout:

```bash
scripts/dev.sh text "What is two plus two?"
scripts/dev.sh pronounce "PR #128 changed src/http_client.py"
scripts/dev.sh text "Is the voice harness healthy?"
scripts/dev.sh setup --defaults
scripts/dev.sh config show audio.wake_threshold
scripts/dev.sh integrations list
scripts/dev.sh text "What voice are you using?"
scripts/dev.sh text "Is Linear enabled?"
```

The self-health request runs bounded, read-only diagnostics and returns only a
non-sensitive summary. This text-only smoke does not use the microphone and does
not require stopping the installed wake listener.

The launcher sets `XDG_CONFIG_HOME` to `.dev/config`, `XDG_STATE_HOME` to
`.dev/state`, `UV_PROJECT_ENVIRONMENT` to `.dev/venv`, and
`VOICE_HARNESS_BRANCH_RUNTIME` to `.dev/runtime`. For example,
branch-specific backend configuration belongs at
`.dev/config/voice-harness/backends.toml`. When that checkout has no
`config.toml` or `backends.toml` yet, the launcher copies those files from the
installed profile so the branch starts from production defaults; later
checkout-local edits stay isolated. Durable jobs are stored below
`.dev/state/voice-harness/`, worker logs and other branch-owned transients live
under `.dev/runtime/`, and launcher dependencies stay in `.dev/venv`.
The checkout-local jobs database is explicitly
`<checkout>/.dev/state/voice-harness/jobs/jobs.sqlite3`.
All launcher commands in one checkout share that database, while launchers in
different checkout or worktree roots use distinct databases, logs, and worker
environments. The launcher also
removes an inherited systemd `STATE_DIRECTORY` before starting Python, so a shell
carrying the installed service environment cannot override `.dev/state` or select
the production jobs database.
Existing `VOICE_HARNESS_*` values are inherited, so temporary overrides remain
available:

```bash
VOICE_HARNESS_LLM_PROVIDER=venice scripts/dev.sh text "Summarize my open work"
```

The primary checkout's `.venv` belongs to the installed wake service. Development
launcher commands never sync or execute that environment; they select Python 3.11
and the `wake` extra in `.dev/venv` instead. Git worktrees already have distinct
checkout-local `.venv` directories, and each worktree launcher likewise uses its own
ignored `.dev/venv`. The foreground `wake` command restores its launcher-local model
only when absent. Installed STT, TTS, and LLM services remain shared.

Only explicit installation and wake synchronization may mutate the installed wake
environment. Run `scripts/sync-wake.sh` to sync its dependencies and restore the
required OpenWakeWord model if it is missing. Existing model files make the restore
step a no-op.

GitHub CLI authentication remains shared with the normal user profile. Before
isolating `XDG_CONFIG_HOME`, the launcher sets `GH_CONFIG_DIR` to the directory
that `gh` would normally use (`$XDG_CONFIG_HOME/gh` or `$HOME/.config/gh`).
An existing `GH_CONFIG_DIR` override is preserved. Run `gh auth login` outside
the launcher if that profile is not authenticated.

To run the checkout's wake daemon in the foreground:

```bash
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
configuration helpers (`setup`, `config`, and `integrations`) and the side-effect-free
`pronounce` preview, so it does not expose service or credential management.

### Conversational configuration-change smoke

This smoke uses the microphone. Before starting it, tell the user that the foreground
wake listener will capture audio and pause until they explicitly confirm they are
ready. Do not infer readiness from an earlier message.

After acknowledgement, record the current branch-local value, stop the installed
listener, and start the checkout listener:

```fish
set original_voice (scripts/dev.sh config show audio.voice)
echo $original_voice
systemctl --user stop voice-harness-wake.service
scripts/dev.sh wake
```

Say “Set the voice to issue_273_smoke.” Verify that the assistant reads back
`audio.voice`, the exact old value, and `issue_273_smoke`, and asks for yes or no without
writing. Say “no” and verify the value is unchanged. Repeat the request, say “yes”,
and verify that it reports the running snapshot is unchanged and names
`voice-harness-wake.service` as requiring a manual restart. In another terminal,
confirm the persisted branch-local value:

```fish
scripts/dev.sh config show audio.voice
```

For a stale-confirmation check, request another voice, change `audio.voice` through
the branch-local `config set` command before answering, then say “yes.” The wake
conversation must reject the stale confirmation and preserve the intervening value.
Press Ctrl-C, restore the original branch-local value if needed, and restart the
installed listener:

```fish
scripts/dev.sh config set audio.voice "$original_voice"
systemctl --user start voice-harness-wake.service
```

### Controlled activation recovery smoke

This smoke restarts the installed wake service and captures microphone audio. Do not
run it as part of automated verification. Explain that the installed wake listener
will be restarted, identify the branch/worktree under test, and pause until the user
explicitly acknowledges readiness.

The foreground `scripts/dev.sh wake` workflow cannot exercise this path because its
safety check requires the installed wake unit to be inactive. Use the controlled
service-double tests by default. A real smoke requires a disposable test user/session
whose already-installed `voice-harness-wake.service` has been independently reviewed
and confirmed to execute this checkout with the intended configuration profile. The
activation path itself never installs, replaces, enables, disables, or removes units.

After acknowledgement, use an allow-listed change that differs from the current
value. Confirm the configuration write with “yes.” Verify that the saved-change
response is delivered before any restart and asks for the distinct phrase
“activate now.” A generic “yes” must not activate it. Say “activate now” and verify
that the complete pre-restart response is delivered before the wake process exits.
After systemd starts the listener again, verify that the durable completion response
is spoken/displayed only after the service is active with the expected immutable
configuration snapshot.

For recovery testing, repeat with one controlled interruption at a time:

- stop the branch wake process before saying “activate now”; no restart request should
  exist;
- terminate it after activation acceptance but before pre-restart delivery; the next
  branch wake process must replay that delivery before dispatch;
- terminate the isolated activation worker while restart state is durable; recovery
  must reconcile service identity and snapshot state without blindly restarting;
- terminate wake after restart observation but before completion acknowledgement; the
  next process must redeliver the stored result without another restart.

Restore the original configuration after the smoke. Failed or partial outcomes must
remain visible in the completion response; do not retry them automatically.

This is branch testing isolation, not a complete runtime profile:

- Isolated per worktree: configuration (`.dev/config`), durable state
  (`.dev/state`), branch-owned transients and worker logs (`.dev/runtime`),
  Cursor job records, spawned worker `TMPDIR`, and the launcher virtualenv
  (`.dev/venv`). Concurrent `scripts/dev.sh text` commands from separate
  worktrees execute that checkout's code and do not read or modify another
  worktree's `.dev/` files.
- Shared: `XDG_RUNTIME_DIR` is preserved for installed STT, TTS, and LLM
  sockets, recording paths, the wake lock, and other microphone-owned runtime.
  Those services still use their installed configuration and endpoints.
- Wake and audio settings are loaded when the process starts. Hot reload is out
  of scope; restart the foreground process after changing an override.
- Only one wake listener should own the microphone and shared wake runtime.
  `scripts/dev.sh wake` still refuses to start when
  `voice-harness-wake.service` is active, without changing systemd state. Stop
  the installed wake unit yourself before using the foreground listener, then
  restore it afterward.
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
set -gx UV_PROJECT_ENVIRONMENT ".dev/quality-$python_version"
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
