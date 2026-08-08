# Development

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
