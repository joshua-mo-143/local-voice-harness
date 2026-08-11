# Reproducible voice replay

Replay bundles capture bounded semantic inputs and recorded nondeterministic
decisions. They omit audio, provider bodies, and complete model payloads by
default. Running a replay does not start services, contact providers, submit
jobs, mutate workspaces, or invoke TTS.

Capture a turn from a raw transcript:

```fish
voice-harness replay capture please work on issues 12 and 18
```

Capture can inspect the currently focused context and call the configured intent
router. Use `--without-context` to record an explicit empty context decision.
Use `--intent` together with `--confidence` to inject the router decision from
an observed failure instead of calling the current router again.
Use `--spoken-response` and optionally `--display-response` to preserve an
already-observed channel-aware response without making another model request.

Bundles are written with owner-only permissions under the harness state
directory unless `--output` is supplied. Inspect or verify one with:

```fish
voice-harness replay inspect /path/to/replay.json
voice-harness replay run /path/to/replay.json
```

`run` recomputes transcript normalization and ticket extraction, then injects
the recorded context and routing decisions. Missing stages, unsupported schema
versions, mismatched deterministic outputs, unknown fields, oversized values,
and credential material fail explicitly.

Export requires reviewing the manifest summary and typing `export`:

```fish
voice-harness replay export /path/to/replay.json /path/to/shared.json
```

To turn a bundle into a regression fixture, use `promote`; it prints the complete
bounded JSON for review and requires typing `reviewed`:

```fish
voice-harness replay promote /path/to/replay.json tests/fixtures/replay/case.json
```

Never promote a fixture merely because it passes automated credential checks.
Manual review is required because sensitive business or personal content may
not resemble a credential.
