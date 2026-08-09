## Security notes

- Voice transcription can be wrong. Review all Cursor changes before committing.
- Herdr agents are started with workspace trust but not Cursor `--force`.
- Ticket and MCP content is treated as untrusted input and inferred paths are
  validated against local Git repositories.
- Focused GitHub issue and pull request content is read through `gh`, and rendered
  Zendesk ticket content is copied from the browser session; all are bounded before
  prompting and treated as untrusted external data.
- Optional context integrations are disabled by default on fresh installations.
  Zendesk is off unless explicitly enabled through `[integrations] zendesk` in
  `config.toml` or `VOICE_HARNESS_INTEGRATION_ZENDESK`; while disabled, Zendesk
  URLs are never inspected and no page text is copied. A misconfigured or
  unreadable configuration fails closed to the disabled defaults, and an
  individual provider failing never breaks an ordinary voice request.
- Repository cloning requires explicit Rofi confirmation, accepts only HTTPS or SSH
  Git URLs, and places the checkout beneath the configured project root.
- Merely focusing a GitHub page cannot create a fork. The original spoken request must
  unambiguously ask for one, and the user must separately confirm before the validated
  public repository is forked.
- Checking out a focused pull request clones or reuses its repository below the GitHub
  root and runs `gh pr checkout` only in a job-unique, reserved worktree. Recovery
  retries that same worktree; failed preparation quarantines it.
- Jobs never automatically commit, push, open pull requests, or remove worktrees.
- Completed-job follow-ups reuse only the parent's exact, verified retained checkout:
  the inherited repository, branch, worktree path, workspace, and pane are immutable;
  the path is confirmed to be an isolated worktree (never the shared clone) that still
  matches Herdr before dispatch; and unresolved live or quarantined evidence blocks
  reuse. Agent startup uses a durably reserved target and retained pane, recovered
  agents must report the exact checkout, and a prompt timeout keeps its target fenced
  until cancellation. The completed parent is never reopened or mutated, prior agent
  output is never injected into routing or worker prompts, opening a pull request
  stays unsupported, and the in-memory follow-up reference is discarded on restart. Set
  `VOICE_HARNESS_CURSOR_FOLLOWUP=0` to disable the feature.
- Intent routing uses the configured LLM backend, but only a high-confidence
  authoritative route can invoke Cursor. Conversation and low-confidence fallback are
  tool-free.
- Runtime job metadata and conversational audio live under
  `$XDG_RUNTIME_DIR/voice-harness`; focused dictation audio lives under
  `$XDG_RUNTIME_DIR/dictation`.
- The unauthenticated llama.cpp API is bound to `127.0.0.1` for this trusted
  single-user workstation. Same-account processes are trusted; loopback is not a
  per-UID boundary on a mutually untrusted multi-user host.
- Shipped services use service-specific systemd hardening and bounded resources; see
  the [hardening policy](docs/service-hardening.md) for deliberate exceptions and
  host checks.


