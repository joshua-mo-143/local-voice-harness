## Security notes

- Voice transcription can be wrong. Review all Cursor changes before committing.
- Herdr agents are started with workspace trust but not Cursor `--force`.
- Ticket and MCP content is treated as untrusted input and inferred paths are
  validated against local Git repositories.
- Focused GitHub issue and pull request content is read through `gh`, and rendered
  Zendesk ticket content is copied from the browser session; all are bounded before
  prompting and treated as untrusted external data.
- Repository cloning requires explicit Rofi confirmation, accepts only HTTPS or SSH
  Git URLs, and places the checkout beneath the configured project root.
- Merely focusing a GitHub page cannot create a fork. The original spoken request must
  unambiguously ask for one, and the user must separately confirm before the validated
  public repository is forked.
- Checking out a focused pull request clones or reuses its repository below the GitHub
  root and runs `gh pr checkout` only in a job-unique, reserved worktree. Recovery
  retries that same worktree; failed preparation quarantines it.
- Jobs never automatically commit, push, open pull requests, or remove worktrees.
- Runtime job metadata and conversational audio live under
  `$XDG_RUNTIME_DIR/voice-harness`; focused dictation audio lives under
  `$XDG_RUNTIME_DIR/dictation`.
- The unauthenticated llama.cpp API is bound to `127.0.0.1` for this trusted
  single-user workstation. Same-account processes are trusted; loopback is not a
  per-UID boundary on a mutually untrusted multi-user host.
- Shipped services use service-specific systemd hardening and bounded resources; see
  the [hardening policy](docs/service-hardening.md) for deliberate exceptions and
  host checks.


