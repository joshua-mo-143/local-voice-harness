# Architecture

```text
PipeWire microphone
  -> OpenWakeWord ("Hey Jarvis")
  -> Parakeet TDT 0.6B v2 via ONNX Runtime (CUDA)
  -> Qwen3.5-4B Q4_K_M via llama.cpp (CUDA)
       -> focused intent classification
       -> ordinary conversational response
       -> Herdr-managed Cursor agent, GitHub CLI, and Linear MCP
  -> Chatterbox Turbo (CUDA)
  -> PipeWire playback
```

The always-on wake daemon verifies OpenWakeWord candidates with the configured
dictation backend to reject false activations. A request that takes longer than five
seconds becomes a persisted background job, and its completion or clarification
question is spoken later. Job transitions are serialized across the daemon and
detached workers; abandoned jobs are recovered at daemon startup and during normal
polling. Spoken background results use at-least-once delivery: playback is
acknowledged only after it succeeds, so a crash at that boundary may repeat a result
but will not silently lose it.

Spoken responses use chunk-level streaming. Chatterbox still generates a complete
waveform for each short sentence or clause, but the next chunk is synthesized while
the current chunk is sent through one low-latency PipeWire playback stream. This is
not native sample streaming from the model. Playback sessions are serialized across
processes so manual commands and daemon announcements cannot overlap. Playback
therefore starts after the first sentence/clause instead of waiting for the whole
response. Chatterbox cannot cancel an active `generate()` call, so a wake-word
interruption may take up to one short chunk to take effect on the server side, while
PipeWire playback and already-queued chunks stop immediately.

## Cursor routing

Cursor routing works as follows:

1. Ask a focused Qwen pass to classify conversation, new work, clarification replies,
   status, and cancellation without rewriting the user's request.
2. Prefer an idle Cursor agent already running in the requested checkout.
3. For a Linear issue without a repository name, ask a dedicated routing agent to
   inspect the ticket through Linear MCP and infer the repository.
4. For a focused or explicitly spoken GitHub issue, validate it through `gh`, reuse
   an exact matching local checkout or clone its repository below the GitHub root,
   and preserve bounded issue context with the job.
5. If no repository can be resolved, open Rofi to select a local repository or paste
   a Git URL; cloning requires a second confirmation.
6. When the user unambiguously asks to fork, ask for a yes-or-no confirmation, then
   validate the focused public GitHub repository, create or reuse the authenticated
   user's fork, and clone it below the configured GitHub root.
7. When a GitHub pull request is focused, clone or reuse its repository below the
   configured GitHub root, create a job-unique `voice/github-pr-<job-id>` worktree,
   and run `gh pr checkout` only inside that reserved worktree.
8. Create or reuse a `voice/<issue-key>` worktree for Linear work, a stable
   `voice/github-issue-<number>` worktree for GitHub issue work, or a unique
   `voice/github-<job-id>` worktree for a GitHub fork task.
9. Start a new Cursor agent through Herdr when no suitable agent exists.
10. Reserve that agent and checkout until it finishes, is blocked, or is cancelled.

The harness never automatically commits, pushes, opens pull requests, modifies Linear,
or deletes generated worktrees. Fork creation is the only supported GitHub write and
is performed only after an unambiguous spoken request and a separate affirmative
confirmation. Checking out a focused pull request only reads from GitHub and writes to
its isolated local worktree. PR worktrees are reused only by recovery or continuation
of the same job. Completed and cancelled worktrees are retained for inspection, while
an invalid or partially prepared checkout is marked quarantined and is never dispatched.

## Runtime privacy and durability

Microphone recordings, recorder ownership files, logs, and service sockets are
transient session data under `$XDG_RUNTIME_DIR`. The bundled STT service accepts only
strictly named UUID generations beneath the two harness recording directories.
Stopping capture atomically moves the writable WAV to its immutable generation while
the recorder lock is still held; wake-mode recording performs the same handoff. A
later capture only replaces the writable path. After acquiring the model slot, STT
atomically moves that generation to a unique private processing path and removes only
the claimed file after the attempt. Cancellation removes writable audio after
recorder termination is confirmed. Recorder ownership includes the Linux process
start identity as well as its PID; it is not durable across login sessions.

The wake-service journal records user and assistant text, raw LLM request payloads,
aggregated responses, and tool-call arguments and results for diagnostics. These logs
can contain conversation, repository, and issue content and follow the system journal's
retention policy. Authorization headers and API keys are never included.

Only one in-process GPU transcription runs at a time. A second fully framed request
receives a structured `server_busy` error immediately instead of waiting behind a
possibly hung model call, without moving or deleting its retryable generation. The
client retries that same immutable generation with bounded backoff for an overall
120-second request window. If STT remains busy, the error prints a safe
`voice-harness transcribe --generation <path>` retry command and leaves the file in
place. The accepted call remains synchronous; Python cannot safely force-cancel a
hung native GPU call, so service supervision must restart the dictation process to
recover that case. Wake capture is suppressed without stopping the daemon while a
manual or focused-dictation recorder owns the shared recording lock. Manual and
focused-dictation starts inspect every configured recorder owner atomically under
that lock, so different capture modes cannot run concurrently.

Cursor job JSON, its lock, and quarantine evidence are durable under the absolute
`$STATE_DIRECTORY/jobs` supplied by systemd. Outside the service they use
`$XDG_STATE_HOME/voice-harness/jobs`, falling back to
`~/.local/state/voice-harness/jobs`. `STATE_DIRECTORY` is service-owned and must
not be set in user environment overrides. Detached worker logs remain private,
session-only files under `$XDG_RUNTIME_DIR/voice-harness/jobs`. On first recovery,
legacy runtime job JSON is imported under both legacy and durable locks; conflicting
same-revision imports are preserved in the durable quarantine instead of replacing
state. Linux boot identity is part of worker and target-release ownership, so a
reused PID after reboot cannot inherit a stale claim. Recovery retains active,
undelivered, uncertain, fenced, manual-review, and quarantined records. It prunes
only delivered terminal jobs whose completion is more than seven days old and never
automatically deletes quarantine evidence.
Unresolved quarantine evidence conservatively fences conflicting target and
worktree reservations. Operators may explicitly release that fence through the
typed `JobStore.acknowledge_quarantine_reservations()` API, which writes a
hash-bound resolution tombstone while preserving the quarantined payload and
metadata.
