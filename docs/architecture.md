# Architecture

```text
PipeWire microphone
  -> OpenWakeWord ("Hey Jarvis")
  -> Parakeet TDT 0.6B v2 via ONNX Runtime (CUDA)
  -> Qwen3.5-4B Q4_K_M via llama.cpp (CUDA)
       -> focused intent classification
       -> ordinary conversational response
       -> Herdr-managed Cursor agent, GitHub CLI, and optional integrations
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

Focused dictation supports both manual and VAD-controlled capture. In VAD mode the
invoking CLI process remains active, repeatedly owns a raw PipeWire stream, frames
it through the same RMS-gated WebRTC detector as the wake daemon, and requires
sustained speech before starting an utterance. It transcribes each utterance after
post-speech silence and rearms, including when STT finds no recognizable speech. A
concurrent invocation signals that owner to stop instead of attempting
WAV handoff itself. The owner closes the WAV and atomically creates the immutable
generation before transcription, which prevents duplicate or partially written
handoffs.

## Cursor routing

Cursor routing works as follows:

1. Ask the configured LLM backend for one focused, forced-tool classification of
   conversation, new work, follow-ups, clarification replies, status, cancellation,
   and ending the conversation without rewriting the user's request. Non-actionable
   routes fall through to tool-free conversation, so only a high-confidence router
   result can mutate a workspace. A high-confidence `end_conversation` result lets the
   assistant voluntarily wind down when nothing further is needed: it speaks a brief
   farewell and closes the conversation, releasing the microphone and models without
   waiting for the inactivity timeout. A spoken close phrase such as "goodbye" or "stop
   listening" remains a fast path that closes immediately.
2. Prefer an idle Cursor agent already running in the requested checkout.
3. Enabled integrations may contribute issue recognition, bounded external context,
   agent instructions, and repository routing. The optional Linear connector asks a
   dedicated routing agent to inspect the ticket through Linear MCP when no repository
   is named; it is never loaded or diagnosed while disabled.
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

### Multi-ticket fan-out

One trusted utterance may explicitly name multiple GitHub or Linear issues. A pure
planner extracts full references in textual order and deduplicates their canonical
identities. Bare positive numbers are accepted only when captured provider metadata
proves exactly one scope: an exact GitHub repository `/issues` page or a Linear
`/team/<KEY>/...` page. Page content never supplies ticket identifiers.

Every unique target is preflighted before any child starts. GitHub targets must
resolve through `gh`; Linear targets must have valid team syntax, an enabled
integration, and a healthy Cursor MCP capability. Rejected targets do not prevent
the remaining valid targets from starting. Valid targets become ordinary,
target-scoped `AgentJob` records through the existing job-creation and worker-launch
path. Starts are bounded by the configured concurrency and immediately background;
single-ticket requests retain the normal foreground window.

Fan-out is deliberately best-effort rather than crash-atomic. There is no durable
batch schema, manifest, rollback, or shared lifecycle: the response only summarizes
each unique target as `accepted`, `rejected`, or `start-failed` in request order.
Each accepted child independently owns planning, questions, recovery, delivery, and
the job-store maintenance fence. Linear repository routing serializes acquisition,
prompting, and marker parsing under a cancellation-aware cross-process lock around
the stable `voice-router` agent.

After checkout preparation, every new ticket enters a durable tiered workflow. A
Plan-mode participant first performs a short read-only inspection and classifies the
ticket. Clear, localized, reversible work is `simple` and goes directly to a fresh
Agent-mode implementer. `medium` work receives a persisted plan and one independent
review from a fresh Ask-mode participant. `high-risk` work receives the same review
gate plus one bounded planner revision. If the final review still rejects the plan,
the job enters an explicit exhausted-review clarification: only a trusted spoken
`approve` queues the unchanged plan for implementation, a trusted `abort` uses the
normal cancellation fences, and any other reply remains awaiting. The job records
whether approval came from the reviewer or from that explicit user override.
Persistence, migration, recovery,
concurrency, lifecycle, worktrees, external writes, security, infrastructure,
destructive behavior, public APIs, and ambiguous requirements force the high-risk
tier. After the latest review approves, the job still pauses at the exact Cursor
Plan Mode Build boundary. In the default `ask` mode it speaks a yes-or-no question.
An affirmative answer queues `lgtm. Implement the approved plan.` through the Herdr
prompt API to the retained planner; a negative answer uses normal cancellation and
retains the plan artifact. Newly discovered scope promotes the tier and returns the
job to planning.

The boundary is not inferred from an idle status. A stable, turn-scoped
`WORKFLOW_PLAN` marker may prove it while the planner still reports `working` or
`blocked`. The job persists a digest-derived gate ID, canonical Herdr session,
state-change sequence, and optional revision. Approval submission rechecks the same
session and sequence and disables the desktop Enter-key fallback. The ordinary
prompt operation fence supplies `planned`, `submitting`, `submitted`, and observed
recovery; only a Herdr-accepted explicit submission enters the preference ledger.
Ambiguous submission is never retried without positive sequence evidence. If the
preference store is temporarily unavailable after implementation, the completed
result is persisted and preference reconciliation retries locally without observing
or prompting the old Herdr session again.

After three distinct accepted explicit submissions, the successful third
implementation speaks a one-time offer for automatic plan approval. Accepted
automatic mode never applies to a high-risk workflow and bypasses only this reviewed
Plan Mode Build gate. Deterministic security, destructive, infrastructure, migration,
authentication, permission, and other hard-risk evidence keeps asking. Reviewer
objections, unresolved product/architecture questions, interactive questionnaires,
and tool permission prompts always retain their existing user decision path.

The selected tier, evidence, current phase, review round, active participant, and
participant targets live in the job record. Full plans and reviews are bounded,
hash-validated, content-addressed sidecar artifacts under the durable jobs
directory, never files in the checkout. New artifacts are published
create-exclusively and are never overwritten; an identical retry reuses the same
reference, while conflicting bytes fail closed. Review artifacts record the exact
plan digest they reviewed. Publication checks the current worker claim, turn,
workflow phase, review round, and prior artifact reference while holding the same
job-directory lock used to update the job, so stale worker output cannot create an
orphan sidecar or advance the workflow. A crash after sidecar creation but before
the job update can leave a harmless immutable orphan that an identical retry
reuses and normal job pruning/deletion removes. Round-only references from earlier
schema-v10 development builds remain read-only compatible; all new writes include
the artifact digest in their filename. Malformed artifacts are quarantined.
Phase-specific, turn-scoped output markers prevent stale terminal text from
advancing a workflow. Prompt delivery is a durable operation with `planned`,
`submitting`, `submitted`, `ambiguous`, and `none` states plus the phase, turn,
target, and observed Herdr sequence. Recovery submits only `planned`, observes only
`submitted`, and never retries `submitting` without positive sequence evidence.
During migration to schema v11, the former schema-v10 prompt boolean is translated:
a true value becomes an ambiguous fail-closed operation because it has no
trustworthy sequence baseline.

Fresh workflow participants persist their deterministic name, role, label, and
workspace intent before pane creation. The returned pane and workspace are persisted
before agent startup. A crash across pane creation retains the identity and requires
reconciliation rather than creating another pane.

Success, failure, and cancellation first persist a terminal intent while the job is
`reconciling`. Cleanup then cancels every deduplicated planner, reviewer, implementer,
active, and pending participant target. Only confirmed cleanup clears those handles
and publishes the terminal status; observation or cancellation uncertainty retains
the release and reservation fences. User decisions always return through the
existing voice clarification flow. One ticket
therefore remains one inbox item across planning, review, implementation, restart,
clarification, cancellation, and delivery.

The harness never automatically commits, pushes, opens pull requests, modifies Linear,
or deletes generated worktrees. Fork creation is the only supported GitHub write and
is performed only after an unambiguous spoken request and a separate affirmative
confirmation. Checking out a focused pull request only reads from GitHub and writes to
its isolated local worktree. PR worktrees are reused only by recovery or continuation
of the same job. Completed and cancelled worktrees are retained for inspection, while
an invalid or partially prepared checkout is marked quarantined and is never dispatched.

### Durable question broker

Agent questions cross a provider-neutral broker contract before entering the Cursor
job flow. A question records its type (free text or multiple choice), choices,
decision sensitivity, provider, opaque job identity, and originating turn token.
The initial Cursor adapter stores that versioned envelope in the same atomically
replaced job JSON as `AWAITING_USER`; the legacy `question` and
`clarification_kind` fields remain mirrored for voice and inbox compatibility.
There is no sidecar transaction that can disagree with job state.

Answers are compare-and-swap fenced by question identity and originating turn.
Multiple-choice matching accepts only an exact choice identifier, exact spoken
label, or unambiguous ordinal. Ambiguous and stale answers do not mutate or launch
the job. Security, destructive, architecture, and product questions have no
automatic/default answer path: only trusted user voice or user-text provenance can
be recorded. Legacy plain-text questions have unspecified sensitivity and therefore
use the same protected policy; automation provenance is rejected without mutation.
“Answer later” leaves the question durably deferred, while “repeat” speaks the
same pending question.

An accepted agent answer preserves the original request and retained agent target.
The answer continuation includes the original question and, for multiple choice,
both the stable choice ID and spoken label. Before prompting, the worker persists a
two-phase operation as `planned` with one immutable next-turn token. Immediately
around subprocess launch it records the Herdr baseline sequence and advances through
`submitted` and `observed`. Recovery reuses that turn: a planned operation is safe
to retry, a proven observed operation is read without prompting, and a submitted
operation is retried only after repeated proof that Herdr did not change. Missing or
contradictory evidence blocks for manual attention rather than risking duplicate
delivery. The answer and question remain durable until matching output resolves the
operation.

Cursor owner-specific transitions are selected by a typed handler registry. Built-in
handlers cover agent continuation, local repository selection, GitHub repository
selection, and fork confirmation. Unknown owners remain awaiting input without a
worker launch. Workflow adapters such as tiered planning can register their own
phase handler without adding planning concepts to the provider-neutral package.

Wake routing obtains question text, owner, ID, and turn token from one immutable job
snapshot and uses that same snapshot for intent context and answer fencing. Broker
controls are forced only while that snapshot proves the job is awaiting the question.
Herdr waiting polls `interactive_ready` while the prompt process runs. An unexpected
questionnaire promptly blocks the job without sending keys or selecting an answer.
Recovery performs read-only checks while it remains open; after the user manually
closes it, the same agent and turn are queued for reconciliation-only output reading.
Completion resolves the question; a new question replaces it; cancellation closes
it atomically.

The broker package imports neither Cursor nor Herdr. Provider adapters own their
persistence and opaque resume handles, allowing OpenCode or tiered planning to use
the same question and answer policy without putting workflow phases into the
broker contract.

After a completed job is announced, the wake conversation retains a bounded, one-shot
reference to it. A referential follow-up within that window ("review the changes",
"run the tests") starts a child job that reuses the completed parent's exact retained
checkout: the child records a `parent_job_id`, inherits the parent's immutable
repository, branch, worktree path, workspace, and root-pane identity, and provisions
an agent only at that verified checkout. The completed parent job is terminal and is
never reopened, mutated, or transitioned; a settled agent already at the checkout is
reused, otherwise a new agent is started only in the retained pane. Missing pane
identity fails closed rather than creating an unrecoverable Herdr side effect. The
reference is volatile in-memory conversation state installed only after a successful,
acknowledged announcement, so it never survives a restart, and
awaiting-clarification replies always take precedence over it.

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
metadata. `voice-harness jobs quarantine list` exposes the recorded worker and
reservation identities for inspection; the confirmation-gated `acknowledge`
subcommand records the operator's reason. Bulk job deletion preflights this
evidence before staging cancellations or stopping workers.
