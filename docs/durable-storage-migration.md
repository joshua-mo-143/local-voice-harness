# Durable storage migration: JSON jobs to SQLite

This is the design/inventory for [#139](https://github.com/joshua-mo-143/local-voice-harness/issues/139).
It is deliberately read/design-only: it does not change runtime code, `cursor/model.py`,
or the public `JobStore` API. The implementation phases are [#140](https://github.com/joshua-mo-143/local-voice-harness/issues/140),
[#141](https://github.com/joshua-mo-143/local-voice-harness/issues/141), and
[#142](https://github.com/joshua-mo-143/local-voice-harness/issues/142), in that order.
[#341](https://github.com/joshua-mo-143/local-voice-harness/issues/341) extends the
#142 effect contract in this document only.

SQLite parity and the typed lifecycle children are now implemented. Schema-v18
state is persisted by database schema v2 as described below. The older v12
inventory remains as historical design context for the compatibility importer.

## Current schema-v18 relational inventory

`CursorJob` currently produces 213 persisted field names, including the
dynamically emitted `pane_retained_at` cleanup and
`grouped_repository_coordinator_id` handoff fields. Database schema v2
assigns each exactly once: 207 named lifecycle columns, two immutable artifact
references, and four import-only compatibility values. The executable exhaustive
inventory is `_NAMED_TABLE_FIELDS` in `cursor/sqlite_store.py`; startup rejects a
v18 value not present in that inventory rather than silently creating generic
state.

| Disposition | Named owner | Count |
|---|---|---:|
| Identity and top-level discriminator | `jobs` plus `job_identity.lifecycle_kind` | 25, including `id` |
| Prompt and question | `job_prompt_question` | 15 |
| Terminal intent and cleanup | `job_terminal_cleanup` | 6 |
| Delivery and announcement | `job_delivery_announcement` | 10 |
| Workflow, review, approval, and participant | `job_workflow_review_approval_participant` | 31 lifecycle values |
| Immutable artifact references | `plan_artifact` and `review_artifact` columns, foreign-key evidence in `artifacts` | 2 |
| Checkout and fork | `job_checkout_fork` | 46 |
| Provider and ticket creation | `job_provider_ticket` | 28 |
| Session, pane, reconciliation, and release | `job_session_pane` | 39 |
| Worker ownership | `job_worker` (with claim projection in `worker_claims`) | 7 |
| Import-only compatibility | `schema_version`, `migration_source_schema_version`, `phase_prompt_active`, `agent_identity_legacy_compatible` | 4 |

The executable ownership inventory is
`cursor/lifecycle_ownership.py`. Tests fail when a schema-v18 field or public
transition entry point is added without a row. That inventory is the #357
baseline; it does not change runtime behavior or treat a lower field count as
success.

| Baseline | Count |
|---|---:|
| Persisted field names | 215 |
| Named table fields | 210 |
| Import-only fields | 4 |
| Directly exposed `CursorJob` properties | 154 |
| Compatibility adapters | 38 |
| Public transition entry points | 74 |
| Documented duplicate authorities | 0 |
| Lifecycle-related module lines | 26147 |

#358 unified ordinary and clarification submission onto one
`prompt_operations.PromptOperationState`. #359 made that typed operation the
ordinary runtime representation; flat fields remain only at the persistence
edge. #360 gave checkout and managed-session operations first-class
`CheckoutState` and `AgentSessionState` labels so retained, quarantined,
confirmed absent, and manual required stay available on the typed objects.
There are no remaining documented duplicate authorities. #360 followed #359
because both overlap `model.py`, `provisioning.py`, and `recovery.py`.

The only JSON-valued columns are the nine intrinsically structured, edge-validated
values: `voice_question`, `clarifications`, `prompt_context_sessions`,
`prompt_manifest`, `grouped_repository_targets`,
`grouped_repository_candidates`, `grouped_repository_launches`,
`target_release_unverified_targets`, and `participant_session_owners`.
`job_field_presence` records serialization presence only; lifecycle values always
come from named columns. `job_fields` is retained unchanged as v1 migration
evidence and is never read or written by canonical schema-v2 operations.

Opening a v1 database first takes the checkout-local
`.sqlite-bootstrap.lock`, then starts one `BEGIN IMMEDIATE` transaction and
rechecks the marker. Every EAV record passes through the current `CursorJob`
compatibility adapter and durable-write normalizer. Legacy settled operations
without typed identity are retained as explicit uncertainty rather than being
blessed as current state. After writing named rows, migration reloads each row
through the canonical relational reader and validates it as native schema v18
before changing `store_meta.schema_version` last. Any invalid or lossy record
rolls back the schema projection and version marker together. Reservations,
delivery and worker claim projections, maintenance, quarantine, artifacts, and
outbox rows are not rewritten by this migration. Retrying either repeats the
whole v1 projection or observes completed v2 state, so cutover is idempotent.
Fresh base schema creation and its marker use that same transaction; no
`executescript` implicit commit is used. A marker-free partial database is
completed only when its existing tables contain no evidence. Populated unknown
or unmarked tables fail closed and remain untouched.

## Pre-migration baseline

`cursor/store.py` is the durable transaction boundary. A per-directory `flock`
serializes readers and writers; a candidate is parsed and normalized, compared
with the previous revision, checked against peer reservations and quarantine
evidence, then atomically replaced as JSON. `cursor/model.py` supplies v12
migration, typed conversion, transition validation, and cross-field invariants.
Recovery, service, provisioning, delivery, and worker lifecycle all submit
closures to that store boundary.

```mermaid
flowchart LR
    S[service.py / jobs.py] -->|closures| L[.lock + flock]
    W[detached worker] -->|CAS-like update| L
    R[recovery.py] -->|reconcile/update| L
    L --> M[model.py<br/>v0..v12 migration + validation]
    M --> J["$STATE/jobs/<job-id>.json"]
    L --> Q[".quarantine/<br/>payload + metadata + resolution"]
    L --> A[".artifacts/<job>/<br/>content-addressed plan/review"]
    D[delivery.py] --> L
    P[provisioning.py] --> L
    X[Herdr / GitHub / filesystem effects] -. observed outside lock .-> R
```

### JSON write-surface verification

The repository-wide search covered production Python under
`src/local_voice_harness/`, repository scripts, and service/CLI entry points
for JSON serialization, file writes, durable-job path constants, and
`JobStore` construction/calls. Tests were checked separately as fixtures and
fault-injection clients, not production writers. The result is:

* Job records, maintenance fences, quarantine metadata, quarantine resolution
  tombstones, and workflow-artifact sidecars are written by the private helpers
  in `cursor/store.py`.
* `cursor/jobs.py::write_job` is a compatibility facade; it delegates to
  `JobStore.create/update` and does not write JSON itself.
* `app.py`, `wake/daemon.py`, `cursor/worker.py`, and the agent-neutral
  `agents/store.py` facade construct or re-export `JobStore`; they do not write
  job JSON directly.
* `model.py`, `recovery.py`, `provisioning.py`, `worker_lifecycle.py`, and
  `questions.py` do not write persisted JSON files. `service.py` emits two
  diagnostic `json.dumps(...)` lines to stdout only.
* `recovery.py` and `provisioning.py` consume provider JSON returned by Herdr;
  that is not local persistence.
* Cursor worker logs are transient files under `$XDG_RUNTIME_DIR`; they are not
  job state.

Thus there are no hidden Cursor job JSON writers outside the documented store
paths. This statement is scoped to Cursor job durability; unrelated vocabulary,
plan-preference, audio, and service configuration JSON stores remain separate
and are not part of this migration.

### Current transition ownership

The existing implementation has one durable write boundary, but transition
decisions are distributed across these callers:

| Module | Transition decisions and durable responsibilities |
|---|---|
| `cursor/store.py` | Owns the lock, load/normalize/validate/write transaction, revision CAS, peer reservation checks, quarantine fencing, maintenance lease, delivery claim CAS, and artifact sidecars. |
| `cursor/model.py` | Owns typed conversion, v0–v12 normalization, legal status/workflow transitions, cross-field invariants, worker ownership validation, question validation, and delivery-state evolution. |
| `cursor/recovery.py` | Reconciles detached workers, external targets, forks, and worktrees from observations; requests state changes through `JobStore`. |
| `cursor/service.py` | Coordinates admission, cancellation, pruning, delivery orchestration, maintenance operations, and worker launch/recovery; it does not write job JSON directly. |
| `cursor/provisioning.py` | Coordinates fork/worktree/pane provisioning and records provider outcomes through store updates; reservation admission remains enforced by the store. |
| `cursor/delivery.py` | Owns the claim/renew/acknowledge/release protocol and playback ordering; the store persists the claim and terminal delivery state. |
| `cursor/worker_lifecycle.py` | Owns detached-process launch, process identity capture, checkpoints, and stop/reconcile observations; it does not decide domain transitions or write job JSON directly. |

The target coordinator consolidates these decisions without changing their
public entry points: callers submit commands or observations, and only the
coordinator performs the resulting durable transition.

## Target state

SQLite becomes authoritative for job state, reservations, claims, leases,
quarantine references, and pending effects. Large plan/review contents remain
immutable content-addressed files. A single coordinator owns transitions;
workers report observations and execute named effects.

```mermaid
flowchart LR
    C[service / worker commands] --> K[Coordinator]
    K -->|BEGIN IMMEDIATE<br/>state + effects| DB[(SQLite WAL)]
    DB --> J[job + submachine rows]
    DB --> R[reservation rows]
    DB --> O[outbox/effect rows]
    DB --> Q[quarantine evidence]
    O --> E[effect executor]
    E --> H[Herdr / GitHub / filesystem]
    H -->|observation/event| K
    J -. hashes/references .-> A["immutable .artifacts files"]
    K --> V[typed state transitions]
```

The coordinator should preserve externally visible statuses and at-least-once
delivery. A crash after a transaction commits resumes from outbox rows; a crash
after an external effect but before its observation remains `OutcomeUnknown`
and is reconciled rather than blindly replayed.

### Target lifecycle boundaries

The top-level lifecycle is a sum type; fields belonging to another variant are
not nullable alternatives on the same state:

```text
Job =
  Queued(checkout)
  | CheckingOut(checkout, operation)
  | Running(session, checkout)
  | AwaitingQuestion(session, question)
  | AwaitingPrompt(session, prompt)
  | Reviewing(session, review)
  | Terminal(intent, delivery)
  | Reconciling(intent, external_operations)
  | Quarantined(evidence, reservations)
```

Each job owns independent submachines whose uncertainty is explicit:

* `Checkout`: `Unrequested | Provisioning | Ready | Failed | Unknown |
  ManualRequired`
* `Session`: `Absent | Starting | Active | Stopping | AbsentUnknown |
  | ManualRequired`
* `Prompt`: `Idle | Submitting | Submitted | OutcomeUnknown | Failed`
* `Question`: `None | Asked | Answered | Expired | Cancelled`
* `Delivery`: `NotReady | Ready | Claimed | Retryable | Delivered`

Commands are `Admit`, `StartCheckout`, `ReportCheckout`, `StartSession`,
`StopSession`, `SubmitPrompt`, `AskQuestion`, `AnswerQuestion`,
`PrepareTerminal`, `ClaimDelivery`, `AcknowledgeDelivery`, `ReleaseDelivery`,
`Reconcile`, and `AcknowledgeQuarantine`. Provider results are observations,
not transitions: they are accepted only when the operation idempotency key and
expected revision match the current row. A transition atomically updates the
top-level variant, affected submachine rows, reservations, an audit event,
and any resulting outbox effects.

## v12 persisted-field inventory

The four typed sets in `cursor/model.py` contain 158 scalar fields. The
structured v9+ layout additionally persists `voice_question`, whose value is
validated as a `Question` object rather than by one scalar type. `loaded_schema_version`
is an in-memory provenance value and is **not** a persisted field.

| Classification | Every v12 field | Proposed treatment |
|---|---|---|
| Boolean (21) | `agent_dispatch_exited`, `announcement_dismissed`, `announcement_repeated`, `cancellation_reconciliation_pending`, `continuation`, `delivered`, `fork_committed`, `fork_confirmed`, `fork_dispatch_exited`, `fork_exists`, `fork_operation_source_private`, `fork_requested`, `interactive_questionnaire_blocked`, `phase_prompt_active`, `plan_approval_completion_pending`, `plan_approval_counted`, `reconcile`, `review_approved`, `target_release_pending`, `worktree_dispatch_exited`, `worktree_manual_inspection_required` | `INTEGER NOT NULL CHECK (value IN (0,1))`; move compatibility `phase_prompt_active` to import-only after #142 |
| Integer (20) | `agent_absent_observations`, `agent_reconcile_attempts`, `delivery_attempts`, `delivery_generation`, `fork_absent_observations`, `fork_reconcile_attempts`, `github_issue`, `github_pull_request`, `plan_approval_revision`, `plan_approval_state_change_sequence`, `prompt_baseline_sequence`, `prompt_operation_turn`, `review_round`, `revision`, `schema_version`, `target_release_owner_pid`, `turn`, `worker_pid`, `worktree_absent_observations`, `worktree_reconcile_attempts` | `INTEGER`; non-negative checks for counters/revision/rounds; nullable for optional identities |
| Float/timestamp (31) | `agent_automatic_reconcile_stopped_at`, `agent_confirmed_absent_at`, `agent_last_reconciled_at`, `agent_next_reconcile_at`, `agent_retained_at`, `attempt_started_at`, `completed_at`, `created_at`, `delivered_at`, `delivery_claimed_at`, `delivery_retry_at`, `foreground_until`, `fork_automatic_reconcile_stopped_at`, `fork_committed_at`, `fork_confirmed_absent_at`, `fork_last_reconciled_at`, `fork_next_reconcile_at`, `fork_retained_at`, `manual_reconcile_required_at`, `manual_reconcile_resolved_at`, `next_reconcile_at`, `queued_at`, `started_at`, `terminal_intent_completed_at`, `updated_at`, `worktree_automatic_reconcile_stopped_at`, `worktree_confirmed_absent_at`, `worktree_last_reconciled_at`, `worktree_next_reconcile_at`, `worktree_quarantine_acknowledged_at`, `worktree_retained_at` | `REAL`; finite-value checks at the adapter boundary; use UTC epoch seconds consistently |
| String/enum (86) | `active_participant`, `agent_dispatch_state`, `agent_hint`, `agent_name`, `clarification_kind`, `context_repository`, `continuation_answer`, `delivery_claim_token`, `error`, `fork_operation_login`, `fork_operation_source`, `fork_operation_source_default_branch`, `fork_operation_source_parent`, `fork_operation_source_url`, `fork_operation_state`, `fork_operation_target`, `fork_repository`, `github_issue_context`, `github_issue_url`, `github_repository`, `harness_kind`, `herdr_pane_id`, `herdr_target`, `herdr_workspace_id`, `id`, `implementer_target`, `issue_key`, `manual_reconcile_operation`, `manual_reconcile_outcome`, `manual_reconcile_token`, `parent_job_id`, `participant_creation_label`, `participant_creation_pane_id`, `participant_creation_participant`, `participant_creation_state`, `participant_creation_target`, `participant_creation_workspace_id`, `plan_approval_agent_session`, `plan_approval_id`, `plan_approval_source`, `plan_approval_state`, `plan_artifact`, `planner_target`, `prompt_operation_phase`, `prompt_operation_state`, `prompt_operation_target`, `pull_request_branch`, `pull_request_worktree_error`, `pull_request_worktree_state`, `question`, `reconciliation_base_error`, `repository`, `repository_hint`, `request`, `result`, `review_approval_source`, `review_artifact`, `review_decision`, `reviewer_target`, `session_id`, `speakable_label`, `status`, `target_release_owner_boot_id`, `target_release_owner_start`, `target_release_token`, `terminal_intent_error`, `terminal_intent_result`, `terminal_intent_status`, `trusted_utterance`, `turn_token`, `utterance`, `worker_boot_id`, `worker_operation`, `worker_process_start`, `worker_token`, `workflow_classification_reason`, `workflow_phase`, `workflow_tier`, `workflow_turn_phase`, `worktree_branch`, `worktree_label`, `worktree_path`, `worktree_provision_error`, `worktree_provision_state`, `worktree_root_pane_id`, `worktree_workspace_id` | `TEXT`; use lookup/check constraints for finite enums; secrets/tokens remain opaque |
| Structured payload (1) | `voice_question` | Separate `question` row or canonical JSON column during #140; #141 makes it a typed question submachine. Keep a canonical serialized copy only at the adapter edge, not as the lifecycle source of truth |

The model also has compatibility aliases and structured input containers
(`harness_state`, `checkout_state`, `provider_state`) that are import/serialization
layouts, not additional v12 fields. `schema_version` is retained by #140 for
import parity and becomes an SQLite schema/import version, not a per-job
compatibility mechanism after #142.

### Exhaustive migration disposition

During #140 all 159 persisted values remain lossless in the validated
`payload_json` parity envelope. The list below assigns every value exactly once
to its final #141/#142 disposition; "side table" means typed columns rather
than a generic JSON lifecycle blob.

| Final disposition | Persisted fields |
|---|---|
| Core `jobs` row (21) | `id`, `parent_job_id`, `harness_kind`, `revision`, `status`, `request`, `utterance`, `trusted_utterance`, `created_at`, `updated_at`, `queued_at`, `started_at`, `completed_at`, `foreground_until`, `continuation`, `continuation_answer`, `reconcile`, `agent_hint`, `context_repository`, `repository_hint`, `speakable_label` |
| Workflow/review/approval side tables (20) | `workflow_tier`, `workflow_phase`, `workflow_classification_reason`, `workflow_turn_phase`, `active_participant`, `planner_target`, `reviewer_target`, `implementer_target`, `review_round`, `review_approved`, `review_approval_source`, `review_decision`, `plan_approval_state`, `plan_approval_id`, `plan_approval_agent_session`, `plan_approval_state_change_sequence`, `plan_approval_revision`, `plan_approval_source`, `plan_approval_counted`, `plan_approval_completion_pending` |
| Provider and checkout side tables (27) | `issue_key`, `github_repository`, `github_issue`, `github_issue_url`, `github_issue_context`, `github_pull_request`, `pull_request_branch`, `repository`, `worktree_branch`, `worktree_path`, `worktree_workspace_id`, `worktree_root_pane_id`, `worktree_label`, `pull_request_worktree_state`, `pull_request_worktree_error`, `fork_repository`, `fork_requested`, `fork_confirmed`, `fork_exists`, `fork_committed`, `fork_committed_at`, `fork_operation_source`, `fork_operation_source_private`, `fork_operation_source_url`, `fork_operation_source_default_branch`, `fork_operation_source_parent`, `fork_operation_login` |
| Session and worker-claim side tables (13) | `session_id`, `agent_name`, `herdr_target`, `herdr_pane_id`, `herdr_workspace_id`, `worker_token`, `worker_pid`, `worker_boot_id`, `worker_process_start`, `worker_operation`, `attempt_started_at`, `turn`, `turn_token` |
| Provider-operation side tables (43) | `agent_dispatch_state`, `agent_dispatch_exited`, `agent_absent_observations`, `agent_reconcile_attempts`, `agent_last_reconciled_at`, `agent_next_reconcile_at`, `agent_automatic_reconcile_stopped_at`, `agent_confirmed_absent_at`, `agent_retained_at`, `fork_operation_state`, `fork_operation_target`, `fork_dispatch_exited`, `fork_absent_observations`, `fork_reconcile_attempts`, `fork_last_reconciled_at`, `fork_next_reconcile_at`, `fork_automatic_reconcile_stopped_at`, `fork_confirmed_absent_at`, `fork_retained_at`, `worktree_provision_state`, `worktree_provision_error`, `worktree_dispatch_exited`, `worktree_absent_observations`, `worktree_reconcile_attempts`, `worktree_last_reconciled_at`, `worktree_next_reconcile_at`, `worktree_automatic_reconcile_stopped_at`, `worktree_confirmed_absent_at`, `worktree_retained_at`, `worktree_manual_inspection_required`, `worktree_quarantine_acknowledged_at`, `participant_creation_state`, `participant_creation_participant`, `participant_creation_target`, `participant_creation_label`, `participant_creation_workspace_id`, `participant_creation_pane_id`, `prompt_operation_state`, `prompt_operation_phase`, `prompt_operation_turn`, `prompt_operation_target`, `prompt_baseline_sequence`, `next_reconcile_at` |
| Question side table (4) | `question`, `voice_question`, `clarification_kind`, `interactive_questionnaire_blocked` |
| Delivery and terminal-intent side tables (15) | `delivered`, `delivered_at`, `delivery_attempts`, `delivery_generation`, `delivery_claim_token`, `delivery_claimed_at`, `delivery_retry_at`, `announcement_dismissed`, `announcement_repeated`, `result`, `error`, `terminal_intent_status`, `terminal_intent_result`, `terminal_intent_error`, `terminal_intent_completed_at` |
| Reconciliation and release side tables (12) | `cancellation_reconciliation_pending`, `manual_reconcile_operation`, `manual_reconcile_outcome`, `manual_reconcile_token`, `manual_reconcile_required_at`, `manual_reconcile_resolved_at`, `reconciliation_base_error`, `target_release_pending`, `target_release_token`, `target_release_owner_pid`, `target_release_owner_boot_id`, `target_release_owner_start` |
| File-backed artifacts with SQLite metadata (2) | `plan_artifact`, `review_artifact` become foreign-key references to `artifacts`; immutable artifact bytes remain content-addressed files |
| Import-only, then delete (2) | `schema_version` selects the one-shot importer and becomes store metadata; `phase_prompt_active` is a compatibility alias. Neither remains on the typed job after #142 |

### Invalid combinations and typed replacements

The following list covers the cross-field and transition checks currently
performed by `CursorJob.from_dict`, `validate_invariants`,
`validate_transition`, and `validate_reservations`. Primitive parsing checks
(field type, finite number, JSON shape, and enum membership) remain adapter
validation rather than lifecycle states.

| Invalid combination rejected today | Typed-state or coordinator replacement |
|---|---|
| A status lacks its required payload or timestamp: queued without `queued_at`; awaiting-user without question/result; blocked/completed/cancelled without result; failed without error/result; terminal without `completed_at` | Constructors for `Queued`, `AwaitingQuestion`, and each `Terminal` result variant require their payload and timestamp; absent fields cannot be constructed |
| Identity, creation time, parent, or harness changes; revision does not advance by one; status/workflow phase takes an illegal edge; workflow tier is downgraded | Immutable `JobIdentity` plus command-specific transition functions; coordinator CAS increments revision and transition tables expose only legal edges |
| A follow-up changes inherited repository, branch, path, workspace, or root pane | Immutable `CheckoutSnapshot` is copied when the child and its reservation are created in one transaction |
| Worker token/PID/boot/process-start is partial, a worker state has no complete owner, or PID is non-positive | `WorkerClaim` is either `Unclaimed` or `Claimed(token, ProcessIdentity, operation, claimed_at)`; no partial representation |
| Delivery token and claim time are unpaired, or a delivered job retains a live claim | `Delivery = Ready | Claimed(token, claimed_at, lease, generation) | Retryable | Delivered(delivered_at)` |
| Workflow is classified without tier/reason, simple enters planning/review, medium exceeds one review, revising remains at round two, or round is outside 0–2 | `Workflow` variants carry classification proof; tier-specific transition tables constrain phases and review-round constructors constrain range |
| Review approval/source are unpaired; approval lacks an artifact; reviewer approval lacks an approve decision; user override is not an exhausted high-risk rejection; approved artifacts do not match the current round | `Review = Pending | Rejected | Approved(ApprovalProof, PlanArtifact, ReviewArtifact, round)` with distinct reviewer and high-risk user-override proof variants |
| Plan approval retains proof while inactive, lacks complete proof while active, appears in the wrong phase/status, is counted without explicit acceptance, or marks completion without durable finished output | `PlanApproval = None | Boundary | Awaiting(QuestionRef, GateProof) | Approved(ApprovalProof) | Observed(ApprovalProof) | Rejected`; workflow commands control allowed embeddings |
| A planned medium/high-risk workflow implements without current approved plan/review artifacts and plan approval | `Implementing` constructor requires an `ApprovedPlan` aggregate referencing matching immutable artifacts and approval proof |
| Active participant target differs from the session target; participant creation lacks role/target/label or a created pane lacks workspace/pane IDs | `ParticipantCreation = None | Creating(ParticipantSpec) | Created(ParticipantSpec, PaneIdentity) | OutcomeUnknown | ManualRequired`; active participant references the created session target |
| Prompt operation lacks phase/turn/target, or a submitting/submitted/ambiguous prompt lacks baseline sequence | `Prompt = Idle | Submitting(PromptSpec, baseline) | Submitted(PromptSpec, baseline) | OutcomeUnknown(PromptSpec, baseline) | Failed` |
| Agent, fork, or worktree operation is uncertain without its target/spec; worktree provisioning lacks repository/branch/path | Operation variants carry their required `SessionTarget`, `ForkSpec`, or `CheckoutSpec`; uncertain outcomes retain the same operation identity |
| Terminal intent exists outside reconciling/release-fenced state; target release owner identity is partial or remains after release | `Reconciling(TerminalIntent, TargetReleaseLease)` requires a complete process identity; confirmation transitions atomically to `Terminal` and removes the lease |
| Manual reconciliation operation/token/state disagree; cancellation reconciliation has no uncertain operation/release fence or an incompatible status | `ManualRequired(OperationRef, token)` and `CancellationReconciling(TerminalIntent, NonEmptySet[OperationRef])` encode the fence directly |
| Artifact reference has the wrong job/kind/round, or approved plan/review artifacts do not share the active round | `ArtifactRef(job_id, kind, round, digest)` is validated on insertion; foreign keys and the `ApprovedPlan` constructor bind matching references |
| Two jobs reserve the same ticket, target, or worktree, including unresolved quarantine evidence | Unique reservation rows are inserted in the same transaction as the state change; quarantine owns reservation rows until hash-bound acknowledgement |

## Invariants to preserve

| Existing invariant | SQLite/coordinator mapping |
|---|---|
| Exclusive per-store lock | One coordinator connection; WAL for readers, `BEGIN IMMEDIATE` for every state/effect transaction. A short migration/maintenance lease fences old writers. |
| Revision CAS | `jobs.revision` increments exactly once per transition. Update uses `WHERE job_id = ? AND revision = ?`; zero rows means stale command. |
| Legal job/workflow transitions | Typed transition functions and transition tables in #141; SQL checks only enforce local facts, never replace domain validation. |
| Typed/finite/JSON validation | Decode at the edge, validate enums and `Question`, reject non-finite numbers and unknown structured fields before a transaction. |
| Delivery claim ownership | `delivery_claim_token`, `claimed_at`, attempts, retry time, and generation are one claim row or one constrained submachine; acknowledge/release require the token and a live lease. |
| Worker ownership | Token, PID, boot ID, and process start are an all-or-none claim. PID is meaningful only with boot and process-start fences; worker checkpoints compare token/status/terminal intent. |
| Target release fence | A release row stores token plus owner PID/boot/start as a complete tuple. Terminal intent remains reconciling until external targets are confirmed or manually resolved. |
| Reservations | Materialized `reservations(resource_kind, resource_key, job_id)` rows with a unique `(resource_kind, resource_key)` index. Active, uncertain, manual, quarantined, and release-fenced jobs retain rows. |
| Ticket uniqueness | A partial unique index over canonical active GitHub/Linear ticket identity replaces peer scanning for active resource-owning jobs. |
| Follow-up checkout identity | Child has a parent foreign key and immutable repository/branch/path/workspace/pane snapshot; creation and reservation are one transaction. Parent stays terminal and unchanged. |
| External uncertainty | Operation rows use `planned`, `submitting/submitted`, `confirmed`, `ambiguous`, `manual_required`, or `confirmed_absent`; absence of evidence never implies absence. |
| Artifact integrity | SQLite stores immutable reference, digest, kind, round, and source-plan digest. Files remain exclusive-create and hash-verified; malformed files create quarantine rows. |
| Quarantine fail-closed | Quarantine evidence is durable and reservation-bearing until a hash-bound operator acknowledgement. It is never silently deleted by pruning or rollback. |
| Maintenance deletion | A maintenance lease has unique token and process identity. It blocks writes, requires no active workers/fences/legacy records/unresolved quarantine, then deletes in one controlled operation. |
| At-least-once delivery | Result state commits before speech; the effect is acknowledged only after playback. Crash/retry can repeat but cannot silently lose an undelivered terminal. |

## Proposed SQLite shape

SQLite should use `PRAGMA foreign_keys=ON`, WAL mode, a busy timeout, and
explicit transactions. Suggested logical tables (exact normalization can wait
for #141):

* `store_meta(key PRIMARY KEY, value TEXT NOT NULL)` — database format,
  importer status, source-directory fingerprint, and cutover marker.
* `jobs(job_id TEXT PRIMARY KEY, parent_job_id TEXT REFERENCES jobs,
  revision INTEGER NOT NULL, status TEXT NOT NULL, harness_kind TEXT NOT NULL,
  created_at REAL NOT NULL, updated_at REAL, payload_json TEXT NOT NULL)` —
  #140 may retain `payload_json` as a parity envelope, but it is transactionally
  updated and validated. #141 progressively moves lifecycle data into typed
  columns/tables; #142 removes the generic blob as source of truth.
* `workflow(job_id PRIMARY KEY REFERENCES jobs, tier, phase, classification_reason,
  review_round, active_participant, plan_artifact, review_artifact, review_decision,
  approval fields...)` with checks for tier/phase, round range, and approval proof.
* `questions(question_id PRIMARY KEY, job_id REFERENCES jobs, turn_token,
  owner, state, kind, sensitivity, choices_json, answer_json, asked_at,
  answered_at)`; unique `(job_id, turn_token, question_id)` plus a partial
  unique index on `(job_id)` for rows in the current `Asked` state. Answered,
  expired, and cancelled rows remain as history.
* `operations(operation_id PRIMARY KEY, job_id REFERENCES jobs, kind, state,
  idempotency_key UNIQUE, target, baseline_sequence, attempts, next_at,
  manual_token, outcome, observed_at)` for agent, fork, worktree, prompt, and pane.
* `worker_claims(job_id PRIMARY KEY REFERENCES jobs, token UNIQUE, pid,
  boot_id, process_start, operation, claimed_at)` with either all identity
  columns populated or all null.
* `delivery(job_id PRIMARY KEY REFERENCES jobs, delivered INTEGER NOT NULL,
  generation, claim_token UNIQUE, claimed_at, retry_at, attempts, delivered_at)`;
  checks pair token/timestamp and forbid a live claim on delivered state.
* `reservations(resource_kind, resource_key, job_id REFERENCES jobs,
  reason, created_at, PRIMARY KEY(resource_kind, resource_key))`; resource kinds
  include `ticket`, `target`, and `worktree`.
* `artifacts(reference PRIMARY KEY, job_id REFERENCES jobs, kind, round,
  path, digest UNIQUE, source_digest, immutable INTEGER NOT NULL)` with checks
  tying kind/round/job to the reference.
* `quarantine(evidence_id PRIMARY KEY, job_id, original_name, payload_path,
  metadata_path, payload_digest, error, resolved_at, resolution_reason)` plus a
  unique unresolved evidence identity. Resolution never mutates the evidence.
* `maintenance(token PRIMARY KEY, operation, owner_pid, owner_boot_id,
  owner_process_start, started_at, lease_expires_at, active INTEGER NOT NULL)`
  with at most one active lease; every write verifies the active lease is not
  expired, and releasing it requires the complete owner identity tuple.
* `outbox(effect_id PRIMARY KEY, job_id REFERENCES jobs, kind, idempotency_key
  UNIQUE, payload_json, status, attempts, next_at, lease_token, leased_at,
  completed_at, outcome_json, last_error)` plus
  `outbox_concurrency(effect_id REFERENCES outbox, concurrency_key)`; effect
  status must distinguish pending, running, succeeded, failed,
  failed-retryable, and unknown.
* `events(event_id PRIMARY KEY, command_id UNIQUE, job_id REFERENCES jobs,
  revision, kind, payload_json, created_at)` for audit/recovery evidence; event
  insertion and state update are atomic. Duplicate `command_id` delivery is a
  no-op.

SQL constraints should enforce identity, uniqueness, null pairing, booleans,
counter ranges, foreign keys, and immutable artifact identity. Domain-specific
transition and reservation decisions remain in typed coordinator code so the
database cannot accidentally encode an incomplete policy.

### Coordinator and effects

The durable API should conceptually be:

```text
command(job snapshot, expected revision)
  -> BEGIN IMMEDIATE
     validate command and current submachines
     write new state, event, reservations, and outbox effects
     COMMIT
  -> execute effects outside transaction
  -> report observation(effect id, idempotency key, outcome)
```

[#342](https://github.com/joshua-mo-143/local-voice-harness/issues/342) implements
the commit half of this API as `JobStore.apply`. A typed `CoordinatorCommand`
against an expected revision produces a `CoordinatorDecision` (canonical job plus
named `DurableEffect`s). State, reservations, one `events` row, and pending
outbox rows commit in one `BEGIN IMMEDIATE` transaction or not at all. Existing
`JobStore.create` and `JobStore.update` record through that same path so the
coordinator is production-used before every caller is migrated. Effect execution
is [#345](https://github.com/joshua-mo-143/local-voice-harness/issues/345).

[#345](https://github.com/joshua-mo-143/local-voice-harness/issues/345) implements
the execution half: `JobStore.claim_outbox` leases a pending row, handlers run
after that transaction commits, and `observe_outbox` records `Confirmed`,
`ConfirmedAbsent`, `Failed`, `OutcomeUnknown`, or `ManualRequired` against the
effect ID and idempotency key. A supervisor renews the lease while a handler is
running, and durable concurrency keys prevent two rows for the same provider
resource from running together. Executors never write canonical job rows.
`recover_jobs` accepts the registered domain handlers, reaps expired leases,
and drains due work. A crash before the submit fence schedules a bounded retry
with backoff; a crash after it becomes `OutcomeUnknown`. Retry exhaustion is
terminal `Failed`, and a post-fence retry request is forced to
`OutcomeUnknown`.

An effect executor must lease an outbox row, renew it, and report `Confirmed`,
`ConfirmedAbsent`, `Failed`, or `OutcomeUnknown`; it must not directly update
`jobs`. Named effects, payload identity, and replay rules are the
[#341](https://github.com/joshua-mo-143/local-voice-harness/issues/341)
contract below. Read-only observations (`reconcile`, `stream_events`,
`worktree list`, `gh`/`git` inspect) are not outbox rows.

The Rust coordinator recommendation is intentionally non-blocking: evaluate a
small Rust process/library after the Python state model and SQLite schema have
stabilized. Rust is attractive for one-writer supervision, explicit enums,
crash-safe effect scheduling, and property testing, but it should not be a
prerequisite for #140 or #141. Keep SQLite and the provider-effect protocol
language-neutral so a later Rust coordinator can replace the Python one
without changing persisted semantics.

## Durable effect and reconciliation contract (#341)

This section closes the remaining coordinator-design decisions. It does not
change runtime code. Production evidence is the current worker, recovery, and
delivery paths in `cursor/provisioning.py`, `cursor/recovery.py`,
`cursor/delivery.py`, `cursor/store.py`, `integrations/herdr/`,
`integrations/github.py`, `integrations/linear.py`, `local_git.py`, and
`agents/harness.py`.

### Coordinator process ownership

The coordinator is one logical writer API backed by SQLite transactions
serialized with `BEGIN IMMEDIATE`. It is not one process or one permanently
shared connection, and it is not a new systemd unit.

In production the long-lived owner is the wake daemon
(`voice-harness-wake.service` / `wake/daemon.py`). That process already owns
startup recovery, delivery claims, speech playback, and desktop notification.
CLI and `scripts/dev.sh text` admission use the same library through independent
short-lived SQLite connections. Concurrent callers do not bypass the coordinator:
`BEGIN IMMEDIATE` serializes their decisions with the wake daemon.

Detached workers may execute opted-in leased effects and submit fenced
observations through their own short-lived connections. Other production paths
continue to use `JobStore` transactions and their existing domain-specific
reconciliation. `CoordinatorCommand` is not mandatory for every caller.

A dedicated coordinator service is rejected at cutover: it would add an
operational boundary without changing the existing sole-writer invariant.

### Outbox consumer and lease concurrency

Outbox leasing is **not** globally single-consumer. Multiple executors may
run, each holding a lease token that must be renewed; an expired lease may be
stolen. Concurrency is limited per effect family so providers that are already
serialized stay serialized:

| Opted-in family | Executor | Concurrency |
|---|---|---|
| Managed-job `session.create`, `task.submit`, and `clarification.reply` | Detached worker | One lease per `(provider, target)` |

`herdr ensure_server` and detached-worker spawn are not job-scoped outbox
effects. Interactive Rofi selection and spoken confirmation are coordinator
`AskQuestion` commands, not effects.

Outbox adoption is risk-based, not universal. Workspace/worktree and provider
operations retain their typed operation state and read-only reconciliation;
delivery retains claim/playback/acknowledgement; artifact publication retains
content-addressed exclusive-create. Those protocols remain authoritative unless
a separate issue demonstrates a concrete crash-safety gap.

### Observation vocabulary

Every mutating effect reports exactly one of:

| Outcome | Meaning | Replay |
|---|---|---|
| `Confirmed` | Positive evidence the intended side effect exists with the payload identity | No; treat as done |
| `ConfirmedAbsent` | Positive evidence the side effect did not occur | Retry only if the provider rule below allows create-once replay after absence |
| `Failed` | Provider rejected before or without a side effect | Do not retry the same payload; fail the operation |
| `OutcomeUnknown` | Crash, timeout, or failed observation after the submit fence | Never replay; reconcile with the operation rule |
| `ManualRequired` | Reconciliation cannot prove identity | Operator fence; never replay |

Absence of evidence is not `ConfirmedAbsent`. That matches today's
`operation_timeout` / `operation_ambiguous` / `failed_observing` handling.

Outbox `status` stores executor progress (`pending`, `running`, `succeeded`,
`failed`, `failed-retryable`, `unknown`). The observation above is `outcome_json`.
`Confirmed` / `ConfirmedAbsent` complete the row as `succeeded`. `Failed`
without a retry allowance completes it as `failed`. `OutcomeUnknown` is
`unknown` and must be reconciled. `ManualRequired` stays `unknown` with an
operator fence. `failed-retryable` is only for a contractually replay-safe
effect and is never accepted after the submit fence.

### Handler boundaries

Each opted-in effect belongs to exactly one handler. `AgentHarness` stays the #63
session contract: `create_session`, `submit_task`, `stream_events`,
`reply_to_clarification`, `cancel`, and `reconcile`. It does not grow
workspace, worktree, pane, repository, delivery, or artifact APIs. Launch
context (`pane_id`, `workspace_id`, checkout) is allocated before
`create_session`.

| Boundary | Handler | Owns |
|---|---|---|
| `AgentHarness` | `agents.AgentHarness` / `HerdrSession` | Session create, task/clarification submit, cancel, read-only reconcile and event streaming; only the first three are outbox effects |
| Herdr workspace/worktree | `HerdrWorkspace` | Worktree create/open, workspace/tab pane create, owned pane close, Cursor MCP-auth launch-context link |
| Repository/provider | GitHub/Linear providers and `LocalGitRepository` | Fork, clone, remote configuration, pull-request ref checkout, GitHub issue create, Linear ticket create/observe |
| Delivery | `cursor/delivery.py` plus the wake daemon playback/notify path | Claimed speech and desktop notification |
| Artifact handling | `JobStore.write_artifact` / `publish_artifact` | Immutable plan/review exclusive-create |

### Effect catalogue

Idempotency keys are `kind` plus the payload identity fields, unique in
`outbox.idempotency_key`. Replay means calling the mutating provider again.
Reconciliation is a read-only observation tied back to the original effect
identity. The catalogue records risk and existing safety contracts; only rows
explicitly identified as opted in are outbox effects.

#### AgentHarness

| Effect | Payload identity | Idempotency key | Provider guarantee | Confirmed | Failed | `OutcomeUnknown` | Replay / reconcile |
|---|---|---|---|---|---|---|---|
| `session.create` | `job_id`, planned `target` name, `pane_id`, `workspace_id`, optional mode | those fields | Herdr `agent start` is not idempotent; the planned name is only a correlation address, never durable session identity | The provider response supplies `(provider, session_id, target)` and a subsequent `reconcile(target, expected_session_id)` returns `active` or `settled` with that exact session | Provider rejected before start (`invalid_session_request`, capability miss) | Timeout or crash before the provider-issued `session_id` is durably observed, or missing session after start (`operation_ambiguous`) | Never adopt or replay from target alone. If the provider-issued session ID was durably observed, reconcile that exact identity. Otherwise inspect the planned target plus pane/workspace only to detect conflicts; absence is not replay proof and any present or indeterminate target becomes `ManualRequired`. A later command may admit a replacement only after operator-confirmed absence. Today's `agent_dispatch_state` `dispatching` / `ambiguous` / `failed_observing` map here. |
| `task.submit` | `PromptIdentity`: `job_id`, phase, turn, `turn_token`, target, `session_id`, `baseline_sequence` | those fields | Not idempotent. `before_submit` / `accepted` remain the durable fence | The live handler returns a matching `TaskSubmission` correlation and baseline, then a terminal event for the same session with `state_sequence > baseline` | Pre-submit identity mismatch (`agent_session_changed`, interactive questionnaire) | Crash between `before_submit` and durable observation, or observation failed | Never replay. Read-only reconciliation currently exposes only shared session sequence, so advancement alone cannot prove this prompt correlation; an unknown outcome remains `ManualRequired`. |
| `clarification.reply` | Same `PromptIdentity` shape on the fenced session | those fields | Same as `task.submit` | Same as `task.submit` | Same as `task.submit` | Same as `task.submit` | Same as `task.submit`. |

`stream_events` and `reconcile` stay on `AgentHarness` and produce
observations, not outbox rows. Session cancellation retains the existing
identity-checked reconcile/cancel/owned-pane cleanup path.

#### Herdr workspace/worktree

| Effect | Payload identity | Idempotency key | Provider guarantee | Confirmed | Failed | `OutcomeUnknown` | Replay / reconcile |
|---|---|---|---|---|---|---|---|
| `worktree.create` | repository path, `voice/...` branch, reserved checkout path, label | those fields | `herdr worktree create` is not idempotent; existing branch/path is observed via `worktree list` | List entry with exact branch and reserved path, plus paired `workspace_id` / `root_pane_id` | Invalid branch, path escape, or Herdr rejection before create | Timeout, or created path ≠ reserved path (`operation_ambiguous`) | Do not replay. List+path: exact match settles; path exists without the expected branch → quarantine / `ManualRequired`; both absent → `ConfirmedAbsent` and a later command may admit a new create. |
| `worktree.open` | repository path, existing checkout path, label | those fields | Opening an already-open matching workspace is reuse | Workspace id matches the retained or newly returned workspace | Follow-up open workspace id disagrees with the retained id | Timeout after `worktree open` | Replay is allowed only after `ConfirmedAbsent` of an open workspace for that path; a matching open workspace is `Confirmed`. |
| `pane.create` | `job_id`, participant or role, planned target, checkout, `workspace_id` or none (workspace create), label | those fields | `workspace create` / `tab create` always allocate a new pane | `accepted(pane_id, workspace_id)` with paired ids | Herdr rejected before create | Timeout after `before_submit`, or missing pane/workspace ids | Never replay. Incomplete identity after submit is `ManualRequired` (`participant_creation_state` `submitting` / `manual_required`). |
| `pane.close` | target, `pane_id`, `workspace_id`, expected checkout | those fields | `pane close` is absent-success (`pane_not_found`) when bindings matched or the pane is already gone | Pane get returns not-found after a matching close, or already absent with complete evidence | `ownership_mismatch` | Timeout or agent/pane disagreement | Reconcile bindings first. Replay close only when the same pane/workspace/checkout still match. Uncertain pane targets stay unverified and block release. |
| `mcp_auth.link` | source workspace, target checkout | those fields | Atomic symlink replace; already-linked matching source is reuse | Target `mcp-auth.json` is the source file | Collision, unsafe mode, or missing source | Crash during `os.replace` | Replay is safe: matching symlink is `Confirmed`; otherwise retry the exclusive replace. This is launch-context prep, not `AgentHarness`. |

#### Repository/provider

| Effect | Payload identity | Idempotency key | Provider guarantee | Confirmed | Failed | `OutcomeUnknown` | Replay / reconcile |
|---|---|---|---|---|---|---|---|
| `github.fork` | `GitHubForkPlan` (source, login, target) | source + login + target | `gh repo fork` is not natively idempotent; `observe_fork` reuses an existing fork whose parent is the source | `observe_fork` returns the target with accepted parent | Private source, invalid plan, or target exists and is not that fork | Timeout after `before_submit` (`GitHubOperationAmbiguous`) | Do not replay after unknown. Observe: matching fork is `Confirmed`; proven absence is `ConfirmedAbsent` and a later command may admit a new fork. Today's `fork_operation_state` `submitted` / `ambiguous` / `failed_observing` map here. |
| `repository.clone` | clone root, relative destination, expected remote identity | those fields | `LocalGitRepository.materialize` reuses a verified existing checkout; clone uses a temp dir then `os.replace` | Destination exists and `verify_checkout` matches | Destination escapes root or remote mismatch | Timeout during clone (`LocalGitOperationAmbiguous` / `repository_clone_ambiguous`) | Replay is safe for a verified existing checkout. After unknown, do not assume absence: inspect destination and temp clone dirs; only `ConfirmedAbsent` of both allows a new clone. |
| `repository.remote.ensure` | verified checkout path, remote name, canonical remote identity and URL | those fields | Setting a named Git remote to the same canonical identity is idempotent | `git remote get-url <name>` canonicalizes to the expected identity | Checkout identity changed, path escaped the clone root, or URL is invalid | Crash after adding or changing the remote | Re-verify the checkout and named remote. Matching identity is `Confirmed`; a missing remote permits replay; a different identity is `ManualRequired` and must never be overwritten automatically. |
| `github.pull_request.checkout` | checkout path, `refs/pull/<n>/head`, expected OID, `voice/github-pr-<job-id>` branch | those fields | Fetch+`checkout -B` is idempotent when `HEAD` already equals the expected OID | `HEAD` OID matches persisted `pull_request_head_oid` | Invalid worktree, shared-clone refusal, or remote/ref/OID validation failure | Timeout during fetch/checkout | Replay is safe when inputs still match GitHub. `LocalGitRefChanged` refreshes the plan once; a second mismatch quarantines. |
| `github.issue.create` | repository, title, body, `correlation_marker` | repository + marker | `gh api` create is not idempotent; marker HTML comment in the body is the observe key | Recent issues contain exactly one matching marker | Validation/auth error before submit, or marker not unique | Timeout or observe failure after submit | Never replay. `observe_issue_creation` settles `Confirmed`; failed observe is `ManualRequired`, not absence. |
| `linear.ticket.create` | team id, team key, title, description, `correlation_marker` | team id + marker | Linear MCP create is not idempotent; marker in the description is the observe key | A separately fenced Linear observation submission finds exactly one ticket | Confirmation missing, capability miss, or invalid plan | Ambiguous router prompt or incomplete MCP result | Never replay. Enqueue a sequence-fenced router `task.submit` to observe the marker; `found` is `Confirmed`, while `not_found` is not proof after a submit fence and `unknown`/`multiple` is `ManualRequired`. |

#### Delivery

| Effect | Payload identity | Idempotency key | Provider guarantee | Confirmed | Failed | `OutcomeUnknown` | Replay / reconcile |
|---|---|---|---|---|---|---|---|
| `delivery.speak` | `job_id`, `delivery_generation`, claim token | job + generation | TTS/PipeWire playback is at-least-once; ack only after successful playback | `acknowledge_delivery` under the live claim | Claim lost or job no longer deliverable | Crash during playback | Replay is allowed. Interrupted or unacked playback releases the claim for retry. Do not ack on interrupt. |
| `delivery.desktop` | `job_id`, `delivery_generation`, claim token | job + generation | `notify-send` is at-least-once | `notify-send` exits zero, then `acknowledge_desktop_delivery` succeeds under the live claim | Nonzero process exit, executable failure, or lost claim; do not acknowledge | Crash after successful process exit but before ack | Replay is allowed; the user may see a duplicate notification. |

Deferred delivery acknowledgement is a coordinator command, not an external
effect.

#### Artifact handling

| Effect | Payload identity | Idempotency key | Provider guarantee | Confirmed | Failed | `OutcomeUnknown` | Replay / reconcile |
|---|---|---|---|---|---|---|---|
| `artifact.publish` | `job_id`, kind (`plan`/`review`), round, content digest, optional source-plan digest | those fields | Exclusive-create: identical bytes reuse the sidecar; different bytes cannot replace it | File exists, hash-verifies, and matches the reference | Workflow fence mismatch (stale worker/turn/phase) before create | Crash after create, before the job reference commits | Replay of the same digest is safe and reuses the sidecar. Never rewrite different bytes. Coordinator stores the artifact reference only after `Confirmed`. |

### Retained compound boundary

`HerdrWorkspace.ensure_agent` continues to own worktree, workspace, pane, and
MCP-auth allocation under its existing fences. It can stop before starting the
provider session. Managed jobs then admit `session.create` with the reserved
target, pane, workspace, checkout, and worker identity. Follow-up participants
reuse the retained checkout and apply the same pane/session seam.

Fork, pull-request, issue-provider, cancellation, delivery, and artifact paths
retain their existing operation-specific protocols. Their catalogue entries
above are not a commitment to outbox migration.

## One-shot JSON to SQLite migration

Migration is an import-on-first-open cutover, not a live dual-writer design.
Before cutover, stop or fence all harness daemons and workers, take a
byte-for-byte backup of the durable jobs directory and legacy runtime jobs
directory, and record paths, timestamps, hashes, and the current application
version.

1. Acquire the legacy and durable locks in the existing order. Create
   `jobs/jobs.sqlite3` inside the durable jobs directory with restrictive permissions,
   foreign keys, WAL, and a migration lease.
2. Inventory ordinary `*.json`, `.maintenance`, `.quarantine` payloads,
   metadata, resolution tombstones, and `.artifacts`. Do not follow symlinks.
3. Import unresolved quarantine evidence before importing reservation-bearing
   rows. Preserve its payload, metadata, digest, error, and resolution status.
   Any candidate competing with unresolved evidence remains fenced.
4. Parse each job through the existing v12 `CursorJob.from_dict` path, apply
   legacy v0–v11 normalization, verify filename/job identity, and validate
   artifacts and reservations. Import valid rows with their original revision,
   timestamps, delivery state, worker identity, and quarantine references;
   reservation insertion is rejected if it conflicts with imported unresolved
   evidence.
5. Import undelivered terminals without pruning or coalescing them. Preserve
   terminal intent/reconciliation rows, delivery generation, retry/claim
   metadata, questions, and result/error text. Do not mark playback delivered.
6. Handle active jobs fail-closed. Inspect each worker using boot ID, process
   start, PID, and command identity. If safely absent/stopped, import as queued
   with the existing reconciliation semantics. If ownership or stopping is
   unsafe, preserve the JSON source and quarantine the import; do not attach the
   current boot to a legacy claim.
7. Import legacy runtime jobs exactly once. Same identity and revision with
   identical canonical content is idempotent; a newer durable row wins and the
   source is retained only until verified; same-revision disagreement or
   identity/created-at conflict is quarantined. Unsafe legacy worker claims
   remain blocked and their source is not deleted.
8. Insert rows, reservations, events, and import-status metadata in one
   transaction per bounded batch (or one transaction for the initial cutover).
   Commit a manifest containing source hashes, row counts, quarantine counts,
   and database schema version. Only after commit, rename imported job JSON to
   an archival suffix or move it to a migration backup; never destroy the
   backup as part of import.
9. Set the cutover marker atomically. New runtime opens use SQLite; a second
   open sees the completed manifest and performs no import. A partial import
   without a committed marker is discarded or resumed from the manifest while
   the JSON source remains authoritative.

### Backup and manual recovery

Stop the wake service and all detached workers before a manual backup or restore.
This drains writers and avoids copying a database while a delivery or release lease
is changing:

```bash
voice-harness services stop
state_home="${XDG_STATE_HOME:-$HOME/.local/state}"
jobs_root="$state_home/voice-harness/jobs"
cp -a "$jobs_root" "$jobs_root.backup-$(date +%Y%m%d-%H%M%S)"
```

For the systemd service, obtain the absolute `STATE_DIRECTORY` from
`systemctl --user show voice-harness-wake.service --property=Environment` and use
its `jobs` child instead. A complete backup includes `jobs.sqlite3`, any
`jobs.sqlite3-wal`/`jobs.sqlite3-shm` files present after shutdown, `.artifacts`,
`.quarantine`, and archived `*.json.imported`/`*.json.failed` sources. Do not back
up only the database: artifact bytes and quarantine payloads intentionally remain
file-backed.

`voice-harness doctor` reports the database path, schema version, migration status,
integrity result, and unresolved import failures without opening the store for
writes. If it reports an unreadable database or failed integrity check, leave the
services stopped, copy the entire jobs directory for forensics, and restore the
most recent complete directory backup. Do not delete WAL files, clear reservations,
mark results delivered, or copy selected rows between databases. Inspect unresolved
quarantine with `voice-harness jobs quarantine list`; acknowledge a reservation
only after verifying its worker, Herdr target, and worktree no longer exist.

### Rollback

Rollback is available only before SQLite has become the sole writer or after a
controlled drain. Stop the coordinator/effect executor, acquire the migration
lease, and verify no post-cutover effects are still running. Export SQLite rows
back to the original v12 structured JSON format only if every row has a
lossless representation; otherwise keep the original backup as the rollback
source and mark the database unusable. Restore the backed-up jobs directory,
legacy directory, quarantine evidence, and artifact references, then remove the
cutover marker and restart the prior JSON implementation.

Do not roll back by blindly copying SQLite over JSON: it can lose newly
delivered/undelivered results or resurrect external effects. Any effect with
`OutcomeUnknown` must be reconciled manually before rollback; outbox rows that
were executed are evidence, not permission to replay. If rollback is required
after new SQLite-only state exists, quarantine the database and retain it for
forensics, preserve all external resources, and require an operator decision
for each active reservation.

## Explicit non-goals

* No runtime implementation or `cursor/model.py` refactor in #139 or #341.
* No product-policy change for planning, review, plan approval, cancellation,
  fork confirmation, worktree retention, or delivery.
* No rewrite of wake, STT, TTS, audio, Herdr, GitHub, or provider integrations.
* No durable multi-ticket batch admission, UI/overlay work, or OpenCode
  implementation.
* No deletion of schema migration/compatibility code before #142 proves the
  one-shot importer and rollback path.
* No generic JSON blob remaining as the final source of truth after #142.
* No automatic deletion of quarantine evidence, generated worktrees, or
  undelivered terminals.
* No assumption that an external effect is absent because an observation failed.

## Open questions

1. Should #140 use a parity `payload_json` column temporarily, or split all 158
   scalar fields immediately while preserving the old adapter?
2. **Decided in #341.** Coordinator process: library-owned single SQLite writer;
   production long-lived owner is the wake daemon; no dedicated coordinator
   service.
3. **Decided in #341.** Outbox leasing: multiple leased executors with
   per-family concurrency; delivery executes in-process in the wake daemon.
4. **Decided in #341.** Provider idempotency: see the effect catalogue.
   Pane/session create and prompt submit are not replay-safe after unknown;
   fork/clone/worktree reconcile by observation; issue/ticket create use
   correlation markers and never replay.
5. Should events be retained forever, sampled, or pruned separately from job
   retention?
6. What is the authoritative artifact backup/restore policy, and should
   artifact bytes eventually move into a separate content store?
7. How should SQLite backups be coordinated with WAL checkpoints and active
   effect leases?
8. Which migration failure categories are operator-retryable versus permanent
   quarantine, and what CLI diagnostics are required?
9. Is a Rust coordinator worth the operational boundary once Python transition
   property tests and hardware recovery checks pass?

## Phased checklist

### #139 — inventory and design

- [x] Inventory v12 persisted fields, structured payloads, state transitions,
  locks, CAS, reservations, worker identity, delivery, quarantine, and artifacts.
- [x] Verify the Cursor job JSON write surface and document non-job JSON stores.
- [x] Define SQLite tables, constraints, submachines, outbox effects, migration,
  rollback, non-goals, and open questions.
- [x] Link the design to [#138](https://github.com/joshua-mo-143/local-voice-harness/issues/138)
  and its ordered implementation phases.

### #140 — SQLite parity

- [x] Add SQLite-backed `JobStore` behind the existing API; do not refactor
  `model.py` yet.
- [x] Preserve locks/maintenance behavior with a migration lease and diagnostics.
- [x] Implement one-shot import for durable JSON, legacy JSON, maintenance,
  quarantine, and artifact metadata.
- [x] Prove parity for CAS, reservations, delivery claims, follow-ups, active
  worker recovery, undelivered terminals, pruning, and quarantine.
- [x] Add backup, import-failure, rollback, and DB-path documentation.

### #141 — typed lifecycle and submachines

- [ ] Replace the optional-field lifecycle with typed top-level states.
- [ ] Extract checkout, agent ownership, prompt, question, and delivery
  submachines with explicit uncertainty states.
- [ ] Move cross-field invariants into transitions/types and retain SQL checks
  only for local relational facts.
- [ ] Preserve externally visible statuses and existing safety behavior.

### #357 — lifecycle ownership inventory

- [x] Inventory every schema-v18 field, typed runtime representation, transition
  owner, compatibility adapter, and production caller in
  `cursor/lifecycle_ownership.py`.
- [x] Record crash boundaries for identity, revision, token, timestamp, counter,
  uncertainty, and reconciliation fields.
- [x] Capture baseline counts and the two prompt-operation vocabularies plus
  checkout/session label collapse as duplicate authorities.
- [x] Sequence #358, then #359, then #360 because `model.py`, `provisioning.py`,
  and `recovery.py` overlap; #360 must follow #359.

### #142 — coordinator and outbox

- [x] [#341](https://github.com/joshua-mo-143/local-voice-harness/issues/341)
  catalogue every durable external effect, assign handler boundaries, and
  decide coordinator ownership, lease concurrency, and replay/reconcile rules.
- [x] [#342](https://github.com/joshua-mo-143/local-voice-harness/issues/342)
  persist `(state transition, event, reservations, outbox effects)` atomically.
- [ ] Make the coordinator the sole durable state writer.
- [x] [#345](https://github.com/joshua-mo-143/local-voice-harness/issues/345)
  execute named provider effects outside transactions with idempotency keys,
  leases, observations, and `OutcomeUnknown` recovery.
- [ ] Remove per-job JSON writes, compatibility layout, and v0–v11 incremental
  migration only after parity and hardware recovery checks pass.
- [ ] Re-evaluate the non-blocking Rust coordinator option after the Python
  model and effect boundary stabilize.
