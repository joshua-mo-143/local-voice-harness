# Agent harness contract

`local_voice_harness.agents.AgentHarness` is the stable provider-neutral boundary
between durable jobs and a coding-agent transport. It covers:

1. creating a provider session in an already allocated launch context;
2. submitting a task with a session and state-sequence fence;
3. streaming typed terminal, clarification, cancellation, and failure events;
4. replying to a clarification on the same fenced session;
5. cancelling that exact session; and
6. read-only reconciliation after a worker or service restart.

Repository selection, checkout validation, workspace and pane ownership, worktree
allocation, reservation policy, and cleanup remain outside this contract. For the
Cursor/Herdr implementation those responsibilities remain in `HerdrWorkspace`;
the contract receives only opaque `pane_id` and `workspace_id` launch context.

## Capabilities

Optional behavior is advertised through `HarnessCapability`. Callers must request
capabilities during preflight rather than discovering missing behavior after a
side effect. The common `require_capabilities` check raises
`UnsupportedCapabilityError` with the harness and missing capability before any
session or task command runs. Cursor/Herdr currently advertises clarification
replies, cancellation, MCP connectors, and recovery.

## Durable identity and recovery

A durable identity is the tuple `(provider, session_id, target)`. `target` is an
address and is not sufficient identity: Herdr may reuse a target for a different
Cursor session. Durable jobs must also persist the observed state sequence before
submission. Every task and clarification reply supplies the expected session ID;
a changed ID fails closed.

The submit boundary has two durable hooks. `before_submit(sequence)` records that
the external call is about to cross, and `accepted()` records positive sequence
evidence that the provider accepted it. A crash between those facts is ambiguous.
On restart, `reconcile()` performs a read-only observation and reports `active`,
`settled`, `missing`, `changed`, or `unknown`. Reconciliation never authorizes a
blind replay:

- the same session with a changed sequence is positive acceptance evidence;
- a changed session, missing identity, or failed observation is not replay proof;
- an ambiguous submission remains fenced for manual reconciliation;
- a settled same-session task may resume event observation without resubmission.

Session identity is provider-issued and remains valid only while reconciliation
continues to observe the same provider and session ID. It does not claim that a
workspace, pane, or worktree is still owned; those bindings are independently
validated by workspace policy before reuse or cleanup.

## Cursor/Herdr mapping

`HerdrSession` implements `AgentHarness`. Session creation maps to
`herdr agent start`; submission maps to the existing sequence-fenced
`herdr agent prompt`; event streaming observes status plus bounded terminal
output; cancellation interrupts and confirms the agent stopped; reconciliation
uses `herdr agent get`. Existing prompt callbacks, interactive-questionnaire
checks, quiet-period completion, ownership validation, and manual ambiguity
fences remain in force.
