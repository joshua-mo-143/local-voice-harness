---
name: write-tickets
description: >-
  Drafts and files GitHub issues and Linear tickets as agent-to-agent contracts
  after investigating the repo. Use when creating, writing, filing, drafting,
  splitting, or converting GitHub issues, Linear tickets, epics, or issue bodies.
---

# Write tickets

Josh files work by asking Cursor to write the ticket. The body is the entire brief for a later agent that has no chat history and may be a different model. Write a contract, not a diary and not a design doc.

Follow `.cursor/rules/issue-scope.mdc` for epic splits, dependency links, and one-PR scope. Do not restate it.

This skill is for Cursor-authored tickets after investigation. Voice-harness drafts must stay faithful to the spoken request and must not invent acceptance criteria.

## Investigate first

Do not file from the user's one-liner. Before drafting:

1. Read the relevant code, tests, docs, logs, or failing job state.
2. Search existing issues for duplicates and related work (`gh issue list` / Linear).
3. Identify the current owner of the behavior and what a later agent would otherwise miss.

If investigation cannot pin the outcome, file a spike or a problem ticket with explicit open questions. Do not invent a 12-item design to look complete.

## What is binding

- **Problem**: what is wrong or missing, with evidence (error string, job id, file, command, reproduction).
- **Outcome**: observable behavior when the issue is done.
- **Acceptance criteria**: testable, behavioral, few. Each item is something a reviewer can check without sharing the author's intent.
- **Non-goals**: adjacent work, named by issue number when it exists.
- **Related / blocked by**: real issue numbers, not prose sequencing.

Do not restate always-on verification from `AGENTS.md` (format, lint, types, full CI matrix). Add only issue-specific tests, smokes, or hardware checks.

## Scope and acceptance gate

Before filing an implementation ticket, verify:

- It has one primary outcome and one coherent ownership/review boundary.
- It has 2–5 observable, pass/fail acceptance criteria.
- Criteria describe required behavior, not suggested implementation steps.
- Every criterion is necessary for the stated outcome.
- No criterion is independently deliverable as a useful change. If one is, split it into its own issue.

Epics use completion conditions rather than implementation acceptance criteria.
Spikes use explicit questions or deliverables that determine when the investigation
is complete.

Treat a brief that combines several of `introduce`, `migrate`, `enforce`,
`remove`, and `prove` as an epic candidate. If a split would create more than
three child issues, present the proposed issue tree and sequencing to the user
before filing any of them.

## What is not binding

Put likely files, APIs, and implementation steps under **Suggested approach**, and mark that section non-binding.

Promote a mechanism into acceptance criteria only when the mechanism *is* the invariant (isolation, fail-closed, no credential sharing, no shared `.venv` mutation). "Replace the 200 ms sleep" and "add a lightweight entrypoint" are suggestions unless the user required that shape.

## Shape

Title: imperative, specific, no repository name. `Epic:` prefix only for coordination parents.

Body template (omit empty sections):

```markdown
## Problem

<what fails, with evidence>

## Outcome

<observable behavior when done>

## Acceptance criteria

- <checkable behavior>
- <issue-specific test or smoke, if any>

## Non-goals

- <adjacent work, with issue numbers>

## Suggested approach

Non-binding. <files or likely path found during investigation>

## Related

- #<n>
```

Epics: outcome, child list, sequencing via dependency links, MVP boundary. No implementation acceptance criteria on the epic itself.

## Agent-to-agent rules

- Assume the implementer never saw this chat. No "as discussed", "the user wants", or "see above".
- Include the smoking gun you actually found. Do not cite internal vocabulary unless it is load-bearing (`speakable_label` is; a class you hope to add is not).
- Bound the blast radius so a later agent does not absorb a related issue.
- Prefer one focused issue. If there are multiple independently deliverable outcomes, stop and split into an epic plus children instead of writing a mega-ticket.
- If the user asked only to draft, show the title and body and wait. If they asked to file or create, open it with `gh issue create` or Linear `save_issue`, then return the URL.

## Anti-patterns

- Prescribing architecture in acceptance criteria.
- Padding criteria with repo-wide CI boilerplate.
- Filing without reading the code.
- Encoding this session's plan as if it were already decided.
- Writing a private note that only the author could implement.
