# Implementation discipline

Before editing:

- State the behavioral invariant being changed and where production code currently owns it.
- Trace the relevant data from input to every affected boundary and output. Search callers, adapters, compatibility facades, legacy paths, and distinct channels such as speech, display, logs, and model context.
- Identify mixed, ambiguous, partial-success, stale-state, and failure cases that could falsify the proposed fix.

During implementation:

- Prefer the smallest change to the owning production path that satisfies the acceptance criteria.
- Preserve information and safety properties across boundaries; do not substitute one channel or representation for another without verifying that contract.
- Do not add APIs, helpers, files, or abstractions unless production code uses them or correctness requires them. If the requested production path does not exist, report that mismatch instead of building speculative infrastructure.

Before handing off:

- Map every acceptance criterion to production code and a test or explicit verification.
- Keep tests proportional, but include the adversarial combinations identified during discovery.
- Review the complete diff independently of the implementation plan. Audit every new symbol for a real caller and remove dead, duplicated, or speculative code.
- Treat passing tests as necessary, not sufficient. Run relevant checkout-based behavioral smoke tests and report residual risks or unverified behavior before claiming completion.

# Verification

After completing code changes, always run the local equivalents of every CI check before handing off. Run formatting checks, linting, type checking, and the full test suite for every Python version in the CI matrix. Do not claim completion until those checks pass, or clearly report any check that could not be run and why.

Use the repository-local development launcher (`scripts/dev.sh`) for relevant behavioral smoke tests so changes are exercised in the current checkout instead of relying solely on the automated test matrix. Follow `docs/development.md` for supported commands, isolation boundaries, and wake-listener safety. Before starting a smoke test that requires user-provided audio through an input source, notify the user and pause until they explicitly acknowledge that they are ready.

# Commits

Use [Angular conventional commits](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) for every commit message.

Format: `<type>(<optional scope>): <short summary>`

Common types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `ci`, `build`, `perf`.

Examples:

- `fix: harden recording and STT ownership`
- `refactor(wake): strengthen agent boundaries`
- `test: add runtime verification foundation`

Use an imperative, present-tense summary. Do not end the subject line with a period.

# Pull requests

When opening a PR, link it back to the issue it addresses. Include a closing keyword and issue number in the PR body, for example:

- `Closes #26`
- `Fixes #42`

Use `Closes` when the PR fully resolves the issue. Use `Fixes` for bug fixes. If a PR only partially addresses an issue, reference it without a closing keyword (for example, `Related to #28`).
