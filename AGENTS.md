# Verification

After completing code changes, always run the local equivalents of every CI check before handing off. Run formatting checks, linting, type checking, and the full test suite for every Python version in the CI matrix. Do not claim completion until those checks pass, or clearly report any check that could not be run and why.

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
