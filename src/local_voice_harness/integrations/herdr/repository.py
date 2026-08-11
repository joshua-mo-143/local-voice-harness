from __future__ import annotations

from pathlib import Path

from ...local_git import (
    LocalGitError,
    LocalGitOperationAmbiguous,
    LocalGitRepository,
)
from ..rofi import choose_repository, confirm_clone
from .types import (
    HOME_ROOT,
    Checkpoint,
    HerdrError,
    HerdrOperations,
    normalize_name,
    repository_name_from_url,
)


class HerdrRepository:
    """Local repository discovery, resolution, and cloning."""

    def __init__(self, operations: HerdrOperations) -> None:
        self._operations = operations

    @staticmethod
    def allowed_repository(path: Path) -> bool:
        try:
            path.relative_to(HOME_ROOT)
        except ValueError:
            return False
        return path != HOME_ROOT and (path / ".git").exists()

    def repository_roots(self) -> list[Path]:
        roots: dict[str, Path] = {}
        for workspace in self._operations.list_workspaces():
            root = (workspace.get("worktree") or {}).get("repo_root")
            if root:
                path = Path(str(root)).resolve()
                if self.allowed_repository(path):
                    roots[str(path)] = path
        try:
            children = HOME_ROOT.iterdir()
        except OSError:
            children = []
        for child in children:
            try:
                if (
                    not child.name.startswith(".")
                    and child.is_dir()
                    and (child / ".git").exists()
                ):
                    path = child.resolve()
                    if self.allowed_repository(path):
                        roots[str(path)] = path
            except OSError:
                continue
        return sorted(roots.values(), key=lambda path: path.name.casefold())

    def resolve_repository(
        self, hint: str | None, task: str, repositories: list[Path] | None = None
    ) -> tuple[Path | None, list[Path]]:
        repositories = self.repository_roots() if repositories is None else repositories
        if hint:
            candidate = Path(hint).expanduser()
            if candidate.is_absolute():
                resolved = candidate.resolve()
                exact = [path for path in repositories if path == resolved]
                return (exact[0], exact) if exact else (None, [])
            normalized = normalize_name(hint)
            exact = [p for p in repositories if normalize_name(p.name) == normalized]
            if len(exact) == 1:
                return exact[0], exact
            partial = [
                p
                for p in repositories
                if normalized
                and (
                    normalized in normalize_name(p.name)
                    or normalize_name(p.name) in normalized
                )
            ]
            return (partial[0], partial) if len(partial) == 1 else (None, partial)
        normalized_task = normalize_name(task)
        matches = [
            path
            for path in repositories
            if normalize_name(path.name) in normalized_task
        ]
        return (matches[0], matches) if len(matches) == 1 else (None, matches)

    def clone_repository(
        self, url: str, *, checkpoint: Checkpoint | None = None
    ) -> Path:
        name = repository_name_from_url(url)
        if name is None:
            raise HerdrError("Only Git HTTPS and SSH repository URLs are supported")
        destination = (HOME_ROOT / name).resolve()
        if destination.parent != HOME_ROOT:
            raise HerdrError(
                "Repository destination is outside the configured project root"
            )
        local_git = LocalGitRepository(
            clone_root=HOME_ROOT,
            allowed_root=HOME_ROOT,
            lock_name=".voice-harness-repository.lock",
        )
        try:
            return local_git.materialize(
                Path(name),
                clone_url=url,
                checkpoint=checkpoint,
            )
        except LocalGitOperationAmbiguous as exc:
            raise HerdrError(
                "Could not clone repository: command timed out; "
                "the clone outcome is ambiguous",
                code="repository_clone_ambiguous",
            ) from exc
        except LocalGitError as exc:
            raise HerdrError(str(exc)) from exc

    def choose_or_clone_repository(
        self,
        repositories: list[Path],
        *,
        checkpoint: Checkpoint | None = None,
    ) -> tuple[Path | None, str]:
        if checkpoint is not None:
            checkpoint()
        selected = choose_repository([path.name for path in repositories])
        if checkpoint is not None:
            checkpoint()
        if selected is None:
            return None, ""
        repository, _matches = self.resolve_repository(selected, "", repositories)
        if repository is not None:
            return repository, ""
        if repository_name_from_url(selected) is None:
            return None, "The Rofi selection was not a local repository or Git URL."
        if checkpoint is not None:
            checkpoint()
        confirmed = confirm_clone(selected)
        if checkpoint is not None:
            checkpoint()
        if not confirmed:
            return None, "Repository cloning was cancelled."
        try:
            return self.clone_repository(selected, checkpoint=checkpoint), ""
        except HerdrError as exc:
            return None, f"Repository cloning failed: {exc}."
