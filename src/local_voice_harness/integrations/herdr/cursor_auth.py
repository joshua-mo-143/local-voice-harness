"""Scoped sharing of Cursor's workspace-local MCP OAuth state."""

from __future__ import annotations

import os
import re
import stat
import uuid
from pathlib import Path

MCP_AUTH_FILENAME = "mcp-auth.json"


class CursorMcpAuthError(ValueError):
    """The configured Cursor MCP auth source or target is unsafe."""


def cursor_project_id(workspace: Path) -> str:
    """Return Cursor's Linux project key for one absolute workspace path.

    Cursor joins ASCII-alphanumeric runs from the absolute path with one hyphen.
    Thus separators, spaces, dots (including hidden-directory prefixes), and
    repeated punctuation all normalize to one delimiter. This mapping is not
    injective; callers comparing multiple workspaces must reject collisions.
    """

    if not workspace.is_absolute():
        raise CursorMcpAuthError("Cursor MCP auth workspace must be an absolute path")
    identifier = re.sub(r"[^A-Za-z0-9]+", "-", workspace.as_posix()).strip("-")
    if not identifier:
        raise CursorMcpAuthError(
            "Cursor MCP auth workspace must not be the filesystem root"
        )
    return identifier


class CursorMcpAuthLinker:
    """Link only MCP OAuth state into explicitly trusted Cursor workspaces."""

    def __init__(
        self,
        source_workspace: Path,
        *,
        projects_root: Path | None = None,
    ) -> None:
        self.source_workspace = source_workspace.expanduser()
        self.projects_root = (
            projects_root or Path.home() / ".cursor" / "projects"
        ).expanduser()

    def _workspace(self, workspace: Path, *, label: str) -> Path:
        expanded = workspace.expanduser()
        if not expanded.is_absolute():
            raise CursorMcpAuthError(f"{label} must be an absolute path")
        try:
            resolved = expanded.resolve(strict=True)
        except OSError as exc:
            raise CursorMcpAuthError(f"{label} does not exist") from exc
        if not resolved.is_dir():
            raise CursorMcpAuthError(f"{label} must be a directory")
        return resolved

    def _auth_path(self, workspace: Path) -> Path:
        root = self.projects_root.resolve()
        return root / cursor_project_id(workspace) / MCP_AUTH_FILENAME

    def validated_source(self) -> tuple[Path, Path]:
        """Return the source checkout and auth file after metadata-only checks."""

        workspace = self._workspace(
            self.source_workspace, label="Cursor MCP auth source workspace"
        )
        auth_path = self._auth_path(workspace)
        try:
            metadata = auth_path.lstat()
        except FileNotFoundError as exc:
            raise CursorMcpAuthError(
                "Cursor MCP auth source has no mcp-auth.json; authenticate Linear "
                "from that workspace first"
            ) from exc
        except OSError as exc:
            raise CursorMcpAuthError("Cursor MCP auth source is unreadable") from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise CursorMcpAuthError(
                "Cursor MCP auth source must be a regular file, not a link"
            )
        if metadata.st_uid != os.geteuid():
            raise CursorMcpAuthError(
                "Cursor MCP auth source must be owned by the harness user"
            )
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise CursorMcpAuthError(
                "Cursor MCP auth source must have owner-only permissions (0600)"
            )
        return workspace, auth_path

    def link(self, target_workspace: Path) -> Path:
        """Atomically point one target workspace at the validated source file."""

        source_workspace, source_auth = self.validated_source()
        target_workspace = self._workspace(
            target_workspace, label="Cursor MCP auth target workspace"
        )
        if target_workspace == source_workspace:
            return source_auth

        source_id = cursor_project_id(source_workspace)
        target_id = cursor_project_id(target_workspace)
        if target_id == source_id:
            raise CursorMcpAuthError(
                "distinct Cursor MCP auth workspaces normalize to the same project ID"
            )

        target_auth = self._auth_path(target_workspace)
        target_directory = target_auth.parent
        target_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory_metadata = target_directory.lstat()
        if (
            not stat.S_ISDIR(directory_metadata.st_mode)
            or directory_metadata.st_uid != os.geteuid()
        ):
            raise CursorMcpAuthError(
                "Cursor MCP auth target directory must be a user-owned directory"
            )
        os.chmod(target_directory, 0o700)

        try:
            if target_auth.is_symlink() and target_auth.resolve() == source_auth:
                return target_auth
        except OSError:
            pass

        temporary = target_directory / f".{MCP_AUTH_FILENAME}.{uuid.uuid4().hex}.tmp"
        try:
            temporary.symlink_to(source_auth)
            os.replace(temporary, target_auth)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return target_auth
