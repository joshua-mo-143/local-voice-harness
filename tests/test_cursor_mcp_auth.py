from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path

from local_voice_harness.integrations.herdr.cursor_auth import (
    CursorMcpAuthError,
    CursorMcpAuthLinker,
    cursor_project_id,
)


class CursorMcpAuthLinkerTests(unittest.TestCase):
    def _source(self, root: Path) -> tuple[Path, Path, Path]:
        source = root / "authenticated checkout"
        source.mkdir()
        projects = root / "cursor-projects"
        auth = projects / cursor_project_id(source.resolve()) / "mcp-auth.json"
        auth.parent.mkdir(parents=True)
        auth.write_text('{"linear": "secret-test-token"}\n')
        auth.chmod(0o600)
        return source, projects, auth

    def test_linux_workspace_path_maps_to_cursor_project_id(self) -> None:
        self.assertEqual(
            cursor_project_id(
                Path(
                    "/home/joshuam/.herdr/worktrees/local-voice-harness/"
                    "test-multi-ticket"
                )
            ),
            ("home-joshuam-herdr-worktrees-local-voice-harness-test-multi-ticket"),
        )
        cases = {
            "/home/user/My Workspace": "home-user-My-Workspace",
            "/home/user/...hidden///ticket": "home-user-hidden-ticket",
            "/home/user/a__b--c..d": "home-user-a-b-c-d",
        }
        for workspace, expected in cases.items():
            with self.subTest(workspace=workspace):
                self.assertEqual(cursor_project_id(Path(workspace)), expected)

        with self.assertRaisesRegex(CursorMcpAuthError, "absolute"):
            cursor_project_id(Path("relative"))
        with self.assertRaisesRegex(CursorMcpAuthError, "filesystem root"):
            cursor_project_id(Path("/"))

    def test_distinct_workspace_collision_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "shared.hidden"
            target = root / "shared hidden"
            source.mkdir()
            target.mkdir()
            projects = root / "projects"
            auth = projects / cursor_project_id(source.resolve()) / "mcp-auth.json"
            auth.parent.mkdir(parents=True)
            auth.write_text("{}")
            auth.chmod(0o600)

            with self.assertRaisesRegex(CursorMcpAuthError, "same project ID"):
                CursorMcpAuthLinker(source, projects_root=projects).link(target)

            self.assertFalse(auth.is_symlink())

    def test_source_must_exist_as_owner_only_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, projects, auth = self._source(root)
            linker = CursorMcpAuthLinker(source, projects_root=projects)

            auth.chmod(0o640)
            with self.assertRaisesRegex(CursorMcpAuthError, "0600"):
                linker.validated_source()

            auth.chmod(0o600)
            moved = auth.with_suffix(".real")
            auth.rename(moved)
            auth.symlink_to(moved)
            with self.assertRaisesRegex(CursorMcpAuthError, "regular file"):
                linker.validated_source()

    def test_target_is_atomically_linked_without_copying_project_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, projects, source_auth = self._source(root)
            target = root / "ticket-worktree"
            target.mkdir()
            target_directory = projects / cursor_project_id(target.resolve())
            target_directory.mkdir()
            unrelated = target_directory / "transcripts.json"
            unrelated.write_text("keep me")
            target_auth = target_directory / "mcp-auth.json"
            target_auth.write_text("stale")

            linked = CursorMcpAuthLinker(source, projects_root=projects).link(target)

            self.assertEqual(linked, target_auth)
            self.assertTrue(linked.is_symlink())
            self.assertEqual(linked.resolve(), source_auth)
            self.assertEqual(unrelated.read_text(), "keep me")
            self.assertEqual(stat.S_IMODE(target_directory.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(source_auth.stat().st_mode), 0o600)
            self.assertFalse(
                any(
                    path.name.startswith(".mcp-auth.json.")
                    for path in target_directory.iterdir()
                )
            )

    def test_source_workspace_is_not_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, projects, source_auth = self._source(root)

            linked = CursorMcpAuthLinker(source, projects_root=projects).link(source)

            self.assertEqual(linked, source_auth)
            self.assertFalse(source_auth.is_symlink())
            self.assertEqual(os.lstat(source_auth).st_nlink, 1)


if __name__ == "__main__":
    unittest.main()
