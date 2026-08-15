from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from local_voice_harness import app, cli, config, vocabulary
from local_voice_harness.browser_context import RequestContext, github_issue_from_text
from local_voice_harness.intent import Intent, IntentRoute
from local_voice_harness.stt import server


class VocabularyStoreTests(unittest.TestCase):
    def _store(self, name: str = "vocabulary.json") -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name) / name

    def test_missing_store_loads_empty(self) -> None:
        store = vocabulary.load(self._store())
        self.assertEqual(store.replacements, ())
        self.assertEqual(store.aliases, ())

    def test_add_normalizes_and_persists_private_json(self) -> None:
        store = self._store()
        with redirect_stdout(io.StringIO()):
            vocabulary.add_replacement("  Herder   Tool ", "herdr tool", path=store)
            vocabulary.add_alias(
                "  The Harness  Repo ",
                "Joshua-MO-143/Local-Voice-Harness.git",
                path=store,
            )

        self.assertEqual(store.stat().st_mode & 0o777, 0o600)
        document = json.loads(store.read_text())
        self.assertEqual(document["version"], vocabulary.SCHEMA_VERSION)
        self.assertEqual(
            document["replacements"],
            [{"spoken": "Herder Tool", "written": "herdr tool"}],
        )
        self.assertEqual(
            document["aliases"],
            [
                {
                    "phrase": "the harness repo",
                    "target": "Joshua-MO-143/Local-Voice-Harness",
                    "kind": "repository",
                }
            ],
        )

    def test_issue_target_infers_issue_kind(self) -> None:
        store = self._store()
        with redirect_stdout(io.StringIO()):
            vocabulary.add_alias("harness bug", "owner/repo#35", path=store)
        alias = vocabulary.load(store).alias_for("harness bug")
        assert alias is not None
        self.assertEqual(alias.kind, vocabulary.AliasKind.ISSUE)
        self.assertEqual(alias.target, "owner/repo#35")

    def test_invalid_alias_target_is_rejected(self) -> None:
        store = self._store()
        with self.assertRaises(vocabulary.VocabularyError):
            vocabulary.add_alias("bad", "not a repository", path=store)

    def test_conflicting_add_is_rejected_without_force(self) -> None:
        store = self._store()
        with redirect_stdout(io.StringIO()):
            vocabulary.add_alias("harness", "owner/one", path=store)
        with self.assertRaises(vocabulary.VocabularyError):
            vocabulary.add_alias("harness", "owner/two", path=store)
        with redirect_stdout(io.StringIO()):
            vocabulary.add_alias("harness", "owner/two", path=store, force=True)
        alias = vocabulary.load(store).alias_for("harness")
        assert alias is not None
        self.assertEqual(alias.target, "owner/two")

    def test_identical_add_is_idempotent(self) -> None:
        store = self._store()
        with redirect_stdout(io.StringIO()):
            vocabulary.add_replacement("kubernetes", "Kubernetes", path=store)
            vocabulary.add_replacement("kubernetes", "Kubernetes", path=store)
        self.assertEqual(len(vocabulary.load(store).replacements), 1)

    def test_ambiguous_stored_file_is_rejected_on_load(self) -> None:
        store = self._store()
        store.write_text(
            json.dumps(
                {
                    "version": 1,
                    "aliases": [
                        {"phrase": "x", "target": "owner/one", "kind": "repository"},
                        {"phrase": "x", "target": "owner/two", "kind": "repository"},
                    ],
                }
            )
        )
        with self.assertRaises(vocabulary.VocabularyError):
            vocabulary.load(store)

    def test_unsupported_version_is_rejected(self) -> None:
        store = self._store()
        store.write_text(json.dumps({"version": 99}))
        with self.assertRaises(vocabulary.VocabularyError):
            vocabulary.load(store)

    def test_invalid_json_is_rejected(self) -> None:
        store = self._store()
        store.write_text("{not json")
        with self.assertRaises(vocabulary.VocabularyError):
            vocabulary.load(store)

    def test_remove_reports_missing_entries(self) -> None:
        store = self._store()
        with self.assertRaises(vocabulary.VocabularyError):
            vocabulary.remove_alias("absent", path=store)
        with self.assertRaises(vocabulary.VocabularyError):
            vocabulary.remove_replacement("absent", path=store)

    def test_remove_deletes_entries(self) -> None:
        store = self._store()
        with redirect_stdout(io.StringIO()):
            vocabulary.add_replacement("herder", "herdr", path=store)
            vocabulary.add_alias("harness", "owner/repo", path=store)
            vocabulary.remove_replacement("Herder", path=store)
            vocabulary.remove_alias("HARNESS", path=store)
        loaded = vocabulary.load(store)
        self.assertEqual(loaded.replacements, ())
        self.assertEqual(loaded.aliases, ())

    def test_list_reports_entries_and_emptiness(self) -> None:
        store = self._store()
        empty = io.StringIO()
        with redirect_stdout(empty):
            vocabulary.list_entries(path=store)
        self.assertIn("(none)", empty.getvalue())

        with redirect_stdout(io.StringIO()):
            vocabulary.add_alias("harness", "owner/repo", path=store)
        populated = io.StringIO()
        with redirect_stdout(populated):
            vocabulary.list_entries("alias", path=store)
        self.assertIn("owner/repo", populated.getvalue())

    def test_export_and_import_round_trip(self) -> None:
        store = self._store("original.json")
        backup = self._store("backup.json")
        with redirect_stdout(io.StringIO()):
            vocabulary.add_replacement("herder", "herdr", path=store)
            vocabulary.add_alias("harness", "owner/repo", path=store)
            vocabulary.export_entries(backup, path=store)

        printed = io.StringIO()
        with redirect_stdout(printed):
            vocabulary.export_entries(path=store)
        self.assertEqual(
            json.loads(printed.getvalue()), vocabulary.load(store).to_document()
        )

        destination = self._store("imported.json")
        with redirect_stdout(io.StringIO()):
            vocabulary.import_entries(backup, path=destination)
        self.assertEqual(
            vocabulary.load(destination).to_document(),
            vocabulary.load(store).to_document(),
        )

    def test_import_replace_overwrites_existing(self) -> None:
        store = self._store()
        backup = self._store("backup.json")
        with redirect_stdout(io.StringIO()):
            vocabulary.add_alias("keep", "owner/keep", path=store)
            vocabulary.add_alias("new", "owner/new", path=backup)
            vocabulary.import_entries(backup, replace=True, path=store)
        loaded = vocabulary.load(store)
        self.assertIsNone(loaded.alias_for("keep"))
        self.assertIsNotNone(loaded.alias_for("new"))

    def test_import_merge_prefers_incoming(self) -> None:
        store = self._store()
        backup = self._store("backup.json")
        with redirect_stdout(io.StringIO()):
            vocabulary.add_alias("harness", "owner/old", path=store)
            vocabulary.add_alias("harness", "owner/new", path=backup)
            vocabulary.import_entries(backup, path=store)
        alias = vocabulary.load(store).alias_for("harness")
        assert alias is not None
        self.assertEqual(alias.target, "owner/new")


class AliasResolutionTests(unittest.TestCase):
    def _store(self) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name) / "vocabulary.json"

    def test_resolution_prefers_the_longest_phrase(self) -> None:
        store = self._store()
        vocabulary.save(
            vocabulary.Vocabulary(
                aliases=(
                    vocabulary.Alias(
                        "harness", "owner/harness", vocabulary.AliasKind.REPOSITORY
                    ),
                    vocabulary.Alias(
                        "harness bug",
                        "owner/harness#35",
                        vocabulary.AliasKind.ISSUE,
                    ),
                )
            ),
            store,
        )
        resolved = vocabulary.resolve_aliases(
            "work on the harness bug in the harness", path=store
        )
        self.assertEqual(resolved, "work on the owner/harness#35 in the owner/harness")

    def test_resolution_is_case_insensitive_and_whole_word(self) -> None:
        store = self._store()
        vocabulary.save(
            vocabulary.Vocabulary(
                aliases=(
                    vocabulary.Alias(
                        "harness", "owner/harness", vocabulary.AliasKind.REPOSITORY
                    ),
                )
            ),
            store,
        )
        self.assertEqual(
            vocabulary.resolve_aliases("The HARNESS is ready", path=store),
            "The owner/harness is ready",
        )
        self.assertEqual(
            vocabulary.resolve_aliases("harnessing power", path=store),
            "harnessing power",
        )

    def test_resolution_tolerates_a_broken_store(self) -> None:
        store = self._store()
        store.write_text("{broken")
        self.assertEqual(
            vocabulary.resolve_aliases("leave me alone", path=store),
            "leave me alone",
        )

    def test_issue_alias_feeds_existing_reference_parsing(self) -> None:
        store = self._store()
        vocabulary.save(
            vocabulary.Vocabulary(
                aliases=(
                    vocabulary.Alias(
                        "harness bug",
                        "octo/harness#35",
                        vocabulary.AliasKind.ISSUE,
                    ),
                )
            ),
            store,
        )
        resolved = vocabulary.resolve_aliases("look at the harness bug", path=store)
        issue = github_issue_from_text(resolved)
        assert issue is not None
        self.assertEqual(issue.reference, "octo/harness#35")


class SpeechToTextCorrectionTests(unittest.TestCase):
    def _store(self) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name) / "vocabulary.json"

    def test_missing_store_preserves_static_replacements(self) -> None:
        store = self._store()
        with mock.patch.object(server.config, "VOCABULARY_PATH", store):
            self.assertEqual(server.normalize("use curser now"), "use Cursor now")

    def test_user_replacement_overrides_static_and_defaults(self) -> None:
        store = self._store()
        vocabulary.save(
            vocabulary.Vocabulary(
                replacements=(
                    vocabulary.Replacement("herder", "Herder-Custom"),
                    vocabulary.Replacement("kubernetes", "Kubernetes"),
                )
            ),
            store,
        )
        with mock.patch.object(server.config, "VOCABULARY_PATH", store):
            self.assertEqual(
                server.normalize("the herder runs kubernetes and curser"),
                "the Herder-Custom runs Kubernetes and Cursor",
            )

    def test_broken_store_falls_back_to_static_replacements(self) -> None:
        store = self._store()
        store.write_text("{broken")
        with mock.patch.object(server.config, "VOCABULARY_PATH", store):
            self.assertEqual(server.normalize("use curser"), "use Cursor")


class RoutingPrepassTests(unittest.TestCase):
    def _store(self) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name) / "vocabulary.json"

    def test_respond_resolves_aliases_before_routing(self) -> None:
        store = self._store()
        vocabulary.save(
            vocabulary.Vocabulary(
                aliases=(
                    vocabulary.Alias(
                        "the harness repo",
                        "owner/harness",
                        vocabulary.AliasKind.REPOSITORY,
                    ),
                )
            ),
            store,
        )
        captured: list[str] = []
        with (
            mock.patch.object(config, "VOCABULARY_PATH", store),
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app,
                "request_context",
                side_effect=lambda text, **_settings: (
                    captured.append(text) or RequestContext(text)
                ),
            ),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ) as route_intent,
            mock.patch.object(app, "qwen_response", return_value="ok"),
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("summarize the harness repo readme")

        self.assertEqual(captured, ["summarize owner/harness readme"])
        route_intent.assert_called_once_with(
            "summarize owner/harness readme",
            mock.ANY,
            cursor_session=None,
            pending_question=None,
            clarification_kind=None,
            settings=mock.ANY,
        )


class SpokenAliasPreparationTests(unittest.TestCase):
    def _store(self) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name) / "vocabulary.json"

    def test_call_this_repo_uses_focused_repository_not_uttered_target(self) -> None:
        store = self._store()
        preparation = vocabulary.prepare_spoken_alias(
            "Call this repo owner/stt-guess the harness",
            focused_repository="joshua-mo-143/local-voice-harness",
            focused_issue="joshua-mo-143/local-voice-harness#419",
            path=store,
        )

        self.assertEqual(preparation.status, vocabulary.SpokenAliasStatus.READY)
        pending = preparation.pending
        assert pending is not None
        self.assertEqual(pending.phrase, "owner/stt-guess the harness")
        self.assertEqual(pending.target, "joshua-mo-143/local-voice-harness")
        self.assertEqual(pending.kind, vocabulary.AliasKind.REPOSITORY)
        self.assertFalse(pending.replace)
        self.assertIsNone(vocabulary.load(store).alias_for("the harness"))

    def test_call_this_issue_uses_focused_issue_identity(self) -> None:
        preparation = vocabulary.prepare_spoken_alias(
            "call this the launcher issue",
            focused_repository="owner/repo",
            focused_issue="owner/repo#35",
            path=self._store(),
        )

        pending = preparation.pending
        assert pending is not None
        self.assertEqual(pending.phrase, "the launcher issue")
        self.assertEqual(pending.target, "owner/repo#35")
        self.assertEqual(pending.kind, vocabulary.AliasKind.ISSUE)

    def test_missing_phrase_or_target_fails_closed(self) -> None:
        store = self._store()
        missing_phrase = vocabulary.prepare_spoken_alias(
            "please alias this",
            focused_repository="owner/repo",
            path=store,
        )
        missing_target = vocabulary.prepare_spoken_alias(
            "Call this repo the harness",
            path=store,
        )
        uttered_only = vocabulary.prepare_spoken_alias(
            "Call this repo owner/stt-guess",
            path=store,
        )

        self.assertEqual(
            missing_phrase.status, vocabulary.SpokenAliasStatus.MISSING_PHRASE
        )
        self.assertEqual(
            missing_target.status, vocabulary.SpokenAliasStatus.MISSING_TARGET
        )
        self.assertEqual(
            uttered_only.status, vocabulary.SpokenAliasStatus.MISSING_TARGET
        )
        self.assertEqual(vocabulary.load(store).aliases, ())

    def test_issue_kind_without_issue_identity_fails_closed(self) -> None:
        preparation = vocabulary.prepare_spoken_alias(
            "call this issue the launcher",
            focused_repository="owner/repo",
            path=self._store(),
        )

        self.assertEqual(
            preparation.status, vocabulary.SpokenAliasStatus.MISSING_TARGET
        )
        self.assertIsNone(preparation.pending)

    def test_invalid_trusted_target_fails_closed(self) -> None:
        preparation = vocabulary.prepare_spoken_alias(
            "Call this repo the harness",
            focused_repository="not a repository",
            path=self._store(),
        )

        self.assertEqual(
            preparation.status, vocabulary.SpokenAliasStatus.INVALID_TARGET
        )

    def test_identical_alias_is_no_change(self) -> None:
        store = self._store()
        with redirect_stdout(io.StringIO()):
            vocabulary.add_alias("the harness", "owner/repo", path=store)
        preparation = vocabulary.prepare_spoken_alias(
            "Call this repo the harness",
            focused_repository="owner/repo",
            path=store,
        )

        self.assertEqual(preparation.status, vocabulary.SpokenAliasStatus.NO_CHANGE)
        self.assertIsNone(preparation.pending)

    def test_conflict_is_ready_but_not_written_until_replace(self) -> None:
        store = self._store()
        with redirect_stdout(io.StringIO()):
            vocabulary.add_alias("the harness", "owner/old", path=store)
        preparation = vocabulary.prepare_spoken_alias(
            "Call this repo the harness",
            focused_repository="owner/new",
            path=store,
        )

        pending = preparation.pending
        assert pending is not None
        self.assertEqual(pending.existing_target, "owner/old")
        self.assertFalse(pending.replace)
        existing = vocabulary.load(store).alias_for("the harness")
        assert existing is not None
        self.assertEqual(existing.target, "owner/old")

        spoken = vocabulary.render_spoken_alias_preparation(preparation)
        self.assertIn("the harness", spoken.spoken_text)
        self.assertIn("owner/new", spoken.spoken_text)
        self.assertNotIn("Replace it", spoken.spoken_text)

    def test_yes_commits_through_add_alias_and_no_keeps_store_empty(self) -> None:
        store = self._store()
        preparation = vocabulary.prepare_spoken_alias(
            "Call this repo the harness",
            focused_repository="owner/repo",
            path=store,
        )
        pending = preparation.pending
        assert pending is not None
        with redirect_stdout(io.StringIO()):
            vocabulary.commit_spoken_alias(pending, path=store)
        alias = vocabulary.load(store).alias_for("the harness")
        assert alias is not None
        self.assertEqual(alias.target, "owner/repo")
        self.assertEqual(alias.kind, vocabulary.AliasKind.REPOSITORY)

    def test_replace_confirmation_uses_force(self) -> None:
        store = self._store()
        with redirect_stdout(io.StringIO()):
            vocabulary.add_alias("the harness", "owner/old", path=store)
        pending = vocabulary.PendingSpokenAlias(
            trusted_utterance="Call this repo the harness",
            phrase="the harness",
            target="owner/new",
            kind=vocabulary.AliasKind.REPOSITORY,
            existing_target="owner/old",
            replace=True,
        )
        with redirect_stdout(io.StringIO()):
            vocabulary.commit_spoken_alias(pending, force=True, path=store)
        alias = vocabulary.load(store).alias_for("the harness")
        assert alias is not None
        self.assertEqual(alias.target, "owner/new")

    def test_spoken_route_does_not_add_replacements_or_pronunciations(self) -> None:
        store = self._store()
        preparation = vocabulary.prepare_spoken_alias(
            "Call this repo the harness",
            focused_repository="owner/repo",
            path=store,
        )
        pending = preparation.pending
        assert pending is not None
        with redirect_stdout(io.StringIO()):
            vocabulary.commit_spoken_alias(pending, path=store)
        loaded = vocabulary.load(store)
        self.assertEqual(loaded.replacements, ())
        self.assertEqual(loaded.pronunciations, ())
        self.assertEqual(len(loaded.aliases), 1)

    def test_parse_rejects_remember_that_and_empty_phrase(self) -> None:
        self.assertIsNone(vocabulary.parse_spoken_alias_request("remember that"))
        self.assertIsNone(vocabulary.parse_spoken_alias_request("call this repo"))
        self.assertIsNone(vocabulary.parse_spoken_alias_request("work on the harness"))


class VocabularyCliTests(unittest.TestCase):
    def test_add_replacement_parses_and_dispatches(self) -> None:
        arguments = cli.parser().parse_args(
            ["vocabulary", "add", "replacement", "herder", "herdr", "--force"]
        )
        self.assertEqual(arguments.vocabulary_command, "add")
        self.assertEqual(arguments.vocabulary_kind, "replacement")
        with mock.patch.object(cli, "add_replacement") as add_replacement:
            cli.dispatch(arguments)
        add_replacement.assert_called_once_with("herder", "herdr", force=True)

    def test_add_alias_dispatches(self) -> None:
        arguments = cli.parser().parse_args(
            ["vocabulary", "add", "alias", "the harness repo", "owner/repo"]
        )
        with mock.patch.object(cli, "add_alias") as add_alias:
            cli.dispatch(arguments)
        add_alias.assert_called_once_with("the harness repo", "owner/repo", force=False)

    def test_add_and_remove_pronunciation_dispatches(self) -> None:
        with mock.patch.object(cli, "add_pronunciation") as add_pronunciation:
            cli.dispatch(
                cli.parser().parse_args(
                    [
                        "vocabulary",
                        "add",
                        "pronunciation",
                        "Herdr",
                        "herder",
                        "--force",
                    ]
                )
            )
        add_pronunciation.assert_called_once_with("Herdr", "herder", force=True)

        with mock.patch.object(cli, "remove_pronunciation") as remove_pronunciation:
            cli.dispatch(
                cli.parser().parse_args(
                    ["vocabulary", "remove", "pronunciation", "Herdr"]
                )
            )
        remove_pronunciation.assert_called_once_with("Herdr")

    def test_remove_and_list_dispatch(self) -> None:
        with mock.patch.object(cli, "remove_alias") as remove_alias:
            cli.dispatch(
                cli.parser().parse_args(["vocabulary", "remove", "alias", "harness"])
            )
        remove_alias.assert_called_once_with("harness")

        with mock.patch.object(cli, "list_entries") as list_entries:
            cli.dispatch(
                cli.parser().parse_args(["vocabulary", "list", "--kind", "replacement"])
            )
        list_entries.assert_called_once_with("replacement")

    def test_export_and_import_dispatch(self) -> None:
        with mock.patch.object(cli, "export_entries") as export_entries:
            cli.dispatch(
                cli.parser().parse_args(
                    ["vocabulary", "export", "--output", "out.json"]
                )
            )
        export_entries.assert_called_once_with(Path("out.json"))

        with mock.patch.object(cli, "import_entries") as import_entries:
            cli.dispatch(
                cli.parser().parse_args(
                    ["vocabulary", "import", "backup.json", "--replace"]
                )
            )
        import_entries.assert_called_once_with(Path("backup.json"), replace=True)

    def test_remove_replacement_dispatch(self) -> None:
        with mock.patch.object(cli, "remove_replacement") as remove_replacement:
            cli.dispatch(
                cli.parser().parse_args(
                    ["vocabulary", "remove", "replacement", "herder"]
                )
            )
        remove_replacement.assert_called_once_with("herder")


if __name__ == "__main__":
    unittest.main()
