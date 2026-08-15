from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest import mock

from local_voice_harness import app, cli, config, vocabulary
from local_voice_harness.browser_context import RequestContext, github_issue_from_text
from local_voice_harness.cursor import service
from local_voice_harness.errors import HarnessError
from local_voice_harness.integrations.registry import (
    build_integration_registry,
    extract_issue_reference,
)
from local_voice_harness.intent import Intent, IntentRoute
from local_voice_harness.stt import server
from local_voice_harness.user_config import IntegrationSettings, default_user_config


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
                    "source": "github",
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
        self.assertEqual(alias.source, "github")

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


class LinearAliasTargetTests(unittest.TestCase):
    def _store(self) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name) / "vocabulary.json"

    def _linear(self) -> IntegrationSettings:
        return IntegrationSettings(github_enabled=True, linear_enabled=True)

    def test_cli_add_stores_canonical_linear_identifier(self) -> None:
        store = self._store()
        with redirect_stdout(io.StringIO()):
            vocabulary.add_alias(
                "launcher ticket",
                "API-79",
                path=store,
                integrations=self._linear(),
            )
        alias = vocabulary.load(store).alias_for("launcher ticket")
        assert alias is not None
        self.assertEqual(alias.target, "API-79")
        self.assertEqual(alias.kind, vocabulary.AliasKind.LINEAR)
        self.assertEqual(alias.source, "linear")
        document = json.loads(store.read_text())
        self.assertEqual(document["aliases"][0]["kind"], "linear")
        self.assertEqual(document["aliases"][0]["source"], "linear")

    def test_linear_target_uses_team_syntax_and_positive_number(self) -> None:
        store = self._store()
        with redirect_stdout(io.StringIO()):
            vocabulary.add_alias(
                "launcher ticket",
                "api-79",
                path=store,
                integrations=self._linear(),
            )
        alias = vocabulary.load(store).alias_for("launcher ticket")
        assert alias is not None
        self.assertEqual(alias.target, "API-79")
        for invalid in ("API", "API-", "1-79", "-79"):
            with self.subTest(target=invalid):
                with self.assertRaises(vocabulary.VocabularyError):
                    vocabulary.add_alias(
                        "bad",
                        invalid,
                        path=store,
                        integrations=self._linear(),
                    )

    def test_load_conflict_and_force_match_github_aliases(self) -> None:
        store = self._store()
        with redirect_stdout(io.StringIO()):
            vocabulary.add_alias(
                "launcher ticket",
                "API-79",
                path=store,
                integrations=self._linear(),
            )
        with self.assertRaises(vocabulary.VocabularyError):
            vocabulary.add_alias(
                "launcher ticket",
                "ENG-12",
                path=store,
                integrations=self._linear(),
            )
        with redirect_stdout(io.StringIO()):
            vocabulary.add_alias(
                "launcher ticket",
                "ENG-12",
                path=store,
                force=True,
                integrations=self._linear(),
            )
        alias = vocabulary.load(store).alias_for("launcher ticket")
        assert alias is not None
        self.assertEqual(alias.target, "ENG-12")

        store.write_text(
            json.dumps(
                {
                    "version": 2,
                    "aliases": [
                        {"phrase": "x", "target": "API-79", "kind": "linear"},
                        {"phrase": "x", "target": "ENG-12", "kind": "linear"},
                    ],
                }
            )
        )
        with self.assertRaises(vocabulary.VocabularyError):
            vocabulary.load(store)

    def test_resolve_aliases_substitutes_stored_linear_identifier(self) -> None:
        store = self._store()
        with redirect_stdout(io.StringIO()):
            vocabulary.add_alias(
                "launcher ticket",
                "API-79",
                path=store,
                integrations=self._linear(),
            )
        self.assertEqual(
            vocabulary.resolve_aliases("work on the launcher ticket", path=store),
            "work on the API-79",
        )

    def test_github_aliases_remain_unchanged(self) -> None:
        store = self._store()
        with redirect_stdout(io.StringIO()):
            vocabulary.add_alias("harness", "owner/repo", path=store)
            vocabulary.add_alias("harness bug", "owner/repo#35", path=store)
            vocabulary.add_alias(
                "launcher ticket",
                "API-79",
                path=store,
                integrations=self._linear(),
            )
        loaded = vocabulary.load(store)
        repo = loaded.alias_for("harness")
        issue = loaded.alias_for("harness bug")
        linear_alias = loaded.alias_for("launcher ticket")
        assert repo is not None and issue is not None and linear_alias is not None
        self.assertEqual(repo.kind, vocabulary.AliasKind.REPOSITORY)
        self.assertEqual(issue.kind, vocabulary.AliasKind.ISSUE)
        self.assertEqual(linear_alias.kind, vocabulary.AliasKind.LINEAR)
        self.assertEqual(repo.source, "github")
        self.assertEqual(linear_alias.source, "linear")

    def test_vocabulary_module_does_not_import_provider_modules(self) -> None:
        source = Path(vocabulary.__file__).read_text(encoding="utf-8")
        self.assertNotIn("LinearIntegration", source)
        self.assertNotIn("ZendeskProvider", source)
        self.assertNotIn("GitHubProvider", source)
        self.assertNotIn("integrations.linear", source)
        self.assertNotIn("integrations.zendesk", source)
        self.assertNotIn("integrations.github", source)
        self.assertIn("integrations.registry", source)

    def test_disabled_linear_fails_like_a_typed_identifier(self) -> None:
        store = self._store()
        with redirect_stdout(io.StringIO()):
            vocabulary.add_alias(
                "launcher ticket",
                "API-79",
                path=store,
                integrations=self._linear(),
            )
        expanded = vocabulary.resolve_aliases("work on launcher ticket", path=store)
        disabled = IntegrationSettings(linear_enabled=False)
        self.assertEqual(expanded, "work on API-79")
        self.assertIsNone(extract_issue_reference(expanded, disabled))
        self.assertIsNone(extract_issue_reference("work on API-79", disabled))
        registry = build_integration_registry(
            replace(default_user_config(), integrations=disabled)
        )
        with (
            mock.patch.object(service, "_job_store") as store_mock,
            mock.patch.object(service, "launch_worker"),
            self.assertRaisesRegex(HarnessError, "provider is unavailable"),
        ):
            service.start_job(expanded, issue_key="API-79", integrations=registry)
        store_mock.return_value.create.assert_not_called()

    def test_spoken_alias_path_rejects_linear_when_provider_is_disabled(self) -> None:
        preparation = vocabulary.prepare_spoken_alias(
            "call this the launcher issue",
            focused_issue="API-79",
            source="linear",
            path=self._store(),
            integrations=IntegrationSettings(github_enabled=True, linear_enabled=False),
        )
        self.assertEqual(
            preparation.status, vocabulary.SpokenAliasStatus.INVALID_TARGET
        )
        self.assertIsNone(preparation.pending)


class ProviderAliasIdentityTests(unittest.TestCase):
    def _store(self) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name) / "vocabulary.json"

    def test_cli_add_is_rejected_when_no_enabled_provider_owns_target(self) -> None:
        store = self._store()
        with self.assertRaisesRegex(vocabulary.VocabularyError, "no enabled provider"):
            vocabulary.add_alias(
                "launcher ticket",
                "API-79",
                path=store,
                integrations=IntegrationSettings(
                    github_enabled=True, linear_enabled=False
                ),
            )
        with self.assertRaisesRegex(vocabulary.VocabularyError, "no enabled provider"):
            vocabulary.add_alias(
                "harness",
                "owner/repo",
                path=store,
                integrations=IntegrationSettings(github_enabled=False),
            )
        self.assertEqual(vocabulary.load(store).aliases, ())

    def test_legacy_untagged_github_and_linear_aliases_still_load(self) -> None:
        store = self._store()
        store.write_text(
            json.dumps(
                {
                    "version": 2,
                    "aliases": [
                        {
                            "phrase": "harness",
                            "target": "owner/repo",
                            "kind": "repository",
                        },
                        {
                            "phrase": "harness bug",
                            "target": "owner/repo#35",
                            "kind": "issue",
                        },
                        {
                            "phrase": "launcher ticket",
                            "target": "API-79",
                            "kind": "linear",
                        },
                    ],
                }
            )
        )
        loaded = vocabulary.load(store)
        repo = loaded.alias_for("harness")
        issue = loaded.alias_for("harness bug")
        linear_alias = loaded.alias_for("launcher ticket")
        assert repo is not None and issue is not None and linear_alias is not None
        self.assertEqual(repo.source, "github")
        self.assertEqual(issue.source, "github")
        self.assertEqual(linear_alias.source, "linear")
        self.assertEqual(repo.target, "owner/repo")
        self.assertEqual(issue.target, "owner/repo#35")
        self.assertEqual(linear_alias.target, "API-79")
        self.assertEqual(
            vocabulary.resolve_aliases(
                "work on harness and the launcher ticket", path=store
            ),
            "work on owner/repo and the API-79",
        )

    def test_tagged_alias_load_preserves_provider_metadata_without_reparsing(
        self,
    ) -> None:
        store = self._store()
        store.write_text(
            json.dumps(
                {
                    "version": 3,
                    "aliases": [
                        {
                            "phrase": "future ticket",
                            "source": "future-provider",
                            "target": "opaque:v1:42",
                            "kind": "issue",
                        }
                    ],
                }
            )
        )
        alias = vocabulary.load(store).alias_for("future ticket")
        assert alias is not None
        self.assertEqual(alias.source, "future-provider")
        self.assertEqual(alias.target, "opaque:v1:42")
        self.assertEqual(alias.kind, vocabulary.AliasKind.ISSUE)

    def test_saved_provider_alias_reloads_without_provider_syntax_knowledge(
        self,
    ) -> None:
        class FutureProvider:
            name = "future-provider"

            @staticmethod
            def owns_issue_reference(reference: str) -> bool:
                return reference.strip().casefold().startswith("opaque:")

            @staticmethod
            def canonicalize_issue_reference(reference: str) -> str:
                return reference.strip().casefold()

        store = self._store()
        with (
            mock.patch.object(
                vocabulary,
                "enabled_integrations",
                return_value=(FutureProvider(),),
            ),
            redirect_stdout(io.StringIO()),
        ):
            vocabulary.add_alias("future ticket", "OPAQUE:V1:42", path=store)

        alias = vocabulary.load(store).alias_for("future ticket")
        assert alias is not None
        self.assertEqual(alias.source, "future-provider")
        self.assertEqual(alias.target, "opaque:v1:42")
        self.assertEqual(alias.kind, vocabulary.AliasKind.ISSUE)

    def test_provider_canonical_value_must_still_be_owned_before_save(self) -> None:
        class InvalidCanonicalProvider:
            name = "linear"

            @staticmethod
            def owns_issue_reference(reference: str) -> bool:
                return reference == "spoken API zero"

            @staticmethod
            def canonicalize_issue_reference(_reference: str) -> str:
                return "API-0"

        store = self._store()
        with (
            mock.patch.object(
                vocabulary,
                "enabled_integrations",
                return_value=(InvalidCanonicalProvider(),),
            ),
            self.assertRaisesRegex(vocabulary.VocabularyError, "no enabled provider"),
        ):
            vocabulary.add_alias("invalid", "spoken API zero", path=store)
        self.assertEqual(vocabulary.load(store).aliases, ())

    def test_multiple_provider_owners_are_rejected_without_registry_precedence(
        self,
    ) -> None:
        class Provider:
            def __init__(self, name: str) -> None:
                self.name = name

            @staticmethod
            def owns_issue_reference(reference: str) -> bool:
                return reference == "shared:42"

            @staticmethod
            def canonicalize_issue_reference(reference: str) -> str:
                return reference

        providers = (Provider("first"), Provider("second"))
        with (
            mock.patch.object(
                vocabulary,
                "enabled_integrations",
                return_value=providers,
            ),
            self.assertRaisesRegex(vocabulary.VocabularyError, "multiple enabled"),
        ):
            vocabulary.resolve_owned_alias_target("shared:42")

        with mock.patch.object(
            vocabulary,
            "enabled_integrations",
            return_value=providers,
        ):
            self.assertEqual(
                vocabulary.resolve_owned_alias_target("shared:42", source="second"),
                ("shared:42", "second", vocabulary.AliasKind.ISSUE),
            )

    def test_tagged_zendesk_alias_loads_when_source_matches_target(self) -> None:
        store = self._store()
        store.write_text(
            json.dumps(
                {
                    "version": 3,
                    "aliases": [
                        {
                            "phrase": "the help ticket",
                            "source": "zendesk",
                            "target": "Help#42",
                            "kind": "issue",
                        }
                    ],
                }
            )
        )
        alias = vocabulary.load(store).alias_for("the help ticket")
        assert alias is not None
        self.assertEqual(alias.source, "zendesk")
        self.assertEqual(alias.target, "Help#42")
        self.assertEqual(alias.kind, vocabulary.AliasKind.ISSUE)

    def test_voice_add_copies_fragment_source_and_issue_reference(self) -> None:
        store = self._store()
        linear = IntegrationSettings(github_enabled=True, linear_enabled=True)
        preparation = vocabulary.prepare_spoken_alias(
            "call this the launcher issue",
            focused_issue="API-79",
            source="linear",
            path=store,
            integrations=linear,
        )
        pending = preparation.pending
        assert pending is not None
        self.assertEqual(pending.source, "linear")
        self.assertEqual(pending.target, "API-79")
        self.assertEqual(pending.kind, vocabulary.AliasKind.LINEAR)
        with redirect_stdout(io.StringIO()):
            vocabulary.commit_spoken_alias(pending, path=store, integrations=linear)
        alias = vocabulary.load(store).alias_for("the launcher issue")
        assert alias is not None
        self.assertEqual(alias.source, "linear")
        self.assertEqual(alias.target, "API-79")

    def test_voice_add_does_not_parse_a_second_stt_guess(self) -> None:
        store = self._store()
        preparation = vocabulary.prepare_spoken_alias(
            "Call this repo owner/stt-guess the harness",
            focused_repository="focused/repo",
            source="github",
            path=store,
            integrations=IntegrationSettings(github_enabled=True),
        )
        pending = preparation.pending
        assert pending is not None
        self.assertEqual(pending.target, "focused/repo")
        self.assertEqual(pending.source, "github")
        self.assertNotIn("stt-guess", pending.target)

    def test_zendesk_can_store_an_owned_identity_without_a_work_on_route(self) -> None:
        store = self._store()
        zendesk = IntegrationSettings(github_enabled=True, zendesk_enabled=True)
        with redirect_stdout(io.StringIO()):
            vocabulary.add_alias(
                "the login ticket",
                "Example#42",
                path=store,
                integrations=zendesk,
            )
        alias = vocabulary.load(store).alias_for("the login ticket")
        assert alias is not None
        self.assertEqual(alias.source, "zendesk")
        self.assertEqual(alias.target, "example#42")
        self.assertEqual(alias.kind, vocabulary.AliasKind.ISSUE)
        expanded = vocabulary.resolve_aliases("work on the login ticket", path=store)
        self.assertEqual(expanded, "work on example#42")
        self.assertIsNone(
            extract_issue_reference(
                expanded,
                IntegrationSettings(github_enabled=True, zendesk_enabled=True),
            )
        )

    def test_disabled_owner_still_expands_but_does_not_start_work(self) -> None:
        store = self._store()
        store.write_text(
            json.dumps(
                {
                    "version": 2,
                    "aliases": [
                        {
                            "phrase": "launcher ticket",
                            "target": "API-79",
                            "kind": "linear",
                        }
                    ],
                }
            )
        )
        expanded = vocabulary.resolve_aliases("work on launcher ticket", path=store)
        disabled = IntegrationSettings(github_enabled=True, linear_enabled=False)
        self.assertEqual(expanded, "work on API-79")
        self.assertIsNone(extract_issue_reference(expanded, disabled))
        registry = build_integration_registry(
            replace(default_user_config(), integrations=disabled)
        )
        with (
            mock.patch.object(service, "_job_store") as store_mock,
            mock.patch.object(service, "launch_worker"),
            self.assertRaisesRegex(HarnessError, "provider is unavailable"),
        ):
            service.start_job(expanded, issue_key="API-79", integrations=registry)
        store_mock.return_value.create.assert_not_called()


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
        self.assertEqual(pending.source, "github")
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
        self.assertEqual(pending.source, "github")

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

    def test_add_linear_alias_dispatches(self) -> None:
        arguments = cli.parser().parse_args(
            ["vocabulary", "add", "alias", "launcher ticket", "API-79"]
        )
        with mock.patch.object(cli, "add_alias") as add_alias:
            cli.dispatch(arguments)
        add_alias.assert_called_once_with("launcher ticket", "API-79", force=False)

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
