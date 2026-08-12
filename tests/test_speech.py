from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from local_voice_harness import app, cli, config, vocabulary
from local_voice_harness.browser_context import RequestContext
from local_voice_harness.intent import Intent, IntentRoute
from local_voice_harness.speech import (
    SpeechRenderer,
    SpeechToken,
    SpeechTokenKind,
    StreamingSpeechRenderer,
)
from local_voice_harness.user_config import default_user_config


class SpeechRendererTests(unittest.TestCase):
    def test_ordinary_and_mixed_language_prose_is_unchanged(self) -> None:
        renderer = SpeechRenderer()

        text = "That sounds good. Déjà vu; 版本 2 is ready."

        self.assertEqual(renderer.render(text), text)

    def test_references_preserve_numbers_and_meaning(self) -> None:
        renderer = SpeechRenderer()

        self.assertEqual(
            renderer.render("Issue #42 and PR #17 are related to #8."),
            "issue 42 and pull request 17 are related to issue 8.",
        )

    def test_commit_and_job_identities_are_spelled_without_omission(self) -> None:
        renderer = SpeechRenderer()

        self.assertEqual(
            renderer.render("Commit ab12cde fixed job 01HF7Z9K2MNP."),
            "commit a b 1 2 c d e fixed job 0 1 H F 7 Z 9 K 2 M N P.",
        )

    def test_paths_urls_and_identifiers_use_bounded_rules(self) -> None:
        renderer = SpeechRenderer(local_checkout=Path("/work/harness"))

        rendered = renderer.render(
            "Open /work/harness/src/http_client.py, then visit "
            "https://api.example.com/v1/job_status. Use local-voice-harness."
        )

        self.assertEqual(
            rendered,
            "Open the local checkout, src slash H T T P client dot py, then visit "
            "secure URL, A P I dot example dot com, v1 slash job status. "
            "Use local voice harness.",
        )

    def test_repository_owners_remain_distinguishable(self) -> None:
        renderer = SpeechRenderer()

        rendered = renderer.render("Compare alpha/tool#7 with beta/tool#7.")

        self.assertEqual(
            rendered,
            "Compare alpha slash tool, issue 7 with beta slash tool, issue 7.",
        )

    def test_github_url_keeps_repository_and_reference_identity(self) -> None:
        renderer = SpeechRenderer()

        rendered = renderer.render("See https://github.com/alpha/tool/pull/17.")

        self.assertEqual(
            rendered,
            "See secure GitHub, alpha slash tool, pull request 17.",
        )

    def test_url_rendering_preserves_scheme_and_query_identity(self) -> None:
        renderer = SpeechRenderer()

        secure = renderer.render("HTTPS://example.com/search?id=1")
        insecure = renderer.render("http://example.com/search?id=2")

        self.assertEqual(
            secure,
            "secure URL, example dot com, search, query id equals 1",
        )
        self.assertEqual(
            insecure,
            "URL, example dot com, search, query id equals 2",
        )
        self.assertNotEqual(secure, insecure)

    def test_malformed_url_is_left_safe_for_tts(self) -> None:
        renderer = SpeechRenderer()

        text = "Try https://example.com:invalid/path."

        self.assertEqual(renderer.render(text), text)

    def test_unknown_home_path_never_aborts_rendering(self) -> None:
        renderer = SpeechRenderer()

        rendered = renderer.render("Open ~voice-harness-user-that-does-not-exist/file.")

        self.assertEqual(
            rendered,
            "Open path ~voice harness user that does not exist slash file.",
        )

    def test_slash_prose_and_dotnet_are_not_rewritten(self) -> None:
        renderer = SpeechRenderer()

        text = (
            "Choose input/output, pass/fail, on/off, left/right, A/B, 24/7, "
            "true/False, and/or .NET as appropriate."
        )

        self.assertEqual(renderer.render(text), text)

    def test_contextual_and_dotted_repositories_remain_distinct_from_paths(
        self,
    ) -> None:
        renderer = SpeechRenderer()

        self.assertEqual(
            renderer.render(
                "Compare repository alpha/tool with repository owner/repo.name."
            ),
            "Compare alpha slash tool with owner slash repo dot name.",
        )

    def test_contextual_arbitrary_relative_path_is_not_a_repository(self) -> None:
        renderer = SpeechRenderer()

        self.assertEqual(
            renderer.render("Open build/output.log."),
            "Open path build slash output dot log.",
        )

    def test_nested_relative_path_is_rendered_as_one_token(self) -> None:
        renderer = SpeechRenderer()

        self.assertEqual(
            renderer.render("Open src/package/http_client.py."),
            "Open path src slash package slash H T T P client dot py.",
        )

    def test_two_part_kebab_pascal_case_and_bare_sha_are_supported(self) -> None:
        renderer = SpeechRenderer()

        self.assertEqual(
            renderer.render("HTTPClient uses kebab-case at ab12cde."),
            "H T T P Client uses kebab case at commit a b 1 2 c d e.",
        )

    def test_versions_decimals_and_ip_addresses_are_not_confused_with_shas(
        self,
    ) -> None:
        renderer = SpeechRenderer()

        text = "Version 1.2.3 uses 10.0.0.8 and scored 0.75."

        self.assertEqual(renderer.render(text), text)

    def test_tokenizer_exposes_typed_technical_values(self) -> None:
        renderer = SpeechRenderer()

        tokens = renderer.tokenize("See #12 and snake_case.")

        self.assertIn(
            SpeechToken(SpeechTokenKind.REFERENCE, "#12"),
            tokens,
        )
        self.assertIn(
            SpeechToken(SpeechTokenKind.IDENTIFIER, "snake_case"),
            tokens,
        )

    def test_explicit_pronunciations_are_local_to_rendering(self) -> None:
        pronunciation = vocabulary.Pronunciation("Herdr", "herder")
        renderer = SpeechRenderer(pronunciations=(pronunciation,))
        source = "Herdr completed owner/repo#9."

        self.assertEqual(
            renderer.render(source),
            "herder completed owner slash repo, issue 9.",
        )
        self.assertEqual(source, "Herdr completed owner/repo#9.")

    def test_pronunciation_aliases_do_not_cascade(self) -> None:
        renderer = SpeechRenderer(
            pronunciations=(
                vocabulary.Pronunciation("Herdr", "herder"),
                vocabulary.Pronunciation("herder", "header"),
            )
        )

        self.assertEqual(renderer.render("Herdr and herder"), "herder and header")


class StreamingSpeechRendererTests(unittest.TestCase):
    def test_split_reference_and_alias_are_buffered_until_complete(self) -> None:
        renderer = StreamingSpeechRenderer(
            SpeechRenderer(
                pronunciations=(
                    vocabulary.Pronunciation("Local Voice Harness", "the harness"),
                )
            )
        )

        self.assertEqual(renderer.feed("PR #"), ())
        self.assertEqual(renderer.feed("42 changed Local "), ())
        self.assertEqual(
            renderer.feed("Voice Harness."),
            ("pull request 42 changed the harness.",),
        )
        self.assertEqual(renderer.flush(), ())

    def test_identifier_split_preserves_original_adjacency(self) -> None:
        renderer = StreamingSpeechRenderer(SpeechRenderer())

        self.assertEqual(renderer.feed("Use snake_"), ())
        self.assertEqual(renderer.feed("case."), ("Use snake case.",))


class PronunciationVocabularyTests(unittest.TestCase):
    def _store(self) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name) / "vocabulary.json"

    def test_version_one_store_migrates_without_losing_entries(self) -> None:
        store = self._store()
        store.write_text(
            '{"version": 1, "replacements": [], "aliases": []}',
            encoding="utf-8",
        )

        loaded = vocabulary.load(store)

        self.assertEqual(loaded.version, vocabulary.SCHEMA_VERSION)
        self.assertEqual(loaded.pronunciations, ())

    def test_pronunciation_is_private_validated_and_round_trips(self) -> None:
        store = self._store()
        with redirect_stdout(io.StringIO()):
            vocabulary.add_pronunciation("Herdr", "herder", path=store)

        loaded = vocabulary.load(store)

        self.assertEqual(store.stat().st_mode & 0o777, 0o600)
        self.assertEqual(
            loaded.pronunciations,
            (vocabulary.Pronunciation("Herdr", "herder"),),
        )

    def test_pronunciation_rejects_markup_and_command_syntax(self) -> None:
        store = self._store()

        for unsafe in (
            "<speak>name</speak>",
            "$(notify-send bad)",
            "one; rm",
            "line\nbreak",
        ):
            with self.subTest(unsafe=unsafe):
                with self.assertRaises(vocabulary.VocabularyError):
                    vocabulary.add_pronunciation("name", unsafe, path=store)

    def test_pronunciation_accepts_unicode_letters_and_combining_marks(self) -> None:
        store = self._store()
        with redirect_stdout(io.StringIO()):
            vocabulary.add_pronunciation("José", "Jose\u0301", path=store)

        pronunciation = vocabulary.load(store).pronunciation_for("José")

        self.assertIsNotNone(pronunciation)
        assert pronunciation is not None
        self.assertEqual(pronunciation.spoken, "Jose\u0301")


class SpeechChannelIntegrationTests(unittest.TestCase):
    def test_foreground_renders_only_the_tts_channel(self) -> None:
        settings = default_user_config(home=Path("/work/harness"))
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app,
                "request_context",
                return_value=RequestContext("status"),
            ),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ),
            mock.patch.object(
                app,
                "qwen_response",
                return_value="Issue #42 changed snake_case.",
            ),
            mock.patch.object(app, "stream_and_play") as play,
            redirect_stdout(io.StringIO()) as output,
        ):
            app.respond("status", user_config=settings)

        self.assertIn("Assistant: Issue #42 changed snake_case.", output.getvalue())
        play.assert_called_once_with(
            "issue 42 changed snake case.",
            settings=settings.audio,
        )

    def test_preview_command_does_not_invoke_playback(self) -> None:
        arguments = cli.parser().parse_args(
            ["pronounce", "PR #9 changed api_client.py"]
        )
        with (
            mock.patch.object(
                config,
                "VOCABULARY_PATH",
                Path("/definitely/missing/vocabulary.json"),
            ),
            redirect_stdout(io.StringIO()) as output,
        ):
            cli.dispatch(arguments)

        self.assertEqual(
            output.getvalue().strip(),
            "pull request 9 changed A P I client dot py",
        )


if __name__ == "__main__":
    unittest.main()
