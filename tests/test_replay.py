from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness.browser_context import RequestContext
from local_voice_harness.intent import Intent, IntentRoute
from local_voice_harness.replay import (
    ReplayBundle,
    ReplayError,
    capture_bundle,
    load_bundle,
    manifest_summary,
    run_replay,
    save_bundle,
)
from local_voice_harness.responses import AssistantResponse
from local_voice_harness.transcript import TranscriptReplacement

FIXTURE = Path(__file__).parent / "fixtures/replay/routing-v1.json"


def example_bundle() -> ReplayBundle:
    raw = "Please ask Cursa to work on issues 12 and 18"
    context = RequestContext(
        "Please ask Cursor to work on issues 12 and 18\n\nuntrusted body",
        focused_repository="example/project",
        github_repository="example/project",
        issue_scope="example/project",
        issue_scope_source="github",
        focused_app_class="cursor",
        focused_app_context="untrusted body",
        focused_app_sources=("selection",),
    )
    return capture_bundle(
        raw,
        replacements=(TranscriptReplacement("Cursa", "Cursor"),),
        context=context,
        route=IntentRoute(Intent.AGENT_SUBMIT, "high"),
        response=AssistantResponse(
            "Two jobs are ready.",
            "Ready to submit example/project#12 and example/project#18.",
        ),
    )


class ReplayBundleTests(unittest.TestCase):
    def test_reviewed_regression_fixture_replays_routing_extraction_and_rendering(
        self,
    ) -> None:
        bundle = load_bundle(FIXTURE)

        response = run_replay(bundle)

        self.assertEqual(bundle.route, IntentRoute(Intent.AGENT_SUBMIT, "high"))
        self.assertEqual(
            [reference.canonical for reference in bundle.ticket_extraction.references],
            ["example/project#12", "example/project#18"],
        )
        self.assertEqual(response, bundle.response)

    def test_round_trip_replays_deterministic_stages_with_private_permissions(
        self,
    ) -> None:
        bundle = example_bundle()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bundle.json"
            save_bundle(bundle, path)

            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            loaded = load_bundle(path)

        self.assertEqual(loaded, bundle)
        self.assertEqual(run_replay(loaded), bundle.response)

    def test_manifest_omits_external_context_body_and_audio(self) -> None:
        document = json.dumps(example_bundle().to_dict())

        self.assertNotIn("untrusted body", document)
        self.assertNotIn("focused_app_context", document)
        self.assertIn("external_context_bodies", document)
        self.assertIn("audio", document)

    def test_replay_uses_recorded_decisions_without_live_boundaries(self) -> None:
        bundle = example_bundle()
        with (
            mock.patch(
                "local_voice_harness.browser_context.request_context"
            ) as request_context,
            mock.patch("local_voice_harness.intent.route_intent") as route_intent,
            mock.patch("urllib.request.urlopen") as urlopen,
            mock.patch("subprocess.run") as subprocess_run,
            mock.patch("subprocess.Popen") as subprocess_popen,
        ):
            response = run_replay(bundle)

        self.assertEqual(response, bundle.response)
        request_context.assert_not_called()
        route_intent.assert_not_called()
        urlopen.assert_not_called()
        subprocess_run.assert_not_called()
        subprocess_popen.assert_not_called()

    def test_tampered_deterministic_output_fails_explicitly(self) -> None:
        document = example_bundle().to_dict()
        transcript = document["transcript"]
        assert isinstance(transcript, dict)
        transcript["normalized"] = "different"
        bundle = ReplayBundle.from_dict(document)

        with self.assertRaisesRegex(ReplayError, "transcript_normalization"):
            run_replay(bundle)

    def test_unknown_version_stage_and_field_are_rejected(self) -> None:
        for mutate, message in (
            (lambda value: value.update(version=2), "unsupported replay schema"),
            (
                lambda value: value["captured_stages"].append("network"),
                "unsupported replay stages",
            ),
            (lambda value: value.update(credentials={}), "unknown fields"),
        ):
            with self.subTest(message=message):
                document = example_bundle().to_dict()
                mutate(document)
                with self.assertRaisesRegex(ReplayError, message):
                    ReplayBundle.from_dict(document)

    def test_incomplete_response_stage_is_rejected(self) -> None:
        document = example_bundle().to_dict()
        document["response"] = None

        with self.assertRaisesRegex(ReplayError, "response_rendering"):
            ReplayBundle.from_dict(document)

    def test_credentials_and_authenticated_urls_are_rejected(self) -> None:
        values = (
            "Authorization: Bearer abcdefghijklmnop",
            "password=correct-horse-battery-staple",
            "Cookie: session=abcdefghijklmnop",
            "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",
            "-----BEGIN PRIVATE KEY-----",
            "https://user:secret@example.com/issues/1",
            "https://example.com/issues/1?access_token=secret",
            "https://example.com/file?X-Amz-Signature=secret",
            "https://hooks.slack.com/services/team/channel/secret",
        )
        for value in values:
            with self.subTest(value=value):
                bundle = example_bundle()
                document = bundle.to_dict()
                transcript = document["transcript"]
                assert isinstance(transcript, dict)
                transcript["raw"] = value
                transcript["normalized"] = value
                with self.assertRaisesRegex(ReplayError, "credential|authenticated"):
                    ReplayBundle.from_dict(document)

    def test_required_omissions_cannot_be_hidden(self) -> None:
        document = example_bundle().to_dict()
        document["omitted_inputs"] = []

        with self.assertRaisesRegex(ReplayError, "required safeguards"):
            ReplayBundle.from_dict(document)

    def test_environment_secret_is_rejected(self) -> None:
        bundle = example_bundle()
        document = bundle.to_dict()
        transcript = document["transcript"]
        assert isinstance(transcript, dict)
        transcript["raw"] = "secret-value-123"
        transcript["normalized"] = "secret-value-123"

        with (
            mock.patch.dict(os.environ, {"EXAMPLE_API_TOKEN": "secret-value-123"}),
            self.assertRaisesRegex(ReplayError, "environment secret"),
        ):
            ReplayBundle.from_dict(document)

    def test_summary_does_not_echo_captured_text(self) -> None:
        summary = manifest_summary(example_bundle())

        self.assertNotIn("Cursa", summary)
        self.assertNotIn("Ready to submit", summary)
        self.assertIn("cursor_submit (high)", summary)
        self.assertIn("example/project#12", summary)


if __name__ == "__main__":
    unittest.main()
