from __future__ import annotations

import contextlib
import io
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from local_voice_harness import app, vocabulary
from local_voice_harness.agents.model import JobStatus
from local_voice_harness.browser_context import RequestContext
from local_voice_harness.cursor import consultation
from local_voice_harness.cursor.service import CursorTurnRequest
from local_voice_harness.diagnostics.help import harness_help_response
from local_voice_harness.errors import SpeechDeliveryError
from local_voice_harness.intent import Intent, IntentRoute
from local_voice_harness.questions import AnswerProvenance
from local_voice_harness.responses import AssistantResponse
from local_voice_harness.ticket_snapshot import TicketSnapshot
from local_voice_harness.user_config import default_user_config


class ForegroundDeliveryTests(unittest.TestCase):
    def test_cursor_result_is_acknowledged_after_playback(self) -> None:
        events: list[str] = []

        def cursor_turn(
            request: CursorTurnRequest,
            *,
            delivery_claims: list[tuple[str, str]],
            integrations: object,
        ) -> tuple[str, None]:
            self.assertEqual(request.utterance, "Use Cursor to inspect this repository")
            self.assertIsNone(request.context_repository)
            delivery_claims.append(("123456789abc", "claim"))
            return "done", None

        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app,
                "request_context",
                side_effect=lambda text, **_settings: RequestContext(text),
            ),
            mock.patch.object(app, "route_intent") as route_intent,
            mock.patch.object(app, "cursor_turn", side_effect=cursor_turn),
            mock.patch.object(
                app,
                "stream_and_play",
                side_effect=lambda _text, **_settings: events.append("played"),
            ),
            mock.patch.object(
                app,
                "acknowledge_deliveries",
                side_effect=lambda _claims: events.append("acknowledged"),
            ),
            mock.patch.object(app, "release_deliveries"),
        ):
            app.respond("Use Cursor to inspect this repository")

        route_intent.assert_not_called()
        self.assertEqual(events, ["played", "acknowledged"])

    def test_playback_failure_releases_cursor_result(self) -> None:
        def cursor_turn(
            request: CursorTurnRequest,
            *,
            delivery_claims: list[tuple[str, str]],
            integrations: object,
        ) -> tuple[str, None]:
            self.assertEqual(request.utterance, "Use Cursor to inspect this repository")
            self.assertIsNone(request.context_repository)
            delivery_claims.append(("123456789abc", "claim"))
            return "done", None

        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app,
                "request_context",
                side_effect=lambda text, **_settings: RequestContext(text),
            ),
            mock.patch.object(app, "route_intent") as route_intent,
            mock.patch.object(app, "cursor_turn", side_effect=cursor_turn),
            mock.patch.object(
                app,
                "stream_and_play",
                side_effect=RuntimeError("playback failed"),
            ),
            mock.patch.object(app, "acknowledge_deliveries") as acknowledge,
            mock.patch.object(app, "release_deliveries") as release,
            self.assertRaisesRegex(SpeechDeliveryError, "speech delivery failed"),
        ):
            app.respond("Use Cursor to inspect this repository")

        route_intent.assert_not_called()
        acknowledge.assert_not_called()
        release.assert_called_once_with([("123456789abc", "claim")])

    def test_exhausted_speech_failure_keeps_display_and_unspoken_claims(self) -> None:
        output = io.StringIO()

        def cursor_turn(
            request: CursorTurnRequest,
            *,
            delivery_claims: list[tuple[str, str]],
            integrations: object,
        ) -> tuple[str, None]:
            delivery_claims.append(("123456789abc", "claim"))
            return "The Cursor job finished.", None

        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app,
                "request_context",
                side_effect=lambda text, **_settings: RequestContext(text),
            ),
            mock.patch.object(app, "cursor_turn", side_effect=cursor_turn),
            mock.patch.object(
                app,
                "stream_and_play",
                side_effect=RuntimeError(
                    "Venice TTS request failed: "
                    "Remote end closed connection without response"
                ),
            ),
            mock.patch.object(app, "acknowledge_deliveries") as acknowledge,
            mock.patch.object(app, "release_deliveries") as release,
            contextlib.redirect_stdout(output),
            self.assertRaises(SpeechDeliveryError) as raised,
        ):
            app.respond("Use Cursor to inspect this repository")

        self.assertIn("Assistant: The Cursor job finished.", output.getvalue())
        self.assertIn("speech delivery failed", str(raised.exception))
        self.assertIn("Remote end closed connection", str(raised.exception))
        acknowledge.assert_not_called()
        release.assert_called_once_with([("123456789abc", "claim")])

    def test_ticket_review_without_identity_asks_which_ticket(self) -> None:
        context = RequestContext("review this ticket")
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.CURSOR_SUBMIT, "high"),
            ),
            mock.patch.object(app, "cursor_turn") as cursor,
            mock.patch.object(app, "qwen_response") as qwen,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("review this ticket")

        cursor.assert_not_called()
        qwen.assert_not_called()
        play.assert_called_once_with("Which ticket should I review?", settings=mock.ANY)

    def test_ticket_review_uses_consultation_not_submit(self) -> None:
        context = RequestContext(
            "review this ticket",
            focused_issue="owner/repo#12",
            focused_repository="owner/repo",
            github_issue_context="untrusted body",
        )
        registry = mock.Mock()
        client = registry.herdr_client.return_value
        target = consultation.WorkspaceTarget(
            checkout=Path("/tmp/project"),
            workspace_id="workspace-1",
            label="project",
        )
        findings = AssistantResponse(
            spoken_text="Scope is too broad.",
            display_text="Acceptance criteria mix two children.",
        )
        snapshot = TicketSnapshot(
            "github",
            "owner/repo#12",
            "https://github.com/owner/repo/issues/12",
            "Bound the scope",
            "fetched body",
            "2026-08-15T10:00:00Z",
            "https://github.com/owner/repo/issues/12",
            "OPEN",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "build_integration_registry", return_value=registry),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app.cursor_consultation,
                "pending_question_snapshot",
                return_value=None,
            ),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.CURSOR_SUBMIT, "high"),
            ),
            mock.patch.object(
                app.cursor_consultation, "workspace_target", return_value=target
            ),
            mock.patch.object(app, "ticket_snapshot", return_value=snapshot) as fetch,
            mock.patch.object(
                app.cursor_consultation, "consult_ticket", return_value=findings
            ) as consult,
            mock.patch.object(app, "cursor_turn") as cursor,
            mock.patch.object(app, "qwen_response") as qwen,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("review this ticket")

        consult.assert_called_once_with(
            client,
            target,
            "review this ticket",
            snapshot=snapshot,
            kind="review",
            adversarial=False,
        )
        fetch.assert_called_once_with(
            "owner/repo#12",
            registry,
            provider="github",
            client=client,
        )
        cursor.assert_not_called()
        qwen.assert_not_called()
        self.assertEqual(
            [call.args[0] for call in play.call_args_list],
            [
                consultation.acknowledgement("review this ticket").spoken_text,
                "Scope is too broad.",
            ],
        )

    def test_adversarial_ticket_review_uses_consultation_not_submit(self) -> None:
        context = RequestContext(
            "adversarially review this ticket",
            focused_issue="owner/repo#12",
            focused_repository="owner/repo",
            github_issue_context="untrusted body",
        )
        registry = mock.Mock()
        client = registry.herdr_client.return_value
        target = consultation.WorkspaceTarget(
            checkout=Path("/tmp/project"),
            workspace_id="workspace-1",
            label="project",
        )
        findings = AssistantResponse(
            spoken_text="Scope mixes two children.",
            display_text="Acceptance criteria hide a second ticket.",
        )
        snapshot = TicketSnapshot(
            "github",
            "owner/repo#12",
            "https://github.com/owner/repo/issues/12",
            "Bound the scope",
            "fetched body",
            "2026-08-15T10:00:00Z",
            "https://github.com/owner/repo/issues/12",
            "OPEN",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "build_integration_registry", return_value=registry),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app.cursor_consultation,
                "pending_question_snapshot",
                return_value=None,
            ),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.CURSOR_SUBMIT, "high"),
            ),
            mock.patch.object(
                app.cursor_consultation, "workspace_target", return_value=target
            ),
            mock.patch.object(app, "ticket_snapshot", return_value=snapshot),
            mock.patch.object(
                app.cursor_consultation, "consult_ticket", return_value=findings
            ) as consult,
            mock.patch.object(app, "cursor_turn") as cursor,
            mock.patch.object(app, "qwen_response") as qwen,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("adversarially review this ticket")

        consult.assert_called_once_with(
            client,
            target,
            "adversarially review this ticket",
            snapshot=snapshot,
            kind="review",
            adversarial=True,
        )
        cursor.assert_not_called()
        qwen.assert_not_called()
        self.assertEqual(
            [call.args[0] for call in play.call_args_list],
            [
                consultation.acknowledgement(
                    "adversarially review this ticket"
                ).spoken_text,
                "Scope mixes two children.",
            ],
        )


class SelfHealthRoutingTests(unittest.TestCase):
    def test_health_route_is_isolated_from_context_cursor_and_conversation(
        self,
    ) -> None:
        response = AssistantResponse.from_text("The voice harness looks healthy.")
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.SELF_HEALTH, "high"),
            ),
            mock.patch.object(app, "request_context") as request_context,
            mock.patch.object(
                app, "self_health_response", return_value=response
            ) as health_response,
            mock.patch.object(app, "cursor_turn") as cursor_turn,
            mock.patch.object(app, "qwen_response") as qwen_response,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("Is the voice harness healthy?")

        health_response.assert_called_once_with()
        request_context.assert_not_called()
        cursor_turn.assert_not_called()
        qwen_response.assert_not_called()
        play.assert_called_once_with(response.spoken_text, settings=mock.ANY)


class HarnessHelpRoutingTests(unittest.TestCase):
    def test_help_route_is_isolated_from_context_cursor_and_conversation(
        self,
    ) -> None:
        response = harness_help_response()
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.HARNESS_HELP, "high"),
            ),
            mock.patch.object(app, "request_context") as request_context,
            mock.patch.object(
                app, "harness_help_response", return_value=response
            ) as help_response,
            mock.patch.object(app, "cursor_turn") as cursor_turn,
            mock.patch.object(app, "qwen_response") as qwen_response,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("What can you do?")

        help_response.assert_called_once_with()
        request_context.assert_not_called()
        cursor_turn.assert_not_called()
        qwen_response.assert_not_called()
        play.assert_called_once()
        played = play.call_args.args[0]
        self.assertIn("talk with you", played)
        self.assertIn("hang up when you're done", played)


class AppContextTests(unittest.TestCase):
    def test_config_inspection_reads_snapshot_without_context_or_cursor(self) -> None:
        config = default_user_config(home=Path("/home/example"))
        config = replace(
            config,
            integrations=replace(config.integrations, linear_enabled=True),
        )
        output = io.StringIO()
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.HARNESS_CONFIG_INSPECT, "high"),
            ),
            mock.patch.object(app, "request_context") as request_context,
            mock.patch.object(app, "cursor_turn") as cursor_turn,
            mock.patch.object(app, "qwen_response") as qwen_response,
            mock.patch.object(app, "stream_and_play") as play,
            contextlib.redirect_stdout(output),
        ):
            app.respond("Is Linear enabled?", user_config=config)

        request_context.assert_not_called()
        cursor_turn.assert_not_called()
        qwen_response.assert_not_called()
        play.assert_called_once_with("Linear is enabled.", settings=config.audio)
        self.assertIn("integrations.linear: enabled", output.getvalue())

    def test_config_change_requires_wake_confirmation_without_cursor(self) -> None:
        config = default_user_config(home=Path("/home/example"))
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(
                    Intent.HARNESS_CONFIG_CHANGE,
                    "high",
                    "vad",
                ),
            ),
            mock.patch.object(app, "request_context") as request_context,
            mock.patch.object(app, "cursor_turn") as cursor_turn,
            mock.patch.object(app, "qwen_response") as qwen_response,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("Set barge-in mode to vad", user_config=config)

        request_context.assert_not_called()
        cursor_turn.assert_not_called()
        qwen_response.assert_not_called()
        self.assertIn("require a wake conversation", play.call_args.args[0])
        self.assertIn("didn't write anything", play.call_args.args[0])

    def test_spoken_alias_requires_wake_confirmation_without_writing(self) -> None:
        config = default_user_config(home=Path("/home/example"))
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context") as request_context,
            mock.patch.object(app, "cursor_turn") as cursor_turn,
            mock.patch.object(app, "qwen_response") as qwen_response,
            mock.patch.object(app, "stream_and_play") as play,
            mock.patch.object(vocabulary, "add_alias") as add_alias,
        ):
            app.respond("Call this repo the harness", user_config=config)

        request_context.assert_not_called()
        cursor_turn.assert_not_called()
        qwen_response.assert_not_called()
        add_alias.assert_not_called()
        self.assertIn("requires a wake conversation", play.call_args.args[0])
        self.assertIn("didn't write anything", play.call_args.args[0])

    def test_text_mapping_answers_single_durable_grouped_question(self) -> None:
        answer = (
            "JOS-30: local-voice-harness-batch-fixture, "
            "JOS-29: local-voice-harness-tiered-batch-fixture"
        )
        grouped = ("grouped12345", "question-grouped", "grouped12345-turn")
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app,
                "request_context",
                return_value=RequestContext(answer),
            ),
            mock.patch.object(
                app,
                "_pending_grouped_repository_question",
                return_value=grouped,
            ),
            mock.patch.object(app, "route_intent") as route_intent,
            mock.patch.object(
                app, "cursor_turn", return_value=("Two jobs started.", None)
            ) as cursor_turn,
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond(answer)

        route_intent.assert_not_called()
        cursor_turn.assert_called_once_with(
            CursorTurnRequest(
                answer,
                action="reply",
                reference=answer,
                utterance=answer,
                job_id="grouped12345",
                expected_question_id="question-grouped",
                expected_question_turn="grouped12345-turn",
                answer_provenance=AnswerProvenance.USER_TEXT,
            ),
            delivery_claims=mock.ANY,
            integrations=mock.ANY,
        )

    def test_grouped_question_lookup_requires_one_matching_owner(self) -> None:
        grouped_job = mock.Mock(id="grouped12345", status=JobStatus.AWAITING_USER)
        unrelated_job = mock.Mock(id="other1234567", status=JobStatus.AWAITING_USER)
        grouped_question = mock.Mock(
            id="question-grouped",
            owner="grouped_repository",
            origin=mock.Mock(turn_token="grouped12345-turn"),
        )
        unrelated_question = mock.Mock(owner="repository")

        with (
            mock.patch.object(
                app.CURSOR_STORE,
                "list",
                return_value=[grouped_job, unrelated_job],
            ),
            mock.patch.object(
                app.cursor_questions,
                "current",
                side_effect=[grouped_question, unrelated_question],
            ),
        ):
            target = app._pending_grouped_repository_question()

        self.assertEqual(
            target,
            ("grouped12345", "question-grouped", "grouped12345-turn"),
        )

        with (
            mock.patch.object(
                app.CURSOR_STORE,
                "list",
                return_value=[grouped_job, grouped_job],
            ),
            mock.patch.object(
                app.cursor_questions,
                "current",
                return_value=grouped_question,
            ),
        ):
            self.assertIsNone(app._pending_grouped_repository_question())

    def test_list_repositories_answers_single_repository_question(self) -> None:
        target = ("repo123456789", "question-repository", "repo123456789-turn")
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app,
                "request_context",
                return_value=RequestContext("list repositories"),
            ),
            mock.patch.object(app, "_pending_repository_question", return_value=target),
            mock.patch.object(app, "route_intent") as route_intent,
            mock.patch.object(
                app,
                "cursor_turn",
                return_value=("Available repositories include: a.", None),
            ) as cursor_turn,
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("list repositories")

        route_intent.assert_not_called()
        cursor_turn.assert_called_once_with(
            CursorTurnRequest(
                "list repositories",
                action="reply",
                reference="list repositories",
                utterance="list repositories",
                job_id="repo123456789",
                expected_question_id="question-repository",
                expected_question_turn="repo123456789-turn",
                answer_provenance=AnswerProvenance.USER_TEXT,
            ),
            delivery_claims=mock.ANY,
            integrations=mock.ANY,
        )

    def test_repository_question_lookup_requires_one_matching_owner(self) -> None:
        repository_job = mock.Mock(id="repo123456789", status=JobStatus.AWAITING_USER)
        unrelated_job = mock.Mock(id="other1234567", status=JobStatus.AWAITING_USER)
        repository_question = mock.Mock(
            id="question-repository",
            owner="repository",
            origin=mock.Mock(turn_token="repo123456789-turn"),
        )
        unrelated_question = mock.Mock(owner="grouped_repository")

        with (
            mock.patch.object(
                app.CURSOR_STORE,
                "list",
                return_value=[repository_job, unrelated_job],
            ),
            mock.patch.object(
                app.cursor_questions,
                "current",
                side_effect=[repository_question, unrelated_question],
            ),
        ):
            self.assertEqual(
                app._pending_repository_question(),
                ("repo123456789", "question-repository", "repo123456789-turn"),
            )

        with (
            mock.patch.object(
                app.CURSOR_STORE,
                "list",
                return_value=[repository_job, repository_job],
            ),
            mock.patch.object(
                app.cursor_questions,
                "current",
                return_value=repository_question,
            ),
        ):
            self.assertIsNone(app._pending_repository_question())

    def test_response_channels_are_selected_at_foreground_boundary(self) -> None:
        output = io.StringIO()
        response = AssistantResponse(
            spoken_text="The job started.",
            display_text="Started job 123456789abc in /tmp/example.",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app, "request_context", return_value=RequestContext("start it")
            ),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ),
            mock.patch.object(app, "qwen_response", return_value=response) as qwen,
            mock.patch.object(app, "stream_and_play") as play,
            contextlib.redirect_stdout(output),
        ):
            app.respond("start it")

        self.assertIn(f"Assistant: {response.display_text}", output.getvalue())
        self.assertNotIn(response.spoken_text, output.getvalue())
        play.assert_called_once_with(response.spoken_text, settings=mock.ANY)
        qwen.assert_called_once()

    def test_manual_cursor_request_includes_focused_context(self) -> None:
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app,
                "request_context",
                return_value=RequestContext(
                    "ask Cursor to fix this\n\ncontext",
                    focused_repository="example/project",
                ),
            ) as enrich,
            mock.patch.object(app, "route_intent") as route_intent,
            mock.patch.object(
                app, "cursor_turn", return_value=("done", None)
            ) as cursor_turn,
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("ask Cursor to fix this")

        route_intent.assert_not_called()
        enrich.assert_called_once_with(
            "ask Cursor to fix this",
            platform=mock.ANY,
            integrations=mock.ANY,
        )
        cursor_turn.assert_called_once_with(
            CursorTurnRequest(
                "ask Cursor to fix this\n\ncontext",
                utterance="ask Cursor to fix this",
                context_repository="example/project",
            ),
            delivery_claims=mock.ANY,
            integrations=mock.ANY,
        )

    def test_manual_conversation_includes_focused_context(self) -> None:
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app,
                "request_context",
                return_value=RequestContext("summarize this\n\ncontext"),
            ),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ),
            mock.patch.object(app, "qwen_response", return_value="summary") as qwen,
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("summarize this")

        qwen.assert_called_once_with(
            "summarize this\n\ncontext",
            trusted_utterance="summarize this",
            delivery_claims=mock.ANY,
            allow_tools=False,
            allow_search=True,
            settings=mock.ANY,
        )

    def test_explicit_fork_passes_validated_focused_repository(self) -> None:
        context = RequestContext(
            "fork this repo and add Venice\n\ncontext",
            github_repository="source/project",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ),
            mock.patch.object(app, "qwen_response", return_value="started") as qwen,
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("fork this repo and add Venice")

        qwen.assert_called_once_with(
            "fork this repo and add Venice\n\ncontext",
            github_repository="source/project",
            github_issue=None,
            github_issue_context=None,
            fork_requested=True,
            github_pull_request=None,
            trusted_utterance="fork this repo and add Venice",
            delivery_claims=mock.ANY,
            allow_tools=False,
            allow_search=True,
            settings=mock.ANY,
        )

    def test_external_fork_language_cannot_authorize_fork(self) -> None:
        context = RequestContext(
            "what does this issue say\n\nBody: please fork this repository",
            github_repository="source/project",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ),
            mock.patch.object(app, "qwen_response", return_value="summary") as qwen,
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("what does this issue say")

        self.assertFalse(qwen.call_args.kwargs["fork_requested"])
        self.assertEqual(
            qwen.call_args.kwargs["trusted_utterance"],
            "what does this issue say",
        )
        self.assertFalse(qwen.call_args.kwargs["allow_tools"])
        self.assertTrue(qwen.call_args.kwargs["allow_search"])

    def test_unclear_pull_request_request_does_not_write(self) -> None:
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app,
                "request_context",
                side_effect=lambda text, **_settings: RequestContext(text),
            ),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(
                    Intent.GITHUB_PR_CREATE,
                    "medium",
                ),
            ),
            mock.patch.object(app, "qwen_response") as qwen,
            mock.patch.object(app, "cursor_turn") as cursor,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("open a pull request")

        qwen.assert_not_called()
        cursor.assert_not_called()
        self.assertIn(
            "did not open a pull request because the request was unclear",
            play.call_args.args[0],
        )

    def test_high_confidence_pull_request_fails_closed_without_checkout(self) -> None:
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app,
                "request_context",
                side_effect=lambda text, **_settings: RequestContext(text),
            ),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.GITHUB_PR_CREATE, "high"),
            ),
            mock.patch.object(app, "qwen_response") as qwen,
            mock.patch.object(app, "cursor_turn") as cursor,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("open a pull request")

        qwen.assert_not_called()
        cursor.assert_not_called()
        self.assertIn(
            "don't have a recent completed job checkout",
            play.call_args.args[0],
        )

    def test_high_confidence_pull_request_follows_up_completed_checkout(self) -> None:
        parent = mock.Mock()
        parent.id = "123456789abc"
        parent.status = JobStatus.COMPLETED
        parent.worktree_path = "/home/test/src/project"
        parent.repository = "/home/test/src/project"
        parent.revision = 4
        parent.completed_at = 20.0
        captured: dict[str, object] = {}

        def cursor_turn(
            request: CursorTurnRequest,
            *,
            delivery_claims: list[tuple[str, str]],
            integrations: object,
        ) -> tuple[str, None]:
            captured["request"] = request
            return "Create private source/project?", None

        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app,
                "request_context",
                side_effect=lambda text, **_settings: RequestContext(text),
            ),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.GITHUB_PR_CREATE, "high"),
            ),
            mock.patch.object(app.CURSOR_STORE, "list", return_value=[parent]),
            mock.patch.object(app, "qwen_response") as qwen,
            mock.patch.object(app, "cursor_turn", side_effect=cursor_turn) as cursor,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("open a pull request")

        qwen.assert_not_called()
        cursor.assert_called_once()
        request = captured["request"]
        assert isinstance(request, CursorTurnRequest)
        self.assertEqual(request.action, "follow_up")
        self.assertEqual(request.job_id, "123456789abc")
        self.assertEqual(request.expected_parent_revision, 4)
        self.assertEqual(request.expected_completed_at, 20.0)
        self.assertTrue(request.github_pr_create_requested)
        self.assertIn("Create private source/project?", play.call_args.args[0])

    def test_medium_confidence_merge_does_not_write(self) -> None:
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app,
                "request_context",
                side_effect=lambda text, **_settings: RequestContext(text),
            ),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.GITHUB_PR_MERGE, "medium"),
            ),
            mock.patch.object(app, "qwen_response") as qwen,
            mock.patch.object(app, "cursor_turn") as cursor,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("merge the pull request")

        qwen.assert_not_called()
        cursor.assert_not_called()
        self.assertIn(
            "did not merge a pull request because the request was unclear",
            play.call_args.args[0],
        )

    def test_high_confidence_merge_starts_job_without_merging_immediately(self) -> None:
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app,
                "request_context",
                side_effect=lambda text, **_settings: RequestContext(
                    text,
                    github_repository="source/project",
                    github_pull_request=7,
                ),
            ),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.GITHUB_PR_MERGE, "high"),
            ),
            mock.patch.object(app, "qwen_response") as qwen,
            mock.patch.object(
                app, "cursor_turn", return_value=("Merge it?", None)
            ) as cursor,
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("merge this pull request")

        qwen.assert_not_called()
        cursor.assert_called_once()
        request = cursor.call_args.args[0]
        self.assertTrue(request.github_pr_merge_requested)
        self.assertEqual(request.github_repository, "source/project")
        self.assertEqual(request.github_pr_merge_number, 7)

    def test_actionable_github_issue_metadata_reaches_cursor(self) -> None:
        context = RequestContext(
            "work on this\n\nIssue: #42",
            focused_repository="source/project",
            focused_issue="source/project#42",
            github_repository="source/project",
            github_issue=42,
            github_issue_context="Issue: #42",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.CURSOR_SUBMIT, "high"),
            ),
            mock.patch.object(
                app, "cursor_turn", return_value=("started", None)
            ) as cursor,
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("work on this")

        cursor.assert_called_once_with(
            CursorTurnRequest(
                context.text,
                utterance="work on this",
                context_repository="source/project",
                github_repository="source/project",
                github_issue=42,
                github_issue_context="Issue: #42",
                fork_requested=False,
                github_pull_request=None,
            ),
            delivery_claims=mock.ANY,
            integrations=mock.ANY,
        )

    def test_focused_github_issue_does_not_submit_at_low_confidence(self) -> None:
        context = RequestContext(
            "work on this issue please\n\nIssue: #56",
            focused_repository="source/project",
            focused_issue="source/project#56",
            github_repository="source/project",
            github_issue=56,
            github_issue_context="Issue: #56",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.CURSOR_SUBMIT, "low"),
            ),
            mock.patch.object(
                app, "cursor_turn", return_value=("started", None)
            ) as cursor,
            mock.patch.object(app, "qwen_response") as qwen,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("work on this issue please")

        qwen.assert_not_called()
        cursor.assert_not_called()
        play.assert_called_once_with(
            app.NON_ACTIONABLE_SUBMIT_RESPONSE, settings=mock.ANY
        )

    def test_low_confidence_submit_without_focus_uses_safe_response(self) -> None:
        context = RequestContext("work on this please")
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.CURSOR_SUBMIT, "low"),
            ),
            mock.patch.object(app, "cursor_turn") as cursor,
            mock.patch.object(app, "qwen_response", return_value="ok") as qwen,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("work on this please")

        cursor.assert_not_called()
        qwen.assert_not_called()
        play.assert_called_once_with(
            app.NON_ACTIONABLE_SUBMIT_RESPONSE, settings=mock.ANY
        )

    def test_uncertain_bare_ticket_batch_requests_repository_scope(self) -> None:
        text = "Can you work on issues 92, 93 and 95?"
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app,
                "request_context",
                return_value=RequestContext(text),
            ),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.UNCERTAIN, "low"),
            ),
            mock.patch.object(app, "cursor_turn") as cursor,
            mock.patch.object(app, "qwen_response") as qwen,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond(text)

        cursor.assert_not_called()
        qwen.assert_not_called()
        play.assert_called_once_with(
            app.MISSING_ISSUE_SCOPE_RESPONSE.replace(
                "repository-scoped", "repository scoped"
            ),
            settings=mock.ANY,
        )

    def test_actionable_linear_issue_metadata_reaches_cursor(self) -> None:
        context = RequestContext(
            "work on this ticket\n\nIdentifier (untrusted external identifier): ENG-123",
            focused_issue="ENG-123",
            external_issue_reference="ENG-123",
            external_issue_source="linear",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.CURSOR_SUBMIT, "high"),
            ),
            mock.patch.object(
                app, "cursor_turn", return_value=("started", None)
            ) as cursor,
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("work on this ticket")

        cursor.assert_called_once_with(
            CursorTurnRequest(
                context.text,
                utterance="work on this ticket",
                context_repository=None,
                issue_key="ENG-123",
            ),
            delivery_claims=mock.ANY,
            integrations=mock.ANY,
        )

    def test_issue_list_scope_reaches_cursor_submission(self) -> None:
        context = RequestContext(
            "work on issues 12 and 18\n\nRepository context",
            focused_repository="source/project",
            github_repository="source/project",
            issue_scope="source/project",
            issue_scope_source="github",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.CURSOR_SUBMIT, "high"),
            ),
            mock.patch.object(
                app, "cursor_turn", return_value=("started", None)
            ) as cursor,
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("work on issues 12 and 18")

        cursor.assert_called_once_with(
            CursorTurnRequest(
                context.text,
                utterance="work on issues 12 and 18",
                context_repository="source/project",
                issue_scope="source/project",
                issue_scope_source="github",
                github_repository="source/project",
                github_issue=None,
                github_issue_context=None,
                fork_requested=False,
                github_pull_request=None,
            ),
            delivery_claims=mock.ANY,
            integrations=mock.ANY,
        )

    def test_focused_linear_issue_does_not_submit_at_low_confidence(self) -> None:
        context = RequestContext(
            "work on this ticket\n\nIdentifier (untrusted external identifier): ENG-123",
            focused_issue="ENG-123",
            external_issue_reference="ENG-123",
            external_issue_source="linear",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.CURSOR_SUBMIT, "low"),
            ),
            mock.patch.object(
                app, "cursor_turn", return_value=("started", None)
            ) as cursor,
            mock.patch.object(app, "qwen_response") as qwen,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("work on this ticket")

        qwen.assert_not_called()
        cursor.assert_not_called()
        play.assert_called_once_with(
            app.NON_ACTIONABLE_SUBMIT_RESPONSE, settings=mock.ANY
        )


class AppConsultationTests(unittest.TestCase):
    def test_workspace_consultation_uses_read_only_dispatch(self) -> None:
        events: list[tuple[str, str]] = []
        context = RequestContext(
            "What do you think about this approach?",
            focused_repository="source/project",
        )
        registry = mock.Mock()
        client = registry.herdr_client.return_value
        target = consultation.WorkspaceTarget(
            checkout=Path("/tmp/project"),
            workspace_id="workspace-1",
            label="project",
        )

        def consult_after_acknowledgement(
            _client: object,
            _target: consultation.WorkspaceTarget,
            _text: str,
        ) -> str:
            self.assertEqual(
                events,
                [
                    (
                        "play",
                        consultation.acknowledgement("review this").spoken_text,
                    )
                ],
            )
            events.append(("consult", _text))
            return "Use the simpler boundary."

        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app,
                "resolve_aliases",
                return_value="What do you think about this approach?",
            ),
            mock.patch.object(app, "build_integration_registry", return_value=registry),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app.cursor_consultation,
                "pending_question_snapshot",
                return_value=None,
            ),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.WORKSPACE_CONSULTATION, "high"),
            ),
            mock.patch.object(
                app.cursor_consultation, "workspace_target", return_value=target
            ),
            mock.patch.object(
                app.cursor_consultation,
                "consult",
                side_effect=consult_after_acknowledgement,
            ) as consult,
            mock.patch.object(app, "cursor_turn") as cursor,
            mock.patch.object(app, "qwen_response") as qwen,
            mock.patch.object(
                app,
                "stream_and_play",
                side_effect=lambda text, **_kwargs: events.append(("play", text)),
            ) as play,
        ):
            app.respond("review this")

        consult.assert_called_once_with(client, target, context.text)
        cursor.assert_not_called()
        qwen.assert_not_called()
        self.assertEqual(
            [call.args[0] for call in play.call_args_list],
            [
                consultation.acknowledgement("review this").spoken_text,
                "Use the simpler boundary.",
            ],
        )

    def test_pending_consultation_supplies_router_context_without_replying(
        self,
    ) -> None:
        events: list[tuple[str, str]] = []
        context = RequestContext("Which option do you recommend?")
        snapshot = mock.Mock(
            job_id="aaaaaaaaaaaa",
            text="Which design?",
            owner="agent",
        )
        registry = mock.Mock()
        client = registry.herdr_client.return_value

        def consult_after_acknowledgement(
            _client: object,
            _store: object,
            _snapshot: object,
            text: str,
        ) -> str:
            self.assertEqual(
                events,
                [
                    (
                        "play",
                        consultation.acknowledgement(
                            "Which option do you recommend?"
                        ).spoken_text,
                    )
                ],
            )
            events.append(("consult", text))
            return "Choose the safe design."

        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "build_integration_registry", return_value=registry),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app.cursor_consultation,
                "pending_question_snapshot",
                return_value=snapshot,
            ),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(
                    Intent.QUESTION_CONSULTATION,
                    "high",
                ),
            ) as route,
            mock.patch.object(
                app.cursor_consultation,
                "consult_pending_question",
                side_effect=consult_after_acknowledgement,
            ) as consult,
            mock.patch.object(app, "cursor_turn") as cursor,
            mock.patch.object(app, "qwen_response") as qwen,
            mock.patch.object(
                app,
                "stream_and_play",
                side_effect=lambda text, **_kwargs: events.append(("play", text)),
            ) as play,
        ):
            app.respond("Which option do you recommend?")

        route.assert_called_once_with(
            "Which option do you recommend?",
            context,
            cursor_session="aaaaaaaaaaaa",
            pending_question="Which design?",
            clarification_kind="agent",
            settings=mock.ANY,
        )
        consult.assert_called_once_with(
            client,
            app.CURSOR_STORE,
            snapshot,
            context.text,
        )
        cursor.assert_not_called()
        qwen.assert_not_called()
        self.assertEqual(
            [call.args[0] for call in play.call_args_list],
            [
                consultation.acknowledgement(
                    "Which option do you recommend?"
                ).spoken_text,
                "Choose the safe design.",
            ],
        )

    def test_ambiguous_pending_consultation_fails_without_fallback(self) -> None:
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app,
                "request_context",
                return_value=RequestContext("Which option do you recommend?"),
            ),
            mock.patch.object(
                app.cursor_consultation,
                "pending_question_snapshot",
                return_value=None,
            ),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(
                    Intent.QUESTION_CONSULTATION,
                    "high",
                ),
            ),
            mock.patch.object(
                app.cursor_consultation, "consult_pending_question"
            ) as consult,
            mock.patch.object(app, "cursor_turn") as cursor,
            mock.patch.object(app, "qwen_response") as qwen,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("Which option do you recommend?")

        consult.assert_not_called()
        cursor.assert_not_called()
        qwen.assert_not_called()
        play.assert_called_once_with(
            consultation.NO_PENDING_QUESTION,
            settings=mock.ANY,
        )

    def test_missing_workspace_does_not_play_acknowledgement(self) -> None:
        registry = mock.Mock()
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "build_integration_registry", return_value=registry),
            mock.patch.object(
                app,
                "request_context",
                return_value=RequestContext("Inspect this checkout"),
            ),
            mock.patch.object(
                app.cursor_consultation,
                "pending_question_snapshot",
                return_value=None,
            ),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.WORKSPACE_CONSULTATION, "high"),
            ),
            mock.patch.object(
                app.cursor_consultation,
                "workspace_target",
                return_value=None,
            ),
            mock.patch.object(app.cursor_consultation, "consult") as consult,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("Inspect this checkout")

        consult.assert_not_called()
        play.assert_called_once_with(consultation.NO_WORKSPACE, settings=mock.ANY)

    def test_later_consultation_failure_keeps_explicit_failure_response(self) -> None:
        registry = mock.Mock()
        target = consultation.WorkspaceTarget(
            checkout=Path("/tmp/project"),
            workspace_id="workspace-1",
            label="project",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "build_integration_registry", return_value=registry),
            mock.patch.object(
                app,
                "request_context",
                return_value=RequestContext("Inspect this checkout"),
            ),
            mock.patch.object(
                app.cursor_consultation,
                "pending_question_snapshot",
                return_value=None,
            ),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.WORKSPACE_CONSULTATION, "high"),
            ),
            mock.patch.object(
                app.cursor_consultation,
                "workspace_target",
                return_value=target,
            ),
            mock.patch.object(
                app.cursor_consultation,
                "consult",
                side_effect=RuntimeError("agent unavailable"),
            ),
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("Inspect this checkout")

        self.assertEqual(
            [call.args[0] for call in play.call_args_list],
            [
                consultation.acknowledgement("Inspect this checkout").spoken_text,
                "I couldn't complete the read only consultation.",
            ],
        )

    def test_ordinary_conversation_does_not_play_acknowledgement(self) -> None:
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app.cursor_consultation,
                "pending_question_snapshot",
                return_value=None,
            ),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ),
            mock.patch.object(
                app,
                "qwen_response",
                return_value="A conversational answer.",
            ),
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("Tell me something")

        play.assert_called_once_with("A conversational answer.", settings=mock.ANY)

    def test_use_your_recommendation_submits_the_stored_choice(self) -> None:
        snapshot = mock.Mock(
            job_id="aaaaaaaaaaaa",
            question_id="question-1",
            turn_token="turn-1",
            text="Which design?",
            owner="agent",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app.cursor_consultation,
                "pending_question_snapshot",
                return_value=snapshot,
            ),
            mock.patch.object(
                app.cursor_consultation,
                "applicable_choice_id",
                return_value="safe",
            ) as applicable,
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ),
            mock.patch.object(
                app, "cursor_turn", return_value=("Queued the safe design.", None)
            ) as cursor,
            mock.patch.object(app, "qwen_response") as qwen,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("use your recommendation")

        applicable.assert_called_once_with(app.CURSOR_STORE, "aaaaaaaaaaaa")
        request = cursor.call_args.args[0]
        self.assertEqual(request.text, "safe")
        self.assertEqual(request.utterance, "safe")
        self.assertEqual(request.action, "reply")
        self.assertEqual(request.job_id, "aaaaaaaaaaaa")
        self.assertEqual(request.expected_question_id, "question-1")
        self.assertEqual(request.expected_question_turn, "turn-1")
        self.assertEqual(request.answer_provenance, AnswerProvenance.USER_TEXT)
        qwen.assert_not_called()
        play.assert_called_once_with("Queued the safe design.", settings=mock.ANY)

    def test_generic_acknowledgment_does_not_apply_a_recommendation(self) -> None:
        snapshot = mock.Mock(
            job_id="aaaaaaaaaaaa",
            question_id="question-1",
            turn_token="turn-1",
            text="Which design?",
            owner="agent",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app.cursor_consultation,
                "pending_question_snapshot",
                return_value=snapshot,
            ),
            mock.patch.object(
                app.cursor_consultation,
                "applicable_choice_id",
                return_value="safe",
            ) as applicable,
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ),
            mock.patch.object(app, "cursor_turn") as cursor,
            mock.patch.object(app, "qwen_response", return_value="Okay."),
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("okay")

        applicable.assert_not_called()
        cursor.assert_not_called()

    def test_stale_recommendation_reference_fails_closed(self) -> None:
        snapshot = mock.Mock(
            job_id="aaaaaaaaaaaa",
            question_id="question-1",
            turn_token="turn-1",
            text="Which design?",
            owner="agent",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app.cursor_consultation,
                "pending_question_snapshot",
                return_value=snapshot,
            ),
            mock.patch.object(
                app.cursor_consultation,
                "applicable_choice_id",
                return_value=None,
            ),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.AGENT_REPLY, "high"),
            ),
            mock.patch.object(app, "cursor_turn") as cursor,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("use your recommendation")

        cursor.assert_not_called()
        play.assert_called_once_with(
            consultation.RECOMMENDATION_UNAVAILABLE,
            settings=mock.ANY,
        )

    def test_use_your_recommendation_is_not_intercepted_by_router_intent(self) -> None:
        snapshot = mock.Mock(
            job_id="aaaaaaaaaaaa",
            question_id="question-1",
            turn_token="turn-1",
            text="Which design?",
            owner="agent",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app.cursor_consultation,
                "pending_question_snapshot",
                return_value=snapshot,
            ),
            mock.patch.object(
                app.cursor_consultation,
                "applicable_choice_id",
                return_value="safe",
            ),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.HARNESS_CONFIG_CHANGE, "high"),
            ),
            mock.patch.object(
                app, "cursor_turn", return_value=("Queued the safe design.", None)
            ) as cursor,
            mock.patch.object(app, "qwen_response") as qwen,
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("use your recommendation")

        request = cursor.call_args.args[0]
        self.assertEqual(request.text, "safe")
        self.assertEqual(request.action, "reply")
        qwen.assert_not_called()

    def test_use_your_recommendation_without_a_pending_question_fails_closed(
        self,
    ) -> None:
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app.cursor_consultation,
                "pending_question_snapshot",
                return_value=None,
            ),
            mock.patch.object(
                app.cursor_consultation,
                "applicable_choice_id",
            ) as applicable,
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ),
            mock.patch.object(app, "cursor_turn") as cursor,
            mock.patch.object(app, "qwen_response") as qwen,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("use your recommendation")

        applicable.assert_not_called()
        cursor.assert_not_called()
        qwen.assert_not_called()
        play.assert_called_once_with(
            consultation.RECOMMENDATION_UNAVAILABLE,
            settings=mock.ANY,
        )


class CursorFastPathTests(unittest.TestCase):
    def test_coding_request_does_not_need_to_name_cursor(self) -> None:
        text = "fix the failing authentication tests"
        context = RequestContext(f"{text}\n\nrepository context")
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context) as capture,
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.AGENT_SUBMIT, "high"),
            ),
            mock.patch.object(
                app, "cursor_turn", return_value=("started", None)
            ) as cursor_turn,
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond(text)

        capture.assert_called_once()
        self.assertEqual(cursor_turn.call_args.args[0].utterance, text)

    def test_job_management_does_not_capture_external_context(self) -> None:
        text = "list my running jobs"
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context") as capture,
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.AGENT_LIST, "high"),
            ) as route_intent,
            mock.patch.object(app, "cursor_turn", return_value=("none", None)),
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond(text)

        capture.assert_not_called()
        route_intent.assert_called_once_with(
            text,
            RequestContext(text),
            cursor_session=None,
            pending_question=None,
            clarification_kind=None,
            settings=mock.ANY,
        )

    def test_explicit_cursor_utterance_skips_router(self) -> None:
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(
                app,
                "request_context",
                side_effect=lambda text, **_settings: RequestContext(text),
            ),
            mock.patch.object(app, "route_intent") as route_intent,
            mock.patch.object(
                app, "cursor_turn", return_value=("done", None)
            ) as cursor_turn,
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("use cursor to refactor auth")

        route_intent.assert_not_called()
        cursor_turn.assert_called_once_with(
            CursorTurnRequest(
                "use cursor to refactor auth",
                utterance="use cursor to refactor auth",
                context_repository=None,
            ),
            delivery_claims=mock.ANY,
            integrations=mock.ANY,
        )

    def test_non_cursor_utterance_uses_router(self) -> None:
        context = RequestContext("what is the weather")
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ) as route_intent,
            mock.patch.object(app, "qwen_response", return_value="ok"),
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("what is the weather")

        route_intent.assert_called_once_with(
            "what is the weather",
            context,
            cursor_session=None,
            pending_question=None,
            clarification_kind=None,
            settings=mock.ANY,
        )

    def test_ticket_update_without_identity_asks_which_ticket(self) -> None:
        context = RequestContext("update this ticket title")
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.GITHUB_ISSUE_UPDATE, "high"),
            ),
            mock.patch.object(app, "cursor_turn") as cursor,
            mock.patch.object(app, "qwen_response") as qwen,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("update this ticket title")

        cursor.assert_not_called()
        qwen.assert_not_called()
        play.assert_called_once_with("Which ticket should I update?", settings=mock.ANY)

    def test_ticket_update_dispatches_dedicated_durable_job(self) -> None:
        context = RequestContext(
            "update this ticket title",
            focused_issue="owner/repo#12",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.GITHUB_ISSUE_UPDATE, "high"),
            ),
            mock.patch.object(
                app, "cursor_turn", return_value=("drafted", "job")
            ) as cursor,
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("update this ticket title")

        request = cursor.call_args.args[0]
        self.assertTrue(request.github_issue_update_requested)
        self.assertEqual(request.github_repository, "owner/repo")
        self.assertEqual(request.github_issue, 12)
        self.assertFalse(request.github_issue_create_requested)

    def test_low_confidence_ticket_update_does_not_write(self) -> None:
        context = RequestContext(
            "update this ticket title",
            focused_issue="owner/repo#12",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.GITHUB_ISSUE_UPDATE, "low"),
            ),
            mock.patch.object(app, "cursor_turn") as cursor,
            mock.patch.object(app, "qwen_response") as qwen,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("update this ticket title")

        cursor.assert_not_called()
        qwen.assert_not_called()
        play.assert_called_once()
        self.assertIn("did not update a ticket", play.call_args.args[0])

    def test_ticket_close_without_identity_asks_which_ticket(self) -> None:
        context = RequestContext("close this ticket")
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.GITHUB_ISSUE_CLOSE, "high"),
            ),
            mock.patch.object(app, "cursor_turn") as cursor,
            mock.patch.object(app, "qwen_response") as qwen,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("close this ticket")

        cursor.assert_not_called()
        qwen.assert_not_called()
        play.assert_called_once_with("Which ticket should I close?", settings=mock.ANY)

    def test_ticket_close_dispatches_dedicated_durable_job(self) -> None:
        context = RequestContext(
            "close this ticket",
            focused_issue="owner/repo#12",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.GITHUB_ISSUE_CLOSE, "high"),
            ),
            mock.patch.object(
                app, "cursor_turn", return_value=("queued", "job")
            ) as cursor,
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("close this ticket")

        request = cursor.call_args.args[0]
        self.assertTrue(request.github_issue_close_requested)
        self.assertEqual(request.github_repository, "owner/repo")
        self.assertEqual(request.github_issue, 12)
        self.assertFalse(request.github_issue_update_requested)

    def test_low_confidence_ticket_close_does_not_write(self) -> None:
        context = RequestContext(
            "close this ticket",
            focused_issue="owner/repo#12",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.GITHUB_ISSUE_CLOSE, "low"),
            ),
            mock.patch.object(app, "cursor_turn") as cursor,
            mock.patch.object(app, "qwen_response") as qwen,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("close this ticket")

        cursor.assert_not_called()
        qwen.assert_not_called()
        play.assert_called_once()
        self.assertIn("did not close a ticket", play.call_args.args[0])

    def test_ticket_split_without_identity_asks_which_ticket(self) -> None:
        context = RequestContext("split this ticket into two issues")
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.GITHUB_ISSUE_SPLIT, "high"),
            ),
            mock.patch.object(app, "cursor_turn") as cursor,
            mock.patch.object(app, "qwen_response") as qwen,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("split this ticket into two issues")

        cursor.assert_not_called()
        qwen.assert_not_called()
        play.assert_called_once_with("Which ticket should I split?", settings=mock.ANY)

    def test_ticket_split_dispatches_dedicated_durable_job(self) -> None:
        context = RequestContext(
            "split this ticket into two issues",
            focused_issue="owner/repo#12",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.GITHUB_ISSUE_SPLIT, "high"),
            ),
            mock.patch.object(
                app, "cursor_turn", return_value=("drafted", "job")
            ) as cursor,
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("split this ticket into two issues")

        request = cursor.call_args.args[0]
        self.assertTrue(request.github_issue_split_requested)
        self.assertEqual(request.github_repository, "owner/repo")
        self.assertEqual(request.github_issue, 12)
        self.assertFalse(request.github_issue_create_requested)

    def test_low_confidence_ticket_split_does_not_write(self) -> None:
        context = RequestContext(
            "split this ticket into two issues",
            focused_issue="owner/repo#12",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.GITHUB_ISSUE_SPLIT, "low"),
            ),
            mock.patch.object(app, "cursor_turn") as cursor,
            mock.patch.object(app, "qwen_response") as qwen,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("split this ticket into two issues")

        cursor.assert_not_called()
        qwen.assert_not_called()
        play.assert_called_once()
        self.assertIn("did not split a ticket", play.call_args.args[0])

    def test_ticket_merge_without_identity_asks_which_tickets(self) -> None:
        context = RequestContext("merge these tickets")
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.GITHUB_ISSUE_MERGE, "high"),
            ),
            mock.patch.object(app, "cursor_turn") as cursor,
            mock.patch.object(app, "qwen_response") as qwen,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("merge these tickets")

        cursor.assert_not_called()
        qwen.assert_not_called()
        play.assert_called_once_with("Which tickets should I merge?", settings=mock.ANY)

    def test_ticket_merge_dispatches_dedicated_durable_job(self) -> None:
        context = RequestContext(
            "merge owner/repo#12 and owner/repo#13",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.GITHUB_ISSUE_MERGE, "high"),
            ),
            mock.patch.object(
                app, "cursor_turn", return_value=("drafted", "job")
            ) as cursor,
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("merge owner/repo#12 and owner/repo#13")

        request = cursor.call_args.args[0]
        self.assertTrue(request.github_issue_merge_requested)
        self.assertEqual(request.github_repository, "owner/repo")
        self.assertEqual(request.github_issue, 12)
        self.assertEqual(request.ticket_merge_survivor, "owner/repo#12")
        self.assertFalse(request.github_issue_create_requested)

    def test_low_confidence_ticket_merge_does_not_write(self) -> None:
        context = RequestContext("merge owner/repo#12 and owner/repo#13")
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.GITHUB_ISSUE_MERGE, "low"),
            ),
            mock.patch.object(app, "cursor_turn") as cursor,
            mock.patch.object(app, "qwen_response") as qwen,
            mock.patch.object(app, "stream_and_play") as play,
        ):
            app.respond("merge owner/repo#12 and owner/repo#13")

        cursor.assert_not_called()
        qwen.assert_not_called()
        play.assert_called_once()
        self.assertIn("did not merge tickets", play.call_args.args[0])

    def test_github_issue_creation_dispatches_dedicated_durable_job(self) -> None:
        context = RequestContext(
            "create an issue in this repo",
            github_repository="example/project",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.GITHUB_ISSUE_CREATE, "high"),
            ),
            mock.patch.object(
                app, "cursor_turn", return_value=("drafted", "job")
            ) as cursor,
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("create an issue in this repo about startup")

        request = cursor.call_args.args[0]
        self.assertTrue(request.github_issue_create_requested)
        self.assertEqual(request.github_repository, "example/project")
        self.assertEqual(
            request.utterance,
            "create an issue in this repo about startup",
        )

    def test_github_repo_creation_dispatches_dedicated_durable_job(self) -> None:
        context = RequestContext("create a GitHub repository called payments")
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.GITHUB_REPO_CREATE, "high"),
            ),
            mock.patch.object(
                app, "cursor_turn", return_value=("drafted", "job")
            ) as cursor,
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("create a GitHub repository called payments")

        request = cursor.call_args.args[0]
        self.assertTrue(request.github_repo_create_requested)
        self.assertFalse(request.github_repo_create_org_requested)
        self.assertEqual(
            request.utterance,
            "create a GitHub repository called payments",
        )

    def test_github_org_repo_creation_ignores_focused_page_repository(self) -> None:
        context = RequestContext(
            "create a GitHub repository in an organization",
            github_repository="focused/page",
            focused_repository="focused/page",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.GITHUB_ORG_REPO_CREATE, "high"),
            ),
            mock.patch.object(
                app, "cursor_turn", return_value=("drafted", "job")
            ) as cursor,
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("create a GitHub repository in an organization")

        request = cursor.call_args.args[0]
        self.assertTrue(request.github_repo_create_requested)
        self.assertTrue(request.github_repo_create_org_requested)
        self.assertIsNone(request.github_repository)

    def test_linear_ticket_creation_dispatches_dedicated_durable_job(self) -> None:
        context = RequestContext(
            "create a Linear ticket in this team",
            issue_scope="API",
            issue_scope_source="linear",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app.cursor_consultation,
                "pending_question_snapshot",
                return_value=None,
            ),
            mock.patch.object(app, "_single_pending_job", return_value=None),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.LINEAR_TICKET_CREATE, "high"),
            ),
            mock.patch.object(
                app, "cursor_turn", return_value=("drafted", "job")
            ) as cursor,
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("create a Linear ticket in this team about startup")

        request = cursor.call_args.args[0]
        self.assertTrue(request.linear_ticket_create_requested)
        self.assertEqual(request.linear_team, "API")
        self.assertEqual(
            request.utterance,
            "create a Linear ticket in this team about startup",
        )

    def test_router_receives_the_single_pending_confirmation(self) -> None:
        context = RequestContext("no")
        pending = mock.Mock(
            id="aaaaaaaaaaaa",
            question="Create this GitHub issue?",
            clarification_kind="github_issue_create_confirmation",
        )
        with (
            mock.patch.object(app, "start_components"),
            mock.patch.object(app, "request_context", return_value=context),
            mock.patch.object(
                app.cursor_consultation,
                "pending_question_snapshot",
                return_value=None,
            ),
            mock.patch.object(app, "_single_pending_job", return_value=pending),
            mock.patch.object(
                app,
                "route_intent",
                return_value=IntentRoute(Intent.END_CONVERSATION, "high"),
            ) as route,
            mock.patch.object(app, "qwen_response", return_value="okay"),
            mock.patch.object(app, "stream_and_play"),
        ):
            app.respond("no")

        route.assert_called_once_with(
            "no",
            context,
            cursor_session="aaaaaaaaaaaa",
            pending_question="Create this GitHub issue?",
            clarification_kind="github_issue_create_confirmation",
            settings=mock.ANY,
        )


if __name__ == "__main__":
    unittest.main()
