from __future__ import annotations

import unittest

from local_voice_harness.responses import (
    AssistantResponse,
    as_assistant_response,
    spoken_utterance_slice,
    with_spoken_utterance_ack,
    without_spoken_utterance_ack,
)


class AssistantResponseTests(unittest.TestCase):
    def test_plain_text_is_adapted_without_duplication_at_call_site(self) -> None:
        response = as_assistant_response("Hello there.")

        self.assertEqual(response.spoken_text, "Hello there.")
        self.assertEqual(response.display_text, "Hello there.")

    def test_typed_response_is_immutable_and_preserved(self) -> None:
        response = AssistantResponse(
            spoken_text="The job failed.",
            display_text="Job 123 failed during repository setup.",
        )

        self.assertIs(as_assistant_response(response), response)
        with self.assertRaises(AttributeError):
            response.spoken_text = "changed"  # type: ignore[misc]

    def test_invalid_channel_value_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "spoken_text must be a string"):
            AssistantResponse(
                spoken_text=object(),  # type: ignore[arg-type]
                display_text="detail",
            )

        with self.assertRaisesRegex(TypeError, "assistant response"):
            as_assistant_response(object())  # type: ignore[arg-type]


class SpokenUtteranceSliceTests(unittest.TestCase):
    def test_short_utterance_is_kept_in_full(self) -> None:
        self.assertEqual(
            spoken_utterance_slice("revert the login change"),
            "revert the login change",
        )

    def test_first_clause_wins_before_the_word_cap(self) -> None:
        self.assertEqual(
            spoken_utterance_slice(
                "revert the login change. Then review the rest of the auth work."
            ),
            "revert the login change",
        )

    def test_long_clause_is_capped_at_about_twelve_words(self) -> None:
        utterance = (
            "revert the login change and then also update the documentation "
            "for the new auth flow please"
        )
        self.assertEqual(
            spoken_utterance_slice(utterance),
            "revert the login change and then also update the documentation for the",
        )
        self.assertEqual(len(spoken_utterance_slice(utterance).split()), 12)

    def test_accept_sentence_names_the_slice_and_can_drop_it(self) -> None:
        spoken = with_spoken_utterance_ack(
            "Cursor accepted the login change and queued it.",
            "revert the login change",
        )
        self.assertEqual(
            spoken,
            "Cursor accepted the login change and queued it for “revert the login change.”",
        )
        self.assertEqual(
            without_spoken_utterance_ack(spoken),
            "Cursor accepted the login change and queued it.",
        )


if __name__ == "__main__":
    unittest.main()
