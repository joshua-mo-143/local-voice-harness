from __future__ import annotations

import unittest

from local_voice_harness.responses import AssistantResponse, as_assistant_response


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


if __name__ == "__main__":
    unittest.main()
