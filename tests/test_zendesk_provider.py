from __future__ import annotations

import unittest
from unittest import mock

from local_voice_harness import context_providers, desktop
from local_voice_harness.context_fragment import ContextFragment
from local_voice_harness.integrations import zendesk
from local_voice_harness.user_config import IntegrationSettings


class ZendeskUrlTests(unittest.TestCase):
    def test_extracts_canonical_ticket_url(self) -> None:
        ticket = zendesk.zendesk_ticket_from_url(
            "https://Example-Help.zendesk.com/agent/tickets/42?foo=bar"
        )
        self.assertEqual(ticket, zendesk.ZendeskTicket("example-help", 42))

        invalid_urls = [
            "http://example.zendesk.com/agent/tickets/42",
            "https://example.zendesk.com.evil.test/agent/tickets/42",
            "https://user@example.zendesk.com/agent/tickets/42",
            "https://example.zendesk.com:444/agent/tickets/42",
            "https://example.zendesk.com/hc/en-us/requests/42",
            "https://example.zendesk.com/agent/tickets/not-a-number",
            "https://-example.zendesk.com/agent/tickets/42",
        ]
        for url in invalid_urls:
            with self.subTest(url=url):
                self.assertIsNone(zendesk.zendesk_ticket_from_url(url))


class ZendeskPageCaptureTests(unittest.TestCase):
    def test_copies_rendered_page_text_and_restores_clipboard(self) -> None:
        url = "https://example.zendesk.com/agent/tickets/42"
        page_text = " Ticket 42 \n\n Customer cannot sign in "
        window = desktop.Window("42", "firefox", 10)
        backend = mock.Mock()
        backend.has_clipboard.return_value = True
        backend.active_window.return_value = window
        backend.read_clipboard.side_effect = [
            (True, "previous"),
            (True, url),
            (True, page_text),
            (True, page_text),
        ]
        backend.send_key.return_value = True
        with (
            mock.patch.object(zendesk, "get_desktop", return_value=backend),
            mock.patch.object(zendesk.time, "sleep"),
        ):
            captured = zendesk._focused_firefox_page_text(url)

        self.assertEqual(captured, "Ticket 42\nCustomer cannot sign in")
        self.assertEqual(
            backend.send_key.call_args_list,
            [
                mock.call("ctrl+l", window=window),
                mock.call("ctrl+c", window=window),
                mock.call("Escape", window=window),
                mock.call("ctrl+a", window=window),
                mock.call("ctrl+c", window=window),
                mock.call("Escape", window=window),
            ],
        )
        backend.write_clipboard.assert_called_once_with("previous")

    def test_focus_change_aborts_page_capture(self) -> None:
        url = "https://example.zendesk.com/agent/tickets/42"
        window = desktop.Window("42", "firefox", 10)
        changed = desktop.Window("99", "foot", 11)
        backend = mock.Mock()
        backend.has_clipboard.return_value = True
        backend.active_window.side_effect = [window, window, window, changed, changed]
        backend.read_clipboard.side_effect = [
            (True, "previous"),
            (True, url),
            (True, url),
        ]
        backend.send_key.return_value = True
        with (
            mock.patch.object(zendesk, "get_desktop", return_value=backend),
            mock.patch.object(zendesk.time, "sleep"),
        ):
            self.assertIsNone(zendesk._focused_firefox_page_text(url))

        self.assertEqual(
            backend.send_key.call_args_list,
            [
                mock.call("ctrl+l", window=window),
                mock.call("ctrl+c", window=window),
                mock.call("Escape", window=window),
            ],
        )
        backend.write_clipboard.assert_called_once_with("previous")

    def test_non_firefox_window_is_never_copied(self) -> None:
        backend = mock.Mock()
        backend.has_clipboard.return_value = True
        backend.active_window.return_value = desktop.Window("42", "alacritty", 1)
        with mock.patch.object(zendesk, "get_desktop", return_value=backend):
            self.assertIsNone(
                zendesk._focused_firefox_page_text(
                    "https://example.zendesk.com/agent/tickets/42"
                )
            )
        backend.send_key.assert_not_called()


class ZendeskProviderTests(unittest.TestCase):
    URL = "https://example.zendesk.com/agent/tickets/42"

    def test_capture_returns_bounded_labelled_fragment(self) -> None:
        with mock.patch.object(
            zendesk, "_focused_firefox_page_text", return_value="x" * 20_000
        ):
            fragment = zendesk.ZendeskProvider().capture(self.URL)

        self.assertIsInstance(fragment, ContextFragment)
        assert fragment is not None
        self.assertEqual(fragment.source, "zendesk")
        self.assertIn("untrusted external context", fragment.text)
        self.assertIn("Tenant: example", fragment.text)
        self.assertIn("Ticket: #42", fragment.text)
        self.assertIn("Rendered ticket text:", fragment.text)
        self.assertLess(len(fragment.text), 10_500)
        self.assertTrue(fragment.text.endswith("…"))

    def test_capture_keeps_ticket_identity_when_page_copy_fails(self) -> None:
        with mock.patch.object(
            zendesk, "_focused_firefox_page_text", return_value=None
        ):
            fragment = zendesk.ZendeskProvider().capture(self.URL)

        assert fragment is not None
        self.assertIn("Tenant: example", fragment.text)
        self.assertIn("Ticket: #42", fragment.text)
        self.assertIn("could not be read", fragment.text)

    def test_capture_ignores_non_zendesk_url_without_copying(self) -> None:
        with mock.patch.object(zendesk, "_focused_firefox_page_text") as page_text:
            fragment = zendesk.ZendeskProvider().capture(
                "https://github.com/example/project/issues/42"
            )

        self.assertIsNone(fragment)
        page_text.assert_not_called()

    def test_matches_only_zendesk_ticket_urls(self) -> None:
        provider = zendesk.ZendeskProvider()
        self.assertTrue(provider.matches(self.URL))
        self.assertFalse(provider.matches("https://example.com/agent/tickets/42"))


class ZendeskRegistryTests(unittest.TestCase):
    def test_enabled_flag_registers_zendesk_provider(self) -> None:
        providers = context_providers.available_context_providers(
            IntegrationSettings(github_enabled=False, zendesk_enabled=True)
        )
        self.assertEqual(len(providers), 1)
        self.assertIsInstance(providers[0], zendesk.ZendeskProvider)

    def test_disabled_flag_registers_no_provider(self) -> None:
        self.assertEqual(
            context_providers.available_context_providers(
                IntegrationSettings(github_enabled=False, zendesk_enabled=False)
            ),
            (),
        )

    def test_disabled_registry_does_not_inspect_or_copy_zendesk_url(self) -> None:
        with (
            mock.patch.object(zendesk, "zendesk_ticket_from_url") as ticket,
            mock.patch.object(zendesk, "_focused_firefox_page_text") as page_text,
        ):
            fragment = context_providers.capture_context(
                ZendeskProviderTests.URL,
                IntegrationSettings(github_enabled=False, zendesk_enabled=False),
            )

        self.assertIsNone(fragment)
        ticket.assert_not_called()
        page_text.assert_not_called()


if __name__ == "__main__":
    unittest.main()
