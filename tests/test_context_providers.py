from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from local_voice_harness.context_fragment import ContextFragment, ContextProvider
from local_voice_harness.integrations import registry as context_providers
from local_voice_harness.user_config import (
    IntegrationSettings,
    UserConfigurationError,
    default_user_config,
)

URL = "https://example.test/thing/42"


class _StubProvider:
    """A minimal provider used to exercise the generic registry mechanics."""

    name = "stub"

    def __init__(self, fragment: ContextFragment | None = None) -> None:
        self._fragment = fragment

    def matches(self, url: str) -> bool:
        return self._fragment is not None

    def capture(self, url: str) -> ContextFragment | None:
        return self._fragment


class RegistryTests(unittest.TestCase):
    def test_stub_provider_satisfies_the_contract(self) -> None:
        self.assertIsInstance(_StubProvider(), ContextProvider)

    def test_no_registered_factories_expose_no_providers(self) -> None:
        with mock.patch.object(context_providers, "_INTEGRATION_FACTORIES", ()):
            self.assertEqual(context_providers.available_context_providers(), ())

    def test_flag_gates_provider_instantiation(self) -> None:
        fragment = ContextFragment(source="stub", text="context")
        factory = mock.Mock(return_value=_StubProvider(fragment))
        with mock.patch.object(
            context_providers,
            "_INTEGRATION_FACTORIES",
            (("zendesk_enabled", factory),),
        ):
            disabled = context_providers.available_context_providers(
                IntegrationSettings(zendesk_enabled=False)
            )
            self.assertEqual(disabled, ())
            factory.assert_not_called()

            enabled = context_providers.available_context_providers(
                IntegrationSettings(zendesk_enabled=True)
            )
            self.assertEqual(len(enabled), 1)
            factory.assert_called_once_with()

    def test_defaults_enable_only_builtin_github_provider(self) -> None:
        providers = context_providers.available_context_providers()
        self.assertEqual(tuple(provider.name for provider in providers), ("github",))

    def test_malformed_config_fails_before_constructing_providers(self) -> None:
        factory = mock.Mock()
        with (
            mock.patch.object(
                context_providers,
                "load_user_config",
                side_effect=UserConfigurationError("invalid config"),
            ),
            mock.patch.object(
                context_providers,
                "_INTEGRATION_FACTORIES",
                (("github_enabled", factory),),
            ),
        ):
            with self.assertRaisesRegex(UserConfigurationError, "invalid config"):
                context_providers.available_context_providers()

        factory.assert_not_called()

    def test_configured_factories_share_the_startup_snapshot(self) -> None:
        config = default_user_config(Path("/home/example"))
        config = replace(
            config,
            platform=replace(
                config.platform,
                project_root=Path("/repositories"),
                github_root=Path("/repositories/github"),
                herdr_worktree_root=Path("/worktrees"),
                gh_bin=Path("/tools/gh"),
                git_bin=Path("/tools/git"),
                herdr_bin=Path("/tools/herdr"),
                cursor_mcp_auth_source=Path("/authenticated/workspace"),
                github_timeout_seconds=12,
                herdr_timeout_seconds=7,
            ),
        )

        registry = context_providers.build_integration_registry(config)
        github = registry.github_client()
        herdr = registry.herdr_client()

        self.assertEqual(github.gh_executable, "/tools/gh")
        self.assertEqual(github.git_executable, "/tools/git")
        self.assertEqual(github.allowed_root, Path("/repositories"))
        self.assertEqual(github.clone_root, Path("/repositories/github"))
        self.assertEqual(github.timeout, 12)
        self.assertEqual(herdr.executable, "/tools/herdr")
        self.assertEqual(herdr.repository.root, Path("/repositories"))
        self.assertEqual(herdr.workspace.worktree_root, Path("/worktrees"))
        self.assertIsNotNone(herdr.workspace._cursor_mcp_auth)
        assert herdr.workspace._cursor_mcp_auth is not None
        self.assertEqual(
            herdr.workspace._cursor_mcp_auth.source_workspace,
            Path("/authenticated/workspace"),
        )
        self.assertEqual(herdr.timeout, 7)

    def test_disabled_github_does_not_construct_client(self) -> None:
        config = replace(
            default_user_config(Path("/home/example")),
            integrations=IntegrationSettings(github_enabled=False),
        )
        registry = context_providers.build_integration_registry(config)
        factory = mock.Mock()
        registry = replace(
            registry,
            factories=(("github_enabled", factory),),
        )
        self.assertEqual(context_providers.available_context_providers(registry), ())
        factory.assert_not_called()


class CaptureContextTests(unittest.TestCase):
    def test_returns_first_matching_fragment(self) -> None:
        fragment = ContextFragment(source="stub", text="matched context")
        provider = mock.Mock()
        provider.capture.return_value = fragment
        with mock.patch.object(
            context_providers,
            "available_context_providers",
            return_value=(provider,),
        ):
            result = context_providers.capture_context(URL)

        self.assertIs(result, fragment)
        provider.matches.assert_called_once_with(URL)
        provider.capture.assert_called_once_with(URL)

    def test_provider_failure_is_isolated(self) -> None:
        boom = mock.Mock()
        boom.capture.side_effect = RuntimeError("provider unavailable")
        fragment = ContextFragment(source="stub", text="recovered")
        healthy = mock.Mock()
        healthy.capture.return_value = fragment
        with mock.patch.object(
            context_providers,
            "available_context_providers",
            return_value=(boom, healthy),
        ):
            result = context_providers.capture_context(URL)

        self.assertIs(result, fragment)

    def test_no_match_returns_none(self) -> None:
        provider = mock.Mock()
        provider.matches.return_value = False
        with mock.patch.object(
            context_providers,
            "available_context_providers",
            return_value=(provider,),
        ):
            self.assertIsNone(context_providers.capture_context(URL))
        provider.matches.assert_called_once_with(URL)
        provider.capture.assert_not_called()

    def test_disabled_registry_inspects_nothing(self) -> None:
        with mock.patch.object(
            context_providers, "available_context_providers", return_value=()
        ):
            self.assertIsNone(context_providers.capture_context(URL))


class CaptureTextContextTests(unittest.TestCase):
    def test_returns_first_text_fragment(self) -> None:
        fragment = ContextFragment(source="stub", text="spoken context")
        provider = mock.Mock()
        provider.capture_text.return_value = fragment
        with mock.patch.object(
            context_providers,
            "available_context_providers",
            return_value=(provider,),
        ):
            result = context_providers.capture_text_context("work on owner/repo#42")

        self.assertIs(result, fragment)
        provider.capture_text.assert_called_once_with("work on owner/repo#42")

    def test_text_provider_failure_is_isolated(self) -> None:
        broken = mock.Mock()
        broken.capture_text.side_effect = RuntimeError("unavailable")
        fragment = ContextFragment(source="stub", text="recovered")
        healthy = mock.Mock()
        healthy.capture_text.return_value = fragment
        with mock.patch.object(
            context_providers,
            "available_context_providers",
            return_value=(broken, healthy),
        ):
            self.assertIs(
                context_providers.capture_text_context("work on owner/repo#42"),
                fragment,
            )


class ContextFragmentTests(unittest.TestCase):
    def test_stringifies_to_its_text(self) -> None:
        fragment = ContextFragment(
            source="stub",
            text="rendered context",
            issue_reference="owner/repo#42",
            repository_reference="owner/repo",
            issue_number=42,
            pull_request_number=7,
        )
        self.assertEqual(str(fragment), "rendered context")
        self.assertEqual(fragment.repository_reference, "owner/repo")
        self.assertEqual(fragment.issue_reference, "owner/repo#42")
        self.assertEqual(fragment.issue_number, 42)
        self.assertEqual(fragment.pull_request_number, 7)


if __name__ == "__main__":
    unittest.main()
