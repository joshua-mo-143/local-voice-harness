from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from local_voice_harness import credentials


def _completed(
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class VeniceCredentialTests(unittest.TestCase):
    def test_store_sends_key_only_over_stdin(self) -> None:
        with (
            mock.patch.object(
                credentials.shutil, "which", return_value="/usr/bin/secret-tool"
            ),
            mock.patch.object(
                credentials.subprocess, "run", return_value=_completed()
            ) as run,
        ):
            credentials.store_venice_api_key("venice-secret")

        command = run.call_args.args[0]
        self.assertNotIn("venice-secret", command)
        self.assertEqual(run.call_args.kwargs["input"], "venice-secret")
        self.assertEqual(command[:2], ["/usr/bin/secret-tool", "store"])
        self.assertEqual(command[-4:], list(credentials.SECRET_ATTRIBUTES))

    def test_lookup_returns_secret_service_value(self) -> None:
        with (
            mock.patch.object(
                credentials.shutil, "which", return_value="/usr/bin/secret-tool"
            ),
            mock.patch.object(
                credentials.subprocess,
                "run",
                return_value=_completed(stdout="venice-secret\n"),
            ) as run,
        ):
            value = credentials.get_venice_api_key()

        self.assertEqual(value, "venice-secret")
        self.assertEqual(
            run.call_args.args[0],
            ["/usr/bin/secret-tool", "lookup", *credentials.SECRET_ATTRIBUTES],
        )

    def test_secret_service_availability_is_capability_not_invocation(self) -> None:
        with mock.patch.object(credentials.shutil, "which", return_value=None):
            self.assertFalse(credentials.secret_service_available())
            self.assertFalse(credentials.SecretServiceStore().available())
        with mock.patch.object(
            credentials.shutil, "which", return_value="/usr/bin/secret-tool"
        ):
            self.assertTrue(credentials.secret_service_available())

    def test_missing_tool_and_missing_key_are_clear_errors(self) -> None:
        with (
            mock.patch.object(credentials.shutil, "which", return_value=None),
            self.assertRaisesRegex(credentials.CredentialError, "install libsecret"),
        ):
            credentials.get_venice_api_key()

        with (
            mock.patch.object(
                credentials.shutil,
                "which",
                return_value="/usr/bin/secret-tool",
            ),
            mock.patch.object(
                credentials.subprocess,
                "run",
                return_value=_completed(returncode=1),
            ),
            self.assertRaisesRegex(credentials.CredentialError, "credentials set"),
        ):
            credentials.get_venice_api_key()

        with (
            mock.patch.object(
                credentials.shutil,
                "which",
                return_value="/usr/bin/secret-tool",
            ),
            mock.patch.object(credentials.subprocess, "run", return_value=_completed()),
            self.assertRaisesRegex(credentials.CredentialError, "credentials set"),
        ):
            credentials.get_venice_api_key()

    def test_missing_secret_service_provider_has_install_guidance(self) -> None:
        with (
            mock.patch.object(
                credentials.shutil,
                "which",
                return_value="/usr/bin/secret-tool",
            ),
            mock.patch.object(
                credentials.subprocess,
                "run",
                return_value=_completed(
                    returncode=1,
                    stderr="secret-tool: The name is not activatable\n",
                ),
            ),
            self.assertRaisesRegex(credentials.CredentialError, "oo7"),
        ):
            credentials.store_venice_api_key("venice-secret")

    def test_delete_clears_only_the_harness_venice_item(self) -> None:
        with (
            mock.patch.object(
                credentials.shutil, "which", return_value="/usr/bin/secret-tool"
            ),
            mock.patch.object(
                credentials.subprocess, "run", return_value=_completed()
            ) as run,
        ):
            credentials.delete_venice_api_key()

        self.assertEqual(
            run.call_args.args[0],
            ["/usr/bin/secret-tool", "clear", *credentials.SECRET_ATTRIBUTES],
        )


if __name__ == "__main__":
    unittest.main()
