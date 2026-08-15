from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness import service_manager
from local_voice_harness.diagnostics import checks
from local_voice_harness.install_distro import (
    DistroError,
    DistroFamily,
    DistroPlan,
    detect_distro_family,
    discover_install_paths,
    main,
    parse_os_release,
    resolve_distro_plan,
)
from local_voice_harness.install_profile import resolve_installation_plan
from local_voice_harness.integrations.registry import build_integration_registry
from local_voice_harness.user_config import default_user_config

ARCH_RELEASE = 'ID=arch\nID_LIKE=arch\nNAME="Arch Linux"\n'
UBUNTU_RELEASE = 'ID=ubuntu\nID_LIKE=debian\nNAME="Ubuntu"\n'
FEDORA_RELEASE = 'ID=fedora\nNAME="Fedora Linux"\n'
OS_RELEASES = {
    DistroFamily.ARCH: ARCH_RELEASE,
    DistroFamily.DEBIAN: UBUNTU_RELEASE,
    DistroFamily.FEDORA: FEDORA_RELEASE,
}


def _plan(
    family: DistroFamily, *, profile: str = "showcase", checkout: Path
) -> DistroPlan:
    return resolve_distro_plan(
        resolve_installation_plan(profile=profile),
        checkout=checkout,
        os_release=parse_os_release(OS_RELEASES[family]),
        home=checkout.parent / "home",
        environment={"HOME": str(checkout.parent / "home")},
    )


class DistroDetectionTests(unittest.TestCase):
    def test_detects_arch_ubuntu_and_fedora_families(self) -> None:
        self.assertEqual(
            detect_distro_family(parse_os_release(ARCH_RELEASE)), DistroFamily.ARCH
        )
        self.assertEqual(
            detect_distro_family(parse_os_release(UBUNTU_RELEASE)), DistroFamily.DEBIAN
        )
        self.assertEqual(
            detect_distro_family(parse_os_release(FEDORA_RELEASE)), DistroFamily.FEDORA
        )

    def test_unknown_distro_is_rejected(self) -> None:
        with self.assertRaisesRegex(DistroError, "unsupported distribution"):
            detect_distro_family({"ID": "alpine"})


class DistroPackageAdapterTests(unittest.TestCase):
    def test_hosted_installs_omit_cuda_on_every_family(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary) / "checkout"
            checkout.mkdir()
            for family in DistroFamily:
                plan = _plan(family, profile="showcase", checkout=checkout)
                self.assertEqual(plan.cuda_packages, ())
                self.assertEqual(plan.nvidia_packages(), ())
                self.assertNotIn("cuda", plan.packages)
                self.assertNotIn("llama.cpp-cuda", plan.packages)
                self.assertNotIn("nvidia-cuda-toolkit", plan.packages)

    def test_package_names_are_mapped_per_family(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary) / "checkout"
            checkout.mkdir()
            arch = _plan(DistroFamily.ARCH, checkout=checkout)
            debian = _plan(DistroFamily.DEBIAN, checkout=checkout)
            fedora = _plan(DistroFamily.FEDORA, checkout=checkout)

        self.assertEqual(arch.package_manager, "paru")
        self.assertIn("github-cli", arch.packages)
        self.assertIn("libsecret", arch.packages)
        self.assertEqual(debian.package_manager, "apt-get")
        self.assertIn("gh", debian.packages)
        self.assertIn("libnotify-bin", debian.packages)
        self.assertIn("libsecret-tools", debian.packages)
        self.assertIn("gnome-keyring", debian.packages)
        self.assertTrue(debian.uv_bootstrap)
        self.assertEqual(fedora.package_manager, "dnf")
        self.assertIn("gh", fedora.packages)
        self.assertIn("gnome-keyring", fedora.packages)
        self.assertTrue(fedora.uv_bootstrap)
        self.assertFalse(arch.uv_bootstrap)

    def test_local_cuda_cuda_packages_stay_arch_specific(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary) / "checkout"
            checkout.mkdir()
            arch = _plan(DistroFamily.ARCH, profile="local-cuda", checkout=checkout)
            debian = _plan(DistroFamily.DEBIAN, profile="local-cuda", checkout=checkout)

        self.assertIn("cuda", arch.cuda_packages)
        self.assertEqual(debian.cuda_packages, ())
        self.assertIn("cuda", debian.skipped_packages)
        self.assertIn("llama.cpp-cuda", debian.skipped_packages)

    def test_resolving_the_same_inputs_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary) / "checkout"
            checkout.mkdir()
            first = _plan(DistroFamily.DEBIAN, checkout=checkout)
            second = _plan(DistroFamily.DEBIAN, checkout=checkout)
        self.assertEqual(first, second)


class InstallPathDiscoveryTests(unittest.TestCase):
    def test_paths_follow_checkout_and_xdg_instead_of_maintainer_home(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout = root / "src" / "voice-harness"
            checkout.mkdir(parents=True)
            home = root / "user"
            paths = discover_install_paths(
                checkout=checkout,
                home=home,
                environment={
                    "HOME": str(home),
                    "XDG_CONFIG_HOME": str(root / "xdg-config"),
                },
                chatterbox_dir=root / "tts",
            )

        self.assertEqual(paths.checkout, checkout)
        self.assertEqual(paths.user_bin, home / ".local" / "bin")
        self.assertEqual(paths.voice_harness, home / ".local" / "bin" / "voice-harness")
        self.assertEqual(
            paths.systemd_user_dir, root / "xdg-config" / "systemd" / "user"
        )
        self.assertEqual(paths.chatterbox_dir, root / "tts")
        self.assertNotIn("local-voice-harness", str(paths.checkout))

    def test_installer_relinks_stale_unit_checkout_and_preserves_directories(
        self,
    ) -> None:
        installer = (
            Path(__file__).resolve().parents[1] / "scripts" / "install.sh"
        ).read_text()

        self.assertIn('if [[ -L "$UNIT_CHECKOUT" ]]', installer)
        self.assertIn('ln -sfn "$INSTALL_CHECKOUT" "$UNIT_CHECKOUT"', installer)
        self.assertIn('elif [[ -e "$UNIT_CHECKOUT" ]]', installer)
        self.assertIn("refusing to replace it", installer)
        self.assertIn("exit 1", installer)
        self.assertIn("paru is required on Arch", installer)
        self.assertNotIn('PACKAGE_COMMAND="sudo pacman', installer)

    def test_runtime_systemd_path_matches_discovered_xdg_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            xdg_config = root / "xdg-config"
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from local_voice_harness.config import SYSTEMD_USER_DIR; "
                        "print(SYSTEMD_USER_DIR)"
                    ),
                ],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "XDG_CONFIG_HOME": str(xdg_config)},
            )

        self.assertEqual(
            result.stdout.strip(),
            str(xdg_config / "systemd" / "user"),
        )


class DistroServiceLifecycleTests(unittest.TestCase):
    def test_service_install_diagnostics_and_uninstall_for_each_family(self) -> None:
        config = default_user_config()
        snapshot = service_manager.ServiceManagementSnapshot(
            config,
            build_integration_registry(config),
        )
        supervisor = mock.Mock()
        supervisor.available.return_value = True
        supervisor.is_active.return_value = "inactive"
        supervisor.show.return_value = {}
        for family in DistroFamily:
            with (
                self.subTest(family=family.value),
                tempfile.TemporaryDirectory() as temporary,
            ):
                checkout = Path(temporary) / "checkout"
                checkout.mkdir()
                plan = _plan(family, checkout=checkout)
                systemd = plan.paths.systemd_user_dir
                with (
                    mock.patch.object(service_manager, "SYSTEMD_USER_DIR", systemd),
                    mock.patch.object(service_manager, "systemctl"),
                    mock.patch.object(checks, "user_services", return_value=supervisor),
                    mock.patch.object(checks, "SYSTEMD_USER_DIR", systemd),
                    mock.patch.object(checks, "_systemctl_show", return_value={}),
                ):
                    service_manager.install_services(force=True, snapshot=snapshot)
                    installed = sorted(path.name for path in systemd.glob("*.service"))
                    self.assertIn("voice-harness-wake.service", installed)
                    diagnostics = checks.check_systemd_units()
                    self.assertTrue(
                        any(result.name.startswith("unit:") for result in diagnostics)
                    )
                    service_manager.uninstall_services(
                        include_herdr=False, snapshot=snapshot
                    )
                    remaining = [
                        path.name
                        for path in systemd.glob("*.service")
                        if path.read_text() == service_manager.unit_text(path.name)
                    ]
                    self.assertEqual(remaining, [])


class DistroCliTests(unittest.TestCase):
    def test_cli_emits_env_for_ubuntu_showcase(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary) / "checkout"
            checkout.mkdir()
            release = Path(temporary) / "os-release"
            release.write_text(UBUNTU_RELEASE)
            with mock.patch.dict(
                os.environ, {"HOME": str(Path(temporary) / "home")}, clear=False
            ):
                code = main(
                    [
                        "--checkout",
                        str(checkout),
                        "--os-release",
                        str(release),
                        "--profile",
                        "showcase",
                        "--format",
                        "env",
                    ]
                )
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
