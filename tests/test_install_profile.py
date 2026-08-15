from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from local_voice_harness.install_profile import (
    CUDA_PACKAGES,
    NVIDIA_PACKAGE_MARKERS,
    InstallProfileError,
    main,
    resolve_installation_plan,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class InstallationProfileDecisionTests(unittest.TestCase):
    def test_showcase_installs_no_cuda_or_nvidia_packages(self) -> None:
        plan = resolve_installation_plan(profile="showcase")

        self.assertEqual(plan.profile, "showcase")
        self.assertEqual(plan.llm_provider, "venice")
        self.assertEqual(plan.tts_provider, "venice")
        self.assertEqual(plan.dictation_extra, "dictation")
        self.assertEqual(plan.dictation_device, "cpu")
        self.assertEqual(plan.cuda_packages, ())
        self.assertEqual(plan.nvidia_packages(), ())
        self.assertFalse(
            any(
                marker in package.casefold()
                for package in plan.system_packages
                for marker in NVIDIA_PACKAGE_MARKERS
            )
        )

    def test_showcase_omits_local_llm_tts_extras_models_and_services(self) -> None:
        plan = resolve_installation_plan(profile="showcase", llm_provider="local")

        self.assertNotIn("tts", plan.python_extras)
        self.assertNotIn("dictation-cuda", plan.python_extras)
        self.assertFalse(plan.install_tts_extra)
        self.assertFalse(plan.download_qwen)
        self.assertFalse(plan.download_chatterbox)
        self.assertFalse(plan.install_llm_service)

    def test_venice_providers_without_profile_use_showcase_decisions(self) -> None:
        plan = resolve_installation_plan(llm_provider="venice", tts_provider="venice")

        self.assertEqual(plan.profile, "showcase")
        self.assertEqual(plan.dictation_extra, "dictation")
        self.assertEqual(plan.cuda_packages, ())
        self.assertFalse(plan.install_tts_extra)
        self.assertFalse(plan.download_qwen)

    def test_local_cuda_path_preserves_cuda_dictation_and_local_assets(self) -> None:
        plan = resolve_installation_plan(profile="local-cuda")

        self.assertEqual(plan.profile, "local-cuda")
        self.assertEqual(plan.llm_provider, "local")
        self.assertEqual(plan.tts_provider, "local")
        self.assertEqual(plan.dictation_extra, "dictation-cuda")
        self.assertEqual(plan.dictation_device, "cuda")
        self.assertEqual(plan.cuda_packages, CUDA_PACKAGES)
        self.assertIn("cuda", plan.nvidia_packages())
        self.assertTrue(plan.install_tts_extra)
        self.assertTrue(plan.download_qwen)
        self.assertTrue(plan.download_chatterbox)
        self.assertTrue(plan.install_llm_service)
        self.assertIn("tts", plan.python_extras)

    def test_mixed_local_llm_keeps_cuda_and_skips_chatterbox(self) -> None:
        plan = resolve_installation_plan(llm_provider="local", tts_provider="venice")

        self.assertEqual(plan.profile, "local-cuda")
        self.assertEqual(plan.cuda_packages, CUDA_PACKAGES)
        self.assertTrue(plan.download_qwen)
        self.assertTrue(plan.install_llm_service)
        self.assertFalse(plan.install_tts_extra)
        self.assertFalse(plan.download_chatterbox)

    def test_unknown_profile_and_provider_fail_closed(self) -> None:
        with self.assertRaisesRegex(InstallProfileError, "local-cuda, showcase"):
            resolve_installation_plan(profile="gpu-free")
        with self.assertRaisesRegex(InstallProfileError, "LLM provider"):
            resolve_installation_plan(llm_provider="openai")
        with self.assertRaisesRegex(InstallProfileError, "TTS provider"):
            resolve_installation_plan(tts_provider="cloud")
        with self.assertRaisesRegex(InstallProfileError, "LLM device"):
            resolve_installation_plan(llm_provider="local", llm_device="metal")

    def test_cli_env_format_is_shell_evaluable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "eval-plan.sh"
            script.write_text(
                "#!/bin/sh\n"
                'eval "$(python3 -m local_voice_harness.install_profile '
                '--profile showcase --format env)"\n'
                'printf "%s %s %s %s\\n" '
                '"$INSTALL_PROFILE" "$INSTALL_DICTATION_EXTRA" '
                '"$INSTALL_CUDA_PACKAGES" "$INSTALL_TTS_EXTRA"\n'
            )
            result = subprocess.run(
                ["sh", str(script)],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(PROJECT_ROOT / "src"),
                },
            )

        self.assertEqual(result.stdout.strip(), "showcase dictation  0")

    def test_cli_rejects_unknown_profile(self) -> None:
        self.assertEqual(main(["--profile", "mystery"]), 1)

    def test_installer_consults_profile_module_and_documents_showcase(self) -> None:
        installer = (PROJECT_ROOT / "scripts/install.sh").read_text()
        readme = (PROJECT_ROOT / "README.md").read_text()
        installation = (PROJECT_ROOT / "docs/installation.md").read_text()

        self.assertIn("local_voice_harness.install_profile", installer)
        self.assertIn('--extra "$INSTALL_DICTATION_EXTRA"', installer)
        self.assertIn("INSTALL_CUDA_PACKAGES", installer)
        self.assertIn("--profile showcase", installer)
        self.assertIn("PROFILE=showcase", readme)
        self.assertIn("showcase", installation.casefold())
        self.assertIn("local-cuda", installer)
        self.assertIn('requested = os.environ["TTS_CACHE_DEVICE"]', installer)
        self.assertIn('requested == "auto" and torch.cuda.is_available()', installer)
        self.assertIn('else "cpu" if requested == "auto"', installer)


if __name__ == "__main__":
    unittest.main()
