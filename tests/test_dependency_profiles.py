from __future__ import annotations

import subprocess
import tomllib
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolved_profile(extra: str) -> str:
    result = subprocess.run(
        [
            "uv",
            "export",
            "--locked",
            "--no-dev",
            "--no-emit-project",
            "--extra",
            extra,
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.casefold()


class DictationDependencyProfileTests(unittest.TestCase):
    def test_every_declared_extra_resolves_from_the_lockfile(self) -> None:
        from local_voice_harness.production_extras import declared_extras

        for extra in declared_extras(PROJECT_ROOT / "pyproject.toml"):
            with self.subTest(extra=extra):
                resolved = _resolved_profile(extra)
                self.assertTrue(resolved.strip())

    def test_cpu_profile_excludes_gpu_and_nvidia_packages(self) -> None:
        resolved = _resolved_profile("dictation")

        self.assertIn("onnxruntime==", resolved)
        self.assertNotIn("onnxruntime-gpu", resolved)
        self.assertNotIn("nvidia-", resolved)

    def test_cuda_profile_selects_gpu_onnx_runtime(self) -> None:
        resolved = _resolved_profile("dictation-cuda")

        self.assertIn("onnxruntime-gpu==", resolved)

    def test_wake_uses_onnx_and_limits_tflite_to_supported_python(self) -> None:
        metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
        overrides = metadata["tool"]["uv"]["override-dependencies"]
        daemon = (
            PROJECT_ROOT / "src" / "local_voice_harness" / "wake" / "daemon.py"
        ).read_text()

        self.assertIn(
            "tflite-runtime==2.14.0; python_version < '3.12'",
            overrides,
        )
        self.assertIn('inference_framework="onnx"', daemon)


if __name__ == "__main__":
    unittest.main()
