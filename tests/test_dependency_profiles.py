from __future__ import annotations

import subprocess
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
    def test_cpu_profile_excludes_gpu_and_nvidia_packages(self) -> None:
        resolved = _resolved_profile("dictation")

        self.assertIn("onnxruntime==", resolved)
        self.assertNotIn("onnxruntime-gpu", resolved)
        self.assertNotIn("nvidia-", resolved)

    def test_cuda_profile_selects_gpu_onnx_runtime(self) -> None:
        resolved = _resolved_profile("dictation-cuda")

        self.assertIn("onnxruntime-gpu==", resolved)


if __name__ == "__main__":
    unittest.main()
