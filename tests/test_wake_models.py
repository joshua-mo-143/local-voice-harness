from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNC_WAKE = PROJECT_ROOT / "scripts" / "sync-wake.sh"


class WakeModelSyncTests(unittest.TestCase):
    def test_wake_sync_restores_missing_model_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout = root / "primary-checkout"
            scripts = checkout / "scripts"
            scripts.mkdir(parents=True)
            sync_wake = scripts / "sync-wake.sh"
            shutil.copy2(SYNC_WAKE, sync_wake)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            uv_record = root / "uv.jsonl"
            download_record = root / "downloads.txt"
            fake_site = root / "site"
            package = fake_site / "openwakeword"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text(
                """
MODELS = {
    "hey_jarvis": {
        "download_url": "https://example.invalid/hey_jarvis_v0.1.tflite"
    }
}
"""
            )
            (package / "utils.py").write_text(
                """
import os
from pathlib import Path


def download_models(model_names):
    package = Path(__file__).parent
    model = package / "resources" / "models" / "hey_jarvis_v0.1.tflite"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_text("downloaded")
    with Path(os.environ["FAKE_DOWNLOAD_RECORD"]).open("a") as record:
        record.write("models:" + ",".join(model_names) + "\\n")


def download_file(url, target_directory):
    model = Path(target_directory) / url.rsplit("/", 1)[-1]
    model.write_text("downloaded")
    with Path(os.environ["FAKE_DOWNLOAD_RECORD"]).open("a") as record:
        record.write("file:" + model.name + "\\n")
"""
            )
            uv = bin_dir / "uv"
            uv.write_text(
                f"""#!{sys.executable}
import json
import os
import sys
from pathlib import Path

environment = Path(os.environ["UV_PROJECT_ENVIRONMENT"])
(environment / "bin").mkdir(parents=True, exist_ok=True)
python = environment / "bin" / "python"
if not python.exists():
    python.symlink_to(sys.executable)
with Path(os.environ["FAKE_UV_RECORD"]).open("a") as record:
    record.write(json.dumps({{"arguments": sys.argv[1:]}}) + "\\n")
"""
            )
            uv.chmod(0o755)
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{bin_dir}{os.pathsep}{environment['PATH']}",
                    "PYTHONPATH": os.pathsep.join(
                        (str(PROJECT_ROOT / "src"), str(fake_site))
                    ),
                    "FAKE_UV_RECORD": str(uv_record),
                    "FAKE_DOWNLOAD_RECORD": str(download_record),
                }
            )

            for _ in range(2):
                process = subprocess.run(
                    [str(sync_wake)],
                    cwd=root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(process.returncode, 0, process.stderr)

            model = package / "resources" / "models" / "hey_jarvis_v0.1.onnx"
            self.assertEqual(model.read_text(), "downloaded")
            self.assertEqual(
                download_record.read_text().splitlines(),
                [
                    "models:hey_jarvis_v0.1",
                    "file:hey_jarvis_v0.1.onnx",
                ],
            )
            invocations = [
                json.loads(line)["arguments"]
                for line in uv_record.read_text().splitlines()
            ]
            self.assertEqual(
                invocations,
                [
                    [
                        "sync",
                        "--project",
                        str(checkout),
                        "--python",
                        "3.11",
                        "--extra",
                        "wake",
                        "--no-dev",
                    ],
                    [
                        "sync",
                        "--project",
                        str(checkout),
                        "--python",
                        "3.11",
                        "--extra",
                        "wake",
                        "--no-dev",
                    ],
                ],
            )


if __name__ == "__main__":
    unittest.main()
