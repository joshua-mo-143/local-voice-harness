from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness import production_extras
from local_voice_harness.config import PROJECT_ROOT


class ProductionExtraSpecTests(unittest.TestCase):
    def test_every_declared_extra_has_an_install_and_entry_point_spec(self) -> None:
        pyproject = PROJECT_ROOT / "pyproject.toml"
        extras = production_extras.declared_extras(pyproject)
        self.assertEqual(extras, tuple(sorted(production_extras.EXTRA_SPECS)))
        self.assertEqual(production_extras.missing_extra_specs(pyproject), ())
        for extra, spec in production_extras.EXTRA_SPECS.items():
            with self.subTest(extra=extra):
                self.assertTrue(spec.imports)
                self.assertTrue(spec.commands)

    def test_core_and_extra_commands_use_check_or_help_without_model_load(self) -> None:
        for name, arguments in (
            *production_extras.CORE_COMMANDS,
            *(
                command
                for spec in production_extras.EXTRA_SPECS.values()
                for command in spec.commands
            ),
        ):
            with self.subTest(entry_point=name):
                self.assertIn(arguments[0], {"--check", "--help"})


class ProductionExtraCommandTests(unittest.TestCase):
    def test_export_and_install_commands_keep_extras_in_separate_environments(
        self,
    ) -> None:
        wheel = Path("/tmp/local_voice_harness-0.1.0-py3-none-any.whl")
        python = Path("/tmp/extra-wake/bin/python")
        requirements = Path("/tmp/wake.txt")
        self.assertEqual(
            production_extras.export_command("wake"),
            [
                "uv",
                "export",
                "--locked",
                "--no-dev",
                "--no-emit-project",
                "--extra",
                "wake",
            ],
        )
        self.assertEqual(
            production_extras.install_commands(python, wheel, requirements),
            [
                ["uv", "pip", "install", "--python", str(python), str(wheel)],
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    "-r",
                    str(requirements),
                ],
            ],
        )
        self.assertEqual(
            production_extras.import_command(python, ("openwakeword", "numpy")),
            [str(python), "-c", "import openwakeword; import numpy"],
        )
        self.assertEqual(
            production_extras.show_distribution_command(python, "onnxruntime-gpu"),
            ["uv", "pip", "show", "--python", str(python), "onnxruntime-gpu"],
        )

    def test_unknown_extra_fails_before_installing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(SystemExit) as raised:
                production_extras.check_environments(
                    project=PROJECT_ROOT,
                    python="3.11",
                    extras=("not-an-extra",),
                    work=Path(temporary),
                )
        self.assertIn("no extra spec", str(raised.exception))

    def test_check_environments_installs_core_then_each_extra(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            wheel = work / "dist" / "pkg.whl"
            recorded: list[list[str]] = []

            def fake_run(command: list[str], **_options: object) -> mock.Mock:
                recorded.append(list(command))
                if command[:3] == ["uv", "build", "--wheel"]:
                    wheel.parent.mkdir(parents=True, exist_ok=True)
                    wheel.write_text("wheel")
                if command[:2] == ["uv", "export"]:
                    Path(command[-1]).write_text("deps\n")
                if command[:2] == ["uv", "venv"]:
                    destination = Path(command[-1])
                    (destination / "bin").mkdir(parents=True)
                    (destination / "bin" / "python").write_text("python")
                    for name, _arguments in (
                        *production_extras.CORE_COMMANDS,
                        *production_extras.EXTRA_SPECS["wake"].commands,
                    ):
                        (destination / "bin" / name).write_text(name)
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch.object(production_extras, "_run", side_effect=fake_run):
                production_extras.check_environments(
                    project=PROJECT_ROOT,
                    python="3.11",
                    extras=("wake",),
                    work=work,
                )

        exported = [command for command in recorded if command[:2] == ["uv", "export"]]
        self.assertEqual(exported[0][-2:], ["-o", str(work / "core.txt")])
        self.assertIn("--extra", exported[1])
        self.assertIn("wake", exported[1])
        self.assertTrue(any("voice-harness-wake" in command[0] for command in recorded))
        self.assertTrue(
            any(
                command[-1] == "import openwakeword; import numpy"
                for command in recorded
            )
        )


if __name__ == "__main__":
    unittest.main()
