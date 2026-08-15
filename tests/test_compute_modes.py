from __future__ import annotations

import unittest
from dataclasses import replace
from unittest import mock

from local_voice_harness import llm_launcher
from local_voice_harness.diagnostics import checks
from local_voice_harness.diagnostics.model import Severity
from local_voice_harness.install_profile import resolve_installation_plan
from local_voice_harness.tts import server as tts_server
from local_voice_harness.user_config import (
    ComputeDevice,
    default_user_config,
    resolve_local_compute,
)


class ResolveLocalComputeTests(unittest.TestCase):
    def test_cpu_cuda_and_auto_are_deterministic(self) -> None:
        self.assertEqual(
            resolve_local_compute(ComputeDevice.CPU, cuda_available=True, label="LLM"),
            "cpu",
        )
        self.assertEqual(
            resolve_local_compute(ComputeDevice.CUDA, cuda_available=True, label="TTS"),
            "cuda",
        )
        self.assertEqual(
            resolve_local_compute(ComputeDevice.AUTO, cuda_available=True, label="LLM"),
            "cuda",
        )
        self.assertEqual(
            resolve_local_compute(
                ComputeDevice.AUTO, cuda_available=False, label="TTS"
            ),
            "cpu",
        )

    def test_explicit_cuda_fails_clearly_when_unavailable(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "CUDA LLM was requested"):
            resolve_local_compute(ComputeDevice.CUDA, cuda_available=False, label="LLM")
        with self.assertRaisesRegex(RuntimeError, "CUDA TTS was requested"):
            resolve_local_compute(ComputeDevice.CUDA, cuda_available=False, label="TTS")


class LocalLlmComputeTests(unittest.TestCase):
    def _snapshot(self, device: ComputeDevice):
        snapshot = default_user_config()
        return replace(
            snapshot,
            providers=replace(snapshot.providers, llm_provider="local"),
            compute=replace(snapshot.compute, llm_device=device, cuda_device="CUDA3"),
        )

    def test_cpu_command_omits_cuda_device_and_does_not_probe(self) -> None:
        with mock.patch.object(llm_launcher, "llama_cuda_available") as probe:
            command = llm_launcher.command(self._snapshot(ComputeDevice.CPU))

        probe.assert_not_called()
        self.assertIn("--n-gpu-layers", command)
        self.assertEqual(command[command.index("--n-gpu-layers") + 1], "0")
        self.assertNotIn("--device", command)
        self.assertEqual(command[command.index("--flash-attn") + 1], "off")

    def test_explicit_cuda_uses_configured_device(self) -> None:
        command = llm_launcher.command(
            self._snapshot(ComputeDevice.CUDA), cuda_available=True
        )

        self.assertEqual(command[command.index("--n-gpu-layers") + 1], "99")
        self.assertEqual(command[command.index("--device") + 1], "CUDA3")

    def test_explicit_cuda_fails_when_unavailable(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "CUDA LLM was requested"):
            llm_launcher.command(
                self._snapshot(ComputeDevice.CUDA), cuda_available=False
            )

    def test_auto_falls_back_to_cpu_without_device_flag(self) -> None:
        command = llm_launcher.command(
            self._snapshot(ComputeDevice.AUTO), cuda_available=False
        )

        self.assertEqual(command[command.index("--n-gpu-layers") + 1], "0")
        self.assertNotIn("--device", command)


class LocalTtsComputeTests(unittest.TestCase):
    def test_cpu_path_does_not_call_torch_cuda(self) -> None:
        compute = replace(default_user_config().compute, tts_device=ComputeDevice.CPU)
        with mock.patch.object(tts_server, "torch_cuda_available") as probe:
            device = tts_server.resolve_tts_device(compute)

        probe.assert_not_called()
        self.assertEqual(device, "cpu")

    def test_explicit_cuda_fails_when_unavailable(self) -> None:
        compute = replace(default_user_config().compute, tts_device=ComputeDevice.CUDA)
        with self.assertRaisesRegex(RuntimeError, "CUDA TTS was requested"):
            tts_server.resolve_tts_device(compute, cuda_available=False)

    def test_auto_uses_cuda_when_available(self) -> None:
        compute = replace(default_user_config().compute, tts_device=ComputeDevice.AUTO)
        self.assertEqual(
            tts_server.resolve_tts_device(compute, cuda_available=True), "cuda"
        )
        self.assertEqual(
            tts_server.resolve_tts_device(compute, cuda_available=False), "cpu"
        )


class InstallAndDiagnosticComputeTests(unittest.TestCase):
    def test_cpu_local_install_omits_cuda_packages(self) -> None:
        plan = resolve_installation_plan(
            llm_provider="local",
            tts_provider="local",
            llm_device="cpu",
            tts_device="cpu",
            dictation_device="cpu",
        )

        self.assertEqual(plan.cuda_packages, ())
        self.assertEqual(plan.dictation_extra, "dictation")
        self.assertEqual(plan.llm_device, "cpu")
        self.assertEqual(plan.tts_device, "cpu")

    def test_diagnostics_report_configured_and_effective_cpu_modes(self) -> None:
        config = default_user_config()
        snapshot = checks.DiagnosticSnapshot(
            config=replace(
                config,
                providers=replace(
                    config.providers, llm_provider="local", tts_provider="local"
                ),
                compute=replace(
                    config.compute,
                    llm_device=ComputeDevice.CPU,
                    tts_device=ComputeDevice.CPU,
                    dictation_device=ComputeDevice.CPU,
                ),
            ),
            registry=checks.build_integration_registry(config),
        )

        modes = checks.check_compute_modes(snapshot)
        cuda = checks.check_cuda(snapshot)

        self.assertTrue(all(result.severity is Severity.OK for result in modes))
        self.assertTrue(
            any("effective compute=cpu" in result.detail for result in modes)
        )
        self.assertEqual(cuda[0].severity, Severity.OK)
        self.assertIn("CUDA tools were not invoked", cuda[0].detail)

    def test_cpu_cuda_check_does_not_call_nvidia_smi(self) -> None:
        config = default_user_config()
        snapshot = checks.DiagnosticSnapshot(
            config=replace(
                config,
                compute=replace(config.compute, dictation_device=ComputeDevice.CPU),
            ),
            registry=checks.build_integration_registry(config),
        )
        with mock.patch.object(checks, "_which") as which:
            results = checks.check_cuda(snapshot)

        which.assert_not_called()
        self.assertIn("CUDA tools were not invoked", results[0].detail)
