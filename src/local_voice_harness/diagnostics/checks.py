from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

from ..components import llm_ready
from ..config import (
    DEFAULT_SOURCE,
    JOBS_DIR,
    PROJECT_ROOT,
    RUNTIME,
    SERVICE_FILES,
    START_SERVICES,
    STATE_DIR,
    STT_SOCKET,
    SYSTEMD_USER_DIR,
    TTS_SOCKET,
    BackendConfigurationError,
    load_backend_settings,
)
from ..credentials import CredentialError, get_venice_api_key
from ..errors import HarnessError
from ..integrations.herdr import HERDR_BIN, HerdrClient, HerdrError
from ..integrations.registry import capability_statuses
from ..ipc import socket_ready
from .model import CheckResult, Repair, Severity

Check = Callable[[], list[CheckResult]]

# A worker-owned job that has not advanced for this long is treated as stuck.
STUCK_JOB_SECONDS = 60 * 60
# systemd rate-limits restarts at this burst; matches COMMON_SECURITY_POLICY.
RESTART_LOOP_THRESHOLD = 5

MODEL_FILE = PROJECT_ROOT / "models" / "Qwen3.5-4B-Q4_K_M.gguf"
HUGGINGFACE_CACHE = Path.home() / ".cache" / "huggingface"

_ACTIVE_JOB_STATUSES = frozenset(
    {"queued", "routing", "running", "reconciling", "awaiting_user", "blocked"}
)
_WORKER_JOB_STATUSES = frozenset({"routing", "running", "reconciling"})
_ATTENTION_JOB_STATUSES = frozenset({"awaiting_user", "blocked"})

REQUIRED_EXECUTABLES: tuple[tuple[str, str], ...] = (
    ("pw-record", "record microphone audio through PipeWire"),
    ("pw-play", "play synthesized speech through PipeWire"),
    ("llama-server", "serve the Qwen conversational model with llama.cpp"),
    ("ffmpeg", "adjust Venice speech speed without changing pitch"),
    ("uv", "manage reproducible Python environments"),
    ("herdr", "manage Cursor agents"),
    ("agent", "run delegated work through the Cursor CLI"),
)
OPTIONAL_EXECUTABLES: tuple[tuple[str, str], ...] = (
    ("gh", "read focused GitHub issue/PR context and create requested forks"),
    ("rofi", "select repositories and confirm clone URLs"),
)
INSTALL_HINTS: dict[str, str] = {
    "pw-record": "paru -S --needed pipewire",
    "pw-play": "paru -S --needed pipewire",
    "llama-server": "paru -S --needed cuda llama.cpp-cuda",
    "ffmpeg": "paru -S --needed ffmpeg",
    "uv": "paru -S --needed uv",
    "herdr": "curl -fsSL https://herdr.dev/install.sh | sh",
    "agent": "curl https://cursor.com/install -fsS | bash",
    "gh": "paru -S --needed github-cli",
    "rofi": "paru -S --needed rofi",
    "xdotool": "paru -S --needed xdotool",
    "xclip": "paru -S --needed xclip",
    "wtype": "paru -S --needed wtype",
    "wl-copy": "paru -S --needed wl-clipboard",
    "wl-paste": "paru -S --needed wl-clipboard",
    "wpctl": "paru -S --needed wireplumber",
    "nvidia-smi": "install the proprietary NVIDIA driver for your kernel",
}


def _which(name: str) -> str | None:
    return shutil.which(name)


def _run(
    command: Sequence[str], *, timeout: float = 5.0
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _systemctl_show(name: str, properties: Sequence[str]) -> dict[str, str]:
    command = ["systemctl", "--user", "show", name]
    command.extend(f"--property={prop}" for prop in properties)
    process = _run(command)
    values: dict[str, str] = {}
    if process is None or process.returncode:
        return values
    for line in process.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def _service_active_state(name: str) -> str:
    return _systemctl_show(name, ["ActiveState"]).get("ActiveState", "unknown")


def _restart_service_repair(name: str) -> Repair:
    def action() -> str:
        process = _run(["systemctl", "--user", "restart", name], timeout=30)
        if process is None:
            raise HarnessError(f"could not run systemctl --user restart {name}")
        if process.returncode:
            raise HarnessError(
                process.stderr.strip() or f"systemctl --user restart {name} failed"
            )
        return f"restarted {name}"

    return Repair(summary=f"restart {name}", action=action)


def _remove_stale_socket_repair(path: Path) -> Repair:
    def action() -> str:
        try:
            if path.is_socket():
                path.unlink()
                return f"removed stale socket {path}"
        except OSError as exc:
            raise HarnessError(f"could not remove {path}: {exc}") from exc
        return f"{path} is no longer a stale socket; nothing to do"

    return Repair(summary=f"remove stale socket {path}", action=action)


def check_required_executables() -> list[CheckResult]:
    try:
        settings = load_backend_settings()
    except BackendConfigurationError:
        settings = None
    results: list[CheckResult] = []
    for name, purpose in REQUIRED_EXECUTABLES:
        if name == "llama-server" and settings is not None:
            if settings.llm_provider == "venice":
                continue
        if name == "ffmpeg" and settings is not None:
            if settings.tts_provider != "venice":
                continue
        location = _which(name)
        if location is not None:
            results.append(
                CheckResult(
                    name=f"executable:{name}",
                    category="executables",
                    severity=Severity.OK,
                    detail=f"{name} found at {location}",
                )
            )
        else:
            results.append(
                CheckResult(
                    name=f"executable:{name}",
                    category="executables",
                    severity=Severity.FATAL,
                    detail=f"{name} is not on PATH; required to {purpose}",
                    suggestion=INSTALL_HINTS.get(name),
                )
            )
    return results


def check_backend_configuration() -> list[CheckResult]:
    try:
        settings = load_backend_settings()
        if "venice" in {settings.llm_provider, settings.tts_provider}:
            get_venice_api_key()
    except (BackendConfigurationError, CredentialError) as exc:
        return [
            CheckResult(
                name="configuration:backends",
                category="configuration",
                severity=Severity.FATAL,
                detail=str(exc),
                suggestion="see README backend configuration",
            )
        ]
    return [
        CheckResult(
            name="configuration:backends",
            category="configuration",
            severity=Severity.OK,
            detail=(
                f"LLM provider={settings.llm_provider} model={settings.llm_model}; "
                f"TTS provider={settings.tts_provider} model={settings.tts_model}"
            ),
        )
    ]


def check_optional_executables() -> list[CheckResult]:
    results: list[CheckResult] = []
    for name, purpose in OPTIONAL_EXECUTABLES:
        location = _which(name)
        if location is not None:
            results.append(
                CheckResult(
                    name=f"executable:{name}",
                    category="executables",
                    severity=Severity.OK,
                    detail=f"{name} found at {location}",
                )
            )
        else:
            results.append(
                CheckResult(
                    name=f"executable:{name}",
                    category="executables",
                    severity=Severity.WARNING,
                    detail=f"{name} is not on PATH; needed to {purpose}",
                    suggestion=INSTALL_HINTS.get(name),
                )
            )
    return results


def check_focus_automation() -> list[CheckResult]:
    """One complete X11 or Wayland stack enables focused-window dictation."""

    x11 = {tool: _which(tool) is not None for tool in ("xdotool", "xclip")}
    wayland = {
        tool: _which(tool) is not None for tool in ("wtype", "wl-copy", "wl-paste")
    }
    x11_ready = all(x11.values())
    wayland_ready = all(wayland.values())
    if x11_ready or wayland_ready:
        stacks = []
        if x11_ready:
            stacks.append("X11 (xdotool, xclip)")
        if wayland_ready:
            stacks.append("Wayland (wtype, wl-clipboard)")
        return [
            CheckResult(
                name="focus-automation",
                category="executables",
                severity=Severity.OK,
                detail=f"focused-window automation available: {', '.join(stacks)}",
            )
        ]
    missing_x11 = sorted(tool for tool, present in x11.items() if not present)
    missing_wayland = sorted(tool for tool, present in wayland.items() if not present)
    return [
        CheckResult(
            name="focus-automation",
            category="executables",
            severity=Severity.WARNING,
            detail=(
                "no complete focused-window automation stack; dictation insertion "
                f"will fall back to stdout. Missing X11 tools: {missing_x11}; "
                f"missing Wayland tools: {missing_wayland}"
            ),
            suggestion=(
                "paru -S --needed xdotool xclip  # X11, or: "
                "paru -S --needed wtype wl-clipboard  # Wayland"
            ),
        )
    ]


def _python_environments() -> tuple[tuple[str, Path, Path, str], ...]:
    home = Path.home()
    return (
        (
            "wake/management (.venv)",
            PROJECT_ROOT / ".venv",
            PROJECT_ROOT / ".venv" / "bin" / "voice-harness-wake",
            "uv sync --python 3.11 --extra wake --no-dev",
        ),
        (
            "dictation (.venv-dictation)",
            PROJECT_ROOT / ".venv-dictation",
            PROJECT_ROOT / ".venv-dictation" / "bin" / "voice-harness-dictation",
            "env UV_PROJECT_ENVIRONMENT=.venv-dictation "
            "uv sync --python 3.11 --extra dictation --no-dev",
        ),
        (
            "chatterbox tts",
            home / "chatterbox-audition" / ".venv",
            home / "chatterbox-audition" / ".venv" / "bin" / "voice-harness-tts",
            'UV_PROJECT_ENVIRONMENT="$HOME/chatterbox-audition/.venv" '
            "uv sync --python 3.11 --extra tts --no-dev",
        ),
    )


def check_python_environments() -> list[CheckResult]:
    results: list[CheckResult] = []
    for label, venv_dir, executable, suggestion in _python_environments():
        if executable.exists():
            results.append(
                CheckResult(
                    name=f"venv:{label}",
                    category="python-env",
                    severity=Severity.OK,
                    detail=f"{label} environment is present ({executable})",
                )
            )
        elif venv_dir.exists():
            results.append(
                CheckResult(
                    name=f"venv:{label}",
                    category="python-env",
                    severity=Severity.WARNING,
                    detail=(
                        f"{label} directory exists but its console script is missing "
                        f"({executable}); the environment may be incomplete"
                    ),
                    suggestion=suggestion,
                )
            )
        else:
            results.append(
                CheckResult(
                    name=f"venv:{label}",
                    category="python-env",
                    severity=Severity.FATAL,
                    detail=f"{label} environment is missing ({venv_dir})",
                    suggestion=suggestion,
                )
            )
    return results


def check_model_file() -> list[CheckResult]:
    try:
        if load_backend_settings().llm_provider == "venice":
            return [
                CheckResult(
                    name="model:qwen",
                    category="models",
                    severity=Severity.OK,
                    detail="local Qwen model is not required by the Venice LLM backend",
                )
            ]
    except BackendConfigurationError:
        pass
    if MODEL_FILE.is_file():
        try:
            size_gb = MODEL_FILE.stat().st_size / 1024**3
        except OSError:
            size_gb = 0.0
        return [
            CheckResult(
                name="model:qwen",
                category="models",
                severity=Severity.OK,
                detail=f"Qwen model present at {MODEL_FILE} ({size_gb:.1f} GiB)",
            )
        ]
    return [
        CheckResult(
            name="model:qwen",
            category="models",
            severity=Severity.FATAL,
            detail=f"Qwen model is missing at {MODEL_FILE}",
            suggestion=(
                "hf download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf "
                "--local-dir models"
            ),
        )
    ]


def check_model_caches() -> list[CheckResult]:
    if HUGGINGFACE_CACHE.is_dir():
        return [
            CheckResult(
                name="cache:huggingface",
                category="models",
                severity=Severity.OK,
                detail=f"Hugging Face cache present at {HUGGINGFACE_CACHE}",
            )
        ]
    try:
        local_tts = load_backend_settings().tts_provider == "local"
    except BackendConfigurationError:
        local_tts = True
    cache_users = (
        "Chatterbox runs offline and Parakeet downloads on first use"
        if local_tts
        else "Parakeet downloads on first use"
    )
    return [
        CheckResult(
            name="cache:huggingface",
            category="models",
            severity=Severity.WARNING,
            detail=(f"Hugging Face cache {HUGGINGFACE_CACHE} is absent; {cache_users}"),
            suggestion="see README steps 2 and 3 to pre-download the model caches",
        )
    ]


def check_cuda() -> list[CheckResult]:
    try:
        settings = load_backend_settings()
    except BackendConfigurationError:
        settings = None
    selected_models = ["STT"]
    if settings is None or settings.llm_provider == "local":
        selected_models.append("LLM")
    if settings is None or settings.tts_provider == "local":
        selected_models.append("TTS")
    model_label = "/".join(selected_models)
    if _which("nvidia-smi") is None:
        return [
            CheckResult(
                name="gpu:cuda",
                category="gpu",
                severity=Severity.WARNING,
                detail=(
                    f"nvidia-smi not found; GPU acceleration for {model_label} is "
                    "unavailable and CPU fallback is substantially slower"
                ),
                suggestion=INSTALL_HINTS.get("nvidia-smi"),
            )
        ]
    process = _run(["nvidia-smi"], timeout=10)
    if process is None or process.returncode:
        detail = "nvidia-smi failed to query the GPU"
        if process is not None and process.stderr.strip():
            detail = f"{detail}: {process.stderr.strip().splitlines()[0]}"
        return [
            CheckResult(
                name="gpu:cuda",
                category="gpu",
                severity=Severity.WARNING,
                detail=detail,
                suggestion="nvidia-smi  # confirm the driver is loaded",
            )
        ]
    return [
        CheckResult(
            name="gpu:cuda",
            category="gpu",
            severity=Severity.OK,
            detail=(
                f"nvidia-smi reports a working GPU for {model_label}; "
                "the local LLM service uses CUDA0 when selected"
            ),
        )
    ]


def check_pipewire_devices() -> list[CheckResult]:
    configured_source = os.environ.get("VOICE_HARNESS_SOURCE", DEFAULT_SOURCE)
    if _which("wpctl") is None:
        return [
            CheckResult(
                name="audio:pipewire",
                category="audio",
                severity=Severity.WARNING,
                detail=(
                    "wpctl not found; cannot enumerate PipeWire capture/playback "
                    f"devices. Configured source: {configured_source}"
                ),
                suggestion=INSTALL_HINTS.get("wpctl"),
            )
        ]
    process = _run(["wpctl", "status"], timeout=5)
    if process is None or process.returncode:
        return [
            CheckResult(
                name="audio:pipewire",
                category="audio",
                severity=Severity.WARNING,
                detail=(
                    "wpctl status failed; PipeWire may not be running for this "
                    f"session. Configured source: {configured_source}"
                ),
                suggestion="systemctl --user status pipewire wireplumber",
            )
        ]
    return [
        CheckResult(
            name="audio:pipewire",
            category="audio",
            severity=Severity.OK,
            detail=(
                "PipeWire is responding to wpctl; configured capture source is "
                f"{configured_source} (override with VOICE_HARNESS_SOURCE)"
            ),
        )
    ]


def check_systemd_units() -> list[CheckResult]:
    results: list[CheckResult] = []
    for name in SERVICE_FILES:
        always_on = name in START_SERVICES
        installed = (SYSTEMD_USER_DIR / name).exists()
        properties = _systemctl_show(
            name, ["ActiveState", "SubState", "Result", "NRestarts", "UnitFileState"]
        )
        load_failed = not installed and not properties
        if load_failed:
            results.append(
                CheckResult(
                    name=f"unit:{name}",
                    category="systemd",
                    severity=Severity.FATAL,
                    detail=f"{name} is not installed",
                    suggestion="voice-harness services install",
                )
            )
            continue

        active = properties.get("ActiveState", "unknown")
        sub = properties.get("SubState", "unknown")
        result = properties.get("Result", "unknown")
        unit_file_state = properties.get("UnitFileState", "unknown")
        try:
            restarts = int(properties.get("NRestarts", "0") or "0")
        except ValueError:
            restarts = 0

        if restarts >= RESTART_LOOP_THRESHOLD:
            results.append(
                CheckResult(
                    name=f"unit:{name}",
                    category="systemd",
                    severity=Severity.FATAL,
                    detail=(
                        f"{name} has restarted {restarts} times "
                        f"(result={result}); it is likely crash-looping"
                    ),
                    suggestion=f"voice-harness services logs  # inspect {name}",
                    repair=_restart_service_repair(name),
                )
            )
            continue
        if active == "failed" or result not in {"success", ""}:
            results.append(
                CheckResult(
                    name=f"unit:{name}",
                    category="systemd",
                    severity=Severity.FATAL,
                    detail=f"{name} is failed (result={result}, sub={sub})",
                    suggestion=f"voice-harness services logs  # inspect {name}",
                    repair=_restart_service_repair(name),
                )
            )
            continue
        if always_on and unit_file_state not in {"enabled", "enabled-runtime"}:
            results.append(
                CheckResult(
                    name=f"unit:{name}",
                    category="systemd",
                    severity=Severity.WARNING,
                    detail=(f"{name} should start at login but is {unit_file_state!r}"),
                    suggestion="voice-harness services install",
                )
            )
            continue
        if always_on and active != "active":
            results.append(
                CheckResult(
                    name=f"unit:{name}",
                    category="systemd",
                    severity=Severity.WARNING,
                    detail=f"{name} is an always-on service but is {active}",
                    suggestion="voice-harness services start",
                    repair=_restart_service_repair(name),
                )
            )
            continue
        results.append(
            CheckResult(
                name=f"unit:{name}",
                category="systemd",
                severity=Severity.OK,
                detail=(
                    f"{name} active={active} sub={sub} restarts={restarts} "
                    f"({unit_file_state})"
                ),
            )
        )
    return results


def _socket_result(
    *,
    name: str,
    label: str,
    socket_path: Path,
    unit: str,
    always_on: bool,
) -> CheckResult:
    if socket_ready(socket_path):
        return CheckResult(
            name=name,
            category="runtime",
            severity=Severity.OK,
            detail=f"{label} socket {socket_path} is responding",
        )
    active = _service_active_state(unit)
    if active == "active":
        return CheckResult(
            name=name,
            category="runtime",
            severity=Severity.WARNING,
            detail=(
                f"{label} socket {socket_path} is not responding although {unit} "
                "is active"
            ),
            suggestion=f"voice-harness services logs  # inspect {unit}",
            repair=_restart_service_repair(unit),
        )
    if socket_path.exists():
        return CheckResult(
            name=name,
            category="runtime",
            severity=Severity.WARNING,
            detail=(
                f"{label} socket {socket_path} is stale: the file exists but {unit} "
                f"is {active}"
            ),
            suggestion="voice-harness services start",
            repair=_remove_stale_socket_repair(socket_path),
        )
    severity = Severity.WARNING if always_on else Severity.OK
    note = "not running" if always_on else "not running (starts on demand)"
    return CheckResult(
        name=name,
        category="runtime",
        severity=severity,
        detail=f"{label} socket absent; {unit} is {active} ({note})",
        suggestion="voice-harness services start" if always_on else None,
    )


def check_runtime_sockets() -> list[CheckResult]:
    try:
        settings = load_backend_settings()
    except BackendConfigurationError:
        settings = None
    results = [
        _socket_result(
            name="socket:stt",
            label="STT (dictation)",
            socket_path=STT_SOCKET,
            unit="dictation.service",
            always_on=True,
        ),
        _socket_result(
            name="socket:tts",
            label=(f"TTS ({settings.tts_provider})" if settings is not None else "TTS"),
            socket_path=TTS_SOCKET,
            unit="voice-harness-tts.service",
            always_on=False,
        ),
    ]

    llm_unit = "voice-harness-llm.service"
    if settings is not None and llm_ready():
        results.append(
            CheckResult(
                name="socket:llm",
                category="runtime",
                severity=Severity.OK,
                detail=(
                    "Venice LLM credentials are available"
                    if settings is not None and settings.llm_provider == "venice"
                    else "LLM health endpoint is responding on 127.0.0.1:8090"
                ),
            )
        )
    else:
        active = _service_active_state(llm_unit)
        if active == "active":
            results.append(
                CheckResult(
                    name="socket:llm",
                    category="runtime",
                    severity=Severity.WARNING,
                    detail=(
                        f"LLM health endpoint is down although {llm_unit} is active"
                    ),
                    suggestion=f"voice-harness services logs  # inspect {llm_unit}",
                    repair=_restart_service_repair(llm_unit),
                )
            )
        else:
            results.append(
                CheckResult(
                    name="socket:llm",
                    category="runtime",
                    severity=Severity.OK,
                    detail=(
                        f"LLM endpoint not serving; {llm_unit} is {active} "
                        "(starts on demand)"
                    ),
                )
            )
    return results


def _directory_result(name: str, path: Path, *, required: bool) -> CheckResult:
    if not path.exists():
        severity = Severity.WARNING if required else Severity.OK
        note = "will be created on next use" if not required else "expected to exist"
        return CheckResult(
            name=name,
            category="runtime",
            severity=severity,
            detail=f"{path} does not exist ({note})",
        )
    try:
        info = path.stat()
    except OSError as exc:
        return CheckResult(
            name=name,
            category="runtime",
            severity=Severity.WARNING,
            detail=f"could not stat {path}: {exc}",
        )
    if info.st_uid != os.getuid():
        return CheckResult(
            name=name,
            category="runtime",
            severity=Severity.WARNING,
            detail=(
                f"{path} is owned by uid {info.st_uid}, not the current user "
                f"({os.getuid()}); it may be stale state from another session"
            ),
        )
    if info.st_mode & 0o077:
        return CheckResult(
            name=name,
            category="runtime",
            severity=Severity.WARNING,
            detail=f"{path} is group/other accessible (mode {info.st_mode & 0o777:o})",
        )
    return CheckResult(
        name=name,
        category="runtime",
        severity=Severity.OK,
        detail=f"{path} exists with private permissions",
    )


def check_runtime_directories() -> list[CheckResult]:
    results: list[CheckResult] = []
    if not RUNTIME.exists() or not os.access(RUNTIME, os.W_OK):
        results.append(
            CheckResult(
                name="runtime:xdg",
                category="runtime",
                severity=Severity.FATAL,
                detail=(
                    f"runtime directory {RUNTIME} is missing or not writable; "
                    "sockets and recorder state cannot be created"
                ),
            )
        )
    else:
        results.append(
            CheckResult(
                name="runtime:xdg",
                category="runtime",
                severity=Severity.OK,
                detail=f"runtime directory {RUNTIME} is writable",
            )
        )
    results.append(_directory_result("runtime:state", STATE_DIR, required=False))
    results.append(_directory_result("runtime:jobs", JOBS_DIR, required=False))
    return results


def check_cursor_jobs() -> list[CheckResult]:
    """Read job JSON read-only; never quarantine or mutate durable state."""

    results: list[CheckResult] = []
    if not JOBS_DIR.is_dir():
        return [
            CheckResult(
                name="jobs:store",
                category="jobs",
                severity=Severity.OK,
                detail=f"no durable job store yet at {JOBS_DIR}",
            )
        ]
    statuses: Counter[str] = Counter()
    unreadable: list[str] = []
    stuck: list[str] = []
    now = time.time()
    for path in sorted(JOBS_DIR.glob("*.json")):
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            unreadable.append(path.name)
            continue
        if not isinstance(raw, dict):
            unreadable.append(path.name)
            continue
        status = str(raw.get("status") or "unknown")
        statuses[status] += 1
        if status in _WORKER_JOB_STATUSES:
            timestamp = raw.get("updated_at") or raw.get("started_at")
            if isinstance(timestamp, int | float) and now - timestamp > (
                STUCK_JOB_SECONDS
            ):
                stuck.append(path.name)

    active = sum(statuses[status] for status in _ACTIVE_JOB_STATUSES)
    attention = sum(statuses[status] for status in _ATTENTION_JOB_STATUSES)
    summary = ", ".join(
        f"{status}={count}" for status, count in sorted(statuses.items())
    )
    results.append(
        CheckResult(
            name="jobs:store",
            category="jobs",
            severity=Severity.OK,
            detail=(
                f"{sum(statuses.values())} durable jobs ({active} active); "
                f"{summary or 'none'}"
            ),
        )
    )
    if attention:
        results.append(
            CheckResult(
                name="jobs:attention",
                category="jobs",
                severity=Severity.WARNING,
                detail=(
                    f"{attention} job(s) are awaiting user input or blocked; speak a "
                    "status or cancellation request to resolve them"
                ),
            )
        )
    if stuck:
        results.append(
            CheckResult(
                name="jobs:stuck",
                category="jobs",
                severity=Severity.WARNING,
                detail=(
                    f"{len(stuck)} worker-owned job(s) have not advanced in over "
                    f"{STUCK_JOB_SECONDS // 3600}h: {stuck}"
                ),
                suggestion="voice-harness services logs  # inspect the wake daemon",
            )
        )
    if unreadable:
        results.append(
            CheckResult(
                name="jobs:unreadable",
                category="jobs",
                severity=Severity.WARNING,
                detail=f"{len(unreadable)} job file(s) could not be read: {unreadable}",
            )
        )
    quarantine = JOBS_DIR / ".quarantine"
    if quarantine.is_dir():
        pending = sorted(quarantine.glob("*.metadata.json"))
        unresolved = [
            metadata.name
            for metadata in pending
            if not metadata.with_name(
                metadata.name.removesuffix(".metadata.json")
                + ".reservation-resolution.json"
            ).exists()
        ]
        if unresolved:
            results.append(
                CheckResult(
                    name="jobs:quarantine",
                    category="jobs",
                    severity=Severity.WARNING,
                    detail=(
                        f"{len(unresolved)} quarantined job record(s) may fence "
                        f"reservations: {unresolved}"
                    ),
                )
            )
    return results


def check_herdr() -> list[CheckResult]:
    if _which(HERDR_BIN) is None and not Path(HERDR_BIN).exists():
        return []
    try:
        running = HerdrClient().is_running()
    except HerdrError as exc:
        return [
            CheckResult(
                name="integration:herdr",
                category="integrations",
                severity=Severity.WARNING,
                detail=f"could not query the Herdr server: {exc}",
                suggestion="herdr status server",
            )
        ]
    if running:
        return [
            CheckResult(
                name="integration:herdr",
                category="integrations",
                severity=Severity.OK,
                detail="Herdr server is running",
            )
        ]
    return [
        CheckResult(
            name="integration:herdr",
            category="integrations",
            severity=Severity.WARNING,
            detail="Herdr server is not running (the wake daemon starts it on demand)",
            suggestion="herdr status server",
        )
    ]


def check_cursor_cli() -> list[CheckResult]:
    if _which("agent") is None:
        return []
    process = _run(["agent", "--version"], timeout=10)
    if process is None or process.returncode:
        return [
            CheckResult(
                name="integration:cursor",
                category="integrations",
                severity=Severity.WARNING,
                detail="the Cursor CLI (agent) did not respond to --version",
                suggestion="agent login",
            )
        ]
    return [
        CheckResult(
            name="integration:cursor",
            category="integrations",
            severity=Severity.OK,
            detail=f"Cursor CLI available: {process.stdout.strip().splitlines()[0]}"
            if process.stdout.strip()
            else "Cursor CLI available",
        )
    ]


def check_github_auth() -> list[CheckResult]:
    if _which("gh") is None:
        return []
    process = _run(["gh", "auth", "status"], timeout=10)
    if process is None or process.returncode:
        return [
            CheckResult(
                name="integration:github",
                category="integrations",
                severity=Severity.WARNING,
                detail=(
                    "GitHub CLI is not authenticated; focused issue/PR context and "
                    "fork creation will be unavailable"
                ),
                suggestion="gh auth login",
            )
        ]
    return [
        CheckResult(
            name="integration:github",
            category="integrations",
            severity=Severity.OK,
            detail="GitHub CLI is authenticated",
        )
    ]


def check_optional_integrations() -> list[CheckResult]:
    results: list[CheckResult] = []
    for name, status in capability_statuses():
        results.append(
            CheckResult(
                name=f"integration:{name}",
                category="integrations",
                severity=Severity.OK if status.available else Severity.FATAL,
                detail=status.detail,
                suggestion=status.suggestion,
            )
        )
    return results


def check_mcp_linear() -> list[CheckResult]:
    """Backward-compatible entry point; disabled Linear yields no diagnostic."""

    return [
        result
        for result in check_optional_integrations()
        if result.name == "integration:linear"
    ]


ALL_CHECKS: tuple[Check, ...] = (
    check_backend_configuration,
    check_required_executables,
    check_optional_executables,
    check_focus_automation,
    check_python_environments,
    check_model_file,
    check_model_caches,
    check_cuda,
    check_pipewire_devices,
    check_systemd_units,
    check_runtime_sockets,
    check_runtime_directories,
    check_cursor_jobs,
    check_herdr,
    check_cursor_cli,
    check_github_auth,
    check_optional_integrations,
)
