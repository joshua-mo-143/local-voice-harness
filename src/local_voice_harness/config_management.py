"""Configuration and integration management for the voice-harness CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from .config import PROJECT_ROOT, SERVICE_FILES, backend_config_path, xdg_config_home
from .credentials import CredentialError, get_venice_api_key
from .integrations.registry import capability_statuses
from .user_config import (
    IntegrationSettings,
    UserConfig,
    UserConfigurationError,
    default_user_config,
    load_user_config,
    render_user_config,
    user_config_path,
    write_user_config,
)

_INTEGRATION_NAMES = ("github", "zendesk", "linear")
_WAKE_SERVICE = "voice-harness-wake.service"
_LLM_SERVICE = "voice-harness-llm.service"
_TTS_SERVICE = "voice-harness-tts.service"
_DICTATION_SERVICE = "dictation.service"


@dataclass(frozen=True)
class ConfigPaths:
    config: Path
    backends: Path
    backend_env: Path
    home: Path


@dataclass(frozen=True)
class ConfigChangeResult:
    config: UserConfig
    changed_keys: tuple[str, ...]
    restart_services: tuple[str, ...]


Parser = Callable[[str], object]
Applier = Callable[[UserConfig, object], UserConfig]


@dataclass(frozen=True)
class ConfigField:
    parse: Parser
    apply: Applier
    describe: Callable[[UserConfig], object]
    services: tuple[str, ...] = ()


def _parse_bool(value: str) -> bool:
    text = value.strip().casefold()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise UserConfigurationError(f"expected a boolean, got {value!r}")


def _parse_str(value: str) -> str:
    return value.strip()


def _parse_int(value: str) -> int:
    try:
        return int(value.strip())
    except ValueError as exc:
        raise UserConfigurationError(f"expected an integer, got {value!r}") from exc


def _parse_float(value: str) -> float:
    try:
        return float(value.strip())
    except ValueError as exc:
        raise UserConfigurationError(f"expected a number, got {value!r}") from exc


def _parse_path(value: str) -> Path:
    return Path(value.strip()).expanduser()


def _parse_classes(value: str) -> tuple[str, ...]:
    parts = tuple(part.strip().casefold() for part in value.split(",") if part.strip())
    if not parts:
        raise UserConfigurationError("expected at least one window class")
    return parts


def _parse_replacements(value: str) -> tuple[tuple[str, str], ...]:
    entries = tuple(
        tuple(entry.split(":", 1))
        for entry in value.split(";")
        if entry.strip() and ":" in entry
    )
    return tuple((source.strip(), target.strip()) for source, target in entries)


def _replace_providers(config: UserConfig, **changes: object) -> UserConfig:
    from .config import BackendSettings

    providers = replace(config.providers, **changes)
    if not isinstance(providers, BackendSettings):
        raise UserConfigurationError("invalid provider update")
    return replace(config, providers=providers)


def _replace_section(
    config: UserConfig,
    section: str,
    **changes: object,
) -> UserConfig:
    current = getattr(config, section)
    updated = replace(current, **changes)
    return replace(config, **{section: updated})


def _field(
    *,
    parse: Parser,
    section: str,
    attribute: str,
    services: tuple[str, ...],
    transform: Callable[[object], object] | None = None,
) -> ConfigField:
    def apply(config: UserConfig, value: object) -> UserConfig:
        payload = transform(value) if transform is not None else value
        return _replace_section(config, section, **{attribute: payload})

    def describe(config: UserConfig) -> object:
        return getattr(getattr(config, section), attribute)

    return ConfigField(
        parse=parse,
        apply=apply,
        describe=describe,
        services=services,
    )


def _provider_field(section: str, attribute: str, *, parse: Parser) -> ConfigField:
    llm = section == "llm"

    def apply(config: UserConfig, value: object) -> UserConfig:
        key = f"llm_{attribute}" if llm else f"tts_{attribute}"
        return _replace_providers(config, **{key: value})

    def describe(config: UserConfig) -> object:
        providers = config.providers
        return getattr(providers, f"llm_{attribute}" if llm else f"tts_{attribute}")

    services = (_WAKE_SERVICE, _LLM_SERVICE) if llm else (_WAKE_SERVICE, _TTS_SERVICE)
    return ConfigField(
        parse=parse,
        apply=apply,
        describe=describe,
        services=services,
    )


_CONFIG_FIELDS: dict[str, ConfigField] = {
    "providers.llm.provider": _provider_field("llm", "provider", parse=_parse_str),
    "providers.llm.model": _provider_field("llm", "model", parse=_parse_str),
    "providers.llm.endpoint": _provider_field("llm", "endpoint", parse=_parse_str),
    "providers.llm.timeout": _provider_field("llm", "timeout", parse=_parse_float),
    "providers.tts.provider": _provider_field("tts", "provider", parse=_parse_str),
    "providers.tts.model": _provider_field("tts", "model", parse=_parse_str),
    "providers.tts.voice": _provider_field("tts", "voice", parse=_parse_str),
    "providers.tts.speed": _provider_field("tts", "speed", parse=_parse_float),
    "providers.tts.endpoint": _provider_field("tts", "endpoint", parse=_parse_str),
    "providers.tts.timeout": _provider_field("tts", "timeout", parse=_parse_float),
    "integrations.github": _field(
        parse=_parse_bool,
        section="integrations",
        attribute="github_enabled",
        services=(_WAKE_SERVICE,),
    ),
    "integrations.zendesk": _field(
        parse=_parse_bool,
        section="integrations",
        attribute="zendesk_enabled",
        services=(_WAKE_SERVICE,),
    ),
    "integrations.linear": _field(
        parse=_parse_bool,
        section="integrations",
        attribute="linear_enabled",
        services=(_WAKE_SERVICE,),
    ),
    "compute.cuda_device": _field(
        parse=_parse_str,
        section="compute",
        attribute="cuda_device",
        services=(_LLM_SERVICE,),
    ),
    "compute.dictation_backend": _field(
        parse=_parse_str,
        section="compute",
        attribute="dictation_backend",
        services=(_DICTATION_SERVICE,),
    ),
    "compute.dictation_model": _field(
        parse=_parse_str,
        section="compute",
        attribute="dictation_model",
        services=(_DICTATION_SERVICE,),
    ),
    "compute.dictation_quantization": _field(
        parse=_parse_str,
        section="compute",
        attribute="dictation_quantization",
        services=(_DICTATION_SERVICE,),
    ),
    "compute.dictation_compute": _field(
        parse=_parse_str,
        section="compute",
        attribute="dictation_compute",
        services=(_DICTATION_SERVICE,),
    ),
    "compute.dictation_language": _field(
        parse=_parse_str,
        section="compute",
        attribute="dictation_language",
        services=(_DICTATION_SERVICE,),
    ),
    "audio.source": _field(
        parse=_parse_str,
        section="audio",
        attribute="source",
        services=(_WAKE_SERVICE,),
    ),
    "audio.voice": _field(
        parse=_parse_str,
        section="audio",
        attribute="voice",
        services=(_WAKE_SERVICE,),
    ),
    "audio.wake_threshold": _field(
        parse=_parse_float,
        section="audio",
        attribute="wake_threshold",
        services=(_WAKE_SERVICE,),
    ),
    "audio.min_speech_rms": _field(
        parse=_parse_float,
        section="audio",
        attribute="min_speech_rms",
        services=(_WAKE_SERVICE,),
    ),
    "audio.barge_in_mode": _field(
        parse=_parse_str,
        section="audio",
        attribute="barge_in_mode",
        services=(_WAKE_SERVICE,),
    ),
    "audio.barge_in_speech_frames": _field(
        parse=_parse_int,
        section="audio",
        attribute="barge_in_speech_frames",
        services=(_WAKE_SERVICE,),
    ),
    "audio.playback_quiet_frames": _field(
        parse=_parse_int,
        section="audio",
        attribute="playback_quiet_frames",
        services=(_WAKE_SERVICE,),
    ),
    "audio.playback_quiet_timeout_seconds": _field(
        parse=_parse_float,
        section="audio",
        attribute="playback_quiet_timeout_seconds",
        services=(_WAKE_SERVICE,),
    ),
    "audio.playback_latency": _field(
        parse=_parse_str,
        section="audio",
        attribute="playback_latency",
        services=(_WAKE_SERVICE,),
    ),
    "dictation.source": _field(
        parse=_parse_str,
        section="dictation",
        attribute="source",
        services=(),
    ),
    "dictation.inject": _field(
        parse=_parse_str,
        section="dictation",
        attribute="inject",
        services=(),
    ),
    "dictation.prompt": _field(
        parse=_parse_str,
        section="dictation",
        attribute="prompt",
        services=(_DICTATION_SERVICE,),
    ),
    "dictation.replacements": _field(
        parse=_parse_replacements,
        section="dictation",
        attribute="replacements",
        services=(_DICTATION_SERVICE,),
    ),
    "dictation.vad_end_silence_ms": _field(
        parse=_parse_float,
        section="dictation",
        attribute="vad_end_silence_ms",
        services=(),
    ),
    "dictation.vad_max_seconds": _field(
        parse=_parse_float,
        section="dictation",
        attribute="vad_max_seconds",
        services=(),
    ),
    "dictation.vad_min_speech_rms": _field(
        parse=_parse_float,
        section="dictation",
        attribute="vad_min_speech_rms",
        services=(),
    ),
    "dictation.vad_start_speech_frames": _field(
        parse=_parse_int,
        section="dictation",
        attribute="vad_start_speech_frames",
        services=(),
    ),
    "platform.project_root": _field(
        parse=_parse_path,
        section="platform",
        attribute="project_root",
        services=(_WAKE_SERVICE,),
    ),
    "platform.github_root": _field(
        parse=_parse_path,
        section="platform",
        attribute="github_root",
        services=(_WAKE_SERVICE,),
    ),
    "platform.herdr_worktree_root": _field(
        parse=_parse_path,
        section="platform",
        attribute="herdr_worktree_root",
        services=(_WAKE_SERVICE,),
    ),
    "platform.gh_bin": _field(
        parse=_parse_path,
        section="platform",
        attribute="gh_bin",
        services=(_WAKE_SERVICE,),
    ),
    "platform.git_bin": _field(
        parse=_parse_path,
        section="platform",
        attribute="git_bin",
        services=(_WAKE_SERVICE,),
    ),
    "platform.herdr_bin": _field(
        parse=_parse_path,
        section="platform",
        attribute="herdr_bin",
        services=(_WAKE_SERVICE,),
    ),
    "platform.github_timeout_seconds": _field(
        parse=_parse_float,
        section="platform",
        attribute="github_timeout_seconds",
        services=(_WAKE_SERVICE,),
    ),
    "platform.herdr_timeout_seconds": _field(
        parse=_parse_float,
        section="platform",
        attribute="herdr_timeout_seconds",
        services=(_WAKE_SERVICE,),
    ),
    "platform.focused_app_context": _field(
        parse=_parse_bool,
        section="platform",
        attribute="focused_app_context_enabled",
        services=(_WAKE_SERVICE,),
    ),
    "platform.focused_app_deny_classes": _field(
        parse=_parse_classes,
        section="platform",
        attribute="focused_app_deny_classes",
        services=(_WAKE_SERVICE,),
    ),
    "platform.focused_app_max_chars": _field(
        parse=_parse_int,
        section="platform",
        attribute="focused_app_max_chars",
        services=(_WAKE_SERVICE,),
    ),
    "platform.cursor_followup": _field(
        parse=_parse_bool,
        section="platform",
        attribute="cursor_followup_enabled",
        services=(_WAKE_SERVICE,),
    ),
    "platform.cursor_followup_window_seconds": _field(
        parse=_parse_float,
        section="platform",
        attribute="cursor_followup_window_seconds",
        services=(_WAKE_SERVICE,),
    ),
    "platform.cursor_foreground_seconds": _field(
        parse=_parse_float,
        section="platform",
        attribute="cursor_foreground_seconds",
        services=(_WAKE_SERVICE,),
    ),
    "platform.cursor_agent_inactivity_seconds": _field(
        parse=_parse_float,
        section="platform",
        attribute="cursor_agent_inactivity_seconds",
        services=(_WAKE_SERVICE,),
    ),
    "platform.cursor_agent_max_runtime_seconds": _field(
        parse=_parse_float,
        section="platform",
        attribute="cursor_agent_max_runtime_seconds",
        services=(_WAKE_SERVICE,),
    ),
    "platform.agent_job_start_concurrency": _field(
        parse=_parse_int,
        section="platform",
        attribute="agent_job_start_concurrency",
        services=(_WAKE_SERVICE,),
    ),
}


def _resolved_environment(
    environment: Mapping[str, str] | None,
) -> Mapping[str, str]:
    return os.environ if environment is None else environment


def _file_environment(
    environment: Mapping[str, str] | None,
) -> Mapping[str, str]:
    return {} if environment is None else environment


def config_paths(
    environment: Mapping[str, str] | None = None,
    *,
    home: Path | None = None,
) -> ConfigPaths:
    resolved_home = home or Path.home()
    env = _resolved_environment(environment)
    return ConfigPaths(
        config=user_config_path(env, home=resolved_home),
        backends=backend_config_path(env, home=resolved_home),
        backend_env=xdg_config_home(env, home=resolved_home)
        / "dictation"
        / "backend.env",
        home=resolved_home,
    )


def load_managed_config(
    paths: ConfigPaths,
    environment: Mapping[str, str] | None = None,
) -> UserConfig:
    env = _file_environment(environment)
    return load_user_config(
        env,
        path=paths.config,
        backends_path=paths.backends,
        backend_env_path=paths.backend_env,
        home=paths.home,
    )


def validate_managed_config(
    config: UserConfig,
    paths: ConfigPaths,
    environment: Mapping[str, str] | None = None,
) -> UserConfig:
    """Validate ``config`` by round-tripping through a temporary write."""

    env = _file_environment(environment)
    temporary = paths.config.with_suffix(".validate.tmp")
    write_user_config(config, temporary)
    try:
        return load_user_config(
            env,
            path=temporary,
            backends_path=paths.backends,
            backend_env_path=Path(os.devnull),
            home=paths.home,
        )
    finally:
        temporary.unlink(missing_ok=True)


def restart_services_for_keys(
    config: UserConfig,
    keys: Iterable[str],
) -> tuple[str, ...]:
    services: list[str] = []
    for key in keys:
        normalized = key.strip().casefold()
        field = _CONFIG_FIELDS.get(normalized)
        if field is None:
            continue
        for service in field.services:
            services.append(service)
    return tuple(dict.fromkeys(services))


def active_services(services: Sequence[str]) -> tuple[str, ...]:
    active: list[str] = []
    for name in services:
        if name not in SERVICE_FILES:
            continue
        process = subprocess.run(
            ["systemctl", "--user", "is-active", name],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.stdout.strip() == "active":
            active.append(name)
    return tuple(active)


def format_restart_notice(services: Sequence[str]) -> str:
    if not services:
        return "No running services require a restart for this change."
    names = ", ".join(services)
    return (
        f"Restart to apply: {names}. "
        "Run `voice-harness services restart` when convenient; always-on services "
        "restart now and active on-demand services restart on their next use."
    )


def apply_config_values(
    config: UserConfig,
    assignments: Mapping[str, str],
) -> UserConfig:
    updated = config
    for key, raw_value in assignments.items():
        normalized = key.strip().casefold()
        field = _CONFIG_FIELDS.get(normalized)
        if field is None:
            known = ", ".join(sorted(_CONFIG_FIELDS))
            raise UserConfigurationError(
                f"unknown configuration key {key!r}; expected one of: {known}"
            )
        parsed = field.parse(raw_value)
        updated = field.apply(updated, parsed)
    return updated


def commit_config_change(
    assignments: Mapping[str, str],
    *,
    paths: ConfigPaths | None = None,
    environment: Mapping[str, str] | None = None,
) -> ConfigChangeResult:
    resolved_paths = paths or config_paths(environment)
    env = _file_environment(environment)
    current = load_managed_config(resolved_paths, env)
    updated = apply_config_values(current, assignments)
    validated = validate_managed_config(updated, resolved_paths, env)
    write_user_config(validated, resolved_paths.config)
    restart = active_services(restart_services_for_keys(validated, tuple(assignments)))
    return ConfigChangeResult(
        config=validated,
        changed_keys=tuple(assignments),
        restart_services=restart,
    )


def show_config(
    *,
    key: str | None = None,
    json_output: bool = False,
    paths: ConfigPaths | None = None,
    environment: Mapping[str, str] | None = None,
) -> str:
    resolved_paths = paths or config_paths(environment)
    config = load_managed_config(resolved_paths, _file_environment(environment))
    if key is not None:
        normalized = key.strip().casefold()
        field = _CONFIG_FIELDS.get(normalized)
        if field is None:
            raise UserConfigurationError(f"unknown configuration key {key!r}")
        value = field.describe(config)
        if json_output:
            return json.dumps(
                {normalized: _json_value(value)}, indent=2, sort_keys=True
            )
        return _format_scalar(value)
    if json_output:
        payload = {
            name: _json_value(field.describe(config))
            for name, field in sorted(_CONFIG_FIELDS.items())
        }
        return json.dumps(payload, indent=2, sort_keys=True)
    return render_user_config(config)


def reset_config(
    *,
    section: str | None = None,
    paths: ConfigPaths | None = None,
    environment: Mapping[str, str] | None = None,
) -> ConfigChangeResult:
    resolved_paths = paths or config_paths(environment)
    env = _file_environment(environment)
    defaults = default_user_config(home=resolved_paths.home)
    current = load_managed_config(resolved_paths, env)
    if section is None:
        validated = validate_managed_config(defaults, resolved_paths, env)
        changed = tuple(sorted(_CONFIG_FIELDS))
    else:
        normalized = section.strip().casefold()
        if normalized not in {
            "providers",
            "integrations",
            "compute",
            "audio",
            "dictation",
            "platform",
        }:
            raise UserConfigurationError(
                "section must be one of: providers, integrations, compute, audio, "
                "dictation, platform"
            )
        replacements: dict[str, str] = {}
        for name in _CONFIG_FIELDS:
            if not name.startswith(f"{normalized}."):
                continue
            field = _CONFIG_FIELDS[name]
            value = field.describe(defaults)
            replacements[name] = _format_scalar(value)
        updated = apply_config_values(current, replacements)
        validated = validate_managed_config(updated, resolved_paths, env)
        changed = tuple(replacements)
    write_user_config(validated, resolved_paths.config)
    restart = active_services(restart_services_for_keys(validated, changed))
    return ConfigChangeResult(
        config=validated,
        changed_keys=changed,
        restart_services=restart,
    )


def _json_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _format_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        if all(isinstance(item, tuple) and len(item) == 2 for item in value):
            return ";".join(f"{item[0]}:{item[1]}" for item in value)
        return ",".join(str(item) for item in value)
    return str(value)


def _choice(
    prompt: str,
    options: Sequence[str],
    *,
    default: str,
    input_fn: Callable[[str], str] = input,
) -> str:
    labels = "/".join(options)
    while True:
        answer = input_fn(f"{prompt} ({labels}) [{default}]: ").strip().casefold()
        if not answer:
            return default
        if answer in options:
            return answer
        print(f"Choose one of: {', '.join(options)}")


def _available_llm_providers() -> tuple[str, ...]:
    options: list[str] = []
    if shutil.which("llama-server") is not None:
        options.append("local")
    if _venice_available():
        options.append("venice")
    return tuple(options or ("local", "venice"))


def _available_tts_providers() -> tuple[str, ...]:
    return _available_llm_providers()


def _venice_available() -> bool:
    try:
        get_venice_api_key()
        return True
    except CredentialError:
        return shutil.which("secret-tool") is not None


def _available_dictation_backends() -> tuple[str, ...]:
    backends: list[str] = []
    if (PROJECT_ROOT / ".venv-dictation").is_dir():
        backends.append("parakeet")
    if (
        shutil.which("faster-whisper") is not None
        or (PROJECT_ROOT / ".venv-dictation-whisper").is_dir()
    ):
        backends.append("whisper")
    return tuple(backends or ("parakeet", "whisper"))


def run_setup(
    *,
    defaults_only: bool = False,
    paths: ConfigPaths | None = None,
    environment: Mapping[str, str] | None = None,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> ConfigChangeResult:
    resolved_paths = paths or config_paths(environment)
    llm_options = _available_llm_providers()
    tts_options = _available_tts_providers()
    dictation_options = _available_dictation_backends()

    if defaults_only:
        llm_provider = "local"
        tts_provider = "local"
        dictation_backend = dictation_options[0]
        github_enabled = True
        zendesk_enabled = False
        linear_enabled = False
        wake_threshold = "0.55"
    else:
        print_fn("Voice harness setup writes ~/.config/voice-harness/config.toml.")
        print_fn(
            "Existing backends.toml values still override provider fields until removed."
        )
        llm_provider = _choice(
            "LLM provider",
            llm_options,
            default="local",
            input_fn=input_fn,
        )
        tts_provider = _choice(
            "TTS provider",
            tts_options,
            default="local",
            input_fn=input_fn,
        )
        dictation_backend = _choice(
            "Dictation backend",
            dictation_options,
            default=dictation_options[0],
            input_fn=input_fn,
        )
        github_enabled = (
            _choice(
                "Enable GitHub integration",
                ("true", "false"),
                default="true",
                input_fn=input_fn,
            )
            == "true"
        )
        zendesk_enabled = (
            _choice(
                "Enable Zendesk integration",
                ("true", "false"),
                default="false",
                input_fn=input_fn,
            )
            == "true"
        )
        linear_enabled = (
            _choice(
                "Enable Linear integration",
                ("true", "false"),
                default="false",
                input_fn=input_fn,
            )
            == "true"
        )
        wake_threshold = input_fn("Wake threshold (0.0-1.0) [0.55]: ").strip() or "0.55"

    assignments = {
        "providers.llm.provider": llm_provider,
        "providers.tts.provider": tts_provider,
        "compute.dictation_backend": dictation_backend,
        "integrations.github": "true" if github_enabled else "false",
        "integrations.zendesk": "true" if zendesk_enabled else "false",
        "integrations.linear": "true" if linear_enabled else "false",
        "audio.wake_threshold": wake_threshold,
    }
    result = commit_config_change(
        assignments, paths=resolved_paths, environment=environment
    )
    print_fn("Wrote unified configuration.")
    print_fn(format_restart_notice(result.restart_services))
    if llm_provider == "venice" or tts_provider == "venice":
        print_fn(
            "Store a Venice API key with `voice-harness credentials set` if needed."
        )
    if linear_enabled:
        print_fn(
            "Linear requires Cursor MCP: "
            "`agent mcp login linear && agent mcp enable linear`."
        )
    return result


def list_integrations(
    *,
    json_output: bool = False,
    paths: ConfigPaths | None = None,
    environment: Mapping[str, str] | None = None,
) -> str:
    config = load_managed_config(
        paths or config_paths(environment),
        _file_environment(environment),
    )
    rows = [
        {
            "name": name,
            "enabled": getattr(config.integrations, f"{name}_enabled"),
        }
        for name in _INTEGRATION_NAMES
    ]
    if json_output:
        return json.dumps(rows, indent=2, sort_keys=True)
    lines = [
        f"{row['name']}: {'enabled' if row['enabled'] else 'disabled'}" for row in rows
    ]
    return "\n".join(lines)


def set_integration_enabled(
    name: str,
    *,
    enabled: bool,
    paths: ConfigPaths | None = None,
    environment: Mapping[str, str] | None = None,
) -> ConfigChangeResult:
    normalized = name.strip().casefold()
    if normalized not in _INTEGRATION_NAMES:
        options = ", ".join(_INTEGRATION_NAMES)
        raise UserConfigurationError(
            f"unknown integration {name!r}; expected: {options}"
        )
    return commit_config_change(
        {f"integrations.{normalized}": "true" if enabled else "false"},
        paths=paths,
        environment=environment,
    )


@dataclass(frozen=True)
class IntegrationDiagnostic:
    name: str
    ok: bool
    detail: str
    suggestion: str | None = None


def integration_diagnostics(
    integrations: IntegrationSettings | None = None,
) -> tuple[IntegrationDiagnostic, ...]:
    settings = integrations
    if settings is None:
        settings = load_managed_config(config_paths()).integrations
    results: list[IntegrationDiagnostic] = []
    for name, status in capability_statuses(settings):
        results.append(
            IntegrationDiagnostic(
                name=name,
                ok=status.available,
                detail=status.detail,
                suggestion=status.suggestion,
            )
        )
    if settings.github_enabled and shutil.which("gh") is not None:
        process = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode:
            results.append(
                IntegrationDiagnostic(
                    name="github",
                    ok=False,
                    detail=(
                        "GitHub is enabled but the GitHub CLI is not authenticated; "
                        "focused issue/PR context and fork creation will be unavailable."
                    ),
                    suggestion="gh auth login",
                )
            )
        else:
            results.append(
                IntegrationDiagnostic(
                    name="github",
                    ok=True,
                    detail="GitHub CLI is authenticated.",
                )
            )
    elif settings.github_enabled:
        results.append(
            IntegrationDiagnostic(
                name="github",
                ok=False,
                detail="GitHub is enabled but the GitHub CLI is not installed.",
                suggestion="paru -S --needed github-cli",
            )
        )
    if settings.zendesk_enabled:
        results.append(
            IntegrationDiagnostic(
                name="zendesk",
                ok=True,
                detail="Zendesk browser capture is enabled.",
            )
        )
    return tuple(results)


def format_integration_diagnostics(
    diagnostics: Sequence[IntegrationDiagnostic],
    *,
    json_output: bool = False,
) -> str:
    if json_output:
        payload = [
            {
                "name": item.name,
                "ok": item.ok,
                "detail": item.detail,
                "suggestion": item.suggestion,
            }
            for item in diagnostics
        ]
        return json.dumps(payload, indent=2, sort_keys=True)
    if not diagnostics:
        return "No enabled integrations to inspect."
    lines: list[str] = []
    for item in diagnostics:
        state = "ok" if item.ok else "needs attention"
        lines.append(f"{item.name}: {state}")
        lines.append(f"  {item.detail}")
        if item.suggestion:
            lines.append(f"  suggestion: {item.suggestion}")
    return "\n".join(lines)


def run_integration_doctor(
    *,
    json_output: bool = False,
    paths: ConfigPaths | None = None,
    environment: Mapping[str, str] | None = None,
) -> tuple[int, str]:
    config = load_managed_config(
        paths or config_paths(environment),
        _file_environment(environment),
    )
    diagnostics = integration_diagnostics(config.integrations)
    output = format_integration_diagnostics(diagnostics, json_output=json_output)
    return (0 if all(item.ok for item in diagnostics) else 1, output)
