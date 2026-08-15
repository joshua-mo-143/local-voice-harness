"""Durable, narrowly targeted activation of confirmed configuration changes."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from .config import SERVICE_FILES, durable_state_dir
from .config_management import ConfigChangeResult, commit_config_change
from .errors import HarnessError
from .platform_services import user_services
from .responses import AssistantResponse
from .self_management import PendingConfigChange
from .tts.client import VoiceValidationResult, validate_voice
from .user_config import UserConfig, load_user_config, render_user_config

_WAKE_SERVICE = "voice-harness-wake.service"
_FINAL_STATES = frozenset({"succeeded", "partial", "failed", "declined"})


class ActivationStatus(StrEnum):
    OFFERED = "offered"
    ACCEPTED = "accepted"
    READY = "ready"
    VALIDATING = "validating"
    RESTARTING = "restarting"
    ROLLING_BACK = "rolling_back"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    DECLINED = "declined"


class ActivationDecision(StrEnum):
    ACTIVATE = "activate"
    DECLINE = "decline"
    NONE = "none"


class ActivationDeliveryKind(StrEnum):
    OFFER = "offer"
    PRE_RESTART = "pre_restart"
    RESULT = "result"


class ServiceOutcomeStatus(StrEnum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ServiceSnapshot:
    installed: bool
    active_state: str
    sub_state: str
    invocation_id: str
    process_start: str
    config_digest: str
    voice: str = ""

    @property
    def identity(self) -> tuple[str, str]:
        return self.invocation_id, self.process_start


@dataclass(frozen=True, slots=True)
class ServiceOutcome:
    service: str
    status: ServiceOutcomeStatus = ServiceOutcomeStatus.PENDING
    attempted: bool = False
    before: ServiceSnapshot | None = None
    after: ServiceSnapshot | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ActivationRecord:
    id: str
    setting_key: str
    old_value: str | bool
    new_value: str | bool
    expected_config_digest: str
    targets: tuple[str, ...]
    status: ActivationStatus
    outcomes: tuple[ServiceOutcome, ...]
    offer_delivered: bool = False
    pre_restart_delivered: bool = False
    acknowledged: bool = False
    voice_validation_attempted: bool = False
    voice_validated: bool = False
    rollback_config_restored: bool = False
    rollback_config_digest: str = ""
    rollback_outcomes: tuple[ServiceOutcome, ...] = ()
    detail: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass(frozen=True, slots=True)
class ActivationDelivery:
    record: ActivationRecord
    kind: ActivationDeliveryKind


class ActivationStateError(HarnessError):
    """The durable activation journal cannot accept the requested transition."""


class ServiceController(Protocol):
    def snapshot(self, service: str) -> ServiceSnapshot: ...

    def restart(self, service: str) -> subprocess.CompletedProcess[str]: ...


def config_digest(config: UserConfig) -> str:
    """Fingerprint one immutable configuration snapshot."""

    return hashlib.sha256(render_user_config(config).encode("utf-8")).hexdigest()


def service_snapshot_path(
    service: str,
    *,
    state_dir: Path | None = None,
) -> Path:
    if service not in SERVICE_FILES:
        raise ActivationStateError(f"unsupported user service {service!r}")
    root = state_dir or durable_state_dir()
    return root / "service-snapshots" / f"{service}.json"


def publish_service_snapshot(
    service: str,
    config: UserConfig,
    *,
    pid: int,
    process_start: str,
    state_dir: Path | None = None,
) -> None:
    """Publish the immutable snapshot loaded by one running managed service."""

    path = service_snapshot_path(service, state_dir=state_dir)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "service": service,
                "pid": pid,
                "process_start": process_start,
                "config_digest": config_digest(config),
                "voice": config.audio.voice,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _snapshot_from_dict(value: object) -> ServiceSnapshot | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ActivationStateError("activation snapshot must be an object")
    try:
        return ServiceSnapshot(
            installed=bool(value["installed"]),
            active_state=str(value["active_state"]),
            sub_state=str(value["sub_state"]),
            invocation_id=str(value["invocation_id"]),
            process_start=str(value["process_start"]),
            config_digest=str(value["config_digest"]),
            voice=str(value.get("voice") or ""),
        )
    except KeyError as exc:
        raise ActivationStateError("activation snapshot is incomplete") from exc


def _outcome_from_dict(value: object) -> ServiceOutcome:
    if not isinstance(value, dict):
        raise ActivationStateError("activation outcome must be an object")
    try:
        return ServiceOutcome(
            service=str(value["service"]),
            status=ServiceOutcomeStatus(str(value["status"])),
            attempted=bool(value["attempted"]),
            before=_snapshot_from_dict(value.get("before")),
            after=_snapshot_from_dict(value.get("after")),
            detail=str(value.get("detail") or ""),
        )
    except (KeyError, ValueError) as exc:
        raise ActivationStateError("activation outcome is invalid") from exc


def _record_from_dict(value: object) -> ActivationRecord:
    if not isinstance(value, dict):
        raise ActivationStateError("activation journal must contain an object")
    try:
        old_value = value["old_value"]
        new_value = value["new_value"]
        if not isinstance(old_value, (str, bool)) or not isinstance(
            new_value, (str, bool)
        ):
            raise ActivationStateError("activation values must be strings or booleans")
        targets = value["targets"]
        outcomes = value["outcomes"]
        rollback_outcomes = value.get("rollback_outcomes", [])
        if (
            not isinstance(targets, list)
            or not isinstance(outcomes, list)
            or not isinstance(rollback_outcomes, list)
        ):
            raise ActivationStateError("activation targets and outcomes must be arrays")
        record_id = str(value["id"])
        target_values = tuple(str(item) for item in targets)
        outcome_values = tuple(_outcome_from_dict(item) for item in outcomes)
        rollback_config_digest = str(value.get("rollback_config_digest") or "")
        if (
            len(record_id) != 32
            or any(character not in "0123456789abcdef" for character in record_id)
            or (
                rollback_config_digest
                and (
                    len(rollback_config_digest) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in rollback_config_digest
                    )
                )
            )
            or not target_values
            or any(service not in SERVICE_FILES for service in target_values)
            or tuple(outcome.service for outcome in outcome_values) != target_values
            or (
                rollback_outcomes
                and tuple(
                    str(item.get("service", ""))
                    for item in rollback_outcomes
                    if isinstance(item, dict)
                )
                != target_values
            )
        ):
            raise ActivationStateError("activation identity or targets are invalid")
        return ActivationRecord(
            id=record_id,
            setting_key=str(value["setting_key"]),
            old_value=old_value,
            new_value=new_value,
            expected_config_digest=str(value["expected_config_digest"]),
            targets=target_values,
            status=ActivationStatus(str(value["status"])),
            outcomes=outcome_values,
            offer_delivered=bool(value.get("offer_delivered")),
            pre_restart_delivered=bool(value.get("pre_restart_delivered")),
            acknowledged=bool(value.get("acknowledged")),
            voice_validation_attempted=bool(value.get("voice_validation_attempted")),
            voice_validated=bool(value.get("voice_validated")),
            rollback_config_restored=bool(value.get("rollback_config_restored")),
            rollback_config_digest=rollback_config_digest,
            rollback_outcomes=tuple(
                _outcome_from_dict(item) for item in rollback_outcomes
            ),
            detail=str(value.get("detail") or ""),
            created_at=float(value["created_at"]),
            updated_at=float(value["updated_at"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ActivationStateError("activation journal is invalid") from exc


class ActivationStore:
    """One atomic durable activation journal."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or durable_state_dir() / "config-activation.json"
        self.lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")
        self.execution_lock_path = self.path.with_suffix(
            f"{self.path.suffix}.execute.lock"
        )

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock:
            os.chmod(self.lock_path, 0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def execution_lock(self) -> Iterator[None]:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self.execution_lock_path.open("a+b") as lock:
            os.chmod(self.execution_lock_path, 0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _read(self) -> ActivationRecord | None:
        try:
            raw = self.path.read_text()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ActivationStateError(f"could not read {self.path}: {exc}") from exc
        try:
            return _record_from_dict(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise ActivationStateError(
                f"activation journal {self.path} is malformed"
            ) from exc

    def _write(self, record: ActivationRecord) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        payload = (
            json.dumps(asdict(record), sort_keys=True, separators=(",", ":")) + "\n"
        )
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        with os.fdopen(descriptor, "w") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, self.path)
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def current(self) -> ActivationRecord | None:
        with self._locked():
            return self._read()

    def _update(
        self,
        record_id: str,
        transition: Callable[[ActivationRecord], ActivationRecord],
    ) -> ActivationRecord:
        with self._locked():
            current = self._read()
            if current is None or current.id != record_id:
                raise ActivationStateError("activation request no longer exists")
            updated = replace(transition(current), updated_at=time.time())
            self._write(updated)
            return updated

    def create_offer(
        self,
        pending: PendingConfigChange,
        result: ConfigChangeResult,
        *,
        expected_config: UserConfig | None = None,
    ) -> ActivationRecord | None:
        targets = tuple(dict.fromkeys(result.restart_services))
        if not targets:
            return None
        if any(
            service not in SERVICE_FILES
            or service not in pending.affected_services
            or service.casefold() == "herdr"
            for service in targets
        ):
            raise ActivationStateError(
                "activation target was not reported by the typed configuration change"
            )
        candidate_config = expected_config or result.config
        now = time.time()
        record = ActivationRecord(
            id=uuid.uuid4().hex,
            setting_key=pending.setting.value,
            old_value=pending.old_value,
            new_value=pending.new_value,
            expected_config_digest=config_digest(candidate_config),
            targets=targets,
            status=ActivationStatus.OFFERED,
            outcomes=tuple(ServiceOutcome(service) for service in targets),
            created_at=now,
            updated_at=now,
        )
        with self._locked():
            current = self._read()
            if current is not None and not current.acknowledged:
                raise ActivationStateError(
                    "a previous activation result is still awaiting delivery"
                )
            self._write(record)
        return record

    def mark_offer_delivered(self, record_id: str) -> ActivationRecord:
        def transition(record: ActivationRecord) -> ActivationRecord:
            if record.status != ActivationStatus.OFFERED:
                raise ActivationStateError("activation offer is no longer pending")
            return replace(record, offer_delivered=True)

        return self._update(record_id, transition)

    def accept(self, record_id: str) -> ActivationRecord:
        def transition(record: ActivationRecord) -> ActivationRecord:
            if record.status != ActivationStatus.OFFERED or not record.offer_delivered:
                raise ActivationStateError("activation offer is not ready to accept")
            return replace(record, status=ActivationStatus.ACCEPTED)

        return self._update(record_id, transition)

    def decline(self, record_id: str) -> ActivationRecord:
        def transition(record: ActivationRecord) -> ActivationRecord:
            if record.status != ActivationStatus.OFFERED or not record.offer_delivered:
                raise ActivationStateError("activation offer is not ready to decline")
            return replace(
                record,
                status=ActivationStatus.DECLINED,
                detail="The saved change was not activated; no service was restarted.",
            )

        return self._update(record_id, transition)

    def mark_pre_restart_delivered(self, record_id: str) -> ActivationRecord:
        def transition(record: ActivationRecord) -> ActivationRecord:
            if record.status != ActivationStatus.ACCEPTED:
                raise ActivationStateError(
                    "activation request is not awaiting delivery"
                )
            return replace(
                record,
                status=ActivationStatus.READY,
                pre_restart_delivered=True,
            )

        return self._update(record_id, transition)

    def begin_voice_validation(self, record_id: str) -> ActivationRecord:
        def transition(record: ActivationRecord) -> ActivationRecord:
            if (
                record.status != ActivationStatus.READY
                or record.setting_key != "audio.voice"
                or record.voice_validation_attempted
            ):
                raise ActivationStateError("voice activation is not ready to validate")
            return replace(
                record,
                status=ActivationStatus.VALIDATING,
                voice_validation_attempted=True,
            )

        return self._update(record_id, transition)

    def mark_voice_validated(self, record_id: str) -> ActivationRecord:
        def transition(record: ActivationRecord) -> ActivationRecord:
            if record.status != ActivationStatus.VALIDATING:
                raise ActivationStateError("voice activation is not validating")
            return replace(
                record,
                status=ActivationStatus.READY,
                voice_validated=True,
            )

        return self._update(record_id, transition)

    def update_voice_activation_digest(
        self,
        record_id: str,
        expected_config_digest: str,
    ) -> ActivationRecord:
        def transition(record: ActivationRecord) -> ActivationRecord:
            if (
                record.status != ActivationStatus.READY
                or record.setting_key != "audio.voice"
                or not record.voice_validated
            ):
                raise ActivationStateError(
                    "voice activation snapshot is not ready to update"
                )
            if len(expected_config_digest) != 64 or any(
                character not in "0123456789abcdef"
                for character in expected_config_digest
            ):
                raise ActivationStateError("voice activation digest is invalid")
            return replace(record, expected_config_digest=expected_config_digest)

        return self._update(record_id, transition)

    def begin_rollback(self, record_id: str, detail: str) -> ActivationRecord:
        def transition(record: ActivationRecord) -> ActivationRecord:
            if record.setting_key != "audio.voice" or record.status not in {
                ActivationStatus.READY,
                ActivationStatus.VALIDATING,
                ActivationStatus.RESTARTING,
            }:
                raise ActivationStateError("voice activation cannot be rolled back")
            return replace(
                record,
                status=ActivationStatus.ROLLING_BACK,
                rollback_outcomes=tuple(
                    ServiceOutcome(service) for service in record.targets
                ),
                detail=detail,
            )

        return self._update(record_id, transition)

    def mark_rollback_config_restored(
        self,
        record_id: str,
        rollback_config_digest: str,
    ) -> ActivationRecord:
        def transition(record: ActivationRecord) -> ActivationRecord:
            if record.status != ActivationStatus.ROLLING_BACK:
                raise ActivationStateError("voice activation is not rolling back")
            if len(rollback_config_digest) != 64 or any(
                character not in "0123456789abcdef"
                for character in rollback_config_digest
            ):
                raise ActivationStateError("voice rollback digest is invalid")
            return replace(
                record,
                rollback_config_restored=True,
                rollback_config_digest=rollback_config_digest,
            )

        return self._update(record_id, transition)

    def update_rollback_outcome(
        self,
        record_id: str,
        outcome: ServiceOutcome,
    ) -> ActivationRecord:
        def transition(record: ActivationRecord) -> ActivationRecord:
            if record.status != ActivationStatus.ROLLING_BACK:
                raise ActivationStateError("voice activation is not rolling back")
            if outcome.service not in record.targets:
                raise ActivationStateError("service is not a rollback target")
            return replace(
                record,
                rollback_outcomes=tuple(
                    outcome if item.service == outcome.service else item
                    for item in record.rollback_outcomes
                ),
            )

        return self._update(record_id, transition)

    def begin_restart(
        self,
        record_id: str,
        outcomes: tuple[ServiceOutcome, ...],
    ) -> ActivationRecord:
        def transition(record: ActivationRecord) -> ActivationRecord:
            if record.status != ActivationStatus.READY:
                raise ActivationStateError("activation request is not ready")
            return replace(
                record,
                status=ActivationStatus.RESTARTING,
                outcomes=outcomes,
            )

        return self._update(record_id, transition)

    def update_outcome(
        self,
        record_id: str,
        outcome: ServiceOutcome,
    ) -> ActivationRecord:
        def transition(record: ActivationRecord) -> ActivationRecord:
            if record.status != ActivationStatus.RESTARTING:
                raise ActivationStateError("activation request is not restarting")
            if outcome.service not in record.targets:
                raise ActivationStateError("service is not an activation target")
            return replace(
                record,
                outcomes=tuple(
                    outcome if item.service == outcome.service else item
                    for item in record.outcomes
                ),
            )

        return self._update(record_id, transition)

    def finish(
        self,
        record_id: str,
        *,
        status: ActivationStatus,
        detail: str,
    ) -> ActivationRecord:
        if status.value not in _FINAL_STATES - {"declined"}:
            raise ActivationStateError("activation result is not terminal")
        return self._update(
            record_id,
            lambda record: replace(record, status=status, detail=detail),
        )

    def fail_worker(self, record_id: str, detail: str) -> ActivationRecord:
        def transition(record: ActivationRecord) -> ActivationRecord:
            if record.status not in {
                ActivationStatus.READY,
                ActivationStatus.VALIDATING,
                ActivationStatus.RESTARTING,
                ActivationStatus.ROLLING_BACK,
            }:
                raise ActivationStateError("activation request is not executable")
            return replace(record, status=ActivationStatus.FAILED, detail=detail)

        return self._update(record_id, transition)

    def acknowledge(self, record_id: str) -> ActivationRecord:
        def transition(record: ActivationRecord) -> ActivationRecord:
            if record.status.value not in _FINAL_STATES:
                raise ActivationStateError("activation result is not final")
            return replace(record, acknowledged=True)

        return self._update(record_id, transition)

    def next_delivery(self) -> ActivationDelivery | None:
        record = self.current()
        if record is None:
            return None
        if record.status == ActivationStatus.OFFERED and not record.offer_delivered:
            return ActivationDelivery(record, ActivationDeliveryKind.OFFER)
        if record.status == ActivationStatus.ACCEPTED:
            return ActivationDelivery(record, ActivationDeliveryKind.PRE_RESTART)
        if record.status.value in _FINAL_STATES and not record.acknowledged:
            return ActivationDelivery(record, ActivationDeliveryKind.RESULT)
        return None


def resolve_activation_decision(utterance: str) -> ActivationDecision:
    normalized = " ".join(utterance.casefold().strip().split())
    if normalized in {"activate now", "restart it now"}:
        return ActivationDecision.ACTIVATE
    if normalized in {"not now", "cancel activation", "do not restart"}:
        return ActivationDecision.DECLINE
    return ActivationDecision.NONE


def _display_value(value: str | bool) -> str:
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    return value or "the configured default"


def _bounded_detail(detail: str) -> str:
    normalized = " ".join(detail.split())
    if len(normalized) <= 240:
        return normalized
    return normalized[:239].rstrip() + "…"


def render_activation_delivery(delivery: ActivationDelivery) -> AssistantResponse:
    record = delivery.record
    targets = ", ".join(record.targets)
    if delivery.kind == ActivationDeliveryKind.OFFER:
        wake_notice = (
            " Because this affects the active wake listener, activation requires a "
            "separate explicit confirmation."
            if _WAKE_SERVICE in record.targets
            else ""
        )
        text = (
            f"Saved {record.setting_key} as {_display_value(record.new_value)}. "
            "The running configuration snapshot is unchanged."
            f"{wake_notice} Say activate now to restart {targets}, or say not now "
            "to leave the saved change inactive."
        )
    elif delivery.kind == ActivationDeliveryKind.PRE_RESTART:
        text = (
            f"Activation accepted. I will restart only {targets} after this message "
            "is delivered. The completion result will be delivered after recovery."
        )
    elif record.status == ActivationStatus.SUCCEEDED:
        if record.setting_key == "audio.voice":
            text = (
                f"Voice activation succeeded. The selected TTS provider accepted "
                f"{_display_value(record.new_value)}, and {targets} is active with "
                "the expected voice snapshot."
            )
        else:
            text = (
                f"Activation succeeded. {targets} is active with the expected "
                "configuration snapshot."
            )
    elif record.status == ActivationStatus.DECLINED:
        text = (
            "Activation was cancelled. The configuration remains saved, but no "
            "service was restarted."
        )
    else:
        text = (
            f"Activation {record.status.value}. {_bounded_detail(record.detail)} "
            "I will not repeat any ambiguous restart automatically."
        )
    return AssistantResponse(spoken_text=text, display_text=text)


class SystemdUserServiceController:
    """Observe and restart allow-listed installed user services only."""

    def __init__(self, *, state_dir: Path | None = None) -> None:
        self.state_dir = state_dir

    def snapshot(self, service: str) -> ServiceSnapshot:
        if service not in SERVICE_FILES:
            raise ActivationStateError(f"unsupported user service {service!r}")
        properties = user_services().show(
            service,
            (
                "LoadState",
                "ActiveState",
                "SubState",
                "InvocationID",
                "MainPID",
            ),
        )
        process_start = ""
        digest = ""
        voice = ""
        marker = service_snapshot_path(service, state_dir=self.state_dir)
        try:
            value = json.loads(marker.read_text())
        except (OSError, json.JSONDecodeError):
            value = None
        if (
            isinstance(value, dict)
            and value.get("service") == service
            and str(value.get("pid") or "") == properties.get("MainPID")
        ):
            process_start = str(value.get("process_start") or "")
            digest = str(value.get("config_digest") or "")
            voice = str(value.get("voice") or "")
        return ServiceSnapshot(
            installed=properties.get("LoadState") == "loaded",
            active_state=properties.get("ActiveState", "unknown"),
            sub_state=properties.get("SubState", "unknown"),
            invocation_id=properties.get("InvocationID", ""),
            process_start=process_start,
            config_digest=digest,
            voice=voice,
        )

    def restart(self, service: str) -> subprocess.CompletedProcess[str]:
        if service not in SERVICE_FILES:
            raise ActivationStateError(f"unsupported user service {service!r}")
        return user_services().try_restart(service)


def _matches_expected_snapshot(
    snapshot: ServiceSnapshot,
    expected_digest: str,
    expected_voice: str | None,
) -> bool:
    if snapshot.config_digest == expected_digest:
        return True
    return (
        expected_voice is not None
        and snapshot.voice == expected_voice
        and len(snapshot.config_digest) == 64
        and all(character in "0123456789abcdef" for character in snapshot.config_digest)
    )


def _observe_restart(
    controller: ServiceController,
    outcome: ServiceOutcome,
    expected_digest: str,
    *,
    expected_voice: str | None = None,
    timeout: float,
) -> ServiceOutcome:
    deadline = time.monotonic() + timeout
    after = controller.snapshot(outcome.service)
    while time.monotonic() < deadline:
        identity_changed = (
            outcome.before is not None
            and after.identity != outcome.before.identity
            and any(after.identity)
        )
        if (
            after.active_state == "active"
            and identity_changed
            and _matches_expected_snapshot(after, expected_digest, expected_voice)
        ):
            break
        time.sleep(0.2)
        after = controller.snapshot(outcome.service)
    identity_changed = (
        outcome.before is not None
        and after.identity != outcome.before.identity
        and any(after.identity)
    )
    if (
        after.installed
        and after.active_state == "active"
        and identity_changed
        and _matches_expected_snapshot(after, expected_digest, expected_voice)
    ):
        return replace(
            outcome,
            status=ServiceOutcomeStatus.SUCCEEDED,
            after=after,
            detail=(
                "active with the expected immutable snapshot"
                if after.config_digest == expected_digest
                else "active with the expected voice snapshot"
            ),
        )
    reasons = []
    if not after.installed:
        reasons.append("service is no longer installed")
    if after.active_state != "active":
        reasons.append(f"service state is {after.active_state}")
    if not identity_changed:
        reasons.append("restart invocation was not observed")
    if not _matches_expected_snapshot(after, expected_digest, expected_voice):
        reasons.append("expected configuration or voice snapshot was not observed")
    return replace(
        outcome,
        status=ServiceOutcomeStatus.FAILED,
        after=after,
        detail=", ".join(reasons) or "restart outcome is ambiguous",
    )


def _restore_previous_voice(record: ActivationRecord) -> UserConfig:
    if not isinstance(record.old_value, str) or not isinstance(record.new_value, str):
        raise ActivationStateError("voice rollback values are invalid")
    commit_config_change(
        {"audio.voice": record.old_value},
        expected_values={"audio.voice": record.new_value},
    )
    return load_user_config()


def _execute_voice_rollback(
    record: ActivationRecord,
    *,
    store: ActivationStore,
    controller: ServiceController,
    load_config: Callable[[], UserConfig],
    restore_voice: Callable[[ActivationRecord], UserConfig],
    observation_timeout: float,
) -> ActivationRecord:
    if not isinstance(record.old_value, str) or not isinstance(record.new_value, str):
        return store.finish(
            record.id,
            status=ActivationStatus.FAILED,
            detail=f"{record.detail} The voice rollback values were invalid.",
        )
    current_config = load_config()
    if not record.rollback_config_restored:
        if current_config.audio.voice == record.old_value:
            restored = current_config
        elif current_config.audio.voice == record.new_value:
            try:
                restored = restore_voice(record)
            except Exception as exc:  # noqa: BLE001 - bounded rollback failure
                return store.finish(
                    record.id,
                    status=ActivationStatus.FAILED,
                    detail=(
                        f"{record.detail} Restoring the previous voice failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            if restored.audio.voice != record.old_value:
                return store.finish(
                    record.id,
                    status=ActivationStatus.FAILED,
                    detail=(
                        f"{record.detail} Restoring the previous voice did not "
                        "restore the expected voice value."
                    ),
                )
        else:
            return store.finish(
                record.id,
                status=ActivationStatus.FAILED,
                detail=(
                    f"{record.detail} The stored voice changed during recovery, "
                    "so the replacement voice was not overwritten."
                ),
            )
        record = store.mark_rollback_config_restored(
            record.id,
            config_digest(restored),
        )
    elif not record.rollback_config_digest:
        if current_config.audio.voice != record.old_value:
            return store.finish(
                record.id,
                status=ActivationStatus.FAILED,
                detail=(
                    f"{record.detail} The restored voice could not be proven after "
                    "journal recovery."
                ),
            )
        record = store.mark_rollback_config_restored(
            record.id,
            config_digest(current_config),
        )
    rollback_digest = record.rollback_config_digest

    for outcome in record.rollback_outcomes:
        if outcome.status != ServiceOutcomeStatus.PENDING:
            continue
        before = outcome.before or controller.snapshot(outcome.service)
        if (
            before.installed
            and before.active_state == "active"
            and before.config_digest == rollback_digest
        ):
            record = store.update_rollback_outcome(
                record.id,
                replace(
                    outcome,
                    status=ServiceOutcomeStatus.SUCCEEDED,
                    before=before,
                    after=before,
                    detail="already active with the previous immutable snapshot",
                ),
            )
            continue
        if not before.installed or before.active_state != "active":
            detail = (
                "service is not installed"
                if not before.installed
                else f"service is {before.active_state}; it was not started"
            )
            record = store.update_rollback_outcome(
                record.id,
                replace(
                    outcome,
                    status=ServiceOutcomeStatus.FAILED,
                    before=before,
                    detail=detail,
                ),
            )
            continue
        pending = replace(outcome, before=before)
        if pending.attempted:
            observed = _observe_restart(
                controller,
                pending,
                rollback_digest,
                timeout=observation_timeout,
            )
            record = store.update_rollback_outcome(record.id, observed)
            continue
        attempted = replace(pending, attempted=True)
        record = store.update_rollback_outcome(record.id, attempted)
        try:
            controller.restart(outcome.service)
        except Exception as exc:  # noqa: BLE001 - reconcile ambiguous systemd result
            observed = _observe_restart(
                controller,
                attempted,
                rollback_digest,
                timeout=observation_timeout,
            )
            if observed.status == ServiceOutcomeStatus.FAILED:
                observed = replace(
                    observed,
                    detail=f"rollback restart command failed: {exc}; {observed.detail}",
                )
            record = store.update_rollback_outcome(record.id, observed)
            continue
        observed = _observe_restart(
            controller,
            attempted,
            rollback_digest,
            timeout=observation_timeout,
        )
        record = store.update_rollback_outcome(record.id, observed)

    failed = tuple(
        outcome
        for outcome in record.rollback_outcomes
        if outcome.status != ServiceOutcomeStatus.SUCCEEDED
    )
    if config_digest(load_config()) != rollback_digest:
        return store.finish(
            record.id,
            status=ActivationStatus.FAILED,
            detail=(
                f"{record.detail} The previous voice could not be proven after "
                "restoration."
            ),
        )
    if failed:
        failures = "; ".join(
            f"{outcome.service}: {outcome.detail}" for outcome in failed
        )
        detail = (
            f"{record.detail} The previous voice was restored, but its service "
            f"snapshot could not be fully restored: {failures}"
        )
    else:
        detail = (
            f"{record.detail} The previous voice and service snapshot were restored."
        )
    return store.finish(record.id, status=ActivationStatus.FAILED, detail=detail)


def execute_activation(
    record_id: str,
    *,
    store: ActivationStore | None = None,
    controller: ServiceController | None = None,
    load_config: Callable[[], UserConfig] = load_user_config,
    voice_validator: Callable[[str], VoiceValidationResult] = validate_voice,
    restore_voice: Callable[[ActivationRecord], UserConfig] = _restore_previous_voice,
    observation_timeout: float = 30.0,
) -> ActivationRecord:
    """Execute or reconcile one durable request without blind restart repetition."""

    resolved_store = store or ActivationStore()
    resolved_controller = controller or SystemdUserServiceController()
    with resolved_store.execution_lock():
        return _execute_activation(
            record_id,
            store=resolved_store,
            controller=resolved_controller,
            load_config=load_config,
            voice_validator=voice_validator,
            restore_voice=restore_voice,
            observation_timeout=observation_timeout,
        )


def _execute_activation(
    record_id: str,
    *,
    store: ActivationStore,
    controller: ServiceController,
    load_config: Callable[[], UserConfig],
    voice_validator: Callable[[str], VoiceValidationResult],
    restore_voice: Callable[[ActivationRecord], UserConfig],
    observation_timeout: float,
) -> ActivationRecord:
    record = store.current()
    if record is None or record.id != record_id:
        raise ActivationStateError("activation request no longer exists")
    if record.status.value in _FINAL_STATES:
        return record
    if record.status == ActivationStatus.ROLLING_BACK:
        return _execute_voice_rollback(
            record,
            store=store,
            controller=controller,
            load_config=load_config,
            restore_voice=restore_voice,
            observation_timeout=observation_timeout,
        )
    if record.status == ActivationStatus.VALIDATING:
        # The attempt is durable, but its result is not. The provider call and this
        # journal cannot commit atomically, so retrying could duplicate an external
        # request and accepting could promote an unproven voice.
        record = store.begin_rollback(
            record.id,
            "Voice usability validation was attempted, but its result was not "
            "durably recorded. Validation was not repeated.",
        )
        return _execute_voice_rollback(
            record,
            store=store,
            controller=controller,
            load_config=load_config,
            restore_voice=restore_voice,
            observation_timeout=observation_timeout,
        )
    stored_config = load_config()
    if config_digest(stored_config) != record.expected_config_digest:
        if record.setting_key == "audio.voice" and record.status in {
            ActivationStatus.READY,
            ActivationStatus.RESTARTING,
        }:
            if not isinstance(record.new_value, str) or (
                stored_config.audio.voice != record.new_value
            ):
                record = store.begin_rollback(
                    record.id,
                    "The stored voice changed before voice activation completed.",
                )
                return _execute_voice_rollback(
                    record,
                    store=store,
                    controller=controller,
                    load_config=load_config,
                    restore_voice=restore_voice,
                    observation_timeout=observation_timeout,
                )
        else:
            if record.status == ActivationStatus.READY:
                record = store.begin_restart(record.id, record.outcomes)
            return store.finish(
                record.id,
                status=ActivationStatus.FAILED,
                detail="The stored configuration changed before activation.",
            )
    if (
        record.setting_key == "audio.voice"
        and record.status == ActivationStatus.READY
        and not record.voice_validated
    ):
        if not isinstance(record.new_value, str):
            record = store.begin_rollback(
                record.id,
                "Voice activation carried an invalid candidate value.",
            )
        else:
            candidate_voice = record.new_value
            record = store.begin_voice_validation(record.id)
            try:
                validation = voice_validator(candidate_voice)
            except Exception as exc:  # noqa: BLE001 - provider result is unproven
                validation = VoiceValidationResult(
                    False,
                    f"TTS validation failed: {type(exc).__name__}: {exc}",
                )
            if validation.usable:
                record = store.mark_voice_validated(record.id)
            else:
                record = store.begin_rollback(
                    record.id,
                    f"Voice {candidate_voice!r} was not usable: {validation.detail}",
                )
        if record.status == ActivationStatus.ROLLING_BACK:
            return _execute_voice_rollback(
                record,
                store=store,
                controller=controller,
                load_config=load_config,
                restore_voice=restore_voice,
                observation_timeout=observation_timeout,
            )
    if record.setting_key == "audio.voice" and record.status == ActivationStatus.READY:
        effective_config = load_config()
        if not isinstance(record.new_value, str) or (
            effective_config.audio.voice != record.new_value
        ):
            record = store.begin_rollback(
                record.id,
                "The stored voice changed before the service restart.",
            )
            return _execute_voice_rollback(
                record,
                store=store,
                controller=controller,
                load_config=load_config,
                restore_voice=restore_voice,
                observation_timeout=observation_timeout,
            )
        effective_digest = config_digest(effective_config)
        if effective_digest != record.expected_config_digest:
            record = store.update_voice_activation_digest(
                record.id,
                effective_digest,
            )
    if record.status == ActivationStatus.READY:
        initial: list[ServiceOutcome] = []
        for outcome in record.outcomes:
            before = controller.snapshot(outcome.service)
            if not before.installed:
                initial.append(
                    replace(
                        outcome,
                        status=ServiceOutcomeStatus.FAILED,
                        before=before,
                        detail="service is not installed",
                    )
                )
            elif before.active_state != "active":
                initial.append(
                    replace(
                        outcome,
                        status=ServiceOutcomeStatus.FAILED,
                        before=before,
                        detail=f"service is {before.active_state}; it was not started",
                    )
                )
            else:
                initial.append(replace(outcome, before=before))
        record = store.begin_restart(record.id, tuple(initial))
    if record.status != ActivationStatus.RESTARTING:
        raise ActivationStateError("activation request is not executable")
    expected_voice = (
        record.new_value
        if record.setting_key == "audio.voice" and isinstance(record.new_value, str)
        else None
    )
    for outcome in record.outcomes:
        if outcome.status != ServiceOutcomeStatus.PENDING:
            continue
        if outcome.attempted:
            reconciled = _observe_restart(
                controller,
                outcome,
                record.expected_config_digest,
                expected_voice=expected_voice,
                timeout=observation_timeout,
            )
            record = store.update_outcome(record.id, reconciled)
            continue
        attempted = replace(outcome, attempted=True)
        record = store.update_outcome(record.id, attempted)
        try:
            controller.restart(outcome.service)
        except Exception as exc:  # noqa: BLE001 - reconcile ambiguous systemd result
            observed = _observe_restart(
                controller,
                attempted,
                record.expected_config_digest,
                expected_voice=expected_voice,
                timeout=observation_timeout,
            )
            if observed.status == ServiceOutcomeStatus.FAILED:
                observed = replace(
                    observed,
                    detail=f"restart command failed: {exc}; {observed.detail}",
                )
            record = store.update_outcome(record.id, observed)
            continue
        observed = _observe_restart(
            controller,
            attempted,
            record.expected_config_digest,
            expected_voice=expected_voice,
            timeout=observation_timeout,
        )
        record = store.update_outcome(record.id, observed)
    outcomes = record.outcomes
    succeeded = sum(
        outcome.status == ServiceOutcomeStatus.SUCCEEDED for outcome in outcomes
    )
    final_config = load_config()
    stored_matches = config_digest(final_config) == record.expected_config_digest
    voice_owned = (
        record.setting_key == "audio.voice"
        and isinstance(record.new_value, str)
        and final_config.audio.voice == record.new_value
    )
    if record.setting_key == "audio.voice" and (
        not voice_owned or succeeded != len(outcomes)
    ):
        failures = "; ".join(
            f"{outcome.service}: {outcome.detail}"
            for outcome in outcomes
            if outcome.status == ServiceOutcomeStatus.FAILED
        )
        detail = (
            "Voice usability was proven, but stored voice ownership was lost."
            if not voice_owned
            else (
                "Voice usability was proven, but the candidate service snapshot was "
                f"not fully activated{f': {failures}' if failures else '.'}"
            )
        )
        record = store.begin_rollback(record.id, detail)
        return _execute_voice_rollback(
            record,
            store=store,
            controller=controller,
            load_config=load_config,
            restore_voice=restore_voice,
            observation_timeout=observation_timeout,
        )
    if record.setting_key == "audio.voice" and succeeded == len(outcomes):
        status = ActivationStatus.SUCCEEDED
        detail = (
            "The selected TTS provider accepted the voice and all targeted services "
            "loaded the expected voice snapshot."
        )
    elif not stored_matches:
        status = ActivationStatus.PARTIAL if succeeded else ActivationStatus.FAILED
        detail = (
            "Service actions completed, but the expected config is no longer stored."
        )
    elif succeeded == len(outcomes):
        status = ActivationStatus.SUCCEEDED
        detail = "All targeted services loaded the expected configuration."
    elif succeeded:
        status = ActivationStatus.PARTIAL
        detail = "; ".join(
            f"{outcome.service}: {outcome.detail}"
            for outcome in outcomes
            if outcome.status == ServiceOutcomeStatus.FAILED
        )
    else:
        status = ActivationStatus.FAILED
        detail = "; ".join(
            f"{outcome.service}: {outcome.detail}" for outcome in outcomes
        )
    return store.finish(record.id, status=status, detail=detail)


def launch_activation_worker(record_id: str) -> subprocess.Popen[bytes]:
    """Launch the executor in a transient scope outside the wake service cgroup."""

    unit = f"voice-harness-config-activation-{record_id[:16]}"
    return subprocess.Popen(
        [
            "systemd-run",
            "--user",
            "--scope",
            "--collect",
            "--quiet",
            f"--unit={unit}",
            sys.executable,
            "-m",
            "local_voice_harness.config_activation",
            "--execute",
            record_id,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", metavar="ACTIVATION_ID", required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    execute_activation(arguments.execute)


if __name__ == "__main__":
    main()
