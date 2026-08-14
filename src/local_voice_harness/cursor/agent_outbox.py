"""AgentHarness-only handlers for durable session effects."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

from ..agents.harness import (
    AgentHarness,
    HarnessCapability,
    HarnessEvent,
    HarnessSession,
    HarnessTask,
    SessionRequest,
)
from .coordinator import (
    CoordinatorCommand,
    CoordinatorDecision,
    EffectObservation,
    OutboxLease,
    OutboxResult,
)
from .model import CursorJob
from .outbox import EffectHandler
from .store import JobStore

SESSION_CREATE = "session.create"
TASK_SUBMIT = "task.submit"
CLARIFICATION_REPLY = "clarification.reply"
AGENT_EFFECT_KINDS = (
    SESSION_CREATE,
    TASK_SUBMIT,
    CLARIFICATION_REPLY,
)

AgentResultReducer = Callable[[CursorJob, OutboxResult], CursorJob | None]
HarnessFactory = Callable[[], AgentHarness]


class AgentEffectPayloadError(ValueError):
    """A durable agent effect payload is incomplete or malformed."""


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise AgentEffectPayloadError(f"agent effect requires {name}")
    return value


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AgentEffectPayloadError(f"agent effect requires nonnegative {name}")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise AgentEffectPayloadError(f"agent effect requires boolean {name}")
    return value


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name)


def session_payload(session: HarnessSession) -> dict[str, object]:
    return {
        "provider": session.provider,
        "session_id": session.session_id,
        "target": session.target,
        "state_sequence": session.state_sequence,
        "metadata": dict(session.metadata),
    }


def load_session(payload: Mapping[str, object]) -> HarnessSession:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise AgentEffectPayloadError("agent effect has invalid session metadata")
    return HarnessSession(
        provider=_string(payload.get("provider"), "provider"),
        session_id=_string(payload.get("session_id"), "session_id"),
        target=_string(payload.get("target"), "target"),
        state_sequence=_integer(payload.get("state_sequence"), "state_sequence"),
        metadata=cast(dict[str, str], metadata),
    )


def event_payload(event: HarnessEvent) -> dict[str, object]:
    return {
        "kind": event.kind.value,
        "session": session_payload(event.session),
        "status": event.status,
        "output": event.output,
        "summary": event.summary,
        "question": event.question,
        "error": event.error,
        "revision": event.revision,
    }


def _request(payload: Mapping[str, object]) -> SessionRequest:
    launch_context = payload.get("launch_context", {})
    capabilities = payload.get("required_capabilities", [])
    if not isinstance(launch_context, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in launch_context.items()
    ):
        raise AgentEffectPayloadError("session.create has invalid launch_context")
    if not isinstance(capabilities, list) or not all(
        isinstance(value, str) for value in capabilities
    ):
        raise AgentEffectPayloadError("session.create has invalid capabilities")
    mode = payload.get("mode")
    if mode is not None and not isinstance(mode, str):
        raise AgentEffectPayloadError("session.create has invalid mode")
    try:
        required = frozenset(HarnessCapability(value) for value in capabilities)
    except ValueError as exc:
        raise AgentEffectPayloadError("session.create has unknown capability") from exc
    return SessionRequest(
        name=_string(payload.get("name"), "name"),
        provider=_string(payload.get("provider"), "provider"),
        mode=mode,
        launch_context=cast(dict[str, str], launch_context),
        required_capabilities=required,
    )


def _task(payload: Mapping[str, object], session: HarnessSession) -> HarnessTask:
    baseline = _integer(payload.get("baseline_sequence"), "baseline_sequence")
    if baseline != session.state_sequence:
        raise AgentEffectPayloadError("task baseline does not match durable session")
    return HarnessTask(
        text=_string(payload.get("text"), "text"),
        correlation_id=_string(payload.get("correlation_id"), "correlation_id"),
        expected_session_id=session.session_id,
        baseline_sequence=baseline,
        completion_marker=_optional_string(
            payload.get("completion_marker"),
            "completion_marker",
        ),
        allow_interactive_boundary=_boolean(
            payload.get("allow_interactive_boundary", False),
            "allow_interactive_boundary",
        ),
        allow_fallback_submit=_boolean(
            payload.get("allow_fallback_submit", True),
            "allow_fallback_submit",
        ),
    )


def _session_create_handler(
    harness: AgentHarness,
) -> EffectHandler:
    def handle(
        lease: OutboxLease, mark_dispatched: Callable[[], None]
    ) -> EffectObservation:
        dispatched = False

        def before_submit() -> None:
            nonlocal dispatched
            mark_dispatched()
            dispatched = True

        session = harness.create_session(
            _request(lease.payload),
            before_submit=before_submit,
        )
        if not dispatched:
            return EffectObservation(
                outcome="OutcomeUnknown",
                detail={"error": "harness omitted the session dispatch fence"},
            )
        return EffectObservation(
            outcome="Confirmed",
            detail={"session": session_payload(session)},
        )

    return handle


def _task_handler(
    harness: AgentHarness,
    *,
    clarification: bool,
) -> EffectHandler:
    def handle(
        lease: OutboxLease, mark_dispatched: Callable[[], None]
    ) -> EffectObservation:
        session_value = lease.payload.get("session")
        if not isinstance(session_value, dict):
            raise AgentEffectPayloadError("task effect requires session")
        session = load_session(session_value)
        task = _task(lease.payload, session)
        baseline = task.baseline_sequence
        assert baseline is not None
        accepted = False

        def before_submit(observed_baseline: int) -> None:
            if observed_baseline != baseline:
                raise AgentEffectPayloadError(
                    "provider observed a different task baseline"
                )
            mark_dispatched()

        def record_accepted() -> None:
            nonlocal accepted
            accepted = True

        submit = (
            harness.reply_to_clarification if clarification else harness.submit_task
        )
        submission = submit(
            session,
            task,
            before_submit=before_submit,
            accepted=record_accepted,
        )
        if not accepted:
            return EffectObservation(
                outcome="OutcomeUnknown",
                detail={"error": "provider returned without positive acceptance"},
            )
        if (
            submission.session.provider != session.provider
            or submission.session.session_id != session.session_id
            or submission.session.target != session.target
            or submission.session.state_sequence != baseline
            or submission.correlation_id != task.correlation_id
            or submission.baseline_sequence != baseline
        ):
            return EffectObservation(
                outcome="OutcomeUnknown",
                detail={"error": "provider returned a mismatched task submission"},
            )
        events = tuple(harness.stream_events(submission))
        if len(events) != 1:
            return EffectObservation(
                outcome="OutcomeUnknown",
                detail={"error": "provider returned an incomplete task event stream"},
            )
        event = events[0]
        if (
            event.session.provider != session.provider
            or event.session.session_id != session.session_id
            or event.session.target != session.target
            or event.session.state_sequence <= baseline
        ):
            return EffectObservation(
                outcome="OutcomeUnknown",
                detail={"error": "task event did not advance the fenced session"},
            )
        return EffectObservation(
            outcome="Confirmed",
            detail={
                "submission": {
                    "correlation_id": submission.correlation_id,
                    "baseline_sequence": submission.baseline_sequence,
                },
                "event": event_payload(event),
            },
        )

    return handle


def agent_effect_handlers(
    harness: AgentHarness,
) -> dict[str, EffectHandler]:
    """Return the complete AgentHarness-owned effect family."""

    return {
        SESSION_CREATE: _session_create_handler(harness),
        TASK_SUBMIT: _task_handler(
            harness,
            clarification=False,
        ),
        CLARIFICATION_REPLY: _task_handler(
            harness,
            clarification=True,
        ),
    }


def lazy_agent_effect_handlers(factory: HarnessFactory) -> dict[str, EffectHandler]:
    """Create built-in handlers without constructing a provider until claimed."""

    handlers: dict[str, EffectHandler] = {}
    for kind in AGENT_EFFECT_KINDS:

        def handle(
            lease: OutboxLease,
            mark_dispatched: Callable[[], None],
            effect_kind: str = kind,
        ) -> EffectObservation:
            return agent_effect_handlers(factory())[effect_kind](
                lease,
                mark_dispatched,
            )

        handlers[kind] = handle
    return handlers


def consume_agent_results(
    store: JobStore,
    reducer: AgentResultReducer,
    *,
    limit: int = 32,
) -> int:
    """Reduce terminal agent observations through revision-fenced commands."""

    consumed = 0
    for result in store.unconsumed_outbox_results(AGENT_EFFECT_KINDS, limit=limit):
        consume_agent_result(store, result, reducer)
        consumed += 1
    return consumed


def consume_agent_result(
    store: JobStore,
    result: OutboxResult,
    reducer: AgentResultReducer,
) -> str:
    """Consume one exact result through its persisted identity and revision fence."""

    expected = result.payload.get("expected_revision")
    if not isinstance(expected, int) or isinstance(expected, bool) or expected < 0:
        store.mark_outbox_consumed(result, "invalid")
        return "invalid"
    called = False

    def decide(job: CursorJob) -> CoordinatorDecision | None:
        nonlocal called
        called = True
        candidate = reducer(job, result)
        if candidate is None:
            return None
        return CoordinatorDecision(
            job=candidate,
            event_kind=f"{result.kind}.observed",
            event_payload={
                "effect_id": result.effect_id,
                "idempotency_key": result.idempotency_key,
                "effect_status": result.status,
            },
        )

    def apply_at(revision: int, command_id: str) -> CursorJob | None:
        return store.apply(
            CoordinatorCommand(
                job_id=result.job_id,
                expected_revision=revision,
                command_id=command_id,
                kind=f"{result.kind}.observe",
            ),
            decide,
        )

    updated = apply_at(expected, f"observe:{result.effect_id}")
    disposition = (
        "applied" if updated is not None else ("rejected" if called else "stale")
    )
    if disposition == "stale":
        for _attempt in range(4):
            current = store.get(result.job_id)
            called = False
            updated = apply_at(
                current.revision,
                f"observe:{result.effect_id}:revision:{current.revision}",
            )
            disposition = (
                "applied"
                if updated is not None
                else ("rejected" if called else "stale")
            )
            if disposition != "stale":
                break
    if disposition != "stale":
        store.mark_outbox_consumed(result, disposition)
    return disposition
