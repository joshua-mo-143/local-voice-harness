from __future__ import annotations

import hashlib
import subprocess
import time
from typing import Any, Protocol

from .types import (
    AGENT_COMPLETION_POLL_SECONDS,
    AGENT_COMPLETION_QUIET_SECONDS,
    AGENT_PROMPT_WAIT_SECONDS,
    OBSERVABLE_AGENT_STATES,
    SETTLED,
    BeforePromptSubmit,
    Checkpoint,
    HerdrError,
    PromptAccepted,
    PromptBoundary,
    PromptOutcome,
    agent_session_identity,
    extract_marker,
)


class HerdrSessionClient(Protocol):
    def command(self, *args: str) -> list[str]: ...

    def decode(self, text: str) -> dict[str, Any]: ...

    def run_json(self, *args: str, timeout: float = 30) -> dict[str, Any]: ...

    def run_text(self, *args: str, timeout: float = 30) -> str: ...

    def get_agent(self, target: str) -> dict[str, Any]: ...


class HerdrSession:
    """Agent prompt, completion, and cancellation handling."""

    def __init__(self, client: HerdrSessionClient) -> None:
        self._client = client

    def prompt_and_wait(
        self,
        target: str,
        text: str,
        *,
        token: str,
        timeout: float = 15 * 60,
        max_runtime: float = 60 * 60,
        checkpoint: Checkpoint | None = None,
        baseline_sequence: int | None = None,
        expected_agent_session: str | None = None,
        before_submit: BeforePromptSubmit | None = None,
        accepted: PromptAccepted | None = None,
        before_agent: PromptBoundary | None = None,
        after_submit: PromptBoundary | None = None,
        active_marker: str | None = None,
        allow_interactive_plan_boundary: bool = False,
        allow_enter_fallback: bool = True,
    ) -> PromptOutcome:
        started_at = time.monotonic()
        if checkpoint is not None:
            checkpoint()
        before = self._client.get_agent(target)
        before_session = agent_session_identity(before.get("agent_session"))
        if (
            expected_agent_session is not None
            and before_session != expected_agent_session
        ):
            raise HerdrError(
                f"Herdr agent {target} no longer has the expected session",
                code="agent_session_changed",
            )
        if before.get("interactive_ready") is False:
            raise HerdrError(
                f"Herdr agent {target} is showing an interactive questionnaire",
                code="interactive_questionnaire",
            )
        observed_baseline = int(before.get("state_change_seq") or 0)
        if baseline_sequence is not None and observed_baseline != baseline_sequence:
            raise HerdrError(
                "Herdr agent changed before the planned prompt was submitted",
                code="operation_ambiguous",
            )
        if before_agent is not None:
            before_agent(before)
        if checkpoint is not None:
            checkpoint()
            checkpoint()
        if before_submit is not None:
            before_submit(observed_baseline)
        process = subprocess.Popen(
            self._client.command(
                "agent",
                "prompt",
                target,
                text,
                "--wait",
                "--timeout",
                str(int(AGENT_PROMPT_WAIT_SECONDS * 1000)),
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        acceptance_recorded = False
        observed_acceptance = False
        interactive_plan_boundary = False

        def expected_interactive_plan_boundary(agent: dict[str, Any]) -> bool:
            if (
                not allow_interactive_plan_boundary
                or active_marker != "WORKFLOW_PLAN"
                or str(agent.get("agent_status") or "") not in {"working", "blocked"}
            ):
                return False
            output = self._client.run_text(
                "agent",
                "read",
                target,
                "--source",
                "recent-unwrapped",
                "--lines",
                "160",
            )
            return bool(
                extract_marker(output, "WORKFLOW_PLAN", token)
                and not extract_marker(output, "VOICE_QUESTION", token)
            )

        try:
            if after_submit is not None:
                after_submit(before)
            if checkpoint is not None:
                checkpoint()
            time.sleep(0.35)
            if checkpoint is not None:
                checkpoint()
            current = self._client.get_agent(target)
            if (
                expected_agent_session is not None
                and agent_session_identity(current.get("agent_session"))
                != expected_agent_session
            ):
                raise HerdrError(
                    f"Herdr agent {target} changed sessions during prompt submission",
                    code="agent_session_changed",
                )
            observed_acceptance = current.get("state_change_seq") != before.get(
                "state_change_seq"
            )
            if checkpoint is not None:
                checkpoint()
            if current.get("interactive_ready") is False:
                interactive_plan_boundary = expected_interactive_plan_boundary(current)
                if not interactive_plan_boundary:
                    raise HerdrError(
                        f"Herdr agent {target} opened an interactive questionnaire",
                        code="interactive_questionnaire",
                    )
            if (
                current.get("state_change_seq") == before.get("state_change_seq")
                and current.get("agent_status") in SETTLED
                and allow_enter_fallback
            ):
                if checkpoint is not None:
                    checkpoint()
                self._client.run_json("agent", "send-keys", target, "enter")
                if checkpoint is not None:
                    checkpoint()
            elif observed_acceptance and accepted is not None:
                accepted()
                acceptance_recorded = True
            deadline = time.monotonic() + AGENT_PROMPT_WAIT_SECONDS + 5
            while process.poll() is None and not interactive_plan_boundary:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    process.wait(timeout=min(1.0, remaining))
                except subprocess.TimeoutExpired:
                    if checkpoint is not None:
                        checkpoint()
                    current = self._client.get_agent(target)
                    if (
                        expected_agent_session is not None
                        and agent_session_identity(current.get("agent_session"))
                        != expected_agent_session
                    ):
                        raise HerdrError(
                            f"Herdr agent {target} changed sessions during "
                            "prompt submission",
                            code="agent_session_changed",
                        ) from None
                    observed_acceptance = observed_acceptance or (
                        current.get("state_change_seq")
                        != before.get("state_change_seq")
                    )
                    if current.get("interactive_ready") is False:
                        interactive_plan_boundary = expected_interactive_plan_boundary(
                            current
                        )
                        if not interactive_plan_boundary:
                            raise HerdrError(
                                f"Herdr agent {target} opened an interactive "
                                "questionnaire",
                                code="interactive_questionnaire",
                            ) from None
            if interactive_plan_boundary and process.poll() is None:
                process.kill()
            stdout, stderr = process.communicate(timeout=AGENT_PROMPT_WAIT_SECONDS + 5)
            if checkpoint is not None:
                checkpoint()
        except Exception:
            process.kill()
            process.wait()
            raise
        if interactive_plan_boundary:
            pass
        elif process.returncode:
            try:
                self._client.decode(stdout or stderr)
            except HerdrError as exc:
                if exc.code not in {"operation_timeout", "timeout"}:
                    raise
        else:
            self._client.decode(stdout)
        if expected_agent_session is not None or (
            accepted is not None and not acceptance_recorded
        ):
            if checkpoint is not None:
                checkpoint()
            current = self._client.get_agent(target)
            if (
                expected_agent_session is not None
                and agent_session_identity(current.get("agent_session"))
                != expected_agent_session
            ):
                raise HerdrError(
                    f"Herdr agent {target} changed sessions during prompt submission",
                    code="agent_session_changed",
                )
            if accepted is not None and not acceptance_recorded:
                observed_acceptance = observed_acceptance or (
                    current.get("state_change_seq") != before.get("state_change_seq")
                )
            if checkpoint is not None:
                checkpoint()
        if accepted is not None and not acceptance_recorded and observed_acceptance:
            accepted()
            acceptance_recorded = True
        if accepted is not None and not acceptance_recorded:
            raise HerdrError(
                f"Herdr did not accept the prompt for agent {target}",
                code="agent_prompt_stalled",
            )
        return self.wait_for_stable_completion(
            target,
            token=token,
            inactivity_timeout=timeout,
            max_runtime=max_runtime,
            started_at=started_at,
            checkpoint=checkpoint,
            expected_agent_session=expected_agent_session,
            active_marker=active_marker,
            allow_interactive_plan_boundary=allow_interactive_plan_boundary,
        )

    def wait_for_stable_completion(
        self,
        target: str,
        *,
        token: str,
        inactivity_timeout: float = 15 * 60,
        max_runtime: float = 60 * 60,
        quiet_period: float = AGENT_COMPLETION_QUIET_SECONDS,
        started_at: float | None = None,
        checkpoint: Checkpoint | None = None,
        expected_agent_session: str | None = None,
        active_marker: str | None = None,
        allow_interactive_plan_boundary: bool = False,
    ) -> PromptOutcome:
        """Wait for stable completion or an opt-in active marker boundary."""

        started = time.monotonic() if started_at is None else started_at
        last_activity = started
        last_signature: tuple[object, str, str] | None = None
        initial_session: str | None = None
        settled_since: float | None = None
        boundary_since: float | None = None
        marker_output: str | None = None

        while True:
            if checkpoint is not None:
                checkpoint()
            agent = self._client.get_agent(target)
            if checkpoint is not None:
                checkpoint()
            output = self._client.run_text(
                "agent",
                "read",
                target,
                "--source",
                "recent-unwrapped",
                "--lines",
                "160",
            )
            if checkpoint is not None:
                checkpoint()

            now = time.monotonic()
            status = str(agent.get("agent_status") or "unknown")
            session = agent_session_identity(agent.get("agent_session"))
            if expected_agent_session is not None and session != expected_agent_session:
                raise HerdrError(
                    f"Herdr agent {target} no longer has the expected session",
                    code="agent_session_changed",
                )
            if initial_session is None:
                initial_session = session
            elif session is not None and session != initial_session:
                raise HerdrError(
                    f"Herdr agent {target} changed sessions while work was active",
                    code="agent_session_changed",
                )

            signature = (
                agent.get("state_change_seq"),
                status,
                hashlib.sha256(output.encode()).hexdigest(),
            )
            if signature != last_signature:
                last_signature = signature
                last_activity = now
                settled_since = now if status in OBSERVABLE_AGENT_STATES else None
                boundary_since = None

            summary = extract_marker(output, "VOICE_SUMMARY", token)
            question = extract_marker(output, "VOICE_QUESTION", token)
            if summary or question:
                marker_output = output
            boundary = (
                extract_marker(output, active_marker, token) if active_marker else None
            )
            expected_plan_boundary = bool(
                allow_interactive_plan_boundary
                and active_marker == "WORKFLOW_PLAN"
                and boundary
                and not question
                and status in {"working", "blocked"}
            )
            if agent.get("interactive_ready") is False and not expected_plan_boundary:
                raise HerdrError(
                    f"Herdr agent {target} opened an interactive questionnaire",
                    code="interactive_questionnaire",
                )
            if boundary and not question and status in {"working", "blocked"}:
                marker_output = output
                if boundary_since is None:
                    boundary_since = now
                if now - boundary_since >= quiet_period:
                    sequence = agent.get("state_change_seq")
                    revision = agent.get("revision")
                    return PromptOutcome(
                        status=status,
                        summary=summary,
                        question=None,
                        output=output,
                        boundary_marker=active_marker,
                        agent_session=session,
                        state_change_sequence=(
                            sequence
                            if isinstance(sequence, int)
                            and not isinstance(sequence, bool)
                            else None
                        ),
                        revision=(
                            revision
                            if isinstance(revision, int)
                            and not isinstance(revision, bool)
                            else None
                        ),
                    )
            else:
                boundary_since = None

            if status in OBSERVABLE_AGENT_STATES:
                if settled_since is None:
                    settled_since = now
                if now - settled_since >= quiet_period:
                    selected_output = marker_output or output
                    return PromptOutcome(
                        status=status,
                        summary=extract_marker(selected_output, "VOICE_SUMMARY", token),
                        question=extract_marker(
                            selected_output, "VOICE_QUESTION", token
                        ),
                        output=selected_output,
                        boundary_marker=(
                            active_marker
                            if active_marker
                            and extract_marker(selected_output, active_marker, token)
                            and not extract_marker(
                                selected_output, "VOICE_QUESTION", token
                            )
                            else None
                        ),
                        agent_session=session,
                        state_change_sequence=(
                            agent.get("state_change_seq")
                            if isinstance(agent.get("state_change_seq"), int)
                            and not isinstance(agent.get("state_change_seq"), bool)
                            else None
                        ),
                        revision=(
                            agent.get("revision")
                            if isinstance(agent.get("revision"), int)
                            and not isinstance(agent.get("revision"), bool)
                            else None
                        ),
                    )
            else:
                settled_since = None

            inactivity_expired = now - last_activity >= inactivity_timeout
            runtime_expired = now - started >= max_runtime
            if inactivity_expired or runtime_expired:
                reason = (
                    "inactivity timeout" if inactivity_expired else "maximum runtime"
                )
                raise HerdrError(
                    f"Herdr agent {target} exceeded its {reason}",
                    code="agent_stalled",
                )

            remaining = min(
                inactivity_timeout - (now - last_activity),
                max_runtime - (now - started),
                AGENT_COMPLETION_POLL_SECONDS,
            )
            time.sleep(max(0.0, remaining))
            if checkpoint is not None:
                checkpoint()

    def cancel_agent(self, target: str) -> None:
        self._client.run_json("agent", "send-keys", target, "ctrl+c")
        result = self._client.run_json(
            "agent", "wait", target, "--timeout", "5000", timeout=10
        )
        agent = dict(result.get("agent") or {})
        if agent.get("agent_status") == "working":
            raise HerdrError(f"Herdr agent {target} did not stop")

    def close_owned_pane(
        self,
        target: str,
        pane_id: str,
        workspace_id: str,
    ) -> None:
        """Close one pane only when every observable ownership binding matches."""

        if not target or not pane_id or not workspace_id:
            raise HerdrError(
                "owned pane cleanup requires target, pane, and workspace identity",
                code="ownership_mismatch",
            )

        agent: dict[str, Any] | None
        try:
            agent = self._client.get_agent(target)
        except HerdrError as exc:
            if exc.code not in {"agent_not_found", "not_found"}:
                raise
            agent = None
        if agent is not None:
            observed_target = str(agent.get("name") or agent.get("pane_id") or "")
            if (
                target not in {observed_target, str(agent.get("pane_id") or "")}
                or str(agent.get("pane_id") or "") != pane_id
                or str(agent.get("workspace_id") or "") != workspace_id
            ):
                raise HerdrError(
                    "Herdr agent no longer matches the owned pane binding",
                    code="ownership_mismatch",
                )

        try:
            pane = dict(self._client.run_json("pane", "get", pane_id).get("pane") or {})
        except HerdrError as exc:
            if exc.code not in {"pane_not_found", "not_found"}:
                raise
            if agent is not None:
                raise HerdrError(
                    "owned agent exists but its pane is absent",
                    code="ownership_mismatch",
                ) from exc
            return
        if (
            str(pane.get("pane_id") or "") != pane_id
            or str(pane.get("workspace_id") or "") != workspace_id
        ):
            raise HerdrError(
                "Herdr pane no longer matches the owned workspace binding",
                code="ownership_mismatch",
            )

        if agent is not None:
            try:
                self.cancel_agent(target)
            except HerdrError as exc:
                if exc.code not in {"agent_not_found", "not_found"}:
                    raise
        try:
            self._client.run_json("pane", "close", pane_id)
        except HerdrError as exc:
            if exc.code not in {"pane_not_found", "not_found"}:
                raise

        try:
            self._client.run_json("pane", "get", pane_id)
        except HerdrError as exc:
            if exc.code not in {"pane_not_found", "not_found"}:
                raise
        else:
            raise HerdrError(
                f"Herdr pane {pane_id} still exists after close",
                code="pane_close_unconfirmed",
            )

        try:
            self._client.get_agent(target)
        except HerdrError as exc:
            if exc.code in {"agent_not_found", "not_found"}:
                return
            raise
        raise HerdrError(
            f"Herdr agent {target} still exists after pane close",
            code="pane_close_unconfirmed",
        )
