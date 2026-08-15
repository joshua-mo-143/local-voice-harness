from __future__ import annotations

import json
import unittest
from unittest import mock

from local_voice_harness.agents import (
    HarnessCapability,
    HarnessEvent,
    HarnessEventKind,
    HarnessSession,
    HarnessTask,
    ReconciliationState,
    SessionRequest,
    TaskSubmission,
    UnsupportedCapabilityError,
    require_capabilities,
)
from local_voice_harness.integrations.herdr import (
    HerdrSession,
    OpenCodeSession,
    PromptOutcome,
)

from .harness_contract import HarnessContractTests


class _FakeHerdr:
    def __init__(self) -> None:
        self.agent = {
            "agent": "cursor",
            "name": "contract-agent",
            "pane_id": "workspace:pane",
            "workspace_id": "workspace",
            "cwd": "/checkout",
            "interactive_ready": True,
            "agent_status": "working",
            "agent_session": {"conversation_id": "durable-session"},
            "state_change_seq": 1,
        }
        self.calls: list[tuple[object, ...]] = []

    def command(self, *args: str) -> list[str]:
        return ["herdr", *args]

    @staticmethod
    def decode(text: str) -> dict[str, object]:
        return dict(json.loads(text).get("result") or {})

    def run_json(self, *args: str, timeout: float | None = None) -> dict[str, object]:
        self.calls.append((*args, timeout))
        if args[:2] == ("agent", "start"):
            return {"agent": dict(self.agent)}
        if args[:2] == ("agent", "wait"):
            return {"agent": {**self.agent, "agent_status": "idle"}}
        return {}

    def run_text(self, *args: str, timeout: float | None = None) -> str:
        self.calls.append((*args, timeout))
        return ""

    def get_agent(self, target: str) -> dict[str, object]:
        if target != self.agent["name"]:
            raise AssertionError(f"unexpected target {target}")
        return dict(self.agent)


class _HerdrScenario:
    def __init__(self) -> None:
        self.client = _FakeHerdr()
        self.harness = HerdrSession(self.client)
        self.session: HarnessSession | None = None

    def create(self) -> HarnessSession:
        if self.session is None:
            self.session = self.harness.create_session(
                SessionRequest(
                    name="contract-agent",
                    provider="cursor/herdr",
                    launch_context={
                        "pane_id": "workspace:pane",
                        "workspace_id": "workspace",
                    },
                )
            )
        return self.session

    def _submission(self) -> TaskSubmission:
        session = self.create()
        return TaskSubmission(session, "turn", 1, 0.0)

    def events(self, kind: HarnessEventKind) -> list[HarnessEvent]:
        outcomes = {
            HarnessEventKind.SUCCEEDED: PromptOutcome(
                "done", "complete", None, "VOICE_SUMMARY[turn]: complete"
            ),
            HarnessEventKind.CLARIFICATION: PromptOutcome(
                "blocked", None, "Which branch?", "VOICE_QUESTION[turn]: Which branch?"
            ),
            HarnessEventKind.FAILED: PromptOutcome(
                "unknown", None, None, "provider output unavailable"
            ),
        }
        with mock.patch.object(
            self.harness,
            "wait_for_stable_completion",
            return_value=outcomes[kind],
        ):
            return list(self.harness.stream_events(self._submission()))

    def reply(self) -> None:
        session = self.create()
        task = HarnessTask("Use main", "reply", session.session_id, 1)
        submission = TaskSubmission(session, "reply", 1, 0.0)
        with mock.patch.object(
            self.harness, "submit_task", return_value=submission
        ) as submit:
            self.harness.reply_to_clarification(session, task)
        submit.assert_called_once()

    def cancel(self) -> None:
        self.harness.cancel(self.create())
        self._assert_called("agent", "send-keys", "contract-agent", "ctrl+c")
        self._assert_called("agent", "wait", "contract-agent", "--timeout", "5000")

    def restart_state(self) -> ReconciliationState:
        restarted = HerdrSession(self.client)
        return restarted.reconcile(
            self.create().target,
            expected_session_id=self.create().session_id,
        ).state

    def _assert_called(self, *prefix: str) -> None:
        if not any(call[: len(prefix)] == prefix for call in self.client.calls):
            raise AssertionError(
                f"missing Herdr call {prefix!r}: {self.client.calls!r}"
            )


class _OpenCodeScenario(_HerdrScenario):
    def __init__(self) -> None:
        super().__init__()
        self.client.agent["agent"] = "opencode"
        self.harness = OpenCodeSession(self.client)

    def create(self) -> HarnessSession:
        if self.session is None:
            self.session = self.harness.create_session(
                SessionRequest(
                    name="contract-agent",
                    provider="opencode/herdr",
                    launch_context={
                        "pane_id": "workspace:pane",
                        "workspace_id": "workspace",
                    },
                )
            )
        return self.session

    def restart_state(self) -> ReconciliationState:
        restarted = OpenCodeSession(self.client)
        return restarted.reconcile(
            self.create().target,
            expected_session_id=self.create().session_id,
        ).state


class CursorHerdrHarnessContractTests(HarnessContractTests):
    __test__ = True

    def scenario(self) -> _HerdrScenario:
        return _HerdrScenario()

    def test_unsupported_capability_fails_before_side_effect(self) -> None:
        with self.assertRaises(UnsupportedCapabilityError) as raised:
            require_capabilities(
                "minimal-harness",
                frozenset(),
                frozenset({HarnessCapability.RECOVERY}),
            )
        self.assertEqual(raised.exception.code, "unsupported_capability")
        self.assertIn("choose a compatible harness", str(raised.exception))

    def test_session_creation_uses_cursor_mcp_transport_capability(self) -> None:
        scenario = self.scenario()
        scenario.create()
        start = next(
            call for call in scenario.client.calls if call[:2] == ("agent", "start")
        )
        self.assertIn("--approve-mcps", start)
        self.assertIn(HarnessCapability.MCP_CONNECTORS, scenario.harness.capabilities)


class OpenCodeHerdrHarnessContractTests(HarnessContractTests):
    __test__ = True

    def scenario(self) -> _OpenCodeScenario:
        return _OpenCodeScenario()

    def test_advertised_capabilities_exclude_cursor_mcp(self) -> None:
        scenario = self.scenario()
        self.assertNotIn(
            HarnessCapability.MCP_CONNECTORS, scenario.harness.capabilities
        )
        self.assertEqual(
            scenario.harness.capabilities,
            frozenset(
                {
                    HarnessCapability.CLARIFICATION_REPLIES,
                    HarnessCapability.CANCELLATION,
                    HarnessCapability.RECOVERY,
                }
            ),
        )

    def test_mcp_capability_fails_before_session_creation(self) -> None:
        scenario = self.scenario()
        with self.assertRaises(UnsupportedCapabilityError) as raised:
            scenario.harness.create_session(
                SessionRequest(
                    name="contract-agent",
                    provider="opencode/herdr",
                    launch_context={
                        "pane_id": "workspace:pane",
                        "workspace_id": "workspace",
                    },
                    required_capabilities=frozenset({HarnessCapability.MCP_CONNECTORS}),
                )
            )
        self.assertEqual(raised.exception.code, "unsupported_capability")
        self.assertFalse(
            any(call[:2] == ("agent", "start") for call in scenario.client.calls)
        )

    def test_session_creation_uses_opencode_kind_without_cursor_flags(self) -> None:
        scenario = self.scenario()
        scenario.create()
        start = next(
            call for call in scenario.client.calls if call[:2] == ("agent", "start")
        )
        self.assertIn("--kind", start)
        self.assertEqual(start[start.index("--kind") + 1], "opencode")
        self.assertNotIn("--trust", start)
        self.assertNotIn("--approve-mcps", start)
        self.assertNotIn("--mode", start)
        self.assertIn("--agent", start)
        self.assertEqual(start[start.index("--agent") + 1], "build")


if __name__ == "__main__":
    unittest.main()
