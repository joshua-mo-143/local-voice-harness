"""Reusable lifecycle assertions for any AgentHarness implementation."""

from __future__ import annotations

import unittest
from typing import Protocol

from local_voice_harness.agents import (
    AgentHarness,
    HarnessEvent,
    HarnessEventKind,
    HarnessSession,
    ReconciliationState,
)


class HarnessScenario(Protocol):
    @property
    def harness(self) -> object: ...

    def create(self) -> HarnessSession: ...

    def events(self, kind: HarnessEventKind) -> list[HarnessEvent]: ...

    def reply(self) -> None: ...

    def cancel(self) -> None: ...

    def restart_state(self) -> ReconciliationState: ...


class HarnessContractTests(unittest.TestCase):
    """Mixin reusable by concrete harness test suites."""

    __test__ = False

    def scenario(self) -> HarnessScenario:
        raise NotImplementedError

    def test_contract_success(self) -> None:
        scenario = self.scenario()
        session = scenario.create()
        events = scenario.events(HarnessEventKind.SUCCEEDED)
        self.assertTrue(isinstance(scenario.harness, AgentHarness))
        self.assertEqual(events[-1].kind, HarnessEventKind.SUCCEEDED)
        self.assertEqual(events[-1].session.session_id, session.session_id)

    def test_contract_clarification_reply(self) -> None:
        scenario = self.scenario()
        scenario.create()
        events = scenario.events(HarnessEventKind.CLARIFICATION)
        self.assertEqual(events[-1].kind, HarnessEventKind.CLARIFICATION)
        self.assertTrue(events[-1].question)
        scenario.reply()

    def test_contract_cancellation(self) -> None:
        scenario = self.scenario()
        scenario.create()
        scenario.cancel()

    def test_contract_failure(self) -> None:
        scenario = self.scenario()
        scenario.create()
        events = scenario.events(HarnessEventKind.FAILED)
        self.assertEqual(events[-1].kind, HarnessEventKind.FAILED)
        self.assertTrue(events[-1].error)

    def test_contract_restart_reconciliation(self) -> None:
        scenario = self.scenario()
        scenario.create()
        self.assertEqual(scenario.restart_state(), ReconciliationState.ACTIVE)
