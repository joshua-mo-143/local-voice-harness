from __future__ import annotations

from pathlib import Path

import pytest

from local_voice_harness.cursor import lifecycle_ownership as ownership
from local_voice_harness.cursor.model import JobStatus
from local_voice_harness.prompt_operations import PromptOperationState as JobPromptState
from local_voice_harness.questions import PromptOperationState as QuestionPromptState

REPO = Path(__file__).resolve().parent.parent


def test_every_persisted_field_has_exactly_one_owner() -> None:
    persisted = ownership.persisted_field_names()
    inventoried = {row.name for row in ownership.FIELD_OWNERSHIP}

    assert inventoried == persisted
    assert len(ownership.FIELD_OWNERSHIP) == len(inventoried)


def test_field_owners_point_at_real_tables_and_crash_boundaries() -> None:
    fence_kinds = {
        ownership.CrashKind.IDENTITY,
        ownership.CrashKind.REVISION,
        ownership.CrashKind.TOKEN,
        ownership.CrashKind.TIMESTAMP,
        ownership.CrashKind.COUNTER,
        ownership.CrashKind.UNCERTAINTY,
        ownership.CrashKind.RECONCILIATION,
    }
    for row in ownership.FIELD_OWNERSHIP:
        assert row.persisted == ownership.named_table_for(row.name)
        assert row.submachine in ownership.SUBMACHINES
        assert row.typed_runtime
        assert row.transition_owner
        assert row.compatibility_adapter
        assert row.production_callers
        assert row.crash_boundary
        if row.crash_kind in fence_kinds:
            assert "Not a crash-recovery fence" not in row.crash_boundary


def test_top_level_and_submachine_inventory_is_complete() -> None:
    assert ownership.TOP_LEVEL_STATES == {status.value for status in JobStatus}
    assert ownership.SUBMACHINES == {
        "identity",
        "prompt",
        "question",
        "terminal",
        "delivery",
        "workflow",
        "checkout",
        "ticket",
        "session",
        "worker",
        "import",
    }


def test_duplicate_authorities_include_prompt_and_checkout_overlaps() -> None:
    names = {item.name for item in ownership.DUPLICATE_AUTHORITIES}
    assert names == {
        "prompt-operation-vocabularies",
        "prompt-flat-runtime-reconstruction",
        "checkout-session-label-collapse",
    }
    job_states = {item.value for item in JobPromptState}
    question_states = {item.value for item in QuestionPromptState}
    assert job_states != question_states
    assert {"none", "submitting", "ambiguous"} <= job_states
    assert {"observed", "resolved"} <= question_states
    assert {"planned", "submitted"} <= job_states & question_states

    prompt = next(
        item
        for item in ownership.DUPLICATE_AUTHORITIES
        if item.name == "prompt-operation-vocabularies"
    )
    assert prompt.child_issue == 358
    checkout = next(
        item
        for item in ownership.DUPLICATE_AUTHORITIES
        if item.name == "checkout-session-label-collapse"
    )
    assert checkout.child_issue == 360


def test_child_sequence_stacks_because_files_overlap() -> None:
    sequence = {item.issue: item for item in ownership.CHILD_SEQUENCE}
    assert sequence[358].blocked_by == (357,)
    assert sequence[359].blocked_by == (357, 358)
    assert sequence[360].blocked_by == (357, 359)
    shared = set(sequence[359].overlapping_files) & set(sequence[360].overlapping_files)
    assert {
        "src/local_voice_harness/cursor/model.py",
        "src/local_voice_harness/cursor/provisioning.py",
        "src/local_voice_harness/cursor/recovery.py",
    } <= shared


def test_every_discovered_transition_entry_point_is_inventoried() -> None:
    discovered = ownership.discover_transition_entry_points()
    inventoried = ownership.TRANSITION_ENTRY_POINT_NAMES
    assert inventoried == discovered


@pytest.mark.parametrize("qualname", sorted(ownership.TRANSITION_ENTRY_POINT_NAMES))
def test_inventoried_transition_entry_points_exist(qualname: str) -> None:
    assert callable(ownership.resolve_qualname(qualname))


@pytest.mark.parametrize("qualname", sorted(ownership.COMPATIBILITY_ADAPTERS))
def test_compatibility_adapters_exist_and_are_not_transition_authorities(
    qualname: str,
) -> None:
    target = ownership.resolve_qualname(qualname)
    assert callable(target) or isinstance(target, property)
    assert qualname not in ownership.TRANSITION_ENTRY_POINT_NAMES


def test_baseline_counts_match_live_code() -> None:
    measured = ownership.measured_baseline_counts()
    assert measured == ownership.BASELINE_COUNTS
    assert measured["persisted_field_names"] == len(ownership.persisted_field_names())
    assert measured["cursor_job_public_properties"] == len(
        ownership.cursor_job_public_properties()
    )
    assert measured["transition_entry_points"] == len(
        ownership.TRANSITION_ENTRY_POINT_NAMES
    )


def test_lifecycle_module_sizes_are_recorded() -> None:
    counts = ownership.module_line_counts(str(REPO))
    assert counts.keys() == set(ownership.LIFECYCLE_MODULE_PATHS)
    assert all(count > 0 for count in counts.values())
    assert sum(counts.values()) == ownership.BASELINE_COUNTS["lifecycle_module_lines"]


def test_durable_storage_doc_records_ownership_baseline_and_sequence() -> None:
    document = (REPO / "docs/durable-storage-migration.md").read_text()
    assert "lifecycle_ownership.py" in document
    assert "#357" in document
    assert "#358" in document
    assert "#359" in document
    assert "#360" in document
    for key, value in ownership.BASELINE_COUNTS.items():
        assert str(value) in document, key
    assert "must follow #359" in document
