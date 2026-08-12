from __future__ import annotations

from local_voice_harness.browser_context import RequestContext
from local_voice_harness.critical_targets import (
    READBACK_TIMEOUT_SECONDS,
    ReadbackReply,
    TargetSelection,
    new_candidate,
    parse_target,
    readback_response,
    resolve_readback,
    select_submit_target,
)
from local_voice_harness.ticket_targets import extract_ticket_targets


def test_single_ticket_submit_requires_concise_exact_readback() -> None:
    context = RequestContext("work on example/payments#42")
    selection = select_submit_target(
        extract_ticket_targets("work on example/payments#42"),
        context,
    )

    assert selection is not None
    candidate = new_candidate(selection, origin_turn="turn-1", now=10)
    response = readback_response(candidate)

    assert candidate.target.canonical == "example/payments#42"
    assert candidate.question.origin.turn_token == "turn-1"
    assert response.spoken_text == "Work on issue 42 in payments?"
    assert "example/payments#42" in response.display_text


def test_affirmation_is_bound_to_current_focused_identity() -> None:
    context = RequestContext(
        "work on this",
        focused_repository="example/payments",
        focused_issue="example/payments#42",
        issue_scope_source="github",
        issue_scope="example/payments",
    )
    selection = select_submit_target(extract_ticket_targets("work on this"), context)
    assert selection is not None
    candidate = new_candidate(selection, origin_turn="turn-1", now=10)

    accepted = resolve_readback(candidate, "yes", context, now=11)
    changed = resolve_readback(
        candidate,
        "yes",
        RequestContext(
            "yes",
            focused_repository="example/payments",
            focused_issue="example/payments#43",
            issue_scope_source="github",
            issue_scope="example/payments",
        ),
        now=11,
    )

    assert accepted.reply == ReadbackReply.AFFIRMATIVE
    assert changed.reply == ReadbackReply.EXPIRED


def test_number_correction_never_approves_original_target() -> None:
    context = RequestContext("work on example/payments#42")
    candidate = new_candidate(
        TargetSelection(parse_target("example/payments#42"), (None,) * 4),
        origin_turn="turn-1",
        now=10,
    )

    resolution = resolve_readback(candidate, "No, issue 43", context, now=11)

    assert resolution.reply == ReadbackReply.CORRECTION
    assert resolution.replacement is not None
    assert resolution.replacement.canonical == "example/payments#43"


def test_negative_and_stale_replies_fail_closed() -> None:
    context = RequestContext("work on example/payments#42")
    candidate = new_candidate(
        TargetSelection(parse_target("example/payments#42"), (None,) * 4),
        origin_turn="turn-1",
        now=10,
    )

    assert (
        resolve_readback(candidate, "no", context, now=11).reply
        == ReadbackReply.NEGATIVE
    )
    assert (
        resolve_readback(
            candidate,
            "yes",
            context,
            now=10 + READBACK_TIMEOUT_SECONDS + 0.1,
        ).reply
        == ReadbackReply.EXPIRED
    )


def test_explicit_batch_is_not_read_back() -> None:
    extraction = extract_ticket_targets(
        "work on issues 42 and 43",
        scope_source="github",
        scope="example/payments",
    )

    assert (
        select_submit_target(
            extraction,
            RequestContext(
                "work on issues 42 and 43",
                issue_scope_source="github",
                issue_scope="example/payments",
            ),
        )
        is None
    )


def test_unrelated_or_read_only_routing_has_no_target_policy_side_effect() -> None:
    assert (
        select_submit_target(
            extract_ticket_targets("what time is it"),
            RequestContext("what time is it"),
        )
        is None
    )
