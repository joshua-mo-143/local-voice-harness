from __future__ import annotations

from local_voice_harness.ticket_targets import extract_ticket_targets


def test_scoped_github_numbers_preserve_order_and_deduplicate() -> None:
    extraction = extract_ticket_targets(
        "Work on issues 12, 18, and 12",
        scope_source="github",
        scope="Example/Project",
    )

    assert extraction.requested_count == 3
    assert extraction.batch_requested
    assert [reference.canonical for reference in extraction.references] == [
        "Example/Project#12",
        "Example/Project#18",
    ]
    assert all(reference.scoped for reference in extraction.references)


def test_spoken_scoped_github_numbers_are_extracted() -> None:
    extraction = extract_ticket_targets(
        "Can you work on issues sixty six and sixty-seven?",
        scope_source="github",
        scope="Example/Project",
    )

    assert extraction.requested_count == 2
    assert [reference.canonical for reference in extraction.references] == [
        "Example/Project#66",
        "Example/Project#67",
    ]


def test_spoken_hundreds_distinguish_internal_and_list_conjunctions() -> None:
    extraction = extract_ticket_targets(
        "Work on issues one hundred and fifty, two hundred and seven, "
        "and one thousand two hundred and thirty-four",
        scope_source="github",
        scope="example/project",
    )

    assert [reference.canonical for reference in extraction.references] == [
        "example/project#150",
        "example/project#207",
        "example/project#1234",
    ]


def test_spoken_bare_batch_without_scope_is_rejected() -> None:
    extraction = extract_ticket_targets("Work on issues sixty six and sixty seven")

    assert extraction.batch_requested
    assert extraction.has_unresolved_scope


def test_scoped_linear_numbers_use_canonical_team_key() -> None:
    extraction = extract_ticket_targets(
        "Work on tickets #7 and 9",
        scope_source="linear",
        scope="eng",
    )

    assert [reference.canonical for reference in extraction.references] == [
        "ENG-7",
        "ENG-9",
    ]


def test_full_references_are_ordered_across_providers() -> None:
    extraction = extract_ticket_targets(
        "Work on ENG-4, https://github.com/example/project/issues/8, and other/repo#3"
    )

    assert [(item.source, item.canonical) for item in extraction.references] == [
        ("linear", "ENG-4"),
        ("github", "example/project#8"),
        ("github", "other/repo#3"),
    ]


def test_linear_like_substrings_inside_github_references_are_ignored() -> None:
    extraction = extract_ticket_targets(
        "Work on joshua-mo-143/local-voice-harness#999998 and "
        "https://github.com/example/eng-42/issues/7"
    )

    assert extraction.requested_count == 2
    assert [(item.source, item.canonical) for item in extraction.references] == [
        ("github", "joshua-mo-143/local-voice-harness#999998"),
        ("github", "example/eng-42#7"),
    ]


def test_standalone_linear_reference_survives_github_overlap_filtering() -> None:
    extraction = extract_ticket_targets(
        "Work on MO-143 and "
        "https://github.com/joshua-mo-143/local-voice-harness/issues/999998"
    )

    assert extraction.requested_count == 2
    assert [(item.source, item.canonical) for item in extraction.references] == [
        ("linear", "MO-143"),
        ("github", "joshua-mo-143/local-voice-harness#999998"),
    ]


def test_issue_in_repository_is_not_duplicated_as_bare_number() -> None:
    extraction = extract_ticket_targets(
        "Please handle issue 42 in example/project",
        scope_source="github",
        scope="other/project",
    )

    assert extraction.requested_count == 1
    assert extraction.references[0].canonical == "example/project#42"
    assert not extraction.references[0].scoped


def test_bare_batch_without_scope_is_rejected_deterministically() -> None:
    extraction = extract_ticket_targets("Work on issues 12 and 18")

    assert extraction.batch_requested
    assert extraction.has_unresolved_scope
    assert [reference.canonical for reference in extraction.references] == [None, None]
    assert all(
        reference.error == "bare issue number requires an unambiguous issues-page scope"
        for reference in extraction.references
    )


def test_scoped_batch_does_not_report_unresolved_scope() -> None:
    extraction = extract_ticket_targets(
        "Work on issues 12 and 18",
        scope_source="github",
        scope="example/project",
    )

    assert extraction.batch_requested
    assert not extraction.has_unresolved_scope


def test_nonpositive_scoped_number_is_rejected() -> None:
    extraction = extract_ticket_targets(
        "Work on issues 0 and 2",
        scope_source="github",
        scope="example/project",
    )

    assert extraction.references[0].canonical is None
    assert extraction.references[0].error == "issue number must be positive"
    assert extraction.references[1].canonical == "example/project#2"


def test_unrelated_numbers_are_not_targets() -> None:
    extraction = extract_ticket_targets(
        "Use 3 workers and finish by 2027",
        scope_source="github",
        scope="example/project",
    )

    assert extraction.requested_count == 0
    assert extraction.references == ()


def test_malformed_github_references_are_not_truncated() -> None:
    extraction = extract_ticket_targets(
        "Work on example/project#12abc and "
        "https://github.com/example/project/issues/34/files"
    )

    assert extraction.requested_count == 0
    assert extraction.references == ()


def test_single_scoped_number_is_identified_without_becoming_a_batch() -> None:
    extraction = extract_ticket_targets(
        "Work on issue 12",
        scope_source="github",
        scope="example/project",
    )

    assert not extraction.batch_requested
    assert extraction.requested_count == 1
    assert extraction.references[0].canonical == "example/project#12"
    assert extraction.references[0].scoped
