"""Tests for filename sanitization and the confident/UNMATCHED rename targets."""
from pathlib import Path

import identifier as I


def test_sanitize_name_strips_unsafe_chars_and_trailing_dots():
    assert I.sanitize_name('Foo: Bar/Baz*?"<>|') == "Foo BarBaz"
    assert I.sanitize_name("trailing dot. ") == "trailing dot"


def test_build_confident_path():
    src = Path("/Shows/X/title01.mkv")
    ep = {"season": 1, "number": 5}
    assert I.build_confident_path(src, ep) == Path("/Shows/X/S01E05.mkv")


def test_build_unmatched_path_with_guess():
    src = Path("/Shows/X/title01.mkv")
    ep = {"season": 1, "number": 4}
    target = I.build_unmatched_path(src, ep, confidence=0.42)
    assert target.name == "UNMATCHED_S01E04_42pct_title01.mkv"


def test_build_unmatched_path_no_guess_at_all():
    src = Path("/Shows/X/title01.mkv")
    target = I.build_unmatched_path(src, None, confidence=0.0)
    assert target.name == "UNMATCHED_UNKNOWN_title01.mkv"


def test_strip_unmatched_prefix_prevents_nesting_on_rerun():
    stem = "UNMATCHED_S01E04_42pct_title01"
    assert I._strip_unmatched_prefix(stem) == "title01"


def test_build_unmatched_path_rerun_does_not_nest_prefixes():
    # Re-running on an already-UNMATCHED file with a new (different) guess
    # should replace the old guess, not stack another UNMATCHED_ on top.
    src = Path("/Shows/X/UNMATCHED_S01E04_42pct_title01.mkv")
    ep = {"season": 1, "number": 5}
    target = I.build_unmatched_path(src, ep, confidence=0.60)
    assert target.name == "UNMATCHED_S01E05_60pct_title01.mkv"
