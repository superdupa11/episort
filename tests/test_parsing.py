"""Tests for directory/filename inference and CLI-adjacent parsing helpers."""
from pathlib import Path

import identifier as I


# ---------------------------------------------------------------------------
# infer_show_season
# ---------------------------------------------------------------------------
def test_infer_show_season_from_season_folder():
    show, season = I.infer_show_season(Path("/Shows/Breaking Bad/Season 1"), None, None)
    assert show == "Breaking Bad"
    assert season == 1


def test_infer_show_season_no_season_folder():
    show, season = I.infer_show_season(Path("/Shows/Breaking Bad"), None, None)
    assert show == "Breaking Bad"
    assert season is None


def test_infer_show_season_overrides_take_precedence():
    show, season = I.infer_show_season(Path("/Shows/X/Season 1"), "Override Name", 9)
    assert show == "Override Name"
    assert season == 9


# ---------------------------------------------------------------------------
# infer_file_season / infer_disc_number
# ---------------------------------------------------------------------------
def test_infer_file_season_from_subfolder():
    root = Path("/Shows/X")
    f = root / "Season 2" / "Disc 1" / "title01.mkv"
    assert I.infer_file_season(f, root, default_season=None) == 2


def test_infer_file_season_falls_back_to_default():
    root = Path("/Shows/X/Season 1")
    f = root / "title01.mkv"
    assert I.infer_file_season(f, root, default_season=1) == 1


def test_infer_disc_number_variants():
    root = Path("/Shows/X")
    assert I.infer_disc_number(root / "Disc 2" / "title01.mkv", root) == 2
    assert I.infer_disc_number(root / "Disk 3" / "title01.mkv", root) == 3
    assert I.infer_disc_number(root / "title_D4.mkv", root) == 4
    assert I.infer_disc_number(root / "title01.mkv", root) is None


def test_infer_disc_number_does_not_confuse_track_suffix():
    # "_t01" right after the disc number shouldn't break the disc-number match.
    root = Path("/Shows/X")
    assert I.infer_disc_number(root / "Disc 6_t02.mkv", root) == 6


def test_infer_part_number_variants():
    root = Path("/Shows/X")
    assert I.infer_part_number(root / "Show- Season 5, Part 2 - Disc 1.mkv", root) == 2
    assert I.infer_part_number(root / "Part 1" / "title01.mkv", root) == 1
    assert I.infer_part_number(root / "title01.mkv", root) is None


# ---------------------------------------------------------------------------
# narrow_by_disc
# ---------------------------------------------------------------------------
def _eps(numbers):
    return [{"season": 1, "number": n} for n in numbers]


def test_narrow_by_disc_even_split():
    eps = _eps(range(1, 11))  # 10 episodes, 2 discs -> 5 each
    narrowed = I.narrow_by_disc(eps, disc=1, total_discs=2, window=0)
    assert [e["number"] for e in narrowed] == [1, 2, 3, 4, 5]
    narrowed2 = I.narrow_by_disc(eps, disc=2, total_discs=2, window=0)
    assert [e["number"] for e in narrowed2] == [6, 7, 8, 9, 10]


def test_narrow_by_disc_window_padding():
    eps = _eps(range(1, 11))
    narrowed = I.narrow_by_disc(eps, disc=2, total_discs=2, window=2)
    assert [e["number"] for e in narrowed] == [4, 5, 6, 7, 8, 9, 10]


def test_narrow_by_disc_no_disc_or_single_disc_returns_unchanged():
    eps = _eps(range(1, 6))
    assert I.narrow_by_disc(eps, disc=None, total_discs=2, window=0) == eps
    assert I.narrow_by_disc(eps, disc=1, total_discs=1, window=0) == eps
    assert I.narrow_by_disc([], disc=1, total_discs=2, window=0) == []


def test_narrow_by_disc_falls_back_to_full_list_if_window_empties_it():
    eps = _eps([1, 2, 3])
    # A degenerate window computation should never return an empty result when
    # episodes exist; it falls back to the unfiltered list rather than
    # eliminating every candidate.
    narrowed = I.narrow_by_disc(eps, disc=1, total_discs=100, window=0)
    assert narrowed  # non-empty


# ---------------------------------------------------------------------------
# parse_season_selection / _format_season_ranges
# ---------------------------------------------------------------------------
def test_format_season_ranges():
    assert I._format_season_ranges([1, 2, 3, 5, 7, 8]) == "1-3, 5, 7-8"
    assert I._format_season_ranges([1]) == "1"
    assert I._format_season_ranges([]) == ""


def test_parse_season_selection_all_and_none():
    available = [1, 2, 3, 4]
    assert I.parse_season_selection("all", available) == [1, 2, 3, 4]
    assert I.parse_season_selection("", available) == []
    assert I.parse_season_selection("none", available) == []


def test_parse_season_selection_ranges_and_lists():
    available = [1, 2, 3, 4, 5]
    assert I.parse_season_selection("1-3", available) == [1, 2, 3]
    assert I.parse_season_selection("1,3-4", available) == [1, 3, 4]
    assert I.parse_season_selection("2, 2, 3", available) == [2, 3]


def test_parse_season_selection_ignores_out_of_range():
    available = [1, 2, 3]
    assert I.parse_season_selection("1,9", available) == [1]


# ---------------------------------------------------------------------------
# parse_series_files
# ---------------------------------------------------------------------------
def test_parse_series_files_groups_by_season_sorted_by_guess(tmp_path):
    for name in ["Show.S01E05.mkv", "Show.S01E01.mkv", "Show.S02E01.mkv", "no_pattern.mkv"]:
        (tmp_path / name).touch()

    result = I.parse_series_files(tmp_path)
    assert set(result.keys()) == {1, 2}
    season1_names = [p.name for p, _ in result[1]]
    assert season1_names == ["Show.S01E01.mkv", "Show.S01E05.mkv"]  # sorted by guessed episode
