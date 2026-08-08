"""
Higher-level tests for process_file_group and run_series_review: the disc/
season-order alignment, the per-file window safety guard, and the season-wide
deferred-ranking grouping. Whisper/ffmpeg/network calls are all monkeypatched
out; everything else (scoring, alignment, renaming) runs for real.
"""
import argparse
from pathlib import Path

import identifier as I


def _write_srt(path: Path, text: str, span_minutes: float = 43.0) -> None:
    end_h, rem = divmod(int(span_minutes * 60), 3600)
    end_m, end_s = divmod(rem, 60)
    path.write_text(
        f"1\n00:00:01,000 --> 00:00:02,000\n{text}\n\n"
        f"2\n{end_h:02d}:{end_m:02d}:{end_s:02d},000 --> {end_h:02d}:{end_m:02d}:{end_s:02d},500\n(end)\n"
    )


def _group_args(**overrides):
    defaults = dict(
        full_scan=False, whisper_model="base", whisper_duration=20, whisper_skip=60,
        no_whisper=False, threshold=0.75, dry_run=True, alt_candidates=0, local_ai_url="",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_process_file_group_aligns_files_in_rip_order(tmp_path, monkeypatch):
    files = [tmp_path / f"disc_t0{i}.mkv" for i in range(3)]
    for f in files:
        f.touch()

    filler = "the and of to a in is that it was for on with " * 8
    transcripts = {
        files[0]: filler + "aaron betrayed charlie doorway engine forest goblin harbor igloo jungle",
        files[1]: filler + "aaron betrayed lighthouse marble nectar octopus pancake quartz",
        files[2]: filler + "rooster sunlight tornado umbrella volcano walnut xylophone yellow zebra",
    }
    monkeypatch.setattr(I, "get_transcript", lambda f, *a, **k: (transcripts[f], 44 * 60))

    ref_text = {
        "S01E04": "unrelated filler words alpha bravo charlie delta echo foxtrot",
        "S01E05": "aaron betrayed charlie doorway engine forest goblin harbor igloo jungle romeo sierra",
        "S01E06": "aaron betrayed lighthouse marble nectar octopus pancake quartz tango uniform victor",
        "S01E07": "rooster sunlight tornado umbrella volcano walnut xylophone yellow zebra whiskey",
        "S01E08": "totally different content nothing overlaps here at all zzz",
    }
    reference_subtitles = {}
    for key, text in ref_text.items():
        p = tmp_path / f"{key}.srt"
        _write_srt(p, text)
        reference_subtitles[key] = p

    candidate_episodes = [
        {"season": 1, "number": n, "name": f"Ep{n}", "runtime": 44} for n in range(4, 9)
    ]

    results = I.process_file_group(
        files=files, candidate_episodes=candidate_episodes, reference_subtitles=reference_subtitles,
        show_name="Test Show", args=_group_args(), ai_api_key=None, local_ai_model=None,
        os_api_key=None, cache_dir=tmp_path / "subcache", matched_this_run=set(),
        pending_renames=[], processing_paths=set(files), log_dir=None,
    )

    matched = {r["file"]: (r["matched"] or "").split(" - ")[0] for r in results}
    assert matched == {
        "disc_t00.mkv": "S01E05",
        "disc_t01.mkv": "S01E06",
        "disc_t02.mkv": "S01E07",
    }
    assert all(r["status"] == "renamed" for r in results)


def test_process_file_group_respects_per_file_window_even_under_perfect_lure(tmp_path, monkeypatch):
    files = [tmp_path / "disc_t00.mkv", tmp_path / "disc_t01.mkv"]
    for f in files:
        f.touch()

    filler = "the and of to a in is that it was for on with " * 8
    # file 0 is guessed as E05 (window locked to exactly E05) but its transcript
    # is a PERFECT copy of E04's reference -- the window must still refuse E04.
    transcripts = {
        files[0]: filler + "unrelated filler words alpha bravo charlie delta echo foxtrot",
        files[1]: filler + "rooster sunlight tornado umbrella volcano walnut xylophone yellow zebra",
    }
    monkeypatch.setattr(I, "get_transcript", lambda f, *a, **k: (transcripts[f], 44 * 60))

    ref_text = {
        "S01E04": "unrelated filler words alpha bravo charlie delta echo foxtrot",
        "S01E05": "aaron betrayed charlie doorway engine forest goblin harbor igloo jungle",
        "S01E06": "rooster sunlight tornado umbrella volcano walnut xylophone yellow zebra",
    }
    reference_subtitles = {}
    for key, text in ref_text.items():
        p = tmp_path / f"{key}.srt"
        _write_srt(p, text)
        reference_subtitles[key] = p

    candidate_episodes = [{"season": 1, "number": n, "name": f"Ep{n}", "runtime": 44} for n in (4, 5, 6)]

    results = I.process_file_group(
        files=files, candidate_episodes=candidate_episodes, reference_subtitles=reference_subtitles,
        show_name="Test Show", args=_group_args(), ai_api_key=None, local_ai_model=None,
        os_api_key=None, cache_dir=tmp_path / "subcache", matched_this_run=set(),
        pending_renames=[], processing_paths=set(files), log_dir=None,
        per_file_windows=[(5, 5), (6, 6)],
    )

    r0 = next(r for r in results if r["file"] == "disc_t00.mkv")
    matched_key = (r0["matched"] or "").split(" - ")[0]
    assert matched_key != "S01E04"  # never escapes its declared window, even scoring 0% there
    assert r0["status"] == "unmatched"


def test_run_series_review_makes_one_season_wide_call_and_skips_bogus_guess(tmp_path, monkeypatch):
    for name in ["Show.S01E01.mkv", "Show.S01E02.mkv", "Show.S01E05.mkv", "Show.S01E08.mkv", "Show.S01E20.mkv"]:
        (tmp_path / name).touch()

    all_episodes = [{"season": 1, "number": n, "name": f"Ep{n}"} for n in range(1, 9)]

    captured_calls = []

    def fake_process_file_group(**kwargs):
        captured_calls.append(kwargs)
        return [
            {"file": f.name, "status": "renamed", "matched": "S01E01", "confidence": 0.9}
            for f in kwargs["files"]
        ]

    monkeypatch.setattr(I, "process_file_group", fake_process_file_group)
    monkeypatch.setattr(
        I, "prefetch_reference_subtitles",
        lambda api_key, token, show_name, eps, cache_dir: {
            f"S{ep['season']:02d}E{ep['number']:02d}": Path("/dev/null") for ep in eps
        },
    )

    args = argparse.Namespace(
        no_confirm=True, dry_run=True, review_window=2, threshold=0.75,
        full_scan=False, whisper_model="base", whisper_duration=20, whisper_skip=60,
        no_whisper=False, alt_candidates=0, local_ai_url="",
    )

    I.run_series_review(
        args=args, scan_path=tmp_path, show_name="Test Show", all_episodes=all_episodes,
        api_key="k", token="t", cache_dir=tmp_path / "cache", ai_api_key=None,
        local_ai_model=None, log_dir=None,
    )

    assert len(captured_calls) == 1
    call = captured_calls[0]
    file_names = sorted(f.name for f in call["files"])
    assert "Show.S01E20.mkv" not in file_names  # no valid window -> filtered before the call
    assert len(file_names) == 4
    assert call["candidate_episodes"] == all_episodes  # whole season, not a per-disc slice


def test_print_summary_renders_top_candidates_section(capsys):
    results = [
        {"file": "S01E03.mkv", "status": "renamed", "matched": "S01E03 - Homecoming", "confidence": 0.91,
         "top_scores": [("S01E03", "Homecoming", 0.91), ("S01E04", "Departure", 0.12)]},
        {"file": "UNMATCHED_S01E04_42pct_x.mkv", "status": "unmatched",
         "matched": "S01E04 - Departure", "confidence": 0.10,
         "top_scores": [("S01E04", "Departure", 0.42), ("S01E05", "Arrival", 0.41)]},
        {"file": "skipped.mkv", "status": "skipped", "matched": None, "confidence": 0.0},
    ]
    I.print_summary(results, dry_run=True)
    out = capsys.readouterr().out
    assert "TOP CANDIDATES" in out
    assert "S01E03.mkv" in out
    assert "Homecoming" in out
    assert "Departure" in out
    assert "Arrival" in out
    assert "skipped.mkv" not in out.split("TOP CANDIDATES")[1]  # no scores -> not in the breakdown
