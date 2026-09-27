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
        no_whisper=False, whisper_url="", local_whisper=False,
        threshold=0.75, dry_run=True, alt_candidates=0, local_ai_url="",
        no_ocr=True, no_fill_unmatched=False, fill_min_score=I.FILL_MIN_SCORE,
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


def test_process_file_group_fixes_one_reversed_disc_among_normal_ones(tmp_path, monkeypatch):
    # Disc 1's titles count DOWN from its own last episode (a real MakeMKV rip
    # pattern -- see align_monotonic_bidirectional / _reorder_disc_blocks_for_
    # local_direction) while disc 2's count up normally. A season-wide
    # bidirectional flip alone can only pick one direction for the whole
    # group, so without the per-disc fix, disc 1's files get force-assigned
    # backwards despite each one's own text match being unambiguous.
    disc1 = [tmp_path / f"disc1_t0{i}.mkv" for i in range(3)]  # rip order -> true E03,E02,E01
    disc2 = [tmp_path / f"disc2_t0{i}.mkv" for i in range(3)]  # rip order -> true E04,E05,E06
    files = disc1 + disc2
    for f in files:
        f.touch()

    filler = "the and of to a in is that it was for on with " * 8
    words = {
        1: "aaron betrayed charlie doorway engine forest goblin harbor",
        2: "igloo jungle kettle lantern marble nectar octopus pancake",
        3: "quartz rooster sunlight tornado umbrella volcano walnut xray",
        4: "yellow zebra amber bronze copper diamond ebony flint",
        5: "granite hazel ivory jasper karma lilac mango nectarine",
        6: "opal pearl quartzite ruby sapphire topaz umber velvet",
    }
    transcripts = {
        disc1[0]: filler + words[3], disc1[1]: filler + words[2], disc1[2]: filler + words[1],
        disc2[0]: filler + words[4], disc2[1]: filler + words[5], disc2[2]: filler + words[6],
    }
    monkeypatch.setattr(I, "get_transcript", lambda f, *a, **k: (transcripts[f], 44 * 60))

    reference_subtitles = {}
    for n, w in words.items():
        key = f"S01E{n:02d}"
        p = tmp_path / f"{key}.srt"
        _write_srt(p, w)
        reference_subtitles[key] = p

    candidate_episodes = [{"season": 1, "number": n, "name": f"Ep{n}", "runtime": 44} for n in range(1, 7)]

    results = I.process_file_group(
        files=files, candidate_episodes=candidate_episodes, reference_subtitles=reference_subtitles,
        show_name="Test Show", args=_group_args(), ai_api_key=None, local_ai_model=None,
        os_api_key=None, cache_dir=tmp_path / "subcache", matched_this_run=set(),
        pending_renames=[], processing_paths=set(files), log_dir=None,
        file_discs=[1, 1, 1, 2, 2, 2],
    )

    matched = {r["file"]: (r["matched"] or "").split(" - ")[0] for r in results}
    assert matched == {
        "disc1_t00.mkv": "S01E03", "disc1_t01.mkv": "S01E02", "disc1_t02.mkv": "S01E01",
        "disc2_t00.mkv": "S01E04", "disc2_t01.mkv": "S01E05", "disc2_t02.mkv": "S01E06",
    }
    assert all(r["status"] == "renamed" for r in results)


def test_find_video_files_orders_by_original_name_not_previous_guess(tmp_path):
    # A --force rerun sees files a previous run renamed to UNMATCHED_SxxExx_YYpct_<original>.
    # Physical rip order must come from the original name (Disc 1, Disc 1_t01, Disc 2),
    # not from the previous guess baked into the prefix (E01 < E03 < E05 here would
    # otherwise put the Disc 1_t01 file first).
    names = [
        "UNMATCHED_S05E05_62pct_Show Disc 1.mkv",
        "UNMATCHED_S05E01_0pct_Show Disc 1_t01.mkv",
        "UNMATCHED_S05E03_50pct_Show Disc 2.mkv",
    ]
    for n in names:
        (tmp_path / n).touch()

    ordered = [f.name for f in I.find_video_files(tmp_path, force=True)]

    assert ordered == [names[0], names[1], names[2]]


def test_process_file_group_does_not_force_rows_without_a_disc_position(tmp_path, monkeypatch):
    # "S01E04.mkv" was confidently identified by an earlier run and no longer carries
    # a "Disc N" marker, so it has no recoverable physical position -- and it sorts
    # FIRST. Its text also weakly overlaps E01 (above the alignment floor), so if it
    # takes part in the disc-order alignment, the ascending chain E01,E02,E03 scores
    # higher than leaving it out, and it gets force-assigned E01 despite its own
    # text pointing unambiguously at E04. It must be matched on its own text instead.
    nodisc = tmp_path / "S01E04.mkv"
    disc = [tmp_path / "disc1_t00.mkv", tmp_path / "disc1_t01.mkv"]
    files = [nodisc] + disc
    for f in files:
        f.touch()

    filler = "the and of to a in is that it was for on with " * 8
    words = {
        1: "aaron betrayed charlie doorway engine forest goblin harbor",
        2: "igloo jungle kettle lantern marble nectar octopus pancake",
        3: "quartz rooster sunlight tornado umbrella volcano walnut xray",
        4: "yellow zebra amber bronze copper diamond ebony flint",
    }
    transcripts = {
        nodisc: filler + words[4] + " aaron betrayed",  # E04 text + a weak E01 overlap
        disc[0]: filler + words[2],
        disc[1]: filler + words[3],
    }
    monkeypatch.setattr(I, "get_transcript", lambda f, *a, **k: (transcripts[f], 44 * 60))

    reference_subtitles = {}
    for n, w in words.items():
        p = tmp_path / f"S01E{n:02d}.srt"
        _write_srt(p, w)
        reference_subtitles[f"S01E{n:02d}"] = p
    candidate_episodes = [{"season": 1, "number": n, "name": f"Ep{n}", "runtime": 44} for n in (1, 2, 3, 4)]

    def run(file_discs):
        return I.process_file_group(
            files=files, candidate_episodes=candidate_episodes, reference_subtitles=reference_subtitles,
            show_name="Test Show", args=_group_args(), ai_api_key=None, local_ai_model=None,
            os_api_key=None, cache_dir=tmp_path / "subcache", matched_this_run=set(),
            pending_renames=[], processing_paths=set(files), log_dir=None,
            file_discs=file_discs,
        )

    results = run([None, 1, 1])
    matched = {r["file"]: (r["matched"] or "").split(" - ")[0] for r in results}
    assert matched == {"S01E04.mkv": "S01E04", "disc1_t00.mkv": "S01E02", "disc1_t01.mkv": "S01E03"}
    assert all(r["status"] == "renamed" for r in results)


def test_process_file_group_places_a_textless_file_by_elimination(tmp_path, monkeypatch):
    # S01E02.mkv and S01E04.mkv were identified by an earlier run and carry no disc
    # marker; each decisively claims its own episode. The three disc files then line
    # up against exactly the three episodes left (E01, E03, E05) -- so d1_b, whose
    # transcript matches NOTHING (its episode's reference subtitle is bad), lands on
    # E03 by elimination rather than being wedged onto an already-identified episode.
    # (Its own text can't vouch for it, so it stays UNMATCHED -- with the right guess.)
    unplaced = [tmp_path / "S01E02.mkv", tmp_path / "S01E04.mkv"]
    disc = [tmp_path / "d1_a.mkv", tmp_path / "d1_b.mkv", tmp_path / "d1_c.mkv"]
    files = unplaced + disc
    for f in files:
        f.touch()

    common = "common1 common2 common3"
    own = {
        1: "aaron betrayed charlie doorway engine forest goblin harbor",
        2: "igloo jungle kettle lantern marble nectar octopus pancake",
        3: "quartz rooster sunlight tornado umbrella volcano walnut xray",
        4: "yellow zebra amber bronze copper diamond ebony flint",
        5: "granite hazel ivory jasper karma lilac mango nectarine",
    }
    ref = {n: f"{w} {common}" for n, w in own.items()}
    # No real match anywhere, but it weakly resembles E04 -- the already-claimed
    # episode (the alignment-level effect is pinned by test_align_group_rows_*).
    noise = f"{common} qqqa qqqb qqqc ebony flint"
    filler = "the and of to a in is that it was for on with " * 8  # stop words: clears the 100-word minimum
    transcripts = {
        unplaced[0]: filler + ref[2], unplaced[1]: filler + ref[4],
        disc[0]: filler + ref[1], disc[1]: filler + noise, disc[2]: filler + ref[5],
    }
    monkeypatch.setattr(I, "get_transcript", lambda f, *a, **k: (transcripts[f], 44 * 60))

    reference_subtitles = {}
    for n, text in ref.items():
        p = tmp_path / f"S01E{n:02d}.srt"
        _write_srt(p, text)
        reference_subtitles[f"S01E{n:02d}"] = p
    candidate_episodes = [{"season": 1, "number": n, "name": f"Ep{n}", "runtime": 44} for n in range(1, 6)]

    def run():
        return I.process_file_group(
            files=files, candidate_episodes=candidate_episodes, reference_subtitles=reference_subtitles,
            show_name="Test Show", args=_group_args(), ai_api_key=None, local_ai_model=None,
            os_api_key=None, cache_dir=tmp_path / "subcache", matched_this_run=set(),
            pending_renames=[], processing_paths=set(files), log_dir=None,
            file_discs=[None, None, 1, 1, 1],
        )

    results = {r["file"]: r for r in run()}
    matched = {name: (r["matched"] or "").split(" - ")[0] for name, r in results.items()}
    assert matched == {
        "S01E02.mkv": "S01E02", "S01E04.mkv": "S01E04",
        "d1_a.mkv": "S01E01", "d1_b.mkv": "S01E03", "d1_c.mkv": "S01E05",
    }
    assert results["d1_b.mkv"]["status"] == "unmatched"  # no text support -> not auto-renamed
    assert all(results[n]["status"] == "renamed" for n in ("S01E02.mkv", "S01E04.mkv", "d1_a.mkv", "d1_c.mkv"))


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
        no_ocr=True,
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


# ---------------------------------------------------------------------------
# Fill-in pass: below-threshold files map onto their best unclaimed episode
# ---------------------------------------------------------------------------
def _run_group_with_fixed_scores(tmp_path, monkeypatch, per_file_scores, **arg_overrides):
    """Run process_file_group for real (renames included, dry_run off) with text
    scoring pinned to `per_file_scores` -- one {episode_number: jaccard} dict per
    file, in file order -- so a test can dial in an exact best-vs-runner-up gap."""
    files = [tmp_path / f"disc_t{i:02d}.mkv" for i in range(len(per_file_scores))]
    for f in files:
        f.touch()
    filler = "the and of to a in is that it was for on with " * 12  # clears the 100-word minimum
    transcripts = {f: f"{filler} tag{i}" for i, f in enumerate(files)}
    by_transcript = {transcripts[f]: s for f, s in zip(files, per_file_scores)}
    monkeypatch.setattr(I, "get_transcript", lambda f, *a, **k: (transcripts[f], 44 * 60))

    def fake_score_episodes(transcript, episodes, reference_subtitles, video_seconds=0.0):
        table = by_transcript[transcript]
        scored = [(ep, table.get(ep["number"], 0.0)) for ep in episodes]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    monkeypatch.setattr(I, "_score_episodes", fake_score_episodes)
    numbers = sorted({n for s in per_file_scores for n in s})
    candidates = [{"season": 5, "number": n, "name": f"Ep{n}", "runtime": 44} for n in numbers]
    matched_this_run: set = set()
    results = I.process_file_group(
        files=files, candidate_episodes=candidates, reference_subtitles={},
        show_name="Test Show", args=_group_args(dry_run=False, **arg_overrides),
        ai_api_key=None, local_ai_model=None, os_api_key=None,
        cache_dir=tmp_path / "subcache", matched_this_run=matched_this_run,
        pending_renames=[], processing_paths=set(files), log_dir=None,
    )
    return results, matched_this_run


def test_fill_in_renames_a_high_jaccard_file_stuck_just_under_threshold(tmp_path, monkeypatch):
    # 94.6% raw text match, but a 52% runner-up holds ratio-confidence at ~71% --
    # under the 75% threshold. Used to be parked as UNMATCHED_S05E24_71pct_...
    results, claimed = _run_group_with_fixed_scores(
        tmp_path, monkeypatch, [{23: 0.52, 24: 0.946}]
    )
    (r,) = results
    assert 0.70 < r["confidence"] < 0.75  # really is below threshold on its own
    assert r["status"] == "filled"
    assert r["matched"].startswith("S05E24")
    assert (tmp_path / "S05E24.mkv").exists()
    assert not list(tmp_path.glob("UNMATCHED_*"))
    assert claimed == {"S05E24"}


def test_no_fill_unmatched_keeps_the_old_unmatched_behavior(tmp_path, monkeypatch):
    results, claimed = _run_group_with_fixed_scores(
        tmp_path, monkeypatch, [{23: 0.52, 24: 0.946}], no_fill_unmatched=True
    )
    assert results[0]["status"] == "unmatched"
    assert (tmp_path / "UNMATCHED_S05E24_71pct_disc_t00.mkv").exists()
    assert not claimed


def test_fill_in_skips_an_episode_a_confident_sibling_already_claimed(tmp_path, monkeypatch):
    # disc_t00 matches E05 outright. disc_t01's own best text match is ALSO E05
    # (0.80) but it can't have it; among what's left E06/E07 are a near-tie, so
    # it stays below threshold on its own (and the group's picks, E05 then E07,
    # aren't consecutive, so no group-confirmation floor rescues it either).
    # It gets its highest-scoring unclaimed episode, E07 -- never the taken E05.
    results, claimed = _run_group_with_fixed_scores(
        tmp_path, monkeypatch,
        [{5: 0.95, 6: 0.10, 7: 0.10}, {5: 0.80, 6: 0.44, 7: 0.45}],
    )
    by_file = {r["file"]: r for r in results}
    assert by_file["disc_t00.mkv"]["status"] == "renamed"
    assert by_file["disc_t00.mkv"]["matched"].startswith("S05E05")
    assert by_file["disc_t01.mkv"]["status"] == "filled"
    assert by_file["disc_t01.mkv"]["matched"].startswith("S05E07")
    assert (tmp_path / "S05E05.mkv").exists() and (tmp_path / "S05E07.mkv").exists()
    assert claimed == {"S05E05", "S05E07"}


def test_fill_in_leaves_a_file_with_no_text_support_unmatched(tmp_path, monkeypatch):
    results, claimed = _run_group_with_fixed_scores(
        tmp_path, monkeypatch, [{23: 0.20, 24: 0.22}]
    )
    assert results[0]["status"] == "unmatched"
    assert list(tmp_path.glob("UNMATCHED_*"))
    assert not claimed


def test_fill_min_score_lowers_the_bar(tmp_path, monkeypatch):
    results, _ = _run_group_with_fixed_scores(
        tmp_path, monkeypatch, [{23: 0.20, 24: 0.22}], fill_min_score=0.0
    )
    assert results[0]["status"] == "filled"
    assert results[0]["matched"].startswith("S05E24")
