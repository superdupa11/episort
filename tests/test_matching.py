"""
Tests for the pure scoring/alignment logic: no network, ffmpeg, or Whisper
required. These are the functions most of this session's bugs were found in
(a negative-confidence clamp, a missing forced_ep branch) — this suite exists
so the next one is caught automatically instead of by manual reasoning.
"""
from pathlib import Path

import identifier as I


# ---------------------------------------------------------------------------
# align_monotonic
# ---------------------------------------------------------------------------
def test_align_monotonic_resolves_order_ambiguity():
    # File 2's transcript, scored in isolation, would look more like E05 than
    # its true match E06 (0.35 > 0.30) — a greedy/independent matcher would let
    # file 1 or file 2 collide on E05. The order constraint must resolve it.
    #        E04   E05   E06   E07   E08
    matrix = [
        [0.05, 0.55, 0.10, 0.05, 0.05],
        [0.05, 0.35, 0.30, 0.05, 0.05],
        [0.05, 0.05, 0.10, 0.50, 0.05],
    ]
    picks = I.align_monotonic(matrix)
    labels = ["E04", "E05", "E06", "E07", "E08"]
    assert [labels[p] for p in picks] == ["E05", "E06", "E07"]


def test_align_monotonic_leaves_subfloor_file_unassigned():
    # Row 1 never clears QUALITY_FLOOR (0.10) against anything — it should be
    # left as a gap rather than forced onto whatever's left over.
    matrix = [
        [0.60, 0.05, 0.05],
        [0.02, 0.03, 0.01],
        [0.05, 0.05, 0.55],
    ]
    picks = I.align_monotonic(matrix)
    assert picks[1] is None
    assert picks[0] == 0 and picks[2] == 2


def test_align_monotonic_single_file_reduces_to_argmax():
    matrix = [[0.10, 0.80, 0.20]]
    assert I.align_monotonic(matrix) == [1]


def test_align_monotonic_empty_inputs():
    assert I.align_monotonic([]) == []
    assert I.align_monotonic([[]]) == [None]


# ---------------------------------------------------------------------------
# align_monotonic_bidirectional
# ---------------------------------------------------------------------------
def test_align_monotonic_bidirectional_prefers_forward_on_normal_rip():
    #        E04   E05   E06
    matrix = [
        [0.60, 0.05, 0.05],
        [0.05, 0.55, 0.05],
        [0.05, 0.05, 0.50],
    ]
    assert I.align_monotonic_bidirectional(matrix) == [0, 1, 2]


def test_align_monotonic_bidirectional_detects_reversed_rip_order():
    # Files physically ripped in descending episode order (e.g. MakeMKV's
    # unsuffixed title being the disc's last episode, _t01.._tN counting
    # backwards) -- each file's own match is unambiguous, but a pure ascending
    # assumption would force every file onto the wrong episode.
    #        E04   E05   E06
    matrix = [
        [0.05, 0.05, 0.60],
        [0.05, 0.55, 0.05],
        [0.50, 0.05, 0.05],
    ]
    assert I.align_monotonic_bidirectional(matrix) == [2, 1, 0]


def test_align_monotonic_bidirectional_matches_plain_version_when_unambiguous_forward():
    # Same fixture as test_align_monotonic_resolves_order_ambiguity -- the
    # bidirectional wrapper shouldn't change behavior on an already-correct
    # forward rip just because it now also checks the reverse.
    matrix = [
        [0.05, 0.55, 0.10, 0.05, 0.05],
        [0.05, 0.35, 0.30, 0.05, 0.05],
        [0.05, 0.05, 0.10, 0.50, 0.05],
    ]
    assert I.align_monotonic_bidirectional(matrix) == I.align_monotonic(matrix)


# ---------------------------------------------------------------------------
# _reorder_disc_blocks_for_local_direction
# ---------------------------------------------------------------------------
def _pf(*names):
    return [Path(f"/x/{n}.mkv") for n in names]


def test_reorder_disc_blocks_fixes_one_reversed_disc_among_normal_ones():
    # Disc 1's titles count DOWN (E03,E02,E01) while disc 2's count up
    # normally (E04,E05,E06) -- a season-wide bidirectional flip can only
    # pick one direction for everything, so this needs the per-disc fix.
    #                E01   E02   E03   E04   E05   E06
    matrix = [
        [0.05, 0.05, 0.60, 0.05, 0.05, 0.05],  # disc 1, row 0 -> true E03
        [0.05, 0.60, 0.05, 0.05, 0.05, 0.05],  # disc 1, row 1 -> true E02
        [0.60, 0.05, 0.05, 0.05, 0.05, 0.05],  # disc 1, row 2 -> true E01
        [0.05, 0.05, 0.05, 0.60, 0.05, 0.05],  # disc 2, row 0 -> true E04
        [0.05, 0.05, 0.05, 0.05, 0.60, 0.05],  # disc 2, row 1 -> true E05
        [0.05, 0.05, 0.05, 0.05, 0.05, 0.60],  # disc 2, row 2 -> true E06
    ]
    files = _pf("d1_base", "d1_t01", "d1_t02", "d2_base", "d2_t01", "d2_t02")
    file_discs = [1, 1, 1, 2, 2, 2]
    transcripts = [(f"transcript {i}", 0.0) for i in range(6)]

    new_files, new_matrix, _, new_transcripts = I._reorder_disc_blocks_for_local_direction(
        files, matrix, None, transcripts, file_discs,
    )

    # Disc 1's block is physically reversed; disc 2's is left alone.
    assert [f.name for f in new_files] == [
        "d1_t02.mkv", "d1_t01.mkv", "d1_base.mkv", "d2_base.mkv", "d2_t01.mkv", "d2_t02.mkv",
    ]
    assert new_transcripts == [transcripts[2], transcripts[1], transcripts[0],
                                transcripts[3], transcripts[4], transcripts[5]]
    # After reordering, feeding this into the normal aligner resolves ascending.
    picks = I.align_monotonic_bidirectional(new_matrix)
    assert picks == [0, 1, 2, 3, 4, 5]


def test_reorder_disc_blocks_noop_without_disc_info():
    files = _pf("a", "b")
    matrix = [[0.6, 0.05], [0.05, 0.6]]
    transcripts = ["t0", "t1"]
    result = I._reorder_disc_blocks_for_local_direction(files, matrix, None, transcripts, None)
    assert result == (files, matrix, None, transcripts)


def test_reorder_disc_blocks_leaves_single_file_disc_alone():
    # A lone file on its own disc has nothing to reorder against.
    files = _pf("solo")
    matrix = [[0.05, 0.6, 0.05]]
    transcripts = ["t0"]
    new_files, new_matrix, _, new_transcripts = I._reorder_disc_blocks_for_local_direction(
        files, matrix, None, transcripts, [1],
    )
    assert new_files == files
    assert new_matrix == matrix
    assert new_transcripts == transcripts


def test_reorder_disc_blocks_keeps_per_file_windows_in_lockstep():
    matrix = [
        [0.05, 0.60],  # -> true E02
        [0.60, 0.05],  # -> true E01
    ]
    files = _pf("t01", "t02")
    windows = [(1, 2), (1, 2)]
    transcripts = ["a", "b"]
    new_files, new_matrix, new_windows, new_transcripts = I._reorder_disc_blocks_for_local_direction(
        files, matrix, windows, transcripts, [1, 1],
    )
    assert [f.name for f in new_files] == ["t02.mkv", "t01.mkv"]
    assert new_windows == [(1, 2), (1, 2)]  # identical windows here, but list stays aligned
    assert new_transcripts == ["b", "a"]


# ---------------------------------------------------------------------------
# _claim_unplaced_rows
# ---------------------------------------------------------------------------
def test_claim_unplaced_rows_claims_only_decisive_rows():
    #        E01   E02   E03
    matrix = [
        [0.95, 0.22, 0.20],  # row 0: unmistakably E01
        [0.55, 0.50, 0.20],  # row 1: weak and clustered -- no claim
        [0.20, 0.30, 0.90],  # row 2: placed (not in unplaced list) -- never considered
    ]
    assert I._claim_unplaced_rows(matrix, [0, 1]) == {0: 0}


def test_claim_unplaced_rows_drops_ambiguous_double_claims():
    # Two files both decisively claim E02 -- one of them is wrong, and there's
    # no telling which, so neither may pull that episode out of the pool.
    matrix = [
        [0.20, 0.95, 0.20],
        [0.20, 0.93, 0.25],
    ]
    assert I._claim_unplaced_rows(matrix, [0, 1]) == {}


# ---------------------------------------------------------------------------
# _align_group_rows
# ---------------------------------------------------------------------------
def _group_matrix():
    #            E01   E02   E03   E04   E05
    return [
        [0.20, 0.95, 0.20, 0.20, 0.20],  # row 0: no disc position, decisively E02
        [0.20, 0.20, 0.20, 0.95, 0.20],  # row 1: no disc position, decisively E04
        [0.95, 0.20, 0.20, 0.20, 0.20],  # row 2: disc file, E01
        [0.20, 0.20, 0.25, 0.35, 0.20],  # row 3: disc file, no real match; leans toward E04
        [0.20, 0.20, 0.20, 0.20, 0.95],  # row 4: disc file, E05
    ]


def test_align_group_rows_places_a_textless_file_by_elimination():
    # E02 and E04 are spoken for by rows 0/1, so the three disc files line up against
    # exactly E01, E03, E05 -- row 3 (whose text resembles the claimed E04) lands on
    # E03 by elimination instead of being wedged onto an already-identified episode.
    picks, confirmed, n_unplaced, n_claims = I._align_group_rows(_group_matrix(), [None, None, 1, 1, 1])
    assert picks == [None, None, 0, 2, 4]
    assert confirmed is True   # 3 rows, one unbroken run in the reduced candidate list
    assert (n_unplaced, n_claims) == (2, 2)


def test_align_group_rows_without_claims_the_same_file_takes_a_claimed_episode():
    # Documents what the claims prevent: aligned against ALL five episodes, row 3
    # would take E04 (its weak lean) -- the episode row 1 already identifies.
    picks, confirmed, _, _ = I._align_group_rows(_group_matrix()[2:], [1, 1, 1])
    assert picks[1] == 3
    assert confirmed is False  # E01, E04, E05: skips episodes -> not an unbroken run


def test_align_group_rows_with_no_disc_info_aligns_every_row():
    matrix = [[0.9, 0.2], [0.2, 0.9]]
    assert I._align_group_rows(matrix, None) == ([0, 1], True, 0, 0)
    assert I._align_group_rows(matrix, [None, None]) == ([0, 1], True, 0, 0)


# ---------------------------------------------------------------------------
# _alignment_is_conclusive
# ---------------------------------------------------------------------------
def _eps(*numbers):
    return [{"season": 5, "number": n} for n in numbers]


def test_alignment_is_conclusive_true_for_unbroken_run():
    picks = [0, 1, 2, 3, 4, 5]
    assert I._alignment_is_conclusive(picks) is True


def test_alignment_is_conclusive_true_for_unbroken_descending_run():
    # A reversed-rip-order group (see align_monotonic_bidirectional) lands on
    # a descending run of episode numbers -- still one unbroken run, so this
    # should count as conclusive just like the ascending case.
    picks = [5, 4, 3, 2, 1, 0]
    assert I._alignment_is_conclusive(picks) is True


def test_alignment_is_conclusive_false_for_single_file():
    assert I._alignment_is_conclusive([0]) is False


def test_alignment_is_conclusive_false_when_unassigned_row_leaves_a_gap():
    # Row 1 unassigned, but the assigned rows land on E09 then E11 -- a gap
    # (E10 skipped), not one unbroken run, so this stays inconclusive even
    # though the unassigned row itself is now tolerated on its own.
    picks = [0, None, 2]
    assert I._alignment_is_conclusive(picks) is False


def test_alignment_is_conclusive_false_when_episode_skipped():
    # Picks land on E09 then E11 -- a gap (E10 skipped), not one unbroken run.
    picks = [0, 2]
    assert I._alignment_is_conclusive(picks) is False


def test_alignment_is_conclusive_false_for_empty_picks():
    assert I._alignment_is_conclusive([]) is False


def test_alignment_is_conclusive_true_despite_unassigned_duplicate_titles():
    # 5 files ripped for 4 episodes (a duplicate/bonus title in the middle
    # that lost its column to a stronger row) -- the real-world MakeMKV
    # situation this tolerance exists for. The assigned majority still forms
    # an unbroken run, so this should now read as confirmed.
    picks = [0, None, 1, 2, 3]
    assert I._alignment_is_conclusive(picks) is True


def test_alignment_is_conclusive_false_when_assigned_rows_are_not_a_majority():
    # Only 2 of 10 rows assigned, even though those two happen to be
    # consecutive -- too little of the group actually lined up to count as
    # independent corroboration.
    picks = [None, None, None, None, None, None, None, None, 0, 1]
    assert I._alignment_is_conclusive(picks) is False


# ---------------------------------------------------------------------------
# _confidence_from_scores / _confidence_for_forced_pick
# ---------------------------------------------------------------------------
def test_confidence_from_scores_empty():
    assert I._confidence_from_scores([]) == (None, 0.0, 0.0)


def test_confidence_from_scores_below_quality_floor():
    ep = {"season": 1, "number": 1}
    best_ep, conf, raw = I._confidence_from_scores([(ep, 0.05)])
    assert best_ep is ep
    assert conf == 0.0
    assert raw == 0.05


def test_confidence_from_scores_ratio_scales_with_margin():
    ep1, ep2 = {"season": 1, "number": 1}, {"season": 1, "number": 2}
    # 5x better -> full confidence
    _, conf_high, _ = I._confidence_from_scores([(ep1, 0.50), (ep2, 0.10)])
    assert conf_high == 1.0
    # nearly tied -> near-zero confidence
    _, conf_low, _ = I._confidence_from_scores([(ep1, 0.31), (ep2, 0.30)])
    assert conf_low < 0.05


def test_confidence_for_forced_pick_below_competitor_clamps_to_zero_not_negative():
    # Forced pick scores LOWER than a competitor -- the naive ratio formula
    # would go negative here; it must be clamped to 0, not left negative.
    scores = [
        ({"season": 1, "number": 5}, 0.35),
        ({"season": 1, "number": 6}, 0.30),
    ]
    forced = {"season": 1, "number": 6}
    ep, conf, raw = I._confidence_for_forced_pick(scores, forced)
    assert raw == 0.30
    assert conf == 0.0


def test_confidence_for_forced_pick_when_also_the_best():
    scores = [
        ({"season": 1, "number": 6}, 0.50),
        ({"season": 1, "number": 5}, 0.10),
    ]
    forced = {"season": 1, "number": 6}
    ep, conf, raw = I._confidence_for_forced_pick(scores, forced)
    assert raw == 0.50
    assert conf == 1.0


def test_confidence_for_forced_pick_falls_back_when_forced_ep_missing():
    scores = [
        ({"season": 1, "number": 5}, 0.35),
        ({"season": 1, "number": 6}, 0.30),
    ]
    # Not present in scores at all (e.g. claimed by an earlier group) -> falls
    # back to plain _confidence_from_scores behavior (top of the sorted list).
    ep, conf, raw = I._confidence_for_forced_pick(scores, {"season": 1, "number": 9})
    assert ep == {"season": 1, "number": 5}
    assert raw == 0.35


def test_confidence_for_forced_pick_empty_scores():
    assert I._confidence_for_forced_pick([], {"season": 1, "number": 1}) == (None, 0.0, 0.0)


def test_confidence_for_forced_pick_group_confirmed_boosts_mediocre_ratio():
    # A middling-but-real match (0.50 vs a 0.30 runner-up: 1.7x, and below the
    # decisive-match floor so the margin rule doesn't apply): text alone stays
    # well short of confident, but the whole disc lining up end-to-end is
    # independent evidence that should lift it.
    scores = [
        ({"season": 5, "number": 9}, 0.50),
        ({"season": 5, "number": 7}, 0.30),
        ({"season": 5, "number": 12}, 0.28),
    ]
    forced = {"season": 5, "number": 9}
    ep, conf_plain, _ = I._confidence_for_forced_pick(scores, forced)
    assert conf_plain < I.GROUP_CONFIRMED_FLOOR
    ep, conf_boosted, raw = I._confidence_for_forced_pick(scores, forced, group_confirmed=True)
    assert ep == forced
    assert raw == 0.50
    assert conf_boosted == I.GROUP_CONFIRMED_FLOOR


def test_decisive_margin_is_confident_without_group_confirmation():
    # Mirrors real OCR results: ~0.87-0.97 against a ~0.23 noise floor is only
    # a ~3.7-4.2x ratio, which the plain ratio formula caps at ~70% -- below the
    # 75% threshold -- even though nothing else comes close. The absolute gap
    # (0.63) makes it unmistakable.
    scores = [
        ({"season": 5, "number": 9}, 0.869),
        ({"season": 5, "number": 7}, 0.230),
        ({"season": 5, "number": 12}, 0.234),
    ]
    ep, conf, raw = I._confidence_from_scores(scores)
    assert ep["number"] == 9 and raw == 0.869
    assert conf >= 0.75
    forced = {"season": 5, "number": 9}
    assert I._confidence_for_forced_pick(scores, forced)[1] >= 0.75


def test_margin_confidence_never_lifts_a_clustered_or_weak_pick():
    # Two candidates that score alike (e.g. two reference uploads that are the
    # same text) have no gap; a weak winner has nothing to be decisive about.
    assert I._margin_confidence(0.87, 0.85) < 0.1
    assert I._margin_confidence(0.55, 0.05) == 0.0  # below DECISIVE_MATCH_FLOOR
    assert I._margin_confidence(0.95, 0.25) == 1.0


def test_confidence_for_forced_pick_group_confirmed_rescues_an_exact_tie():
    # OpenSubtitles sometimes serves another episode's dialogue as an episode's
    # reference (seen: E07's upload was E12's text), so the true episode and the
    # imposter tie EXACTLY. A tie isn't the text contradicting the alignment, so
    # group confirmation must be able to rescue it -- unlike the ratio<1 case.
    scores = [
        ({"season": 5, "number": 7}, 0.869),   # imposter reference, ties the true one
        ({"season": 5, "number": 12}, 0.869),  # the order-consistent pick
        ({"season": 5, "number": 3}, 0.22),
    ]
    forced = {"season": 5, "number": 12}
    assert I._confidence_for_forced_pick(scores, forced)[1] == 0.0
    assert I._confidence_for_forced_pick(scores, forced, group_confirmed=True)[1] == I.GROUP_CONFIRMED_FLOOR


def test_confidence_for_forced_pick_group_confirmed_does_not_rescue_contradicted_pick():
    # The forced pick scores LOWER than a competitor -- group corroboration
    # must not override that; text actively disagrees, so it stays at 0.
    scores = [
        ({"season": 1, "number": 5}, 0.60),
        ({"season": 1, "number": 6}, 0.50),
    ]
    forced = {"season": 1, "number": 6}
    ep, conf, raw = I._confidence_for_forced_pick(scores, forced, group_confirmed=True)
    assert conf == 0.0


def test_confidence_for_forced_pick_group_confirmed_does_not_rescue_thin_text_support():
    # Forced pick "wins" against its only competitor but never clears
    # JACCARD_AI_SKIP_FLOOR (0.40) -- too little real text support for the
    # group's order agreement to paper over on its own.
    scores = [
        ({"season": 1, "number": 6}, 0.15),
        ({"season": 1, "number": 5}, 0.05),
    ]
    forced = {"season": 1, "number": 6}
    ep, conf, raw = I._confidence_for_forced_pick(scores, forced, group_confirmed=True)
    assert raw == 0.15
    assert conf < I.GROUP_CONFIRMED_FLOOR


def test_confidence_for_forced_pick_group_confirmed_never_lowers_an_already_high_ratio():
    scores = [
        ({"season": 1, "number": 6}, 0.50),
        ({"season": 1, "number": 5}, 0.05),
    ]
    forced = {"season": 1, "number": 6}
    ep, conf, raw = I._confidence_for_forced_pick(scores, forced, group_confirmed=True)
    assert conf == 1.0


# ---------------------------------------------------------------------------
# match_score / _content_words / _fuzzy_bonus_matches
# ---------------------------------------------------------------------------
def test_content_words_strips_stopwords_and_punctuation():
    words = I._content_words("The Quick, Brown Fox! It was fast.")
    assert "quick" in words and "brown" in words and "fox" in words and "fast" in words
    assert "the" not in words and "it" not in words and "was" not in words


def test_match_score_identical_text_is_one():
    text = "aaron betrayed charlie in the doorway of the old lighthouse"
    assert I.match_score(text, text) == 1.0


def test_match_score_disjoint_text_is_zero():
    assert I.match_score("aaron betrayed charlie", "xander destroyed yolanda") == 0.0


def test_match_score_no_content_words_is_zero():
    # Union of content words is empty on both sides -> defined as 0, not a
    # division-by-zero crash.
    assert I.match_score("the a is", "it was to") == 0.0


def test_fuzzy_bonus_matches_credits_near_typos_once_each():
    # "recieve"/"receive" is a classic OCR/typo pair; each word should be
    # credited at most once (greedy highest-similarity-first pairing).
    bonus = I._fuzzy_bonus_matches(["recieve", "aaron"], ["receive", "aaron2"])
    assert bonus == 1  # only the recieve/receive pair clears FUZZY_SCORE_CUTOFF


def test_fuzzy_bonus_matches_empty_inputs():
    assert I._fuzzy_bonus_matches([], ["anything"]) == 0
    assert I._fuzzy_bonus_matches(["anything"], []) == 0


# ---------------------------------------------------------------------------
# _duration_bonus
# ---------------------------------------------------------------------------
def test_duration_bonus_tiers():
    assert I._duration_bonus(44 * 60, 44) == I.DURATION_BONUS_MAX
    assert I._duration_bonus(44 * 60 + 6 * 60, 44) == I.DURATION_BONUS_MAX * 0.5
    assert I._duration_bonus(44 * 60 + 20 * 60, 44) == 0.0


def test_duration_bonus_missing_inputs():
    assert I._duration_bonus(0, 44) == 0.0
    assert I._duration_bonus(44 * 60, None) == 0.0


# ---------------------------------------------------------------------------
# _score_episodes: plausibility check must apply here (single source of truth
# for real scoring — process_file_group's alignment matrix reuses this same
# function specifically so it can never drift out of sync with it).
# ---------------------------------------------------------------------------
def _write_srt(path: Path, text: str, span_minutes: float) -> None:
    end_h, rem = divmod(int(span_minutes * 60), 3600)
    end_m, end_s = divmod(rem, 60)
    path.write_text(
        f"1\n00:00:01,000 --> 00:00:02,000\n{text}\n\n"
        f"2\n{end_h:02d}:{end_m:02d}:{end_s:02d},000 --> {end_h:02d}:{end_m:02d}:{end_s:02d},500\n(end)\n"
    )


def test_score_episodes_excludes_implausible_reference(tmp_path):
    good = tmp_path / "good.srt"
    _write_srt(good, "aaron betrayed charlie in the doorway", span_minutes=43)
    bad = tmp_path / "bad.srt"
    _write_srt(bad, "aaron betrayed charlie in the doorway", span_minutes=1.5)  # sync-test-length

    episodes = [
        {"season": 1, "number": 1, "runtime": 44},
        {"season": 1, "number": 2, "runtime": 44},
    ]
    reference_subtitles = {"S01E01": good, "S01E02": bad}
    scores = I._score_episodes(
        "aaron betrayed charlie in the doorway", episodes, reference_subtitles, video_seconds=0.0
    )
    keys_scored = {(ep["season"], ep["number"]) for ep, _ in scores}
    assert (1, 1) in keys_scored
    assert (1, 2) not in keys_scored  # excluded as implausible, not scored at all


def test_score_episodes_keeps_plausible_reference_when_no_runtime_on_record(tmp_path):
    srt = tmp_path / "x.srt"
    _write_srt(srt, "aaron betrayed charlie", span_minutes=1.5)
    episodes = [{"season": 1, "number": 1, "runtime": None}]
    scores = I._score_episodes("aaron betrayed charlie", episodes, {"S01E01": srt})
    assert len(scores) == 1  # no runtime on record -> plausibility check can't judge, assumes True


# ---------------------------------------------------------------------------
# refine_top_candidates_with_alternates: forced_ep must be honored on BOTH the
# early-return path and the post-refinement-loop return path. A prior version
# of this function only handled the early-return branch, silently discarding
# the disc-order alignment's pick whenever alt-candidate refinement actually ran.
# ---------------------------------------------------------------------------
def test_refine_honors_forced_ep_after_refinement_loop_runs(monkeypatch):
    monkeypatch.setattr(I, "get_alternate_reference_subtitles", lambda *a, **k: [])

    scores = [
        ({"season": 1, "number": 5, "name": "Ep5"}, 0.50),  # raw top
        ({"season": 1, "number": 6, "name": "Ep6"}, 0.30),  # forced (order-consistent) pick
    ]
    forced_ep = {"season": 1, "number": 6, "name": "Ep6"}

    alt_ep, alt_conf, alt_raw, refined = I.refine_top_candidates_with_alternates(
        transcript="irrelevant, no alternates available",
        scores=scores,
        show_name="Test",
        api_key="k",
        token="t",
        cache_dir=Path("/tmp"),
        max_alternates=2,
        forced_ep=forced_ep,
    )
    assert alt_ep == forced_ep
    assert alt_raw == 0.30  # must NOT silently revert to the raw top-of-list score (0.50)
    assert refined == scores  # nothing to refine (no alternates found) -> scores unchanged


def test_refine_early_return_when_max_alternates_zero():
    scores = [({"season": 1, "number": 1}, 0.5)]
    ep, conf, raw, returned_scores = I.refine_top_candidates_with_alternates(
        transcript="x", scores=scores, show_name="Test", api_key="k", token="t",
        cache_dir=Path("/tmp"), max_alternates=0,
    )
    assert returned_scores == scores
    assert raw == 0.5


# ---------------------------------------------------------------------------
# match_with_local_ai
# ---------------------------------------------------------------------------
class _FakeResponse:
    """Minimal requests.Response stand-in for match_with_local_ai tests."""

    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.text = str(body)

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.exceptions.HTTPError(
                f"{self.status_code} Client Error: Not Found for url: X", response=self
            )

    def json(self):
        return self._body


def test_match_with_local_ai_surfaces_error_body_on_http_error(monkeypatch, caplog):
    # A wrong/unpulled model name (e.g. "qwen2.5:14b" instead of the actually-pulled
    # "qwen2.5-coder:14b") 404s at the HTTP layer. requests' default exception message
    # alone ("404 Client Error: Not Found for url: ...") reads exactly like a proxy
    # routing failure -- the real cause is in the JSON error body, which must make it
    # into the log instead of being swallowed.
    monkeypatch.setattr(
        I.requests, "post",
        lambda url, json, timeout: _FakeResponse(
            404, {"error": {"message": "model 'qwen2.5:14b' not found", "type": "not_found_error"}}
        ),
    )
    with caplog.at_level("ERROR"):
        ep, conf = I.match_with_local_ai(
            "transcript", [{"season": 5, "number": 1, "name": "Ep"}], "Show",
            "qwen2.5:14b", "https://ollama.example.net",
        )
    assert ep is None and conf == 0.0
    assert "model 'qwen2.5:14b' not found" in caplog.text


def test_match_with_local_ai_falls_back_to_raw_text_on_non_json_error_body(monkeypatch, caplog):
    # A misconfigured reverse proxy can 404 with an HTML error page rather than a
    # JSON body -- the fallback must not crash and should still surface something.
    monkeypatch.setattr(
        I.requests, "post",
        lambda url, json, timeout: _FakeResponse(404, "<html>not found</html>"),
    )
    with caplog.at_level("ERROR"):
        ep, conf = I.match_with_local_ai(
            "transcript", [{"season": 5, "number": 1, "name": "Ep"}], "Show",
            "some-model", "https://ollama.example.net",
        )
    assert ep is None and conf == 0.0
    assert "local-ai matching failed" in caplog.text


# ---------------------------------------------------------------------------
# _plan_fill_ins
# ---------------------------------------------------------------------------
def _eps(*numbers, season=5):
    return {n: {"season": season, "number": n, "name": f"Ep{n}"} for n in numbers}


def _ranked(eps, scores):
    """[(episode, score)] sorted best-first, from {episode_number: score}."""
    return sorted(((eps[n], s) for n, s in scores.items()), key=lambda x: x[1], reverse=True)


def test_plan_fill_ins_takes_next_highest_unclaimed_episode():
    eps = _eps(23, 24, 25)
    ranked = _ranked(eps, {24: 0.95, 23: 0.60, 25: 0.15})
    plan = I._plan_fill_ins([(0, ranked, None)], claimed={"S05E24"}, min_score=0.40)
    assert plan[0][0]["number"] == 23  # E24 is spoken for -> next highest, not E25


def test_plan_fill_ins_leaves_file_unmatched_when_nothing_clears_the_floor():
    eps = _eps(23, 24)
    ranked = _ranked(eps, {24: 0.95, 23: 0.30})
    assert I._plan_fill_ins([(0, ranked, None)], {"S05E24"}, min_score=0.40) == {}


def test_plan_fill_ins_stronger_text_match_keeps_a_contested_episode():
    eps = _eps(23, 24)
    weak = _ranked(eps, {24: 0.72, 23: 0.55})
    strong = _ranked(eps, {24: 0.95, 23: 0.10})
    # The weaker file is listed FIRST -- processing order must not decide it.
    plan = I._plan_fill_ins([(0, weak, None), (1, strong, None)], set(), min_score=0.40)
    assert plan[1][0]["number"] == 24
    assert plan[0][0]["number"] == 23


def test_plan_fill_ins_never_assigns_one_episode_twice():
    eps = _eps(24)
    a, b = _ranked(eps, {24: 0.90}), _ranked(eps, {24: 0.80})
    plan = I._plan_fill_ins([(0, a, None), (1, b, None)], set(), min_score=0.40)
    assert list(plan) == [0]


def test_plan_fill_ins_honors_each_files_window():
    eps = _eps(10, 24)
    ranked = _ranked(eps, {10: 0.99, 24: 0.50})
    plan = I._plan_fill_ins([(0, ranked, (20, 30))], set(), min_score=0.40)
    assert plan[0][0]["number"] == 24  # E10 scores higher but is outside this file's disc


def test_plan_fill_ins_min_score_zero_takes_any_unclaimed_episode():
    eps = _eps(1, 2)
    ranked = _ranked(eps, {1: 0.05, 2: 0.02})
    assert I._plan_fill_ins([(0, ranked, None)], set(), min_score=0.0)[0][0]["number"] == 1
