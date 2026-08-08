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
# _alignment_is_conclusive
# ---------------------------------------------------------------------------
def _eps(*numbers):
    return [{"season": 5, "number": n} for n in numbers]


def test_alignment_is_conclusive_true_for_unbroken_run():
    candidates = _eps(9, 10, 11, 12, 13, 14)
    picks = [0, 1, 2, 3, 4, 5]
    assert I._alignment_is_conclusive(picks, candidates) is True


def test_alignment_is_conclusive_false_for_single_file():
    candidates = _eps(9)
    assert I._alignment_is_conclusive([0], candidates) is False


def test_alignment_is_conclusive_false_when_any_file_unassigned():
    candidates = _eps(9, 10, 11)
    picks = [0, None, 2]
    assert I._alignment_is_conclusive(picks, candidates) is False


def test_alignment_is_conclusive_false_when_episode_skipped():
    # Picks land on E09 then E11 -- a gap (E10 skipped), not one unbroken run.
    candidates = _eps(9, 10, 11)
    picks = [0, 2]
    assert I._alignment_is_conclusive(picks, candidates) is False


def test_alignment_is_conclusive_false_for_empty_picks():
    assert I._alignment_is_conclusive([], _eps(9)) is False


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
    # Mirrors the real Yellowstone S05 case: ~86% Jaccard vs. a ~23% runner-up
    # is a ~3.7x ratio -- comfortably below the 4x the plain ratio formula
    # needs for 75% confidence, even though the pick is unambiguous.
    scores = [
        ({"season": 5, "number": 9}, 0.869),
        ({"season": 5, "number": 7}, 0.230),
        ({"season": 5, "number": 12}, 0.234),
    ]
    forced = {"season": 5, "number": 9}
    ep, conf_plain, _ = I._confidence_for_forced_pick(scores, forced)
    assert conf_plain < I.GROUP_CONFIRMED_FLOOR
    ep, conf_boosted, raw = I._confidence_for_forced_pick(scores, forced, group_confirmed=True)
    assert ep == forced
    assert raw == 0.869
    assert conf_boosted == I.GROUP_CONFIRMED_FLOOR


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
