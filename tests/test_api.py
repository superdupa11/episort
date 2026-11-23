"""
The library API (episort.api): identify() decides without touching files,
apply_renames() renames after review, results come back as EpisodeMatch and
JSON. Network, ffmpeg and Whisper are monkeypatched out, as in
test_pipeline_integration.py; scoring, alignment and the fill pass run for real.
"""
import json
from pathlib import Path

import pytest

import identifier as I
from episort import api
from episort.api import EpisodeMatch, IdentifyConfig, Show

FILLER = "the and of to a in is that it was for on with " * 8


def _write_srt(path: Path, text: str, span_minutes: float = 43.0) -> None:
    end_h, rem = divmod(int(span_minutes * 60), 3600)
    end_m, end_s = divmod(rem, 60)
    path.write_text(
        f"1\n00:00:01,000 --> 00:00:02,000\n{text}\n\n"
        f"2\n{end_h:02d}:{end_m:02d}:{end_s:02d},000 --> {end_h:02d}:{end_m:02d}:{end_s:02d},500\n(end)\n"
    )


@pytest.fixture
def season(tmp_path, monkeypatch):
    """A season folder with three rips whose text matches episodes 5-7, a fake
    OpenSubtitles (login + reference subtitles) and a fake transcript source."""
    folder = tmp_path / "Test Show" / "Season 1"
    folder.mkdir(parents=True)
    files = [folder / f"title_t0{i}.mkv" for i in range(3)]
    for f in files:
        f.write_bytes(b"x")
    words = {
        files[0]: "aaron betrayed charlie doorway engine forest goblin harbor igloo jungle",
        files[1]: "aaron betrayed lighthouse marble nectar octopus pancake quartz",
        files[2]: "rooster sunlight tornado umbrella volcano walnut xylophone yellow zebra",
    }
    transcripts = {f: FILLER + w for f, w in words.items()}

    def fake_get_transcript(mkv_path, *a, **k):
        I._TRANSCRIPT_SOURCES[str(mkv_path)] = "subtitles (stream 3, subrip)"
        return transcripts[mkv_path], 44 * 60

    monkeypatch.setattr(I, "get_transcript", fake_get_transcript)
    monkeypatch.setattr(I, "os_login", lambda *a, **k: "token")
    monkeypatch.setattr(I, "get_alternate_reference_subtitles", lambda *a, **k: [])

    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    ref_text = {
        "S01E04": "unrelated filler words alpha bravo charlie delta echo foxtrot",
        "S01E05": "aaron betrayed charlie doorway engine forest goblin harbor igloo jungle romeo sierra",
        "S01E06": "aaron betrayed lighthouse marble nectar octopus pancake quartz tango uniform victor",
        "S01E07": "rooster sunlight tornado umbrella volcano walnut xylophone yellow zebra whiskey",
        "S01E08": "totally different content nothing overlaps here at all zzz",
    }
    refs = {}
    for key, text in ref_text.items():
        p = refs_dir / f"{key}.srt"
        _write_srt(p, text)
        refs[key] = p
    monkeypatch.setattr(I, "prefetch_reference_subtitles", lambda *a, **k: refs)

    show = Show(name="Test Show", tvmaze_id=1, episodes=[
        {"season": 1, "number": n, "name": f"Ep{n}", "runtime": 44, "summary": ""} for n in range(4, 9)
    ])
    config = IdentifyConfig(opensubtitles_api_key="k", opensubtitles_username="u",
                            opensubtitles_password="p", ai="none", ocr=False,
                            alt_candidates=0, cache_dir=tmp_path / "subcache")
    return folder, files, show, config


def test_identify_matches_in_rip_order_and_renames_nothing(season):
    folder, files, show, config = season
    before = sorted(p.name for p in folder.iterdir())

    matches = api.identify(folder, show=show, config=config)

    assert [m.path.name for m in matches] == [f.name for f in files]
    assert [m.key for m in matches] == ["S01E05", "S01E06", "S01E07"]
    assert all(m.method == "direct" for m in matches)
    assert [m.title for m in matches] == ["Ep5", "Ep6", "Ep7"]
    assert all(m.confidence >= 0.75 for m in matches)
    assert sorted(p.name for p in folder.iterdir()) == before  # nothing renamed, nothing added
    ev = matches[0].evidence
    assert ev["status"] == "matched"
    assert ev["transcript_source"] == "subtitles (stream 3, subrip)"
    assert ev["top_scores"][0]["key"] == "S01E05"
    assert ev["group_confirmed"] is True


def test_identify_ignores_existing_names_on_disk(season):
    """A file already called S01E05.mkv in the folder isn't a rename conflict
    (identify never renames); with exclude_named=False its episode stays a candidate."""
    folder, files, show, config = season
    (folder / "S01E05.mkv").write_bytes(b"x")

    matches = api.identify(folder, show=show, config=config, files=files, exclude_named=False)

    assert [m.key for m in matches] == ["S01E05", "S01E06", "S01E07"]
    assert all(m.evidence["status"] == "matched" for m in matches)


def test_identify_leaves_named_episodes_out_by_default(season):
    folder, files, show, config = season
    (folder / "S01E05.mkv").write_bytes(b"x")

    matches = api.identify(folder, show=show, config=config, files=files)

    # E05 is taken by the existing file, so the first rip can't be it
    assert "S01E05" not in [m.key for m in matches]
    assert matches[1].key == "S01E06" and matches[2].key == "S01E07"


def test_episode_range_bounds_the_alignment_but_text_still_decides(season):
    """A per-file range (e.g. from a disc's title map) bounds the rip-order
    alignment and the fill pass. A decisive text match outside it still wins,
    and the window is in the evidence, so the caller can flag the disagreement."""
    folder, files, show, config = season

    matches = api.identify(folder, show=show, config=config, episode_range={files[0]: (7, 8)})

    assert matches[0].key == "S01E05"
    assert matches[0].evidence["window"] == [7, 8]
    assert matches[1].evidence["window"] == [4, 8]   # no range given: the whole season
    whole = api.identify(folder, show=show, config=config, episode_range=(5, 6))
    assert [m.evidence["window"] for m in whole] == [[5, 6]] * 3


def test_unmatched_file_keeps_its_best_guess_as_evidence(season, monkeypatch):
    folder, files, show, config = season
    junk = FILLER + "nothing here resembles any reference at all qqq"
    orig = I.get_transcript

    def get(mkv_path, *a, **k):
        if mkv_path == files[2]:
            return junk, 44 * 60
        return orig(mkv_path, *a, **k)

    monkeypatch.setattr(I, "get_transcript", get)
    matches = api.identify(folder, show=show, config=config, fill_unmatched=False)

    last = matches[2]
    assert last.method == "unmatched"
    assert last.key is None and last.season is None
    assert last.evidence["status"] == "unmatched"
    assert files[2].exists()  # no UNMATCHED_ prefix in library mode


def test_no_transcript_is_unmatched_with_its_reason(season, monkeypatch):
    folder, files, show, config = season
    orig = I.get_transcript
    monkeypatch.setattr(I, "get_transcript",
                        lambda p, *a, **k: (None, 0.0) if p == files[1] else orig(p, *a, **k))

    matches = api.identify(folder, show=show, config=config)

    assert matches[1].method == "unmatched"
    assert matches[1].evidence["status"] == "no_transcript"


def test_progress_reports_each_file(season):
    folder, files, show, config = season
    seen = []
    api.identify(folder, show=show, config=config, progress=lambda msg, frac: seen.append((msg, frac)))
    assert [m for m, _ in seen[:3]] == [f"text of {f.name}" for f in files]
    assert seen[-1] == ("matching", 1.0)


def test_identify_errors_are_explicit(season, monkeypatch):
    folder, files, show, config = season
    with pytest.raises(api.IdentifyError, match="credentials"):
        api.identify(folder, show=show, config=IdentifyConfig(ai="none"))
    with pytest.raises(api.IdentifyError, match="no episodes"):
        api.identify(folder, show=Show(name="Test Show"), config=config)
    monkeypatch.setattr(I, "os_login", lambda *a, **k: None)
    with pytest.raises(api.IdentifyError, match="login"):
        api.identify(folder, show=show, config=config)


def test_methods_from_engine_results():
    base = {"file": "a.mkv", "path": "/x/a.mkv", "confidence": 0.9,
            "episode": {"season": 2, "number": 3, "name": "Three"}}
    assert api.match_from_result({**base, "status": "renamed"}).method == "direct"
    assert api.match_from_result({**base, "status": "matched", "ai_picked": True}).method == "ai_fallback"
    filled = api.match_from_result({**base, "status": "filled", "fill_score": 0.512345})
    assert filled.method == "filled" and filled.key == "S02E03"
    assert filled.evidence["fill_score"] == 0.5123
    un = api.match_from_result({**base, "status": "unmatched"})
    assert un.method == "unmatched" and un.key is None
    assert un.evidence["best_guess"] == {"season": 2, "episode": 3, "title": "Three"}
    skipped = api.match_from_result({"file": "b.mkv", "status": "sparse_transcript", "confidence": 0.0})
    assert skipped.method == "unmatched" and skipped.path == Path("b.mkv")


def test_json_round_trip_and_write(tmp_path, capsys):
    m = EpisodeMatch(Path("/x/t00.mkv"), 1, 5, "Ep5", 0.91234, "direct", {"status": "matched"})
    d = json.loads(api.matches_to_json([m]))[0]
    assert d == {"path": "/x/t00.mkv", "season": 1, "episode": 5, "key": "S01E05", "title": "Ep5",
                 "confidence": 0.9123, "method": "direct", "evidence": {"status": "matched"}}
    assert EpisodeMatch.from_dict(d).key == "S01E05"
    api.write_json([m], tmp_path / "out.json")
    assert json.loads((tmp_path / "out.json").read_text())[0]["key"] == "S01E05"
    api.write_json([m], "-")
    assert json.loads(capsys.readouterr().out)[0]["method"] == "direct"


def test_cli_json_output_uses_the_same_shape(tmp_path):
    results = [{"file": "t00.mkv", "path": str(tmp_path / "t00.mkv"), "status": "renamed",
                "matched": "S01E05 - Ep5", "confidence": 0.9,
                "episode": {"season": 1, "number": 5, "name": "Ep5"}}]
    I._write_json_results(results, str(tmp_path / "r.json"))
    out = json.loads((tmp_path / "r.json").read_text())
    assert out[0]["key"] == "S01E05" and out[0]["method"] == "direct"


def test_cli_parses_new_flags():
    args = I.parse_args(["/tmp/x", "--json", "-", "--mlx-whisper"])
    assert args.json == "-" and args.mlx_whisper is True
    assert I.parse_args(["/tmp/x"]).json is None


def test_apply_renames_after_review_with_template_and_swaps(tmp_path):
    a, b, c = (tmp_path / n for n in ("S01E02.mkv", "S01E01.mkv", "t02.mkv"))
    for f in (a, b, c):
        f.write_text(f.name)
    matches = [
        EpisodeMatch(a, 1, 1, "Pilot", 0.9, "direct"),   # the two swap names
        EpisodeMatch(b, 1, 2, "Threads", 0.9, "direct"),
        EpisodeMatch(c, None, None, None, 0.2, "unmatched"),   # left alone
    ]
    done = api.apply_renames(matches)

    assert sorted((o.name, n.name) for o, n in done) == [("S01E01.mkv", "S01E02.mkv"),
                                                          ("S01E02.mkv", "S01E01.mkv")]
    assert (tmp_path / "S01E01.mkv").read_text() == "S01E02.mkv"   # a's content
    assert (tmp_path / "S01E02.mkv").read_text() == "S01E01.mkv"
    assert c.exists()


def test_target_name_template_fields():
    m = EpisodeMatch(Path("/x/title_t00.mkv"), 1, 1, 'Pilot: "Part 1"', 0.9, "direct")
    assert api.target_name(m) == "S01E01.mkv"
    assert api.target_name(m, "{show} - S{season:02d}E{episode:02d} - {title}", show="White Collar") \
        == "White Collar - S01E01 - Pilot Part 1.mkv"
    with pytest.raises(ValueError):
        api.target_name(EpisodeMatch(Path("/x/a.mkv"), None, None, None, 0, "unmatched"))


def test_apply_renames_dry_run_changes_nothing(tmp_path):
    f = tmp_path / "t00.mkv"
    f.write_text("x")
    done = api.apply_renames([EpisodeMatch(f, 1, 4, "Four", 0.9, "direct")], dry_run=True)
    assert done == [(f, tmp_path / "S01E04.mkv")]
    assert f.exists() and not (tmp_path / "S01E04.mkv").exists()


def test_config_from_env_and_engine_args(tmp_path):
    env = {"OPENSUBTITLES_API_KEY": "k", "OPENSUBTITLES_USERNAME": "u", "OPENSUBTITLES_PASSWORD": "p",
           "TMDB_API_KEY": "t"}
    c = IdentifyConfig.from_env(env, whisper="mlx")
    assert (c.opensubtitles_api_key, c.tmdb_api_key, c.anthropic_api_key) == ("k", "t", None)
    assert c.resolved_whisper_model() == I.WHISPER_LOCAL_DEFAULT_MODEL
    assert IdentifyConfig().resolved_whisper_model() == I.WHISPER_REMOTE_DEFAULT_MODEL

    args = api._engine_args(tmp_path, c, 0.8, False, 0.3, 2, None)
    assert args.local_whisper == "mlx" and args.identify_only is True and args.dry_run is True
    assert args.no_fill_unmatched is True and args.threshold == 0.8
    assert api._engine_args(tmp_path, IdentifyConfig(whisper="openai"), .75, True, .4, 3, None).local_whisper is True
    assert api._engine_args(tmp_path, IdentifyConfig(whisper="none"), .75, True, .4, 3, None).no_whisper is True


def test_mlx_whisper_repo_names():
    assert I.mlx_whisper_repo("medium") == "mlx-community/whisper-medium-mlx"
    assert I.mlx_whisper_repo("large-v3-turbo") == "mlx-community/whisper-large-v3-turbo"
    assert I.mlx_whisper_repo("mlx-community/whisper-small.en-mlx") == "mlx-community/whisper-small.en-mlx"


def test_transcribe_audio_dispatches_mlx(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(I, "transcribe_with_mlx_whisper", lambda *a, **k: calls.append("mlx") or "t")
    monkeypatch.setattr(I, "transcribe_with_whisper", lambda *a, **k: calls.append("openai") or "t")
    monkeypatch.setattr(I, "transcribe_with_remote_whisper", lambda *a, **k: calls.append("remote") or "t")
    for backend in ("mlx", True, False):
        I.transcribe_audio(tmp_path / "x.mkv", "medium", 0, "http://w", backend)
    assert calls == ["mlx", "openai", "remote"]


def test_transcript_cache_keeps_the_source_and_keys_mlx_apart(tmp_path, monkeypatch):
    f = tmp_path / "t00.mkv"
    f.write_bytes(b"x")

    def fake(mkv_path, *a, **k):
        I._TRANSCRIPT_SOURCES[str(mkv_path)] = "whisper (mlx, medium)"
        return "words " * 200, 2640.0

    monkeypatch.setattr(I, "get_transcript", fake)
    cache = tmp_path / "tc"
    I.get_transcript_cached(f, "medium", 0, 60, False, "", "mlx", cache)
    I._TRANSCRIPT_SOURCES.clear()
    monkeypatch.setattr(I, "get_transcript", lambda *a, **k: pytest.fail("cache missed"))
    I.get_transcript_cached(f, "medium", 0, 60, False, "", "mlx", cache)
    assert I.transcript_source(f) == "whisper (mlx, medium)"

    key_mlx = I._transcript_cache_key(f, "medium", 0, 60, "", "mlx", None)
    key_local = I._transcript_cache_key(f, "medium", 0, 60, "", True, None)
    assert key_mlx != key_local


def test_load_show_by_tvmaze_id_with_tmdb_summaries(monkeypatch):
    class R:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    monkeypatch.setattr(api.requests, "get", lambda url, **k: R({"name": "White Collar"}))
    monkeypatch.setattr(I, "tvmaze_get_episodes", lambda sid, season: [
        {"season": 1, "number": 1, "name": "Pilot", "summary": "<p>short</p>", "runtime": 60}])
    monkeypatch.setattr(I, "tmdb_search_show", lambda name, key: {"id": 108, "name": name})
    monkeypatch.setattr(I, "tmdb_get_episodes", lambda sid, season, key: [
        {"season": 1, "number": 1, "name": "Pilot", "summary": "a much longer summary", "runtime": 60}])

    show = api.load_show(tvmaze_id=5, season=1, tmdb_api_key="t")
    assert (show.name, show.tvmaze_id, show.tmdb_id) == ("White Collar", 5, 108)
    assert show.episodes[0]["summary"] == "a much longer summary"

    tmdb_only = api.load_show(tmdb_id=108, season=1, tmdb_api_key="t")
    assert tmdb_only.tvmaze_id is None and tmdb_only.episodes[0]["name"] == "Pilot"
