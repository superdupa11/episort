"""
Tests for the Whisper-adjacent accuracy features: initial_prompt hinting,
no_speech_prob segment filtering, and the on-disk transcript cache.
"""
import identifier as I


# ---------------------------------------------------------------------------
# build_whisper_initial_prompt
# ---------------------------------------------------------------------------
def test_initial_prompt_picks_recurring_names_only():
    episodes = [
        {"name": "Pilot", "summary": "Aaron meets Betty for the first time."},
        {"name": "Return", "summary": "Aaron confronts Betty about the letter."},
        {"name": "Finale", "summary": "A one-off Xerxes appears just this once."},
    ]
    prompt = I.build_whisper_initial_prompt(episodes)
    assert prompt is not None
    assert "Aaron" in prompt
    assert "Betty" in prompt
    assert "Xerxes" not in prompt  # only appears once -> filtered out


def test_initial_prompt_none_when_nothing_recurs():
    episodes = [{"name": "", "summary": ""}]
    assert I.build_whisper_initial_prompt(episodes) is None


def test_initial_prompt_excludes_stop_words():
    episodes = [
        {"name": "The Return", "summary": "The Return happens. The Return matters."},
    ]
    prompt = I.build_whisper_initial_prompt(episodes)
    # "The" recurs a lot but is a stop word and must never end up in the prompt.
    assert prompt is None or "The" not in (prompt or "").split(", ")


# ---------------------------------------------------------------------------
# _filter_whisper_segments
# ---------------------------------------------------------------------------
def test_filter_whisper_segments_drops_high_no_speech_prob():
    result = {
        "segments": [
            {"text": " real dialogue here", "no_speech_prob": 0.05, "avg_logprob": -0.2},
            {"text": " hallucinated over music", "no_speech_prob": 0.95, "avg_logprob": -1.5},
            {"text": " more real dialogue", "no_speech_prob": 0.1, "avg_logprob": -0.3},
        ],
        "text": "fallback text should not be used when segments exist",
    }
    text = I._filter_whisper_segments(result)
    assert "real dialogue here" in text
    assert "more real dialogue" in text
    assert "hallucinated" not in text


def test_filter_whisper_segments_falls_back_to_plain_text_without_segments():
    result = {"text": "plain transcript, no segment data"}
    assert I._filter_whisper_segments(result) == "plain transcript, no segment data"


def test_filter_whisper_segments_all_dropped_returns_empty():
    result = {"segments": [{"text": "noise", "no_speech_prob": 0.99}]}
    assert I._filter_whisper_segments(result) == ""


# ---------------------------------------------------------------------------
# get_transcript_cached / _transcript_cache_key
# ---------------------------------------------------------------------------
def test_transcript_cache_hit_avoids_recomputation(tmp_path, monkeypatch):
    mkv = tmp_path / "episode.mkv"
    mkv.write_bytes(b"fake video content")
    cache_dir = tmp_path / ".transcript_cache"

    calls = []

    def fake_get_transcript(mkv_path, model, duration, skip, no_whisper, initial_prompt=None):
        calls.append(1)
        return "the real transcript", 2640.0

    monkeypatch.setattr(I, "get_transcript", fake_get_transcript)

    t1, d1 = I.get_transcript_cached(mkv, "medium", 0, 60, False, cache_dir)
    t2, d2 = I.get_transcript_cached(mkv, "medium", 0, 60, False, cache_dir)

    assert t1 == t2 == "the real transcript"
    assert d1 == d2 == 2640.0
    assert len(calls) == 1  # second call was a cache hit, not a recomputation


def test_transcript_cache_miss_when_whisper_model_changes(tmp_path, monkeypatch):
    mkv = tmp_path / "episode.mkv"
    mkv.write_bytes(b"fake video content")
    cache_dir = tmp_path / ".transcript_cache"

    calls = []

    def fake_get_transcript(mkv_path, model, duration, skip, no_whisper, initial_prompt=None):
        calls.append(model)
        return f"transcript from {model}", 100.0

    monkeypatch.setattr(I, "get_transcript", fake_get_transcript)

    I.get_transcript_cached(mkv, "base", 0, 60, False, cache_dir)
    I.get_transcript_cached(mkv, "medium", 0, 60, False, cache_dir)

    assert calls == ["base", "medium"]  # different settings -> different cache key -> both ran


def test_transcript_cache_none_dir_always_calls_through(tmp_path, monkeypatch):
    mkv = tmp_path / "episode.mkv"
    mkv.write_bytes(b"fake video content")

    calls = []

    def fake_get_transcript(mkv_path, model, duration, skip, no_whisper, initial_prompt=None):
        calls.append(1)
        return "x", 1.0

    monkeypatch.setattr(I, "get_transcript", fake_get_transcript)

    I.get_transcript_cached(mkv, "base", 0, 60, False, None)
    I.get_transcript_cached(mkv, "base", 0, 60, False, None)

    assert len(calls) == 2  # no cache dir -> caching disabled, always recomputes


def test_transcript_cache_does_not_persist_none_results(tmp_path, monkeypatch):
    mkv = tmp_path / "episode.mkv"
    mkv.write_bytes(b"fake video content")
    cache_dir = tmp_path / ".transcript_cache"

    monkeypatch.setattr(
        I, "get_transcript",
        lambda *a, **k: (None, 0.0),
    )

    transcript, duration = I.get_transcript_cached(mkv, "base", 0, 60, False, cache_dir)
    assert transcript is None
    # A "too short/no usable content" result is cheap to recompute -> not cached.
    assert not cache_dir.exists() or not any(cache_dir.iterdir())
