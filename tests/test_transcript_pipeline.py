"""
Tests for the Whisper-adjacent accuracy features: initial_prompt hinting,
no_speech_prob segment filtering, and the on-disk transcript cache, plus the
PGS subtitle OCR path (clean-room .sup decoder, tesseract wrapper, and the
get_transcript branch that prefers OCR over Whisper).
"""
import subprocess
from pathlib import Path

import pytest

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

    def fake_get_transcript(mkv_path, model, duration, skip, no_whisper, whisper_url, local_whisper, initial_prompt=None, no_ocr=False):
        calls.append(1)
        return "the real transcript", 2640.0

    monkeypatch.setattr(I, "get_transcript", fake_get_transcript)

    t1, d1 = I.get_transcript_cached(mkv, "medium", 0, 60, False, "http://x.invalid", True, cache_dir)
    t2, d2 = I.get_transcript_cached(mkv, "medium", 0, 60, False, "http://x.invalid", True, cache_dir)

    assert t1 == t2 == "the real transcript"
    assert d1 == d2 == 2640.0
    assert len(calls) == 1  # second call was a cache hit, not a recomputation


def test_transcript_cache_miss_when_whisper_model_changes(tmp_path, monkeypatch):
    mkv = tmp_path / "episode.mkv"
    mkv.write_bytes(b"fake video content")
    cache_dir = tmp_path / ".transcript_cache"

    calls = []

    def fake_get_transcript(mkv_path, model, duration, skip, no_whisper, whisper_url, local_whisper, initial_prompt=None, no_ocr=False):
        calls.append(model)
        return f"transcript from {model}", 100.0

    monkeypatch.setattr(I, "get_transcript", fake_get_transcript)

    I.get_transcript_cached(mkv, "base", 0, 60, False, "http://x.invalid", True, cache_dir)
    I.get_transcript_cached(mkv, "medium", 0, 60, False, "http://x.invalid", True, cache_dir)

    assert calls == ["base", "medium"]  # different settings -> different cache key -> both ran


def test_transcript_cache_none_dir_always_calls_through(tmp_path, monkeypatch):
    mkv = tmp_path / "episode.mkv"
    mkv.write_bytes(b"fake video content")

    calls = []

    def fake_get_transcript(mkv_path, model, duration, skip, no_whisper, whisper_url, local_whisper, initial_prompt=None, no_ocr=False):
        calls.append(1)
        return "x", 1.0

    monkeypatch.setattr(I, "get_transcript", fake_get_transcript)

    I.get_transcript_cached(mkv, "base", 0, 60, False, "http://x.invalid", True, None)
    I.get_transcript_cached(mkv, "base", 0, 60, False, "http://x.invalid", True, None)

    assert len(calls) == 2  # no cache dir -> caching disabled, always recomputes


def test_transcript_cache_does_not_persist_none_results(tmp_path, monkeypatch):
    mkv = tmp_path / "episode.mkv"
    mkv.write_bytes(b"fake video content")
    cache_dir = tmp_path / ".transcript_cache"

    monkeypatch.setattr(
        I, "get_transcript",
        lambda *a, **k: (None, 0.0),
    )

    transcript, duration = I.get_transcript_cached(mkv, "base", 0, 60, False, "http://x.invalid", True, cache_dir)
    assert transcript is None
    # A "too short/no usable content" result is cheap to recompute -> not cached.
    assert not cache_dir.exists() or not any(cache_dir.iterdir())


# ---------------------------------------------------------------------------
# probe_remote_whisper retry-on-timeout
# ---------------------------------------------------------------------------
class _FakeProbeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"text": "hello there"}


def test_probe_remote_whisper_retries_once_after_timeout(monkeypatch):
    # First attempt times out (model still loading), second succeeds -- should
    # report the backend as healthy rather than giving up after one timeout.
    calls = {"n": 0}

    def fake_post(url, files, data, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise I.requests.exceptions.Timeout()
        return _FakeProbeResponse()

    monkeypatch.setattr(I.requests, "post", fake_post)
    assert I.probe_remote_whisper("http://fake", "large-v3", timeout=1) is True
    assert calls["n"] == 2


def test_probe_remote_whisper_gives_up_after_two_timeouts(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, files, data, timeout):
        calls["n"] += 1
        raise I.requests.exceptions.Timeout()

    monkeypatch.setattr(I.requests, "post", fake_post)
    assert I.probe_remote_whisper("http://fake", "large-v3", timeout=1) is False
    assert calls["n"] == 2  # retried exactly once, not forever


def test_probe_remote_whisper_connection_error_does_not_retry(monkeypatch):
    # An unreachable server is a different failure mode than a slow/cold one --
    # retrying won't help, so this should fail fast on the first attempt.
    calls = {"n": 0}

    def fake_post(url, files, data, timeout):
        calls["n"] += 1
        raise I.requests.exceptions.ConnectionError()

    monkeypatch.setattr(I.requests, "post", fake_post)
    assert I.probe_remote_whisper("http://fake", "large-v3", timeout=1) is False
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# PGS OCR: pure logic
# ---------------------------------------------------------------------------
def test_assemble_ocr_transcript_joins_in_order_skipping_empty():
    texts = ["Hello there.", None, "", "   ", "General Kenobi!"]
    assert I._assemble_ocr_transcript(texts) == "Hello there. General Kenobi!"


def test_assemble_ocr_transcript_all_empty_returns_empty_string():
    assert I._assemble_ocr_transcript([None, "", "   "]) == ""


# ---------------------------------------------------------------------------
# PGS OCR: clean-room .sup decoder
# ---------------------------------------------------------------------------
def _pgs_segment(seg_type, pts_ms, payload):
    pts90 = int(pts_ms * 90).to_bytes(4, "big")
    return b"PG" + pts90 + b"\x00\x00\x00\x00" + bytes([seg_type]) + len(payload).to_bytes(2, "big") + payload


def _tiny_pgs_stream():
    # One 3x2 cue at t=1000ms: row 1 = opaque-white, opaque-white, transparent;
    # row 2 = three transparent pixels. Followed by a bare "clear screen"
    # display set (END with no ODS) at t=2000ms, which must NOT yield a cue.
    palette = bytes([0, 0]) + bytes([1, 235, 128, 128, 255])  # index 1 -> white, alpha 255
    rle = b"\x01\x01\x00\x01" + b"\x00\x00" + b"\x00\x03" + b"\x00\x00"
    ods = (
        b"\x00\x00" + b"\x00" + b"\xc0" + (len(rle) + 4).to_bytes(3, "big")
        + (3).to_bytes(2, "big") + (2).to_bytes(2, "big") + rle
    )
    return (
        _pgs_segment(0x14, 1000, palette)
        + _pgs_segment(0x15, 1000, ods)
        + _pgs_segment(0x80, 1000, b"")
        + _pgs_segment(0x80, 2000, b"")
    )


def test_decode_pgs_cues_decodes_bitmap_timestamp_and_skips_clear_screens(tmp_path):
    pytest.importorskip("PIL")
    sup = tmp_path / "tiny.sup"
    sup.write_bytes(_tiny_pgs_stream())

    cues = I.decode_pgs_cues(sup)

    assert len(cues) == 1  # the trailing clear-screen display set yields nothing
    image, pts_ms = cues[0]
    assert pts_ms == 1000.0
    assert image.size == (3, 2)
    assert image.getpixel((0, 0)) == (235, 235, 235, 255)  # palette index 1: opaque white
    assert image.getpixel((2, 0))[3] == 0                   # RLE transparent run
    assert image.getpixel((1, 1))[3] == 0                   # whole second row transparent


def test_decode_pgs_cues_stops_cleanly_at_truncated_or_corrupt_tail(tmp_path):
    pytest.importorskip("PIL")
    sup = tmp_path / "truncated.sup"
    sup.write_bytes(_tiny_pgs_stream() + b"NOT-A-PGS-SEGMENT-AT-ALL")

    assert len(I.decode_pgs_cues(sup)) == 1  # garbage tail ignored, earlier cue kept


# ---------------------------------------------------------------------------
# PGS OCR: tesseract wrapper
# ---------------------------------------------------------------------------
def test_composite_for_ocr_flattens_transparency_onto_black():
    Image = pytest.importorskip("PIL.Image")
    img = Image.new("RGBA", (2, 1))
    img.putpixel((0, 0), (255, 255, 255, 255))  # opaque white text pixel
    img.putpixel((1, 0), (255, 255, 255, 0))    # fully transparent background pixel

    flat = I._composite_for_ocr(img)

    assert flat.mode == "L"
    assert flat.getpixel((0, 0)) == 255  # text stays white...
    assert flat.getpixel((1, 0)) == 0    # ...on black, not white (see docstring)


class _FakeRun:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _blank_cue_image():
    Image = pytest.importorskip("PIL.Image")
    return Image.new("RGBA", (4, 4))


def test_ocr_image_with_tesseract_returns_stripped_text_and_pipes_png_over_stdin(monkeypatch):
    seen = {}

    def fake_run(cmd, input, capture_output, timeout):
        seen.update(cmd=cmd, input=input)
        return _FakeRun(stdout=b"  Hello there.\n")

    monkeypatch.setattr(I.subprocess, "run", fake_run)
    assert I.ocr_image_with_tesseract(_blank_cue_image()) == "Hello there."
    assert seen["cmd"][:3] == ["tesseract", "stdin", "stdout"]
    assert "--psm" in seen["cmd"]
    assert seen["input"].startswith(b"\x89PNG")


def test_ocr_image_with_tesseract_empty_output_is_empty_string_not_none(monkeypatch):
    # "tesseract ran fine, found no text" must stay distinguishable from
    # "tesseract is broken" -- ocr_transcribe_subtitle_stream counts only the
    # latter toward its consecutive-failure bail-out.
    monkeypatch.setattr(I.subprocess, "run", lambda *a, **k: _FakeRun(stdout=b"\n"))
    assert I.ocr_image_with_tesseract(_blank_cue_image()) == ""


def test_ocr_image_with_tesseract_nonzero_exit_returns_none(monkeypatch, caplog):
    monkeypatch.setattr(I.subprocess, "run", lambda *a, **k: _FakeRun(returncode=1, stderr=b"boom"))
    with caplog.at_level("ERROR"):
        assert I.ocr_image_with_tesseract(_blank_cue_image()) is None
    assert "boom" in caplog.text


def test_ocr_image_with_tesseract_timeout_returns_none(monkeypatch):
    def fake_run(*a, **k):
        raise subprocess.TimeoutExpired(cmd="tesseract", timeout=30)

    monkeypatch.setattr(I.subprocess, "run", fake_run)
    assert I.ocr_image_with_tesseract(_blank_cue_image()) is None


# ---------------------------------------------------------------------------
# PGS OCR: ocr_transcribe_subtitle_stream orchestration
# ---------------------------------------------------------------------------
def _patch_stream_pipeline(monkeypatch, tmp_path, n_cues, ocr_results):
    monkeypatch.setattr(I, "extract_pgs_subtitle", lambda *a, **k: tmp_path / "x.sup")
    monkeypatch.setattr(I, "decode_pgs_cues", lambda p: [(object(), float(i)) for i in range(n_cues)])
    calls = []

    def fake_ocr(image):
        calls.append(1)
        return ocr_results[len(calls) - 1]

    monkeypatch.setattr(I, "ocr_image_with_tesseract", fake_ocr)
    return calls


def test_ocr_transcribe_subtitle_stream_assembles_cues_in_order(monkeypatch, tmp_path):
    calls = _patch_stream_pipeline(monkeypatch, tmp_path, 3, ["First line.", "", "Third line."])
    text = I.ocr_transcribe_subtitle_stream(Path("ep.mkv"), 4, tmp_path)
    assert text == "First line. Third line."
    assert len(calls) == 3


def test_ocr_transcribe_subtitle_stream_bails_after_consecutive_failures(monkeypatch, tmp_path):
    calls = _patch_stream_pipeline(monkeypatch, tmp_path, 10, [None] * 10)
    assert I.ocr_transcribe_subtitle_stream(Path("ep.mkv"), 4, tmp_path) is None
    assert len(calls) == I._OCR_MAX_CONSECUTIVE_FAILURES  # stopped early, not all 10


def test_ocr_transcribe_subtitle_stream_failure_streak_resets_after_a_success(monkeypatch, tmp_path):
    # Two failures, a success, two more failures: never 3 in a row, so it keeps going.
    calls = _patch_stream_pipeline(monkeypatch, tmp_path, 6, [None, None, "a", None, None, "b"])
    assert I.ocr_transcribe_subtitle_stream(Path("ep.mkv"), 4, tmp_path) == "a b"
    assert len(calls) == 6


def test_ocr_transcribe_subtitle_stream_returns_none_when_extraction_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(I, "extract_pgs_subtitle", lambda *a, **k: None)
    assert I.ocr_transcribe_subtitle_stream(Path("ep.mkv"), 4, tmp_path) is None


def test_ocr_transcribe_subtitle_stream_returns_none_when_decode_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(I, "extract_pgs_subtitle", lambda *a, **k: tmp_path / "x.sup")

    def boom(p):
        raise ValueError("corrupt")

    monkeypatch.setattr(I, "decode_pgs_cues", boom)
    assert I.ocr_transcribe_subtitle_stream(Path("ep.mkv"), 4, tmp_path) is None


# ---------------------------------------------------------------------------
# get_transcript: OCR branch selection
# ---------------------------------------------------------------------------
def _media_info(streams, duration=2700):
    return {"format": {"duration": str(duration)}, "streams": streams}


def _pgs_stream(index=2):
    return {"index": index, "codec_name": "hdmv_pgs_subtitle", "codec_type": "subtitle", "tags": {}}


def test_get_transcript_uses_ocr_when_only_pgs_present_and_skips_whisper(tmp_path, monkeypatch):
    mkv = tmp_path / "episode.mkv"
    mkv.touch()
    monkeypatch.setattr(I, "get_media_info", lambda p: _media_info([_pgs_stream()]))
    monkeypatch.setattr(I, "ocr_transcribe_subtitle_stream", lambda *a, **k: "word " * 150)
    monkeypatch.setattr(
        I, "transcribe_audio",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("whisper must not run")),
    )
    text, duration = I.get_transcript(mkv, "medium", 0, 60, False, "http://x", True)
    assert len(text.split()) == 150
    assert duration == 2700.0


def test_get_transcript_falls_back_to_whisper_when_ocr_too_sparse(tmp_path, monkeypatch):
    mkv = tmp_path / "episode.mkv"
    mkv.touch()
    monkeypatch.setattr(I, "get_media_info", lambda p: _media_info([_pgs_stream()]))
    monkeypatch.setattr(I, "ocr_transcribe_subtitle_stream", lambda *a, **k: "only three words")
    monkeypatch.setattr(I, "transcribe_audio", lambda *a, **k: "whisper transcript")
    text, _ = I.get_transcript(mkv, "medium", 0, 60, False, "http://x", True)
    assert text == "whisper transcript"


def test_get_transcript_falls_back_to_whisper_when_ocr_libs_unavailable(tmp_path, monkeypatch):
    mkv = tmp_path / "episode.mkv"
    mkv.touch()
    monkeypatch.setattr(I, "get_media_info", lambda p: _media_info([_pgs_stream()]))
    monkeypatch.setattr(I, "OCR_LIBS_AVAILABLE", False)
    called = []
    monkeypatch.setattr(I, "ocr_transcribe_subtitle_stream", lambda *a, **k: called.append(1))
    monkeypatch.setattr(I, "transcribe_audio", lambda *a, **k: "whisper transcript")
    text, _ = I.get_transcript(mkv, "medium", 0, 60, False, "http://x", True)
    assert called == []
    assert text == "whisper transcript"


def test_get_transcript_skips_non_pgs_image_codecs_in_v1(tmp_path, monkeypatch):
    mkv = tmp_path / "episode.mkv"
    mkv.touch()
    vobsub_stream = {"index": 2, "codec_name": "dvd_subtitle", "codec_type": "subtitle", "tags": {}}
    monkeypatch.setattr(I, "get_media_info", lambda p: _media_info([vobsub_stream]))
    called = []
    monkeypatch.setattr(I, "ocr_transcribe_subtitle_stream", lambda *a, **k: called.append(1))
    monkeypatch.setattr(I, "transcribe_audio", lambda *a, **k: "whisper transcript")
    text, _ = I.get_transcript(mkv, "medium", 0, 60, False, "http://x", True)
    assert called == []
    assert text == "whisper transcript"


def test_get_transcript_respects_no_ocr_flag(tmp_path, monkeypatch):
    mkv = tmp_path / "episode.mkv"
    mkv.touch()
    monkeypatch.setattr(I, "get_media_info", lambda p: _media_info([_pgs_stream()]))
    called = []
    monkeypatch.setattr(I, "ocr_transcribe_subtitle_stream", lambda *a, **k: called.append(1))
    monkeypatch.setattr(I, "transcribe_audio", lambda *a, **k: "whisper transcript")
    text, _ = I.get_transcript(mkv, "medium", 0, 60, False, "http://x", True, no_ocr=True)
    assert called == []
    assert text == "whisper transcript"
