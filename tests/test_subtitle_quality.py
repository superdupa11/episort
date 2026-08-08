"""Tests for SRT text extraction and the reference-subtitle plausibility check."""
import identifier as I


def test_srt_to_text_strips_metadata_and_noise(tmp_path):
    srt = tmp_path / "x.srt"
    srt.write_text(
        "1\n"
        "00:00:01,000 --> 00:00:03,000\n"
        "[MUSIC PLAYING] JOHN: Hello <i>there</i>\n"
        "\n"
        "2\n"
        "00:00:04,000 --> 00:00:06,000\n"
        "(laughs) General Kenobi.\n"
    )
    text = I.srt_to_text(srt)
    assert "Hello there" in text
    assert "General Kenobi." in text
    assert "MUSIC" not in text
    assert "JOHN" not in text
    assert "<i>" not in text
    assert "laughs" not in text


def test_srt_to_text_missing_file_returns_empty():
    assert I.srt_to_text(I.Path("/does/not/exist.srt")) == ""


def test_srt_span_minutes_reads_last_end_timestamp(tmp_path):
    srt = tmp_path / "x.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nfirst\n\n"
        "2\n00:43:10,500 --> 00:43:12,000\nlast\n"
    )
    span = I._srt_span_minutes(srt)
    assert span is not None
    assert abs(span - 43.2) < 0.05


def test_srt_span_minutes_no_timestamps_returns_none(tmp_path):
    srt = tmp_path / "x.srt"
    srt.write_text("not a valid subtitle file at all")
    assert I._srt_span_minutes(srt) is None


def test_is_plausible_reference_subtitle_accepts_realistic_span(tmp_path):
    srt = tmp_path / "x.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nx\n\n2\n00:42:00,000 --> 00:42:01,000\ny\n")
    assert I.is_plausible_reference_subtitle(srt, expected_runtime_minutes=44) is True


def test_is_plausible_reference_subtitle_rejects_sync_test_length(tmp_path):
    srt = tmp_path / "x.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nx\n\n2\n00:01:30,000 --> 00:01:31,000\ny\n")
    assert I.is_plausible_reference_subtitle(srt, expected_runtime_minutes=44) is False


def test_is_plausible_reference_subtitle_rejects_movie_length_upload(tmp_path):
    srt = tmp_path / "x.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nx\n\n2\n02:10:00,000 --> 02:10:01,000\ny\n")
    assert I.is_plausible_reference_subtitle(srt, expected_runtime_minutes=44) is False


def test_is_plausible_reference_subtitle_assumes_true_without_enough_info(tmp_path):
    srt = tmp_path / "x.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nx\n")
    # No runtime on record at all -> can't judge, assume plausible.
    assert I.is_plausible_reference_subtitle(srt, expected_runtime_minutes=None) is True
    # Runtime known but the subtitle has no parseable timestamps -> also assume True.
    unparseable = tmp_path / "y.srt"
    unparseable.write_text("garbage, no timestamps here")
    assert I.is_plausible_reference_subtitle(unparseable, expected_runtime_minutes=44) is True
