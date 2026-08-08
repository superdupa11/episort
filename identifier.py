#!/usr/bin/env python3
"""
TV Show Episode Identifier

Scans a directory of MKV files ripped from BluRay discs, extracts dialogue via
embedded subtitles or local Whisper transcription, compares against reference
subtitles from OpenSubtitles, and renames each file to standard S01E01 - Title.mkv
format when confidence meets the threshold.

Files below the threshold are renamed with an UNMATCHED_ prefix so they are
visible for manual review but do not get silently lost.

Usage:
    python identifier.py "/path/to/Show/Season 1"
    python identifier.py /path/to/Show --season 1 --dry-run
    python identifier.py "/path/to/Show/Season 2" --threshold 0.85 --whisper-model small

Secrets file (recommended):
    Create a .secrets file next to identifier.py (KEY=VALUE, one per line).
    Environment variables override values from the file if both are set.

    OPENSUBTITLES_API_KEY=xxxxx
    OPENSUBTITLES_USERNAME=youruser
    OPENSUBTITLES_PASSWORD=yourpass
    ANTHROPIC_API_KEY=sk-ant-xxxxx        # only needed with --ai-fallback
    TMDB_API_KEY=xxxxx                    # optional; enriches AI-fallback summaries

Required credentials:
    OPENSUBTITLES_API_KEY      from https://www.opensubtitles.com/consumers (free)
    OPENSUBTITLES_USERNAME     your opensubtitles.com login
    OPENSUBTITLES_PASSWORD     your opensubtitles.com password

Optional credentials:
    ANTHROPIC_API_KEY          required only with --ai-fallback
    OPENSUBTITLES_APP_NAME     shown in API requests (default: tv-episode-identifier)
    TMDB_API_KEY               from https://www.themoviedb.org/settings/api (free)
                               Provides richer episode summaries for the AI fallback
                               and acts as a complete metadata source if TVMaze misses
                               the show.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from rapidfuzz import fuzz, process

# ---------------------------------------------------------------------------
# Optional dependency: openai-whisper
# ---------------------------------------------------------------------------
try:
    import whisper as _whisper  # type: ignore
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Secrets file loader
# ---------------------------------------------------------------------------
def _load_secrets(secrets_path: Optional[Path] = None) -> None:
    """
    Load KEY=VALUE pairs from a secrets file into os.environ.
    Environment variables already set take precedence over the file.
    Looks for .secrets next to this script unless an explicit path is given.
    Lines starting with # and blank lines are ignored.
    """
    if secrets_path is None:
        secrets_path = Path(__file__).with_name("tv.secrets")
    if not secrets_path.exists():
        return
    with secrets_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OPENSUBTITLES_BASE = "https://api.opensubtitles.com/api/v1"
TVMAZE_BASE = "https://api.tvmaze.com"
TMDB_BASE = "https://api.themoviedb.org/3"

MIN_DURATION_SECONDS = 300       # 5 minutes — shorter clips are treated as extras
MIN_TRANSCRIPT_WORDS = 100       # sparse transcript → probably not a full episode
OS_RATE_DELAY = 1.1              # seconds between OpenSubtitles API calls (5 req/s limit)
SAMPLE_WORDS = 50000             # words used for text matching per side — effectively
                                  # unbounded for any real TV episode (even a 90-minute
                                  # special tops out around 12-15k spoken words). Accuracy
                                  # is the priority here, not compute: rapidfuzz handles
                                  # sets this size in a fraction of a second, so there's no
                                  # reason to throw away potentially-discriminating dialogue
                                  # from later in an episode just to keep the word set small.
QUALITY_FLOOR = 0.10             # minimum Jaccard score expected for a correct match
JACCARD_AI_SKIP_FLOOR = 0.40     # if best raw Jaccard reaches this, the top pick is
                                  # text-supported enough that AI shouldn't be allowed to
                                  # override it with a hallucinated guess. This only ever
                                  # suppresses AI — it does NOT force the match to pass the
                                  # confidence threshold; a genuinely ambiguous match (low
                                  # ratio-confidence between the top two candidates) still
                                  # falls through to UNMATCHED for manual review rather than
                                  # being declared confident just because the raw score
                                  # cleared this floor.
GROUP_CONFIRMED_FLOOR = 0.85     # confidence floor granted when a multi-file disc-order
                                  # alignment resolves end-to-end with no gaps and no
                                  # unassigned files (see _alignment_is_conclusive) — several
                                  # files' physical rip order landing on one unbroken run of
                                  # the season's episode order isn't something a mismatched
                                  # file could produce by chance, so it's trusted even when
                                  # any single file's own best-vs-runner-up ratio is mediocre.
                                  # Only applied when the file's own text still agrees with
                                  # the forced pick (see the confidence>0 check at its use
                                  # site) and clears JACCARD_AI_SKIP_FLOOR, so a near-empty
                                  # transcript can't ride the group's coattails to a false
                                  # confident match.

DURATION_BONUS_MAX = 0.05        # max additive score boost when file duration matches episode runtime
DURATION_TIGHT_MIN = 3.0         # delta ≤ 3 min → full bonus (BluRay rips vary slightly)
DURATION_LOOSE_MIN = 8.0         # delta ≤ 8 min → half bonus; beyond → no bonus

FUZZY_SCORE_CUTOFF = 88.0        # rapidfuzz 0-100 similarity floor for a "same word" bonus
                                  # match (tolerates 1-2 char OCR/typo noise, e.g. "recieve")
ALT_CANDIDATE_TOP_K = 5          # when confidence is low, re-score this many top episodes
                                  # against alternate community subtitle uploads — wide enough
                                  # that the true match still gets a second look even when a
                                  # bad primary subtitle upload buried it a few slots down

# Common English words that appear in all subtitle files and don't help discriminate episodes.
# Filtering these out leaves only content words (names, places, plot-specific vocabulary).
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "nor", "so", "yet", "for", "as",
    "at", "by", "in", "of", "on", "to", "up", "be", "is", "am", "are",
    "was", "were", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "shall", "can",
    "that", "this", "these", "those", "it", "its",
    "i", "me", "my", "we", "our", "you", "your",
    "he", "him", "his", "she", "her", "they", "them", "their",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "no", "not",
    "only", "other", "same", "than", "then", "too", "very", "just",
    "if", "with", "about", "into", "from", "out", "over", "under", "here", "there",
    "get", "got", "go", "went", "come", "came", "make", "made", "think",
    "know", "see", "look", "want", "use", "find", "give", "tell",
    "work", "call", "try", "ask", "seem", "feel", "leave", "put",
    "mean", "keep", "let", "set", "turn", "like", "take", "say", "said",
    "now", "back", "right", "well", "also", "even", "still", "never",
    "always", "though", "because", "while", "without", "yes", "no",
    "ok", "okay", "oh", "ah", "um", "uh", "yeah", "don", "doesn", "didn",
    "won", "can", "couldn", "wouldn", "shouldn", "isn", "aren", "wasn",
    "weren", "hasn", "haven", "hadn", "s", "re", "ve", "ll", "d", "t",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
})

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("tv-identifier")


# ---------------------------------------------------------------------------
# 1. CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Identify and rename TV show MKV files using OpenSubtitles matching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("path", help="Directory to scan (season folder or show root)")
    p.add_argument("--show", metavar="NAME",
                   help="Override TV show name (default: inferred from directory structure)")
    p.add_argument("--season", type=int, metavar="N",
                   help="Override season number (default: inferred from directory structure)")
    p.add_argument("--disc-window", type=int, default=2, metavar="N",
                   help="When a file's path contains a disc number (a 'Disc N' folder, or a "
                        "D<n> token in the filename), bound its candidate episodes to that "
                        "disc's estimated range — the season's episode count split evenly "
                        "across the discs seen — padded by N episodes on each side to absorb "
                        "uneven disc splits (default: 2). This only bounds which episodes a "
                        "file's own alignment can land on, not what gets downloaded (the "
                        "whole season's reference subtitles are always fetched up front), so "
                        "a wider window costs a few extra comparisons, not extra network calls "
                        "— err on the side of wider. Set 0 for an exact split, or a large value "
                        "to effectively disable disc-based narrowing.")
    p.add_argument("--threshold", type=float, default=0.75, metavar="0.0-1.0",
                   help="Confidence required for renaming (default: 0.75)")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview what would be renamed without touching files")
    p.add_argument("--whisper-model", default="medium",
                   choices=["tiny", "base", "small", "medium", "large"],
                   help="Whisper model size used when no embedded subtitle is found (default: "
                        "medium — a large accuracy jump over base/small at a moderate speed "
                        "cost; use --whisper-model large for the lowest transcription error "
                        "rate if you can tolerate it being slower still, or a smaller model "
                        "when you need speed more than accuracy for a given run).")
    p.add_argument("--whisper-duration", type=int, default=0, metavar="MINUTES",
                   help="Minutes of audio to extract before Whisper transcription (default: 0, "
                        "meaning the full audio track — same as --full-scan). More transcribed "
                        "dialogue means more content words to match against, which is why this "
                        "defaults to everything; pass a positive value (e.g. 20) to transcribe "
                        "only that many minutes instead, trading accuracy for speed.")
    p.add_argument("--whisper-skip", type=int, default=60, metavar="SECONDS",
                   help="Seconds to skip at the start of audio before transcribing (default: 60). "
                        "Skips opening theme and credits, which can cause Whisper to hallucinate "
                        "dialogue — especially on music-heavy intros like Star Trek.")
    p.add_argument("--full-scan", action="store_true",
                   help="Transcribe the entire audio track. This is already the default "
                        "(--whisper-duration defaults to 0, i.e. full audio); this flag is kept "
                        "for explicitness/backward compatibility and is a no-op unless you've "
                        "also passed --whisper-duration.")
    p.add_argument("--no-whisper", action="store_true",
                   help="Skip Whisper fallback; only process files with embedded subtitles")
    p.add_argument("--cache-dir", default=".subtitle_cache", metavar="DIR",
                   help="Where to store downloaded reference subtitles (default: .subtitle_cache)")
    p.add_argument("--alt-candidates", type=int, default=4, metavar="N",
                   help="When text-match confidence is low, fetch up to N alternate "
                        "community subtitle uploads per candidate episode and keep the "
                        "best-scoring one. Community subs vary in quality, so this reduces "
                        "false negatives caused by one noisy upload. Set 0 to disable "
                        "(default: 4). Each alternate costs one OpenSubtitles download quota.")
    p.add_argument("--check-subtitles", action="store_true",
                   help="Only scan files and report embedded subtitle streams (text vs "
                        "image-based, language) — no matching, no renaming, no network calls.")
    p.add_argument("--force", action="store_true",
                   help="Re-process files that are already named (S01E01 or UNMATCHED_). "
                        "Useful after improving match settings or adding --ai-fallback.")
    p.add_argument("--ai-fallback", action="store_true",
                   help="Use Claude AI to identify episodes when text matching confidence is low. "
                        "Requires ANTHROPIC_API_KEY environment variable.")
    p.add_argument("--local-ai", metavar="MODEL",
                   help="Use a local Ollama/LM Studio model as AI fallback instead of Claude. "
                        "Example: --local-ai llama3.1 or --local-ai mistral. "
                        "No API key required. Requires Ollama or LM Studio running locally.")
    p.add_argument("--local-ai-url", default="http://localhost:11434",
                   metavar="URL",
                   help="Base URL for the local AI server (default: http://localhost:11434 for Ollama). "
                        "Use http://localhost:1234 for LM Studio.")
    p.add_argument("--log-dir", metavar="DIR", default=None,
                   help="Write a per-file match log (transcript + episode scores + AI prompt) to "
                        "this directory. Useful for debugging low-confidence matches. "
                        "Default: 'match_logs' folder next to identifier.py when omitted.")
    p.add_argument("--no-confirm", action="store_true",
                   help="Skip the interactive show-selection prompt and auto-select the top TVMaze "
                        "result. Useful for scripted/automated runs. The default is to always "
                        "show the top matches and ask you to confirm the correct show.")
    p.add_argument("--temp-dir", metavar="DIR", default=None,
                   help="Override the directory used for all temporary files (subtitle extraction "
                        "and Whisper audio clips). Use this when the system temp partition is full. "
                        "Example: --temp-dir /Volumes/MyDrive/tmp")
    p.add_argument("--series-review", action="store_true",
                   help="Review an entire series stored in one flat folder. Files must already be "
                        "named SxxExx (even if the episode numbers are guesses). The tool groups "
                        "files by season, asks which seasons to verify, then re-identifies each "
                        "file against a ±--review-window episode window around the guessed number "
                        "and renames those that reach the confidence threshold.")
    p.add_argument("--review-window", type=int, default=5, metavar="N",
                   help="Episode search window for --series-review: bound each file's alignment "
                        "to ±N episodes around its guessed number (default: 5, matching a rip "
                        "off a disc with up to ~5 episodes on it — a guess should never be off "
                        "by more than that). This only bounds candidates, not downloads (the "
                        "whole season's reference subtitles are fetched regardless), so a wider "
                        "window is nearly free — increase it if guesses might drift further.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Show debug-level logging")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 2. Directory inference
# ---------------------------------------------------------------------------
_SEASON_PATTERNS = [
    re.compile(r"[Ss]eason\s*(\d+)"),
    re.compile(r"[Ss]eries\s*(\d+)"),
    re.compile(r"^[Ss](\d{1,2})$"),
]


def infer_show_season(
    scan_path: Path,
    show_override: Optional[str],
    season_override: Optional[int],
) -> tuple[str, Optional[int]]:
    """
    Derive show name and season number from directory structure.

    /Shows/Breaking Bad/Season 1/  →  show='Breaking Bad', season=1
    /Shows/Breaking Bad/           →  show='Breaking Bad', season=None
    """
    season = season_override
    if season is None:
        for pat in _SEASON_PATTERNS:
            m = pat.search(scan_path.name)
            if m:
                season = int(m.group(1))
                break

    show = show_override or (scan_path.parent.name if season is not None else scan_path.name)
    return show, season


_DISC_PATTERNS = [
    # (?!\d) rather than \b: real rips append "_t01" etc. right after the disc
    # number (e.g. "Disc 6_t02.mkv"), and \b doesn't break between a digit and
    # an underscore since both are word characters.
    re.compile(r"[Dd]isc\s*(\d{1,2})(?!\d)"),
    re.compile(r"[Dd]isk\s*(\d{1,2})(?!\d)"),
    re.compile(r"(?<![A-Za-z0-9])[Dd](\d{1,2})(?![A-Za-z0-9])"),
]

_PART_PATTERNS = [
    re.compile(r"[Pp]art\s*(\d{1,2})(?!\d)"),
]


def _search_path_parts(path: Path, scan_root: Path, patterns: list[re.Pattern]) -> Optional[int]:
    """
    Search each path component between scan_root and path — directories first,
    then the filename — for the first pattern match. Used to pull a season or
    disc number out of a file's location even when it wasn't set at the top level
    (e.g. a show root containing per-season subfolders, or Disc N rip folders).
    """
    try:
        rel_parts = path.relative_to(scan_root).parts
    except ValueError:
        rel_parts = (path.name,)
    for part in rel_parts:
        for pat in patterns:
            m = pat.search(part)
            if m:
                return int(m.group(1))
    return None


def infer_file_season(mkv_path: Path, scan_root: Path, default_season: Optional[int]) -> Optional[int]:
    """
    Determine the season a specific file belongs to from its path (e.g.
    .../Season 2/Disc 1/title03.mkv), falling back to default_season when no
    season marker is found in the path. This only changes behavior when scanning
    a show root with multiple season subfolders — a single season folder already
    has its season fixed by infer_show_season.
    """
    found = _search_path_parts(mkv_path, scan_root, _SEASON_PATTERNS)
    return found if found is not None else default_season


def infer_disc_number(mkv_path: Path, scan_root: Path) -> Optional[int]:
    """
    Extract a disc number (Disc 1, Disc1, D1, Disk 2, ...) from the file's path —
    typically a "Disc N" rip folder or a D<n> token left in the filename by
    ripping software. Returns None when no disc marker is present.
    """
    return _search_path_parts(mkv_path, scan_root, _DISC_PATTERNS)


def infer_part_number(mkv_path: Path, scan_root: Path) -> Optional[int]:
    """
    Extract a "Part N" marker from the file's path — seasons released or ripped
    in multiple parts (e.g. a strike-delayed back half, common on recent shows)
    are commonly labelled "Part 2 - Disc 1". Used to recognize when a disc
    number can't be trusted to represent the season's overall disc layout: a
    part's own disc numbering restarts from 1 with no way to know how many
    discs an earlier, unseen part used (see narrow_by_disc's caller).
    """
    return _search_path_parts(mkv_path, scan_root, _PART_PATTERNS)


def build_disc_totals(
    mkv_files: list[Path], scan_root: Path, default_season: Optional[int]
) -> dict[int, int]:
    """
    Scan all discovered files up front and record the highest disc number seen
    per season. This estimates how many discs a season was split across, which
    narrow_by_disc uses to guess how many episodes land on each one.
    """
    totals: dict[int, int] = {}
    for f in mkv_files:
        season = infer_file_season(f, scan_root, default_season)
        disc = infer_disc_number(f, scan_root)
        if season is not None and disc is not None:
            totals[season] = max(totals.get(season, 0), disc)
    return totals


def narrow_by_disc(
    season_episodes: list[dict], disc: Optional[int], total_discs: int, window: int
) -> list[dict]:
    """
    Restrict a season's episode list to the range expected on one disc, assuming
    episodes split evenly across total_discs (ceil(n / total_discs) per disc).
    `window` pads both ends of the computed range to absorb uneven disc splits
    (bonus features, discs authored with different episode counts). Returns
    season_episodes unchanged when disc/total_discs can't be used.
    """
    if disc is None or total_discs <= 1 or not season_episodes:
        return season_episodes

    numbers = sorted(ep["number"] for ep in season_episodes)
    n = len(numbers)
    per_disc = -(-n // total_discs)  # ceil division
    lo_idx = min((disc - 1) * per_disc, n - 1)
    hi_idx = min(disc * per_disc - 1, n - 1)
    lo = numbers[lo_idx] - window
    hi = numbers[hi_idx] + window

    narrowed = [ep for ep in season_episodes if lo <= ep["number"] <= hi]
    return narrowed or season_episodes


# ---------------------------------------------------------------------------
# 3. MKV discovery
# ---------------------------------------------------------------------------
_ALREADY_NAMED = re.compile(r"[Ss]\d{2}[Ee]\d{2}")


_VIDEO_EXTENSIONS = ("*.mkv", "*.mp4", "*.m4v")


def find_video_files(scan_path: Path, force: bool = False) -> list[Path]:
    """Return sorted MKV/MP4 files. Already-identified files are skipped unless force=True."""
    all_files = sorted(
        f for ext in _VIDEO_EXTENSIONS for f in scan_path.rglob(ext)
    )
    to_process: list[Path] = []
    for f in all_files:
        if not force and _ALREADY_NAMED.search(f.stem):
            log.info(f"Skip (already named): {f.name}")
        else:
            to_process.append(f)
    return to_process


# ---------------------------------------------------------------------------
# 3b. Series review helpers
# ---------------------------------------------------------------------------
_SERIES_FILE_RE = re.compile(r"[Ss](\d{1,2})[Ee](\d{1,2})")


def parse_series_files(scan_path: Path) -> dict[int, list[tuple[Path, int]]]:
    """
    Find all video files whose stem contains SxxExx and group them by season.
    Returns {season_num: [(path, guessed_episode_num), ...]} sorted by episode.
    Files without a SxxExx pattern are skipped (logged at debug level).
    """
    result: dict[int, list[tuple[Path, int]]] = {}
    all_files = sorted(f for ext in _VIDEO_EXTENSIONS for f in scan_path.rglob(ext))
    for f in all_files:
        m = _SERIES_FILE_RE.search(f.stem)
        if not m:
            log.debug(f"Series review: skipping (no SxxExx in name): {f.name}")
            continue
        season = int(m.group(1))
        ep = int(m.group(2))
        result.setdefault(season, []).append((f, ep))
    for season in result:
        result[season].sort(key=lambda x: x[1])
    return result


def _format_season_ranges(seasons: list[int]) -> str:
    """Format [1,2,3,5,7,8] as '1-3, 5, 7-8'."""
    if not seasons:
        return ""
    ranges: list[str] = []
    start = prev = seasons[0]
    for s in seasons[1:]:
        if s == prev + 1:
            prev = s
        else:
            ranges.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = s
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ", ".join(ranges)


def parse_season_selection(raw: str, available: list[int]) -> list[int]:
    """
    Parse user input into a list of season numbers drawn from `available`.
    Accepts: "all", "1-3", "1,2,3", "1,3-5,7", "none"/"" (→ empty).
    """
    raw = raw.strip().lower()
    if not raw or raw in ("none", "n", "0"):
        return []
    
    if raw in ("all", "a", "*"):
        return sorted(available)
    selected: set[int] = set()
    for part in re.split(r"[,\s]+", raw):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, _, hi_s = part.partition("-")
            try:
                lo, hi = int(lo_s), int(hi_s)
                selected.update(s for s in available if lo <= s <= hi)
            except ValueError:
                pass
        else:
            try:
                s = int(part)
                if s in available:
                    selected.add(s)
            except ValueError:
                pass
    return sorted(selected)


def prompt_series_season_selection(
    season_files: dict[int, list],
    auto: bool = False,
) -> list[int]:
    """
    Display detected seasons and ask which to review.
    With auto=True (or non-TTY), all seasons are selected without prompting.
    Returns sorted list of season numbers to process; empty list means abort.
    """
    seasons = sorted(season_files.keys())
    total_files = sum(len(v) for v in season_files.values())
    print(f"\nFound {len(seasons)} season(s): {_format_season_ranges(seasons)}  ({total_files} file(s) total)")
    print()
    for s in seasons:
        n = len(season_files[s])
        print(f"  Season {s:2d}: {n} file(s)")
    print()

    if auto or not sys.stdin.isatty():
        print(f"  Auto-selecting all seasons (--no-confirm)")
        return seasons

    while True:
        raw = input(
            'Which seasons should be reviewed?\n'
            '  Enter "all", a range like "1-3", a list like "1,2,3", or "q" to quit: '
        ).strip()
        if raw.lower() in ("q", "quit", "exit"):
            return []
        selected = parse_season_selection(raw, seasons)
        if selected:
            print(f"  Will review: Season(s) {_format_season_ranges(selected)}")
            print()
            return selected
        print(f"  No valid seasons matched '{raw}'. Available: {_format_season_ranges(seasons)}")


# ---------------------------------------------------------------------------
# 4. FFmpeg / ffprobe utilities
# ---------------------------------------------------------------------------
def get_media_info(mkv_path: Path) -> dict:
    """Run ffprobe and return parsed JSON stream/format info."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            str(mkv_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.debug(f"ffprobe error: {result.stderr.strip()}")
        return {}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}


def get_duration(media_info: dict) -> float:
    return float(media_info.get("format", {}).get("duration", 0))


def find_subtitle_streams(media_info: dict) -> list[dict]:
    """Return subtitle streams, English/undefined first."""
    streams = media_info.get("streams", [])
    subs = [s for s in streams if s.get("codec_type") == "subtitle"]

    def lang_priority(s: dict) -> int:
        lang = s.get("tags", {}).get("language", "und").lower()
        return 0 if lang in ("eng", "en", "und") else 1

    return sorted(subs, key=lang_priority)


def find_best_audio_stream(media_info: dict) -> Optional[int]:
    """
    Return the absolute stream index of the best audio track for transcription.
    Prefers English, deprioritizes commentary/description tracks.
    Returns None when there's only one audio track (no choice to make).
    """
    streams = media_info.get("streams", [])
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if len(audio) <= 1:
        return None

    def _score(s: dict) -> tuple:
        tags = s.get("tags", {})
        lang = tags.get("language", "und").lower()
        title = tags.get("title", "").lower()
        is_english = lang in ("eng", "en", "und")
        is_commentary = any(w in title for w in ("commentary", "director", "cast", "description", "audio description"))
        return (not is_english, is_commentary, s.get("index", 999))

    best = min(audio, key=_score)
    default_idx = audio[0].get("index")
    best_idx = best.get("index")
    if best_idx != default_idx:
        tags = best.get("tags", {})
        lang = tags.get("language", "und")
        title = tags.get("title", "")
        label = f"{lang} '{title}'" if title else lang
        log.info(f"  Audio: preferring stream {best_idx} ({label}) over default stream {default_idx}")
    return best_idx


def extract_subtitle(mkv_path: Path, stream_index: int, out_dir: Path) -> Optional[Path]:
    """Extract one subtitle stream to SRT. Returns the path or None on failure."""
    out = out_dir / f"sub_{stream_index}.srt"
    r = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "quiet",
            "-i", str(mkv_path),
            "-map", f"0:{stream_index}",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode == 0 and out.exists() and out.stat().st_size > 50:
        return out
    return None


_IMAGE_SUBTITLE_CODECS = {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub"}


def check_subtitles(scan_path: Path) -> None:
    """
    Read-only audit: report each MKV's embedded subtitle streams (if any) — text vs
    image-based, language — with no network calls, matching, or renaming. Answers
    "does this file already carry subtitles ffmpeg can actually read as text?"
    """
    files = sorted(f for ext in _VIDEO_EXTENSIONS for f in scan_path.rglob(ext))
    if not files:
        log.info("No .mkv or .mp4 files found.")
        return

    print(f"\n{'FILE':<55} {'DURATION':>9}  SUBTITLE STREAMS")
    print("-" * 110)
    for f in files:
        media_info = get_media_info(f)
        if not media_info:
            print(f"{f.name[:55]:<55} {'?':>9}  (ffprobe failed — is ffmpeg installed?)")
            continue
        duration = get_duration(media_info)
        subs = find_subtitle_streams(media_info)
        if not subs:
            desc = "none"
        else:
            parts = []
            for s in subs:
                lang = s.get("tags", {}).get("language", "und")
                codec = s.get("codec_name", "?")
                kind = "IMAGE (unusable)" if codec in _IMAGE_SUBTITLE_CODECS else "text"
                parts.append(f"{lang}:{codec}[{kind}]")
            desc = ", ".join(parts)
        print(f"{f.name[:55]:<55} {duration / 60:>7.1f}m  {desc}")
    print("-" * 110)
    print(
        f"{len(files)} file(s) scanned. 'text' streams are usable as-is; "
        f"'IMAGE' streams (PGS/DVD/DVB/xsub) require OCR and are always skipped in "
        f"favor of Whisper transcription."
    )


# ---------------------------------------------------------------------------
# 5. SRT text extraction
# ---------------------------------------------------------------------------
_TIMING_RE = re.compile(
    r"^\d{1,2}:\d{2}:\d{2}[,\.]\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,\.]\d{3}"
)
_INDEX_RE = re.compile(r"^\d+$")
_TAG_RE = re.compile(r"<[^>]+>|\{[^}]+\}")
# Hearing-impaired sound cues ("[MUSIC PLAYING]", "(laughs)") and speaker labels
# ("JOHN:", "- NARRATOR:") are common in community subtitles but never appear in a
# Whisper transcript. Left in, they pollute the reference vocabulary with words the
# transcript can never contain, dragging down scores for genuinely correct matches.
_SOUND_CUE_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")
_SPEAKER_LABEL_RE = re.compile(r"^-?\s*[A-Z][A-Z0-9 .'’-]{1,30}:\s*")


def srt_to_text(srt_path: Path) -> str:
    """Strip SRT timing/index metadata, inline tags, sound cues, and speaker labels."""
    try:
        content = srt_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines: list[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line or _INDEX_RE.match(line) or _TIMING_RE.match(line):
            continue
        line = _TAG_RE.sub("", line)
        line = _SOUND_CUE_RE.sub("", line)
        line = _SPEAKER_LABEL_RE.sub("", line)
        line = line.strip()
        if line:
            lines.append(line)
    return " ".join(lines)


_SRT_END_TIMESTAMP_RE = re.compile(
    r"-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def _srt_span_minutes(srt_path: Path) -> Optional[float]:
    """
    Return a subtitle file's total time span (its last cue's end timestamp) in
    minutes, or None if it has no parseable timestamps at all.
    """
    try:
        content = srt_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    last_end: Optional[float] = None
    for m in _SRT_END_TIMESTAMP_RE.finditer(content):
        h, mnt, s, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        last_end = h * 3600 + mnt * 60 + s + ms / 1000
    return last_end / 60.0 if last_end is not None else None


def is_plausible_reference_subtitle(srt_path: Path, expected_runtime_minutes) -> bool:
    """
    Cheap plausibility check on a reference subtitle before it's trusted for
    scoring: does its time span roughly match the episode's listed runtime?
    Catches a sync-test upload, a trailer, or a completely unrelated file
    mislabeled under the right episode ID — any of which would otherwise be
    scored normally and could produce a false negative for the real match (if
    it's the correct episode's slot but garbage content) or a false positive
    (if the garbage happens to overlap with the wrong file's transcript).

    Returns True — "assume plausible" — whenever there isn't enough information
    to judge (no runtime on record, or the subtitle has no parseable
    timestamps); this is a filter for clearly-broken data, not a strict
    validator, so it should never block a normal match for lack of metadata.
    """
    if not expected_runtime_minutes:
        return True
    span = _srt_span_minutes(srt_path)
    if span is None:
        return True
    # Subtitles rarely cover every silent second of an episode, so the low end
    # gets generous slack. The useful catch is a file wildly too short (a
    # 2-minute sync test attached to a 45-minute episode) or too long (a
    # movie-length or multi-episode compilation upload).
    return (expected_runtime_minutes * 0.4) <= span <= (expected_runtime_minutes * 1.5)


# ---------------------------------------------------------------------------
# 6. Whisper transcription
# ---------------------------------------------------------------------------
NO_SPEECH_PROB_FLOOR = 0.6   # Whisper segments this confident there's no speech are usually
                             # hallucinated text over music/silence/action rather than real
                             # dialogue — drop them instead of letting them pollute the word
                             # set used for matching.


def _filter_whisper_segments(result: dict) -> str:
    """
    Build the transcript text from Whisper's per-segment output, dropping any
    segment flagged with a high no_speech_prob. Falls back to the plain
    result["text"] when segments aren't available (shouldn't normally happen,
    but keeps this defensive rather than crashing on an unexpected result shape).
    Also logs the kept segments' average avg_logprob — Whisper's own estimate of
    how confident it was — purely for visibility into transcript quality; it's
    not used as a hard gate since it hasn't been empirically tuned into a
    reliable cutoff the way MIN_TRANSCRIPT_WORDS has.
    """
    segments = result.get("segments") or []
    if not segments:
        return (result.get("text") or "").strip()

    kept = [s for s in segments if s.get("no_speech_prob", 0.0) < NO_SPEECH_PROB_FLOOR]
    dropped = len(segments) - len(kept)
    if dropped:
        log.debug(f"  Whisper: dropped {dropped}/{len(segments)} segment(s) flagged as likely non-speech")
    if kept:
        avg_logprob = sum(s.get("avg_logprob", 0.0) for s in kept) / len(kept)
        log.debug(f"  Whisper: avg_logprob={avg_logprob:.2f} over {len(kept)} kept segment(s)")
    return " ".join(s.get("text", "").strip() for s in kept).strip()


def transcribe_with_whisper(
    mkv_path: Path,
    model_size: str,
    duration_minutes: int,
    skip_seconds: int = 0,
    audio_stream: Optional[int] = None,
    initial_prompt: Optional[str] = None,
) -> Optional[str]:
    if not WHISPER_AVAILABLE:
        log.warning("openai-whisper not installed — run: pip install openai-whisper")
        return None
    import warnings
    skip_label = f"skip {skip_seconds}s, " if skip_seconds else ""
    dur_label = "full audio" if not duration_minutes else f"next {duration_minutes} min of audio"
    log.info(f"  Transcribing with Whisper ({model_size}, {skip_label}{dur_label})...")
    try:
        model = _whisper.load_model(model_size)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            clip_path = Path(tmp.name)
        try:
            # Build ffmpeg command. -ss before -i is a fast input seek; -t limits duration.
            # Omit -t entirely when duration_minutes is 0 (full-scan mode).
            # Specifying the audio stream index avoids accidentally using a commentary track.
            cmd = ["ffmpeg", "-y", "-v", "quiet"]
            if skip_seconds:
                cmd += ["-ss", str(skip_seconds)]
            cmd += ["-i", str(mkv_path)]
            if audio_stream is not None:
                cmd += ["-map", f"0:{audio_stream}"]
            if duration_minutes:
                cmd += ["-t", str(duration_minutes * 60)]
            cmd += ["-vn", "-ar", "16000", "-ac", "1", str(clip_path)]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0 or not clip_path.exists():
                log.warning("  ffmpeg audio extraction failed; falling back to full file")
                clip_path = mkv_path
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
                result = model.transcribe(
                    str(clip_path), verbose=False, language="en", initial_prompt=initial_prompt,
                )
        finally:
            if clip_path != mkv_path:
                clip_path.unlink(missing_ok=True)
        return _filter_whisper_segments(result)
    except Exception as exc:
        log.error(f"  Whisper failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# 7. Transcript pipeline
# ---------------------------------------------------------------------------
def get_transcript(
    mkv_path: Path,
    whisper_model: str,
    whisper_duration: int,
    whisper_skip: int,
    no_whisper: bool,
    initial_prompt: Optional[str] = None,
) -> tuple[Optional[str], float]:
    """
    Try embedded subtitle streams first, fall back to Whisper transcription.
    Returns (transcript, duration_seconds). transcript is None when the file
    should be skipped (too short, no usable content); duration_seconds is 0.0
    in that case.
    """
    media_info = get_media_info(mkv_path)
    if not media_info:
        log.warning(f"  ffprobe returned no data — is ffmpeg installed?")
        return None, 0.0

    duration = get_duration(media_info)
    if duration < MIN_DURATION_SECONDS:
        log.info(f"  Skipping: too short ({duration / 60:.1f} min < 5 min)")
        return None, 0.0

    sub_streams = find_subtitle_streams(media_info)

    # Separate text-based from image-based subtitle streams.
    # PGS (hdmv_pgs_subtitle) and DVD bitmaps (dvd_subtitle) cannot be converted
    # to text by ffmpeg — they require OCR. Only text codecs (srt, ass, subrip,
    # webvtt, etc.) are useful here.
    text_streams = [s for s in sub_streams if s.get("codec_name") not in _IMAGE_SUBTITLE_CODECS]
    image_streams = [s for s in sub_streams if s.get("codec_name") in _IMAGE_SUBTITLE_CODECS]

    stream_summary = f"{len(text_streams)} text"
    if image_streams:
        codecs = ", ".join(s.get("codec_name", "?") for s in image_streams)
        stream_summary += f", {len(image_streams)} image-based ({codecs} — skipped)"
    log.info(f"  Duration: {duration / 60:.1f} min  |  Subtitle streams: {stream_summary}")

    try:
        _tmp_ctx = tempfile.TemporaryDirectory()
    except OSError as exc:
        if exc.errno == 28:  # ENOSPC
            log.error(
                "No space left on the temp partition.\n"
                "Re-run with --temp-dir pointing to a drive that has free space.\n"
                "Example: --temp-dir /Volumes/MyDrive/tmp"
            )
        else:
            log.error(f"Failed to create temp directory: {exc}")
        return None, 0.0
    with _tmp_ctx as tmp:
        tmp_path = Path(tmp)
        for stream in text_streams:
            idx = stream["index"]
            srt = extract_subtitle(mkv_path, idx, tmp_path)
            if not srt:
                continue
            text = srt_to_text(srt)
            word_count = len(text.split())
            if word_count >= MIN_TRANSCRIPT_WORDS:
                log.info(f"  Using subtitle stream {idx} ({word_count} words)")
                return text, duration
            log.debug(f"  Stream {idx}: only {word_count} words — skipping")

    if no_whisper:
        log.info("  No usable embedded subtitles and --no-whisper set. Skipping file.")
        return None, 0.0

    audio_idx = find_best_audio_stream(media_info)
    return transcribe_with_whisper(
        mkv_path, whisper_model, whisper_duration,
        skip_seconds=whisper_skip, audio_stream=audio_idx, initial_prompt=initial_prompt,
    ), duration


_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-zA-Z']{2,}\b")


def build_whisper_initial_prompt(episodes: list[dict], limit: int = 40) -> Optional[str]:
    """
    Build a short Whisper initial_prompt from proper nouns that recur across this
    season's episode titles/summaries (character names, places). Whisper biases
    transcription toward vocabulary present in the prompt, which matters most for
    exactly the words that survive stop-word filtering and do the discriminating
    in the text match — a mistranscribed character name silently becomes a
    non-match on both sides otherwise.

    Requiring a word to appear more than once across the season's summaries is a
    cheap way to prefer actual recurring names over a one-off capitalized
    sentence-starter that only happens to show up in a single episode's summary.
    Returns None when nothing usable was found (e.g. no summaries available).
    """
    counts: Counter = Counter()
    for ep in episodes:
        summary = re.sub(r"<[^>]+>", "", ep.get("summary") or "")
        text = f"{ep.get('name') or ''} {summary}"
        for word in _PROPER_NOUN_RE.findall(text):
            if word.lower() not in _STOP_WORDS:
                counts[word] += 1

    recurring = [word for word, n in counts.most_common(limit) if n > 1]
    return ", ".join(recurring) if recurring else None


def _transcript_cache_key(
    mkv_path: Path, whisper_model: str, whisper_duration: int, whisper_skip: int,
    initial_prompt: Optional[str],
) -> str:
    """
    Identity for a cached transcript: the file's path/size/mtime plus every
    setting that can change what get_transcript actually produces. Including the
    whisper settings and prompt means a changed --whisper-model (or a different
    season's initial_prompt) naturally misses the cache instead of serving a
    transcript generated under different conditions — a cache invalidation
    mistake here would silently corrupt matching, which is worse than the cost
    of an occasional unnecessary re-transcription.
    """
    st = mkv_path.stat()
    raw = (
        f"{mkv_path.resolve()}|{st.st_size}|{st.st_mtime}|"
        f"{whisper_model}|{whisper_duration}|{whisper_skip}|{initial_prompt or ''}"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def get_transcript_cached(
    mkv_path: Path,
    whisper_model: str,
    whisper_duration: int,
    whisper_skip: int,
    no_whisper: bool,
    transcript_cache_dir: Optional[Path],
    initial_prompt: Optional[str] = None,
) -> tuple[Optional[str], float]:
    """
    Wraps get_transcript with an on-disk cache keyed by file identity plus the
    whisper settings used, so an interrupted run — now costlier than it used to
    be, with full-scan and a larger Whisper model both defaults — doesn't re-pay
    transcription time for files it already transcribed. Only successful
    transcriptions are cached; a "too short" / "no usable content" result is
    cheap to recompute (it short-circuits before Whisper ever runs) so there's no
    need to persist it.
    """
    if transcript_cache_dir is None:
        return get_transcript(mkv_path, whisper_model, whisper_duration, whisper_skip, no_whisper, initial_prompt)

    key = _transcript_cache_key(mkv_path, whisper_model, whisper_duration, whisper_skip, initial_prompt)
    cache_path = transcript_cache_dir / f"{key}.json"
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            transcript, duration = data["transcript"], data["duration"]
            log.info(f"  Using cached transcript ({len((transcript or '').split())} words)")
            return transcript, duration
        except (OSError, json.JSONDecodeError, KeyError):
            pass  # corrupt/partial cache entry — fall through and regenerate it

    transcript, duration = get_transcript(
        mkv_path, whisper_model, whisper_duration, whisper_skip, no_whisper, initial_prompt
    )
    if transcript is not None:
        try:
            transcript_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({"file": mkv_path.name, "transcript": transcript, "duration": duration}),
                encoding="utf-8",
            )
        except OSError:
            pass  # caching is an optimization, not a correctness requirement
    return transcript, duration


# ---------------------------------------------------------------------------
# 8. TVMaze — episode metadata
# ---------------------------------------------------------------------------
def tvmaze_search_shows(show_name: str) -> list[dict]:
    """Return all TVMaze search results for show_name (up to 10), sorted by relevance."""
    try:
        r = requests.get(
            f"{TVMAZE_BASE}/search/shows",
            params={"q": show_name},
            timeout=10,
        )
        r.raise_for_status()
        return [entry["show"] for entry in r.json()]
    except Exception as exc:
        log.error(f"TVMaze search failed: {exc}")
        return []


def _format_show_line(show: dict) -> str:
    year = (show.get("premiered") or "")[:4]
    network = (show.get("network") or {}).get("name") or \
              (show.get("webChannel") or {}).get("name") or "?"
    genres = ", ".join((show.get("genres") or [])[:2])
    year_str = f" ({year})" if year else ""
    genre_str = f"  [{genres}]" if genres else ""
    return f"{show['name']}{year_str} — {network}{genre_str}"


def prompt_show_selection(show_name: str, candidates: list[dict], auto: bool = False) -> Optional[dict]:
    """
    Show a numbered list of TVMaze candidates and ask the user to confirm the right one.
    Offers a 'search again' option in case the folder name didn't surface the right show.
    With auto=True (or when stdin is not a TTY), silently takes the first result.
    Returns the selected show dict, or None if the user aborts.
    """
    if not candidates:
        return None

    if auto or not sys.stdin.isatty():
        show = candidates[0]
        log.info(f"TVMaze: auto-selected {_format_show_line(show)}  (use --show to override)")
        return show

    current_candidates = candidates
    current_query = show_name

    while True:
        print()
        if current_candidates:
            print(f"TVMaze results for {current_query!r}:")
            print()
            for i, show in enumerate(current_candidates, 1):
                print(f"  [{i}] {_format_show_line(show)}")
        else:
            print(f"  No results found for {current_query!r}.")
        print()
        n = len(current_candidates)
        prompt = f"  Enter 1-{n} to select, S to search again, or 0 to abort: " if n else \
                 "  Enter S to search again or 0 to abort: "
        try:
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if raw == "0":
            return None

        if raw.upper() == "S":
            try:
                new_query = input("  Search query: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return None
            if new_query:
                current_query = new_query
                current_candidates = tvmaze_search_shows(current_query)
            continue

        if raw.isdigit() and 1 <= int(raw) <= n:
            chosen = current_candidates[int(raw) - 1]
            print(f"  Selected: {_format_show_line(chosen)}")
            print()
            return chosen

        print(f"  Invalid input — enter a number between 1 and {n}, S, or 0.")


def tvmaze_get_episodes(show_id: int, season: Optional[int]) -> list[dict]:
    """Fetch all episodes for a show, optionally filtered to a single season."""
    try:
        r = requests.get(
            f"{TVMAZE_BASE}/shows/{show_id}/episodes",
            timeout=10,
        )
        r.raise_for_status()
        episodes: list[dict] = r.json()
        if season is not None:
            episodes = [e for e in episodes if e["season"] == season]
        return episodes
    except Exception as exc:
        log.error(f"TVMaze episodes fetch failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# 8b. TMDB — richer episode summaries (optional; requires TMDB_API_KEY)
# ---------------------------------------------------------------------------
def tmdb_search_show(show_name: str, api_key: str) -> Optional[dict]:
    """Search TMDB for a TV show. Returns {id, name} or None."""
    try:
        r = requests.get(
            f"{TMDB_BASE}/search/tv",
            params={"query": show_name, "api_key": api_key, "language": "en-US"},
            timeout=10,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None
        show = results[0]
        return {"id": show["id"], "name": show.get("name") or show.get("original_name", "")}
    except Exception as exc:
        log.debug(f"TMDB show search failed: {exc}")
        return None


def tmdb_get_episodes(show_id: int, season: Optional[int], api_key: str) -> list[dict]:
    """
    Fetch episodes from TMDB, normalized to the same dict shape used by TVMaze:
    {season, number, name, summary}. TMDB summaries are plain text (no HTML).
    When season is None, fetches all seasons by querying show details first.
    """
    try:
        if season is not None:
            seasons_to_fetch = [season]
        else:
            r = requests.get(
                f"{TMDB_BASE}/tv/{show_id}",
                params={"api_key": api_key, "language": "en-US"},
                timeout=10,
            )
            r.raise_for_status()
            n_seasons = r.json().get("number_of_seasons", 0)
            seasons_to_fetch = list(range(1, n_seasons + 1))

        episodes: list[dict] = []
        for s in seasons_to_fetch:
            r = requests.get(
                f"{TMDB_BASE}/tv/{show_id}/season/{s}",
                params={"api_key": api_key, "language": "en-US"},
                timeout=10,
            )
            if r.status_code == 404:
                continue
            r.raise_for_status()
            for ep in r.json().get("episodes", []):
                episodes.append({
                    "season": ep["season_number"],
                    "number": ep["episode_number"],
                    "name": ep.get("name") or "",
                    "summary": ep.get("overview") or "",
                    "runtime": ep.get("runtime"),
                })
        return episodes
    except Exception as exc:
        log.debug(f"TMDB episodes fetch failed: {exc}")
        return []


def merge_episode_summaries(primary: list[dict], tmdb: list[dict]) -> list[dict]:
    """
    Merge TMDB summaries into the primary episode list. Prefers the TMDB summary
    when it is longer than the TVMaze one (TMDB tends to be more detailed and is
    already plain text, whereas TVMaze summaries contain HTML tags).
    Episodes present only in primary are kept as-is; TMDB-only episodes are ignored
    since TVMaze is authoritative for season/episode numbering.
    """
    _html_re = re.compile(r"<[^>]+>")
    tmdb_index = {(ep["season"], ep["number"]): ep.get("summary") or "" for ep in tmdb}

    merged = []
    for ep in primary:
        key = (ep["season"], ep["number"])
        tmdb_summary = tmdb_index.get(key, "")
        current_clean = _html_re.sub("", ep.get("summary") or "").strip()
        if len(tmdb_summary) > len(current_clean):
            ep = {**ep, "summary": tmdb_summary}
        merged.append(ep)
    return merged


# ---------------------------------------------------------------------------
# 9. OpenSubtitles integration
# ---------------------------------------------------------------------------
_os_token: Optional[str] = None  # JWT cached for the session


def _os_get_headers(api_key: str, token: Optional[str] = None) -> dict:
    """Headers for GET requests — no Content-Type (no body)."""
    app = os.environ.get("OPENSUBTITLES_APP_NAME", "tv-episode-identifier v1.0")
    h = {"Api-Key": api_key, "User-Agent": app}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _os_post_headers(api_key: str, token: Optional[str] = None) -> dict:
    """Headers for POST requests — includes Content-Type."""
    return {**_os_get_headers(api_key, token), "Content-Type": "application/json"}


def os_login(api_key: str, username: str, password: str) -> Optional[str]:
    global _os_token
    try:
        r = requests.post(
            f"{OPENSUBTITLES_BASE}/login",
            json={"username": username, "password": password},
            headers=_os_post_headers(api_key),
            timeout=10,
        )
        if not r.ok:
            try:
                body = r.json()
            except Exception:
                body = r.text[:200]
            log.error(
                f"OpenSubtitles login failed: {r.status_code} {r.reason}\n"
                f"  Response: {body}\n"
                f"  Check that your API key is correct at "
                f"https://www.opensubtitles.com/consumers"
            )
            return None
        token = r.json().get("token")
        _os_token = token
        log.info("OpenSubtitles: authenticated")
        return token
    except Exception as exc:
        log.error(f"OpenSubtitles login failed: {exc}")
        return None


def _os_pick_best_file_ids(data: list, limit: int) -> list[int]:
    """From a subtitle search result list, return up to `limit` file_ids, most-downloaded first."""
    if not data:
        return []
    ranked = sorted(
        data,
        key=lambda item: item.get("attributes", {}).get("download_count", 0),
        reverse=True,
    )
    file_ids: list[int] = []
    for item in ranked:
        files = item.get("attributes", {}).get("files", [])
        if files:
            file_ids.append(files[0]["file_id"])
        if len(file_ids) >= limit:
            break
    return file_ids


def os_search_subtitles(
    api_key: str,
    token: str,
    show_name: str,
    season: int,
    episode: int,
    limit: int = 1,
) -> list[int]:
    """
    Search for an English subtitle and return up to `limit` file_ids, most-downloaded
    first. Tries a focused search first (type=episode + season/episode numbers), then
    falls back to a broader query if that returns nothing.
    """
    base_params = {
        "query": show_name,
        "season_number": season,
        "episode_number": episode,
        "languages": "en",
    }

    for attempt, params in enumerate([
        {**base_params, "type": "episode"},   # focused: TV episodes only
        base_params,                           # broader: any content type
    ]):
        try:
            r = requests.get(
                f"{OPENSUBTITLES_BASE}/subtitles",
                params=params,
                headers=_os_get_headers(api_key, token),
                timeout=10,
            )
            r.raise_for_status()
            payload = r.json()
            data = payload.get("data", [])
            if data:
                file_ids = _os_pick_best_file_ids(data, limit)
                if file_ids:
                    return file_ids
            if attempt == 0 and not data:
                log.debug(
                    f"  S{season:02d}E{episode:02d}: focused search empty "
                    f"(total_count={payload.get('total_count', '?')}), trying broader query"
                )
        except Exception as exc:
            log.debug(f"  OS search error (attempt {attempt + 1}) S{season:02d}E{episode:02d}: {exc}")

    log.debug(f"  S{season:02d}E{episode:02d}: no subtitle found on OpenSubtitles")
    return []


def os_download_subtitle(
    api_key: str,
    token: str,
    file_id: int,
    dest: Path,
) -> bool:
    """Download a subtitle by file_id to dest. Returns True on success."""
    try:
        r = requests.post(
            f"{OPENSUBTITLES_BASE}/download",
            json={"file_id": file_id},
            headers=_os_post_headers(api_key, token),
            timeout=10,
        )
        r.raise_for_status()
        payload = r.json()
        link = payload.get("link")
        remaining = payload.get("remaining", "?")
        log.debug(f"  Download link obtained (daily quota remaining: {remaining})")
        if not link:
            return False
        content_r = requests.get(link, timeout=30)
        content_r.raise_for_status()
        dest.write_bytes(content_r.content)
        return True
    except Exception as exc:
        log.debug(f"  OS download error (file_id={file_id}): {exc}")
        return False


def get_reference_subtitle(
    api_key: str,
    token: str,
    show_name: str,
    season: int,
    episode: int,
    cache_dir: Path,
) -> tuple[Optional[Path], str]:
    """
    Return (path, source) for a reference SRT where source is 'cache', 'download',
    or 'unavailable'. Downloads from OpenSubtitles only when not already cached.
    """
    show_safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", show_name)
    cache_path = cache_dir / show_safe / f"S{season:02d}E{episode:02d}.srt"

    if cache_path.exists() and cache_path.stat().st_size > 50:
        return cache_path, "cache"

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    file_ids = os_search_subtitles(api_key, token, show_name, season, episode, limit=1)
    time.sleep(OS_RATE_DELAY)

    if not file_ids:
        return None, "unavailable"

    success = os_download_subtitle(api_key, token, file_ids[0], cache_path)
    time.sleep(OS_RATE_DELAY)

    if success:
        return cache_path, "download"
    return None, "unavailable"


def get_alternate_reference_subtitles(
    api_key: str,
    token: str,
    show_name: str,
    season: int,
    episode: int,
    cache_dir: Path,
    max_alternates: int,
) -> list[Path]:
    """
    Fetch up to `max_alternates` additional community subtitle uploads for one episode,
    beyond the primary (most-downloaded) one already cached by get_reference_subtitle.
    Community subtitles vary in transcription quality — scoring a transcript against
    several candidates and keeping the best match reduces false negatives caused by one
    noisy upload. Costs one OpenSubtitles download quota per alternate actually fetched.
    """
    if max_alternates <= 0:
        return []

    show_safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", show_name)
    base = cache_dir / show_safe / f"S{season:02d}E{episode:02d}"

    cached = [
        p for n in range(2, max_alternates + 2)
        if (p := base.with_name(f"{base.name}.alt{n}.srt")).exists() and p.stat().st_size > 50
    ]
    if len(cached) >= max_alternates:
        return cached[:max_alternates]

    # Search sorts by download_count, which is stable within a session, so index 0
    # should be the same file already cached as the primary reference — skip it.
    file_ids = os_search_subtitles(
        api_key, token, show_name, season, episode, limit=max_alternates + 1
    )
    time.sleep(OS_RATE_DELAY)

    fetched = list(cached)
    for slot, file_id in enumerate(file_ids[1:], start=2):
        if len(fetched) >= max_alternates:
            break
        dest = base.with_name(f"{base.name}.alt{slot}.srt")
        if dest.exists() and dest.stat().st_size > 50:
            fetched.append(dest)
            continue
        if os_download_subtitle(api_key, token, file_id, dest):
            fetched.append(dest)
        time.sleep(OS_RATE_DELAY)

    return fetched


def prefetch_reference_subtitles(
    api_key: str,
    token: str,
    show_name: str,
    episodes: list[dict],
    cache_dir: Path,
) -> dict[str, Path]:
    """
    Ensure reference subtitles exist for all candidate episodes (from cache or
    OpenSubtitles download). Returns mapping of 'S01E01' → SRT path.
    """
    refs: dict[str, Path] = {}
    counts = {"cache": 0, "download": 0, "unavailable": 0}

    needs_download = []
    for ep in episodes:
        show_safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", show_name)
        p = cache_dir / show_safe / f"S{ep['season']:02d}E{ep['number']:02d}.srt"
        if p.exists() and p.stat().st_size > 50:
            needs_download.append(False)
        else:
            needs_download.append(True)

    cached_count = needs_download.count(False)
    download_count = needs_download.count(True)

    if cached_count and download_count:
        log.info(
            f"Reference subtitles: {cached_count} cached locally, "
            f"{download_count} to download from OpenSubtitles"
        )
    elif cached_count:
        log.info(f"Reference subtitles: all {cached_count} loaded from local cache")
    else:
        log.info(f"Reference subtitles: downloading {download_count} from OpenSubtitles...")

    for ep in episodes:
        season = ep["season"]
        ep_num = ep["number"]
        key = f"S{season:02d}E{ep_num:02d}"
        srt, source = get_reference_subtitle(api_key, token, show_name, season, ep_num, cache_dir)
        counts[source] += 1
        if srt:
            refs[key] = srt
            if source == "download":
                log.info(f"  Downloaded: {key}")
        else:
            log.debug(f"  No subtitle: {key}")

    parts = []
    if counts["cache"]:
        parts.append(f"{counts['cache']} from cache")
    if counts["download"]:
        parts.append(f"{counts['download']} downloaded")
    if counts["unavailable"]:
        parts.append(f"{counts['unavailable']} unavailable")
    log.info(f"Reference subtitles ready: {len(refs)}/{len(episodes)}  ({', '.join(parts)})")
    return refs


# ---------------------------------------------------------------------------
# 10. Episode matching
# ---------------------------------------------------------------------------
_CLEAN_RE = re.compile(r"[^a-z0-9\s]")


def _content_words(text: str, limit: int = SAMPLE_WORDS) -> frozenset[str]:
    """
    Return the set of content words from `text`: lowercase, punctuation stripped,
    stop words removed. Using a set (order-independent) avoids penalising
    differences in line segmentation between a Whisper transcript and an SRT file.
    """
    cleaned = _CLEAN_RE.sub(" ", text.lower())
    words = cleaned.split()[:limit]
    return frozenset(w for w in words if w not in _STOP_WORDS and len(w) > 1)


def _fuzzy_bonus_matches(exclusive_a: list[str], exclusive_b: list[str]) -> int:
    """
    Count near-identical word pairs between two already-disjoint vocabularies —
    e.g. 'recieve'/'receive' or 'l'/'1' OCR noise in community subtitles. Greedy
    highest-similarity-first pairing so no word is credited toward more than one match.
    """
    if not exclusive_a or not exclusive_b:
        return 0

    matrix = process.cdist(
        exclusive_a, exclusive_b,
        scorer=fuzz.ratio, score_cutoff=FUZZY_SCORE_CUTOFF, workers=-1,
    )
    pairs = np.argwhere(matrix >= FUZZY_SCORE_CUTOFF)
    if pairs.size == 0:
        return 0

    scores = matrix[pairs[:, 0], pairs[:, 1]]
    order = np.argsort(-scores)

    used_a: set[int] = set()
    used_b: set[int] = set()
    bonus = 0
    for idx in order:
        i, j = int(pairs[idx, 0]), int(pairs[idx, 1])
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        bonus += 1
    return bonus


def _duration_bonus(video_seconds: float, episode_runtime_minutes) -> float:
    """
    Additive score bonus when file duration is close to the episode's listed runtime.
    Uses a tiered scale: full bonus within DURATION_TIGHT_MIN, half bonus out to
    DURATION_LOOSE_MIN, zero beyond that. Returns 0.0 when either value is absent.
    """
    if not episode_runtime_minutes or not video_seconds:
        return 0.0
    delta = abs(video_seconds / 60.0 - episode_runtime_minutes)
    if delta <= DURATION_TIGHT_MIN:
        return DURATION_BONUS_MAX
    if delta <= DURATION_LOOSE_MIN:
        return DURATION_BONUS_MAX * 0.5
    return 0.0


def match_score(transcript_text: str, reference_text: str) -> float:
    """
    Fuzzy-tolerant Jaccard similarity on content-word sets.
    Order-independent, so it handles the difference in word flow between a
    Whisper transcript (continuous prose) and a reference SRT (line fragments).
    Exact word matches count directly; words that almost-but-not-quite match are
    given credit too, since community subtitles frequently contain typos and OCR
    artifacts that would otherwise cause a genuinely correct match to score low.
    Returns 0–1 where 1 = identical vocabulary.
    """
    a = _content_words(transcript_text)
    b = _content_words(reference_text)
    union = len(a | b)
    if not union:
        return 0.0

    exact = len(a & b)
    bonus = _fuzzy_bonus_matches(list(a - b), list(b - a))
    return min((exact + bonus) / union, 1.0)


def _score_episodes(
    transcript: str,
    episodes: list[dict],
    reference_subtitles: dict[str, Path],
    video_seconds: float = 0.0,
) -> list[tuple[dict, float]]:
    """Score transcript against each episode's reference subtitle. Best first.
    Applies a small duration bonus when video_seconds is provided and the episode
    has a runtime field, so length-distinct episodes (pilots, finales) score
    higher. Skips episodes whose reference subtitle's time span doesn't plausibly
    match their listed runtime (see is_plausible_reference_subtitle) rather than
    scoring against what's likely mislabeled/corrupt data — this is the one place
    all real scoring happens, so the check lives here rather than being
    duplicated (and risking drifting out of sync) in every caller."""
    scores: list[tuple[dict, float]] = []

    for ep in episodes:
        key = f"S{ep['season']:02d}E{ep['number']:02d}"
        ref_path = reference_subtitles.get(key)
        if not ref_path:
            continue
        if not is_plausible_reference_subtitle(ref_path, ep.get("runtime")):
            log.debug(f"  {key}: reference subtitle implausible for its listed runtime — skipping")
            continue
        ref_text = srt_to_text(ref_path)
        score = match_score(transcript, ref_text)
        bonus = _duration_bonus(video_seconds, ep.get("runtime"))
        score = min(score + bonus, 1.0)
        scores.append((ep, score))
        log.debug(f"  text {key} jaccard={score:.4f}" + (f" (dur+{bonus:.3f})" if bonus else ""))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def _confidence_from_scores(scores: list[tuple[dict, float]]) -> tuple[Optional[dict], float, float]:
    """
    Derive (episode, confidence, raw_best_score) from sorted (episode, score) pairs.
    Confidence is the ratio-based discriminability metric (0–1).
    raw_best_score is the plain Jaccard similarity of the top candidate.
    Both are needed: confidence alone can be low even when raw_best_score is high
    (e.g. scores are clustered), in which case the text match is actually trustworthy.
    """
    if not scores:
        return None, 0.0, 0.0

    best_ep, best_score = scores[0]
    second_score = scores[1][1] if len(scores) > 1 else 0.0

    if best_score < QUALITY_FLOOR:
        return best_ep, 0.0, best_score  # scores too low to be meaningful

    # How many times better is the best match than the runner-up?
    # 5× better → ~100% confidence; 2× better → ~33% confidence.
    ratio = best_score / second_score if second_score > 0 else 10.0
    confidence = min((ratio - 1.0) / 4.0, 1.0)

    return best_ep, confidence, best_score


def _confidence_for_forced_pick(
    scores: list[tuple[dict, float]], forced_ep: dict, group_confirmed: bool = False
) -> tuple[Optional[dict], float, float]:
    """
    Like _confidence_from_scores, but the caller (a cross-file disc-order alignment,
    see align_monotonic) has already decided which episode this file should use
    instead of taking the raw top score. Confidence is still the best-vs-runner-up
    ratio from _confidence_from_scores — it's just measured against the forced pick
    rather than whatever scored highest in isolation, so the existing
    threshold/AI-fallback behavior downstream is unaffected by where the pick came
    from. Falls back to _confidence_from_scores if forced_ep isn't among the scored
    candidates (e.g. another file already claimed it in an earlier group).

    group_confirmed (see _alignment_is_conclusive) means this file was part of a
    multi-file group whose alignment resolved end-to-end with no gaps — independent
    evidence beyond this file's own ratio, applied via GROUP_CONFIRMED_FLOOR below.
    """
    if not scores:
        return None, 0.0, 0.0

    fkey = (forced_ep["season"], forced_ep["number"])
    forced_score = next((s for ep, s in scores if (ep["season"], ep["number"]) == fkey), None)
    if forced_score is None:
        return _confidence_from_scores(scores)

    if forced_score < QUALITY_FLOOR:
        return forced_ep, 0.0, forced_score

    # Unlike _confidence_from_scores, forced_score isn't guaranteed to be the max
    # in `scores` — the alignment can pick an episode that isn't the raw top score
    # when order says it should win anyway. When some other candidate actually
    # scored higher on text alone, ratio < 1 and confidence is clamped to 0 rather
    # than going negative: the order constraint may still be right, but text alone
    # doesn't vouch for it, so it shouldn't auto-clear the threshold on that basis —
    # it should fall through to alt-candidate refinement / AI fallback instead.
    second_score = max((s for ep, s in scores if (ep["season"], ep["number"]) != fkey), default=0.0)
    ratio = forced_score / second_score if second_score > 0 else 10.0
    confidence = max(0.0, min((ratio - 1.0) / 4.0, 1.0))

    # Group corroboration only rescues a pick the file's own text already agrees
    # with (confidence > 0 — never the ratio<1 contradicted case above) and that
    # clears the existing "text-supported enough" bar, so a near-empty transcript
    # can't ride the rest of the group's coattails to a false confident match.
    if group_confirmed and confidence > 0.0 and forced_score >= JACCARD_AI_SKIP_FLOOR:
        confidence = max(confidence, GROUP_CONFIRMED_FLOOR)

    return forced_ep, confidence, forced_score


def align_monotonic(matrix: list[list[float]], floor: float = QUALITY_FLOOR) -> list[Optional[int]]:
    """
    Assign each row (a file, in physical rip order) to at most one column (a
    candidate episode, ascending order) so assigned columns strictly increase with
    row index, maximizing total assigned score. Standard global sequence alignment
    (Needleman-Wunsch style) with a gap allowed on either side: a file can be left
    unassigned (nothing here fits well) and an episode can be skipped (not present
    among these files — e.g. a disc that starts mid-season).

    This encodes the physical constraint that files ripped off one disc appear in
    the same order as the episodes they contain. Even when one file's Jaccard score
    alone doesn't clearly point to the right episode, its neighbors' scores plus
    the order constraint usually do — whereas matching each file independently
    lets an earlier file steal a later file's correct episode, or two files drift
    onto the same wrong one.

    `floor` is subtracted from every score before comparison, so a match only beats
    leaving a file unassigned once it clears QUALITY_FLOOR — otherwise a genuinely
    bad file (an extra, a failed transcription) would get forced onto whatever
    episode is left just to preserve order.

    Returns a list the length of matrix: the chosen column index per row, or None
    where the row was left unassigned.
    """
    m = len(matrix)
    n = len(matrix[0]) if m else 0
    if m == 0 or n == 0:
        return [None] * m

    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = dp[i - 1][j - 1] + (matrix[i - 1][j - 1] - floor)
            dp[i][j] = max(match, dp[i - 1][j], dp[i][j - 1])

    picks: list[Optional[int]] = [None] * m
    i, j = m, n
    while i > 0 and j > 0:
        match = dp[i - 1][j - 1] + (matrix[i - 1][j - 1] - floor)
        if dp[i][j] == match:
            picks[i - 1] = j - 1
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j]:
            i -= 1
        else:
            j -= 1
    return picks


def _alignment_is_conclusive(picks: list[Optional[int]], candidate_episodes: list[dict]) -> bool:
    """
    True when align_monotonic placed every file in a multi-file group onto one
    unbroken run of consecutive episode numbers, with none left unassigned.
    This is independent structural corroboration on top of any single file's
    own text-match ratio (see GROUP_CONFIRMED_FLOOR): several files' physical
    rip order lining up end-to-end with the season's episode order is strong
    evidence a mismatched file is unlikely to produce by chance. A single-file
    "group" carries none of that corroboration (there's nothing to line up
    against), and neither does a group with an unassigned row or a skipped
    episode between two assigned ones.
    """
    if len(picks) < 2 or any(p is None for p in picks):
        return False
    numbers = [candidate_episodes[p]["number"] for p in picks]
    return all(b == a + 1 for a, b in zip(numbers, numbers[1:]))


def match_by_text(
    transcript: str,
    episodes: list[dict],
    reference_subtitles: dict[str, Path],
) -> tuple[Optional[dict], float, float]:
    """
    Compare transcript against available reference subtitles using fuzzy-tolerant
    Jaccard similarity. Returns (episode, confidence, raw_jaccard).
    """
    return _confidence_from_scores(_score_episodes(transcript, episodes, reference_subtitles))


def refine_top_candidates_with_alternates(
    transcript: str,
    scores: list[tuple[dict, float]],
    show_name: str,
    api_key: str,
    token: str,
    cache_dir: Path,
    max_alternates: int,
    top_k: int = ALT_CANDIDATE_TOP_K,
    video_seconds: float = 0.0,
    forced_ep: Optional[dict] = None,
    group_confirmed: bool = False,
) -> tuple[Optional[dict], float, float, list[tuple[dict, float]]]:
    """
    Re-score the top-K candidate episodes against alternate community subtitle
    uploads, keeping each episode's best score across all candidates tried. Only
    called when the initial single-candidate confidence is below threshold — this
    keeps OpenSubtitles download quota usage limited to genuinely ambiguous files.
    Returns (episode, confidence, raw_jaccard, scores) — the trailing scores list
    is the full per-episode ranking with any alternate-subtitle improvements
    folded in, sorted descending, so callers can report an accurate top-N
    breakdown even when the winning pick didn't change from the initial pass.

    When forced_ep is given (a disc-order alignment pick — see align_monotonic —
    that may not be the raw top score), it's added to the refined set even if it
    fell outside top_k, so a low-scoring-but-order-correct episode still gets a
    chance at a cleaner alternate subtitle before its confidence is judged.
    """
    if max_alternates <= 0 or not scores:
        picked = (
            _confidence_for_forced_pick(scores, forced_ep, group_confirmed)
            if forced_ep else _confidence_from_scores(scores)
        )
        return (*picked, scores)

    refined = list(scores)
    indices = list(range(min(top_k, len(refined))))
    if forced_ep is not None:
        fkey = (forced_ep["season"], forced_ep["number"])
        if not any((refined[i][0]["season"], refined[i][0]["number"]) == fkey for i in indices):
            forced_idx = next(
                (i for i, (ep, _) in enumerate(refined) if (ep["season"], ep["number"]) == fkey), None
            )
            if forced_idx is not None:
                indices.append(forced_idx)

    for i in indices:
        ep, score = refined[i]
        season, number = ep["season"], ep["number"]
        key = f"S{season:02d}E{number:02d}"
        alt_paths = get_alternate_reference_subtitles(
            api_key, token, show_name, season, number, cache_dir, max_alternates
        )
        best_score = score
        for alt_path in alt_paths:
            if not is_plausible_reference_subtitle(alt_path, ep.get("runtime")):
                log.debug(f"  {key}: alternate subtitle's time span looks implausible — skipping it")
                continue
            alt_text = srt_to_text(alt_path)
            alt_score = min(
                match_score(transcript, alt_text) + _duration_bonus(video_seconds, ep.get("runtime")),
                1.0,
            )
            log.debug(f"  text {key} alt jaccard={alt_score:.4f}")
            best_score = max(best_score, alt_score)
        if best_score > score:
            log.info(f"  {key}: alternate subtitle improved score {score:.4f} → {best_score:.4f}")
        refined[i] = (ep, best_score)

    refined.sort(key=lambda x: x[1], reverse=True)
    picked = (
        _confidence_for_forced_pick(refined, forced_ep, group_confirmed)
        if forced_ep else _confidence_from_scores(refined)
    )
    return (*picked, refined)


_AI_PROMPT_SUMMARY_CHARS = 800    # generous vs. the old 250 — accuracy over prompt size
_AI_PROMPT_TRANSCRIPT_WORDS = 6000  # bounded (not the full multi-thousand-word transcript
                                     # unconditionally) so this doesn't blow out a small local
                                     # model's context window via --local-ai; still covers the
                                     # vast majority of a full episode's dialogue, and AI
                                     # fallback only ever runs on a small, already-ambiguous
                                     # subset of files, so the extra tokens are cheap in practice


def _build_episode_match_prompt(
    transcript: str, episodes: list[dict], show_name: str
) -> str:
    ep_lines = []
    for ep in episodes:
        s, e = ep["season"], ep["number"]
        title = ep.get("name") or ""
        summary = re.sub(r"<[^>]+>", "", ep.get("summary") or "").strip()
        ep_lines.append(f"S{s:02d}E{e:02d} - {title}: {summary[:_AI_PROMPT_SUMMARY_CHARS]}")

    sample = " ".join(transcript.split()[:_AI_PROMPT_TRANSCRIPT_WORDS])

    return (
        f'I have a transcript from an episode of "{show_name}".\n\n'
        f"Episode list:\n" + "\n".join(ep_lines) +
        f"\n\nTranscript:\n{sample}\n\n"
        "Which episode does this transcript match? "
        "Reply with JSON only, no other text:\n"
        '{"episode": "S01E01", "confidence": 0.95, "reasoning": "one sentence"}\n'
        'If you cannot determine the episode: {"episode": null, "confidence": 0.0, "reasoning": "reason"}'
    )


def _parse_ai_episode_response(
    raw: str, episodes: list[dict], label: str
) -> tuple[Optional[dict], float]:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        log.debug(f"  {label}: unparseable response: {raw[:100]}")
        return None, 0.0
    data = json.loads(m.group())
    ep_code = data.get("episode")
    confidence = float(data.get("confidence", 0.0))
    reasoning = data.get("reasoning", "")

    if not ep_code:
        log.info(f"  {label}: no match — {reasoning}")
        return None, 0.0

    match = re.match(r"S(\d+)E(\d+)", ep_code, re.IGNORECASE)
    if not match:
        return None, 0.0
    s_num, e_num = int(match.group(1)), int(match.group(2))

    for ep in episodes:
        if ep["season"] == s_num and ep["number"] == e_num:
            log.info(
                f"  {label}: {ep_code} - {ep.get('name', '')} "
                f"({confidence:.0%}) — {reasoning}"
            )
            return ep, confidence

    log.debug(f"  {label} identified {ep_code} but it's not in the episode list")
    return None, 0.0


def match_with_claude(
    transcript: str,
    episodes: list[dict],
    show_name: str,
    api_key: str,
) -> tuple[Optional[dict], float]:
    """Ask Claude to identify the episode. No reference subtitles required."""
    try:
        import anthropic
    except ImportError:
        log.warning("anthropic package not installed — run: pip install anthropic")
        return None, 0.0

    prompt = _build_episode_match_prompt(transcript, episodes, show_name)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        return _parse_ai_episode_response(raw, episodes, "Claude")
    except Exception as exc:
        log.error(f"  Claude matching failed: {exc}")
        return None, 0.0


def match_with_local_ai(
    transcript: str,
    episodes: list[dict],
    show_name: str,
    model: str,
    base_url: str,
) -> tuple[Optional[dict], float]:
    """Ask a local Ollama/LM Studio model to identify the episode.

    Both servers expose an OpenAI-compatible /v1/chat/completions endpoint,
    so we talk to them directly with requests — no extra packages needed.
    """
    prompt = _build_episode_match_prompt(transcript, episodes, show_name)
    url = base_url.rstrip("/") + "/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.0,
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        return _parse_ai_episode_response(raw, episodes, f"local-ai({model})")
    except requests.exceptions.ConnectionError:
        log.error(
            f"  local-ai: could not connect to {base_url}. "
            "Is Ollama/LM Studio running? (ollama serve)"
        )
        return None, 0.0
    except Exception as exc:
        log.error(f"  local-ai matching failed: {exc}")
        return None, 0.0


# ---------------------------------------------------------------------------
# 11. Ollama lifecycle helpers
# ---------------------------------------------------------------------------
def _ollama_is_running(base_url: str) -> bool:
    try:
        r = requests.get(base_url.rstrip("/") + "/", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def ollama_start(base_url: str) -> Optional[subprocess.Popen]:
    """Start `ollama serve` if it isn't already running.

    Returns the Popen handle if we started it (caller must stop it), or None
    if it was already running (caller must leave it alone).
    Only attempted when the base URL looks like a local Ollama instance.
    """
    if not base_url.startswith("http://localhost") and not base_url.startswith("http://127.0.0.1"):
        return None  # remote or LM Studio — not our responsibility

    if _ollama_is_running(base_url):
        log.info("Ollama: already running")
        return None

    import shutil
    if not shutil.which("ollama"):
        log.warning("Ollama: 'ollama' not found in PATH — cannot auto-start")
        return None

    log.info("Ollama: starting 'ollama serve'...")
    proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(20):          # wait up to ~10 s
        time.sleep(0.5)
        if _ollama_is_running(base_url):
            log.info("Ollama: ready")
            return proc

    log.warning("Ollama: started but not responding after 10 s — continuing anyway")
    return proc


def ollama_stop(proc: Optional[subprocess.Popen]) -> None:
    """Terminate an Ollama process we started. No-op if proc is None."""
    if proc is None:
        return
    log.info("Ollama: stopping...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------------------------------------------------------------------------
# 12. Rename logic
# ---------------------------------------------------------------------------
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_name(name: str) -> str:
    return _UNSAFE_CHARS.sub("", name).strip(" .")


def build_confident_path(mkv_path: Path, episode: dict) -> Path:
    season = episode["season"]
    ep_num = episode["number"]
    return mkv_path.parent / f"S{season:02d}E{ep_num:02d}{mkv_path.suffix}"


def _strip_unmatched_prefix(stem: str) -> str:
    """Remove any existing UNMATCHED_ prefix so reruns don't nest them."""
    import re
    return re.sub(r"^UNMATCHED_(?:S\d{2}E\d{2}_\d+pct_|UNKNOWN_)+", "", stem)


def build_unmatched_path(
    mkv_path: Path, best_ep: Optional[dict], confidence: float
) -> Path:
    pct = int(confidence * 100)
    base = _strip_unmatched_prefix(mkv_path.stem)
    if best_ep:
        guess = f"S{best_ep['season']:02d}E{best_ep['number']:02d}"
        stem = f"UNMATCHED_{guess}_{pct}pct_{sanitize_name(base)}"
    else:
        stem = f"UNMATCHED_UNKNOWN_{sanitize_name(base)}"
    return mkv_path.parent / f"{stem}{mkv_path.suffix}"


def do_rename(
    src: Path,
    dst: Path,
    dry_run: bool,
    pending_renames: Optional[list] = None,
    processing_paths: Optional[set] = None,
) -> bool:
    """
    Rename src → dst.  Returns True if the rename happened (or was queued).

    When the target already exists AND that target is still in the processing
    queue (i.e. it will be renamed away later in this run), we avoid the conflict
    by renaming src to a unique temp name and recording the final rename in
    pending_renames so it can be applied once the slot is free.
    """
    if dst == src:
        return True
    if dst.exists():
        if pending_renames is not None and processing_paths is not None and dst in processing_paths:
            # Target will be renamed away later — park src under a temp name for now.
            tmp = src.parent / f"_tmp_{src.stem}_{id(src)}{src.suffix}"
            if not dry_run:
                src.rename(tmp)
                log.info(f"  Renamed (temp): {src.name}  →  {tmp.name}  (final: {dst.name})")
            else:
                log.info(f"  [DRY RUN] {src.name}  →  {tmp.name}  (final: {dst.name})")
            pending_renames.append((tmp, dst))
            return True
        log.warning(f"  Target already exists, skipping rename: {dst.name}")
        return False
    if dry_run:
        log.info(f"  [DRY RUN] {src.name}  →  {dst.name}")
    else:
        src.rename(dst)
        log.info(f"  Renamed: {src.name}  →  {dst.name}")
    return True


# ---------------------------------------------------------------------------
# 12. Match log writer
# ---------------------------------------------------------------------------
def write_match_log(
    log_dir: Path,
    video_path: Path,
    show_name: str,
    transcript: Optional[str],
    scores: list[tuple[dict, float]],
    ai_prompt: Optional[str],
    ai_response: Optional[str],
    final_match: Optional[dict],
    confidence: float,
    video_seconds: float = 0.0,
) -> None:
    """Write a human-readable match log for one video file to log_dir/Show/Season N/."""
    try:
        safe_show = re.sub(r'[\\/:*?"<>|]', "_", show_name).strip()
        season_num = (
            final_match["season"] if final_match
            else scores[0][0]["season"] if scores
            else None
        )
        season_label = f"Season {season_num}" if season_num is not None else "Season Unknown"
        out_dir = log_dir / safe_show / season_label
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_stem = re.sub(r"[^\w\-.]", "_", video_path.stem)
        out = out_dir / f"{safe_stem}.txt"
        with out.open("w", encoding="utf-8") as fh:
            sep = "=" * 80
            thin = "-" * 80

            fh.write(f"{sep}\n")
            fh.write(f"File:    {video_path.name}\n")
            fh.write(f"Show:    {show_name}\n")
            if video_seconds:
                ep_runtime = final_match.get("runtime") if final_match else None
                runtime_str = f"  (episode listed: {ep_runtime} min)" if ep_runtime else ""
                fh.write(f"Video:   {video_seconds / 60:.1f} min{runtime_str}\n")
            fh.write(f"Result:  ")
            if final_match:
                key = f"S{final_match['season']:02d}E{final_match['number']:02d}"
                fh.write(f"{key} - {final_match.get('name', '')}  ({confidence:.0%} confidence)\n")
            else:
                fh.write(f"UNMATCHED  ({confidence:.0%})\n")
            fh.write(f"{sep}\n\n")

            fh.write(f"TRANSCRIPT ({len((transcript or '').split())} words)\n")
            fh.write(f"{thin}\n")
            fh.write((transcript or "(none)") + "\n\n")

            fh.write(f"EPISODE SCORES — all candidates ranked by Jaccard similarity\n")
            fh.write(f"{thin}\n")
            if scores:
                for ep, score in scores:
                    key = f"S{ep['season']:02d}E{ep['number']:02d}"
                    title = ep.get("name") or ""
                    raw_summary = ep.get("summary") or ""
                    summary = re.sub(r"<[^>]+>", "", raw_summary).strip()
                    runtime = ep.get("runtime")
                    runtime_str = f"  runtime: {runtime} min" if runtime else ""
                    fh.write(f"\n{key} - {title}  (score: {score:.4f}{runtime_str})\n")
                    if summary:
                        fh.write(f"  Summary: {summary}\n")
                    else:
                        fh.write(f"  Summary: (none)\n")
            else:
                fh.write("(no scores — no reference subtitles available)\n")
            fh.write("\n")

            if ai_prompt:
                fh.write(f"AI PROMPT\n")
                fh.write(f"{thin}\n")
                fh.write(ai_prompt + "\n\n")

            if ai_response:
                fh.write(f"AI RESPONSE\n")
                fh.write(f"{thin}\n")
                fh.write(ai_response + "\n\n")

    except Exception as exc:
        log.debug(f"  match log write failed: {exc}")


# ---------------------------------------------------------------------------
# 13. Per-file processing
# ---------------------------------------------------------------------------
def process_file(
    mkv_path: Path,
    episodes: list[dict],
    show_name: str,
    reference_subtitles: dict[str, Path],
    threshold: float,
    whisper_model: str,
    whisper_duration: int,
    whisper_skip: int,
    no_whisper: bool,
    ai_api_key: Optional[str],
    local_ai_model: Optional[str],
    local_ai_url: str,
    dry_run: bool,
    os_api_key: Optional[str] = None,
    os_token: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    alt_candidates: int = 0,
    matched_this_run: Optional[set] = None,
    pending_renames: Optional[list] = None,
    processing_paths: Optional[set] = None,
    log_dir: Optional[Path] = None,
    forced_episode: Optional[dict] = None,
    precomputed_transcript: Optional[tuple[Optional[str], float]] = None,
    group_confirmed: bool = False,
) -> dict:
    result: dict = {
        "file": mkv_path.name,
        "status": "skipped",
        "matched": None,
        "confidence": 0.0,
    }

    log.info(f"\n[{mkv_path.name}]")

    # Exclude episodes already confidently matched earlier in this run so that
    # each episode can only be assigned to one file per run.
    available_eps = episodes
    if matched_this_run:
        available_eps = [
            ep for ep in episodes
            if f"S{ep['season']:02d}E{ep['number']:02d}" not in matched_this_run
        ]
        excluded = len(episodes) - len(available_eps)
        if excluded:
            log.debug(f"  Excluding {excluded} already-matched episode(s) from candidates")
        if not available_eps:
            log.info("  All episodes already matched in this run — skipping")
            result["status"] = "skipped"
            return result

    # Step 1: extract transcript
    if precomputed_transcript is not None:
        transcript, video_seconds = precomputed_transcript
    else:
        transcript, video_seconds = get_transcript(mkv_path, whisper_model, whisper_duration, whisper_skip, no_whisper)
    if transcript is None:
        result["status"] = "no_transcript"
        return result

    word_count = len(transcript.split())
    if word_count < MIN_TRANSCRIPT_WORDS:
        log.info(f"  Transcript too sparse ({word_count} words) — skipping")
        result["status"] = "sparse_transcript"
        return result

    log.info(f"  Transcript: {word_count} words")

    # Step 2a: text matching against reference subtitles
    scores = _score_episodes(transcript, available_eps, reference_subtitles, video_seconds)
    if forced_episode is not None:
        best_ep, confidence, raw_jaccard = _confidence_for_forced_pick(scores, forced_episode, group_confirmed)
        naive_best = scores[0][0] if scores else None
        if naive_best and (naive_best["season"], naive_best["number"]) != \
                (forced_episode["season"], forced_episode["number"]):
            log.info(
                f"  Disc-order alignment: independent best was "
                f"S{naive_best['season']:02d}E{naive_best['number']:02d} — "
                f"using order-consistent S{forced_episode['season']:02d}E{forced_episode['number']:02d} instead"
            )
    else:
        best_ep, confidence, raw_jaccard = _confidence_from_scores(scores)
    if best_ep:
        key = f"S{best_ep['season']:02d}E{best_ep['number']:02d}"
        log.info(
            f"  Text match: {key} - {best_ep.get('name','')}  "
            f"({confidence:.1%} conf, {raw_jaccard:.1%} Jaccard)"
        )

    # Step 2a-refine: below threshold — try alternate community subtitle uploads for
    # the top candidates before falling back to AI (cheaper and often decisive when
    # the issue is one noisy OpenSubtitles upload rather than a genuine mismatch).
    if confidence < threshold and alt_candidates > 0 and os_api_key and os_token and cache_dir:
        log.info("  Text confidence below threshold — trying alternate subtitle uploads...")
        alt_ep, alt_conf, alt_raw, scores = refine_top_candidates_with_alternates(
            transcript, scores, show_name, os_api_key, os_token, cache_dir, alt_candidates,
            video_seconds=video_seconds, forced_ep=forced_episode, group_confirmed=group_confirmed,
        )
        # Adopt the refined scores for reporting (top_scores, match logs) even when
        # the winning pick doesn't change — they're strictly better information,
        # since any candidate that got refined now shows its best score across
        # both the primary and alternate subtitle uploads tried for it.
        if alt_ep and alt_conf > confidence:
            best_ep, confidence, raw_jaccard = alt_ep, alt_conf, alt_raw

    # Step 2b: AI fallback — suppressed when the raw Jaccard score is already
    # decent, since calling AI there risks hallucination overriding a text match
    # that's basically right — as seen with llama models confidently fabricating
    # wrong episode details. This ONLY withholds AI, though: it never declares the
    # match confident on its own. A file can have a decent raw score and still be
    # genuinely ambiguous (the top two candidates are close enough that
    # ratio-confidence stays below threshold — e.g. a two-parter, or a recap
    # episode) and in that case it's left below threshold on purpose, so it lands
    # in UNMATCHED for manual review instead of being auto-renamed on a guess.
    ai_skipped_high_jaccard = False
    ai_prompt: Optional[str] = None
    ai_response: Optional[str] = None

    if confidence < threshold and raw_jaccard >= JACCARD_AI_SKIP_FLOOR:
        log.info(
            f"  Jaccard score is {raw_jaccard:.1%} but still ambiguous "
            f"({confidence:.1%} conf) — skipping AI (risk of hallucination) rather "
            f"than guessing; leaving for manual review unless alt-candidates already "
            f"resolved it"
        )
        ai_skipped_high_jaccard = True

    if not ai_skipped_high_jaccard:
        if confidence < threshold and (local_ai_model or ai_api_key):
            ai_prompt = _build_episode_match_prompt(transcript, available_eps, show_name)

        if confidence < threshold and local_ai_model:
            log.info(f"  Text confidence below threshold — trying local AI ({local_ai_model})...")
            ai_ep, ai_conf = match_with_local_ai(
                transcript, available_eps, show_name, local_ai_model, local_ai_url
            )
            if ai_ep:
                key = f"S{ai_ep['season']:02d}E{ai_ep['number']:02d}"
                ai_response = f"Matched: {key} - {ai_ep.get('name', '')} ({ai_conf:.0%})"
            else:
                ai_response = "No match"
            if ai_ep and ai_conf > confidence:
                best_ep, confidence = ai_ep, ai_conf
        elif confidence < threshold and ai_api_key:
            log.info("  Text confidence below threshold — trying Claude AI...")
            ai_ep, ai_conf = match_with_claude(transcript, available_eps, show_name, ai_api_key)
            if ai_ep:
                key = f"S{ai_ep['season']:02d}E{ai_ep['number']:02d}"
                ai_response = f"Matched: {key} - {ai_ep.get('name', '')} ({ai_conf:.0%})"
            else:
                ai_response = "No match"
            if ai_ep and ai_conf > confidence:
                best_ep, confidence = ai_ep, ai_conf

    result["confidence"] = confidence
    result["top_scores"] = [
        (f"S{ep['season']:02d}E{ep['number']:02d}", ep.get("name") or "", s)
        for ep, s in scores[:3]
    ]
    if best_ep:
        key = f"S{best_ep['season']:02d}E{best_ep['number']:02d}"
        title = best_ep.get("name") or ""
        result["matched"] = f"{key} - {title}" if title else key
        log.info(f"  Best match: {result['matched']}  ({confidence:.1%} confidence)")

    # Step 3: rename or mark unmatched
    if confidence >= threshold and best_ep:
        target = build_confident_path(mkv_path, best_ep)
        renamed = do_rename(mkv_path, target, dry_run, pending_renames, processing_paths)
        result["status"] = "renamed" if renamed else "conflict"
        if renamed and matched_this_run is not None:
            ep_key = f"S{best_ep['season']:02d}E{best_ep['number']:02d}"
            matched_this_run.add(ep_key)
    else:
        log.info(f"  Below threshold ({confidence:.1%} < {threshold:.1%}) — marking UNMATCHED")
        target = build_unmatched_path(mkv_path, best_ep, confidence)
        do_rename(mkv_path, target, dry_run)
        result["status"] = "unmatched"

    if log_dir is not None:
        write_match_log(
            log_dir=log_dir,
            video_path=mkv_path,
            show_name=show_name,
            transcript=transcript,
            scores=scores,
            ai_prompt=ai_prompt,
            ai_response=ai_response,
            final_match=best_ep,
            confidence=confidence,
            video_seconds=video_seconds,
        )

    return result


# ---------------------------------------------------------------------------
# 13b. Disc/rip-order group processing
# ---------------------------------------------------------------------------
def process_file_group(
    files: list[Path],
    candidate_episodes: list[dict],
    reference_subtitles: dict[str, Path],
    show_name: str,
    args: argparse.Namespace,
    ai_api_key: Optional[str],
    local_ai_model: Optional[str],
    os_api_key: Optional[str],
    cache_dir: Optional[Path],
    matched_this_run: set,
    pending_renames: list,
    processing_paths: set,
    log_dir: Optional[Path],
    os_token: Optional[str] = None,
    per_file_windows: Optional[list[tuple[int, int]]] = None,
) -> list[dict]:
    """
    Process a set of files expected to be in physical rip order (one disc, or an
    entire season when no disc boundaries were found) and therefore expected to
    map onto increasing episode numbers. Extracts every file's transcript once,
    scores all of them against every candidate episode, and uses align_monotonic
    to find the order-preserving assignment with the highest total score — rather
    than letting each file independently claim whichever episode scores highest
    for it alone, which lets an earlier file steal a later file's correct episode,
    or two files drift onto the same wrong one.

    Maximizing total score across the whole group has its own failure mode: it can
    in principle pull an earlier file off an excellent, unambiguous match to free
    that episode up for a later file with an even higher score there (e.g. a recap
    episode that quotes heavily from an earlier one). per_file_windows guards
    against that — when given (one (min, max) episode-number pair per file, same
    order as `files`), a file can never be assigned outside its own window no
    matter how the group-wide total would improve, so alignment can only resolve
    ambiguity *within* each file's own plausible range, never reassign it further
    out than its own guess already allowed.

    Files the alignment can't confidently place (every candidate at/below
    QUALITY_FLOOR) get no forced pick and fall through process_file's normal
    independent-match / AI-fallback path, same as before this existed.

    When the alignment resolves the whole group onto one unbroken run of
    consecutive episodes with every file placed (see _alignment_is_conclusive),
    that agreement is passed to every file in the group as group_confirmed,
    granting a confidence floor (GROUP_CONFIRMED_FLOOR) on top of each file's own
    ratio-confidence — several files' rip order lining up end-to-end with the
    season's order is real evidence a single file's ratio score can't capture.
    """
    candidate_episodes = sorted(candidate_episodes, key=lambda ep: ep["number"])
    whisper_duration = 0 if args.full_scan else args.whisper_duration
    initial_prompt = build_whisper_initial_prompt(candidate_episodes)
    transcript_cache_dir = cache_dir.parent / ".transcript_cache" if cache_dir else None
    transcripts = [
        get_transcript_cached(
            f, args.whisper_model, whisper_duration, args.whisper_skip, args.no_whisper,
            transcript_cache_dir, initial_prompt=initial_prompt,
        )
        for f in files
    ]

    # Score each file against every candidate via _score_episodes — the same
    # function process_file itself uses for the real decision — rather than
    # duplicating the scoring/plausibility logic here. Keeping it in one place
    # means the alignment matrix and the final per-file match can never fall out
    # of sync with each other (e.g. the matrix silently excluding a reference
    # subtitle as implausible while process_file's own scoring still trusted it).
    matrix: list[list[float]] = []
    for row_idx, (transcript, video_seconds) in enumerate(transcripts):
        window = per_file_windows[row_idx] if per_file_windows else None
        if transcript is None:
            matrix.append([0.0] * len(candidate_episodes))
            continue
        file_scores = {
            (ep["season"], ep["number"]): score
            for ep, score in _score_episodes(transcript, candidate_episodes, reference_subtitles, video_seconds)
        }
        row = []
        for ep in candidate_episodes:
            if window is not None and not (window[0] <= ep["number"] <= window[1]):
                row.append(-1.0)  # hard-excluded: outside this file's own plausible range
                continue
            row.append(file_scores.get((ep["season"], ep["number"]), 0.0))
        matrix.append(row)

    picks = align_monotonic(matrix)
    group_confirmed = _alignment_is_conclusive(picks, candidate_episodes)

    results: list[dict] = []
    for f, transcript_and_dur, pick in zip(files, transcripts, picks):
        forced_ep = candidate_episodes[pick] if pick is not None else None
        result = process_file(
            mkv_path=f,
            episodes=candidate_episodes,
            show_name=show_name,
            reference_subtitles=reference_subtitles,
            threshold=args.threshold,
            whisper_model=args.whisper_model,
            whisper_duration=whisper_duration,
            whisper_skip=args.whisper_skip,
            no_whisper=args.no_whisper,
            ai_api_key=ai_api_key,
            local_ai_model=local_ai_model,
            local_ai_url=args.local_ai_url,
            dry_run=args.dry_run,
            os_api_key=os_api_key,
            os_token=os_token,
            cache_dir=cache_dir,
            alt_candidates=args.alt_candidates,
            matched_this_run=matched_this_run,
            pending_renames=pending_renames,
            processing_paths=processing_paths,
            log_dir=log_dir,
            forced_episode=forced_ep,
            precomputed_transcript=transcript_and_dur,
            group_confirmed=group_confirmed,
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# 13. Summary report
# ---------------------------------------------------------------------------
def print_summary(results: list[dict], dry_run: bool) -> None:
    label = " (DRY RUN)" if dry_run else ""
    print(f"\n{'=' * 72}")
    print(f"SUMMARY{label}")
    print(f"{'=' * 72}")
    print(f"{'FILE':<36} {'STATUS':<14} {'MATCHED':<16} {'CONF':>5}")
    print(f"{'-' * 72}")
    for r in results:
        fname = r["file"][:35]
        status = r["status"]
        matched = (r["matched"] or "")[:15]
        conf = f"{r['confidence']:.0%}" if r["confidence"] else "—"
        print(f"{fname:<36} {status:<14} {matched:<16} {conf:>5}")
    print(f"{'=' * 72}")

    counts = Counter(r["status"] for r in results)
    summary = "  ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    print(f"Totals: {summary}")

    scored = [r for r in results if r.get("top_scores")]
    if scored:
        print(f"\n{'-' * 72}")
        print("TOP CANDIDATES (per file, ranked by text-match score)")
        print("For UNMATCHED files: if one candidate clearly leads the rest, that's")
        print("probably the right episode — worth a manual rename.")
        print(f"{'-' * 72}")
        for r in scored:
            print(f"{r['file']}  [{r['status']}]")
            for i, (key, title, score) in enumerate(r["top_scores"], 1):
                label_str = f"{key} - {title}" if title else key
                print(f"    {i}. {label_str[:50]:<50} {score:.1%}")
        print(f"{'-' * 72}")


# ---------------------------------------------------------------------------
# 14. SMB / network path resolver (macOS)
# ---------------------------------------------------------------------------
def _resolve_smb_path(smb_url: str) -> Path:
    """
    Translate an smb:// URL to its local /Volumes/ mount path on macOS.
    macOS mounts SMB shares as /Volumes/<share-name>; the rest of the URL
    becomes a subdirectory under that mount point.

    smb://server/vault/Media/TV/Show  →  /Volumes/vault/Media/TV/Show

    If the share is not yet mounted, prints a clear error rather than
    trying to trigger a mount (which would pop up Finder dialogs).
    """
    import urllib.parse
    parsed = urllib.parse.urlparse(smb_url)
    path_parts = parsed.path.lstrip("/").split("/", 1)
    share = urllib.parse.unquote(path_parts[0])
    subpath = urllib.parse.unquote(path_parts[1]) if len(path_parts) > 1 else ""

    volumes = Path("/Volumes")
    # Exact match first; fall back to prefix match because macOS appends
    # " 1", " 2" etc. when a share name conflicts with an existing mount.
    mount: Optional[Path] = volumes / share
    if not mount.is_dir():
        candidates = sorted(
            d for d in volumes.iterdir()
            if d.name.startswith(share) and d.is_dir()
        )
        mount = candidates[0] if candidates else None

    if mount is None:
        log.error(
            f"SMB share '{share}' is not mounted.\n"
            f"Connect to it in Finder first (⌘K → {parsed.scheme}://{parsed.netloc}/{share}),\n"
            f"then re-run. Expected local path: /Volumes/{share}/{subpath}"
        )
        sys.exit(1)

    local = (mount / subpath) if subpath else mount
    log.info(f"SMB path resolved: {smb_url}  →  {local}")
    return local


# ---------------------------------------------------------------------------
# 15. Series review orchestrator
# ---------------------------------------------------------------------------
def run_series_review(
    args: argparse.Namespace,
    scan_path: Path,
    show_name: str,
    all_episodes: list[dict],
    api_key: str,
    token: str,
    cache_dir: Path,
    ai_api_key: Optional[str],
    local_ai_model: Optional[str],
    log_dir: Optional[Path],
) -> None:
    """
    Group already-named SxxExx files by season, ask which seasons to verify,
    then for each selected season run the normal matching pipeline with a
    ±review_window episode search window around each file's guessed number.
    """
    season_files = parse_series_files(scan_path)
    if not season_files:
        log.error(
            "No SxxExx-named video files found in the scan path.\n"
            "--series-review requires files already named with season/episode guesses "
            "(e.g. S01E03.mkv). Run without --series-review to identify unnamed files."
        )
        return

    selected = prompt_series_season_selection(season_files, auto=args.no_confirm)
    if not selected:
        log.info("No seasons selected — nothing to do.")
        return

    if args.dry_run:
        log.info("DRY RUN — no files will be renamed\n")

    ollama_proc = ollama_start(args.local_ai_url) if local_ai_model else None
    all_results: list[dict] = []

    try:
        for season_num in selected:
            files = season_files[season_num]
            season_eps = [ep for ep in all_episodes if ep["season"] == season_num]

            if not season_eps:
                log.warning(f"Season {season_num}: no episode metadata found — skipping")
                continue

            print(f"\n{'─' * 68}")
            print(f"  SEASON {season_num}  —  {len(files)} file(s)  |  {len(season_eps)} known episode(s)")
            print(f"{'─' * 68}")

            reference_subtitles = prefetch_reference_subtitles(
                api_key, token, show_name, season_eps, cache_dir
            )
            if not reference_subtitles:
                log.warning(
                    f"Season {season_num}: no reference subtitles available — "
                    "text matching will be skipped for this season."
                )
                if not ai_api_key and not local_ai_model:
                    log.warning(f"Season {season_num}: skipping (no subtitles and no AI fallback).")
                    continue

            matched_this_run: set[str] = set()
            pending_renames: list[tuple[Path, Path]] = []
            processing_paths: set[Path] = {f for f, _ in files}

            # Evaluate and rank every file in the season together, in one pass,
            # before renaming any of them — rather than settling discs one at a
            # time, which lets an earlier disc's file lock in an episode that a
            # later disc's file (not yet evaluated) would have matched better,
            # especially where two discs' ±review_window ranges overlap at the
            # seam. Each file still keeps its own ±review_window around its own
            # guessed number as a hard bound (per_file_windows) — the alignment
            # can only resolve ambiguity within that range, never reassign a file
            # further out just because doing so raises the season's total score.
            ep_numbers = {ep["number"] for ep in season_eps}
            valid_files: list[Path] = []
            per_file_windows: list[tuple[int, int]] = []
            for file_path, guessed_ep in files:
                lo = max(1, guessed_ep - args.review_window)
                hi = guessed_ep + args.review_window
                if not any(lo <= n <= hi for n in ep_numbers):
                    log.warning(
                        f"  {file_path.name}: no candidates in window "
                        f"[E{lo:02d}–E{hi:02d}] (season has E01–E{season_eps[-1]['number']:02d}) "
                        f"— skipping"
                    )
                    continue
                valid_files.append(file_path)
                per_file_windows.append((lo, hi))

            if not valid_files:
                log.warning(f"Season {season_num}: no files with a valid candidate window — skipping")
                continue

            log.info(
                f"  Aligning all {len(valid_files)} file(s) together "
                f"(±{args.review_window} episodes per file's own guess)"
            )

            season_results = process_file_group(
                files=valid_files,
                candidate_episodes=season_eps,
                reference_subtitles=reference_subtitles,
                show_name=show_name,
                args=args,
                ai_api_key=ai_api_key,
                local_ai_model=local_ai_model,
                os_api_key=api_key,
                os_token=token,
                cache_dir=cache_dir,
                matched_this_run=matched_this_run,
                pending_renames=pending_renames,
                processing_paths=processing_paths,
                log_dir=log_dir,
                per_file_windows=per_file_windows,
            )

            if pending_renames:
                log.info(f"\nApplying {len(pending_renames)} deferred rename(s)...")
                for tmp_path, final_path in pending_renames:
                    do_rename(tmp_path, final_path, args.dry_run)

            print_summary(season_results, args.dry_run)
            all_results.extend(season_results)

    finally:
        ollama_stop(ollama_proc)

    if len(selected) > 1:
        counts = Counter(r["status"] for r in all_results)
        label = " (DRY RUN)" if args.dry_run else ""
        print(f"\n{'=' * 72}")
        print(f"SERIES REVIEW COMPLETE{label}  —  {len(selected)} season(s) processed")
        print("  ".join(f"{v} {k}" for k, v in sorted(counts.items())))
        print(f"{'=' * 72}")


# ---------------------------------------------------------------------------
# 15. Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    _load_secrets()

    _DEFAULT_TEMP = Path("/Volumes/NVME/MakeMKV")
    tmp_override: Optional[Path] = None
    if args.temp_dir:
        tmp_override = Path(args.temp_dir).resolve()
    elif _DEFAULT_TEMP.is_dir():
        tmp_override = _DEFAULT_TEMP
    if tmp_override:
        tmp_override.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(tmp_override)
        log.info(f"Temp dir: {tmp_override}")

    if args.path.startswith("smb://") or args.path.startswith("afp://"):
        scan_path = _resolve_smb_path(args.path)
    else:
        scan_path = Path(args.path).resolve()
    if not scan_path.exists():
        log.error(f"Path does not exist: {scan_path}")
        sys.exit(1)

    if args.check_subtitles:
        check_subtitles(scan_path)
        return

    # Cache lives inside (or alongside) the scan directory, falling back to the
    # local temp dir when the scan path (e.g. a read-restricted SMB share) can't
    # be written to.
    cache_dir = (scan_path / args.cache_dir).resolve()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        fallback_root = tmp_override or Path(tempfile.gettempdir())
        cache_dir = (fallback_root / "subtitle_cache" / scan_path.name).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        log.warning(f"Cannot write cache under {scan_path} ({e}); using {cache_dir} instead")

    # --- Infer show / season from directory structure ---
    show_name, season = infer_show_season(scan_path, args.show, args.season)
    if args.series_review:
        # Series review groups by season from the filenames — fetch all seasons.
        season = None
    log.info(f"Show:   {show_name!r}")
    log.info(f"Season: {season if season is not None else 'all'}")

    # --- TVMaze lookup ---
    log.info("Searching TVMaze for show metadata...")
    candidates = tvmaze_search_shows(show_name)
    show_data = prompt_show_selection(show_name, candidates, auto=args.no_confirm)
    actual_name: str = show_name
    episodes: list[dict] = []

    if show_data:
        actual_name = show_data["name"]
        show_id: int = show_data["id"]
        log.info(f"TVMaze: {actual_name!r}  (ID {show_id})")
        episodes = tvmaze_get_episodes(show_id, season)
    elif candidates:
        log.error("No show selected — cannot continue. Re-run and select the correct show.")
        sys.exit(1)
    else:
        log.warning(f"TVMaze: show not found — {show_name!r}")

    # --- TMDB enrichment / fallback ---
    tmdb_api_key = os.environ.get("TMDB_API_KEY", "")
    if tmdb_api_key:
        tmdb_show = tmdb_search_show(actual_name if show_data else show_name, tmdb_api_key)
        if tmdb_show:
            tmdb_eps = tmdb_get_episodes(tmdb_show["id"], season, tmdb_api_key)
            if tmdb_eps:
                if episodes:
                    before = sum(1 for ep in episodes if (ep.get("summary") or "").strip())
                    episodes = merge_episode_summaries(episodes, tmdb_eps)
                    after = sum(1 for ep in episodes if (ep.get("summary") or "").strip())
                    log.info(
                        f"TMDB: enriched summaries "
                        f"({before} → {after} episodes with summaries)"
                    )
                else:
                    actual_name = tmdb_show["name"]
                    episodes = tmdb_eps
                    log.info(
                        f"TMDB: using as primary source — "
                        f"{actual_name!r}, {len(episodes)} episode(s)"
                    )
        else:
            log.debug(f"TMDB: show not found — {actual_name!r}")
    elif not show_data:
        log.info("Tip: set TMDB_API_KEY in tv.secrets to try TMDB as a fallback source")

    if not episodes:
        log.error(
            f"No episodes found for {show_name!r} / season {season}.\n"
            "Tip: use --show 'Exact Show Name' to override, or add TMDB_API_KEY to tv.secrets"
        )
        sys.exit(1)
    log.info(f"Loaded {len(episodes)} episode(s)")

    # --- OpenSubtitles credentials ---
    api_key = os.environ.get("OPENSUBTITLES_API_KEY", "")
    username = os.environ.get("OPENSUBTITLES_USERNAME", "")
    password = os.environ.get("OPENSUBTITLES_PASSWORD", "")

    if not (api_key and username and password):
        log.error(
            "OpenSubtitles credentials are required.\n\n"
            "Set these environment variables:\n"
            "  OPENSUBTITLES_API_KEY   — get a free key at https://www.opensubtitles.com/consumers\n"
            "  OPENSUBTITLES_USERNAME  — your opensubtitles.com username\n"
            "  OPENSUBTITLES_PASSWORD  — your opensubtitles.com password\n"
        )
        sys.exit(1)

    token = os_login(api_key, username, password)
    if not token:
        sys.exit(1)

    # --- AI fallback setup (shared by both modes) ---
    ai_api_key: Optional[str] = None
    local_ai_model: Optional[str] = args.local_ai

    if args.local_ai and args.ai_fallback:
        log.error("--local-ai and --ai-fallback are mutually exclusive; pick one.")
        sys.exit(1)

    if args.ai_fallback:
        ai_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not ai_api_key:
            log.error(
                "--ai-fallback requires ANTHROPIC_API_KEY to be set.\n"
                "Get a key at https://console.anthropic.com/"
            )
            sys.exit(1)
        log.info("Claude AI fallback: enabled")

    if local_ai_model:
        log.info(f"Local AI fallback: {local_ai_model} @ {args.local_ai_url}")

    # --- Match log directory (shared by both modes) ---
    log_dir: Optional[Path] = None
    if args.log_dir is not None:
        log_dir = Path(args.log_dir).resolve()
    else:
        log_dir = Path(__file__).parent / "match_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Match logs: {log_dir}")

    # --- Series review mode ---
    if args.series_review:
        run_series_review(
            args=args,
            scan_path=scan_path,
            show_name=actual_name,
            all_episodes=episodes,
            api_key=api_key,
            token=token,
            cache_dir=cache_dir,
            ai_api_key=ai_api_key,
            local_ai_model=local_ai_model,
            log_dir=log_dir,
        )
        return

    # --- Pre-fetch reference subtitles for the whole season ---
    reference_subtitles = prefetch_reference_subtitles(
        api_key, token, actual_name, episodes, cache_dir
    )
    if not reference_subtitles:
        log.warning(
            "No reference subtitles available from OpenSubtitles for this season. "
            "Text matching will be skipped."
        )
        if not args.ai_fallback and not args.local_ai:
            log.error(
                "Cannot identify episodes without reference subtitles or an AI fallback. "
                "Re-run with --ai-fallback (Claude) or --local-ai MODEL (Ollama/LM Studio)."
            )
            sys.exit(1)

    if args.full_scan:
        log.info("Full scan: transcribing complete audio track (--full-scan)")

    # --- Discover MKV files ---
    mkv_files = find_video_files(scan_path, force=args.force)
    log.info(f"\nFound {len(mkv_files)} video file(s) to process")
    if not mkv_files:
        log.info("Nothing to process. Done.")
        return

    if args.dry_run:
        log.info("DRY RUN — no files will be renamed\n")

    # Highest disc number seen per season, used to estimate each disc's episode range.
    disc_totals = build_disc_totals(mkv_files, scan_path, season)

    # --- Start Ollama if needed ---
    ollama_proc = ollama_start(args.local_ai_url) if local_ai_model else None

    # Group files by season only. Each file's own disc (if any) still bounds
    # which episodes it can be assigned — via per_file_windows below — but the
    # alignment itself spans the whole season in one pass, evaluated and ranked
    # before anything is renamed. Settling one disc at a time would let an
    # earlier disc's file lock in an episode that a later disc's file (not yet
    # evaluated) would have matched better, especially where two discs'
    # estimated ranges overlap at the seam after --disc-window padding.
    season_groups: dict[Optional[int], list[Path]] = {}
    file_disc: dict[Path, Optional[int]] = {}
    for mkv_path in mkv_files:
        file_season = infer_file_season(mkv_path, scan_path, season)
        disc = infer_disc_number(mkv_path, scan_path)
        season_groups.setdefault(file_season, []).append(mkv_path)
        file_disc[mkv_path] = disc

    # --- Process each season ---
    results: list[dict] = []
    matched_this_run: set[str] = set()  # prevents duplicate episode assignments within one run
    pending_renames: list[tuple[Path, Path]] = []  # (temp_path, final_path) for deferred renames
    processing_paths: set[Path] = set(mkv_files)   # all source files in this run
    try:
        for file_season, group_paths in season_groups.items():
            # Matters when `season` is None, i.e. scanning a show root with
            # multiple season subfolders — otherwise this is just `episodes`.
            season_episodes = (
                [ep for ep in episodes if ep["season"] == file_season]
                if file_season is not None else episodes
            )
            if not season_episodes:
                log.warning(
                    f"No episode metadata for inferred season {file_season} — "
                    f"skipping {len(group_paths)} file(s)"
                )
                results.extend({
                    "file": p.name, "status": "skipped",
                    "matched": None, "confidence": 0.0,
                } for p in group_paths)
                continue

            per_file_windows: list[tuple[int, int]] = []
            discs_seen: set[int] = set()
            parts_skipped = 0
            for p in group_paths:
                disc = file_disc[p]
                # build_disc_totals only counts discs seen in THIS scan, so a "Part
                # N" folder/filename (a season ripped/released in multiple parts —
                # e.g. a strike-delayed back half) makes that count badly wrong: a
                # part's own disc numbering restarts from 1, so narrow_by_disc would
                # assume this part's 2 discs are the WHOLE season's discs and split
                # all 14 episodes across them — wrongly excluding this part's real
                # (later) episodes from their own files' windows. Skip narrowing
                # for these files rather than trust a split we know is wrong; the
                # cross-file order-alignment in process_file_group still resolves
                # them correctly without it.
                if disc is not None and infer_part_number(p, scan_path) is not None:
                    narrowed = season_episodes
                    parts_skipped += 1
                else:
                    narrowed = narrow_by_disc(
                        season_episodes, disc, disc_totals.get(file_season, 0), args.disc_window
                    )
                per_file_windows.append((narrowed[0]["number"], narrowed[-1]["number"]))
                if disc is not None:
                    discs_seen.add(disc)

            if parts_skipped:
                log.info(
                    f"  Season {file_season}: {len(group_paths)} file(s) — 'Part N' marker "
                    f"detected on {parts_skipped}, skipping disc-based episode narrowing for "
                    f"them (disc numbers restart per part, so this scan's disc count can't "
                    f"represent the season's real disc layout) — aligning by rip order instead"
                )
            elif discs_seen:
                log.info(
                    f"  Season {file_season}: {len(group_paths)} file(s) across "
                    f"{len(discs_seen)} disc(s) — aligning together, each file bounded "
                    f"to its own disc's estimated range"
                )
            elif len(group_paths) > 1:
                log.info(f"  Season {file_season}: aligning {len(group_paths)} file(s) in rip order")

            group_results = process_file_group(
                files=group_paths,
                candidate_episodes=season_episodes,
                reference_subtitles=reference_subtitles,
                show_name=actual_name,
                args=args,
                ai_api_key=ai_api_key,
                local_ai_model=local_ai_model,
                os_api_key=api_key,
                os_token=token,
                cache_dir=cache_dir,
                matched_this_run=matched_this_run,
                pending_renames=pending_renames,
                processing_paths=processing_paths,
                log_dir=log_dir,
                per_file_windows=per_file_windows,
            )
            results.extend(group_results)

        # Apply deferred renames — slots that were occupied by other source files
        # earlier in the run are now free.
        if pending_renames:
            log.info(f"\nApplying {len(pending_renames)} deferred rename(s)...")
            for tmp_path, final_path in pending_renames:
                do_rename(tmp_path, final_path, args.dry_run)
    finally:
        ollama_stop(ollama_proc)

    print_summary(results, args.dry_run)


if __name__ == "__main__":
    main()
