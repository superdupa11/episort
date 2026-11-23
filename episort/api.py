"""
EpiSort as a library: identify a folder's episodes without touching the files,
then rename them later, after someone has reviewed the matches.

    from episort import IdentifyConfig, identify, apply_renames

    config = IdentifyConfig.from_env()            # OPENSUBTITLES_*, TMDB_API_KEY, ...
    matches = identify("/rips/White Collar/Season 1", tmdb_id=..., season=1, config=config)
    for m in matches:
        print(m.path.name, m.key, m.method, f"{m.confidence:.0%}")
    apply_renames([m for m in matches if m.method == "direct"])

identify() runs the same pipeline as the CLI (reference subtitles from
OpenSubtitles, text from embedded subtitles, PGS OCR or Whisper, the season-wide
rip-order alignment, alternates, AI fallback and the fill pass) with three
differences: it never renames (status "matched" where the CLI says "renamed"),
it never prompts (the show comes from an id or the first search hit), and every
setting arrives through IdentifyConfig instead of tv.secrets and the command line.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterator, Literal, Mapping, Optional, Sequence, Union

import requests

from episort import identifier as engine

log = logging.getLogger("tv-identifier")

Method = Literal["direct", "filled", "ai_fallback", "unmatched"]
Progress = Callable[[str, float], None]
EpisodeRange = Union[tuple[int, int], Mapping[Union[str, Path], tuple[int, int]]]

_MATCHED_STATUSES = {"renamed", "matched", "conflict"}


class IdentifyError(RuntimeError):
    """identify() couldn't run at all: no episodes for the show, an OpenSubtitles
    login failure, or no reference subtitles and no AI to fall back on."""


@dataclass
class IdentifyConfig:
    """Everything identify() needs from outside: credentials, services and folders.

    ai: "local" asks an Ollama/LM Studio server (local_ai_url, local_ai_model),
        "claude" the Anthropic API (anthropic_api_key), "none" skips the AI step.
    whisper: "remote" uses a speaches server (whisper_url), "openai" the local
        openai-whisper package, "mlx" mlx-whisper on the Apple Silicon GPU, and
        "none" skips files without usable subtitles. whisper_model defaults to
        the right model for the backend.
    """

    opensubtitles_api_key: str = ""
    opensubtitles_username: str = ""
    opensubtitles_password: str = ""
    tmdb_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    ai: Literal["local", "claude", "none"] = "local"
    local_ai_url: str = engine.OLLAMA_BASE_URL
    local_ai_model: str = engine.OLLAMA_DEFAULT_MODEL
    whisper: Literal["remote", "openai", "mlx", "none"] = "remote"
    whisper_url: str = engine.WHISPER_BASE_URL
    whisper_model: Optional[str] = None
    whisper_duration: int = 0          # minutes of audio to transcribe; 0 = all of it
    whisper_skip: int = 60             # seconds skipped at the start (cold open, titles)
    ocr: bool = True                   # OCR PGS subtitles (needs Pillow and tesseract)
    alt_candidates: int = 4            # alternate OpenSubtitles uploads tried when unsure
    cache_dir: Optional[Path] = None   # reference subtitles; transcripts go next to it
    log_dir: Optional[Path] = None     # per-file match logs (none when unset)
    temp_dir: Optional[Path] = None    # audio clips and extracted subtitles

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None, **overrides) -> "IdentifyConfig":
        """Credentials from the environment (the same names tv.secrets uses),
        then any field given as a keyword."""
        env = os.environ if env is None else env
        config = cls(
            opensubtitles_api_key=env.get("OPENSUBTITLES_API_KEY", ""),
            opensubtitles_username=env.get("OPENSUBTITLES_USERNAME", ""),
            opensubtitles_password=env.get("OPENSUBTITLES_PASSWORD", ""),
            tmdb_api_key=env.get("TMDB_API_KEY") or None,
            anthropic_api_key=env.get("ANTHROPIC_API_KEY") or None,
        )
        return replace(config, **overrides)

    def resolved_whisper_model(self) -> str:
        if self.whisper_model:
            return self.whisper_model
        if self.whisper == "remote":
            return engine.WHISPER_REMOTE_DEFAULT_MODEL
        return engine.WHISPER_LOCAL_DEFAULT_MODEL


@dataclass
class EpisodeMatch:
    """One file's identification. season/episode/title are set unless the method
    is "unmatched"; an unmatched file's best guess, if any, is in
    evidence["best_guess"].

    method: "direct" (confident text match), "ai_fallback" (the AI step picked
    it), "filled" (below threshold, mapped onto the best unclaimed episode:
    weaker evidence, review it), "unmatched" (nothing qualified, or no usable
    text: evidence["status"] says which).
    """

    path: Path
    season: Optional[int]
    episode: Optional[int]
    title: Optional[str]
    confidence: float
    method: Method
    evidence: dict = field(default_factory=dict)

    @property
    def key(self) -> Optional[str]:
        """'S01E05', or None when unmatched."""
        if self.season is None or self.episode is None:
            return None
        return f"S{self.season:02d}E{self.episode:02d}"

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "season": self.season,
            "episode": self.episode,
            "key": self.key,
            "title": self.title,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "EpisodeMatch":
        return cls(
            path=Path(d["path"]), season=d.get("season"), episode=d.get("episode"),
            title=d.get("title"), confidence=float(d.get("confidence") or 0.0),
            method=d["method"], evidence=dict(d.get("evidence") or {}),
        )


@dataclass
class Show:
    """A show and its episodes (the engine's dicts: season, number, name,
    summary, runtime), from TVMaze with TMDB summaries, or TMDB alone."""

    name: str
    tvmaze_id: Optional[int] = None
    tmdb_id: Optional[int] = None
    episodes: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def match_from_result(result: Mapping) -> EpisodeMatch:
    """An EpisodeMatch from one of the engine's per-file result dicts (what
    process_file / process_file_group return, CLI runs included)."""
    status = result.get("status", "skipped")
    episode = result.get("episode")
    if status in _MATCHED_STATUSES and episode:
        method: Method = "ai_fallback" if result.get("ai_picked") else "direct"
    elif status == "filled" and episode:
        method = "filled"
    else:
        method = "unmatched"

    evidence: dict = {
        "status": status,
        "top_scores": [
            {"key": key, "title": title, "score": round(score, 4)}
            for key, title, score in result.get("top_scores") or []
        ],
    }
    for name in ("raw_jaccard", "fill_score"):
        if result.get(name) is not None:
            evidence[name] = round(result[name], 4)
    for name in ("transcript_source", "words", "video_seconds", "window", "forced",
                 "group_confirmed", "ai_response", "fill_note"):
        if result.get(name) is not None:
            evidence[name] = result[name]

    path = Path(result.get("path") or result.get("file", ""))
    confidence = float(result.get("confidence") or 0.0)
    if method == "unmatched":
        if episode:
            evidence["best_guess"] = {"season": episode["season"], "episode": episode["number"],
                                      "title": episode.get("name") or None}
        return EpisodeMatch(path, None, None, None, confidence, method, evidence)
    return EpisodeMatch(path, episode["season"], episode["number"], episode.get("name") or None,
                        confidence, method, evidence)


def matches_from_results(results: Sequence[Mapping]) -> list[EpisodeMatch]:
    return [match_from_result(r) for r in results]


def matches_to_json(matches: Sequence[EpisodeMatch]) -> str:
    return json.dumps([m.to_dict() for m in matches], indent=2)


def write_json(matches: Sequence[EpisodeMatch], dest: Union[str, Path]) -> None:
    """Write matches as a JSON list to a file, or to stdout when dest is '-'."""
    text = matches_to_json(matches) + "\n"
    if str(dest) == "-":
        sys.stdout.write(text)
        sys.stdout.flush()
    else:
        Path(dest).write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Show lookup (no prompts)
# ---------------------------------------------------------------------------
def load_show(
    *,
    name: Optional[str] = None,
    tvmaze_id: Optional[int] = None,
    tmdb_id: Optional[int] = None,
    season: Optional[int] = None,
    tmdb_api_key: Optional[str] = None,
) -> Show:
    """The show's episodes (one season, or all with season=None).

    TVMaze is the primary source, as in the CLI: by tvmaze_id, or else the first
    search hit for name (unless only a tmdb_id was given). TMDB, with a key, adds
    longer summaries, and is the whole source when TVMaze has nothing (looked up
    by tmdb_id, or by name)."""
    show = Show(name=name or "", tvmaze_id=tvmaze_id, tmdb_id=tmdb_id)

    if tvmaze_id is None and name and tmdb_id is None:
        hits = engine.tvmaze_search_shows(name)
        if hits:
            show.tvmaze_id = hits[0]["id"]
            show.name = hits[0].get("name") or name
            log.info(f"TVMaze: first hit for {name!r} is {show.name!r} (ID {show.tvmaze_id})")
    elif tvmaze_id is not None:
        try:
            r = requests.get(f"{engine.TVMAZE_BASE}/shows/{tvmaze_id}", timeout=10)
            r.raise_for_status()
            show.name = r.json().get("name") or show.name
        except Exception as exc:
            log.error(f"TVMaze show {tvmaze_id} lookup failed: {exc}")
    if show.tvmaze_id is not None:
        show.episodes = engine.tvmaze_get_episodes(show.tvmaze_id, season)

    if tmdb_api_key:
        if show.tmdb_id is None and (show.name or name):
            found = engine.tmdb_search_show(show.name or name, tmdb_api_key)
            if found:
                show.tmdb_id = found["id"]
                if not show.name:
                    show.name = found["name"]
        elif show.tmdb_id is not None and not show.name:
            try:
                r = requests.get(f"{engine.TMDB_BASE}/tv/{show.tmdb_id}",
                                 params={"api_key": tmdb_api_key, "language": "en-US"}, timeout=10)
                r.raise_for_status()
                show.name = r.json().get("name") or ""
            except Exception as exc:
                log.error(f"TMDB show {show.tmdb_id} lookup failed: {exc}")
        if show.tmdb_id is not None:
            tmdb_eps = engine.tmdb_get_episodes(show.tmdb_id, season, tmdb_api_key)
            if show.episodes and tmdb_eps:
                show.episodes = engine.merge_episode_summaries(show.episodes, tmdb_eps)
            elif tmdb_eps:
                show.episodes = tmdb_eps
    return show


# ---------------------------------------------------------------------------
# identify()
# ---------------------------------------------------------------------------
def _engine_args(folder: Path, config: IdentifyConfig, threshold: float, fill_unmatched: bool,
                 fill_min_score: float, disc_window: int, progress: Optional[Progress]):
    """The argparse namespace process_file_group reads, from the CLI's own
    defaults, so a setting added to the CLI is never missing here."""
    args = engine.parse_args([str(folder)])
    args.threshold = threshold
    args.no_fill_unmatched = not fill_unmatched
    args.fill_min_score = fill_min_score
    args.disc_window = disc_window
    args.dry_run = True                       # belt and braces: identify_only never renames
    args.no_whisper = config.whisper == "none"
    args.local_whisper = {"openai": True, "mlx": "mlx"}.get(config.whisper, False)
    args.whisper_url = config.whisper_url
    args.whisper_model = config.resolved_whisper_model()
    args.whisper_duration = config.whisper_duration
    args.whisper_skip = config.whisper_skip
    args.full_scan = config.whisper_duration == 0
    args.no_ocr = not config.ocr
    args.alt_candidates = config.alt_candidates
    args.local_ai_url = config.local_ai_url
    args.identify_only = True
    args.progress = progress
    return args


@contextmanager
def _tempdir(path: Optional[Path]) -> Iterator[None]:
    """Point tempfile at path for the run (the engine's audio clips and
    subtitle extracts go there), and put the old setting back afterwards."""
    if path is None:
        yield
        return
    saved = tempfile.tempdir
    Path(path).mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(path)
    try:
        yield
    finally:
        tempfile.tempdir = saved


def _windows(files: Sequence[Path], folder: Path, season_eps: list[dict], season: Optional[int],
             episode_range: Optional[EpisodeRange], disc_totals: dict[int, int],
             disc_window: int) -> list[tuple[int, int]]:
    """Each file's (min, max) episode window: from episode_range when given (one
    range for every file, or per file, e.g. from a disc's title map), otherwise
    from its disc number the way the CLI estimates it."""
    numbers = [ep["number"] for ep in season_eps]
    lo_all, hi_all = min(numbers), max(numbers)
    out = []
    for f in files:
        given = None
        if isinstance(episode_range, tuple):
            given = episode_range
        elif episode_range is not None:
            given = episode_range.get(f) or episode_range.get(str(f)) or episode_range.get(f.name)
        if given is not None:
            lo, hi = max(given[0], lo_all), min(given[1], hi_all)
            if lo <= hi:
                out.append((lo, hi))
                continue
        disc = engine.infer_disc_number(f, folder)
        if disc is not None and engine.infer_part_number(f, folder) is not None:
            narrowed = season_eps
        else:
            narrowed = engine.narrow_by_disc(season_eps, disc, disc_totals.get(season, 0), disc_window)
        out.append((narrowed[0]["number"], narrowed[-1]["number"]))
    return out


def identify(
    folder: Union[str, Path],
    *,
    show: Optional[Show] = None,
    show_name: Optional[str] = None,
    tvmaze_id: Optional[int] = None,
    tmdb_id: Optional[int] = None,
    season: Optional[int] = None,
    files: Optional[Sequence[Union[str, Path]]] = None,
    threshold: float = 0.75,
    fill_unmatched: bool = True,
    fill_min_score: float = engine.FILL_MIN_SCORE,
    episode_range: Optional[EpisodeRange] = None,
    disc_window: int = 3,
    exclude_named: bool = True,
    config: Optional[IdentifyConfig] = None,
    progress: Optional[Progress] = None,
) -> list[EpisodeMatch]:
    """Identify the episodes in folder (a season folder, or a show folder with
    season subfolders) and return one EpisodeMatch per file. Never renames.

    The show is `show` when given, else looked up by tvmaze_id / tmdb_id, else
    by show_name or the folder's name (first search hit). season defaults to
    what the folder name says (Season 1, S01). files limits the run to those
    files, in that order (the rip order the alignment relies on); by default
    every video under folder is used, sorted as the CLI sorts them. With
    exclude_named, episodes that other files in folder are already named after
    (S01E03.mkv) aren't offered to these files.

    episode_range (e.g. (5, 8) for disc 2; one range for every file, or a
    mapping per file path) bounds the rip-order alignment and the fill pass, as
    the CLI's disc estimate does. A decisive text match outside it still wins:
    evidence["window"] shows the range used, so a caller can flag that
    disagreement instead of trusting either side blindly.

    progress(message, fraction) is called as files are read.
    """
    folder = Path(folder).resolve()
    config = config or IdentifyConfig.from_env()
    inferred_name, inferred_season = engine.infer_show_season(folder, show_name, season)
    season = inferred_season if season is None else season

    if show is None:
        show = load_show(name=show_name or inferred_name, tvmaze_id=tvmaze_id, tmdb_id=tmdb_id,
                         season=season, tmdb_api_key=config.tmdb_api_key)
    episodes = [ep for ep in show.episodes if season is None or ep["season"] == season]
    if not episodes:
        raise IdentifyError(f"no episodes found for {show.name or show_name or inferred_name!r}"
                            + (f" season {season}" if season is not None else ""))

    if not (config.opensubtitles_api_key and config.opensubtitles_username
            and config.opensubtitles_password):
        raise IdentifyError("OpenSubtitles credentials are missing (OPENSUBTITLES_API_KEY, "
                            "OPENSUBTITLES_USERNAME, OPENSUBTITLES_PASSWORD)")
    token = engine.os_login(config.opensubtitles_api_key, config.opensubtitles_username,
                            config.opensubtitles_password)
    if not token:
        raise IdentifyError("OpenSubtitles login failed")

    ai_api_key = config.anthropic_api_key if config.ai == "claude" else None
    local_ai_model = config.local_ai_model if config.ai == "local" else None
    if config.ai == "claude" and not ai_api_key:
        raise IdentifyError("ai='claude' needs anthropic_api_key")

    cache_dir = Path(config.cache_dir) if config.cache_dir else folder / ".subtitle_cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        cache_dir = Path(tempfile.gettempdir()) / "subtitle_cache" / folder.name
        cache_dir.mkdir(parents=True, exist_ok=True)
    if config.log_dir:
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    references = engine.prefetch_reference_subtitles(
        config.opensubtitles_api_key, token, show.name, episodes, cache_dir)
    if not references and not (ai_api_key or local_ai_model):
        raise IdentifyError("no reference subtitles on OpenSubtitles for this season, and no AI "
                            "fallback to identify by")

    if files is None:
        paths = engine.find_video_files(folder, force=True)
    else:
        paths = [Path(f).resolve() for f in files]
    if not paths:
        return []

    args = _engine_args(folder, config, threshold, fill_unmatched, fill_min_score, disc_window, progress)
    claimed = engine.find_confidently_matched_episodes(folder, exclude=set(paths)) if exclude_named else {}
    disc_totals = engine.build_disc_totals(paths, folder, season)

    groups: dict[Optional[int], list[Path]] = {}
    for p in paths:
        groups.setdefault(engine.infer_file_season(p, folder, season), []).append(p)

    results: list[dict] = []
    matched_this_run: set[str] = set()
    ollama = engine.ollama_start(config.local_ai_url) if local_ai_model else None
    try:
        with _tempdir(config.temp_dir):
            for file_season, group in groups.items():
                season_eps = [ep for ep in episodes if file_season is None or ep["season"] == file_season]
                taken = claimed.get(file_season, set()) if file_season is not None else set()
                season_eps = sorted((ep for ep in season_eps if ep["number"] not in taken),
                                    key=lambda ep: ep["number"])
                if not season_eps:
                    results.extend({"file": p.name, "path": str(p), "status": "skipped",
                                    "matched": None, "confidence": 0.0} for p in group)
                    continue
                results.extend(engine.process_file_group(
                    files=group,
                    candidate_episodes=season_eps,
                    reference_subtitles=references,
                    show_name=show.name,
                    args=args,
                    ai_api_key=ai_api_key,
                    local_ai_model=local_ai_model,
                    os_api_key=config.opensubtitles_api_key,
                    os_token=token,
                    cache_dir=cache_dir,
                    matched_this_run=matched_this_run,
                    pending_renames=[],
                    processing_paths=set(group),
                    log_dir=Path(config.log_dir) if config.log_dir else None,
                    per_file_windows=_windows(group, folder, season_eps, file_season, episode_range,
                                              disc_totals, disc_window),
                    file_discs=[engine.infer_disc_number(p, folder) for p in group],
                ))
    finally:
        engine.ollama_stop(ollama)
    return matches_from_results(results)


# ---------------------------------------------------------------------------
# Renaming, after review
# ---------------------------------------------------------------------------
DEFAULT_TEMPLATE = "S{season:02d}E{episode:02d}"


def target_name(match: EpisodeMatch, template: str = DEFAULT_TEMPLATE, show: str = "") -> str:
    """The new file name for a match: template (fields: season, episode, title,
    show, stem) plus the file's own extension. Characters that aren't allowed in
    file names are dropped from the text fields."""
    if match.key is None:
        raise ValueError(f"{match.path.name} is unmatched; it has no episode to name it after")
    name = template.format(
        season=match.season, episode=match.episode,
        title=engine.sanitize_name(match.title or ""), show=engine.sanitize_name(show),
        stem=match.path.stem,
    )
    return engine.sanitize_name(name) + match.path.suffix


def apply_renames(
    matches: Sequence[EpisodeMatch],
    template: str = DEFAULT_TEMPLATE,
    *,
    show: str = "",
    dry_run: bool = False,
) -> list[tuple[Path, Path]]:
    """Rename each matched file in place (same folder) after the template, and
    return the (old, new) paths that changed. Unmatched files are left alone.
    Files that swap names with each other are handled; a target that exists and
    isn't one of these files is skipped with a warning."""
    todo = [m for m in matches if m.key is not None and m.method != "unmatched"]
    processing = {m.path for m in todo}
    pending: list[tuple[Path, Path]] = []
    done: list[tuple[Path, Path]] = []
    for m in todo:
        target = m.path.with_name(target_name(m, template, show))
        if target == m.path:
            continue
        before = len(pending)
        if engine.do_rename(m.path, target, dry_run, pending, processing):
            if len(pending) == before:
                done.append((m.path, target))
    for parked, final in pending:
        original = next(m.path for m in todo if m.path.with_name(target_name(m, template, show)) == final)
        if engine.do_rename(parked, final, dry_run):
            done.append((original, final))
    return done
