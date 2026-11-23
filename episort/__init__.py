"""EpiSort: identify TV episodes in ripped files by their dialogue.

The command line is `python identifier.py ...` (or `episort ...` once
installed). As a library, see episort.api:

    from episort import IdentifyConfig, identify, apply_renames
"""

from episort.api import (
    DEFAULT_TEMPLATE,
    EpisodeMatch,
    IdentifyConfig,
    IdentifyError,
    Show,
    apply_renames,
    identify,
    load_show,
    match_from_result,
    matches_from_results,
    matches_to_json,
    target_name,
    write_json,
)

__version__ = "0.2.0"

__all__ = [
    "DEFAULT_TEMPLATE",
    "EpisodeMatch",
    "IdentifyConfig",
    "IdentifyError",
    "Show",
    "apply_renames",
    "identify",
    "load_show",
    "match_from_result",
    "matches_from_results",
    "matches_to_json",
    "target_name",
    "write_json",
    "__version__",
]
