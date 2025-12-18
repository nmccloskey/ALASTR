import yaml
from pathlib import Path
import pandas as pd
from alastr.backend.tools.logger import logger, get_root, early_log, _rel
import argparse


def project_path(*parts) -> Path:
    """Return an absolute path anchored to the project root."""
    return (Path.cwd().resolve() / Path(*parts)).resolve()

def as_path(p: str | Path) -> Path:
    """
    Normalize a path to be relative to the current working directory (project root).

    If the target lies outside the working directory, returns its resolved absolute path.
    This ensures all internal references stay project-root–relative without failures.
    """
    try:
        p = Path(p).expanduser()
        cwd = Path.cwd().resolve()
        resolved = (cwd / p).resolve() if not p.is_absolute() else p.resolve()

        # Try to make relative if possible
        try:
            rel = resolved.relative_to(cwd)
            logger.debug(f"Resolved relative path: {rel}")
            return rel
        except ValueError:
            # Not under cwd — fine, just return absolute
            logger.debug(f"Resolved absolute path (outside cwd): {resolved}")
            return resolved

    except Exception as e:
        logger.error(f"Failed to resolve path {p}: {e}")
        raise

def find_config_file(base_dir: Path, user_arg: str | None = None) -> Path | None:
    """
    Find a YAML configuration file.
    Priority:
      1. User-specified path via --config
      2. config.yaml in current directory
      3. Any *.yaml file under input/config/
    """
    if user_arg:
        cfg = Path(user_arg)
        if cfg.exists():
            return cfg.resolve()
        else:
            raise FileNotFoundError(f"Specified config not found: {cfg}")

    # Default: search in current working directory
    cwd_cfg = Path("config.yaml")
    if cwd_cfg.exists():
        return cwd_cfg.resolve()

    # Fallback: search recursively under input/config/
    for p in Path("input/config").rglob("*.yaml"):
        return p.resolve()  # first match

    raise FileNotFoundError("No configuration file found. Use --config to specify one.")

def load_config(config_file: str | Path) -> dict:
    """
    Load configuration settings from a YAML file.

    Parameters
    ----------
    config_file : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Configuration dictionary loaded from YAML.
    """
    config_file = find_config_file(get_root(), config_file)
    try:
        with open(config_file, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        early_log("info", f"Loaded configuration from {_rel(config_file)}")
        return config
    except FileNotFoundError:
        early_log("error", f"Configuration file not found: {_rel(config_file)}")
        raise
    except yaml.YAMLError as e:
        early_log("error", f"YAML parsing error in {_rel(config_file)}: {e}")
        raise
    except Exception as e:
        early_log("error", f"Unexpected error loading {_rel(config_file)}: {e}")
        raise


# -------------------------------------------------------------
# File handling utilities
# -------------------------------------------------------------
def find_files(
    match_tiers=None,
    directories=None,
    search_base="",
    search_ext=".xlsx",
    deduplicate=True,
):
    """
    Recursively find files matching tier labels and a base pattern.

    Behavior
    --------
    • Searches all provided directories for filenames containing both
      `search_base` and every label in `match_tiers` (case-sensitive).
    • Returns a list[Path] of matches (empty if none found).
    • Optionally deduplicates identical filenames across directories,
      logging which duplicates were removed.

    Parameters
    ----------
    match_tiers : list[str] | None
        Tier labels (e.g., ["AC", "PreTx"]). None/empty ignored.
    directories : Path | str | list[Path | str] | None
        One or more directories to search (default: CWD).
    search_base : str
        Core substring to match in filenames.
    search_ext : str, default ".xlsx"
        File extension (with dot).
    deduplicate : bool, default True
        Remove duplicate filenames across directories.

    Returns
    -------
    list[Path]
        Matching file paths (may be empty).
    """
    match_tiers = [str(mt) for mt in (match_tiers or []) if mt]
    if directories is None:
        directories = [Path.cwd()]
    elif isinstance(directories, (str, Path)):
        directories = [directories]

    all_matches = []
    for d in directories:
        try:
            d = Path(d)
            if not d.exists():
                logger.warning(f"Directory not found: {_rel(d)} (skipping).")
                continue

            for f in d.rglob(f"*{search_base}*{search_ext}"):
                if all(mt in f.name for mt in match_tiers):
                    all_matches.append(f)
        except Exception as e:
            logger.error(f"Error searching in {_rel(d)}: {e}")

    if not all_matches:
        logger.warning(f"No matches found for base '{search_base}' with tiers {match_tiers}.")
        return []

    if deduplicate:
        seen = {}
        duplicates = {}
        for f in all_matches:
            if f.name in seen:
                duplicates.setdefault(f.name, []).append(f)
            else:
                seen[f.name] = f

        unique_matches = list(seen.values())

        if duplicates:
            logger.warning(
                f"Removed {sum(len(v) for v in duplicates.values())} duplicate filename(s) across directories."
            )
            for fname, paths in duplicates.items():
                logger.warning(f"Duplicate filename '{fname}' found in:")
                for p in [seen[fname], *paths]:
                    logger.warning(f"  - {_rel(p)}")

    else:
        unique_matches = all_matches

    if len(unique_matches) == 1:
        logger.info(f"Matched file for '{search_base}': {_rel(unique_matches[0])}")
    else:
        logger.info(
            f"Multiple ({len(unique_matches)}) files matched '{search_base}' and {match_tiers}."
        )
        for f in unique_matches:
            logger.debug(f"  - {_rel(f)}")

    return unique_matches


def extract_transcript_data(
    transcript_table_path: str | Path,
    type: str = "joined"
) -> pd.DataFrame:
    """
    Load data from a transcript table Excel file.

    Parameters
    ----------
    transcript_table_path : str or Path
        Path to an Excel file produced by `make_transcript_tables`.
    type : {'utterance', 'sample', 'joined'}, default='joined'
        Which dataset to return:
          - 'utterance': utterance-level data
          - 'sample': sample-level metadata
          - 'joined': merged table of both (inner join on 'sample_id')

    Returns
    -------
    pandas.DataFrame
        The requested DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the `type` argument is invalid.
    """
    path = as_path(transcript_table_path)
    if not path.exists():
        logger.error(f"Transcript table not found: {path}")
        raise FileNotFoundError(f"Transcript table not found: {path}")

    try:
        # Read available sheets once to avoid multiple disk I/O operations
        xls = pd.ExcelFile(path, engine="openpyxl")
        sheet_names = [s.lower() for s in xls.sheet_names]

        sample_df = pd.read_excel(xls, sheet_name="samples") if "samples" in sheet_names else None
        utt_df = pd.read_excel(xls, sheet_name="utterances") if "utterances" in sheet_names else None
        xls.close()

        if type == "sample":
            if sample_df is None:
                raise ValueError("Sample sheet not found in transcript table.")
            logger.info(f"Loaded sample data from {path}")
            return sample_df

        elif type == "utterance":
            if utt_df is None:
                raise ValueError("Utterance sheet not found in transcript table.")
            logger.info(f"Loaded utterance data from {path}")
            return utt_df

        elif type == "joined":
            if sample_df is None or utt_df is None:
                raise ValueError("Both sheets required for joined type are missing.")
            joined = sample_df.merge(utt_df, on="sample_id", how="inner")
            logger.info(f"Loaded joined transcript data from {path}")
            return joined

        else:
            raise ValueError(f"Invalid type '{type}'. Must be 'sample', 'utterance', or 'joined'.")

    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        raise
