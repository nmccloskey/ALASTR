from __future__ import annotations

from pathlib import Path
import pandas as pd
import docx2txt as dx
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from alastr.backend.nlp.data_processing import scrub_raw_text, get_text_from_cha
from alastr.backend.tools.logger import logger

# -------------------------
# Text file readers
# -------------------------

def read_chat_file(file_path: Path, exclude_speakers: list) -> str:
    text_content = get_text_from_cha(file_path, exclude_speakers)
    logger.info(f"Processed CHAT file: {file_path}")
    return text_content

def read_text_file(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        text_content = file.read()
        text_content = scrub_raw_text(text_content)
        logger.info(f"Processed TXT file: {file_path}")
        return text_content

def read_docx_file(file_path: Path) -> str:
    text_content = dx.process(file_path)
    text_content = scrub_raw_text(text_content)
    logger.info(f"Processed DOCX file: {file_path}")
    return text_content


# -------------------------
# Generic helpers
# -------------------------

def _require_columns(df: pd.DataFrame, required: Iterable[str], context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required column(s): {missing}")


def _read_tabular(file_path: Path) -> pd.DataFrame:
    try:
        suffix = file_path.suffix.lower()
        if suffix == ".xlsx":
            return pd.read_excel(file_path)
        if suffix == ".csv":
            return pd.read_csv(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed reading tabular file: {file_path!r}") from e

    raise ValueError(f"Unsupported tabular extension: {file_path.suffix!r}")


def _update_tiers_from_df(df: pd.DataFrame, OM: Any, non_tier_cols: Set[str]) -> None:
    """
    Create/update tiers in OM.tm based on dataframe columns excluding non-tier cols.
    """
    if OM is None or not hasattr(OM, "tm"):
        raise ValueError("OM must be provided and must have a TierManager at OM.tm")

    tier_cols = [c for c in df.columns if c not in non_tier_cols]
    new_tiers = []
    for col in tier_cols:
        try:
            tier = OM.tm.make_tier(col)
        except Exception as e:
            raise RuntimeError(f"TierManager.make_tier failed for column {col!r}") from e
        if tier:
            new_tiers.append(tier)

    OM.tm.tiers.update({t.name: t for t in new_tiers if getattr(t, "name", None)})
    logger.info("TierManager tiers now: %s", [t.name for t in OM.tm.tiers.values()])


def _insert_doc_ids(df: pd.DataFrame, start_doc_id: int) -> Tuple[pd.DataFrame, int]:
    """
    Insert doc_id column at position 0 and return (df, next_doc_id).
    """
    df = df.copy()
    if "doc_id" in df.columns:
        df = df.drop(columns=["doc_id"])
    df.insert(0, "doc_id", range(start_doc_id, start_doc_id + len(df)))
    return df, start_doc_id + len(df)


def _insert_doc_label(df: pd.DataFrame, label_series: pd.Series) -> pd.DataFrame:
    """
    Insert doc_label column at position 0 (or replace if exists).
    """
    df = df.copy()
    if "doc_label" in df.columns:
        df = df.drop(columns=["doc_label"])
    df.insert(0, "doc_label", label_series.astype(str))
    return df


# -------------------------
# Spreadsheet importer
# -------------------------

def read_spreadsheet(file_path: Path, doc_id: int, OM: Any) -> List[Dict[str, Any]]:
    """
    Read a spreadsheet (.xlsx or .csv) with a required 'text' column and optional tier columns.
    'speaking_time' is allowed but is never treated as a tier.
    """
    df = _read_tabular(file_path)

    if df.empty:
        raise ValueError(f"{file_path.name} contains no rows.")

    _require_columns(df, required=["text"], context=f"{file_path.name} (spreadsheet)")

    # Drop rows with missing text early so doc_id + labels map to actual docs
    df = df.dropna(subset=["text"]).copy()
    if df.empty:
        raise ValueError(f"{file_path.name} contains no valid 'text' entries after dropping NA.")

    # Update tiers from all non-text columns, excluding speaking_time as well
    non_tier_cols = {"text", "speaking_time"}
    _update_tiers_from_df(df, OM, non_tier_cols=non_tier_cols)

    # Build doc_label: <file_name>|<all non-text cols joined>|<row_index>
    other_columns = [c for c in df.columns if c != "text"]
    if other_columns:
        meta_part = df[other_columns].astype(str).agg("|".join, axis=1)
        label_series = file_path.name + "|" + meta_part + "|" + df.index.astype(str)
    else:
        label_series = file_path.name + "|" + df.index.astype(str)

    df = _insert_doc_label(df, label_series)
    df, _next_doc_id = _insert_doc_ids(df, doc_id)

    records = df.to_dict(orient="records")
    logger.info("Processed %d rows from file: %s", len(records), file_path.name)
    return records


# -------------------------
# Transcript-table importer (RASCAL -> ALASTR)
# -------------------------

def _read_transcript_sheets(file_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        in_samples = pd.read_excel(file_path, sheet_name="samples")
        in_utterances = pd.read_excel(file_path, sheet_name="utterances")
    except Exception as e:
        raise RuntimeError(
            f"Failed reading transcript table xlsx: {file_path!r}. "
            f"Expected sheets: 'samples' and 'utterances'."
        ) from e
    return in_samples, in_utterances


def _concat_utterances(
    utter_df: pd.DataFrame,
    exclude_speakers: Optional[Sequence[str]],
) -> pd.DataFrame:
    exclude_set = set(exclude_speakers or [])
    df = utter_df.copy()

    df["speaker"] = df["speaker"].astype(str)
    if exclude_set:
        df = df[~df["speaker"].isin(exclude_set)].copy()

    df["utterance"] = df["utterance"].fillna("").astype(str)

    # sample_id -> text (concatenated)
    out = (
        df.groupby("sample_id", as_index=False)["utterance"]
        .agg(lambda s: " ".join(x for x in s if x))
        .rename(columns={"utterance": "text"})
    )
    return out


def read_transcript_table(
    file_path: Path,
    doc_id: int,
    exclude_speakers: Optional[Sequence[str]],
    OM: Any,
) -> List[Dict[str, Any]]:
    """
    Read a RASCAL transcript-table .xlsx and return ALASTR-ready sample records.

    'samples' sheet must include: sample_id, file
    'speaking_time' may exist but is optional and never treated as a tier.
    'utterances' sheet must include: sample_id, speaker, utterance
    """
    if OM is None or not hasattr(OM, "tm"):
        raise ValueError("OM must be provided and must have a TierManager at OM.tm")

    in_samples, in_utterances = _read_transcript_sheets(file_path)

    _require_columns(in_samples, required=["sample_id", "file"], context="'samples' sheet")
    _require_columns(in_utterances, required=["sample_id", "speaker", "utterance"], context="'utterances' sheet")

    # Tiers come from sample metadata columns excluding these:
    non_tier_cols = {"sample_id", "file", "speaking_time"}
    _update_tiers_from_df(in_samples, OM, non_tier_cols=non_tier_cols)

    base = in_samples.copy()

    # doc_label = <transcript_table_filename>|<original_chat_filename>
    base = _insert_doc_label(base, file_path.name + "|" + base["file"].astype(str))

    base, _next_doc_id = _insert_doc_ids(base, doc_id)

    # concat utterances (after speaker exclusion) and merge
    text_df = _concat_utterances(in_utterances, exclude_speakers)
    merged = base.merge(text_df, on="sample_id", how="left")
    merged["text"] = merged["text"].fillna("")

    return merged.to_dict(orient="records")


# -------------------------
# Dispatcher
# -------------------------

def prep_samples(
    file_path: Path,
    doc_id: int,
    exclude_speakers,
    OM,
) -> List[Dict[str, Any]]:
    
    name = file_path.name
    suffix = file_path.suffix.lower()
    name_lower = name.lower()

    if suffix not in {".xlsx", ".cha", ".txt", ".docx", ".csv"}:
        logger.warning("Unsupported file format: %s. Skipping.", name)
        return []

    if suffix == ".xlsx" and "transcript_table" in name_lower:
        return read_transcript_table(file_path, doc_id, exclude_speakers, OM)

    if suffix in {".xlsx", ".csv"}:
        return read_spreadsheet(file_path, doc_id, OM)

    # Non-tabular single-doc inputs
    if suffix == ".cha":
        text_content = read_chat_file(file_path, exclude_speakers)
    elif suffix == ".txt":
        text_content = read_text_file(file_path)
    elif suffix == ".docx":
        text_content = read_docx_file(file_path)
    else:
        logger.warning("Unsupported file type after filtering: %s", name)
        return []

    sample_data = {
        "doc_id": doc_id,
        "doc_label": name,
        "text": text_content,
        **OM.tm.match_tiers(name),
    }
    return [sample_data]
