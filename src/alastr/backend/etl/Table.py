from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from alastr.backend.tools.logger import logger, _rel


ColumnsArg = Union[str, Sequence[str]]


@dataclass
class Table:
    """
    A thin metadata wrapper around one SQL table that can be exported to Excel.

    Notes
    -----
    - `name` is the SQL table name (unique).
    - `sheet_name` is the Excel sheet name.
    - `file_name` is the Excel workbook filename (e.g., "lex_measures_doc.xlsx").
    - `section` and `subdir` control output directory organization.

    New-architecture metadata (safe to ignore until PipelineManager is updated):
    - analysis_id: canonical ID like "lexicon.readability"
    - level: "doc" or "sent"
    - variant: e.g., "cleaned", "tokenized", "chat_phon" (optional)
    - workbook_stem / sheet_stem: optional semantic names that help build outputs consistently
    """

    om: Any
    name: str
    sheet_name: str
    section: str
    subdir: str
    file_name: str
    primary_keys: List[str]
    pivot: Optional[dict] = None

    # ---- New-architecture fields (optional) ----
    analysis_id: Optional[str] = None
    level: Optional[str] = None
    variant: Optional[str] = None
    workbook_stem: Optional[str] = None
    sheet_stem: Optional[str] = None

    # ---- Existing metadata you were tracking ----
    source_fn: Optional[str] = None
    family: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    fact: bool = False
    fact_table: Optional[str] = None
    grouping_table: Optional[str] = None

    created_at: Optional[str] = None
    num_rows: int = 0
    num_cols: int = 0

    aggregation_level: Optional[str] = None
    granularity: Optional[str] = None

    def __post_init__(self) -> None:
        # Derive file_base without extension
        self.file_base = re.sub(r"\.[^.]*$", "", self.file_name)

        # Back-compat: if caller never sets these, we keep old semantics
        if self.granularity is None and self.level is not None:
            self.granularity = self.level

        # If family isn't explicitly set, a reasonable default is workbook stem or file_base
        if self.family is None:
            self.family = self.workbook_stem or self.file_base

    # -------------------
    # Paths / identifiers
    # -------------------

    def get_subdir(self) -> str:
        """Return subfolder(s) relative to the OutputManager root output directory."""
        return self.subdir

    def get_file_path(self) -> Path:
        """
        Return absolute directory path where this table's workbook should live.

        Uses pathlib so it behaves across OSes.
        """
        return Path(self.om.output_dir) / self.subdir

    def get_workbook_path(self) -> Path:
        """Full path to the Excel workbook this table exports into."""
        return self.get_file_path() / self.file_name

    def get_pks(self) -> List[str]:
        return list(self.primary_keys or [])

    # -------------
    # Data access
    # -------------

    def get_data(self, columns: ColumnsArg = "*", filters: Optional[dict] = None):
        """
        Retrieve table data from the SQLite DB through OutputManager/SQLDaemon.

        Parameters
        ----------
        columns : "*" or sequence[str] or str
            - "*" for all columns
            - a single column name string (e.g., "doc_id") OR
            - list/tuple of column names (e.g., ["doc_id","sent_id","mtld"])
        filters : dict | None
            Column filters passed through to SQLDaemon.access_data.

        Returns
        -------
        pd.DataFrame | None
        """
        if columns == "*" or columns is None:
            cols = "*"
        else:
            # Normalize user input to a list[str]
            if isinstance(columns, str):
                requested = [columns]
            else:
                requested = list(columns)

            # Ensure PKs are included first, without duplicates
            pk = self.get_pks()
            cols_list = pk + [c for c in requested if c not in set(pk)]
            cols = cols_list

        return self.om.access_data(self.name, cols, filters)

    def update_data(self, data: Union[dict, List[dict]]) -> None:
        """
        Insert/update one row or many rows into the SQLite table.

        Delegates to SQLDaemon.update_database (which supports AUTO insert-only mode).
        """
        self.om.update_database(self.name, data)

    # -------------
    # Export
    # -------------

    def export_to_excel(self) -> None:
        """Export this table's current SQL contents to its workbook/sheet."""
        self.om.export_sql_to_excel(self.name)
