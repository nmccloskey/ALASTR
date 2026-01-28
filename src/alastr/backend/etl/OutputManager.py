import yaml
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from alastr.backend.tools.logger import logger
from alastr.backend.tools.auxiliary import project_path
from alastr.backend.tools.Tier import TierManager
from alastr.backend.etl.SQLDaemon import SQLDaemon
from alastr.backend.etl.Table import Table


class OutputManager:
    """
    OutputManager governs:
      - config + run directories
      - table registry (Table objects)
      - thin wrappers around SQLDaemon I/O
      - Excel export

    Intended control flow:
      PipelineManager (PM) -> OutputManager (OM) -> SQLDaemon (DB)

    "De facto singleton":
      - Use OutputManager.get_instance(...) for typical program runs
      - You can still instantiate OutputManager() directly for tests (no global lock-in)
        by using reset=True or bypassing get_instance entirely.
    """

    _instance: Optional["OutputManager"] = None
    _initialized: bool = False

    # -------------------------
    # De facto singleton factory
    # -------------------------

    @classmethod
    def get_instance(
        cls,
        config_file: str = "config.yaml",
        *,
        config: Optional[dict] = None,
        reset: bool = False,
    ) -> "OutputManager":
        """
        Preferred construction method for normal runs.

        Parameters
        ----------
        config_file:
            Path to YAML config. Ignored if `config` is provided.
        config:
            Pre-loaded config dict. If provided, OM will not read YAML.
        reset:
            If True, discards any cached instance and creates a new one.
            Useful for tests and Streamlit reruns.
        """
        if reset or cls._instance is None:
            cls._instance = cls(config_file=config_file, config=config)
        return cls._instance

    # -----------------
    # Construction
    # -----------------

    def __init__(self, config_file: str = "config.yaml", config: Optional[dict] = None):
        # Allow normal instantiation in tests; keep a guard to avoid double init
        if self.__class__._initialized and self is self.__class__._instance:
            return

        self.num_samples: int = 0
        self.tables: Dict[str, Table] = {}

        self._load_config(config_file=config_file, config=config)
        self._init_output_dir()
        self._init_db()

        # DB and TierManager are "governed" here
        self.db = SQLDaemon(self)
        self.tm = TierManager(self)

        self.__class__._instance = self
        self.__class__._initialized = True
        logger.info("OutputManager initialized successfully.")

    # -----------------
    # Config / paths
    # -----------------

    def _load_yaml(self, file_path: str) -> dict:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading config file {file_path}: {e}")
            return {}

    def _load_config(self, config_file: str, config: Optional[dict]) -> None:
        self.config: dict = config if config is not None else self._load_yaml(config_file)

        # Keep legacy keys, but do not assume they exist forever
        self.input_dir = project_path(self.config.get("input_dir", "alastr_data/input"))
        self.output_root = project_path(self.config.get("output_dir", "alastr_data/output"))

        # Old style (may be deprecated): sections dict
        self.sections = self.config.get("sections", {})

        logger.info(f"Loaded config (keys): {list(self.config.keys())}")

    def _init_output_dir(self) -> None:
        self.timestamp = datetime.now().strftime("%y%m%d_%H%M")
        self.output_dir: Path = self.output_root / f"alastr_output_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set at {self.output_dir}")

    def _init_db(self) -> None:
        """Initializes database path (DB is created lazily by SQLDaemon operations)."""
        self.db_path = self.output_dir / f"alastr_database_{self.timestamp}.sqlite"
        logger.info(f"Database set at {self.db_path}")

    # -----------------
    # Table management
    # -----------------

    def create_table(
        self,
        name: str,
        sheet_name: str,
        section: str,
        subdir: str,
        file_name: str,
        primary_keys: List[str],
        pivot: Optional[dict] = None,
        *,
        # New-architecture metadata (optional)
        analysis_id: Optional[str] = None,
        level: Optional[str] = None,
        variant: Optional[str] = None,
        workbook_stem: Optional[str] = None,
        sheet_stem: Optional[str] = None,
        # Existing/extended metadata (optional)
        tags: Optional[List[str]] = None,
        fact: bool = False,
        fact_table: Optional[str] = None,
        grouping_table: Optional[str] = None,
        source_fn: Optional[str] = None,
        family: Optional[str] = None,
    ) -> Table:
        """
        Register a Table and ensure its SQL backing exists.

        This is intended to be called lazily by PipelineManager right before an
        analysis writes records (so we avoid creating unused empty tables).
        """
        if name in self.tables:
            return self.tables[name]

        table_dir = Path(self.output_dir, subdir)
        table_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Creating table {name} with PKs {primary_keys} at {subdir}.")

        t = Table(
            self,
            name=name,
            sheet_name=sheet_name,
            section=section,
            subdir=subdir,
            file_name=file_name,
            primary_keys=primary_keys,
            pivot=pivot,
            analysis_id=analysis_id,
            level=level,
            variant=variant,
            workbook_stem=workbook_stem,
            sheet_stem=sheet_stem,
            source_fn=source_fn,
            family=family,
            tags=tags or [],
            fact=fact,
            fact_table=fact_table,
            grouping_table=grouping_table,
        )
        self.tables[name] = t

        # Ensure SQL backing table exists
        self.db.create_empty_table(name, primary_keys)

        return t

    def get_fact_tables(self) -> List[Table]:
        return [t for t in self.tables.values() if getattr(t, "fact", False)]

    # -----------------
    # Simple file output
    # -----------------

    def save_text(self, subdir: str, filename: str, content: str) -> Optional[Path]:
        try:
            filepath = Path(self.output_dir, subdir, filename)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(content, encoding="utf-8")
            logger.info(f"Saved text file: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving text file {filename}: {e}")
            return None

    def save_image(self, file_path: Path, plt) -> None:
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(file_path)
            logger.info(f"Saved image file: {file_path}")
        except Exception as e:
            logger.error(f"Error saving image to {file_path}: {e}")

    # -----------------
    # SQLDaemon wrappers
    # -----------------

    def update_database(self, table_name: str, update_data: Any) -> None:
        """Delegates database update to SQLDaemon."""
        self.db.update_database(table_name, update_data)

    def access_data(self, table_name: str, columns: Any = "*", filters: Optional[dict] = None):
        """Delegates database retrieval to SQLDaemon."""
        return self.db.access_data(table_name, columns, filters)

    def sanitize_column_name(self, col_name: str) -> str:
        return self.db.sanitize_column_name(col_name)

    # -----------------
    # Export
    # -----------------

    def export_sql_to_excel(self, table_name: str) -> None:
        """
        Export one SQL table to its configured workbook/sheet.

        Uses append mode and replaces the target sheet if it already exists.
        """
        if table_name not in self.tables:
            logger.error(f"Cannot export unknown table '{table_name}'.")
            return

        table = self.tables[table_name]
        file_path = table.get_workbook_path()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            df = table.get_data()
            if df is None or df.empty:
                logger.warning(f"Dataframe for '{table_name}' is empty. Skipping export.")
                return

            mode = "a" if file_path.exists() else "w"

            # pandas supports if_sheet_exists in append mode
            with pd.ExcelWriter(
                file_path,
                engine="openpyxl",
                mode=mode,
                if_sheet_exists="replace" if mode == "a" else None,
            ) as writer:
                df.to_excel(writer, sheet_name=table.sheet_name, index=False)

            logger.info(f"Exported table '{table_name}' to {file_path}")
        except TypeError:
            # Older pandas without if_sheet_exists
            try:
                with pd.ExcelWriter(file_path, engine="openpyxl", mode=mode) as writer:
                    df.to_excel(writer, sheet_name=table.sheet_name, index=False)
                logger.info(f"Exported table '{table_name}' to {file_path} (no replace support)")
            except Exception as e:
                logger.error(f"Failed to export table '{table_name}': {e}.")
        except Exception as e:
            logger.error(f"Failed to export table '{table_name}': {e}.")

    def export_tables_by_filter(
        self,
        *,
        section: Optional[str] = None,
        family: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        analysis_id: Optional[str] = None,
        level: Optional[str] = None,
        variant: Optional[str] = None,
    ) -> int:
        """
        Export multiple tables to Excel based on filters.

        Returns
        -------
        int
            Number of exported tables.
        """
        count = 0
        tag_list = list(tags) if tags else None

        for table_name, table in self.tables.items():
            if section and table.section != section:
                continue
            if family and getattr(table, "family", None) != family:
                continue
            if analysis_id and getattr(table, "analysis_id", None) != analysis_id:
                continue
            if level and getattr(table, "level", None) != level and getattr(table, "granularity", None) != level:
                continue
            if variant and getattr(table, "variant", None) != variant:
                continue
            if tag_list and not all(tag in getattr(table, "tags", []) for tag in tag_list):
                continue

            try:
                self.export_sql_to_excel(table_name)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to export table '{table_name}': {e}")

        logger.info(
            f"Exported {count} table(s) matching filters: "
            f"section={section}, family={family}, analysis_id={analysis_id}, level={level}, variant={variant}, tags={tags}"
        )
        return count

    # Convenience exports expected by PipelineManager (can expand later)

    def export_section(self, section_id: str) -> int:
        """Export all tables belonging to one section."""
        return self.export_tables_by_filter(section=section_id)

    def export_all(self) -> int:
        """Export all tables."""
        return self.export_tables_by_filter()
