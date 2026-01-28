import yaml
from pathlib import Path
import pandas as pd
from datetime import datetime
from alastr.backend.tools.logger import logger, _rel
from alastr.backend.tools.auxiliary import project_path, as_path, find_config_file, load_config, find_files
from alastr.backend.tools.Tier import TierManager
from alastr.backend.etl.SQLDaemon import SQLDaemon
from alastr.backend.etl.Table import Table


class OutputManager:
    _instance = None

    def __new__(cls):
        """
        Singleton OutputManager class to handle data processing configurations, directory structures, and database connections.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)

            cls._instance.num_samples = 0
            cls._instance.tables = {}
            cls._instance._load_config()
            cls._instance._init_output_dir()

            cls._instance._init_db()
            cls._instance.db = SQLDaemon(cls._instance)
            cls._instance.tm = TierManager(cls._instance)

            logger.info("OutputManager initialized successfully.")
        return cls._instance

    def _load_yaml(self, file_path):
        try:
            with open(file_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading config file {file_path}: {e}")
            return {}
    
    def _load_config(self, config_file="config.yaml"):
        self.config = self._load_yaml(config_file)
        logger.info(f"Loaded config: {self.config}")

        self.input_dir = project_path(self.config.get('input_dir', 'alastr_data/input'))
        self.output_dir = project_path(self.config.get('output_dir', 'alastr_data/output'))
        self.sections = self.config.get("sections", {})

    def _init_output_dir(self):
        self.timestamp = datetime.now().strftime("%y%m%d_%H%M")
        self.output_dir = self.output_dir / f"alastr_output_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set at {self.output_dir}")
    
    def _init_db(self):
        """Initializes database path and creates empty tables via SQLDaemon."""
        self.db_path = self.output_dir / f"alastr_database_{self.timestamp}.sqlite"
        logger.info(f"Database set at {self.db_path}")

    def create_table(self, name, sheet_name, section, subdir, file_name, primary_keys, pivot):
        table_dir = Path(self.output_dir, subdir)
        logger.info(f"Creating table {name} with PKs {primary_keys} at {subdir}.")
        self.tables[name] = Table(self, name, sheet_name, section, subdir, file_name, primary_keys, pivot)
        self.db.create_empty_table(name, primary_keys)

        try:
            table_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory created at {table_dir}")
        except OSError as e:
            logger.error(f"Error creating directory: {e}")
    
    def get_fact_tables(self):
        return [t for t in self.tables if t.fact]

    def save_text(self, subdir, filename, content):
        try:
            filepath = Path(subdir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Saved text file: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving text file {filename}: {e}")
            return None

    def save_image(self, file_path, plt):
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(file_path)
            logger.info(f"Saved image file: {file_path}")
        except Exception as e:
            logger.error(f"Error saving image to {file_path}: {e}")

    def update_database(self, table_name, update_data):
        """Delegates database update to SQLDaemon."""
        self.db.update_database(table_name, update_data)

    def access_data(self, table_name, columns='*', filters=None):
        """Delegates database retrieval to SQLDaemon."""
        df = self.db.access_data(table_name, columns, filters)
        return df
    
    def sanitize_column_name(self, col_name):
        return self.db.sanitize_column_name(col_name)

    def export_sql_to_excel(self, table_name):
        """Exports database tables to Excel."""
        try:
            table = self.tables[table_name]
            table_dir = table.get_file_path()
            file_path = Path(table_dir, table.file_name)
            file_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Cannot get file path for table '{table_name}': {e}.")
            return

        try:
            df = table.get_data()
            if df is not None and not df.empty:
                mode = "a" if file_path.exists() else "w"
                with pd.ExcelWriter(file_path, engine="openpyxl", mode=mode) as writer:
                    df.to_excel(writer, sheet_name=table.sheet_name, index=False)
                logger.info(f"Exported table '{table_name}' to {file_path}")
            else:
                logger.warning(f"Dataframe for '{table_name}' is empty. Skipping export.")
        except Exception as e:
            logger.error(f"Failed to export table '{table_name}': {e}.")
    
    def export_tables_by_filter(self, section=None, family=None, tags=None):
        """
        Export multiple tables to Excel based on filters.

        Args:
            section (str, optional): Filter by section name.
            family (str, optional): Filter by table.family value.
            tags (list[str], optional): Filter tables containing all listed tags.
        """
        count = 0
        for table_name, table in self.tables.items():
            if section and table.section != section:
                continue
            if family and table.family != family:
                continue
            if tags and not all(tag in table.tags for tag in tags):
                continue
            try:
                table.export_to_excel()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to export table '{table_name}': {e}")
        
        logger.info(f"Exported {count} table(s) matching filters: "
                    f"section={section}, family={family}, tags={tags}")
