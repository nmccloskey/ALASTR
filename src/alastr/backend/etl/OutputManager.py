import os
import yaml
import pandas as pd
from datetime import datetime
import logging
logger = logging.getLogger("CustomLogger")
from alastr.backend.tools.logger_old import logger, configure_file_handler
from alastr.backend.tools.Tier_old import TierManager
from alastr.backend.eda.EDADaemon import EDADaemon
from alastr.backend.etl.SQLDaemon import SQLDaemon
from alastr.backend.etl.Table import Table
from alastr.backend.eda.visualization import visualize_distinctive_features, generate_corr_maps, generate_data_heatmaps


class OutputManager:
    _instance = None

    def __new__(cls):
        """
        Singleton OutputManager class to handle data processing configurations, directory structures,
        database connections, and visualization options.
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
            cls._instance.eda = EDADaemon(cls._instance)

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

        self.output_label = self.config.get("output_label", "")
        configure_file_handler(self.output_label)

        self.input_dir = os.path.abspath(os.path.expanduser(self.config.get('input_dir', 'alastr_data/input')))
        self.output_dir = os.path.abspath(os.path.expanduser(self.config.get('output_dir', 'alastr_data/output')))
        self.database_dir = os.path.abspath(os.path.expanduser(self.config.get('database_dir', 'alastr_data/database')))
        self.sections = self.config.get("sections", {})
        
        self.cluster = self.config.get("cluster", False)
        self.visualize = self.config.get("visualize", False)
        self.aggregate = self.config.get("aggregate", False)
        self.compare_groups = self.config.get("compare_groups", False)

        self.cohen_d_threshold = self.config.get("cohen_d_threshold", 0.8)
        self.max_feature_visuals = self.config.get("max_feature_visuals", 5)

        self.aggregation_combos = self.config.get("aggregation_combos", [])
        self.all_aggregation_combos = self.config.get("all_aggregation_combos", False)
        self.aggregate_with_clusters = self.config.get("aggregate_with_clusters", False)
        self.aggregation_cols = list(set([col for combo in self.aggregation_combos for col in combo]))
        
        self.comparison_combos = self.config.get("comparison_combos", [])
        self.all_comparison_combos = self.config.get("all_comparison_combos", False)
        self.compare_with_clusters = self.config.get("compare_with_clusters", False)
        self.comparison_cols = list(set([col for combo in self.comparison_combos for col in combo]))

    def _init_output_dir(self):
        self.timestamp = datetime.now().strftime("%y%m%d_%H%M")
        self.output_dir = os.path.join(self.output_dir, f"{self.output_label}_{self.timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory set at {self.output_dir}")
    
    def _init_db(self):
        """Initializes database path and creates empty tables via SQLDaemon."""
        self.db_path = os.path.join(self.database_dir, f"{self.output_label}_{self.timestamp}.sqlite")
        os.makedirs(self.database_dir, exist_ok=True)
        logger.info(f"Database set at {self.db_path}")

    def create_table(self, name, sheet_name, section, subdir, file_name, primary_keys, pivot):
        table_dir = os.path.join(self.output_dir, subdir)
        logger.info(f"Creating table {name} with PKs {primary_keys} at {subdir}.")
        self.tables[name] = Table(self, name, sheet_name, section, subdir, file_name, primary_keys, pivot)
        self.db.create_empty_table(name, primary_keys)

        try:
            os.makedirs(table_dir, exist_ok=True)
            logger.info(f"Directory created at {table_dir}")
        except OSError as e:
            logger.error(f"Error creating directory: {e}")
    
    def get_fact_tables(self):
        return [t for t in self.tables if t.fact]

    def save_text(self, subdir, filename, content):
        try:
            filepath = os.path.join(subdir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Saved text file: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving text file {filename}: {e}")
            return None

    def save_image(self, file_path, plt):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
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
    
    def get_partition_tiers(self):
        return self.tm.get_partition_tiers()

    def sanitize_column_name(self, col_name):
        return self.db.sanitize_column_name(col_name)

    def get_data_with_groupings(self, fact_table: str, dim_table: str, cluster_table: str = None, grouping_cols: list = []) -> pd.DataFrame:
        """
        Merges a fact table with a dimension table (and optionally a cluster table)
        based on common primary keys across all involved tables.

        Args:
            fact_table (str): Name of the fact table.
            dim_table (str): Name of the dimension table.
            cluster_table (str, optional): Name of the cluster table. Defaults to None.

        Returns:
            pd.DataFrame: Merged DataFrame if successful, otherwise None.
        """
        try:
            if grouping_cols == []:
                logger.error(f"No grouping columns specified - selecting all.")
                grouping_cols = '*'

            fact_pks = self.tables[fact_table].get_pks()
            dim_pks = self.tables[dim_table].get_pks()
            common_pks = set(fact_pks) & set(dim_pks)

            if cluster_table:
                cluster_pks = self.tables[cluster_table].get_pks()
                common_pks &= set(cluster_pks)

            common_pks = list(common_pks)

            if not common_pks:
                msg = f"Cannot merge - no matching primary keys found between '{fact_table}', '{dim_table}'"
                if cluster_table:
                    msg += f", and '{cluster_table}'"
                logger.error(msg)
                return None

            if len(common_pks) != len(fact_pks) or len(common_pks) != len(dim_pks):
                logger.warning(f"Partial PK match detected for merging '{fact_table}' and '{dim_table}'. Using {common_pks}.")
            if cluster_table and len(common_pks) != len(cluster_pks):
                logger.warning(f"Partial PK match detected for merging '{fact_table}', '{dim_table}', and '{cluster_table}'. Using {common_pks}.")

            fact_df = self.tables[fact_table].get_data(columns=grouping_cols)
            dim_df = self.tables[dim_table].get_data()

            merged_df = fact_df
            if cluster_table:
                cluster_df = self.tables[cluster_table].get_data()
                merged_df = merged_df.merge(cluster_df, on=common_pks, how="inner")
            merged_df = merged_df.merge(dim_df, on=common_pks, how="inner")

            if len(merged_df) != len(fact_df) or len(merged_df) != len(dim_df):
                logger.warning(f"Row count mismatch after merging '{fact_table}' and '{dim_table}'. "
                            f"fact_df: {fact_df.shape}, dim_df: {dim_df.shape}, merged: {merged_df.shape}.")
            if cluster_table and len(merged_df) != len(cluster_df):
                logger.warning(f"Row count mismatch after merging with '{cluster_table}'. "
                            f"merged: {merged_df.shape}, cluster_df: {cluster_df.shape}")

            logger.info(f"Final merged DataFrame has shape {merged_df.shape}.")
            return merged_df

        except KeyError as e:
            logger.error(f"Table not found: {e}")
        except AttributeError as e:
            logger.error(f"Missing expected method or attribute: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error during data merging: {e}")
        return None

    def export_sql_to_excel(self, table_name):
        """Exports database tables to Excel."""
        try:
            table = self.tables[table_name]
            table_dir = table.get_file_path()
            file_path = os.path.join(table_dir, table.file_name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        except Exception as e:
            logger.error(f"Cannot get file path for table '{table_name}': {e}.")
            return

        try:
            df = table.get_data()
            if df is not None and not df.empty:
                mode = "a" if os.path.exists(file_path) else "w"
                with pd.ExcelWriter(file_path, mode=mode) as writer:
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

    def run_clustering(self, table_name, section):
        """Calls EDADaemon to cluster data."""
        try:
            df = self.tables[table_name].get_data()

            if df is not None and not df.empty:
                pk_columns = self.tables[table_name].get_pks()
                cluster_df = self.eda.run_clustering(df, pk_columns)

            if cluster_df is not None and not cluster_df.empty:
                cluster_table_name = f"{table_name}_clusters"
                self.create_table(name=cluster_table_name, sheet_name=table_name,
                                section=section, subdir=self.tables[table_name].get_subdir(),
                                file_name= f"{self.tables[table_name].file_base}_clusters.xlsx", 
                                primary_keys=self.tables[table_name].get_pks())
                
                self.tables[cluster_table_name].update_data(cluster_df.to_dict(orient="records"))
                self.tables[cluster_table_name].export_to_excel()
        
        except Exception as e:
            logger.error(f"OutputManager failed to run clustering: {e}.")
    
    def run_aggregate_analyses(self, results, section):
        for table_name in results:
            grouping_table_name = "sample_data_sent" if self.tables[table_name].granularity == "sent" else "sample_data_doc"
            cluster_table_name = f"{table_name}_clusters" if self.cluster else None

            if self.aggregate:
                logger.info(f"Aggregating data from table '{table_name}'.")
                grouping_cols = self.aggregation_cols
                merged_df = self.get_data_with_groupings(
                    fact_table=grouping_table_name, dim_table=table_name,
                    cluster_table=cluster_table_name, grouping_cols=grouping_cols)
                # group_by = self.get_partition_tiers()                
                self.run_aggregation(merged_df, grouping_cols, table_name, section)
            
            if self.compare_groups:
                logger.info(f"Comparing group data from table '{table_name}'.")
                grouping_cols = self.comparison_cols
                merged_df = self.get_data_with_groupings(
                    fact_table=grouping_table_name, dim_table=table_name,
                    cluster_table=cluster_table_name, grouping_cols=grouping_cols)                
                self.run_group_comparison(merged_df, grouping_cols, table_name, section)
    
    def run_aggregation(self, df: pd.DataFrame, group_by: list, base_table_name: str,
                    section: str, agg_pks=["group_id"], agg_subdir="aggregated"):
        """Calls EDADaemon to aggregate data."""
        self.eda.aggregate_data(df, group_by, base_table_name, section, agg_pks, agg_subdir)

    def run_group_comparison(self, df: pd.DataFrame, group_by: list, base_table_name: str,
                    section: str, gcomp_pks=["group_id"], gcomp_subdir="group_comps"):
        """Calls EDADaemon to compare groups."""
        self.eda.compare_groups(df, group_by, base_table_name, section, gcomp_pks, gcomp_subdir)

    def generate_visuals(self, section):
        """Calls visualization functions."""
        ## PATCH
        if self.compare_groups and self.comparison_cols:
            visualize_distinctive_features(self, section, self.comparison_cols)
        generate_corr_maps(self, section)
        generate_data_heatmaps(self, section)
