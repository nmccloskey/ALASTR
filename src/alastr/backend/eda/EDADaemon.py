import os
import logging
logger = logging.getLogger("CustomLogger")
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
import scipy.cluster.hierarchy as sch
from scipy.stats import entropy
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import jensenshannon


class EDADaemon:

    def __init__(self, OM):
        self.cluster_cols = ["KMeans_Cluster", "Hier_Cluster", "DBSCAN_Cluster"]
        self.om = OM

    def run_kmeans(self, data: pd.DataFrame, k_range: range = range(2, 11)) -> list:
        """
        Runs K-Means clustering on the dataset and selects the optimal number of clusters using the silhouette score.

        Args:
            data (pd.DataFrame): Input DataFrame.
            k_range (range): Range of k values to consider (default: 2-10).

        Returns:
            list: Cluster assignments for each sample.
        """   
        if data.shape[0] < 2:
            logger.warning("Not enough data for KMeans clustering.")
            return [np.nan] * data.shape[0]

        # Adjust k_range if too small
        max_k = min(data.shape[0] - 1, max(k_range))  # cap at n_samples
        if max_k < 2:
            return [np.nan] * data.shape[0]
        k_range = range(2, max_k + 1)

        models = {}
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=8).fit(data)
            models[k] = kmeans

        try:
            best_k = max(k_range, key=lambda k: silhouette_score(data, models[k].labels_))
            logger.info(f"Best k: {best_k}.")
        except ValueError as e:
            logger.error(f"Silhouette scoring failed: {e}")
            return [np.nan] * data.shape[0]
        
        # Assign clusters using the best k
        best_kmeans = models[best_k]
        cluster_assignments = best_kmeans.labels_

        logger.info(f"K-Means selected k={best_k} with silhouette score: {silhouette_score(data, cluster_assignments):.4f}")
        return cluster_assignments.tolist()

    def run_hierarchical_clustering(self, data: pd.DataFrame, distance_threshold: float = None) -> list:
        """
        Runs Hierarchical Agglomerative Clustering and determines cluster assignments.

        Args:
            data (pd.DataFrame): Input DataFrame.
            distance_threshold (float, optional): Threshold to determine clusters. If None, it selects an optimal cut-off.

        Returns:
            list: Cluster assignments for each sample.
        """
        if data.shape[0] < 2:
            logger.warning("Not enough data for Hierarchical clustering.")
            return [np.nan] * data.shape[0]

        # Compute the linkage matrix
        linkage_matrix = sch.linkage(data, method='ward')

        # Automatically determine distance threshold if not provided
        if distance_threshold is None:
            # Use inconsistency criterion to determine a reasonable cutoff
            distance_threshold = 0.7 * max(linkage_matrix[:, 2])  # 70% of max distance

        clusters = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')

        logger.info(f"Hierarchical Clustering completed with distance threshold: {distance_threshold:.4f}")
        return clusters.tolist()

    def run_dbscan(self, data: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> list:
        """
        Runs DBSCAN clustering on the dataset.

        Args:
            data (pd.DataFrame): Input DataFrame.
            eps (float): The maximum distance between two samples for one to be considered as in the neighborhood (default: 0.5).
            min_samples (int): The number of samples required to form a dense region (default: 5).

        Returns:
            list: Cluster assignments (where -1 represents noise).
        """
        if data.shape[0] < 5:
            logger.warning(f"Not enough data for DBSCAN clustering.")
            return [np.nan] * data.shape[0] 
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
        labels = dbscan.labels_  # -1 represents noise

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"DBSCAN found {num_clusters} clusters (including noise).")
        return labels.tolist()

    def calculate_entropy(self, cluster_labels: list, predefined_groups: list) -> float:
        """
        Computes entropy to measure agreement between predefined groups and computed clusters.

        Args:
            cluster_labels (list): List of computed cluster labels.
            predefined_groups (list): List of predefined group labels.

        Returns:
            float: Entropy score (lower is better).
        """
        if len(cluster_labels) != len(predefined_groups):
            raise ValueError("Cluster labels and predefined groups must have the same length.")

        # Compute joint distribution
        joint_distribution = Counter(zip(cluster_labels, predefined_groups))
        total = sum(joint_distribution.values())
        joint_probs = np.array(list(joint_distribution.values())) / total

        entropy_score = entropy(joint_probs, base=2)  # Log base 2 for interpretability
        logger.info(f"Entropy score computed: {entropy_score:.4f}")
        return entropy_score

    def run_clustering(self, df: pd.DataFrame, pk_columns: list) -> pd.DataFrame:
        """
        Cleans, imputes, scales, and clusters the input DataFrame using KMeans, Hierarchical, and DBSCAN.
        """
        try:
            if df is None or df.empty:
                logger.error("Input DataFrame is empty. Clustering cannot proceed.")
                raise ValueError("Input DataFrame is empty.")

            # Step 1: Extract numeric features, drop PKs
            data = df.drop(columns=pk_columns, errors='ignore').select_dtypes(include=[np.number])

            # Step 2: Replace infinities with NaN
            data.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Step 3: Impute missing values with median
            imputer = SimpleImputer(strategy="median")
            data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

            # Step 4: Standardize features
            scaler = StandardScaler()
            data_scaled = pd.DataFrame(scaler.fit_transform(data_imputed), columns=data.columns)

            # Step 5: Clustering
            cluster_data = df[pk_columns].copy()

            logger.info(f"Running clustering on {data_scaled.shape[0]} samples with {data_scaled.shape[1]} features.")
            logger.info(f"Any NaNs left? {data_scaled.isna().any().any()}")
            logger.info(f"Value range: min={data_scaled.min().min()}, max={data_scaled.max().max()}")

            kmeans_clusters = self.run_kmeans(data_scaled)
            hier_clusters = self.run_hierarchical_clustering(data_scaled)
            dbscan_clusters = self.run_dbscan(data_scaled)

            num_samples = df.shape[0]
            for name, labels in {
                "KMeans_Cluster": kmeans_clusters,
                "Hier_Cluster": hier_clusters,
                "DBSCAN_Cluster": dbscan_clusters,
            }.items():
                if len(labels) != num_samples:
                    logger.error(f"{name}: expected {num_samples} labels, got {len(labels)}")
                    raise ValueError(f"{name} produced incorrect number of labels.")
                cluster_data[name] = labels

            logger.info("Clustering completed successfully.")
            return cluster_data

        except Exception as e:
            logger.error(f"Error during clustering: {e}")
            raise

    def cv(self, x):
        mean = np.nanmean(x)
        return np.nanstd(x) / mean if mean > 0 else None

    def skew(self, x):
        return stats.skew(x) if len(x) > 1 else None

    def kurtosis(self, x):
        return stats.kurtosis(x) if len(x) > 1 else None

    def sem(self, x):
        return stats.sem(x) if len(x) > 1 else None

    def mode(self, x):
        most_common = Counter(x).most_common(1)
        return most_common[0][0] if most_common else None

    def mode_count(self, x):
        most_common = Counter(x).most_common(1)
        return most_common[0][1] if most_common else None

    def entropy(self, x):
        value_counts = Counter(x).values()
        return stats.entropy(list(value_counts), base=2)

    def cohen_d(self, x, y):
        """
        Computes Cohen's d effect size between two distributions.

        Args:
            x (np.array or pd.Series): First group.
            y (np.array or pd.Series): Second group.

        Returns:
            float: Cohen's d value.
        """
        mean_x, mean_y = np.nanmean(x), np.nanmean(y)
        pooled_std = np.sqrt((np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2)
        
        if pooled_std == 0:  # Avoid division by zero
            return np.nan

        return (mean_x - mean_y) / pooled_std

    def percent_difference_of_means(self, x, y):
        """
        Computes the absolute percent difference between the means of two arrays.

        Args:
            x (array-like): First sample of numeric values.
            y (array-like): Second sample of numeric values.

        Returns:
            float: Percent difference, or NaN if undefined.
        """
        try:
            mx = np.nanmean(x)
            my = np.nanmean(y)
            denom = (mx + my) / 2
            pdiff = abs(mx - my) / denom if denom > 0 else np.nan
            return pdiff
        except Exception as e:
            logger.error(f"Failed to calculate mean percent difference: {e} | x: {x}, y: {y}")
            return np.nan
    
    def set_jaccard_distance(self, x, y):
        """
        Computes Jaccard distance (1 - overlap) between unique value sets of two arrays.

        Args:
            x (array-like): First categorical sample.
            y (array-like): Second categorical sample.

        Returns:
            float: Jaccard distance, or NaN if undefined.
        """
        try:
            set_x = set(pd.Series(x).dropna())
            set_y = set(pd.Series(y).dropna())
            union = set_x | set_y
            intersection = set_x & set_y
            return 1 - len(intersection) / len(union) if union else np.nan
        except Exception as e:
            logger.error(f"Set-based Jaccard failed: {e}")
            return np.nan

    def jensen_shannon_distance(self, x, y):
        """
        Computes Jensen-Shannon distance between category distributions of two samples.

        Args:
            x (array-like): First categorical sample.
            y (array-like): Second categorical sample.

        Returns:
            float: Jensen-Shannon distance, or NaN if undefined.
        """
        try:
            x = pd.Series(x).dropna()
            y = pd.Series(y).dropna()
            categories = sorted(set(x) | set(y))
            px = np.array([np.mean(x == c) for c in categories])
            py = np.array([np.mean(y == c) for c in categories])
            return jensenshannon(px, py)
        except Exception as e:
            logger.error(f"Jensen-Shannon distance failed: {e}")
            return np.nan

    def get_grouping_combos(self, group_by, gtype="agg"):
        """
        Generates combinations of group_by tiers for aggregation or comparison.

        Args:
            group_by (list[str]): Base tier names (e.g., ["test", "narrative"])
            gtype (str): "agg" or "gcomp"

        Returns:
            list[tuple]: List of tier combinations as tuples
        """
        combinations = []

        if gtype == "agg":
            if self.om.all_aggregation_combos:
                combinations += list(itertools.chain.from_iterable(
                    itertools.combinations(group_by, r) for r in range(1, len(group_by) + 1)
                ))
            elif self.om.aggregation_combos:
                combinations += list(set(tuple(combo) for combo in self.om.aggregation_combos))

            if self.om.cluster and self.om.aggregate_with_clusters:
                combinations += [(col,) for col in self.cluster_cols]

        elif gtype == "gcomp":
            if self.om.all_comparison_combos:
                combinations += list(itertools.chain.from_iterable(
                    itertools.combinations(group_by, r) for r in range(1, len(group_by) + 1)
                ))
            elif self.om.comparison_combos:
                combinations += list(set(tuple(combo) for combo in self.om.comparison_combos))

            if self.om.cluster and self.om.compare_with_clusters:
                combinations += [(col,) for col in self.cluster_cols]

        return combinations

    def get_grouping_info(self, df: pd.DataFrame, group_by: list, pk_cols: list, gtype="agg"):
        """
        Extracts grouping and indexing metadata from a DataFrame based on specified groupings and primary keys.

        Args:
            df (pd.DataFrame): Input data.
            group_by (list): Columns to group by.
            pk_cols (list): Primary key columns to exclude from aggregation.

        Returns:
            Tuple: (numeric_cols, categorical_cols, group_keys, group_idxs, group_by_keys)
        """
        if df.empty:
            logger.error("Input DataFrame is empty. Cannot generate grouping info.")
            return None, None, None, None, None

        try:
            logger.info(f"Generating group combinations using {group_by}")
            df[group_by] = df[group_by].fillna("_other")

            no_agg_cols = list(set(group_by + self.cluster_cols + pk_cols))
            numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(no_agg_cols).tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.difference(no_agg_cols).tolist()

            combinations = self.get_grouping_combos(group_by, gtype)

            groupings = [(combo, df.groupby(list(combo))) for combo in combinations]

            group_keys, group_idxs, group_by_keys = [], [], []
            for combo, grouping in groupings:
                for key, idxs in grouping.groups.items():
                    k = "_".join(map(str, key)) if isinstance(key, tuple) else str(key)
                    group_keys.append(k)
                    group_idxs.append(idxs)
                    group_by_keys.append("_".join(combo))

            return numeric_cols, categorical_cols, group_keys, group_idxs, group_by_keys

        except Exception as e:
            logger.exception(f"Failed to compute grouping info: {e}")
            return None, None, None, None, None

    def aggregate_data(self, df: pd.DataFrame, group_by: list, base_table_name: str,
                    section: str, agg_pks=["group_id"], agg_subdir="aggregated"):
        """
        Aggregates data across defined groups using statistical and custom functions.

        Args:
            df (pd.DataFrame): Input data.
            group_by (list): Columns to group by.
            base_table_name (str): Name of base table for metadata and output naming.
            agg_pks (list): Primary key columns for aggregation table.
            agg_subdir (list): Subdirectory path for saving aggregated tables.
        """
        try:
            subdir = self.om.tables[base_table_name].get_subdir()
            path = os.path.join(subdir, agg_subdir)
            pk_cols = self.om.tables[base_table_name].get_pks()

            numeric_cols, categorical_cols, group_keys, group_idxs, group_by_keys = self.get_grouping_info(df, group_by, pk_cols)
            if numeric_cols is None:
                return
            logger.info(f"Numeric columns: {numeric_cols}")
            logger.info(f"Categorical columns: {categorical_cols}")

            num_agg_funcs = {
                "sum": np.nansum, "min": np.nanmin, "max": np.nanmax, "mean": np.nanmean,
                "median": np.nanmedian, "stdev": np.nanstd, "cv": self.cv,
                "skew": self.skew, "kurtosis": self.kurtosis, "sem": self.sem
            }
            cat_agg_funcs = {"mode": self.mode, "mode_count": self.mode_count, "entropy": self.entropy}

            for func_name, func in {**num_agg_funcs, **cat_agg_funcs}.items():
                agg_rows = []

                for i, idxs_i in enumerate(group_idxs):
                    agg_row = {
                        "group_id": i,
                        "group_name": group_keys[i],
                        "group_by": group_by_keys[i],
                        "count": len(idxs_i)
                    }

                    is_cat = func_name in cat_agg_funcs
                    cols = categorical_cols if is_cat else numeric_cols

                    for col in cols:
                        if func_name == "sum" and not col.startswith(("num_", "total_")):
                            # logger.warning(f"Skipping summation for '{col}' using '{func_name}'")
                            continue
                        try:
                            # logger.info(f"Performing '{func_name}' for '{col}'")
                            val = func(df.loc[idxs_i, col])
                            # logger.info(f"val: {val}")
                        except Exception as e:
                            logger.warning(f"Aggregation failed for '{col}' using '{func_name}': {e}")
                            val = np.nan
                        agg_row[f"{col}_{func_name}"] = val

                    agg_rows.append(agg_row)

                agg_df = pd.DataFrame(agg_rows)
                agg_table_name = f"{base_table_name}_agg_{func_name}"
                
                if len([col for col in agg_df.columns if col not in ("group_id", "group_name", "group_by", "count")]) == 0:
                    logger.warning(f"No aggregation data for '{agg_table_name}' - no numeric/categorical aggregates.")
                    continue
                self.om.create_table(name=agg_table_name, sheet_name=func_name,
                                    section=section, subdir=path,
                                    file_name=f"{base_table_name}_agg.xlsx", 
                                    primary_keys=agg_pks)
                self.om.tables[agg_table_name].update_data(agg_df.to_dict(orient="records"))
                self.om.tables[agg_table_name].export_to_excel()
                logger.info(f"Aggregation complete: '{agg_table_name}' shape: {agg_df.shape}")

        except Exception as e:
            logger.exception(f"Failed to aggregate data for '{base_table_name}': {e}")

    def compare_groups(self, df: pd.DataFrame, group_by: list, base_table_name: str,
                    section: str, gcomp_pks=["group_id"], gcomp_subdir="group_commps"):
        """
        Compares each group with every other group using overlap counts and effect size metrics (Cohen's d).

        Args:
            df (pd.DataFrame): Input data.
            group_by (list): Columns to group by.
            base_table_name (str): Name of base table for metadata and output naming.
            gcomp_pks (list): Primary keys for group comparison tables.
            gcomp_subdir (str): Subdirectory path for saving comparison tables.
        """
        try:
            subdir = self.om.tables[base_table_name].get_subdir()
            path = os.path.join(subdir, gcomp_subdir)
            pk_cols = self.om.tables[base_table_name].get_pks()

            numeric_cols, cat_cols, group_keys, group_idxs, group_by_keys = self.get_grouping_info(df, group_by, pk_cols, "gcomp")
            if numeric_cols is None and cat_cols is None:
                return

            count_rows = []
            prop_rows = []
            feature_rows = []
            comp_id = 1

            for i, idxs_i in enumerate(group_idxs):
                count_row_base = {
                    "group_id": i,
                    "group_name": group_keys[i],
                    "group_by": group_by_keys[i],
                    "count": len(idxs_i)
                }
                count_row = count_row_base.copy()
                prop_row = count_row_base.copy()
                feature_row = count_row_base.copy()
                
                for j, idxs_j in enumerate(group_idxs):
                    if i == j:
                        continue
                    
                    try:
                        overlap = len(set(idxs_i) & set(idxs_j))
                        count_row[f"count_{j}"] = overlap
                        prop_row[f"prop_{j}"] = overlap / len(idxs_i)

                        for col in numeric_cols + cat_cols:
                            feature_row = {
                                "comp_id": comp_id,
                                "group_id_1": i,
                                "group_id_2": j,
                                "group_name_1": group_keys[i],
                                "group_name_2": group_keys[j],
                                "group_by_1": group_by_keys[i],
                                "group_by_2": group_by_keys[j],
                                "feature": col
                            }

                            i_group, j_group = df.loc[idxs_i, col], df.loc[idxs_j, col]                 

                            if col in numeric_cols:
                                try:
                                    d_cohen = self.cohen_d(i_group, j_group)
                                    pdiff = self.percent_difference_of_means(i_group, j_group)
                                except Exception as e:
                                    logger.warning(f"Cohen's d / % diff failed for '{col}' between groups {i} and {j}: {e}")
                                    d_cohen = pdiff = np.nan

                                feature_row["cohen_d"] = d_cohen
                                feature_row["perc_diff_means"] = pdiff

                            if col in cat_cols:
                                try:
                                    d_jaccard = self.set_jaccard_distance(i_group, j_group)
                                    js_dist = self.jensen_shannon_distance(i_group, j_group)
                                except Exception as e:
                                    logger.warning(f"Jaccard failed for '{col}' between groups {i} and {j}: {e}")
                                    d_jaccard = js_dist = np.nan
                                
                                feature_row["jaccard_d"] = d_jaccard
                                feature_row["js_dist"] = js_dist
                            
                            feature_rows.append(feature_row)
                            comp_id += 1

                    except Exception as e:
                        logger.warning(f"Group comparison between {i} and {j} failed: {e}")
                
                count_rows.append(count_row)
                prop_rows.append(prop_row)

            for label, result_df in zip(["counts", "props", "effect_sizes"], [pd.DataFrame(count_rows), pd.DataFrame(prop_rows), pd.DataFrame(feature_rows)]):
                table_name = f"{base_table_name}_group_{label}"
                result_pks = ["comp_id"] if label == "effect_sizes" else gcomp_pks
                self.om.create_table(name=table_name, sheet_name=label, section=section,
                                     subdir=path, file_name=f"{base_table_name}_group_comps.xlsx", primary_keys=result_pks)
                self.om.tables[table_name].update_data(result_df.to_dict(orient="records"))
                self.om.tables[table_name].export_to_excel()
                logger.info(f"Group comparison complete: '{table_name}' shape: {result_df.shape}")

        except Exception as e:
            logger.exception(f"Failed to compare groups for '{base_table_name}': {e}")

    def get_distinctive_features(self, table):
        """
        Identifies top features with largest Cohen's d effect sizes across any group pairs.

        Args:
            table (Table): The table to analyze.

        Returns:
            pd.DataFrame: DataFrame with columns [feature, group1, group2, effect_size]
        """
        logger.info(f"Getting distinctive features for {table.name}")
        try:
            gcomp_df = self.om.tables[f"{table.name}_group_effect_sizes"].get_data()

            if gcomp_df is None or gcomp_df.empty:
                logger.warning(f"No group comparison data found for {table.name}")
                return pd.DataFrame()

            max_val_df = gcomp_df.dropna(subset=["cohen_d"])
            max_val_df = max_val_df[max_val_df["cohen_d"] > self.om.cohen_d_threshold]
            max_val_df = max_val_df.sort_values("cohen_d", ascending=False).reset_index(drop=True)


            # max_val_df = pd.DataFrame({
            #     "feature": features,
            #     "group1": group_ids,
            #     "group2": group2,
            #     "effect_size": max_vals.values
            # })

            return max_val_df

        except Exception as e:
            logger.exception(f"Failed to get distinctive features for '{table.name}': {e}")
            return pd.DataFrame()






    # def get_distinctive_features(self, table):
    #     """
    #     Identifies top features with largest Cohen's d effect sizes across any group pairs.

    #     Args:
    #         table (Table): The table to analyze.

    #     Returns:
    #         pd.DataFrame: DataFrame with columns [feature, group1, group2, effect_size]
    #     """
    #     logger.info(f"Getting distinctive features for {table.name}")
    #     try:
    #         gcomp_df = self.om.tables[f"{table.name}_group_effect_sizes"].get_data()

    #         if gcomp_df is None or gcomp_df.empty:
    #             logger.warning(f"No group comparison data found for {table.name}")
    #             return pd.DataFrame()

    #         # metric_cols = list(gcomp_df.columns)[4:]

    #         # # Replace None with np.nan, then safely compute max abs
    #         # gcomp_df[metric_cols] = gcomp_df[metric_cols].apply(pd.to_numeric, errors='coerce')

    #         # group_ids = [
    #         #     gcomp_df.loc[gcomp_df[col].abs().idxmax(), gcomp_df.columns[0]]
    #         #     if gcomp_df[col].notna().any() else None
    #         #     for col in metric_cols
    #         # ]

    #         # max_vals = gcomp_df[metric_cols].abs().max(skipna=True)

    #         # group2 = [col.rsplit("_", 1)[-1] for col in metric_cols]
    #         # features = [col.rsplit("_", 1)[0] for col in metric_cols]


    #         max_val_df = pd.DataFrame({
    #             "feature": features,
    #             "group1": group_ids,
    #             "group2": group2,
    #             "effect_size": max_vals.values
    #         })

    #         max_val_df = max_val_df.dropna(subset=["effect_size"])
    #         max_val_df = max_val_df[max_val_df["effect_size"] > self.om.cohen_d_threshold]
    #         max_val_df = max_val_df.sort_values("effect_size", ascending=False).reset_index(drop=True) #.head(self.om.max_feature_visuals).reset_index(drop=True)

    #         return max_val_df

    #     except Exception as e:
    #         logger.exception(f"Failed to get distinctive features for '{table.name}': {e}")
    #         return pd.DataFrame()