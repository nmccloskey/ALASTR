
    def fetch_grouping_data(self, tiers, basic_table_name):
        """
        Fetches categorical (grouping) data from the 'basic' table.

        Returns:
            pd.DataFrame: Dataframe containing document/sample metadata for grouping.
        """
        try:
            df = self.access_data(output_dir=None, table_name=basic_table_name, columns=tiers)
            if df is None:
                return None

            logger.info(f"Grouping columns identified: {df.columns.tolist()}")

            return df

        except Exception as e:
            logger.error(f"Error fetching grouping data: {e}")
            return None
        

    def fetch_numerical_data(self, table_name):
        """
        Fetches numerical data from a given table.

        Args:
            table_name (str): The name of the table to retrieve numerical values.

        Returns:
            pd.DataFrame: A dataframe containing numerical data from the table.
        """
        try:
            df = self.access_data(output_dir=None, table_name=table_name)
            if df is None:
                return None

            # Select only numeric columns
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            logger.info(f"Numerical columns identified in '{table_name}': {numeric_cols}")

            return df

        except Exception as e:
            logger.error(f"Error fetching numerical data from '{table_name}': {e}")
            return None
    

    def fetch_and_merge_data(self, tiers, table_name):
        """
        Merges categorical data (from 'basic') with numerical data.

        Args:
            numerical_table (str): The table containing numerical values.

        Returns:
            pd.DataFrame: A merged dataframe ready for analysis.
        """
        try:
            # Determine level of analysis.
            sentence = table_name.endswith('sent')

            if sentence:
                tiers = ['sent_id'] + tiers 
                basic_table_name = 'basic_sent'
            else:
                basic_table_name = 'basic_doc'

            grouping_df = self.fetch_grouping_data(['doc_id'] + tiers, basic_table_name)
            numerical_df = self.fetch_numerical_data(table_name)

            if grouping_df is None or numerical_df is None:
                logger.error("Failed to fetch necessary data for merging.")
                return None
            
            merge_cols = ["doc_id"]
            if "sent_id" in grouping_df.columns and "sent_id" in numerical_df.columns:
                merge_cols.append("sent_id")

            merged_df = pd.merge(grouping_df, numerical_df, on=merge_cols+tiers, how="left")
            # logger.info(f"Merged dataframe: {merged_df}")
            logger.info(f"Successfully merged data from {basic_table_name} and {table_name}.")

            return merged_df

        except Exception as e:
            logger.error(f"Error merging data for analysis: {e}")
            return None


def aggregate_sents(sent_data):
    """
    Calculate summary statistics for each numerical column in a list of sentence data dictionaries.
    
    Args:
        sent_data (list of dict): A list of dictionaries where each dictionary represents sentence data with numerical values.

    Returns:
        dict: A dictionary containing summary statistics for each numerical column.
    """
    aggregated = {}
    cols = set(col for sent in sent_data for col in sent.keys())

    for col in cols:
        values = [sent[col] for sent in sent_data if col in sent and isinstance(sent[col], (int, float))]
        
        if values:
            new_col_prefix = f"sent_{col}"
            aggregated[f"avg_{new_col_prefix}"] = np.nanmean(values)
            aggregated[f"median_{new_col_prefix}"] = np.nanmedian(values)
            aggregated[f"min_{new_col_prefix}"] = np.min(values)
            aggregated[f"max_{new_col_prefix}"] = np.max(values)
            aggregated[f"std_{new_col_prefix}"] = np.nanstd(values)
            aggregated[f"cv_{new_col_prefix}"] = (np.nanstd(values) / np.nanmean(values)) if np.nanmean(values) > 0 else None
            aggregated[f"skew_{new_col_prefix}"] = stats.skew(values) if len(values) > 1 else None
            aggregated[f"kurtosis_{new_col_prefix}"] = stats.kurtosis(values) if len(values) > 1 else None
            aggregated[f"std_error_{new_col_prefix}"] = stats.sem(values) if len(values) > 1 else None
    
    # Handling categorical columns
    categorical_cols = {col for sent in sent_data for col in sent.keys() if isinstance(sent[col], str)}
    
    for col in categorical_cols:
        cat_values = [sent[col] for sent in sent_data if col in sent]
        if cat_values:
            mode_value, mode_count = Counter(cat_values).most_common(1)[0]
            aggregated[f"mode_sent_{col}"] = mode_value
            aggregated[f"mode_sent_{col}_count"] = mode_count
            aggregated[f"entropy_sent_{col}"] = stats.entropy(list(Counter(cat_values).values()), base=2)
    
    return aggregated