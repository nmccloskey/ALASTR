import re
import sqlite3
import pandas as pd
import logging
logger = logging.getLogger("CustomLogger")


class SQLDaemon:
    def __init__(self, OM = None):
        self.om = OM
        self.db_path = OM.db_path
    
    def sanitize_column_name(self, col_name):
        """
        Cleans and formats a column name to be SQL-safe.

        - Replaces special characters (`=`, `-`, `space`) with `_`
        - If the column starts with a digit, prefixes it with `_`
        - Wraps the column name in double quotes to handle reserved words

        Args:
            col_name (str): The original column name.

        Returns:
            str: A sanitized and SQL-safe column name.
        """
        col_name = col_name.replace("=", "_").replace("-", "_").replace(" ", "_").replace("*", "")
        if re.match(r"^\d", col_name):
            col_name = f"_{col_name}"
        return col_name

    def create_empty_table(self, table_name, pk):
        """Creates an empty sqlite relation with the correct PK."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name});")
            existing_columns = {row[1] for row in cursor.fetchall()}

            if not existing_columns:
                # If table doesn't exist, create it with proper PKs
                pk_clause = ", ".join(pk)
                create_sql = f"""
                    CREATE TABLE {table_name} (
                        {", ".join(f"{col} INTEGER" for col in pk)},
                        PRIMARY KEY ({pk_clause})
                    );
                """
                cursor.execute(create_sql)
                conn.commit()
                logger.info(f"Created table {table_name} with PK: {pk_clause}")
        except sqlite3.OperationalError as e:
            logger.error(f"SQL error executing query: {e}")
        except Exception as e:
            logger.error(f"Error creating table '{table_name}': {e}")

    def update_database(self, table_name, update_data):
        """
        Updates an SQLite database table with new data.

        Args:
            table_name (str): The name of the table to update.
            update_data (dict or list): The data to insert or update.
                                        If a list, it assumes multiple rows.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if isinstance(update_data, list):
                    for row in update_data:
                        self._update_single_row(cursor, table_name, row)
                else:
                    self._update_single_row(cursor, table_name, update_data)

                conn.commit()

        except sqlite3.OperationalError as e:
            logger.error(f"SQL error updating '{table_name}': {e}")
   
    def _update_single_row(self, cursor, table_name, row_data):
        """
        Updates a single row in the database dynamically, using primary keys inferred from the row data.

        Args:
            cursor (sqlite3.Cursor): SQLite cursor object.
            table_name (str): Name of the table.
            row_data (dict): The row data to insert or update.
        """
        # Identify primary keys (PKs) dynamically
        # PKs = [col for col in row_data.keys() if col.endswith("_id")]
        PKs = self.om.tables[table_name].get_pks()
        if not PKs:
            raise ValueError(f"Missing PKs in update data for table '{table_name}'")

        # Ensure all PKs have values
        pk_values = {pk: row_data.get(pk) for pk in PKs if row_data.get(pk) is not None}
        if len(pk_values) != len(PKs):
            raise ValueError(f"One or more primary keys are missing values in '{table_name}'")

        # Check if the row exists
        where_clause = " AND ".join([f"{pk} = :{pk}" for pk in pk_values.keys()])
        check_sql = f"SELECT 1 FROM {table_name} WHERE {where_clause}"
        cursor.execute(check_sql, pk_values)
        exists = cursor.fetchone()

        # Insert row if it does not exist
        if not exists:
            insert_columns = ", ".join(pk_values.keys())
            placeholders = ", ".join([f":{pk}" for pk in pk_values.keys()])
            insert_sql = f"INSERT INTO {table_name} ({insert_columns}) VALUES ({placeholders})"
            cursor.execute(insert_sql, pk_values)

        # Retrieve existing columns from the table
        cursor.execute(f"PRAGMA table_info({table_name});")
        existing_columns = {row[1] for row in cursor.fetchall()}

        # Sanitize and check for new columns
        sanitized_data = {self.om.sanitize_column_name(col): val for col, val in row_data.items()}
        new_columns = {
            col: "INTEGER" if isinstance(val, int) else "REAL" if isinstance(val, float) else "TEXT"
            for col, val in sanitized_data.items() if col not in existing_columns
        }

        # Add new columns dynamically if needed
        for column, data_type in new_columns.items():
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN \"{column}\" {data_type}")

        # Construct the update statement dynamically
        update_clause = ", ".join([f"\"{col}\" = :{col}" for col in sanitized_data.keys()])
        sql = f"UPDATE {table_name} SET {update_clause} WHERE {where_clause}"

        # Execute the update statement
        cursor.execute(sql, sanitized_data)

    def access_data(self, table_name, columns='*', filters=None):
        """
        Retrieves data from an SQLite database table with optional column selection and filtering.

        Args:
            table_name (str): The name of the table to retrieve data from.
            columns (str or list, optional): Columns to select. Defaults to '*' (all columns).
            filters (dict, optional): A dictionary of column-value pairs to filter the results.

        Returns:
            pd.DataFrame or None: A DataFrame with the retrieved data or None if empty/error.

        Raises:
            sqlite3.OperationalError: If the table does not exist or a query issue arises.
        """
        try:
            logger.info(f"Connecting to database at: {self.db_path}")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Ensure table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = {row[0] for row in cursor.fetchall()}
            if table_name not in tables:
                logger.error(f"Table '{table_name}' does not exist. Available tables: {tables}")
                return None
            
            # Retrieve valid columns
            cursor.execute(f"PRAGMA table_info({table_name});")
            available_columns = {row[1] for row in cursor.fetchall()}
            # logger.info(f"Available columns in '{table_name}': {available_columns}")

            # Sanitize and validate column selection
            if isinstance(columns, list):
                valid_columns = [self.om.sanitize_column_name(col) for col in columns if col in available_columns]
                cols = ', '.join(valid_columns) if valid_columns else '*'
            else:
                cols = columns if columns == '*' else ', '.join([self.om.sanitize_column_name(columns)])

            if not cols.strip():
                logger.error(f"No valid columns provided for query on '{table_name}'.")
                return None

            # Construct WHERE clause dynamically if filters are provided
            where_clause = ""
            params = {}

            if filters and isinstance(filters, dict):
                valid_filters = {col: val for col, val in filters.items() if col in available_columns}
                if valid_filters:
                    where_conditions = [f"{col} = :{col}" for col in valid_filters.keys()]
                    where_clause = " WHERE " + " AND ".join(where_conditions)
                    params = valid_filters

            # Execute query
            query = f"SELECT {cols} FROM {table_name}{where_clause};"
            logger.info(f"Executing query: {query} with parameters: {params}")
            df = pd.read_sql_query(query, conn, params=params)

            if df.empty:
                logger.warning(f"Table '{table_name}' is empty. No data retrieved.")
                return None

            logger.info(f"Successfully retrieved data from table '{table_name}' with {df.shape[0]} rows and {df.shape[1]} columns.")
            return df

        except sqlite3.OperationalError as e:
            logger.error(f"SQL error accessing data from '{table_name}': {e}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error accessing data from '{table_name}': {e}")
            return None

        finally:
            conn.close()
