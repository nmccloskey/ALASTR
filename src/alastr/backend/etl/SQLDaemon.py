import re
import sqlite3
import pandas as pd
from alastr.backend.tools.logger import logger, _rel


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
        """
        Creates an empty sqlite relation with the correct PK.

        - If pk includes "AUTO" (case-insensitive), create a single autoincrement PK column.
        * If exactly one non-AUTO pk name is present, that becomes the autoincrement column.
            Example: pk=["ngram_id","AUTO"] -> ngram_id INTEGER PRIMARY KEY AUTOINCREMENT
        * If no non-AUTO pk names are present, default to "id".
            Example: pk=["AUTO"] -> id INTEGER PRIMARY KEY AUTOINCREMENT
        * If >1 non-AUTO pk names are present, AUTO is ignored and we fall back to normal composite PK.
            (SQLite autoincrement only works with a single INTEGER PRIMARY KEY column.)
        """
        try:
            PKs = list(pk) if pk is not None else []
            auto_pk = any(isinstance(x, str) and x.upper() == "AUTO" for x in PKs)
            non_auto_pks = [x for x in PKs if not (isinstance(x, str) and x.upper() == "AUTO")]

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Does the table already exist / have columns?
                cursor.execute(f'PRAGMA table_info("{table_name}");')
                existing_columns = {row[1] for row in cursor.fetchall()}

                if existing_columns:
                    return

                if auto_pk:
                    if len(non_auto_pks) == 1:
                        auto_col = non_auto_pks[0]
                    elif len(non_auto_pks) == 0:
                        auto_col = "id"
                    else:
                        # Can't do AUTOINCREMENT with composite PKs; fall back to normal mode
                        logger.warning(
                            f"{table_name}: PKs={PKs} include AUTO but also multiple PK columns. "
                            f"Falling back to composite PRIMARY KEY without AUTOINCREMENT."
                        )
                        auto_pk = False  # force normal path

                if auto_pk:
                    create_sql = f'''
                        CREATE TABLE "{table_name}" (
                            "{auto_col}" INTEGER PRIMARY KEY AUTOINCREMENT
                        );
                    '''
                    cursor.execute(create_sql)
                    conn.commit()
                    logger.info(f'Created table "{table_name}" with AUTOINCREMENT PK: {auto_col}')
                    return
                
                if not non_auto_pks:
                    raise ValueError(f"{table_name}: create_empty_table called with no PK columns (pk={pk})")

                pk_clause = ", ".join([f'"{c}"' for c in non_auto_pks])

                create_sql = f'''
                    CREATE TABLE "{table_name}" (
                        {", ".join([f'"{col}" INTEGER' for col in non_auto_pks])},
                        PRIMARY KEY ({pk_clause})
                    );
                '''
                cursor.execute(create_sql)
                conn.commit()
                logger.info(f'Created table "{table_name}" with PK: {", ".join(non_auto_pks)}')

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
                
    def _update_single_row(self, cursor: sqlite3.Cursor, table_name: str, row_data: dict):
        """
        Inserts or updates a single row in the database dynamically.

        Behavior:
        - If table PKs include "AUTO" (case-insensitive), treat the table as insert-only:
            * Ensure columns exist
            * INSERT the row (no PK existence check, no UPDATE)
            * If the inferred auto-id column is present with value None, exclude it so SQLite assigns it.

        - Otherwise, do the usual PK-driven upsert:
            * Validate PK values
            * INSERT stub if missing
            * ALTER TABLE for any new columns
            * UPDATE non-PK columns
        """
        try:
            # ---- PK discovery ----
            PKs = self.om.tables[table_name].get_pks() or []
            auto_pk = any(isinstance(pk, str) and pk.upper() == "AUTO" for pk in PKs)
            non_auto_pks = [pk for pk in PKs if not (isinstance(pk, str) and pk.upper() == "AUTO")]

            # In AUTO mode, infer the autoincrement column name:
            # - If exactly one non-AUTO pk is given, that's the auto-id column.
            # - If none are given, default to "id".
            # - If >1 are given, we can't autoincrement a composite PK; treat as normal mode.
            auto_id_col = None
            if auto_pk:
                if len(non_auto_pks) == 1:
                    auto_id_col = non_auto_pks[0]
                elif len(non_auto_pks) == 0:
                    auto_id_col = "id"
                else:
                    logger.warning(
                        f"{table_name}: PKs={PKs} include AUTO but multiple PK columns were provided. "
                        f"Disabling AUTO insert-only behavior and using normal PK-driven upsert."
                    )
                    auto_pk = False

            # ---- Sanitize incoming data keys once ----
            sanitized_data = {self.om.sanitize_column_name(col): val for col, val in row_data.items()}

            # ---- Discover existing columns ----
            cursor.execute(f'PRAGMA table_info("{table_name}");')
            existing_columns = {row[1] for row in cursor.fetchall()}  # row[1] = column name

            # ---- Add new columns dynamically if needed ----
            new_columns = {
                col: "INTEGER" if isinstance(val, int) else "REAL" if isinstance(val, float) else "TEXT"
                for col, val in sanitized_data.items()
                if col not in existing_columns
            }
            for column, data_type in new_columns.items():
                logger.info(f'Adding new column to "{table_name}": {column} {data_type}')
                cursor.execute(f'ALTER TABLE "{table_name}" ADD COLUMN "{column}" {data_type}')

            # ============================================================
            # AUTO-PK MODE: insert-only
            # ============================================================
            if auto_pk:
                auto_id_col_s = self.om.sanitize_column_name(auto_id_col) if auto_id_col else None

                insert_cols = []
                insert_vals = {}

                for col, val in sanitized_data.items():
                    # If the auto-id column is present but None, omit it so SQLite assigns it.
                    if auto_id_col_s and col == auto_id_col_s and val is None:
                        continue
                    insert_cols.append(col)
                    insert_vals[col] = val

                if not insert_cols:
                    logger.warning(f'{table_name}: AUTO insert skipped because row_data had no insertable columns.')
                    return

                col_clause = ", ".join([f'"{c}"' for c in insert_cols])
                ph_clause = ", ".join([f":{c}" for c in insert_cols])
                insert_sql = f'INSERT INTO "{table_name}" ({col_clause}) VALUES ({ph_clause})'

                cursor.execute(insert_sql, insert_vals)
                logger.debug(
                    f'{table_name}: AUTO insert ok (lastrowid={cursor.lastrowid}). '
                    f'Keys present: {sorted(list(insert_vals.keys()))[:8]}...'
                )
                return

            # ============================================================
            # NORMAL MODE: PK-driven upsert
            # ============================================================
            if not PKs:
                raise ValueError(f"Missing PKs in OutputManager table metadata for '{table_name}'")

            # Sanitize PK names too (critical if sanitize changes names)
            sanitized_pks = [self.om.sanitize_column_name(pk) for pk in PKs]

            # Ensure all PKs have values
            pk_values = {pk: sanitized_data.get(pk) for pk in sanitized_pks if sanitized_data.get(pk) is not None}
            if len(pk_values) != len(sanitized_pks):
                missing = [pk for pk in sanitized_pks if sanitized_data.get(pk) is None]
                raise ValueError(f"Missing PK values in '{table_name}': {missing}")

            # Check if the row exists
            where_clause = " AND ".join([f'"{pk}" = :{pk}' for pk in sanitized_pks])
            check_sql = f'SELECT 1 FROM "{table_name}" WHERE {where_clause}'
            cursor.execute(check_sql, pk_values)
            exists = cursor.fetchone()

            # Insert stub if it doesn't exist
            if not exists:
                insert_columns = ", ".join([f'"{pk}"' for pk in sanitized_pks])
                placeholders = ", ".join([f":{pk}" for pk in sanitized_pks])
                insert_sql = f'INSERT INTO "{table_name}" ({insert_columns}) VALUES ({placeholders})'
                cursor.execute(insert_sql, pk_values)
                logger.debug(f"{table_name}: inserted stub row for PKs={pk_values}")

            # Update statement: update non-PK columns only
            non_pk_cols = [c for c in sanitized_data.keys() if c not in set(sanitized_pks)]
            if not non_pk_cols:
                logger.debug(f"{table_name}: no non-PK columns to update for PKs={pk_values}")
                return

            update_clause = ", ".join([f'"{col}" = :{col}' for col in non_pk_cols])
            sql = f'UPDATE "{table_name}" SET {update_clause} WHERE {where_clause}'

            cursor.execute(sql, sanitized_data)
            logger.debug(f"{table_name}: updated row for PKs={pk_values} (cols={len(non_pk_cols)})")

        except Exception:
            logger.exception(
                f"Failed _update_single_row for table={table_name}. Row keys={list(row_data.keys())[:12]}..."
            )
            raise

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
