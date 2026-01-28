from __future__ import annotations

import re
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from alastr.backend.tools.logger import logger, _rel


ColumnsArg = Union[str, Sequence[str]]


class SQLDaemon:
    """Thin SQLite access layer governed by OutputManager.

    Design intent
    -------------
    - OutputManager owns SQLDaemon and exposes wrapper methods (create table, insert/update, query, export).
    - SQLDaemon should be usable in normal runs via a de facto singleton, while still being easy to
      re-instantiate in tests.

    Notes
    -----
    - Column-name sanitization should match OutputManager/Table behavior. If OutputManager provides
      `sanitize_column_name`, we defer to it; otherwise we fall back to this class method.
    """

    _instance: Optional["SQLDaemon"] = None

    def __init__(self, OM: Any = None, db_path: Optional[str] = None):
        if OM is None and db_path is None:
            raise ValueError("SQLDaemon requires either OM (with db_path) or an explicit db_path.")
        self.om = OM
        self.db_path = str(db_path) if db_path is not None else str(getattr(OM, "db_path"))

    # -------------------------
    # De facto singleton factory
    # -------------------------

    @classmethod
    def get_instance(cls, OM: Any = None, db_path: Optional[str] = None, reset: bool = False) -> "SQLDaemon":
        """Return a cached SQLDaemon instance for typical program runs.

        Use `reset=True` in tests or when you need a clean instance.
        """
        if reset or cls._instance is None:
            cls._instance = cls(OM=OM, db_path=db_path)
        else:
            # If OM changes between calls, update reference (db_path should remain same in-run).
            if OM is not None:
                cls._instance.om = OM
        return cls._instance

    # -------------------
    # Sanitization helpers
    # -------------------

    @staticmethod
    def _sanitize_column_name_fallback(col_name: str) -> str:
        """Make a string SQL-safe as a column name.

        - Replace special characters (`=`, `-`, whitespace) with `_`
        - Replace other non-alphanumerics with `_`
        - Collapse multiple underscores
        - If it starts with a digit, prefix `_`
        """
        if col_name is None:
            return col_name
        s = str(col_name).strip()
        s = re.sub(r"[=\-\s]+", "_", s)
        s = re.sub(r"[^0-9A-Za-z_]", "_", s)
        s = re.sub(r"_+", "_", s)
        if re.match(r"^\d", s):
            s = "_" + s
        return s

    def sanitize_column_name(self, col_name: str) -> str:
        """Defer to OutputManager's sanitization if present; else use fallback."""
        if self.om is not None and hasattr(self.om, "sanitize_column_name"):
            return self.om.sanitize_column_name(col_name)
        return self._sanitize_column_name_fallback(col_name)

    # ------------------------
    # Schema / table creation
    # ------------------------

    def create_empty_table(self, table_name: str, pk: Optional[List[str]]):
        """Create an empty sqlite table if it does not yet exist.

        PK behavior
        ----------
        - If pk includes "AUTO" (case-insensitive), create a single autoincrement PK column.
          * If exactly one non-AUTO pk name is present, that becomes the autoincrement column.
              Example: pk=["ngram_id","AUTO"] -> ngram_id INTEGER PRIMARY KEY AUTOINCREMENT
          * If no non-AUTO pk names are present, default to "id".
              Example: pk=["AUTO"] -> id INTEGER PRIMARY KEY AUTOINCREMENT
          * If >1 non-AUTO pk names are present, AUTO is ignored and we fall back to normal composite PK.
            (SQLite autoincrement only works with a single INTEGER PRIMARY KEY column.)

        Notes
        -----
        - This function creates ONLY the PK columns. Other columns are added dynamically by update_database().
        """
        PKs = list(pk) if pk is not None else []
        auto_pk = any(isinstance(x, str) and x.upper() == "AUTO" for x in PKs)
        non_auto_pks = [x for x in PKs if not (isinstance(x, str) and x.upper() == "AUTO")]

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('SELECT name FROM sqlite_master WHERE type="table" AND name=?;', (table_name,))
                if cursor.fetchone():
                    return

                if auto_pk:
                    if len(non_auto_pks) == 1:
                        auto_col = self.sanitize_column_name(non_auto_pks[0])
                    elif len(non_auto_pks) == 0:
                        auto_col = "id"
                    else:
                        logger.warning(
                            f'{table_name}: PKs={PKs} include AUTO but also multiple PK columns. '
                            f'Falling back to composite PRIMARY KEY without AUTOINCREMENT.'
                        )
                        auto_pk = False

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
                    raise ValueError(f'{table_name}: create_empty_table called with no PK columns (pk={pk})')

                cols = [self.sanitize_column_name(c) for c in non_auto_pks]
                pk_clause = ", ".join([f'"{c}"' for c in cols])

                create_sql = f'''
                    CREATE TABLE "{table_name}" (
                        {", ".join([f'"{col}" INTEGER' for col in cols])},
                        PRIMARY KEY ({pk_clause})
                    );
                '''
                cursor.execute(create_sql)
                conn.commit()
                logger.info(f'Created table "{table_name}" with PK: {", ".join(cols)}')

        except sqlite3.OperationalError:
            logger.exception(f'SQL error creating table "{table_name}".')
            raise
        except Exception:
            logger.exception(f'Error creating table "{table_name}".')
            raise

    # -------------------------
    # Insert/update operations
    # -------------------------

    def update_database(self, table_name: str, update_data: Union[dict, List[dict]]):
        """Insert/update rows into a table.

        - In normal mode (explicit PKs): upsert by PK.
        - In AUTO mode (PK list contains "AUTO"): insert-only (no upsert check).

        Commits once per call.
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

        except sqlite3.OperationalError:
            logger.exception(f'SQL error updating "{table_name}".')
            raise

    def _update_single_row(self, cursor: sqlite3.Cursor, table_name: str, row_data: dict):
        """Insert/update a single row in the database dynamically."""
        try:
            if self.om is None or not hasattr(self.om, "tables") or table_name not in self.om.tables:
                raise KeyError(
                    "SQLDaemon._update_single_row requires OutputManager table metadata. "
                    f'Table "{table_name}" not found in OM.tables.'
                )

            PKs = self.om.tables[table_name].get_pks() or []
            auto_pk = any(isinstance(pk, str) and pk.upper() == "AUTO" for pk in PKs)
            non_auto_pks = [pk for pk in PKs if not (isinstance(pk, str) and pk.upper() == "AUTO")]

            auto_id_col = None
            if auto_pk:
                if len(non_auto_pks) == 1:
                    auto_id_col = non_auto_pks[0]
                elif len(non_auto_pks) == 0:
                    auto_id_col = "id"
                else:
                    logger.warning(
                        f"{table_name}: PKs={PKs} include AUTO but multiple PK columns were provided. "
                        "Disabling AUTO insert-only behavior and using normal PK-driven upsert."
                    )
                    auto_pk = False

            sanitized_data = {self.sanitize_column_name(col): val for col, val in row_data.items()}

            cursor.execute(f'PRAGMA table_info("{table_name}");')
            existing_columns = {row[1] for row in cursor.fetchall()}

            # Add new columns dynamically
            for col, val in sanitized_data.items():
                if col in existing_columns:
                    continue
                if isinstance(val, bool) or isinstance(val, int):
                    dtype = "INTEGER"
                elif isinstance(val, float):
                    dtype = "REAL"
                else:
                    dtype = "TEXT"
                cursor.execute(f'ALTER TABLE "{table_name}" ADD COLUMN "{col}" {dtype}')

            # AUTO mode: insert-only
            if auto_pk:
                auto_id_col_s = self.sanitize_column_name(auto_id_col) if auto_id_col else None

                insert_cols: List[str] = []
                insert_vals: Dict[str, Any] = {}

                for col, val in sanitized_data.items():
                    if auto_id_col_s and col == auto_id_col_s and val is None:
                        continue
                    insert_cols.append(col)
                    insert_vals[col] = val

                if not insert_cols:
                    logger.debug(f'{table_name}: AUTO insert skipped (no insertable columns).')
                    return

                col_clause = ", ".join([f'"{c}"' for c in insert_cols])
                ph_clause = ", ".join([f":{c}" for c in insert_cols])
                insert_sql = f'INSERT INTO "{table_name}" ({col_clause}) VALUES ({ph_clause})'
                cursor.execute(insert_sql, insert_vals)
                return

            # Normal mode: PK-driven upsert
            if not PKs:
                raise ValueError(f"Missing PKs in OutputManager table metadata for '{table_name}'")

            sanitized_pks = [self.sanitize_column_name(pk) for pk in PKs]

            pk_values = {pk: sanitized_data.get(pk) for pk in sanitized_pks if sanitized_data.get(pk) is not None}
            if len(pk_values) != len(sanitized_pks):
                missing = [pk for pk in sanitized_pks if sanitized_data.get(pk) is None]
                raise ValueError(f"Missing PK values in '{table_name}': {missing}")

            where_clause = " AND ".join([f'"{pk}" = :{pk}' for pk in sanitized_pks])
            check_sql = f'SELECT 1 FROM "{table_name}" WHERE {where_clause}'
            cursor.execute(check_sql, pk_values)
            exists = cursor.fetchone()

            if not exists:
                insert_columns = ", ".join([f'"{pk}"' for pk in sanitized_pks])
                placeholders = ", ".join([f":{pk}" for pk in sanitized_pks])
                insert_sql = f'INSERT INTO "{table_name}" ({insert_columns}) VALUES ({placeholders})'
                cursor.execute(insert_sql, pk_values)

            non_pk_cols = [c for c in sanitized_data.keys() if c not in set(sanitized_pks)]
            if not non_pk_cols:
                return

            update_clause = ", ".join([f'"{col}" = :{col}' for col in non_pk_cols])
            sql = f'UPDATE "{table_name}" SET {update_clause} WHERE {where_clause}'
            cursor.execute(sql, sanitized_data)

        except Exception:
            logger.exception(
                f"Failed _update_single_row for table={table_name}. "
                f"Row keys={list(row_data.keys())[:12]}..."
            )
            raise

    # -------------
    # Querying
    # -------------

    def access_data(self, table_name: str, columns: ColumnsArg = "*", filters: Optional[dict] = None):
        """Retrieve data from an SQLite table.

        Returns None if the table is empty.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('SELECT name FROM sqlite_master WHERE type="table" AND name=?;', (table_name,))
                if cursor.fetchone() is None:
                    logger.error(f'Table "{table_name}" does not exist.')
                    return None

                cursor.execute(f'PRAGMA table_info("{table_name}");')
                available_columns = {row[1] for row in cursor.fetchall()}

                # Columns
                if columns == "*" or columns is None:
                    col_clause = "*"
                else:
                    requested = [columns] if isinstance(columns, str) else list(columns)
                    requested_s = [self.sanitize_column_name(c) for c in requested]
                    valid = [c for c in requested_s if c in available_columns]
                    if not valid:
                        logger.error(f'No valid columns requested for "{table_name}". Requested={requested_s}')
                        return None
                    col_clause = ", ".join([f'"{c}"' for c in valid])

                # Filters
                where_clause = ""
                params: Dict[str, Any] = {}
                if filters and isinstance(filters, dict):
                    filt_s = {self.sanitize_column_name(k): v for k, v in filters.items()}
                    filt_valid = {k: v for k, v in filt_s.items() if k in available_columns}
                    if filt_valid:
                        where_clause = " WHERE " + " AND ".join([f'"{k}" = :{k}' for k in filt_valid.keys()])
                        params = filt_valid

                query = f'SELECT {col_clause} FROM "{table_name}"{where_clause};'
                df = pd.read_sql_query(query, conn, params=params)

                if df.empty:
                    return None
                return df

        except sqlite3.OperationalError:
            logger.exception(f'SQL error accessing data from "{table_name}".')
            raise
        except Exception:
            logger.exception(f'Unexpected error accessing data from "{table_name}".')
            raise
