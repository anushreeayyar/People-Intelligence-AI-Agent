"""Read-only DuckDB access layer."""
from __future__ import annotations

import threading
from functools import lru_cache

import duckdb
import pandas as pd

from pi import config

_LOCK = threading.Lock()
_CON: duckdb.DuckDBPyConnection | None = None


def connect() -> duckdb.DuckDBPyConnection:
    """Single shared read-only connection.

    Read-only is not a performance choice, it is a control: the application
    process is structurally incapable of mutating the warehouse, so no prompt,
    no generated SQL and no bug in the agent loop can write to it.
    """
    global _CON
    if _CON is None:
        with _LOCK:
            if _CON is None:
                if not config.WAREHOUSE.exists():
                    raise FileNotFoundError(
                        f"Warehouse not found at {config.WAREHOUSE}. "
                        "Run:  python data/generate_data.py"
                    )
                _CON = duckdb.connect(str(config.WAREHOUSE), read_only=True)
    return _CON


def query(sql: str, params: list | None = None) -> pd.DataFrame:
    con = connect()
    with _LOCK:
        return con.execute(sql, params or []).df()


@lru_cache(maxsize=256)
def cached_query(sql: str) -> pd.DataFrame:
    return query(sql)


def as_of_date() -> str:
    return str(query("SELECT as_of_date FROM meta_asof").iloc[0, 0])


def schema_summary() -> str:
    """Compact schema description handed to the model for SQL generation."""
    objects = [
        "v_employees", "v_headcount_monthly", "v_movement_monthly",
        "v_requisitions", "v_candidates", "v_survey", "v_internal_moves",
    ]
    lines = []
    for obj in objects:
        cols = query(f"DESCRIBE {obj}")
        visible = [
            f"{r.column_name} {r.column_type}"
            for r in cols.itertuples()
            if r.column_name not in config.RESTRICTED_COLUMNS
        ]
        lines.append(f"{obj}({', '.join(visible)})")
    return "\n".join(lines)
