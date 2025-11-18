"""Safe SQLite helper for the assignment.

Provides:
- get_schema(conn)
- table_info(conn, table_name)
- safe_execute(conn, sql, params=None, fetch=1000)

All queries are executed locally on the provided sqlite file.
"""

import sqlite3
from typing import List, Dict, Any, Tuple


def open_conn(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def table_info(conn: sqlite3.Connection, table_name: str) -> List[Dict[str, Any]]:
    cur = conn.execute(f"PRAGMA table_info('{table_name}')")
    rows = [dict(r) for r in cur.fetchall()]
    return rows


def get_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [r[0] for r in cur.fetchall()]


def safe_execute(
    conn: sqlite3.Connection, sql: str, params: Tuple = (), fetch: int = 1000
) -> Dict[str, Any]:
    """Execute SQL safely and return a structured result.

    Returns dict with keys: success (bool), error (str|None), columns (list), rows (list)
    """
    try:
        cur = conn.execute(sql, params)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(fetch)
        # Convert sqlite3.Row to dict
        rows_out = [dict(r) for r in rows]
        return {"success": True, "error": None, "columns": cols, "rows": rows_out}
    except Exception as e:
        return {"success": False, "error": str(e), "columns": [], "rows": []}
