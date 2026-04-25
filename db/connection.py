"""PostgreSQL connection and schema initialization."""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from config import DATABASE_URL

_connection = None


def get_connection():
    """Get or create a PostgreSQL connection."""
    global _connection
    if _connection is None or _connection.closed:
        _connection = psycopg2.connect(DATABASE_URL)
        _connection.autocommit = True
    return _connection


def init_db():
    """Run init.sql to create tables if they don't exist."""
    init_sql_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "init.sql")
    if not os.path.exists(init_sql_path):
        print("[DB] init.sql not found, skipping schema init")
        return

    try:
        conn = get_connection()
        with conn.cursor() as cur:
            with open(init_sql_path, "r") as f:
                cur.execute(f.read())
        print("[DB] Schema initialized successfully")
    except Exception as e:
        print(f"[DB] Schema init failed (pgvector not ready?): {e}")
        print("[DB] App will work without vector features until pgvector is available")


def execute_query(query: str, params: tuple = None, fetch: bool = True):
    """Execute a query and optionally return results."""
    conn = get_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, params)
        if fetch:
            return cur.fetchall()
        return None


def test_connection() -> bool:
    """Test if the database connection works."""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            return True
    except Exception as e:
        print(f"[DB] Connection test failed: {e}")
        return False
