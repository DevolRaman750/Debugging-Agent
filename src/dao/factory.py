"""
DAO Factory
============
Picks SQLite or MongoDB based on environment variable.

Usage:
    from src.dao.factory import create_dao

    # Uses DB_BACKEND env var (default: "sqlite")
    dao = create_dao()

    # Or specify explicitly
    dao = create_dao("sqlite")
    dao = create_dao("mongodb")

Environment Variables:
    DB_BACKEND       — "sqlite" (default) or "mongodb"
    SQLITE_DB_PATH   — Path to SQLite file (default: "traceroot.db")
    MONGODB_URI      — MongoDB connection string
    MONGODB_DB       — MongoDB database name (default: "traceroot")
"""

from src.config import DB_BACKEND, MONGODB_URI, SQLITE_DB_PATH


def create_dao(backend: str | None = None):
    """Create a DAO instance based on the backend type.

    Auto-detection priority:
      1. Explicit `backend` parameter
      2. DB_BACKEND env var
      3. If MONGODB_URI is set  → mongodb
      4. Fallback              → sqlite

    Args:
        backend: "sqlite" or "mongodb". If None, uses config auto-detection.

    Returns:
        TraceRootSQLiteClient or TraceRootMongoDBClient

    Usage:
        dao = create_dao()           # Auto-detect from env
        dao = create_dao("sqlite")   # Force SQLite
        dao = create_dao("mongodb")  # Force MongoDB
    """
    chosen = backend or DB_BACKEND

    if chosen == "mongodb":
        from src.dao.mongodb_dao import TraceRootMongoDBClient
        import os
        uri = MONGODB_URI or "mongodb://localhost:27017"
        db_name = os.getenv("MONGODB_DB", "traceroot")
        return TraceRootMongoDBClient(uri=uri, db_name=db_name)

    elif chosen == "sqlite":
        from src.dao.sqlite_dao import TraceRootSQLiteClient
        return TraceRootSQLiteClient(db_path=SQLITE_DB_PATH)

    else:
        raise ValueError(
            f"Unknown DB_BACKEND: '{chosen}'. Use 'sqlite' or 'mongodb'."
        )
