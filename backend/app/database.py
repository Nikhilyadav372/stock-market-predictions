"""
Database setup using SQLAlchemy 2.x with async-compatible session management.

Falls back to SQLite for local development when PostgreSQL is unavailable.
PostgreSQL is used automatically when DATABASE_URL starts with 'postgresql'.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings

settings = get_settings()

_db_url = settings.DATABASE_URL

# ── SQLite fallback for local dev (no PostgreSQL required) ────────────────────
# If the configured URL uses PostgreSQL and psycopg2 is not available,
# automatically fall back to a local SQLite database so the app still starts.
_engine_kwargs: dict = {"pool_pre_ping": True}

if _db_url.startswith("postgresql"):
    try:
        import psycopg2  # noqa: F401 — just check it's importable
        _engine_kwargs.update({"pool_size": 10, "max_overflow": 20})
    except ImportError:
        import os
        _sqlite_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "dev.db")
        )
        os.makedirs(os.path.dirname(_sqlite_path), exist_ok=True)
        _db_url = f"sqlite:///{_sqlite_path}"
        _engine_kwargs["connect_args"] = {"check_same_thread": False}
        import warnings
        warnings.warn(
            f"psycopg2 not found — falling back to SQLite at {_sqlite_path}. "
            "Install psycopg2-binary and run PostgreSQL for production use.",
            stacklevel=2,
        )
elif _db_url.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(_db_url, **_engine_kwargs)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


def get_db():
    """FastAPI dependency that yields a DB session and ensures cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
