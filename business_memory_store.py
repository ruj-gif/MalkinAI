"""Consent-based SQLite storage for user-isolated business memory profiles.

Profiles are keyed by a random recovery token. Only a one-way hash of the token
is stored, and no conversation transcript or API secret is written to SQLite.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from business_memory import normalize_business_memory


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE_PATH = PROJECT_DIR / "processed" / "business_memory.sqlite3"


def generate_recovery_key() -> str:
    """Create a high-entropy bearer key the user must keep to restore memory."""
    return secrets.token_urlsafe(32)


def _key_hash(recovery_key: str) -> str:
    if not isinstance(recovery_key, str) or len(recovery_key) < 30:
        raise ValueError("Enter a valid business-memory recovery key.")
    return hashlib.sha256(recovery_key.encode("utf-8")).hexdigest()


def _connect(database_path: str | Path) -> sqlite3.Connection:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    try:
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS saved_business_profiles (
                   profile_key_hash TEXT PRIMARY KEY,
                   memory_json TEXT NOT NULL,
                   updated_at TEXT NOT NULL
               )"""
        )
    except sqlite3.Error:
        connection.close()
        raise
    return connection


def save_memory(
    recovery_key: str,
    memory: Any,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> None:
    """Insert or update one profile without storing the recovery key itself."""
    key_hash = _key_hash(recovery_key)
    normalized = normalize_business_memory(memory)
    serialized = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(_connect(database_path)) as connection, connection:
        connection.execute(
            """INSERT INTO saved_business_profiles(profile_key_hash, memory_json, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(profile_key_hash) DO UPDATE SET
                   memory_json = excluded.memory_json,
                   updated_at = excluded.updated_at""",
            (key_hash, serialized, updated_at),
        )


def load_memory(
    recovery_key: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any] | None:
    """Restore one profile; malformed or corrupted rows are treated as unavailable."""
    key_hash = _key_hash(recovery_key)
    if not Path(database_path).exists():
        return None
    try:
        with closing(_connect(database_path)) as connection, connection:
            row = connection.execute(
                "SELECT memory_json FROM saved_business_profiles WHERE profile_key_hash = ?",
                (key_hash,),
            ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    try:
        decoded = json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(decoded, dict) or not isinstance(decoded.get("user_provided"), dict):
        return None
    return normalize_business_memory(decoded)


def delete_memory(
    recovery_key: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> bool:
    """Delete the profile matching a recovery key and report whether it existed."""
    key_hash = _key_hash(recovery_key)
    if not Path(database_path).exists():
        return False
    with closing(_connect(database_path)) as connection, connection:
        cursor = connection.execute(
            "DELETE FROM saved_business_profiles WHERE profile_key_hash = ?",
            (key_hash,),
        )
        return cursor.rowcount > 0
