import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS objects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    reference_image_path TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_id INTEGER NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
    track_id INTEGER,
    event_type TEXT NOT NULL,
    confidence REAL,
    bbox_x1 REAL,
    bbox_y1 REAL,
    bbox_x2 REAL,
    bbox_y2 REAL,
    frame_path TEXT,
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_object_id ON events(object_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)


@contextmanager
def db_cursor() -> Iterator[sqlite3.Cursor]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    finally:
        conn.close()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_object(name: str, display_name: str, reference_image_path: Optional[str]) -> int:
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO objects (name, display_name, reference_image_path, created_at) "
            "VALUES (?, ?, ?, ?)",
            (name, display_name, reference_image_path, now_iso()),
        )
        return cur.lastrowid


def get_object_by_name(name: str) -> Optional[sqlite3.Row]:
    with db_cursor() as cur:
        cur.execute("SELECT * FROM objects WHERE name = ?", (name,))
        return cur.fetchone()


def get_object_by_id(object_id: int) -> Optional[sqlite3.Row]:
    with db_cursor() as cur:
        cur.execute("SELECT * FROM objects WHERE id = ?", (object_id,))
        return cur.fetchone()


def list_objects() -> list[sqlite3.Row]:
    with db_cursor() as cur:
        cur.execute("SELECT * FROM objects ORDER BY created_at DESC")
        return cur.fetchall()


def delete_object(object_id: int) -> None:
    with db_cursor() as cur:
        cur.execute("DELETE FROM objects WHERE id = ?", (object_id,))


def create_event(
    object_id: int,
    track_id: Optional[int],
    event_type: str,
    confidence: Optional[float],
    bbox: Optional[tuple[float, float, float, float]],
    frame_path: Optional[str],
    timestamp: Optional[str] = None,
) -> int:
    bbox = bbox or (None, None, None, None)
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO events "
            "(object_id, track_id, event_type, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, "
            "frame_path, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                object_id,
                track_id,
                event_type,
                confidence,
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                frame_path,
                timestamp or now_iso(),
            ),
        )
        return cur.lastrowid


def get_recent_events(object_id: int, limit: int = 20) -> list[sqlite3.Row]:
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM events WHERE object_id = ? ORDER BY timestamp DESC LIMIT ?",
            (object_id, limit),
        )
        return cur.fetchall()


def get_last_event(object_id: int) -> Optional[sqlite3.Row]:
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM events WHERE object_id = ? ORDER BY timestamp DESC LIMIT 1",
            (object_id,),
        )
        return cur.fetchone()
