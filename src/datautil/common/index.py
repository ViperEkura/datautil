from __future__ import annotations

import sqlite3
from pathlib import Path


class DiskIndex:
    def __init__(self, directory: str) -> None:
        self.conn = sqlite3.connect(str(Path(directory) / "index.sqlite3"))
        self.conn.execute("PRAGMA journal_mode=OFF")
        self.conn.execute("PRAGMA synchronous=OFF")
        self.conn.execute("PRAGMA cache_size=-2048")
        self.conn.execute(
            "CREATE TABLE seen (id TEXT PRIMARY KEY, first_location TEXT)"
        )

    def add_unique(self, record_id: str, location: str | None = None) -> bool:
        try:
            self.conn.execute("INSERT INTO seen VALUES (?, ?)", (record_id, location))
            return True
        except sqlite3.IntegrityError:
            return False

    def first_location(self, record_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT first_location FROM seen WHERE id=?", (record_id,)
        ).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "DiskIndex":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()
