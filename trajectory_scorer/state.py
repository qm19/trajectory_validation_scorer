"""Thread-safe durable row checkpoints. Stage cache lives beside this database."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path


class State:
    def __init__(self, root: Path):
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(root / "checkpoints.sqlite3", check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, status TEXT NOT NULL, value TEXT NOT NULL)")
        self.connection.commit()

    def get(self, key):
        with self.lock:
            row = self.connection.execute("SELECT status,value FROM jobs WHERE id=?", (key,)).fetchone()
        return None if row is None else {"status": row[0], "value": json.loads(row[1])}

    def put(self, key, status, value):
        with self.lock:
            self.connection.execute("INSERT OR REPLACE INTO jobs VALUES (?,?,?)",
                                    (key, status, json.dumps(value, ensure_ascii=False)))
            self.connection.commit()

    def close(self):
        with self.lock:
            self.connection.close()
