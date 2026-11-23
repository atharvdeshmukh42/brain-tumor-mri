import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


def now():
    return datetime.now(UTC).isoformat()


class Store:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "analyses.sqlite3"
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS analyses
                (id TEXT PRIMARY KEY, source TEXT NOT NULL, status TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, error TEXT, report TEXT, metadata TEXT NOT NULL)""")

    def recover_interrupted(self):
        with self.connect() as db:
            db.execute(
                "UPDATE analyses SET status='failed', error='Analysis interrupted by server restart. Run a new analysis.', updated_at=? WHERE status IN ('queued','running')",
                (now(),),
            )

    def connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def create(self, id, source, metadata):
        with self.connect() as db:
            db.execute(
                "INSERT INTO analyses VALUES (?,?,?,?,?,?,?,?)",
                (id, source, "queued", now(), now(), None, None, json.dumps(metadata)),
            )
        return self.get(id)

    def update(self, id, status, *, report=None, error=None):
        with self.connect() as db:
            db.execute(
                "UPDATE analyses SET status=?, updated_at=?, report=?, error=? WHERE id=?",
                (
                    status,
                    now(),
                    json.dumps(report) if report is not None else None,
                    error,
                    id,
                ),
            )

    @staticmethod
    def decode(row):
        if row is None:
            return None
        result = dict(row)
        result["report"] = json.loads(result["report"]) if result["report"] else None
        result["metadata"] = json.loads(result["metadata"])
        return result

    def get(self, id):
        with self.connect() as db:
            return self.decode(
                db.execute("SELECT * FROM analyses WHERE id=?", (id,)).fetchone()
            )

    def history(self):
        with self.connect() as db:
            return [
                self.decode(row)
                for row in db.execute(
                    "SELECT * FROM analyses ORDER BY created_at DESC LIMIT 200"
                )
            ]
