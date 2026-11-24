"""Local append-only forecast history for actual inference requests."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

MAX_HISTORY_RECORDS = 100_000


class PredictionHistoryError(ValueError):
    """Prediction history is unsafe, full, or inconsistent."""


class PredictionHistory:
    def __init__(self, database: Path):
        self.database = database

    def _connect(self) -> sqlite3.Connection:
        if self.database.is_symlink():
            raise PredictionHistoryError("prediction history database must not be a symlink")
        self.database.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database, timeout=5)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                predicted_at_utc TEXT NOT NULL,
                model_id TEXT NOT NULL,
                input_sha256 TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
        """)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS predictions_time_idx ON predictions(predicted_at_utc DESC)"
        )
        return connection

    def append_many(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        with self._connect() as connection:
            current_count = int(
                connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            )
            if current_count + len(records) > MAX_HISTORY_RECORDS:
                raise PredictionHistoryError(
                    f"prediction history reached its {MAX_HISTORY_RECORDS:,}-record retention limit"
                )
            connection.executemany(
                "INSERT INTO predictions(prediction_id,predicted_at_utc,model_id,input_sha256,payload_json) VALUES(?,?,?,?,?)",
                [
                    (
                        record["prediction_id"],
                        record["predicted_at_utc"],
                        record["model_id"],
                        record["input_sha256"],
                        json.dumps(record, allow_nan=False, sort_keys=True),
                    )
                    for record in records
                ],
            )

    def list(self, *, limit: int = 200, input_sha256: str | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        with self._connect() as connection:
            if input_sha256:
                rows = connection.execute(
                    "SELECT payload_json FROM predictions WHERE input_sha256=? ORDER BY predicted_at_utc DESC LIMIT ?",
                    (input_sha256, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT payload_json FROM predictions ORDER BY predicted_at_utc DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])

    def latest_warning(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM predictions WHERE json_extract(payload_json,'$.risk')='warning' "
                "ORDER BY predicted_at_utc DESC LIMIT 1"
            ).fetchone()
        return json.loads(row[0]) if row else None
