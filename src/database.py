"""Histórico das detecções em SQLite."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.paths import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,  -- ISO 8601 em UTC
    source      TEXT    NOT NULL,  -- api, imagem, video, ao-vivo
    label       TEXT    NOT NULL,  -- classe do modelo (bottle, banana…)
    category    TEXT    NOT NULL,  -- categoria da coleta seletiva
    confidence  REAL    NOT NULL,
    weight_g    REAL    NOT NULL,  -- peso médio estimado do item
    track_id    INTEGER            -- id da trilha no vídeo (NULL em imagens)
);
CREATE INDEX IF NOT EXISTS idx_detections_ts ON detections (ts);
"""

COLUMNS = ("id", "ts", "source", "label", "category", "confidence", "weight_g", "track_id")


def _iso(value: datetime | str | None) -> str | None:
    if value is None or isinstance(value, str):
        return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str | Path = DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A API e o dashboard usam o banco de várias threads; o lock serializa as escritas
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    def close(self) -> None:
        self._conn.close()

    def add(
        self,
        label: str,
        category: str,
        confidence: float,
        weight_g: float,
        source: str = "api",
        track_id: int | None = None,
        ts: datetime | str | None = None,
    ) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO detections (ts, source, label, category, confidence, weight_g, track_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (_iso(ts or datetime.now(timezone.utc)), source, label, category, confidence, weight_g, track_id),
            )
            return cur.lastrowid

    def add_detections(self, detections, source: str = "api") -> int:
        """Grava as detecções que são resíduo (com categoria). Devolve quantas gravou."""
        rows = [d for d in detections if d.category]
        for d in rows:
            self.add(d.label, d.category, d.confidence, d.weight_g, source)
        return len(rows)

    def records(self, start: datetime | str | None = None, end: datetime | str | None = None) -> list[dict]:
        where, params = self._period(start, end)
        cur = self._conn.execute(f"SELECT {', '.join(COLUMNS)} FROM detections{where} ORDER BY ts, id", params)
        return [dict(zip(COLUMNS, row)) for row in cur.fetchall()]

    def totals_by_category(self, start=None, end=None) -> dict[str, dict]:
        where, params = self._period(start, end)
        cur = self._conn.execute(
            f"SELECT category, COUNT(*), SUM(weight_g) FROM detections{where} GROUP BY category", params
        )
        return {cat: {"items": n, "weight_kg": (w or 0) / 1000} for cat, n, w in cur.fetchall()}

    def clear(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM detections")

    @staticmethod
    def _period(start, end) -> tuple[str, list]:
        clauses, params = [], []
        if start is not None:
            clauses.append("ts >= ?")
            params.append(_iso(start))
        if end is not None:
            clauses.append("ts < ?")
            params.append(_iso(end))
        return (" WHERE " + " AND ".join(clauses) if clauses else ""), params
