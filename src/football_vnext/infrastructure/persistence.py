"""Persistence for daily predictions, odds snapshots and later CLV settlement.

SQLite is the zero-configuration local default. Set DATABASE_URL to a
PostgreSQL URL in production (psycopg is imported only for PostgreSQL).
All database operations close their connection in a finally block; failed
transactions are rolled back before close.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PredictionStore:
    def __init__(self, database_url: str | None = None):
        self.url = database_url or os.getenv("DATABASE_URL", "sqlite:///data/football_vnext.db")
        if self.url.startswith("sqlite:///"):
            self.kind = "sqlite"
            self.path = Path(self.url[10:])
            self.path.parent.mkdir(parents=True, exist_ok=True)
        elif self.url.startswith(("postgresql://", "postgres://")):
            self.kind = "postgres"
            self.path = None
        else:
            raise ValueError("DATABASE_URL must be sqlite:///... or postgresql://...")
        self._init()

    def _conn(self):
        if self.kind == "sqlite":
            return sqlite3.connect(self.path)
        import psycopg
        return psycopg.connect(self.url)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _init(self) -> None:
        con = self._conn()
        try:
            cur = con.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, kickoff TEXT NOT NULL,
                league TEXT NOT NULL, home TEXT NOT NULL, away TEXT NOT NULL, payload TEXT NOT NULL
            )""")
            cur.execute("""CREATE TABLE IF NOT EXISTS settlements (
                prediction_id TEXT PRIMARY KEY, settled_at TEXT NOT NULL, result TEXT,
                closing_odds TEXT, clv REAL, pnl REAL
            )""")
            cur.execute("""CREATE TABLE IF NOT EXISTS odds_snapshots (
                id TEXT PRIMARY KEY, captured_at TEXT NOT NULL, kickoff TEXT NOT NULL,
                league TEXT NOT NULL, home TEXT NOT NULL, away TEXT NOT NULL, odds TEXT NOT NULL
            )""")
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def save_prediction(self, prediction_id: str, kickoff: datetime, league: str, home: str, away: str, payload: dict) -> None:
        con = self._conn()
        try:
            cur = con.cursor()
            vals = (prediction_id, self._now_iso(), kickoff.isoformat(), league, home, away, json.dumps(payload))
            if self.kind == "sqlite":
                cur.execute(
                    "INSERT OR REPLACE INTO predictions(id,created_at,kickoff,league,home,away,payload) VALUES(?,?,?,?,?,?,?)",
                    vals,
                )
            else:
                cur.execute(
                    "INSERT INTO predictions(id,created_at,kickoff,league,home,away,payload) VALUES(%s,%s,%s,%s,%s,%s,%s) "
                    "ON CONFLICT(id) DO UPDATE SET payload=EXCLUDED.payload",
                    vals,
                )
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def list_recent(self, limit: int = 100):
        if limit < 1:
            raise ValueError("limit must be >= 1")
        con = self._conn()
        try:
            cur = con.cursor()
            if self.kind == "sqlite":
                cur.execute(
                    "SELECT id,created_at,kickoff,league,home,away,payload FROM predictions "
                    "ORDER BY kickoff DESC LIMIT ?", (limit,)
                )
            else:
                cur.execute(
                    "SELECT id,created_at,kickoff,league,home,away,payload FROM predictions "
                    "ORDER BY kickoff DESC LIMIT %s", (limit,)
                )
            return cur.fetchall()
        finally:
            con.close()

    def settle_prediction(
        self, prediction_id: str, result: str, closing_odds: dict | None = None,
        clv: float | None = None, pnl: float | None = None,
    ) -> None:
        con = self._conn()
        try:
            cur = con.cursor()
            payload = json.dumps(closing_odds) if closing_odds is not None else None
            vals = (prediction_id, self._now_iso(), result, payload, clv, pnl)
            if self.kind == "sqlite":
                cur.execute(
                    "INSERT OR REPLACE INTO settlements(prediction_id,settled_at,result,closing_odds,clv,pnl) VALUES(?,?,?,?,?,?)",
                    vals,
                )
            else:
                cur.execute(
                    "INSERT INTO settlements(prediction_id,settled_at,result,closing_odds,clv,pnl) VALUES(%s,%s,%s,%s,%s,%s) "
                    "ON CONFLICT(prediction_id) DO UPDATE SET result=EXCLUDED.result,closing_odds=EXCLUDED.closing_odds,"
                    "clv=EXCLUDED.clv,pnl=EXCLUDED.pnl,settled_at=EXCLUDED.settled_at",
                    vals,
                )
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def list_unsettled(self, before: datetime | None = None, limit: int = 500):
        if limit < 1:
            raise ValueError("limit must be >= 1")
        con = self._conn()
        try:
            cur = con.cursor()
            where = ""
            params: list[Any] = []
            if before is not None:
                where = "WHERE p.kickoff < ?" if self.kind == "sqlite" else "WHERE p.kickoff < %s"
                params.append(before.isoformat())
            sql = (
                "SELECT p.id,p.kickoff,p.league,p.home,p.away,p.payload FROM predictions p "
                "LEFT JOIN settlements s ON p.id=s.prediction_id "
                f"{where}{' AND' if where else 'WHERE'} s.prediction_id IS NULL "
                f"ORDER BY p.kickoff ASC LIMIT {limit}"
            )
            cur.execute(sql, tuple(params))
            return cur.fetchall()
        finally:
            con.close()

    def save_odds_snapshot(
        self, snapshot_id: str, captured_at: datetime, kickoff: datetime,
        league: str, home: str, away: str, odds: dict,
    ) -> None:
        con = self._conn()
        try:
            cur = con.cursor()
            vals = (snapshot_id, captured_at.isoformat(), kickoff.isoformat(), league, home, away, json.dumps(odds))
            if self.kind == "sqlite":
                cur.execute(
                    "INSERT OR REPLACE INTO odds_snapshots(id,captured_at,kickoff,league,home,away,odds) VALUES(?,?,?,?,?,?,?)",
                    vals,
                )
            else:
                cur.execute(
                    "INSERT INTO odds_snapshots(id,captured_at,kickoff,league,home,away,odds) VALUES(%s,%s,%s,%s,%s,%s,%s) "
                    "ON CONFLICT(id) DO UPDATE SET odds=EXCLUDED.odds,captured_at=EXCLUDED.captured_at",
                    vals,
                )
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def latest_odds_before(self, league: str, home: str, away: str, kickoff: datetime):
        """Return the latest captured snapshot strictly before the match kickoff.

        This is the auditable pre-kickoff closing snapshot available to this
        system, rather than comparing the snapshot timestamp with wall-clock now.
        """
        con = self._conn()
        try:
            cur = con.cursor()
            ph = "?" if self.kind == "sqlite" else "%s"
            sql = (
                f"SELECT captured_at,odds FROM odds_snapshots "
                f"WHERE league={ph} AND home={ph} AND away={ph} AND kickoff={ph} "
                f"AND captured_at < {ph} ORDER BY captured_at DESC LIMIT 1"
            )
            # ISO-8601 UTC/local timestamps sort chronologically when the same
            # normalization is used by the writers. kickoff is the hard cutoff.
            cur.execute(sql, (league, home, away, kickoff.isoformat(), kickoff.isoformat()))
            return cur.fetchone()
        finally:
            con.close()
