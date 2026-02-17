"""
database.py

SQLite persistence:
  - strategies, legs, order_log
  - survivability: engine can recover active strategies after crash
  - includes trailing SL fields (lowest_price, trailing_active)
  - includes helpers needed by pending monitor / broker reconciliation

This file is safe to use even if you already have an older DB:
  - It attempts to create tables if missing
  - It attempts to add missing columns via ALTER TABLE
"""

from __future__ import annotations

import sqlite3
import threading
import json
from datetime import datetime
from typing import List, Optional

from app_config import Config
from models import (
    Strategy, StrategyType, StrategyStatus,
    Leg, LegStatus, OrderSide, OptionRight, Greeks
)


class Database:
    def __init__(self, db_path: str = Config.DB_PATH):
        self._path = db_path
        self._local = threading.local()
        self._init_schema()

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._path, check_same_thread=False, timeout=10.0)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
        return self._local.conn

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS strategies (
                    strategy_id   TEXT PRIMARY KEY,
                    strategy_type TEXT NOT NULL,
                    stock_code    TEXT NOT NULL,
                    target_delta  REAL DEFAULT 0,
                    total_pnl     REAL DEFAULT 0,
                    status        TEXT DEFAULT 'deploying',
                    created_at    TEXT,
                    closed_at     TEXT
                );

                CREATE TABLE IF NOT EXISTS legs (
                    leg_id          TEXT PRIMARY KEY,
                    strategy_id     TEXT NOT NULL,
                    stock_code      TEXT NOT NULL,
                    exchange_code   TEXT DEFAULT 'NFO',
                    strike_price    REAL NOT NULL,
                    right           TEXT NOT NULL,
                    expiry_date     TEXT NOT NULL,
                    side            TEXT NOT NULL,
                    quantity        INTEGER NOT NULL,
                    entry_price     REAL DEFAULT 0,
                    current_price   REAL DEFAULT 0,
                    exit_price      REAL DEFAULT 0,
                    sl_price        REAL DEFAULT 0,
                    sl_percentage   REAL DEFAULT 0,
                    lowest_price    REAL DEFAULT 0,
                    trailing_active INTEGER DEFAULT 0,
                    entry_order_id  TEXT DEFAULT '',
                    exit_order_id   TEXT DEFAULT '',
                    status          TEXT DEFAULT 'pending',
                    entry_time      TEXT,
                    exit_time       TEXT,
                    pnl             REAL DEFAULT 0,
                    greeks_json     TEXT DEFAULT '{}',
                    FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
                );

                CREATE TABLE IF NOT EXISTS order_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT NOT NULL,
                    leg_id      TEXT,
                    order_id    TEXT,
                    action      TEXT,
                    status      TEXT,
                    price       REAL,
                    quantity    INTEGER,
                    message     TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_legs_sid ON legs(strategy_id);
                CREATE INDEX IF NOT EXISTS idx_legs_st ON legs(status);
                CREATE INDEX IF NOT EXISTS idx_olog_ts ON order_log(timestamp);
            """)

            # lightweight migrations (add missing columns if DB is old)
            self._ensure_column("legs", "lowest_price", "REAL DEFAULT 0")
            self._ensure_column("legs", "trailing_active", "INTEGER DEFAULT 0")
            self._ensure_column("legs", "greeks_json", "TEXT DEFAULT '{}'")
            self._ensure_column("legs", "pnl", "REAL DEFAULT 0")

    def _ensure_column(self, table: str, col: str, ddl: str) -> None:
        """
        If column doesn't exist, ALTER TABLE ADD COLUMN.
        """
        try:
            cols = [r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})").fetchall()]
            if col not in cols:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
        except Exception:
            # ignore migration failures to keep app bootable
            pass

    # ─────────────────────────────────────────────────────────
    # Strategy CRUD
    # ─────────────────────────────────────────────────────────

    def save_strategy(self, s: Strategy) -> None:
        with self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO strategies
                   (strategy_id, strategy_type, stock_code, target_delta,
                    total_pnl, status, created_at, closed_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    s.strategy_id,
                    s.strategy_type.value,
                    s.stock_code,
                    float(s.target_delta),
                    float(s.total_pnl),
                    s.status.value,
                    s.created_at,
                    s.closed_at,
                ),
            )

    def get_active_strategies(self) -> List[Strategy]:
        rows = self._conn.execute(
            "SELECT * FROM strategies WHERE status IN ('deploying','active','partial_exit')"
        ).fetchall()

        strategies: List[Strategy] = []
        for r in rows:
            s = Strategy(
                strategy_id=r["strategy_id"],
                strategy_type=StrategyType(r["strategy_type"]),
                stock_code=r["stock_code"],
                target_delta=r["target_delta"],
                total_pnl=r["total_pnl"],
                status=StrategyStatus(r["status"]),
                created_at=r["created_at"],
                closed_at=r["closed_at"],
            )
            s.legs = self.get_legs_for_strategy(s.strategy_id)
            strategies.append(s)
        return strategies

    def get_all_strategies(self, limit: int = 50) -> List[Strategy]:
        rows = self._conn.execute(
            "SELECT * FROM strategies ORDER BY created_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()

        out: List[Strategy] = []
        for r in rows:
            s = Strategy(
                strategy_id=r["strategy_id"],
                strategy_type=StrategyType(r["strategy_type"]),
                stock_code=r["stock_code"],
                target_delta=r["target_delta"],
                total_pnl=r["total_pnl"],
                status=StrategyStatus(r["status"]),
                created_at=r["created_at"],
                closed_at=r["closed_at"],
            )
            s.legs = self.get_legs_for_strategy(s.strategy_id)
            out.append(s)
        return out

    # ─────────────────────────────────────────────────────────
    # Leg CRUD
    # ─────────────────────────────────────────────────────────

    def save_leg(self, l: Leg) -> None:
        gj = json.dumps({
            "delta": l.greeks.delta,
            "gamma": l.greeks.gamma,
            "theta": l.greeks.theta,
            "vega": l.greeks.vega,
            "iv": l.greeks.iv,
        })

        with self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO legs
                   (leg_id, strategy_id, stock_code, exchange_code, strike_price,
                    right, expiry_date, side, quantity, entry_price, current_price,
                    exit_price, sl_price, sl_percentage, lowest_price, trailing_active,
                    entry_order_id, exit_order_id, status, entry_time, exit_time,
                    pnl, greeks_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    l.leg_id,
                    l.strategy_id,
                    l.stock_code,
                    l.exchange_code,
                    float(l.strike_price),
                    l.right.value,
                    l.expiry_date,
                    l.side.value,
                    int(l.quantity),
                    float(l.entry_price),
                    float(l.current_price),
                    float(l.exit_price),
                    float(l.sl_price),
                    float(l.sl_percentage),
                    float(l.lowest_price),
                    1 if l.trailing_active else 0,
                    l.entry_order_id or "",
                    l.exit_order_id or "",
                    l.status.value,
                    l.entry_time,
                    l.exit_time,
                    float(l.pnl),
                    gj,
                ),
            )

    def update_leg_price(self, leg_id: str, current_price: float, pnl: float,
                         sl_price: Optional[float] = None,
                         lowest_price: Optional[float] = None,
                         trailing_active: Optional[bool] = None) -> None:
        sets = ["current_price=?", "pnl=?"]
        vals = [float(current_price), float(pnl)]

        if sl_price is not None:
            sets.append("sl_price=?")
            vals.append(float(sl_price))
        if lowest_price is not None:
            sets.append("lowest_price=?")
            vals.append(float(lowest_price))
        if trailing_active is not None:
            sets.append("trailing_active=?")
            vals.append(1 if trailing_active else 0)

        vals.append(leg_id)
        with self._conn:
            self._conn.execute(f"UPDATE legs SET {', '.join(sets)} WHERE leg_id=?", vals)

    def update_leg_status(self, leg_id: str, status: LegStatus, **kwargs) -> None:
        sets = ["status=?"]
        vals = [status.value]
        for k, v in kwargs.items():
            sets.append(f"{k}=?")
            vals.append(v)
        vals.append(leg_id)

        with self._conn:
            self._conn.execute(f"UPDATE legs SET {', '.join(sets)} WHERE leg_id=?", vals)

    def get_legs_for_strategy(self, strategy_id: str) -> List[Leg]:
        rows = self._conn.execute(
            "SELECT * FROM legs WHERE strategy_id=?",
            (strategy_id,),
        ).fetchall()
        return [self._row_to_leg(r) for r in rows]

    def get_active_legs(self) -> List[Leg]:
        rows = self._conn.execute(
            "SELECT * FROM legs WHERE status IN ('active','entering','sl_triggered','exiting')"
        ).fetchall()
        return [self._row_to_leg(r) for r in rows]

    def get_legs_by_status(self, status: LegStatus) -> List[Leg]:
        rows = self._conn.execute(
            "SELECT * FROM legs WHERE status=?",
            (status.value,),
        ).fetchall()
        return [self._row_to_leg(r) for r in rows]

    def _row_to_leg(self, r: sqlite3.Row) -> Leg:
        g = {}
        try:
            g = json.loads(r["greeks_json"] or "{}")
        except Exception:
            g = {}

        return Leg(
            leg_id=r["leg_id"],
            strategy_id=r["strategy_id"],
            stock_code=r["stock_code"],
            exchange_code=r["exchange_code"],
            strike_price=r["strike_price"],
            right=OptionRight(r["right"]),
            expiry_date=r["expiry_date"],
            side=OrderSide(r["side"]),
            quantity=r["quantity"],
            entry_price=r["entry_price"],
            current_price=r["current_price"],
            exit_price=r["exit_price"],
            sl_price=r["sl_price"],
            sl_percentage=r["sl_percentage"],
            lowest_price=r["lowest_price"] if "lowest_price" in r.keys() else 0.0,
            trailing_active=bool(r["trailing_active"]) if "trailing_active" in r.keys() else False,
            entry_order_id=r["entry_order_id"] or "",
            exit_order_id=r["exit_order_id"] or "",
            status=LegStatus(r["status"]),
            entry_time=r["entry_time"],
            exit_time=r["exit_time"],
            pnl=r["pnl"] if "pnl" in r.keys() else 0.0,
            greeks=Greeks(**g) if g else Greeks(),
        )

    # ─────────────────────────────────────────────────────────
    # Order log
    # ─────────────────────────────────────────────────────────

    def log_order(self, leg_id: str, order_id: str, action: str,
                  status: str, price: float, quantity: int, message: str = "") -> None:
        with self._conn:
            self._conn.execute(
                """INSERT INTO order_log
                   (timestamp, leg_id, order_id, action, status, price, quantity, message)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    datetime.now().isoformat(),
                    leg_id,
                    order_id,
                    action,
                    status,
                    float(price),
                    int(quantity),
                    message,
                ),
            )

    def get_recent_logs(self, limit: int = 50) -> list:
        rows = self._conn.execute(
            "SELECT * FROM order_log ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]

    # Backward compatibility with older UI code
    def get_recent_order_logs(self, limit: int = 50) -> list:
        return self.get_recent_logs(limit)

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
