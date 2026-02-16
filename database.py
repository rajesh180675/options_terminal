# ═══════════════════════════════════════════════════════════════
# FILE: database.py
# ═══════════════════════════════════════════════════════════════
"""
SQLite persistence layer with schema versioning.
Fixes: added migration support, proper thread-local connections.
"""

import sqlite3
import threading
import json
from typing import List
from datetime import datetime

from models import (
    Strategy, Leg, StrategyType, StrategyStatus,
    LegStatus, OrderSide, OptionRight, Greeks,
)
from config import Config


class Database:
    SCHEMA_VERSION = 2

    def __init__(self, db_path: str = Config.DB_PATH):
        self._db_path = db_path
        self._local = threading.local()
        self._init_schema()

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self._db_path, check_same_thread=False, timeout=10.0
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
        return self._local.conn

    def _init_schema(self):
        with self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                );

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

                CREATE INDEX IF NOT EXISTS idx_legs_strategy ON legs(strategy_id);
                CREATE INDEX IF NOT EXISTS idx_legs_status ON legs(status);
                CREATE INDEX IF NOT EXISTS idx_order_log_time ON order_log(timestamp);
            """)

            # Version tracking
            row = self._conn.execute(
                "SELECT MAX(version) as v FROM schema_version"
            ).fetchone()
            current = row["v"] if row and row["v"] else 0
            if current < self.SCHEMA_VERSION:
                self._migrate(current)
                self._conn.execute(
                    "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                    (self.SCHEMA_VERSION,),
                )

    def _migrate(self, from_version: int):
        """Run migrations sequentially."""
        if from_version < 2:
            # Add any new columns here in future
            pass

    # ── Strategy CRUD ────────────────────────────────────────

    def save_strategy(self, s: Strategy):
        with self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO strategies
                   (strategy_id, strategy_type, stock_code, target_delta,
                    total_pnl, status, created_at, closed_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (s.strategy_id, s.strategy_type.value, s.stock_code,
                 s.target_delta, s.total_pnl, s.status.value,
                 s.created_at, s.closed_at),
            )

    def update_strategy_status(self, sid: str, status: StrategyStatus, pnl: float = 0):
        with self._conn:
            closed = datetime.now().isoformat() if status == StrategyStatus.CLOSED else None
            self._conn.execute(
                "UPDATE strategies SET status=?, total_pnl=?, closed_at=? WHERE strategy_id=?",
                (status.value, pnl, closed, sid),
            )

    def get_active_strategies(self) -> List[Strategy]:
        rows = self._conn.execute(
            "SELECT * FROM strategies WHERE status IN ('deploying','active','partial_exit')"
        ).fetchall()
        strategies = []
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

    def get_all_strategies(self) -> List[Strategy]:
        rows = self._conn.execute("SELECT * FROM strategies ORDER BY created_at DESC").fetchall()
        strategies = []
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

    # ── Leg CRUD ─────────────────────────────────────────────

    def save_leg(self, leg: Leg):
        greeks_json = json.dumps({
            "delta": leg.greeks.delta, "gamma": leg.greeks.gamma,
            "theta": leg.greeks.theta, "vega": leg.greeks.vega,
            "iv": leg.greeks.iv,
        })
        with self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO legs
                   (leg_id, strategy_id, stock_code, exchange_code,
                    strike_price, right, expiry_date, side, quantity,
                    entry_price, current_price, exit_price, sl_price,
                    sl_percentage, entry_order_id, exit_order_id,
                    status, entry_time, exit_time, pnl, greeks_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (leg.leg_id, leg.strategy_id, leg.stock_code,
                 leg.exchange_code, leg.strike_price, leg.right.value,
                 leg.expiry_date, leg.side.value, leg.quantity,
                 leg.entry_price, leg.current_price, leg.exit_price,
                 leg.sl_price, leg.sl_percentage, leg.entry_order_id,
                 leg.exit_order_id, leg.status.value, leg.entry_time,
                 leg.exit_time, leg.pnl, greeks_json),
            )

    def update_leg_price(self, leg_id: str, current_price: float, pnl: float):
        with self._conn:
            self._conn.execute(
                "UPDATE legs SET current_price=?, pnl=? WHERE leg_id=?",
                (current_price, pnl, leg_id),
            )

    def update_leg_status(self, leg_id: str, status: LegStatus, **kwargs):
        sets = ["status=?"]
        vals = [status.value]
        for k, v in kwargs.items():
            sets.append(f"{k}=?")
            vals.append(v)
        vals.append(leg_id)
        with self._conn:
            self._conn.execute(
                f"UPDATE legs SET {', '.join(sets)} WHERE leg_id=?", vals
            )

    def get_legs_for_strategy(self, sid: str) -> List[Leg]:
        rows = self._conn.execute(
            "SELECT * FROM legs WHERE strategy_id=?", (sid,)
        ).fetchall()
        return [self._row_to_leg(r) for r in rows]

    def get_active_legs(self) -> List[Leg]:
        rows = self._conn.execute(
            "SELECT * FROM legs WHERE status IN ('active','entering','sl_triggered','exiting')"
        ).fetchall()
        return [self._row_to_leg(r) for r in rows]

    def _row_to_leg(self, r) -> Leg:
        g = json.loads(r["greeks_json"] or "{}")
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
            entry_order_id=r["entry_order_id"] or "",
            exit_order_id=r["exit_order_id"] or "",
            status=LegStatus(r["status"]),
            entry_time=r["entry_time"],
            exit_time=r["exit_time"],
            pnl=r["pnl"],
            greeks=Greeks(**g) if g else Greeks(),
        )

    # ── Order log ────────────────────────────────────────────

    def log_order(self, leg_id: str, order_id: str, action: str,
                  status: str, price: float, quantity: int, message: str = ""):
        with self._conn:
            self._conn.execute(
                """INSERT INTO order_log
                   (timestamp, leg_id, order_id, action, status, price, quantity, message)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (datetime.now().isoformat(), leg_id, order_id,
                 action, status, price, quantity, message),
            )

    def get_recent_order_logs(self, limit: int = 50) -> list:
        rows = self._conn.execute(
            "SELECT * FROM order_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
