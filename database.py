# ═══════════════════════════════════════════════════════════════
# FILE: database.py
# ═══════════════════════════════════════════════════════════════
"""
SQLite persistence with trailing SL columns and schema versioning.
"""

import sqlite3
import threading
import json
from typing import List
from datetime import datetime
from models import (Strategy, Leg, StrategyType, StrategyStatus,
                    LegStatus, OrderSide, OptionRight, Greeks)
from app_config import Config


class Database:
    SCHEMA_VERSION = 3

    def __init__(self, db_path: str = Config.DB_PATH):
        self._path = db_path
        self._local = threading.local()
        self._init()

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._path, check_same_thread=False, timeout=10)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
        return self._local.conn

    def _init(self):
        with self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS strategies (
                    strategy_id TEXT PRIMARY KEY, strategy_type TEXT NOT NULL,
                    stock_code TEXT NOT NULL, target_delta REAL DEFAULT 0,
                    total_pnl REAL DEFAULT 0, status TEXT DEFAULT 'deploying',
                    created_at TEXT, closed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS legs (
                    leg_id TEXT PRIMARY KEY, strategy_id TEXT NOT NULL,
                    stock_code TEXT NOT NULL, exchange_code TEXT DEFAULT 'NFO',
                    strike_price REAL NOT NULL, right TEXT NOT NULL,
                    expiry_date TEXT NOT NULL, side TEXT NOT NULL,
                    quantity INTEGER NOT NULL, entry_price REAL DEFAULT 0,
                    current_price REAL DEFAULT 0, exit_price REAL DEFAULT 0,
                    sl_price REAL DEFAULT 0, sl_percentage REAL DEFAULT 0,
                    lowest_price REAL DEFAULT 0, trailing_active INTEGER DEFAULT 0,
                    entry_order_id TEXT DEFAULT '', exit_order_id TEXT DEFAULT '',
                    status TEXT DEFAULT 'pending', entry_time TEXT, exit_time TEXT,
                    pnl REAL DEFAULT 0, greeks_json TEXT DEFAULT '{}',
                    FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
                );
                CREATE TABLE IF NOT EXISTS order_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL, leg_id TEXT, order_id TEXT,
                    action TEXT, status TEXT, price REAL,
                    quantity INTEGER, message TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_legs_sid ON legs(strategy_id);
                CREATE INDEX IF NOT EXISTS idx_legs_st ON legs(status);
            """)

    def save_strategy(self, s: Strategy):
        with self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO strategies
                   (strategy_id,strategy_type,stock_code,target_delta,
                    total_pnl,status,created_at,closed_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (s.strategy_id, s.strategy_type.value, s.stock_code,
                 s.target_delta, s.total_pnl, s.status.value,
                 s.created_at, s.closed_at))

    def save_leg(self, l: Leg):
        gj = json.dumps({"delta": l.greeks.delta, "gamma": l.greeks.gamma,
                          "theta": l.greeks.theta, "vega": l.greeks.vega, "iv": l.greeks.iv})
        with self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO legs
                   (leg_id,strategy_id,stock_code,exchange_code,strike_price,
                    right,expiry_date,side,quantity,entry_price,current_price,
                    exit_price,sl_price,sl_percentage,lowest_price,trailing_active,
                    entry_order_id,exit_order_id,status,entry_time,exit_time,pnl,greeks_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (l.leg_id, l.strategy_id, l.stock_code, l.exchange_code,
                 l.strike_price, l.right.value, l.expiry_date, l.side.value,
                 l.quantity, l.entry_price, l.current_price, l.exit_price,
                 l.sl_price, l.sl_percentage, l.lowest_price,
                 1 if l.trailing_active else 0,
                 l.entry_order_id, l.exit_order_id, l.status.value,
                 l.entry_time, l.exit_time, l.pnl, gj))

    def update_leg_price(self, lid, price, pnl, sl=None, lowest=None, trailing=None):
        sets = ["current_price=?", "pnl=?"]
        vals = [price, pnl]
        if sl is not None:
            sets.append("sl_price=?"); vals.append(sl)
        if lowest is not None:
            sets.append("lowest_price=?"); vals.append(lowest)
        if trailing is not None:
            sets.append("trailing_active=?"); vals.append(1 if trailing else 0)
        vals.append(lid)
        with self._conn:
            self._conn.execute(f"UPDATE legs SET {','.join(sets)} WHERE leg_id=?", vals)

    def update_leg_status(self, lid, status: LegStatus, **kw):
        sets, vals = ["status=?"], [status.value]
        for k, v in kw.items():
            sets.append(f"{k}=?"); vals.append(v)
        vals.append(lid)
        with self._conn:
            self._conn.execute(f"UPDATE legs SET {','.join(sets)} WHERE leg_id=?", vals)

    def get_active_strategies(self) -> List[Strategy]:
        rows = self._conn.execute(
            "SELECT * FROM strategies WHERE status IN ('deploying','active','partial_exit')"
        ).fetchall()
        out = []
        for r in rows:
            s = Strategy(strategy_id=r["strategy_id"],
                         strategy_type=StrategyType(r["strategy_type"]),
                         stock_code=r["stock_code"], target_delta=r["target_delta"],
                         total_pnl=r["total_pnl"], status=StrategyStatus(r["status"]),
                         created_at=r["created_at"], closed_at=r["closed_at"])
            s.legs = self._legs_for(s.strategy_id)
            out.append(s)
        return out

    def _legs_for(self, sid) -> List[Leg]:
        rows = self._conn.execute("SELECT * FROM legs WHERE strategy_id=?", (sid,)).fetchall()
        return [self._to_leg(r) for r in rows]

    def _to_leg(self, r) -> Leg:
        g = json.loads(r["greeks_json"] or "{}")
        return Leg(
            leg_id=r["leg_id"], strategy_id=r["strategy_id"],
            stock_code=r["stock_code"], exchange_code=r["exchange_code"],
            strike_price=r["strike_price"], right=OptionRight(r["right"]),
            expiry_date=r["expiry_date"], side=OrderSide(r["side"]),
            quantity=r["quantity"], entry_price=r["entry_price"],
            current_price=r["current_price"], exit_price=r["exit_price"],
            sl_price=r["sl_price"], sl_percentage=r["sl_percentage"],
            lowest_price=r["lowest_price"],
            trailing_active=bool(r["trailing_active"]),
            entry_order_id=r["entry_order_id"] or "",
            exit_order_id=r["exit_order_id"] or "",
            status=LegStatus(r["status"]),
            entry_time=r["entry_time"], exit_time=r["exit_time"],
            pnl=r["pnl"], greeks=Greeks(**g) if g else Greeks())

    def log_order(self, lid, oid, action, status, price, qty, msg=""):
        with self._conn:
            self._conn.execute(
                "INSERT INTO order_log (timestamp,leg_id,order_id,action,status,price,quantity,message) VALUES (?,?,?,?,?,?,?,?)",
                (datetime.now().isoformat(), lid, oid, action, status, price, qty, msg))

    def get_recent_logs(self, n=50):
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM order_log ORDER BY id DESC LIMIT ?", (n,)).fetchall()]

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
