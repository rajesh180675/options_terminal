"""
database.py — SQLite persistence layer with WAL mode for crash recovery.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any


class TradingDB:
    """Thread-safe SQLite wrapper with WAL journaling."""

    def __init__(self, db_path: str = "trading.db"):
        self._db_path = db_path
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ------------------------------------------------------------------ schema
    def _init_schema(self):
        with self._cursor() as cur:
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS strategies (
                    id             TEXT PRIMARY KEY,
                    name           TEXT NOT NULL,
                    stock_code     TEXT NOT NULL,
                    status         TEXT NOT NULL DEFAULT 'initializing',
                    target_delta   REAL,
                    ce_strike      REAL,
                    pe_strike      REAL,
                    ce_entry_price REAL DEFAULT 0,
                    pe_entry_price REAL DEFAULT 0,
                    ce_sl_price    REAL DEFAULT 0,
                    pe_sl_price    REAL DEFAULT 0,
                    total_premium  REAL DEFAULT 0,
                    current_mtm    REAL DEFAULT 0,
                    expiry_date    TEXT,
                    lots           INTEGER DEFAULT 1,
                    created_at     TEXT,
                    updated_at     TEXT,
                    closed_at      TEXT
                );

                CREATE TABLE IF NOT EXISTS orders (
                    id               TEXT PRIMARY KEY,
                    strategy_id      TEXT,
                    breeze_order_id  TEXT,
                    stock_code       TEXT NOT NULL,
                    exchange_code    TEXT DEFAULT 'NFO',
                    action           TEXT NOT NULL,
                    strike_price     REAL NOT NULL,
                    right_type       TEXT NOT NULL,
                    expiry_date      TEXT,
                    quantity         INTEGER NOT NULL,
                    price            REAL,
                    order_type       TEXT DEFAULT 'limit',
                    status           TEXT DEFAULT 'pending',
                    filled_price     REAL DEFAULT 0,
                    filled_qty       INTEGER DEFAULT 0,
                    leg_tag          TEXT,
                    chase_count      INTEGER DEFAULT 0,
                    created_at       TEXT,
                    updated_at       TEXT
                );

                CREATE TABLE IF NOT EXISTS positions (
                    id             TEXT PRIMARY KEY,
                    strategy_id    TEXT,
                    stock_code     TEXT NOT NULL,
                    strike_price   REAL NOT NULL,
                    right_type     TEXT NOT NULL,
                    expiry_date    TEXT,
                    quantity       INTEGER NOT NULL,
                    entry_price    REAL NOT NULL,
                    current_price  REAL DEFAULT 0,
                    sl_price       REAL DEFAULT 0,
                    status         TEXT DEFAULT 'open',
                    pnl            REAL DEFAULT 0,
                    created_at     TEXT,
                    updated_at     TEXT
                );

                CREATE TABLE IF NOT EXISTS trade_log (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts        TEXT NOT NULL,
                    level     TEXT NOT NULL,
                    source    TEXT,
                    message   TEXT,
                    data      TEXT
                );
            """)

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _new_id() -> str:
        return uuid.uuid4().hex[:12]

    # ------------------------------------------------------------ strategies
    def create_strategy(self, name: str, stock_code: str, target_delta: float,
                        ce_strike: float, pe_strike: float, expiry_date: str,
                        lots: int = 1) -> str:
        sid = self._new_id()
        now = self._now()
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO strategies
                (id, name, stock_code, status, target_delta, ce_strike, pe_strike,
                 expiry_date, lots, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (sid, name, stock_code, "initializing", target_delta,
                  ce_strike, pe_strike, expiry_date, lots, now, now))
        return sid

    def update_strategy(self, sid: str, **kwargs):
        kwargs["updated_at"] = self._now()
        sets = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [sid]
        with self._cursor() as cur:
            cur.execute(f"UPDATE strategies SET {sets} WHERE id=?", vals)

    def get_strategy(self, sid: str) -> dict | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM strategies WHERE id=?", (sid,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_active_strategies(self) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM strategies WHERE status NOT IN ('closed','error')"
            )
            return [dict(r) for r in cur.fetchall()]

    # ---------------------------------------------------------------- orders
    def create_order(self, strategy_id: str, stock_code: str, action: str,
                     strike_price: float, right_type: str, expiry_date: str,
                     quantity: int, price: float, order_type: str = "limit",
                     leg_tag: str = "") -> str:
        oid = self._new_id()
        now = self._now()
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO orders
                (id, strategy_id, stock_code, action, strike_price, right_type,
                 expiry_date, quantity, price, order_type, status, leg_tag,
                 created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (oid, strategy_id, stock_code, action, strike_price, right_type,
                  expiry_date, quantity, price, order_type, "pending", leg_tag,
                  now, now))
        return oid

    def update_order(self, oid: str, **kwargs):
        kwargs["updated_at"] = self._now()
        sets = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [oid]
        with self._cursor() as cur:
            cur.execute(f"UPDATE orders SET {sets} WHERE id=?", vals)

    def get_orders_for_strategy(self, sid: str) -> list[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM orders WHERE strategy_id=? ORDER BY created_at", (sid,))
            return [dict(r) for r in cur.fetchall()]

    def get_pending_orders(self) -> list[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM orders WHERE status IN ('pending','placed','chasing')")
            return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------- positions
    def create_position(self, strategy_id: str, stock_code: str,
                        strike_price: float, right_type: str, expiry_date: str,
                        quantity: int, entry_price: float, sl_price: float) -> str:
        pid = self._new_id()
        now = self._now()
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO positions
                (id, strategy_id, stock_code, strike_price, right_type, expiry_date,
                 quantity, entry_price, current_price, sl_price, status, pnl,
                 created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (pid, strategy_id, stock_code, strike_price, right_type, expiry_date,
                  quantity, entry_price, entry_price, sl_price, "open", 0.0, now, now))
        return pid

    def update_position(self, pid: str, **kwargs):
        kwargs["updated_at"] = self._now()
        sets = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [pid]
        with self._cursor() as cur:
            cur.execute(f"UPDATE positions SET {sets} WHERE id=?", vals)

    def get_open_positions(self) -> list[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM positions WHERE status='open'")
            return [dict(r) for r in cur.fetchall()]

    def get_positions_for_strategy(self, sid: str) -> list[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM positions WHERE strategy_id=?", (sid,))
            return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------ logs
    def log(self, level: str, source: str, message: str, data: Any = None):
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO trade_log (ts, level, source, message, data) VALUES (?,?,?,?,?)",
                (self._now(), level, source, message,
                 json.dumps(data) if data else None),
            )

    def get_recent_logs(self, limit: int = 200) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM trade_log ORDER BY id DESC LIMIT ?", (limit,)
            )
            return [dict(r) for r in cur.fetchall()]
