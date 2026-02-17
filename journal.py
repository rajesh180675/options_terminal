# ═══════════════════════════════════════════════════════════════
# FILE: journal.py  (NEW)
# ═══════════════════════════════════════════════════════════════
"""
Trade journal and performance analytics:
  • Strategy-level outcome tracking
  • Daily P&L snapshots
  • Win rate, profit factor, max drawdown
  • Sharpe-like ratio
  • Strategy-type breakdown
"""

import sqlite3
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass

from app_config import Config
from models import Strategy, StrategyStatus
from utils import LOG


@dataclass
class JournalEntry:
    strategy_id: str
    strategy_type: str
    stock_code: str
    entry_date: str
    exit_date: str
    legs_count: int
    total_premium: float
    pnl: float
    exit_reason: str  # "sl_hit", "profit_target", "panic", "auto_exit", "manual"
    duration_minutes: int = 0


@dataclass
class DailySnapshot:
    date: str
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    strategies_opened: int
    strategies_closed: int
    trades_count: int


@dataclass
class PerformanceStats:
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_duration_min: float = 0.0
    expectancy: float = 0.0  # avg_win * win_rate - avg_loss * loss_rate


class TradeJournal:
    """Persistent trade journal with statistics."""

    def __init__(self, db_path: str = Config.DB_PATH):
        self._path = db_path
        self._local = threading.local()
        self._init()

    @property
    def _conn(self):
        if not hasattr(self._local, "jconn") or self._local.jconn is None:
            self._local.jconn = sqlite3.connect(
                self._path, check_same_thread=False, timeout=10
            )
            self._local.jconn.row_factory = sqlite3.Row
        return self._local.jconn

    def _init(self):
        with self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT, strategy_type TEXT,
                    stock_code TEXT, entry_date TEXT,
                    exit_date TEXT, legs_count INTEGER,
                    total_premium REAL, pnl REAL,
                    exit_reason TEXT, duration_minutes INTEGER
                );
                CREATE TABLE IF NOT EXISTS daily_pnl (
                    date TEXT PRIMARY KEY,
                    realized REAL DEFAULT 0,
                    unrealized REAL DEFAULT 0,
                    total REAL DEFAULT 0,
                    opened INTEGER DEFAULT 0,
                    closed INTEGER DEFAULT 0,
                    trades INTEGER DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_journal_date
                    ON journal(exit_date);
            """)

    def record_strategy(self, strategy: Strategy, exit_reason: str = "manual"):
        """Record a closed strategy into the journal."""
        if strategy.status != StrategyStatus.CLOSED:
            return

        entry_times = [l.entry_time for l in strategy.legs if l.entry_time]
        exit_times = [l.exit_time for l in strategy.legs if l.exit_time]
        entry_date = min(entry_times) if entry_times else strategy.created_at
        exit_date = max(exit_times) if exit_times else datetime.now().isoformat()

        total_premium = sum(
            l.entry_price * l.quantity for l in strategy.legs
        )

        duration = 0
        try:
            t1 = datetime.fromisoformat(entry_date)
            t2 = datetime.fromisoformat(exit_date)
            duration = int((t2 - t1).total_seconds() / 60)
        except Exception:
            pass

        with self._conn:
            self._conn.execute(
                """INSERT INTO journal
                   (strategy_id, strategy_type, stock_code, entry_date,
                    exit_date, legs_count, total_premium, pnl,
                    exit_reason, duration_minutes)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (strategy.strategy_id, strategy.strategy_type.value,
                 strategy.stock_code, entry_date, exit_date,
                 len(strategy.legs), total_premium,
                 strategy.total_pnl, exit_reason, duration),
            )

    def update_daily_snapshot(self, realized: float, unrealized: float,
                              opened: int, closed: int, trades: int):
        today = datetime.now().strftime("%Y-%m-%d")
        total = realized + unrealized
        with self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO daily_pnl
                   (date, realized, unrealized, total, opened, closed, trades)
                   VALUES (?,?,?,?,?,?,?)""",
                (today, realized, unrealized, total, opened, closed, trades),
            )

    def get_entries(self, limit: int = 100) -> List[JournalEntry]:
        rows = self._conn.execute(
            "SELECT * FROM journal ORDER BY exit_date DESC LIMIT ?", (limit,)
        ).fetchall()
        return [JournalEntry(
            strategy_id=r["strategy_id"],
            strategy_type=r["strategy_type"],
            stock_code=r["stock_code"],
            entry_date=r["entry_date"],
            exit_date=r["exit_date"],
            legs_count=r["legs_count"],
            total_premium=r["total_premium"],
            pnl=r["pnl"],
            exit_reason=r["exit_reason"],
            duration_minutes=r["duration_minutes"],
        ) for r in rows]

    def get_daily_pnl(self, days: int = 30) -> List[Dict]:
        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        rows = self._conn.execute(
            "SELECT * FROM daily_pnl WHERE date >= ? ORDER BY date", (since,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self, days: int = 0) -> PerformanceStats:
        """Compute performance statistics from journal entries."""
        if days > 0:
            since = (datetime.now() - timedelta(days=days)).isoformat()
            rows = self._conn.execute(
                "SELECT pnl, duration_minutes FROM journal WHERE exit_date >= ?",
                (since,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT pnl, duration_minutes FROM journal"
            ).fetchall()

        if not rows:
            return PerformanceStats()

        pnls = [r["pnl"] for r in rows]
        durations = [r["duration_minutes"] for r in rows]

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_win = sum(wins) if wins else 0
        total_loss = abs(sum(losses)) if losses else 0

        # Max drawdown
        cumulative = []
        running = 0
        peak = 0
        max_dd = 0
        for p in pnls:
            running += p
            cumulative.append(running)
            if running > peak:
                peak = running
            dd = peak - running
            if dd > max_dd:
                max_dd = dd

        # Consecutive wins/losses
        max_cw, max_cl, cw, cl = 0, 0, 0, 0
        for p in pnls:
            if p > 0:
                cw += 1
                cl = 0
            else:
                cl += 1
                cw = 0
            max_cw = max(max_cw, cw)
            max_cl = max(max_cl, cl)

        n = len(pnls)
        wr = len(wins) / n if n else 0
        avg_w = total_win / len(wins) if wins else 0
        avg_l = total_loss / len(losses) if losses else 0

        return PerformanceStats(
            total_trades=n,
            winners=len(wins),
            losers=len(losses),
            win_rate=round(wr * 100, 1),
            total_pnl=round(sum(pnls), 2),
            avg_win=round(avg_w, 2),
            avg_loss=round(avg_l, 2),
            profit_factor=round(total_win / max(total_loss, 1), 2),
            max_drawdown=round(max_dd, 2),
            max_consecutive_wins=max_cw,
            max_consecutive_losses=max_cl,
            best_trade=round(max(pnls), 2) if pnls else 0,
            worst_trade=round(min(pnls), 2) if pnls else 0,
            avg_duration_min=round(sum(durations) / max(n, 1), 1),
            expectancy=round(avg_w * wr - avg_l * (1 - wr), 2),
        )

    def get_strategy_type_breakdown(self) -> List[Dict]:
        rows = self._conn.execute("""
            SELECT strategy_type,
                   COUNT(*) as count,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(pnl) as total_pnl,
                   AVG(pnl) as avg_pnl,
                   AVG(duration_minutes) as avg_duration
            FROM journal GROUP BY strategy_type
        """).fetchall()
        return [dict(r) for r in rows]
