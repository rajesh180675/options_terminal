"""
shared_state.py

Thread-safe in-memory state for:
  - latest ticks per instrument (keyed by canonical feed_key)
  - strategies snapshot for UI
  - spots per underlying
  - rolling MTM history
  - portfolio greeks
  - cached option chain snapshot
  - adjustment suggestions
  - ring-buffer logs

This module is intentionally lightweight and dependency-free.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple, Any

from models import TickData, Strategy, LogEntry, Greeks


class SharedState:
    def __init__(self, max_logs: int = 800, max_mtm_points: int = 1500):
        self._lock = threading.RLock()

        # feed_key -> TickData
        self._ticks: Dict[str, TickData] = {}

        # stock_code -> spot
        self._spot: Dict[str, float] = {}

        # strategy snapshots (copied from DB periodically)
        self._strategies: List[Strategy] = []

        # cached option chain + timestamp (epoch seconds)
        self._chain_cache: List[dict] = []
        self._chain_cache_time: float = 0.0
        self._chain_cache_key: str = ""  # e.g. "NIFTY|2025-02-18"

        # MTM history points: [{"time": "HH:MM:SS", "mtm": float}, ...]
        self._mtm_history: Deque[dict] = deque(maxlen=max_mtm_points)
        self._total_mtm: float = 0.0

        # portfolio greeks (net)
        self._portfolio_greeks: Greeks = Greeks()

        # adjustments (from adjustment engine)
        self._adjustments: List[Any] = []

        # logs
        self._logs: Deque[LogEntry] = deque(maxlen=max_logs)

        # flags
        self._connected: bool = False
        self._ws_connected: bool = False
        self._engine_running: bool = False
        self._panic_triggered: bool = False
        self._auto_exit_done: bool = False

    # ── Tick store ─────────────────────────────────────────────

    def update_tick(self, tick: TickData) -> None:
        with self._lock:
            self._ticks[tick.feed_key] = tick

    def get_tick(self, feed_key: str) -> Optional[TickData]:
        with self._lock:
            return self._ticks.get(feed_key)

    def get_all_ticks(self) -> Dict[str, TickData]:
        with self._lock:
            return dict(self._ticks)

    # ── Spot store ─────────────────────────────────────────────

    def set_spot(self, stock_code: str, price: float) -> None:
        with self._lock:
            self._spot[stock_code] = float(price)

    def get_spot(self, stock_code: str) -> float:
        with self._lock:
            return float(self._spot.get(stock_code, 0.0))

    def get_all_spots(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._spot)

    # ── Strategies snapshot ────────────────────────────────────

    def set_strategies(self, strategies: List[Strategy]) -> None:
        with self._lock:
            # Store a shallow copy; Strategy/Leg objects are mutable but we
            # treat this list as read-only from UI.
            self._strategies = list(strategies)

    def get_strategies(self) -> List[Strategy]:
        with self._lock:
            return list(self._strategies)

    # ── Chain cache ────────────────────────────────────────────

    def set_chain_cache(self, key: str, chain: List[dict], ts: float) -> None:
        with self._lock:
            self._chain_cache_key = key
            self._chain_cache = list(chain)
            self._chain_cache_time = float(ts)

    def get_chain_cache(self) -> Tuple[str, List[dict], float]:
        with self._lock:
            return self._chain_cache_key, list(self._chain_cache), float(self._chain_cache_time)

    # ── MTM history ────────────────────────────────────────────

    def set_total_mtm(self, value: float) -> None:
        with self._lock:
            self._total_mtm = float(value)
            self._mtm_history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "mtm": float(value),
            })

    def get_total_mtm(self) -> float:
        with self._lock:
            return float(self._total_mtm)

    def get_mtm_history(self) -> List[dict]:
        with self._lock:
            return list(self._mtm_history)

    # ── Portfolio Greeks ───────────────────────────────────────

    def set_portfolio_greeks(self, g: Greeks) -> None:
        with self._lock:
            self._portfolio_greeks = g

    def get_portfolio_greeks(self) -> Greeks:
        with self._lock:
            return self._portfolio_greeks

    # ── Adjustments ────────────────────────────────────────────

    def set_adjustments(self, adjustments: list) -> None:
        with self._lock:
            self._adjustments = list(adjustments) if adjustments else []

    def get_adjustments(self) -> list:
        with self._lock:
            return list(self._adjustments)

    # ── Logs ───────────────────────────────────────────────────

    def add_log(self, level: str, source: str, message: str, data: dict | None = None) -> None:
        with self._lock:
            self._logs.append(LogEntry(level=level, source=source, message=message, data=data))

    def get_logs(self, n: int = 150) -> List[LogEntry]:
        with self._lock:
            if n <= 0:
                return []
            return list(self._logs)[-n:]

    # ── Flags ──────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @connected.setter
    def connected(self, v: bool) -> None:
        with self._lock:
            self._connected = bool(v)

    @property
    def ws_connected(self) -> bool:
        with self._lock:
            return self._ws_connected

    @ws_connected.setter
    def ws_connected(self, v: bool) -> None:
        with self._lock:
            self._ws_connected = bool(v)

    @property
    def engine_running(self) -> bool:
        with self._lock:
            return self._engine_running

    @engine_running.setter
    def engine_running(self, v: bool) -> None:
        with self._lock:
            self._engine_running = bool(v)

    @property
    def panic_triggered(self) -> bool:
        with self._lock:
            return self._panic_triggered

    @panic_triggered.setter
    def panic_triggered(self, v: bool) -> None:
        with self._lock:
            self._panic_triggered = bool(v)

    @property
    def auto_exit_done(self) -> bool:
        with self._lock:
            return self._auto_exit_done

    @auto_exit_done.setter
    def auto_exit_done(self, v: bool) -> None:
        with self._lock:
            self._auto_exit_done = bool(v)
