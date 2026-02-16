# ═══════════════════════════════════════════════════════════════
# FILE: shared_state.py
# ═══════════════════════════════════════════════════════════════
"""
Thread-safe state bus shared between the engine thread and the Streamlit UI.
"""

import threading
from typing import Dict, List, Optional
from collections import deque

from models import TickData, Strategy, LogEntry


class SharedState:
    """Every mutable field is guarded by a single reentrant lock."""

    def __init__(self, max_logs: int = 500):
        self._lock = threading.RLock()

        # Market data: feed_key → TickData
        self._ticks: Dict[str, TickData] = {}

        # Active strategies
        self._strategies: List[Strategy] = []

        # Spot prices
        self._spot_prices: Dict[str, float] = {}

        # Log ring buffer
        self._logs: deque = deque(maxlen=max_logs)

        # Flags
        self._connected: bool = False
        self._ws_connected: bool = False
        self._engine_running: bool = False
        self._kill_switch_armed: bool = False
        self._panic_triggered: bool = False

        # Global P&L
        self._total_mtm: float = 0.0

    # ── Tick data ────────────────────────────────────────────

    def update_tick(self, tick: TickData):
        with self._lock:
            self._ticks[tick.feed_key] = tick

    def get_tick(self, feed_key: str) -> Optional[TickData]:
        with self._lock:
            return self._ticks.get(feed_key)

    def get_all_ticks(self) -> Dict[str, TickData]:
        with self._lock:
            return dict(self._ticks)

    # ── Spot prices ──────────────────────────────────────────

    def set_spot(self, stock_code: str, price: float):
        with self._lock:
            self._spot_prices[stock_code] = price

    def get_spot(self, stock_code: str) -> float:
        with self._lock:
            return self._spot_prices.get(stock_code, 0.0)

    # ── Strategies ───────────────────────────────────────────

    def set_strategies(self, strategies: List[Strategy]):
        with self._lock:
            self._strategies = strategies

    def get_strategies(self) -> List[Strategy]:
        with self._lock:
            return list(self._strategies)

    # ── P&L ──────────────────────────────────────────────────

    def set_total_mtm(self, value: float):
        with self._lock:
            self._total_mtm = value

    def get_total_mtm(self) -> float:
        with self._lock:
            return self._total_mtm

    # ── Logging ──────────────────────────────────────────────

    def add_log(self, level: str, source: str, message: str, data=None):
        entry = LogEntry(level=level, source=source, message=message, data=data)
        with self._lock:
            self._logs.append(entry)

    def get_logs(self, n: int = 100) -> List[LogEntry]:
        with self._lock:
            return list(self._logs)[-n:]

    # ── Flags ────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @connected.setter
    def connected(self, val: bool):
        with self._lock:
            self._connected = val

    @property
    def ws_connected(self) -> bool:
        with self._lock:
            return self._ws_connected

    @ws_connected.setter
    def ws_connected(self, val: bool):
        with self._lock:
            self._ws_connected = val

    @property
    def engine_running(self) -> bool:
        with self._lock:
            return self._engine_running

    @engine_running.setter
    def engine_running(self, val: bool):
        with self._lock:
            self._engine_running = val

    @property
    def panic_triggered(self) -> bool:
        with self._lock:
            return self._panic_triggered

    @panic_triggered.setter
    def panic_triggered(self, val: bool):
        with self._lock:
            self._panic_triggered = val
