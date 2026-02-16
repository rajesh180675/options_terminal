# ═══════════════════════════════════════════════════════════════
# FILE: shared_state.py
# ═══════════════════════════════════════════════════════════════
"""
Thread-safe state bus with MTM history for charting.
"""

import time
import threading
from typing import Dict, List, Optional, Tuple
from collections import deque
from models import TickData, Strategy, LogEntry, Greeks


class SharedState:
    def __init__(self, max_logs=500, max_mtm_history=500):
        self._lock = threading.RLock()
        self._ticks: Dict[str, TickData] = {}
        self._strategies: List[Strategy] = []
        self._spot: Dict[str, float] = {}
        self._logs: deque = deque(maxlen=max_logs)
        self._chain_cache: List[dict] = []
        self._chain_time: float = 0
        self._mtm_history: deque = deque(maxlen=max_mtm_history)
        self._portfolio_greeks: Greeks = Greeks()
        self._connected = False
        self._ws_connected = False
        self._engine_running = False
        self._panic_triggered = False
        self._total_mtm = 0.0
        self._auto_exit_done = False

    # ── Ticks ────────────────────────────────────────────────

    def update_tick(self, tick: TickData):
        with self._lock:
            self._ticks[tick.feed_key] = tick

    def get_tick(self, key: str) -> Optional[TickData]:
        with self._lock:
            return self._ticks.get(key)

    def get_all_ticks(self) -> Dict[str, TickData]:
        with self._lock:
            return dict(self._ticks)

    # ── Spot ─────────────────────────────────────────────────

    def set_spot(self, code: str, price: float):
        with self._lock:
            self._spot[code] = price

    def get_spot(self, code: str) -> float:
        with self._lock:
            return self._spot.get(code, 0.0)

    # ── Strategies ───────────────────────────────────────────

    def set_strategies(self, s: List[Strategy]):
        with self._lock:
            self._strategies = s

    def get_strategies(self) -> List[Strategy]:
        with self._lock:
            return list(self._strategies)

    # ── Chain cache ──────────────────────────────────────────

    def set_chain(self, c: List[dict], t: float):
        with self._lock:
            self._chain_cache = c
            self._chain_time = t

    def get_chain(self) -> Tuple[List[dict], float]:
        with self._lock:
            return list(self._chain_cache), self._chain_time

    # ── MTM ──────────────────────────────────────────────────

    def set_total_mtm(self, v: float):
        from datetime import datetime
        with self._lock:
            self._total_mtm = v
            self._mtm_history.append({"time": datetime.now().strftime("%H:%M:%S"), "mtm": v})

    def get_total_mtm(self) -> float:
        with self._lock:
            return self._total_mtm

    def get_mtm_history(self) -> List[dict]:
        with self._lock:
            return list(self._mtm_history)

    # ── Portfolio Greeks ─────────────────────────────────────

    def set_portfolio_greeks(self, g: Greeks):
        with self._lock:
            self._portfolio_greeks = g

    def get_portfolio_greeks(self) -> Greeks:
        with self._lock:
            return self._portfolio_greeks

    # ── Logs ─────────────────────────────────────────────────

    def add_log(self, level, source, message, data=None):
        with self._lock:
            self._logs.append(LogEntry(level=level, source=source,
                                       message=message, data=data))

    def get_logs(self, n=100) -> List[LogEntry]:
        with self._lock:
            return list(self._logs)[-n:]

    # ── Flags ────────────────────────────────────────────────

    @property
    def connected(self):
        with self._lock: return self._connected
    @connected.setter
    def connected(self, v):
        with self._lock: self._connected = v

    @property
    def ws_connected(self):
        with self._lock: return self._ws_connected
    @ws_connected.setter
    def ws_connected(self, v):
        with self._lock: self._ws_connected = v

    @property
    def engine_running(self):
        with self._lock: return self._engine_running
    @engine_running.setter
    def engine_running(self, v):
        with self._lock: self._engine_running = v

    @property
    def panic_triggered(self):
        with self._lock: return self._panic_triggered
    @panic_triggered.setter
    def panic_triggered(self, v):
        with self._lock: self._panic_triggered = v

    @property
    def auto_exit_done(self):
        with self._lock: return self._auto_exit_done
    @auto_exit_done.setter
    def auto_exit_done(self, v):
        with self._lock: self._auto_exit_done = v
