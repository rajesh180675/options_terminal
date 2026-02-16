"""
shared_state.py — Thread-safe in-memory state container.
All components read/write through this object.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TickData:
    symbol: str = ""
    stock_code: str = ""
    strike_price: float = 0.0
    right: str = ""
    expiry_date: str = ""
    ltp: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: int = 0
    oi: int = 0
    timestamp: float = 0.0


@dataclass
class GreeksData:
    iv: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0


@dataclass
class OptionStrike:
    strike: float = 0.0
    ce_tick: TickData = field(default_factory=TickData)
    pe_tick: TickData = field(default_factory=TickData)
    ce_greeks: GreeksData = field(default_factory=GreeksData)
    pe_greeks: GreeksData = field(default_factory=GreeksData)


class SharedState:
    """Central state object shared across all threads."""

    def __init__(self):
        self._lock = threading.RLock()

        # Market data keyed by (stock_code, strike, right)
        self._ticks: dict[tuple[str, float, str], TickData] = {}

        # Greeks keyed by (stock_code, strike, right)
        self._greeks: dict[tuple[str, float, str], GreeksData] = {}

        # Spot prices keyed by stock_code
        self._spots: dict[str, float] = {}

        # Option chain (sorted strikes) keyed by stock_code
        self._chains: dict[str, list[OptionStrike]] = {}

        # Active strategy states (quick access mirrors DB)
        self._strategies: dict[str, dict] = {}

        # Positions mirror
        self._positions: dict[str, dict] = {}

        # Global P&L
        self._total_mtm: float = 0.0

        # Connection status
        self._ws_connected: bool = False
        self._last_tick_time: float = 0.0
        self._api_connected: bool = False

        # Log buffer for UI (ring buffer)
        self._log_buffer: list[dict] = []
        self._log_max = 500

        # Kill switch
        self._kill_switch_active: bool = False

        # Engine running flag
        self._engine_running: bool = False

    # --------------------------------------------------------- spot price
    def set_spot(self, stock_code: str, price: float):
        with self._lock:
            self._spots[stock_code] = price

    def get_spot(self, stock_code: str) -> float:
        with self._lock:
            return self._spots.get(stock_code, 0.0)

    # ------------------------------------------------------- tick data
    def update_tick(self, stock_code: str, strike: float, right: str, tick: TickData):
        key = (stock_code, strike, right)
        with self._lock:
            self._ticks[key] = tick
            self._last_tick_time = time.time()

    def get_tick(self, stock_code: str, strike: float, right: str) -> TickData | None:
        with self._lock:
            return self._ticks.get((stock_code, strike, right))

    def get_ltp(self, stock_code: str, strike: float, right: str) -> float:
        tick = self.get_tick(stock_code, strike, right)
        return tick.ltp if tick else 0.0

    # -------------------------------------------------------- greeks
    def update_greeks(self, stock_code: str, strike: float, right: str,
                      greeks: GreeksData):
        with self._lock:
            self._greeks[(stock_code, strike, right)] = greeks

    def get_greeks(self, stock_code: str, strike: float, right: str) -> GreeksData:
        with self._lock:
            return self._greeks.get((stock_code, strike, right), GreeksData())

    # ------------------------------------------------------ option chain
    def set_chain(self, stock_code: str, chain: list[OptionStrike]):
        with self._lock:
            self._chains[stock_code] = chain

    def get_chain(self, stock_code: str) -> list[OptionStrike]:
        with self._lock:
            return list(self._chains.get(stock_code, []))

    # ------------------------------------------------------ strategies
    def set_strategy(self, sid: str, data: dict):
        with self._lock:
            self._strategies[sid] = data

    def get_strategy(self, sid: str) -> dict | None:
        with self._lock:
            return self._strategies.get(sid)

    def get_all_strategies(self) -> dict[str, dict]:
        with self._lock:
            return dict(self._strategies)

    def remove_strategy(self, sid: str):
        with self._lock:
            self._strategies.pop(sid, None)

    # ------------------------------------------------------ positions
    def set_position(self, pid: str, data: dict):
        with self._lock:
            self._positions[pid] = data

    def get_open_positions(self) -> dict[str, dict]:
        with self._lock:
            return {k: v for k, v in self._positions.items()
                    if v.get("status") == "open"}

    def remove_position(self, pid: str):
        with self._lock:
            self._positions.pop(pid, None)

    # -------------------------------------------------------- P&L
    def set_total_mtm(self, value: float):
        with self._lock:
            self._total_mtm = value

    def get_total_mtm(self) -> float:
        with self._lock:
            return self._total_mtm

    # ------------------------------------------------- connection status
    def set_ws_connected(self, val: bool):
        with self._lock:
            self._ws_connected = val

    def is_ws_connected(self) -> bool:
        with self._lock:
            return self._ws_connected

    def get_last_tick_age(self) -> float:
        with self._lock:
            if self._last_tick_time == 0:
                return 999.0
            return time.time() - self._last_tick_time

    def set_api_connected(self, val: bool):
        with self._lock:
            self._api_connected = val

    def is_api_connected(self) -> bool:
        with self._lock:
            return self._api_connected

    # ---------------------------------------------------- kill switch
    def activate_kill_switch(self):
        with self._lock:
            self._kill_switch_active = True

    def deactivate_kill_switch(self):
        with self._lock:
            self._kill_switch_active = False

    def is_kill_switch_active(self) -> bool:
        with self._lock:
            return self._kill_switch_active

    # ------------------------------------------------------- engine
    def set_engine_running(self, val: bool):
        with self._lock:
            self._engine_running = val

    def is_engine_running(self) -> bool:
        with self._lock:
            return self._engine_running

    # ----------------------------------------------------------- logs
    def add_log(self, level: str, source: str, message: str,
                data: Any = None):
        entry = {
            "ts": time.strftime("%H:%M:%S"),
            "level": level,
            "source": source,
            "message": message,
            "data": data,
        }
        with self._lock:
            self._log_buffer.append(entry)
            if len(self._log_buffer) > self._log_max:
                self._log_buffer = self._log_buffer[-self._log_max:]

    def get_logs(self, n: int = 100) -> list[dict]:
        with self._lock:
            return list(self._log_buffer[-n:])
