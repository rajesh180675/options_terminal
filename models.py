# ═══════════════════════════════════════════════════════════════
# FILE: models.py
# ═══════════════════════════════════════════════════════════════
"""
Immutable-ish data containers for the entire system.
Using dataclasses for clarity; fields are mutable for in-place updates.
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any


# ── Enums ───────────────────────────────────────────────────

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"

class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    MODIFYING = "modifying"

class LegStatus(str, Enum):
    PENDING = "pending"
    ENTERING = "entering"
    ACTIVE = "active"
    SL_TRIGGERED = "sl_triggered"
    EXITING = "exiting"
    SQUARED_OFF = "squared_off"
    ERROR = "error"

class OptionRight(str, Enum):
    CALL = "call"
    PUT = "put"

class StrategyType(str, Enum):
    SHORT_STRADDLE = "short_straddle"
    SHORT_STRANGLE = "short_strangle"

class StrategyStatus(str, Enum):
    DEPLOYING = "deploying"
    ACTIVE = "active"
    PARTIAL_EXIT = "partial_exit"
    CLOSED = "closed"
    ERROR = "error"


# ── Data containers ─────────────────────────────────────────

@dataclass
class Greeks:
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    iv: float = 0.0


@dataclass
class TickData:
    stock_code: str = ""
    strike_price: float = 0.0
    right: str = ""
    expiry_date: str = ""
    ltp: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    volume: int = 0
    oi: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def feed_key(self) -> str:
        return f"{self.stock_code}|{self.strike_price}|{self.right}|{self.expiry_date}"


@dataclass
class OptionQuote:
    stock_code: str = ""
    strike_price: float = 0.0
    right: OptionRight = OptionRight.CALL
    expiry_date: str = ""
    ltp: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    volume: int = 0
    oi: int = 0
    greeks: Greeks = field(default_factory=Greeks)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Leg:
    leg_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    strategy_id: str = ""
    stock_code: str = "NIFTY"
    exchange_code: str = "NFO"
    strike_price: float = 0.0
    right: OptionRight = OptionRight.CALL
    expiry_date: str = ""
    side: OrderSide = OrderSide.SELL
    quantity: int = 0
    entry_price: float = 0.0
    current_price: float = 0.0
    exit_price: float = 0.0
    sl_price: float = 0.0
    sl_percentage: float = 0.0
    entry_order_id: str = ""
    exit_order_id: str = ""
    status: LegStatus = LegStatus.PENDING
    entry_time: Optional[str] = None
    exit_time: Optional[str] = None
    pnl: float = 0.0
    greeks: Greeks = field(default_factory=Greeks)

    def compute_pnl(self) -> float:
        if self.side == OrderSide.SELL:
            self.pnl = (self.entry_price - self.current_price) * self.quantity
        else:
            self.pnl = (self.current_price - self.entry_price) * self.quantity
        return self.pnl

    @property
    def feed_key(self) -> str:
        return f"{self.stock_code}|{self.strike_price}|{self.right.value}|{self.expiry_date}"


@dataclass
class Strategy:
    strategy_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    strategy_type: StrategyType = StrategyType.SHORT_STRADDLE
    stock_code: str = "NIFTY"
    target_delta: float = 0.0
    legs: List[Leg] = field(default_factory=list)
    total_pnl: float = 0.0
    status: StrategyStatus = StrategyStatus.DEPLOYING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    closed_at: Optional[str] = None

    def compute_total_pnl(self) -> float:
        self.total_pnl = sum(leg.compute_pnl() for leg in self.legs)
        return self.total_pnl


@dataclass
class LogEntry:
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S.%f")[:-3])
    level: str = "INFO"
    source: str = ""
    message: str = ""
    data: Optional[Dict[str, Any]] = None
