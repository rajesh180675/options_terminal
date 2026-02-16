# ═══════════════════════════════════════════════════════════════
# FILE: models.py
# ═══════════════════════════════════════════════════════════════
"""
Data containers with trailing SL support and canonical feed_key.
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"

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
    MANUAL_SELL = "manual_sell"

class StrategyStatus(str, Enum):
    DEPLOYING = "deploying"
    ACTIVE = "active"
    PARTIAL_EXIT = "partial_exit"
    CLOSED = "closed"
    ERROR = "error"


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
        r = self.right.strip().lower()
        exp = self.expiry_date[:10] if self.expiry_date else ""
        return f"{self.stock_code}|{int(self.strike_price)}|{r}|{exp}"


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
    lowest_price: float = 0.0      # for trailing SL
    trailing_active: bool = False   # trailing SL engaged?
    entry_order_id: str = ""
    exit_order_id: str = ""
    status: LegStatus = LegStatus.PENDING
    entry_time: Optional[str] = None
    exit_time: Optional[str] = None
    pnl: float = 0.0
    greeks: Greeks = field(default_factory=Greeks)

    def compute_pnl(self) -> float:
        if self.entry_price <= 0:
            self.pnl = 0.0
            return 0.0
        if self.side == OrderSide.SELL:
            self.pnl = (self.entry_price - self.current_price) * self.quantity
        else:
            self.pnl = (self.current_price - self.entry_price) * self.quantity
        return self.pnl

    @property
    def feed_key(self) -> str:
        r = self.right.value.lower()
        exp = self.expiry_date[:10] if self.expiry_date else ""
        return f"{self.stock_code}|{int(self.strike_price)}|{r}|{exp}"

    @property
    def display_name(self) -> str:
        return f"{self.stock_code} {int(self.strike_price)} {self.right.value.upper()[:1]}E"


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

    @property
    def net_greeks(self) -> Greeks:
        g = Greeks()
        for leg in self.legs:
            if leg.status != LegStatus.ACTIVE:
                continue
            sign = -1.0 if leg.side == OrderSide.SELL else 1.0
            g.delta += leg.greeks.delta * sign * leg.quantity
            g.gamma += leg.greeks.gamma * sign * leg.quantity
            g.theta += leg.greeks.theta * sign * leg.quantity
            g.vega += leg.greeks.vega * sign * leg.quantity
        return g


@dataclass
class LogEntry:
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%H:%M:%S.%f")[:-3]
    )
    level: str = "INFO"
    source: str = ""
    message: str = ""
    data: Optional[Dict[str, Any]] = None
