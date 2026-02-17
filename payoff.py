# ═══════════════════════════════════════════════════════════════
# FILE: payoff.py  (NEW — the one chart every options platform must have)
# ═══════════════════════════════════════════════════════════════
"""
Payoff diagram computation for any combination of option legs.
Generates:
  • Expiry payoff (intrinsic value based)
  • Current payoff (using Black-Scholes for remaining time value)
  • Breakeven points
  • Max profit / max loss
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
from models import Leg, Strategy, OptionRight, OrderSide, LegStatus
from greeks_engine import BlackScholes, time_to_expiry
from app_config import Config


@dataclass
class PayoffResult:
    data: List[Dict]           # [{spot, expiry_pnl, current_pnl}, ...]
    breakevens: List[float]
    max_profit: float
    max_loss: float
    max_profit_spot: float
    max_loss_spot: float


def compute_payoff(strategy: Strategy, spot: float, num_points: int = 100) -> PayoffResult:
    """
    Compute payoff for a strategy across a range of spot prices.
    Works for any number/combination of legs.
    """
    legs = [l for l in strategy.legs
            if l.status in (LegStatus.ACTIVE, LegStatus.ENTERING) and l.entry_price > 0]

    if not legs or spot <= 0:
        return PayoffResult([], [], 0, 0, 0, 0)

    # Determine spot range: ±15% from current spot
    lo = spot * 0.85
    hi = spot * 1.15
    step = (hi - lo) / num_points

    T = time_to_expiry(legs[0].expiry_date) if legs else 0.001
    r = Config.RISK_FREE_RATE

    data = []
    max_profit, max_loss = float("-inf"), float("inf")
    max_profit_spot, max_loss_spot = spot, spot
    prev_pnl = None
    breakevens = []

    s = lo
    while s <= hi:
        expiry_pnl = 0.0
        current_pnl = 0.0

        for leg in legs:
            K = leg.strike_price
            qty = leg.quantity
            entry = leg.entry_price
            right = leg.right
            sign = -1.0 if leg.side == OrderSide.SELL else 1.0

            # Expiry payoff (intrinsic only)
            if right == OptionRight.CALL:
                intrinsic = max(s - K, 0)
            else:
                intrinsic = max(K - s, 0)
            leg_pnl_expiry = (entry - intrinsic) * qty if leg.side == OrderSide.SELL \
                else (intrinsic - entry) * qty
            expiry_pnl += leg_pnl_expiry

            # Current payoff (with time value via BS)
            iv = leg.greeks.iv if leg.greeks.iv > 0 else 0.15
            if T > 0.0001:
                bs_price = BlackScholes.price(s, K, T, r, iv, right)
            else:
                bs_price = intrinsic
            leg_pnl_current = (entry - bs_price) * qty if leg.side == OrderSide.SELL \
                else (bs_price - entry) * qty
            current_pnl += leg_pnl_current

        data.append({
            "spot": round(s, 1),
            "expiry_pnl": round(expiry_pnl, 2),
            "current_pnl": round(current_pnl, 2),
        })

        if expiry_pnl > max_profit:
            max_profit = expiry_pnl
            max_profit_spot = s
        if expiry_pnl < max_loss:
            max_loss = expiry_pnl
            max_loss_spot = s

        # Detect breakeven crossings
        if prev_pnl is not None and prev_pnl * expiry_pnl < 0:
            breakevens.append(round(s - step / 2, 1))
        prev_pnl = expiry_pnl
        s += step

    return PayoffResult(
        data=data, breakevens=breakevens,
        max_profit=round(max_profit, 2), max_loss=round(max_loss, 2),
        max_profit_spot=round(max_profit_spot, 1),
        max_loss_spot=round(max_loss_spot, 1),
    )
