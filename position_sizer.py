"""
position_sizer.py  — NEW MODULE

Professional Position Sizing Engine
  • Kelly Criterion (full / fractional)
  • Volatility-based sizing (target risk per trade)
  • Margin-aware sizing (SPAN + exposure)
  • Max delta-per-lot constraint
  • Portfolio heat limiter (total risk budget)

Zero breaking-changes: pure add-on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, List

from greeks_engine import BlackScholes, time_to_expiry
from models import Strategy, Leg, LegStatus, OrderSide, OptionRight, Greeks
from app_config import Config, INSTRUMENTS
from utils import LOG


# ────────────────────────────────────────────────────────────────
# Results
# ────────────────────────────────────────────────────────────────

@dataclass
class SizingResult:
    recommended_lots: int
    max_lots_margin: int
    max_lots_kelly: int
    max_lots_risk: int
    margin_required: float
    premium_collected: float
    max_loss_estimate: float
    risk_reward: float
    portfolio_heat_pct: float
    notes: List[str]


# ────────────────────────────────────────────────────────────────
# Position Sizer
# ────────────────────────────────────────────────────────────────

class PositionSizer:
    """
    Calculates optimal lot size for an option strategy
    given account constraints and risk parameters.
    """

    # Defaults (can be overridden per call)
    KELLY_FRACTION = 0.25        # Use 25% Kelly (safer)
    MAX_RISK_PER_TRADE_PCT = 5.0 # Max 5% of capital per trade
    MAX_PORTFOLIO_HEAT_PCT = 20.0# Max 20% of capital in open risk
    MARGIN_BUFFER_PCT = 15.0     # Keep 15% margin buffer

    def __init__(self):
        pass

    # ── Primary API ───────────────────────────────────────────

    def size_for_premium(
        self,
        instrument: str,
        spot: float,
        atm_iv: float,
        expiry: str,
        strategy_type: str = "straddle",
        available_capital: float = 0.0,
        used_capital: float = 0.0,
        win_rate: float = 0.60,
        avg_win_pct: float = 0.50,
        avg_loss_pct: float = 1.00,
    ) -> SizingResult:
        """
        Full sizing recommendation considering multiple constraints.

        Args:
            instrument: e.g. "NIFTY"
            spot: current spot price
            atm_iv: ATM implied vol (as decimal, e.g. 0.15 for 15%)
            expiry: expiry date string
            available_capital: free margin available
            used_capital: already deployed capital
            win_rate: historical win rate (0-1)
            avg_win_pct: avg win as % of premium collected
            avg_loss_pct: avg loss as % of premium collected
        """
        inst = INSTRUMENTS.get(instrument.upper(), INSTRUMENTS.get("NIFTY"))
        lot_size = inst["lot_size"]

        notes = []

        # ── 1. Estimate single-lot premium ───────────────────
        T = time_to_expiry(expiry) if expiry else 7.0 / 365.0
        r = Config.RISK_FREE_RATE

        if strategy_type == "straddle":
            ce_price = BlackScholes.price(spot, spot, T, r, atm_iv, OptionRight.CALL)
            pe_price = BlackScholes.price(spot, spot, T, r, atm_iv, OptionRight.PUT)
            premium_1lot = (ce_price + pe_price) * lot_size
        elif strategy_type == "strangle":
            ce_k = spot * 1.02
            pe_k = spot * 0.98
            ce_price = BlackScholes.price(spot, ce_k, T, r, atm_iv * 0.9, OptionRight.CALL)
            pe_price = BlackScholes.price(spot, pe_k, T, r, atm_iv * 1.1, OptionRight.PUT)
            premium_1lot = (ce_price + pe_price) * lot_size
        else:
            ce_price = BlackScholes.price(spot, spot, T, r, atm_iv, OptionRight.CALL)
            premium_1lot = ce_price * lot_size

        # ── 2. Estimate margin for 1 lot ──────────────────────
        # Rule of thumb SPAN margin for short options on index
        # Real SPAN ≈ 10-15% of notional (conservative estimate)
        notional_1lot = spot * lot_size
        margin_1lot = self._estimate_margin(notional_1lot, atm_iv, strategy_type)

        # ── 3. Max lots by margin ──────────────────────────────
        if available_capital > 0 and margin_1lot > 0:
            usable_capital = available_capital * (1 - self.MARGIN_BUFFER_PCT / 100)
            max_lots_margin = int(usable_capital / margin_1lot)
        else:
            max_lots_margin = 0
            notes.append("Available capital not provided → margin-based limit skipped")

        # ── 4. Max lots by risk (stop-loss based) ─────────────
        sl_pct = float(Config.SL_PERCENTAGE) / 100
        max_loss_1lot = premium_1lot * sl_pct
        total_capital = (available_capital + used_capital) if available_capital > 0 else 500_000
        risk_budget = total_capital * (self.MAX_RISK_PER_TRADE_PCT / 100)

        max_lots_risk = int(risk_budget / max_loss_1lot) if max_loss_1lot > 0 else 5
        notes.append(f"Risk budget: ₹{risk_budget:,.0f} ({self.MAX_RISK_PER_TRADE_PCT}% of ₹{total_capital:,.0f})")

        # ── 5. Kelly Criterion ────────────────────────────────
        max_lots_kelly = self._kelly_lots(
            win_rate=win_rate,
            avg_win=premium_1lot * avg_win_pct,
            avg_loss=premium_1lot * avg_loss_pct,
            total_capital=total_capital,
            premium_1lot=premium_1lot,
        )
        notes.append(f"Kelly ({self.KELLY_FRACTION*100:.0f}%): {max_lots_kelly} lots")

        # ── 6. Portfolio heat ──────────────────────────────────
        heat_budget = total_capital * (self.MAX_PORTFOLIO_HEAT_PCT / 100)
        current_heat = used_capital * sl_pct  # approximate
        remaining_heat = heat_budget - current_heat
        max_lots_heat = int(remaining_heat / max_loss_1lot) if max_loss_1lot > 0 else max_lots_risk
        portfolio_heat_pct = (current_heat / total_capital) * 100 if total_capital > 0 else 0

        if portfolio_heat_pct > 15:
            notes.append(f"⚠️ Portfolio heat at {portfolio_heat_pct:.1f}% — consider reducing size")

        # ── 7. Recommended lots (most conservative) ───────────
        limits = [x for x in [max_lots_margin, max_lots_risk, max_lots_kelly, max_lots_heat] if x > 0]
        recommended = max(1, min(limits)) if limits else 1

        return SizingResult(
            recommended_lots=recommended,
            max_lots_margin=max_lots_margin,
            max_lots_kelly=max_lots_kelly,
            max_lots_risk=max_lots_risk,
            margin_required=margin_1lot * recommended,
            premium_collected=premium_1lot * recommended,
            max_loss_estimate=max_loss_1lot * recommended,
            risk_reward=round(premium_1lot / max_loss_1lot, 2) if max_loss_1lot > 0 else 0,
            portfolio_heat_pct=round(portfolio_heat_pct, 1),
            notes=notes,
        )

    def size_by_delta(
        self,
        target_portfolio_delta: float,
        leg_delta: float,
        lot_size: int,
        current_portfolio_delta: float = 0.0,
    ) -> int:
        """
        Calculate lots needed to bring portfolio delta to target.
        Useful for delta-neutral adjustments.
        """
        if abs(leg_delta) < 1e-6:
            return 0

        delta_gap = target_portfolio_delta - current_portfolio_delta
        lots_needed = abs(delta_gap / (leg_delta * lot_size))
        return max(1, round(lots_needed))

    def equal_notional_split(
        self,
        total_budget: float,
        strategies: List[str],
        margins: List[float],
    ) -> List[int]:
        """
        Equal-weight capital allocation across multiple strategies.
        strategies: list of strategy names
        margins: estimated margin per lot for each strategy
        """
        n = len(strategies)
        if n == 0:
            return []

        per_strategy = total_budget / n
        return [
            max(1, int(per_strategy / m)) if m > 0 else 1
            for m in margins
        ]

    # ── Analytics ─────────────────────────────────────────────

    def compute_roi(
        self,
        premium_collected: float,
        margin_blocked: float,
        days_held: float,
    ) -> dict:
        """
        Return on Margin (ROM) annualized.
        """
        if margin_blocked <= 0 or days_held <= 0:
            return {}

        rom_trade = (premium_collected / margin_blocked) * 100
        rom_annual = rom_trade * (365 / days_held)

        return {
            "rom_trade_pct": round(rom_trade, 2),
            "rom_annualized_pct": round(rom_annual, 1),
            "premium": round(premium_collected, 0),
            "margin": round(margin_blocked, 0),
            "days": round(days_held, 0),
        }

    def expected_value(
        self,
        premium_collected: float,
        max_loss: float,
        win_rate: float,
    ) -> dict:
        """
        Simple EV calculation for a short premium strategy.
        """
        ev = (win_rate * premium_collected) - ((1 - win_rate) * max_loss)
        ev_pct = (ev / premium_collected * 100) if premium_collected > 0 else 0

        return {
            "expected_value": round(ev, 0),
            "ev_pct_of_premium": round(ev_pct, 1),
            "break_even_win_rate": round(max_loss / (premium_collected + max_loss) * 100, 1),
        }

    # ── Private ───────────────────────────────────────────────

    def _estimate_margin(self, notional: float, iv: float,
                         strategy_type: str) -> float:
        """
        Rough SPAN margin estimate.
        Real SPAN = (worst loss × 99% VaR) + flat amount.
        This is a conservative approximation for UI guidance only.
        """
        # SPAN-like: 3 std dev move loss on notional
        daily_vol = iv / math.sqrt(252)
        span_move = daily_vol * 3.0 * notional  # 3-sigma daily move

        if strategy_type in ("straddle", "strangle"):
            # Both legs' SPAN; second leg gets partial credit ~70%
            margin = span_move * 1.7
        elif strategy_type == "iron_condor":
            # Defined risk; max loss = wing width × qty
            margin = span_move * 0.8
        else:
            margin = span_move

        return max(margin, notional * 0.05)  # min 5% of notional

    def _kelly_lots(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        total_capital: float,
        premium_1lot: float,
    ) -> int:
        """
        Kelly: f = (p * b - q) / b  where b = win/loss ratio.
        Apply fractional Kelly for safety.
        """
        if avg_loss <= 0 or total_capital <= 0:
            return 1

        b = avg_win / avg_loss if avg_loss > 0 else 1.0
        p = win_rate
        q = 1 - p

        f = (p * b - q) / b if b > 0 else 0
        f_frac = max(0, f * self.KELLY_FRACTION)

        kelly_capital = total_capital * f_frac
        lots = int(kelly_capital / premium_1lot) if premium_1lot > 0 else 1
        return max(1, lots)
