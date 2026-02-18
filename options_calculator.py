"""
options_calculator.py  — NEW MODULE

Standalone Options Calculator (no broker connection needed):
  • Single option pricer with full Greeks
  • Break-even finder
  • Target P&L reverse-solver (find IV needed to reach target price)
  • Spread builder (enter custom legs, see combined payoff)
  • Roll analysis (cost to roll current position to different strike/expiry)
  • Greeks ladder (table of Greeks at each spot level)
  • Time value decay table (option value across DTE)

Zero breaking-changes: pure add-on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

from greeks_engine import BlackScholes, time_to_expiry
from models import OptionRight, Greeks
from app_config import Config


# ────────────────────────────────────────────────────────────────
# Data Classes
# ────────────────────────────────────────────────────────────────

@dataclass
class OptionPriceResult:
    spot: float
    strike: float
    right: str
    expiry: str
    dte: float
    iv: float
    price: float
    intrinsic: float
    time_value: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float          # approx: K * T * e^(-rT) * N(d2)
    charm: float        # dDelta/dTime (approx)
    breakeven_at_expiry: float
    breakeven_spot_now: float   # spot where option = current market price

    def to_dict(self) -> dict:
        return {
            "Spot": self.spot,
            "Strike": int(self.strike),
            "Type": self.right,
            "DTE": self.dte,
            "IV%": round(self.iv * 100, 2),
            "Price": round(self.price, 2),
            "Intrinsic": round(self.intrinsic, 2),
            "Time Value": round(self.time_value, 2),
            "Delta": round(self.delta, 4),
            "Gamma": round(self.gamma, 6),
            "Theta/day": round(self.theta, 3),
            "Vega": round(self.vega, 3),
            "BE (expiry)": round(self.breakeven_at_expiry, 1),
        }


@dataclass
class GreeksLadderRow:
    spot: float
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    iv: float


@dataclass
class DecayTableRow:
    dte: float
    price: float
    delta: float
    theta: float
    time_value: float
    pct_remaining: float    # time value as % of original


@dataclass
class CustomLeg:
    strike: float
    right: str              # "CALL" or "PUT"
    side: str               # "BUY" or "SELL"
    quantity: int
    entry_price: float
    lots: int = 1


@dataclass
class SpreadAnalysis:
    legs: List[CustomLeg]
    net_premium: float      # positive = credit, negative = debit
    max_profit: float
    max_loss: float
    breakeven_upper: float
    breakeven_lower: float
    pop_estimate: float
    payoff_curve: List[dict]    # [{spot, pnl}, ...]


# ────────────────────────────────────────────────────────────────
# Options Calculator
# ────────────────────────────────────────────────────────────────

class OptionsCalculator:
    """
    Purely mathematical calculator — no broker, no live data needed.
    Accepts spot / strike / IV / DTE as inputs.
    """

    def __init__(self, risk_free_rate: float = None):
        self.r = risk_free_rate or Config.RISK_FREE_RATE

    # ── Single Option Pricer ──────────────────────────────────

    def price_option(
        self,
        spot: float,
        strike: float,
        iv: float,            # as decimal (e.g. 0.18 for 18%)
        dte_days: float,
        right: str = "CALL",  # "CALL" or "PUT"
        expiry: str = "",
    ) -> OptionPriceResult:
        """Full option pricing with all Greeks."""
        T = dte_days / 365.0
        right_enum = OptionRight.CALL if right.upper() == "CALL" else OptionRight.PUT

        price = BlackScholes.price(spot, strike, T, self.r, iv, right_enum)
        intrinsic = max(spot - strike, 0) if right_enum == OptionRight.CALL else max(strike - spot, 0)
        time_val = max(price - intrinsic, 0)
        g = BlackScholes.greeks(spot, strike, T, self.r, iv, right_enum)

        # Rho (approximation)
        from scipy.stats import norm
        sqT = math.sqrt(T) if T > 0 else 0
        if T > 0 and iv > 0:
            d2 = (math.log(spot / strike) + (self.r - 0.5 * iv**2) * T) / (iv * sqT)
            if right_enum == OptionRight.CALL:
                rho = strike * T * math.exp(-self.r * T) * norm.cdf(d2) / 100
            else:
                rho = -strike * T * math.exp(-self.r * T) * norm.cdf(-d2) / 100
        else:
            rho = 0.0

        # Charm (dDelta/dTime, approximate via finite difference)
        dt = 1 / 365
        if T > dt:
            g_fwd = BlackScholes.greeks(spot, strike, T - dt, self.r, iv, right_enum)
            charm = g_fwd.delta - g.delta
        else:
            charm = 0.0

        # Breakeven at expiry
        be = (strike + price) if right_enum == OptionRight.CALL else (strike - price)

        return OptionPriceResult(
            spot=spot,
            strike=strike,
            right=right.upper(),
            expiry=expiry,
            dte=round(dte_days, 2),
            iv=iv,
            price=round(price, 2),
            intrinsic=round(intrinsic, 2),
            time_value=round(time_val, 2),
            delta=round(g.delta, 4),
            gamma=round(g.gamma, 6),
            theta=round(g.theta, 4),
            vega=round(g.vega, 4),
            rho=round(rho, 4),
            charm=round(charm, 6),
            breakeven_at_expiry=round(be, 2),
            breakeven_spot_now=spot,  # trivially = spot (would need IV solve)
        )

    # ── Greeks Ladder ─────────────────────────────────────────

    def greeks_ladder(
        self,
        spot: float,
        strike: float,
        iv: float,
        dte_days: float,
        right: str = "CALL",
        spot_range_pct: float = 8.0,
        num_steps: int = 15,
    ) -> List[GreeksLadderRow]:
        """Table of Greeks as spot moves from -range% to +range%."""
        T = dte_days / 365.0
        right_enum = OptionRight.CALL if right.upper() == "CALL" else OptionRight.PUT
        spots = np.linspace(spot * (1 - spot_range_pct / 100), spot * (1 + spot_range_pct / 100), num_steps)

        rows = []
        for S in spots:
            p = BlackScholes.price(S, strike, T, self.r, iv, right_enum)
            g = BlackScholes.greeks(S, strike, T, self.r, iv, right_enum)
            rows.append(GreeksLadderRow(
                spot=round(S, 1),
                price=round(p, 2),
                delta=round(g.delta, 4),
                gamma=round(g.gamma, 6),
                theta=round(g.theta, 3),
                vega=round(g.vega, 3),
                iv=round(iv * 100, 2),
            ))

        return rows

    # ── Decay Table ───────────────────────────────────────────

    def decay_table(
        self,
        spot: float,
        strike: float,
        iv: float,
        max_dte: float = 30,
        right: str = "CALL",
    ) -> List[DecayTableRow]:
        """Show how option price and theta decay as DTE decreases."""
        right_enum = OptionRight.CALL if right.upper() == "CALL" else OptionRight.PUT
        dte_values = list(range(int(max_dte), 0, -1)) + [0.5, 0.1]

        original_price = None
        rows = []

        for dte in dte_values:
            T = max(dte / 365.0, 1e-6)
            p = BlackScholes.price(spot, strike, T, self.r, iv, right_enum)
            g = BlackScholes.greeks(spot, strike, T, self.r, iv, right_enum)
            intrinsic = max(spot - strike, 0) if right_enum == OptionRight.CALL else max(strike - spot, 0)
            tv = max(p - intrinsic, 0)

            if original_price is None:
                original_price = p

            pct = (tv / original_price * 100) if original_price > 0 else 0

            rows.append(DecayTableRow(
                dte=dte,
                price=round(p, 2),
                delta=round(g.delta, 4),
                theta=round(g.theta, 3),
                time_value=round(tv, 2),
                pct_remaining=round(pct, 1),
            ))

        return rows

    # ── Spread Builder ────────────────────────────────────────

    def analyze_spread(
        self,
        legs: List[CustomLeg],
        spot: float,
        dte_days: float = 7,
        iv: float = 0.15,
    ) -> SpreadAnalysis:
        """Analyze a custom multi-leg spread."""
        T = max(dte_days / 365.0, 1e-6)

        # Compute entry prices if not provided
        for leg in legs:
            if leg.entry_price <= 0:
                right_enum = OptionRight.CALL if leg.right.upper() == "CALL" else OptionRight.PUT
                leg.entry_price = BlackScholes.price(spot, leg.strike, T, self.r, iv, right_enum)

        # Net premium
        net_premium = 0.0
        for leg in legs:
            qty = leg.quantity * leg.lots
            if leg.side.upper() == "SELL":
                net_premium += leg.entry_price * qty
            else:
                net_premium -= leg.entry_price * qty

        # Payoff at expiry
        all_strikes = [l.strike for l in legs]
        center = sum(all_strikes) / len(all_strikes)
        spread = max(center * 0.08, 500)
        pts = np.linspace(center - spread, center + spread, 100)

        payoff_curve = []
        pnls = []
        for S in pts:
            total = 0.0
            for leg in legs:
                qty = leg.quantity * leg.lots
                if leg.right.upper() == "CALL":
                    intrinsic = max(0, S - leg.strike)
                else:
                    intrinsic = max(0, leg.strike - S)
                if leg.side.upper() == "SELL":
                    leg_pnl = (leg.entry_price - intrinsic) * qty
                else:
                    leg_pnl = (intrinsic - leg.entry_price) * qty
                total += leg_pnl
            payoff_curve.append({"spot": round(S, 1), "pnl": round(total, 2)})
            pnls.append(total)

        max_profit = max(pnls) if pnls else 0
        max_loss = min(pnls) if pnls else 0

        # Breakevens (find sign changes)
        upper_be = lower_be = 0.0
        for i in range(1, len(pnls)):
            if pnls[i - 1] < 0 <= pnls[i]:
                lower_be = pts[i - 1] + (pts[i] - pts[i - 1]) * abs(pnls[i - 1]) / (abs(pnls[i - 1]) + abs(pnls[i]))
            if pnls[i - 1] >= 0 > pnls[i]:
                upper_be = pts[i - 1] + (pts[i] - pts[i - 1]) * pnls[i - 1] / (pnls[i - 1] - pnls[i])

        # Simple POP: % of spot range in profit
        profitable = sum(1 for p in pnls if p > 0)
        pop = (profitable / len(pnls) * 100) if pnls else 0

        return SpreadAnalysis(
            legs=legs,
            net_premium=round(net_premium, 2),
            max_profit=round(max_profit, 2),
            max_loss=round(max_loss, 2),
            breakeven_upper=round(upper_be, 1),
            breakeven_lower=round(lower_be, 1),
            pop_estimate=round(pop, 1),
            payoff_curve=payoff_curve,
        )

    # ── Implied Vol Finder ────────────────────────────────────

    def find_iv_for_price(
        self,
        target_price: float,
        spot: float,
        strike: float,
        dte_days: float,
        right: str = "CALL",
    ) -> Optional[float]:
        """Find IV that produces a target option price."""
        right_enum = OptionRight.CALL if right.upper() == "CALL" else OptionRight.PUT
        T = max(dte_days / 365.0, 1e-6)
        try:
            return BlackScholes.implied_vol(target_price, spot, strike, T, self.r, right_enum)
        except Exception:
            return None

    # ── Roll Analysis ─────────────────────────────────────────

    def roll_analysis(
        self,
        current_price: float,
        target_price: float,
        current_dte: float,
        target_dte: float,
        spot: float,
        current_strike: float,
        target_strike: float,
        right: str = "CALL",
        iv: float = 0.15,
    ) -> dict:
        """
        Analyze the cost to roll a position.
        Buy back current, sell target.
        """
        right_enum = OptionRight.CALL if right.upper() == "CALL" else OptionRight.PUT

        T_curr = max(current_dte / 365.0, 1e-6)
        T_tgt = max(target_dte / 365.0, 1e-6)

        current_iv = BlackScholes.implied_vol(current_price, spot, current_strike, T_curr, self.r, right_enum)
        target_iv_same = current_iv  # assume same IV for target

        target_price_model = BlackScholes.price(spot, target_strike, T_tgt, self.r, target_iv_same, right_enum)

        # Roll debit/credit = (cost to buy back current) - (credit from selling target)
        roll_cost = current_price - target_price_model
        roll_type = "Roll Credit" if roll_cost < 0 else "Roll Debit"

        return {
            "current_strike": int(current_strike),
            "current_dte": current_dte,
            "current_price": round(current_price, 2),
            "current_iv": round(current_iv * 100, 1),
            "target_strike": int(target_strike),
            "target_dte": target_dte,
            "target_price": round(target_price_model, 2),
            "roll_cost": round(abs(roll_cost), 2),
            "roll_type": roll_type,
            "roll_worthwhile": roll_cost < 0,  # credit roll is always worthwhile
            "new_breakeven": round(
                (target_strike + target_price_model) if right_enum == OptionRight.CALL
                else (target_strike - target_price_model), 1
            ),
        }

    # ── Utility ───────────────────────────────────────────────

    @staticmethod
    def greeks_ladder_to_dicts(rows: List[GreeksLadderRow]) -> List[dict]:
        return [{
            "Spot": r.spot,
            "Price": r.price,
            "Delta": r.delta,
            "Gamma": r.gamma,
            "Theta": r.theta,
            "Vega": r.vega,
        } for r in rows]

    @staticmethod
    def decay_table_to_dicts(rows: List[DecayTableRow]) -> List[dict]:
        return [{
            "DTE": r.dte,
            "Price": r.price,
            "Time Value": r.time_value,
            "Theta": r.theta,
            "Delta": r.delta,
            "TV Remaining%": r.pct_remaining,
        } for r in rows]
