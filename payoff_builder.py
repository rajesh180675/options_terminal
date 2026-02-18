"""
payoff_builder.py  — Layer 3 NEW MODULE

Interactive Payoff Diagram Builder
  • P&L curve at expiry (intrinsic value based)
  • P&L curve at T=now (Black-Scholes mark-to-market)
  • Multi-strategy combined payoff
  • Break-even points (exact interpolation)
  • Max profit / max loss labels
  • What-if scenarios: spot +5%, -5%, flat
  • IV shock overlay (+5%, -5% IV shift)
  • Support for any strategy type (legs-based)
  • Payoff table export (spot → P&L dict)

Zero breaking-changes: pure add-on. No existing file modified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from greeks_engine import BlackScholes, time_to_expiry
from models import Strategy, Leg, LegStatus, OrderSide, OptionRight


# ────────────────────────────────────────────────────────────────
# Data Classes
# ────────────────────────────────────────────────────────────────

@dataclass
class PayoffCurve:
    spot_range: List[float]         # x-axis: spot prices
    pnl_expiry: List[float]         # y-axis: P&L at expiry
    pnl_now: List[float]            # y-axis: P&L now (BS mark)
    pnl_iv_up: List[float]          # IV +5% scenario
    pnl_iv_down: List[float]        # IV -5% scenario
    breakeven_points: List[float]   # exact zero-cross spots
    max_profit: float
    max_loss: float
    max_profit_spot: float
    max_loss_spot: float
    current_pnl: float             # P&L at current spot
    current_spot: float


@dataclass
class ScenarioResult:
    label: str
    spot: float
    spot_change_pct: float
    pnl_expiry: float
    pnl_now: float
    pnl_change_from_current: float


@dataclass
class PayoffSummary:
    strategy_id: str
    strategy_type: str
    instrument: str
    curve: PayoffCurve
    scenarios: List[ScenarioResult]
    net_premium: float              # net credit (+) or debit (-)
    max_profit: float
    max_loss: float
    risk_reward: float
    probability_of_profit_pct: float  # simple log-normal approximation
    expected_move_1sd: float          # 1σ expected move at expiry
    expected_move_range: Tuple[float, float]   # (lower, upper) 1σ bands


# ────────────────────────────────────────────────────────────────
# Payoff Builder
# ────────────────────────────────────────────────────────────────

class PayoffBuilder:
    """
    Builds payoff diagrams and scenario analysis for option strategies.

    Usage:
        builder = PayoffBuilder()
        summary = builder.build(strategy, spot, chain=None)
        multi = builder.build_multi(strategies, spot_map)
    """

    SPOT_RANGE_PCT = 15.0           # ±15% from spot
    SPOT_STEPS = 100                # points in payoff curve
    IV_SHOCK_PCT = 5.0              # IV shift for scenarios

    def __init__(self, risk_free_rate: float = 0.07):
        self.r = risk_free_rate

    # ── Single Strategy ───────────────────────────────────────

    def build(
        self,
        strategy: Strategy,
        spot: float,
        chain: Optional[List[dict]] = None,
        spot_range_pct: float = 0.0,
    ) -> PayoffSummary:
        """
        Build complete payoff analysis for a single strategy.

        Args:
            strategy:       Strategy object with legs
            spot:           current spot price
            chain:          option chain (for live IVs)
            spot_range_pct: override default spot range
        """
        if spot <= 0:
            return self._empty_summary(strategy)

        active_legs = [l for l in strategy.legs if l.status in (LegStatus.ACTIVE, LegStatus.ENTERING)]
        if not active_legs:
            # Use all legs if no active (for analysis)
            active_legs = strategy.legs

        if not active_legs:
            return self._empty_summary(strategy)

        rng = spot_range_pct or self.SPOT_RANGE_PCT
        spot_low = spot * (1 - rng / 100)
        spot_high = spot * (1 + rng / 100)
        spot_grid = np.linspace(spot_low, spot_high, self.SPOT_STEPS)

        # Compute curves
        pnl_expiry = self._compute_expiry_pnl(active_legs, spot_grid)
        pnl_now = self._compute_bs_pnl(active_legs, spot_grid, iv_shift=0.0)
        pnl_iv_up = self._compute_bs_pnl(active_legs, spot_grid, iv_shift=+self.IV_SHOCK_PCT / 100)
        pnl_iv_down = self._compute_bs_pnl(active_legs, spot_grid, iv_shift=-self.IV_SHOCK_PCT / 100)

        # Breakevens from expiry curve
        breakevens = self._find_breakevens(spot_grid.tolist(), pnl_expiry)

        # Extremes
        max_pnl = float(max(pnl_expiry))
        min_pnl = float(min(pnl_expiry))
        max_spot = float(spot_grid[int(np.argmax(pnl_expiry))])
        min_spot = float(spot_grid[int(np.argmin(pnl_expiry))])

        # Current P&L from BS
        current_pnl_idx = int(len(spot_grid) // 2)
        current_pnl = float(pnl_now[current_pnl_idx])

        curve = PayoffCurve(
            spot_range=spot_grid.tolist(),
            pnl_expiry=pnl_expiry,
            pnl_now=pnl_now,
            pnl_iv_up=pnl_iv_up,
            pnl_iv_down=pnl_iv_down,
            breakeven_points=breakevens,
            max_profit=max_pnl,
            max_loss=min_pnl,
            max_profit_spot=max_spot,
            max_loss_spot=min_spot,
            current_pnl=current_pnl,
            current_spot=spot,
        )

        # Scenarios
        scenarios = self._build_scenarios(active_legs, spot)

        # Summary stats
        net_premium = self._compute_net_premium(active_legs)
        risk_reward = abs(max_pnl / min_pnl) if min_pnl != 0 else 999.0
        pop = self._estimate_pop(active_legs, spot, breakevens)
        expected_move, em_range = self._expected_move(active_legs, spot)

        return PayoffSummary(
            strategy_id=strategy.strategy_id,
            strategy_type=strategy.strategy_type.value,
            instrument=strategy.stock_code,
            curve=curve,
            scenarios=scenarios,
            net_premium=round(net_premium, 2),
            max_profit=round(max_pnl, 2),
            max_loss=round(min_pnl, 2),
            risk_reward=round(risk_reward, 2),
            probability_of_profit_pct=round(pop, 1),
            expected_move_1sd=round(expected_move, 2),
            expected_move_range=(round(em_range[0], 2), round(em_range[1], 2)),
        )

    def build_multi(
        self,
        strategies: List[Strategy],
        spot_map: Dict[str, float],
    ) -> Dict[str, PayoffSummary]:
        """Build payoff for each strategy individually."""
        result = {}
        for s in strategies:
            from models import StrategyStatus
            if s.status not in (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT):
                continue
            spot = spot_map.get(s.stock_code, 0.0)
            if spot <= 0:
                continue
            result[s.strategy_id] = self.build(s, spot)
        return result

    def build_combined_curve(
        self,
        strategies: List[Strategy],
        spot: float,
    ) -> Optional[PayoffCurve]:
        """Combine all strategies into a single aggregate payoff curve."""
        all_legs = []
        for s in strategies:
            from models import StrategyStatus
            if s.status not in (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT):
                continue
            all_legs.extend(l for l in s.legs if l.status == LegStatus.ACTIVE)

        if not all_legs or spot <= 0:
            return None

        spot_low = spot * (1 - self.SPOT_RANGE_PCT / 100)
        spot_high = spot * (1 + self.SPOT_RANGE_PCT / 100)
        spot_grid = np.linspace(spot_low, spot_high, self.SPOT_STEPS)

        pnl_expiry = self._compute_expiry_pnl(all_legs, spot_grid)
        pnl_now = self._compute_bs_pnl(all_legs, spot_grid)
        breakevens = self._find_breakevens(spot_grid.tolist(), pnl_expiry)

        return PayoffCurve(
            spot_range=spot_grid.tolist(),
            pnl_expiry=pnl_expiry,
            pnl_now=pnl_now,
            pnl_iv_up=pnl_now,
            pnl_iv_down=pnl_now,
            breakeven_points=breakevens,
            max_profit=float(max(pnl_expiry)),
            max_loss=float(min(pnl_expiry)),
            max_profit_spot=float(spot_grid[int(np.argmax(pnl_expiry))]),
            max_loss_spot=float(spot_grid[int(np.argmin(pnl_expiry))]),
            current_pnl=float(pnl_now[len(pnl_now) // 2]),
            current_spot=spot,
        )

    # ── P&L Computation ───────────────────────────────────────

    def _compute_expiry_pnl(self, legs: List[Leg], spot_grid: np.ndarray) -> List[float]:
        """P&L at expiry = intrinsic value."""
        pnl = np.zeros(len(spot_grid))
        for l in legs:
            if l.entry_price <= 0 and l.current_price <= 0:
                continue
            ref_price = l.entry_price if l.entry_price > 0 else l.current_price
            sign = -1.0 if l.side == OrderSide.SELL else 1.0
            if l.right == OptionRight.CALL:
                intrinsic = np.maximum(spot_grid - l.strike_price, 0)
            else:
                intrinsic = np.maximum(l.strike_price - spot_grid, 0)
            # For sells: collected premium - current cost
            # At expiry: collected premium - intrinsic
            if l.side == OrderSide.SELL:
                pnl += (ref_price - intrinsic) * l.quantity
            else:
                pnl += (intrinsic - ref_price) * l.quantity
        return pnl.tolist()

    def _compute_bs_pnl(
        self, legs: List[Leg], spot_grid: np.ndarray, iv_shift: float = 0.0
    ) -> List[float]:
        """P&L now via Black-Scholes repricing."""
        pnl = np.zeros(len(spot_grid))
        for l in legs:
            if l.entry_price <= 0 and l.current_price <= 0:
                continue
            ref_price = l.entry_price if l.entry_price > 0 else l.current_price
            T = time_to_expiry(l.expiry_date)
            if T <= 0:
                # Use intrinsic if expired
                if l.right == OptionRight.CALL:
                    intrinsic = np.maximum(spot_grid - l.strike_price, 0)
                else:
                    intrinsic = np.maximum(l.strike_price - spot_grid, 0)
                if l.side == OrderSide.SELL:
                    pnl += (ref_price - intrinsic) * l.quantity
                else:
                    pnl += (intrinsic - ref_price) * l.quantity
                continue

            # Get base IV
            try:
                iv_base = l.greeks.iv if l.greeks.iv > 0.01 else 0.18
            except Exception:
                iv_base = 0.18
            iv = max(0.01, iv_base + iv_shift)

            # Reprice across spot grid
            for i, s in enumerate(spot_grid):
                try:
                    bs_price = BlackScholes.price(s, l.strike_price, T, self.r, iv, l.right)
                except Exception:
                    bs_price = 0.0
                if l.side == OrderSide.SELL:
                    pnl[i] += (ref_price - bs_price) * l.quantity
                else:
                    pnl[i] += (bs_price - ref_price) * l.quantity

        return pnl.tolist()

    # ── Helpers ───────────────────────────────────────────────

    def _find_breakevens(self, spots: List[float], pnls: List[float]) -> List[float]:
        """Find zero-crossing points via linear interpolation."""
        breakevens = []
        for i in range(len(pnls) - 1):
            if pnls[i] * pnls[i + 1] < 0:  # sign change
                # Linear interpolation
                t = -pnls[i] / (pnls[i + 1] - pnls[i])
                be = spots[i] + t * (spots[i + 1] - spots[i])
                breakevens.append(round(be, 2))
        return breakevens

    def _build_scenarios(self, legs: List[Leg], spot: float) -> List[ScenarioResult]:
        scenarios = []
        spot_changes = [
            ("Flat", 0.0),
            ("+1%", 1.0), ("+3%", 3.0), ("+5%", 5.0), ("+10%", 10.0),
            ("-1%", -1.0), ("-3%", -3.0), ("-5%", -5.0), ("-10%", -10.0),
        ]
        for label, chg_pct in spot_changes:
            s_new = spot * (1 + chg_pct / 100)
            pnl_exp = self._compute_expiry_pnl(legs, np.array([s_new]))[0]
            pnl_now = self._compute_bs_pnl(legs, np.array([s_new]))[0]
            pnl_curr = self._compute_bs_pnl(legs, np.array([spot]))[0]
            scenarios.append(ScenarioResult(
                label=label, spot=round(s_new, 2),
                spot_change_pct=chg_pct,
                pnl_expiry=round(pnl_exp, 2),
                pnl_now=round(pnl_now, 2),
                pnl_change_from_current=round(pnl_now - pnl_curr, 2),
            ))
        return scenarios

    def _compute_net_premium(self, legs: List[Leg]) -> float:
        net = 0.0
        for l in legs:
            ref = l.entry_price if l.entry_price > 0 else l.current_price
            if l.side == OrderSide.SELL:
                net += ref * l.quantity
            else:
                net -= ref * l.quantity
        return net

    def _estimate_pop(self, legs: List[Leg], spot: float, breakevens: List[float]) -> float:
        """Simple log-normal PoP estimate."""
        if not breakevens:
            return 50.0

        # Get ATM IV for expected move
        atm_iv = 0.18
        for l in legs:
            if abs(l.strike_price - spot) / spot < 0.05:
                try:
                    iv = l.greeks.iv if l.greeks.iv > 0.01 else atm_iv
                    atm_iv = iv
                    break
                except Exception:
                    pass

        T = time_to_expiry(legs[0].expiry_date) if legs else 0.01
        if T <= 0:
            T = 0.01
        sigma = atm_iv * math.sqrt(T)

        # PoP = P(spot within profitable zone at expiry)
        # For short straddle: profitable between the two BEs
        if len(breakevens) == 2:
            be_low, be_high = sorted(breakevens)
            from scipy.stats import norm as _norm
            z_low = (math.log(be_low / spot) + 0.5 * atm_iv**2 * T) / sigma
            z_high = (math.log(be_high / spot) + 0.5 * atm_iv**2 * T) / sigma
            pop = (_norm.cdf(z_high) - _norm.cdf(z_low)) * 100
        elif len(breakevens) == 1:
            # Call/Put spread etc.
            be = breakevens[0]
            from scipy.stats import norm as _norm
            z = (math.log(be / spot) + 0.5 * atm_iv**2 * T) / sigma
            pop = _norm.cdf(z) * 100
            # If short call, PoP = P(below BE)
        else:
            pop = 50.0

        return max(0.0, min(100.0, pop))

    def _expected_move(self, legs: List[Leg], spot: float) -> Tuple[float, Tuple[float, float]]:
        """1σ expected move at expiry."""
        atm_iv = 0.18
        for l in legs:
            try:
                if l.greeks.iv > 0.01:
                    atm_iv = l.greeks.iv
                    break
            except Exception:
                pass

        T = time_to_expiry(legs[0].expiry_date) if legs else 0.01
        if T <= 0:
            T = 0.01
        em = spot * atm_iv * math.sqrt(T)
        return em, (spot - em, spot + em)

    def _empty_summary(self, strategy: Strategy) -> PayoffSummary:
        empty_curve = PayoffCurve(
            spot_range=[], pnl_expiry=[], pnl_now=[],
            pnl_iv_up=[], pnl_iv_down=[], breakeven_points=[],
            max_profit=0, max_loss=0, max_profit_spot=0,
            max_loss_spot=0, current_pnl=0, current_spot=0,
        )
        return PayoffSummary(
            strategy_id=strategy.strategy_id,
            strategy_type=strategy.strategy_type.value,
            instrument=strategy.stock_code,
            curve=empty_curve, scenarios=[],
            net_premium=0, max_profit=0, max_loss=0, risk_reward=0,
            probability_of_profit_pct=0, expected_move_1sd=0,
            expected_move_range=(0, 0),
        )
