"""
portfolio_dashboard.py  — NEW MODULE

Rich portfolio-level dashboard helpers:
  • Strategy heat cards (colour-coded P&L + Greeks + health score)
  • Net portfolio Greek bar gauges
  • Margin utilisation tracker
  • Intraday MTM sparkline
  • Breakeven / range visualisation
  • Position concentration alerts
  • Theta-to-margin efficiency ratio

Zero breaking-changes: pure add-on.
All rendering is done via helper functions that return data structures;
the Streamlit calls happen in main.py tabs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

from models import Strategy, StrategyStatus, LegStatus, OrderSide, OptionRight, Greeks
from greeks_engine import BlackScholes, time_to_expiry
from app_config import Config

# ────────────────────────────────────────────────────────────────
# Data Classes
# ────────────────────────────────────────────────────────────────

@dataclass
class StrategyCard:
    strategy_id: str
    strategy_type: str
    stock_code: str
    status: str
    pnl: float
    pnl_pct: float             # pnl / total_premium * 100
    total_premium: float       # net credit received
    max_loss: float            # theoretical max loss (wings define it for defined-risk)
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float
    num_legs: int
    active_legs: int
    dte: float                 # days to expiry of first leg
    health_score: float        # 0-100 (higher = healthier)
    health_label: str          # "Excellent" / "Good" / "Caution" / "Warning" / "Critical"
    breakeven_upper: float
    breakeven_lower: float
    margin_utilised: float     # estimated ₹ margin used
    theta_margin_ratio: float  # daily theta / margin (efficiency)
    notes: List[str] = field(default_factory=list)

    @property
    def is_profitable(self) -> bool:
        return self.pnl > 0

    @property
    def health_color(self) -> str:
        return {
            "Excellent": "#00c853",
            "Good": "#64dd17",
            "Caution": "#ffab40",
            "Warning": "#ff6d00",
            "Critical": "#d50000",
        }.get(self.health_label, "#90a4ae")


@dataclass
class PortfolioSummary:
    total_pnl: float
    total_premium: float
    total_theta: float          # daily
    total_margin: float         # estimated
    theta_margin_pct: float     # theta efficiency
    net_delta: float
    net_gamma: float
    net_vega: float
    active_strategies: int
    active_legs: int
    strategies_in_profit: int
    strategies_in_loss: int
    biggest_winner: str         # strategy_id
    biggest_loser: str
    max_single_loss: float      # worst-case single-strategy loss
    concentration_risk: float   # 0-1 (how concentrated in one instrument)
    portfolio_health: float     # 0-100
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


# ────────────────────────────────────────────────────────────────
# Dashboard Engine
# ────────────────────────────────────────────────────────────────

class PortfolioDashboard:

    # Rough margin estimation per lot for index options sold (₹ per lot)
    MARGIN_PER_LOT_ESTIMATES = {
        "NIFTY":     90_000,
        "CNXBAN":   120_000,
        "BSESEN":   80_000,
    }

    def build_strategy_card(self, strategy: Strategy, spot: float) -> Optional[StrategyCard]:
        """Build a rich StrategyCard from a Strategy object."""
        if not strategy.legs:
            return None

        active = [l for l in strategy.legs if l.status == LegStatus.ACTIVE]
        all_entered = [l for l in strategy.legs if l.entry_price > 0]

        # Financials
        pnl = strategy.compute_total_pnl()
        sell_legs = [l for l in all_entered if l.side == OrderSide.SELL]
        buy_legs = [l for l in all_entered if l.side == OrderSide.BUY]
        total_premium = sum(l.entry_price * l.quantity for l in sell_legs) - \
                        sum(l.entry_price * l.quantity for l in buy_legs)
        pnl_pct = (pnl / total_premium * 100) if total_premium > 0 else 0

        # Net Greeks (from active legs)
        ng = strategy.net_greeks
        lot_size = Config.lot_size(strategy.stock_code) if hasattr(Config, "lot_size") else 50

        # DTE
        dte = 0.0
        if active:
            T = time_to_expiry(active[0].expiry_date)
            dte = round(T * 365, 1)

        # Breakevens (simple: for short straddle/strangle)
        be_upper, be_lower = self._compute_breakevens(strategy, total_premium)

        # Margin estimate
        margin = self._estimate_margin(strategy)

        # Theta/margin efficiency
        theta_daily = abs(ng.theta)  # net theta as ₹/day from greeks (need to scale)
        theta_margin_ratio = (theta_daily / margin * 100) if margin > 0 else 0

        # Max loss
        max_loss = self._compute_max_loss(strategy, total_premium)

        # Health score
        health_score, health_label, notes = self._compute_health(
            strategy, pnl, total_premium, pnl_pct, dte, ng, spot, be_upper, be_lower
        )

        return StrategyCard(
            strategy_id=strategy.strategy_id,
            strategy_type=strategy.strategy_type.value,
            stock_code=strategy.stock_code,
            status=strategy.status.value,
            pnl=pnl,
            pnl_pct=round(pnl_pct, 1),
            total_premium=round(total_premium, 2),
            max_loss=round(max_loss, 2),
            net_delta=round(ng.delta, 2),
            net_gamma=round(ng.gamma, 4),
            net_theta=round(ng.theta, 2),
            net_vega=round(ng.vega, 2),
            num_legs=len(strategy.legs),
            active_legs=len(active),
            dte=dte,
            health_score=health_score,
            health_label=health_label,
            breakeven_upper=be_upper,
            breakeven_lower=be_lower,
            margin_utilised=margin,
            theta_margin_ratio=round(theta_margin_ratio, 3),
            notes=notes,
        )

    def build_portfolio_summary(self, strategies: List[Strategy], spot_map: Dict[str, float]) -> PortfolioSummary:
        """Aggregate portfolio-level summary."""
        active_strats = [s for s in strategies if s.status in
                         (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT)]

        total_pnl = sum(s.compute_total_pnl() for s in active_strats)
        total_premium = 0.0
        total_theta = 0.0
        total_margin = 0.0
        net_delta = net_gamma = net_vega = 0.0
        active_legs = 0
        in_profit = in_loss = 0
        best_pnl = best_id = ""
        worst_pnl = worst_id = ""
        best_val = float("-inf")
        worst_val = float("inf")

        instrument_exposure: Dict[str, float] = {}

        for s in active_strats:
            ng = s.net_greeks
            net_delta += ng.delta
            net_gamma += ng.gamma
            net_vega += ng.vega

            pnl = s.compute_total_pnl()
            if pnl >= 0:
                in_profit += 1
            else:
                in_loss += 1

            if pnl > best_val:
                best_val = pnl
                best_id = s.strategy_id
            if pnl < worst_val:
                worst_val = pnl
                worst_id = s.strategy_id

            al = [l for l in s.legs if l.status == LegStatus.ACTIVE]
            active_legs += len(al)

            sell_legs = [l for l in s.legs if l.side == OrderSide.SELL and l.entry_price > 0]
            buy_legs = [l for l in s.legs if l.side == OrderSide.BUY and l.entry_price > 0]
            prem = sum(l.entry_price * l.quantity for l in sell_legs) - \
                   sum(l.entry_price * l.quantity for l in buy_legs)
            total_premium += prem

            theta_abs = abs(ng.theta)
            total_theta += theta_abs

            margin = self._estimate_margin(s)
            total_margin += margin

            # Track exposure by stock_code
            instrument_exposure[s.stock_code] = instrument_exposure.get(s.stock_code, 0) + abs(margin)

        # Concentration risk (max instrument / total margin)
        if total_margin > 0 and instrument_exposure:
            max_exp = max(instrument_exposure.values())
            concentration_risk = max_exp / total_margin
        else:
            concentration_risk = 0.0

        theta_margin_pct = (total_theta / total_margin * 100) if total_margin > 0 else 0

        # Max single loss
        max_single_loss = min((s.compute_total_pnl() for s in active_strats), default=0)

        # Portfolio health
        portfolio_health = self._portfolio_health_score(
            total_pnl, total_premium, concentration_risk, active_strats
        )

        return PortfolioSummary(
            total_pnl=round(total_pnl, 2),
            total_premium=round(total_premium, 2),
            total_theta=round(total_theta, 2),
            total_margin=round(total_margin, 2),
            theta_margin_pct=round(theta_margin_pct, 3),
            net_delta=round(net_delta, 2),
            net_gamma=round(net_gamma, 4),
            net_vega=round(net_vega, 2),
            active_strategies=len(active_strats),
            active_legs=active_legs,
            strategies_in_profit=in_profit,
            strategies_in_loss=in_loss,
            biggest_winner=best_id,
            biggest_loser=worst_id,
            max_single_loss=round(max_single_loss, 2),
            concentration_risk=round(concentration_risk, 3),
            portfolio_health=portfolio_health,
        )

    def get_mtm_sparkline_data(self, mtm_history: List[dict]) -> List[dict]:
        """Return last N MTM points suitable for sparkline chart."""
        if not mtm_history:
            return []
        recent = mtm_history[-200:]
        return [{"time": p["time"], "MTM": p["mtm"]} for p in recent]

    def get_greeks_gauges(self, portfolio: PortfolioSummary) -> List[dict]:
        """Return gauge data for each portfolio Greek."""
        return [
            {
                "name": "Net Δ",
                "value": portfolio.net_delta,
                "ideal": 0,
                "warn_level": 30,
                "danger_level": 60,
                "unit": "",
                "description": "Portfolio net delta exposure (short premium ideal near 0)",
            },
            {
                "name": "Net Γ",
                "value": portfolio.net_gamma,
                "ideal": 0,
                "warn_level": None,
                "danger_level": None,
                "unit": "",
                "description": "Net gamma (negative = short gamma, standard for premium sellers)",
            },
            {
                "name": "Daily Θ",
                "value": portfolio.total_theta,
                "ideal": None,
                "warn_level": None,
                "danger_level": None,
                "unit": "₹/day",
                "description": "Total daily theta decay collected",
            },
            {
                "name": "Net V",
                "value": portfolio.net_vega,
                "ideal": 0,
                "warn_level": None,
                "danger_level": None,
                "unit": "",
                "description": "Net vega (negative = hurt by IV expansion)",
            },
        ]

    # ── Private helpers ────────────────────────────────────────

    def _estimate_margin(self, strategy: Strategy) -> float:
        """Rough margin estimate for short legs."""
        base = self.MARGIN_PER_LOT_ESTIMATES.get(strategy.stock_code, 80_000)
        sell_qty = sum(l.quantity for l in strategy.legs if l.side == OrderSide.SELL)
        lot_size = max(sell_qty // 2, 1)  # crude lot approx
        margin = base * lot_size

        # Defined-risk (condor/butterfly) gets a discount since wings cap loss
        from models import StrategyType
        if strategy.strategy_type in (StrategyType.IRON_CONDOR, StrategyType.IRON_BUTTERFLY):
            margin *= 0.55
        return margin

    def _compute_breakevens(self, strategy: Strategy, total_premium: float) -> Tuple[float, float]:
        """Compute upper/lower breakeven for short strategies."""
        active = [l for l in strategy.legs if l.status == LegStatus.ACTIVE]
        ce = [l.strike_price for l in active if l.right == OptionRight.CALL and l.side == OrderSide.SELL]
        pe = [l.strike_price for l in active if l.right == OptionRight.PUT and l.side == OrderSide.SELL]

        qty = max(sum(l.quantity for l in active if l.side == OrderSide.SELL), 1)
        prem_per_qty = total_premium / qty

        be_upper = (max(ce) + prem_per_qty) if ce else 0.0
        be_lower = (min(pe) - prem_per_qty) if pe else 0.0
        return round(be_upper, 1), round(be_lower, 1)

    def _compute_max_loss(self, strategy: Strategy, total_premium: float) -> float:
        """Approximate max loss. For undefined-risk = very large; for defined = wing width - premium."""
        from models import StrategyType
        if strategy.strategy_type in (StrategyType.IRON_CONDOR, StrategyType.IRON_BUTTERFLY):
            # max loss = wing_width * qty - total_premium
            all_legs = [l for l in strategy.legs if l.entry_price > 0]
            sell_strikes_ce = [l.strike_price for l in all_legs if l.right == OptionRight.CALL and l.side == OrderSide.SELL]
            buy_strikes_ce = [l.strike_price for l in all_legs if l.right == OptionRight.CALL and l.side == OrderSide.BUY]
            if sell_strikes_ce and buy_strikes_ce:
                wing = max(buy_strikes_ce) - min(sell_strikes_ce)
                qty = max(sum(l.quantity for l in all_legs if l.side == OrderSide.SELL), 1)
                return max(wing * qty - total_premium, 0)
        # Undefined risk — return 3x total premium as proxy
        return total_premium * 3

    def _compute_health(
        self,
        strategy: Strategy,
        pnl: float,
        total_premium: float,
        pnl_pct: float,
        dte: float,
        ng: Greeks,
        spot: float,
        be_upper: float,
        be_lower: float,
    ) -> Tuple[float, str, List[str]]:
        """Score strategy health 0-100 and produce notes."""
        score = 70.0  # start neutral
        notes = []

        # P&L component
        if pnl_pct >= 50:
            score += 20
        elif pnl_pct >= 25:
            score += 10
        elif pnl_pct >= 0:
            score += 5
        elif pnl_pct >= -25:
            score -= 10
            notes.append(f"Drawdown {pnl_pct:.0f}% of premium")
        elif pnl_pct >= -50:
            score -= 25
            notes.append(f"⚠️ Significant drawdown {pnl_pct:.0f}%")
        else:
            score -= 40
            notes.append(f"🔴 Severe drawdown {pnl_pct:.0f}%")

        # Delta risk
        abs_delta = abs(ng.delta)
        if abs_delta > 60:
            score -= 20
            notes.append(f"⚠️ High net delta {ng.delta:+.0f}")
        elif abs_delta > 30:
            score -= 8
            notes.append(f"Net delta elevated: {ng.delta:+.0f}")

        # Spot vs breakevens
        if spot > 0 and be_upper > 0 and be_lower > 0:
            pct_to_upper = (be_upper - spot) / spot * 100
            pct_to_lower = (spot - be_lower) / spot * 100

            if pct_to_upper < 0 or pct_to_lower < 0:
                score -= 30
                notes.append("🔴 Spot beyond breakeven!")
            elif min(pct_to_upper, pct_to_lower) < 0.5:
                score -= 15
                notes.append("⚠️ Very close to breakeven")
            elif min(pct_to_upper, pct_to_lower) < 1.5:
                score -= 5
                notes.append(f"Near breakeven — upper {pct_to_upper:.1f}%, lower {pct_to_lower:.1f}%")

        # DTE
        if dte <= 0:
            score -= 10
            notes.append("Expiry imminent")
        elif dte <= 1:
            score -= 5
            notes.append("Expiry today/tomorrow — gamma risk")

        # Positive theta is good
        if ng.theta > 5:
            score += 5
        elif ng.theta < 0:
            score -= 5
            notes.append("Negative theta (net long options)")

        score = max(0, min(100, score))

        if score >= 80:
            label = "Excellent"
        elif score >= 65:
            label = "Good"
        elif score >= 45:
            label = "Caution"
        elif score >= 25:
            label = "Warning"
        else:
            label = "Critical"

        if not notes:
            notes.append("Strategy within normal parameters")

        return round(score, 1), label, notes

    def _portfolio_health_score(
        self,
        total_pnl: float,
        total_premium: float,
        concentration_risk: float,
        active_strats: List[Strategy],
    ) -> float:
        score = 60.0
        if total_premium > 0:
            pct = total_pnl / total_premium * 100
            if pct > 30:
                score += 20
            elif pct > 10:
                score += 10
            elif pct < -30:
                score -= 20
            elif pct < -10:
                score -= 10

        if concentration_risk > 0.85:
            score -= 15
        elif concentration_risk > 0.70:
            score -= 7

        if len(active_strats) > 6:
            score -= 5  # over-trading risk

        return round(max(0, min(100, score)), 1)
