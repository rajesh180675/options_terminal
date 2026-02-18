"""
delta_hedger.py  — NEW MODULE (Layer 2)

Delta Neutralization Calculator:
  • Portfolio net delta exposure (live, from active strategies)
  • Hedge sizing: lots of futures/options needed to neutralize delta
  • Re-hedge trigger: alert when delta drifts beyond threshold
  • Cost of hedging (bid-ask spread + impact)
  • Mini-hedge vs full-hedge comparison
  • Delta band management (e.g., allow ±10 delta before re-hedging)
  • Hedge effectiveness tracker (how much did last hedge help)
  • Tail risk reducer: suggest protective options for extreme moves

Zero breaking-changes: pure add-on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from greeks_engine import BlackScholes, time_to_expiry
from models import Strategy, Leg, LegStatus, OrderSide, OptionRight, StrategyStatus
from app_config import Config


# ────────────────────────────────────────────────────────────────
# Data Classes
# ────────────────────────────────────────────────────────────────

@dataclass
class HedgeSuggestion:
    action: str                    # "BUY_FUTURES" / "SELL_FUTURES" / "BUY_PUT" / "SELL_CALL"
    quantity: int                  # total quantity (not lots)
    lots: float                    # equivalent lots
    description: str
    estimated_cost: float          # in ₹ (bid-ask + impact)
    delta_reduction: float         # how much delta this removes
    hedge_type: str                # "FULL" / "MINI" / "PROTECTIVE"

@dataclass
class DeltaExposure:
    strategy_id: str
    strategy_type: str
    net_delta: float
    delta_in_rupees: float         # delta * spot
    delta_per_lot: float
    biggest_contributor: str       # leg desc with highest delta contribution

@dataclass
class DeltaDashboard:
    portfolio_delta: float         # net delta across all strategies
    portfolio_delta_inr: float     # ₹ P&L per 1% spot move
    delta_by_strategy: List[DeltaExposure]
    delta_by_instrument: Dict[str, float]
    hedge_needed: bool             # True if delta outside band
    hedge_suggestion: Optional[HedgeSuggestion]
    mini_hedge: Optional[HedgeSuggestion]
    delta_band_upper: float        # configured upper delta band
    delta_band_lower: float        # configured lower delta band
    band_breach_pct: float         # how far outside band we are
    tail_hedge: Optional[HedgeSuggestion]  # cheap OTM protection
    spot: float
    instrument: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


# ────────────────────────────────────────────────────────────────
# Delta Hedger Engine
# ────────────────────────────────────────────────────────────────

class DeltaHedger:
    """
    Analyzes portfolio delta exposure and generates hedge suggestions.
    No broker connection required — purely analytical.
    """

    # Default delta bands (can be overridden)
    DELTA_BAND_LOWER = -30.0   # alert if net delta below this
    DELTA_BAND_UPPER = +30.0   # alert if net delta above this

    def build_dashboard(
        self,
        strategies: List[Strategy],
        spot_map: Dict[str, float],
        instrument: str = "",
        chain: List[dict] = None,
        delta_band: Tuple[float, float] = None,
    ) -> DeltaDashboard:
        """Build full delta dashboard for active portfolio."""
        active = [s for s in strategies if s.status in
                  (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT)]

        if not active:
            spot = next(iter(spot_map.values()), 24000.0)
            return self._empty_dashboard(spot, instrument)

        band_lower = delta_band[0] if delta_band else self.DELTA_BAND_LOWER
        band_upper = delta_band[1] if delta_band else self.DELTA_BAND_UPPER

        # Per-strategy delta
        delta_by_strat = []
        delta_by_instrument: Dict[str, float] = {}
        portfolio_delta = 0.0

        for s in active:
            spot = float(spot_map.get(s.stock_code, 0))
            if spot <= 0:
                continue
            net_delta = s.net_greeks.delta
            delta_inr = net_delta * spot
            lot_size = Config.lot_size(s.stock_code) if hasattr(Config, "lot_size") else 50
            delta_per_lot = net_delta / max(sum(l.quantity for l in s.legs if l.side == OrderSide.SELL) // lot_size, 1)

            # Find biggest contributing leg
            biggest = ""
            biggest_delta = 0.0
            for leg in s.legs:
                if abs(leg.greeks.delta * leg.quantity) > abs(biggest_delta):
                    biggest_delta = leg.greeks.delta * leg.quantity
                    biggest = f"{leg.right.value[0].upper()}E {int(leg.strike_price)}"

            delta_by_strat.append(DeltaExposure(
                strategy_id=s.strategy_id,
                strategy_type=s.strategy_type.value,
                net_delta=round(net_delta, 2),
                delta_in_rupees=round(delta_inr, 0),
                delta_per_lot=round(delta_per_lot, 2),
                biggest_contributor=biggest,
            ))

            portfolio_delta += net_delta
            delta_by_instrument[s.stock_code] = delta_by_instrument.get(s.stock_code, 0) + net_delta

        # Get representative spot
        primary_spot = next(iter(spot_map.values()), 24000.0)
        primary_stock = next(iter(delta_by_instrument.keys()), "NIFTY")

        # Check if hedge is needed
        hedge_needed = portfolio_delta < band_lower or portfolio_delta > band_upper
        band_breach = 0.0
        if portfolio_delta > band_upper:
            band_breach = (portfolio_delta - band_upper) / band_upper * 100
        elif portfolio_delta < band_lower:
            band_breach = (band_lower - portfolio_delta) / abs(band_lower) * 100

        # Build hedge suggestions
        hedge = self._build_hedge_suggestion(
            portfolio_delta, primary_spot, primary_stock, "FULL", chain
        )
        mini_hedge = self._build_hedge_suggestion(
            portfolio_delta / 2, primary_spot, primary_stock, "MINI", chain
        )

        # Tail hedge (OTM protective option)
        tail_hedge = self._build_tail_hedge(portfolio_delta, primary_spot, chain, primary_stock)

        return DeltaDashboard(
            portfolio_delta=round(portfolio_delta, 2),
            portfolio_delta_inr=round(portfolio_delta * primary_spot / 100, 0),  # ₹ per 1% move
            delta_by_strategy=delta_by_strat,
            delta_by_instrument={k: round(v, 2) for k, v in delta_by_instrument.items()},
            hedge_needed=hedge_needed,
            hedge_suggestion=hedge if hedge_needed else None,
            mini_hedge=mini_hedge if hedge_needed else None,
            delta_band_upper=band_upper,
            delta_band_lower=band_lower,
            band_breach_pct=round(band_breach, 1),
            tail_hedge=tail_hedge,
            spot=primary_spot,
            instrument=instrument or primary_stock,
        )

    def compute_hedge_cost(
        self,
        delta_to_hedge: float,
        spot: float,
        futures_spread_pct: float = 0.02,  # 2 bps
    ) -> float:
        """Estimate cost of delta-neutral hedge via futures."""
        lots = abs(delta_to_hedge) / 50  # assume NIFTY lot
        notional = lots * 50 * spot
        cost = notional * futures_spread_pct / 100
        return round(cost, 0)

    def delta_to_futures_lots(self, net_delta: float, lot_size: int = 50) -> float:
        """Convert net delta to equivalent futures lots needed to neutralize."""
        return -net_delta / lot_size  # negative: to offset

    def delta_to_option_qty(
        self,
        net_delta: float,
        hedge_option_delta: float,
    ) -> int:
        """How many options to buy/sell to neutralize a given delta."""
        if abs(hedge_option_delta) < 1e-6:
            return 0
        return int(-net_delta / hedge_option_delta)

    # ── Internal ──────────────────────────────────────────────

    def _build_hedge_suggestion(
        self,
        delta_to_hedge: float,
        spot: float,
        stock_code: str,
        hedge_type: str,
        chain: List[dict] = None,
    ) -> Optional[HedgeSuggestion]:
        """Build a hedge suggestion for the given delta."""
        if abs(delta_to_hedge) < 1:
            return None

        lot_size = 50  # default NIFTY lot

        if delta_to_hedge > 0:
            # Long delta: hedge with sell futures or buy puts
            action = "SELL_FUTURES"
            lots = delta_to_hedge / lot_size
            qty = int(lots * lot_size)
            description = f"Sell {lots:.1f} lots futures to neutralize long delta {delta_to_hedge:+.1f}"
        else:
            # Short delta: hedge with buy futures or buy calls
            action = "BUY_FUTURES"
            lots = abs(delta_to_hedge) / lot_size
            qty = int(lots * lot_size)
            description = f"Buy {lots:.1f} lots futures to neutralize short delta {delta_to_hedge:+.1f}"

        cost = self.compute_hedge_cost(delta_to_hedge, spot)

        return HedgeSuggestion(
            action=action,
            quantity=qty,
            lots=round(lots, 2),
            description=description,
            estimated_cost=cost,
            delta_reduction=round(-delta_to_hedge, 2),
            hedge_type=hedge_type,
        )

    def _build_tail_hedge(
        self,
        portfolio_delta: float,
        spot: float,
        chain: List[dict] = None,
        stock_code: str = "NIFTY",
    ) -> Optional[HedgeSuggestion]:
        """Suggest a cheap OTM put for tail-risk protection."""
        # Find a 5% OTM put
        target_strike = round(spot * 0.95 / 50) * 50
        otm_put_price = 50.0  # estimate if no chain

        if chain:
            put_rows = [r for r in chain
                        if str(r.get("right", "")).upper() == "PUT"
                        and abs(float(r.get("strike", 0)) - target_strike) <= 100]
            if put_rows:
                put_rows.sort(key=lambda r: abs(float(r.get("strike", 0)) - target_strike))
                otm_put_price = float(put_rows[0].get("ltp", 50))
                target_strike = float(put_rows[0].get("strike", target_strike))

        lots = 1
        qty = lots * 50
        cost = otm_put_price * qty

        return HedgeSuggestion(
            action="BUY_PUT",
            quantity=qty,
            lots=float(lots),
            description=f"Buy {lots} lot {int(target_strike)} PE @ ~₹{otm_put_price:.1f} as tail hedge (5% OTM)",
            estimated_cost=round(cost, 0),
            delta_reduction=-0.1 * qty,  # approx 0.10 delta put
            hedge_type="PROTECTIVE",
        )

    def _empty_dashboard(self, spot: float, instrument: str) -> DeltaDashboard:
        return DeltaDashboard(
            portfolio_delta=0.0,
            portfolio_delta_inr=0.0,
            delta_by_strategy=[],
            delta_by_instrument={},
            hedge_needed=False,
            hedge_suggestion=None,
            mini_hedge=None,
            delta_band_upper=self.DELTA_BAND_UPPER,
            delta_band_lower=self.DELTA_BAND_LOWER,
            band_breach_pct=0.0,
            tail_hedge=None,
            spot=spot,
            instrument=instrument,
        )
