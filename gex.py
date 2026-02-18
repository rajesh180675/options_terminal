"""
gex.py  — NEW MODULE (Layer 2)

Gamma Exposure (GEX) Analyzer:
  • Computes dealer gamma exposure by strike (GEX profile)
  • Identifies key GEX flip points (support/resistance)
  • Net GEX: positive (stabilizing) vs negative (destabilizing)
  • GEX per spot level — shows where dealer hedging magnifies moves
  • Zero-GEX level detection (critical flip point)
  • GEX concentration heatmap data
  • Pin risk zones (strikes with highest gamma near expiry)
  • GEX-based expected range for the day

Market mechanics:
  - Dealers are SHORT calls (when clients buy calls) → positive delta
    → As spot rises, dealers must BUY futures (stabilizing above flip)
  - Dealers are LONG puts (when clients buy puts) → negative delta
    → As spot falls, dealers must BUY futures (stabilizing support)
  - GEX = gamma * OI * lot_size * spot^2 / 100

Zero breaking-changes: pure add-on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from greeks_engine import BlackScholes, time_to_expiry
from models import OptionRight
from app_config import Config


# ────────────────────────────────────────────────────────────────
# Data Classes
# ────────────────────────────────────────────────────────────────

@dataclass
class GEXStrike:
    strike: float
    call_gex: float              # dealer gamma from call OI (positive = dealer short calls)
    put_gex: float               # dealer gamma from put OI (negative = dealer long puts)
    net_gex: float               # call_gex + put_gex
    call_oi: float
    put_oi: float
    is_flip_zone: bool           # sign changes near here
    is_pin_zone: bool            # high absolute GEX near expiry

@dataclass
class GEXProfile:
    instrument: str
    spot: float
    expiry: str
    net_total_gex: float         # aggregate (positive = stabilizing)
    gex_flip_strike: float       # nearest zero-crossing
    largest_call_wall: float     # strike with highest positive GEX (resistance)
    largest_put_wall: float      # strike with largest negative GEX (support)
    gex_range_upper: float       # GEX-based expected upper range
    gex_range_lower: float       # GEX-based expected lower range
    strikes: List[GEXStrike]
    regime: str                  # "POSITIVE" (stable) / "NEGATIVE" (volatile)
    regime_description: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))

    def to_chart_data(self) -> List[dict]:
        """Return chart-ready data for bar chart."""
        return [
            {
                "Strike": int(s.strike),
                "Call GEX (₹M)": round(s.call_gex / 1e6, 2),
                "Put GEX (₹M)": round(s.put_gex / 1e6, 2),
                "Net GEX (₹M)": round(s.net_gex / 1e6, 2),
            }
            for s in self.strikes
        ]


# ────────────────────────────────────────────────────────────────
# GEX Engine
# ────────────────────────────────────────────────────────────────

class GEXEngine:
    """
    Computes Gamma Exposure (GEX) from option chain data.
    GEX quantifies how much dealers need to hedge as spot moves.
    """

    def compute_gex_profile(
        self,
        chain: List[dict],
        spot: float,
        instrument: str = "",
        expiry: str = "",
        lot_size: int = None,
    ) -> Optional[GEXProfile]:
        """
        Build full GEX profile from live option chain.
        chain: list of {strike, right, oi, iv, ltp, delta, gamma}
        """
        if not chain or spot <= 0:
            return None

        from utils import atm_strike
        inst = Config.instrument(instrument) if instrument else {"strike_gap": 50, "lot_size": 50}
        gap = inst.get("strike_gap", 50)
        if lot_size is None:
            lot_size = inst.get("lot_size", 50)

        T = time_to_expiry(expiry) if expiry else 0.05  # fallback 18 days
        r = Config.RISK_FREE_RATE

        # Group by strike
        by_strike: Dict[float, Dict] = {}
        for row in chain:
            k = float(row.get("strike", 0))
            oi = float(row.get("oi", 0))
            iv = float(row.get("iv", 0))
            right = str(row.get("right", "")).upper()
            if k <= 0 or oi <= 0 or iv <= 0:
                continue

            if k not in by_strike:
                by_strike[k] = {"call_oi": 0, "put_oi": 0, "call_iv": 0, "put_iv": 0}

            if right == "CALL":
                by_strike[k]["call_oi"] += oi
                by_strike[k]["call_iv"] = iv
            else:
                by_strike[k]["put_oi"] += oi
                by_strike[k]["put_iv"] = iv

        # Compute GEX per strike
        gex_strikes = []
        for k, data in sorted(by_strike.items()):
            call_gamma = 0.0
            put_gamma = 0.0

            if data["call_iv"] > 0 and data["call_oi"] > 0:
                iv_c = data["call_iv"] / 100
                call_gamma = BlackScholes.gamma(spot, k, T, r, iv_c)

            if data["put_iv"] > 0 and data["put_oi"] > 0:
                iv_p = data["put_iv"] / 100
                put_gamma = BlackScholes.gamma(spot, k, T, r, iv_p)

            # GEX formula: gamma * OI * lot_size * spot² / 100
            call_gex = call_gamma * data["call_oi"] * lot_size * spot * spot / 100
            # Dealer is assumed SHORT calls → positive GEX
            # Dealer is assumed LONG puts → negative GEX
            put_gex = -put_gamma * data["put_oi"] * lot_size * spot * spot / 100

            net_gex = call_gex + put_gex

            gex_strikes.append(GEXStrike(
                strike=k,
                call_gex=round(call_gex, 0),
                put_gex=round(put_gex, 0),
                net_gex=round(net_gex, 0),
                call_oi=data["call_oi"],
                put_oi=data["put_oi"],
                is_flip_zone=False,  # set below
                is_pin_zone=abs(k - spot) < gap * 2 and T * 365 < 5,
            ))

        if not gex_strikes:
            return None

        # Find zero-GEX crossing (flip point)
        flip_strike = self._find_gex_flip(gex_strikes, spot)

        # Mark flip zones
        for gs in gex_strikes:
            if abs(gs.strike - flip_strike) <= gap:
                gs.is_flip_zone = True

        # Total GEX
        net_total = sum(gs.net_gex for gs in gex_strikes)

        # Call wall (highest positive GEX above spot = resistance)
        above_spot = [gs for gs in gex_strikes if gs.strike > spot]
        call_wall = max(above_spot, key=lambda x: x.call_gex).strike if above_spot else spot * 1.02

        # Put wall (most negative GEX below spot = support)
        below_spot = [gs for gs in gex_strikes if gs.strike <= spot]
        put_wall = min(below_spot, key=lambda x: x.put_gex).strike if below_spot else spot * 0.98

        # GEX-based daily range (rough approximation)
        total_abs_gex = sum(abs(gs.net_gex) for gs in gex_strikes) or 1
        # Higher GEX → smaller expected move (dealers stabilize)
        gex_dampening = min(total_abs_gex / 1e9 * 0.3, 0.5)  # 0 to 50% dampening
        base_move = spot * 0.01  # 1% base daily range
        expected_range = base_move * (1 - gex_dampening)
        gex_upper = spot + expected_range
        gex_lower = spot - expected_range

        regime = "POSITIVE" if net_total > 0 else "NEGATIVE"
        regime_desc = (
            "Positive GEX: Market makers are stabilizing. Large moves are dampened by dealer hedging."
            if regime == "POSITIVE" else
            "Negative GEX: Market makers amplify moves. Volatility likely to be higher."
        )

        return GEXProfile(
            instrument=instrument,
            spot=spot,
            expiry=expiry,
            net_total_gex=round(net_total / 1e6, 2),  # in ₹M
            gex_flip_strike=round(flip_strike, 0),
            largest_call_wall=round(call_wall, 0),
            largest_put_wall=round(put_wall, 0),
            gex_range_upper=round(gex_upper, 1),
            gex_range_lower=round(gex_lower, 1),
            strikes=gex_strikes,
            regime=regime,
            regime_description=regime_desc,
        )

    def _find_gex_flip(self, gex_strikes: List[GEXStrike], spot: float) -> float:
        """Find the strike nearest to a zero-GEX crossing."""
        # Sort by proximity to spot
        near_spot = sorted(gex_strikes, key=lambda x: abs(x.strike - spot))[:20]
        near_spot.sort(key=lambda x: x.strike)

        # Find sign change in net_gex
        for i in range(1, len(near_spot)):
            if near_spot[i-1].net_gex * near_spot[i].net_gex < 0:
                # Interpolate
                s1 = near_spot[i-1].strike
                s2 = near_spot[i].strike
                g1 = abs(near_spot[i-1].net_gex)
                g2 = abs(near_spot[i].net_gex)
                flip = s1 + (s2 - s1) * g1 / (g1 + g2)
                return flip

        # No sign change found: return strike with minimum absolute GEX
        return min(near_spot, key=lambda x: abs(x.net_gex), default=near_spot[0]).strike

    def get_key_levels(self, profile: GEXProfile) -> List[dict]:
        """Return key GEX levels for display."""
        return [
            {
                "level": "Call Wall (Resistance)",
                "strike": int(profile.largest_call_wall),
                "type": "resistance",
                "color": "#ff4444",
                "description": f"Strong dealer call hedging → resistance at {int(profile.largest_call_wall)}",
            },
            {
                "level": "Put Wall (Support)",
                "strike": int(profile.largest_put_wall),
                "type": "support",
                "color": "#00c853",
                "description": f"Strong dealer put hedging → support at {int(profile.largest_put_wall)}",
            },
            {
                "level": "GEX Flip Point",
                "strike": int(profile.gex_flip_strike),
                "type": "flip",
                "color": "#ff9800",
                "description": f"Zero-GEX level at {int(profile.gex_flip_strike)} — regime changes here",
            },
        ]
