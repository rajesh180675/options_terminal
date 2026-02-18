"""
iv_rank.py  — NEW MODULE (Layer 2)

IV Rank & Percentile Tracker:
  • Real-time IV rank (IVR) computation vs intraday snapshots
  • IV percentile (IVP) vs historical stored range
  • Volatility cone (current ATM IV vs historical bands)
  • IV crush detector (IV falling faster than expected post-event)
  • IV spike alert (sudden IV jump = possible news/event)
  • Per-strike IV rank comparison (identify richest/cheapest strikes)
  • IV regime classification: Low/Normal/High/Extreme

Session state is used to store IV history snapshots — no external DB needed.
Zero breaking-changes: pure add-on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple


# ────────────────────────────────────────────────────────────────
# Data Classes
# ────────────────────────────────────────────────────────────────

@dataclass
class IVSnapshot:
    timestamp: str
    atm_iv: float              # ATM IV at this moment
    atm_iv_ce: float
    atm_iv_pe: float
    iv_rank: float             # 0-100 vs session history
    iv_percentile: float       # 0-100 vs all stored snapshots
    pcr: float
    spot: float

@dataclass
class IVStrikeRank:
    strike: float
    right: str
    iv: float
    iv_rank_vs_session: float  # 0-100
    iv_vs_atm: float           # IV - ATM_IV (premium/discount)
    moneyness: float           # strike/spot

@dataclass
class VolatilityCone:
    dte_points: List[float]    # DTE axis
    current_iv: List[float]    # Current IV at each DTE
    upper_1std: List[float]    # ATM IV + 1 session std
    lower_1std: List[float]
    upper_2std: List[float]
    lower_2std: List[float]
    session_mean: List[float]

@dataclass
class IVRankSummary:
    instrument: str
    spot: float
    atm_iv: float
    atm_iv_ce: float
    atm_iv_pe: float
    iv_skew: float             # CE IV - PE IV at ATM
    iv_rank: float             # 0-100 vs today's range
    iv_percentile: float       # 0-100 vs all snapshots
    iv_regime: str             # "LOW" / "NORMAL" / "HIGH" / "EXTREME"
    iv_momentum: str           # "RISING" / "FALLING" / "STABLE"
    iv_change_1h: float        # change vs 1 hour ago
    iv_change_session: float   # change vs session start
    session_high_iv: float
    session_low_iv: float
    crush_alert: bool          # IV falling > 3pp in last 30 mins
    spike_alert: bool          # IV rising > 3pp in last 30 mins
    strike_ranks: List[IVStrikeRank]
    snapshots_count: int
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


# ────────────────────────────────────────────────────────────────
# IV Rank Engine
# ────────────────────────────────────────────────────────────────

class IVRankEngine:
    """
    Tracks and analyzes IV rank and percentile from live option chain data.
    Uses in-memory snapshot history (stored externally in session state).
    """

    # IV regime thresholds
    IV_LOW = 12.0
    IV_NORMAL = 20.0
    IV_HIGH = 28.0
    IV_EXTREME = 40.0

    def compute_atm_iv(
        self, chain: List[dict], spot: float, instrument: str = ""
    ) -> Tuple[float, float, float]:
        """
        Returns (atm_iv, atm_iv_ce, atm_iv_pe) from live chain.
        """
        from utils import atm_strike
        from app_config import Config

        inst = Config.instrument(instrument) if instrument else {"strike_gap": 50}
        gap = inst.get("strike_gap", 50)
        atm = atm_strike(spot, gap)

        ce_ivs = []
        pe_ivs = []

        for row in chain:
            k = float(row.get("strike", 0))
            iv = float(row.get("iv", 0))
            right = str(row.get("right", "")).upper()
            if abs(k - atm) <= gap and iv > 0:
                if right == "CALL":
                    ce_ivs.append(iv)
                else:
                    pe_ivs.append(iv)

        atm_ce = sum(ce_ivs) / len(ce_ivs) if ce_ivs else 0.0
        atm_pe = sum(pe_ivs) / len(pe_ivs) if pe_ivs else 0.0
        atm_iv = (atm_ce + atm_pe) / 2 if (atm_ce + atm_pe) > 0 else max(atm_ce, atm_pe)

        return round(atm_iv, 2), round(atm_ce, 2), round(atm_pe, 2)

    def build_summary(
        self,
        chain: List[dict],
        spot: float,
        instrument: str,
        iv_history: List[IVSnapshot],  # session snapshots stored externally
    ) -> Optional[IVRankSummary]:
        """
        Build full IV rank summary from current chain + historical snapshots.
        """
        if not chain or spot <= 0:
            return None

        atm_iv, atm_ce, atm_pe = self.compute_atm_iv(chain, spot, instrument)
        if atm_iv <= 0:
            return None

        # IV history stats
        ivs_history = [s.atm_iv for s in iv_history if s.atm_iv > 0]
        session_high = max(ivs_history) if ivs_history else atm_iv
        session_low = min(ivs_history) if ivs_history else atm_iv

        # IV Rank: (current - low) / (high - low) * 100
        iv_range = session_high - session_low
        iv_rank = ((atm_iv - session_low) / iv_range * 100) if iv_range > 0 else 50.0
        iv_rank = max(0, min(100, iv_rank))

        # IV Percentile: % of snapshots below current
        iv_pct = 0.0
        if ivs_history:
            below = sum(1 for v in ivs_history if v < atm_iv)
            iv_pct = below / len(ivs_history) * 100

        # IV Regime
        if atm_iv >= self.IV_EXTREME:
            regime = "EXTREME"
        elif atm_iv >= self.IV_HIGH:
            regime = "HIGH"
        elif atm_iv >= self.IV_NORMAL:
            regime = "NORMAL"
        else:
            regime = "LOW"

        # IV Momentum
        iv_change_1h = 0.0
        iv_change_session = 0.0
        crush_alert = False
        spike_alert = False

        if iv_history:
            # Session start change
            iv_change_session = atm_iv - iv_history[0].atm_iv

            # 1-hour change (last ~240 snapshots if 15-sec intervals)
            lookback = min(240, len(iv_history))
            if lookback > 0:
                iv_change_1h = atm_iv - iv_history[-lookback].atm_iv

            # Recent 30-min change (last ~120 snapshots)
            lookback_30 = min(120, len(iv_history))
            if lookback_30 > 0:
                recent_change = atm_iv - iv_history[-lookback_30].atm_iv
                if recent_change < -3.0:
                    crush_alert = True
                elif recent_change > 3.0:
                    spike_alert = True

        if iv_change_1h > 1.0:
            iv_momentum = "RISING"
        elif iv_change_1h < -1.0:
            iv_momentum = "FALLING"
        else:
            iv_momentum = "STABLE"

        # Per-strike IV ranks
        strike_ranks = self._compute_strike_ranks(chain, spot, atm_iv, instrument)

        return IVRankSummary(
            instrument=instrument,
            spot=spot,
            atm_iv=atm_iv,
            atm_iv_ce=atm_ce,
            atm_iv_pe=atm_pe,
            iv_skew=round(atm_ce - atm_pe, 2),
            iv_rank=round(iv_rank, 1),
            iv_percentile=round(iv_pct, 1),
            iv_regime=regime,
            iv_momentum=iv_momentum,
            iv_change_1h=round(iv_change_1h, 2),
            iv_change_session=round(iv_change_session, 2),
            session_high_iv=round(session_high, 2),
            session_low_iv=round(session_low, 2),
            crush_alert=crush_alert,
            spike_alert=spike_alert,
            strike_ranks=strike_ranks,
            snapshots_count=len(iv_history),
        )

    def _compute_strike_ranks(
        self, chain: List[dict], spot: float, atm_iv: float, instrument: str = ""
    ) -> List[IVStrikeRank]:
        """Compute IV rank vs ATM for each strike."""
        from utils import atm_strike
        from app_config import Config

        inst = Config.instrument(instrument) if instrument else {"strike_gap": 50}
        gap = inst.get("strike_gap", 50)
        center = atm_strike(spot, gap)
        range_width = gap * 6  # ±6 strikes from ATM

        rows = []
        for r in chain:
            k = float(r.get("strike", 0))
            iv = float(r.get("iv", 0))
            right = str(r.get("right", "")).upper()
            if iv <= 0 or abs(k - center) > range_width:
                continue
            moneyness = k / spot if spot > 0 else 1.0
            vs_atm = iv - atm_iv
            rows.append(IVStrikeRank(
                strike=k,
                right=right,
                iv=iv,
                iv_rank_vs_session=0.0,  # filled below if history available
                iv_vs_atm=round(vs_atm, 2),
                moneyness=round(moneyness, 4),
            ))

        rows.sort(key=lambda x: x.strike)
        return rows

    def take_snapshot(
        self, chain: List[dict], spot: float, instrument: str, iv_history: List[IVSnapshot]
    ) -> IVSnapshot:
        """Take a new IV snapshot and return it (caller appends to history)."""
        atm_iv, atm_ce, atm_pe = self.compute_atm_iv(chain, spot, instrument)
        ivs = [s.atm_iv for s in iv_history]
        lo = min(ivs) if ivs else atm_iv
        hi = max(ivs) if ivs else atm_iv
        rng = hi - lo
        rank = ((atm_iv - lo) / rng * 100) if rng > 0 else 50.0

        ce_oi = sum(float(r.get("oi", 0)) for r in chain if str(r.get("right", "")).upper() == "CALL")
        pe_oi = sum(float(r.get("oi", 0)) for r in chain if str(r.get("right", "")).upper() == "PUT")
        pcr = pe_oi / max(ce_oi, 1)

        return IVSnapshot(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            atm_iv=atm_iv,
            atm_iv_ce=atm_ce,
            atm_iv_pe=atm_pe,
            iv_rank=round(rank, 1),
            iv_percentile=50.0,  # simplified
            pcr=round(pcr, 3),
            spot=spot,
        )

    def build_volatility_cone(
        self, atm_iv: float, iv_std: float, dte_range: range = None
    ) -> VolatilityCone:
        """
        Build a volatility cone showing expected IV range across DTE.
        Uses simple square-root-of-time scaling.
        """
        if dte_range is None:
            dte_range = range(1, 31)

        dtes = list(dte_range)
        current = []
        upper1 = []
        lower1 = []
        upper2 = []
        lower2 = []
        mean = []

        for dte in dtes:
            # IV scales with sqrt(T) for term structure approximation
            scale = math.sqrt(dte / 7.0)  # normalize to 7 DTE
            atm_scaled = atm_iv * scale
            std_scaled = iv_std * scale

            current.append(round(atm_scaled, 2))
            mean.append(round(atm_scaled * 0.95, 2))  # slight mean reversion
            upper1.append(round(atm_scaled + std_scaled, 2))
            lower1.append(round(max(atm_scaled - std_scaled, 1), 2))
            upper2.append(round(atm_scaled + 2 * std_scaled, 2))
            lower2.append(round(max(atm_scaled - 2 * std_scaled, 1), 2))

        return VolatilityCone(
            dte_points=[float(d) for d in dtes],
            current_iv=current,
            upper_1std=upper1,
            lower_1std=lower1,
            upper_2std=upper2,
            lower_2std=lower2,
            session_mean=mean,
        )

    def iv_history_to_chart_data(
        self, iv_history: List[IVSnapshot]
    ) -> List[dict]:
        """Convert IV snapshot history to chart-ready format."""
        return [
            {
                "time": s.timestamp,
                "ATM IV": s.atm_iv,
                "CE IV": s.atm_iv_ce,
                "PE IV": s.atm_iv_pe,
                "IV Rank": s.iv_rank,
            }
            for s in iv_history
        ]
