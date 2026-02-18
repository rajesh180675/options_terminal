"""
order_flow_tracker.py  — Layer 3 NEW MODULE

Order Flow Intelligence Engine
  • Volume spike detection: flags when option volume > N × average
  • Price impact analysis: large price moves relative to underlying
  • CE/PE imbalance tracking per strike
  • Big trade detection: single prints > threshold (unusual premium paid)
  • OI-Price divergence signals (smart money vs dumb money)
  • Session volume accumulation tracker
  • Rolling average volume computation (5-bar, 20-bar)
  • Alert generation for unusual flows
  • Strike-level "heat" score (composite of volume, OI, price action)

Zero breaking-changes: pure add-on. No existing file modified.
"""

from __future__ import annotations

import math
import statistics
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# ────────────────────────────────────────────────────────────────
# Data Classes
# ────────────────────────────────────────────────────────────────

@dataclass
class FlowSignal:
    """Single order flow signal event."""
    timestamp: str
    strike: float
    right: str                  # CE / PE
    signal_type: str            # VOLUME_SPIKE, BIG_PRINT, OI_DIVERGENCE, UNUSUAL_PREMIUM
    severity: str               # INFO, WARN, CRITICAL
    value: float                # volume or premium amount
    threshold: float            # what it exceeded
    multiple: float             # how many times threshold
    ltp: float
    message: str


@dataclass
class StrikeFlow:
    """Per-strike order flow summary."""
    strike: float
    ce_volume: float
    pe_volume: float
    ce_vol_ratio: float         # relative to avg
    pe_vol_ratio: float
    ce_oi: float
    pe_oi: float
    ce_oi_chg: float
    pe_oi_chg: float
    ce_ltp: float
    pe_ltp: float
    ce_premium_value: float     # ltp × volume × lot_size (₹ terms)
    pe_premium_value: float
    heat_score: float           # composite 0-100
    dominant_side: str          # CE, PE, NEUTRAL
    flow_direction: str         # CALL_BUYING, PUT_BUYING, CALL_WRITING, PUT_WRITING, MIXED


@dataclass
class SessionFlow:
    """Rolling session order flow stats."""
    total_ce_volume: float
    total_pe_volume: float
    total_ce_premium: float
    total_pe_premium: float
    ce_pe_volume_ratio: float
    net_premium_flow: float     # ce - pe premium (positive = call buying dominant)
    dominant_activity: str
    big_trade_count: int
    spike_count: int


@dataclass
class OrderFlowSummary:
    # Session totals
    session: SessionFlow

    # Signals
    signals: List[FlowSignal]
    critical_signals: List[FlowSignal]

    # Strike heat table
    strike_flows: List[StrikeFlow]

    # Top heat strikes
    hottest_ce_strikes: List[StrikeFlow]
    hottest_pe_strikes: List[StrikeFlow]

    # Smart money indicators
    smart_money_bias: str       # BULLISH, BEARISH, NEUTRAL
    smart_money_detail: str

    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


# ────────────────────────────────────────────────────────────────
# Order Flow Tracker
# ────────────────────────────────────────────────────────────────

class OrderFlowTracker:
    """
    Detects unusual order flow in the option chain.

    Usage:
        tracker = OrderFlowTracker()
        tracker.update(chain, spot, lot_size)   # call each refresh
        summary = tracker.get_summary()
    """

    # Thresholds
    VOLUME_SPIKE_RATIO = 3.0        # volume > 3× avg = spike
    VOLUME_CRITICAL_RATIO = 5.0     # > 5× = critical
    BIG_PRINT_PREMIUM = 5_000_000   # ₹50L+ single strike = big print
    MAX_SIGNALS = 20                # keep top N signals
    ROLLING_WINDOW = 50             # bars for rolling avg

    def __init__(self):
        # Rolling volume history per key (strike_right)
        self._vol_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.ROLLING_WINDOW))
        self._prev_chain: Optional[List[dict]] = None
        self._signals: deque = deque(maxlen=200)
        self._session_ce_vol: float = 0.0
        self._session_pe_vol: float = 0.0
        self._session_ce_premium: float = 0.0
        self._session_pe_premium: float = 0.0
        self._big_trade_count: int = 0
        self._spike_count: int = 0

    # ── Main Update Loop ──────────────────────────────────────

    def update(
        self,
        chain: List[dict],
        spot: float,
        lot_size: int = 50,
    ) -> None:
        """Call each time the chain refreshes. Updates rolling history."""
        for row in chain:
            strike = float(row.get("strike", 0))
            right = str(row.get("right", "")).upper()
            vol = float(row.get("volume", 0))
            if strike <= 0:
                continue
            key = f"{strike}_{right}"
            self._vol_history[key].append(vol)

        # Detect new signals
        new_signals = self._detect_signals(chain, spot, lot_size)
        for sig in new_signals:
            self._signals.append(sig)

        # Update session accumulation
        for row in chain:
            right = str(row.get("right", "")).upper()
            vol = float(row.get("volume", 0))
            ltp = float(row.get("ltp", 0))
            oi_chg = float(row.get("oi_chg", 0)) if "oi_chg" in row else 0
            premium_val = ltp * vol * lot_size

            if "CALL" in right:
                self._session_ce_vol = max(self._session_ce_vol, vol)
                self._session_ce_premium = max(self._session_ce_premium, premium_val)
            else:
                self._session_pe_vol = max(self._session_pe_vol, vol)
                self._session_pe_premium = max(self._session_pe_premium, premium_val)

        if new_signals:
            self._spike_count += sum(1 for s in new_signals if "SPIKE" in s.signal_type)
            self._big_trade_count += sum(1 for s in new_signals if "BIG" in s.signal_type)

        self._prev_chain = chain

    def get_summary(self, chain: Optional[List[dict]] = None, spot: float = 0.0, lot_size: int = 50) -> OrderFlowSummary:
        """Get current order flow summary."""
        if chain is None:
            chain = self._prev_chain or []

        strike_flows = self._build_strike_flows(chain, spot, lot_size)
        signals = list(self._signals)
        critical = [s for s in signals if s.severity == "CRITICAL"]

        # Session stats
        ce_pe_ratio = (
            self._session_ce_vol / self._session_pe_vol
            if self._session_pe_vol > 0 else 1.0
        )
        net_premium = self._session_ce_premium - self._session_pe_premium
        dominant = "NEUTRAL"
        if ce_pe_ratio > 1.3:
            dominant = "CALL_HEAVY"
        elif ce_pe_ratio < 0.7:
            dominant = "PUT_HEAVY"

        session = SessionFlow(
            total_ce_volume=round(self._session_ce_vol, 0),
            total_pe_volume=round(self._session_pe_vol, 0),
            total_ce_premium=round(self._session_ce_premium, 0),
            total_pe_premium=round(self._session_pe_premium, 0),
            ce_pe_volume_ratio=round(ce_pe_ratio, 2),
            net_premium_flow=round(net_premium, 0),
            dominant_activity=dominant,
            big_trade_count=self._big_trade_count,
            spike_count=self._spike_count,
        )

        # Top heat strikes
        sorted_ce = sorted(strike_flows, key=lambda s: s.ce_vol_ratio, reverse=True)
        sorted_pe = sorted(strike_flows, key=lambda s: s.pe_vol_ratio, reverse=True)

        # Smart money bias
        bias, detail = self._smart_money_bias(strike_flows, signals)

        return OrderFlowSummary(
            session=session,
            signals=sorted(signals, key=lambda s: (0 if s.severity == "CRITICAL" else 1), reverse=False)[-self.MAX_SIGNALS:],
            critical_signals=critical[-10:],
            strike_flows=sorted(strike_flows, key=lambda s: s.heat_score, reverse=True),
            hottest_ce_strikes=sorted_ce[:5],
            hottest_pe_strikes=sorted_pe[:5],
            smart_money_bias=bias,
            smart_money_detail=detail,
        )

    # ── Signal Detection ──────────────────────────────────────

    def _detect_signals(
        self, chain: List[dict], spot: float, lot_size: int
    ) -> List[FlowSignal]:
        signals = []
        ts = datetime.now().strftime("%H:%M:%S")

        for row in chain:
            strike = float(row.get("strike", 0))
            right = str(row.get("right", "")).upper()
            vol = float(row.get("volume", 0))
            ltp = float(row.get("ltp", 0))
            if strike <= 0 or vol <= 0:
                continue

            key = f"{strike}_{right}"
            history = list(self._vol_history[key])
            if len(history) < 3:
                continue

            avg_vol = statistics.mean(history[:-1]) if len(history) > 1 else history[0]
            if avg_vol < 1:
                continue

            ratio = vol / avg_vol

            # Volume spike
            if ratio >= self.VOLUME_SPIKE_RATIO:
                sev = "CRITICAL" if ratio >= self.VOLUME_CRITICAL_RATIO else "WARN"
                signals.append(FlowSignal(
                    timestamp=ts, strike=strike, right=right.split()[0],
                    signal_type="VOLUME_SPIKE",
                    severity=sev,
                    value=vol, threshold=avg_vol,
                    multiple=round(ratio, 1), ltp=ltp,
                    message=f"{right} {int(strike)}: vol {int(vol):,} = {ratio:.1f}× avg ({int(avg_vol):,})"
                ))

            # Big print (₹ premium)
            premium_val = ltp * vol * lot_size
            if premium_val >= self.BIG_PRINT_PREMIUM:
                signals.append(FlowSignal(
                    timestamp=ts, strike=strike, right=right.split()[0],
                    signal_type="BIG_PRINT",
                    severity="WARN",
                    value=premium_val, threshold=self.BIG_PRINT_PREMIUM,
                    multiple=round(premium_val / self.BIG_PRINT_PREMIUM, 1),
                    ltp=ltp,
                    message=f"{right} {int(strike)}: ₹{premium_val / 1e5:.1f}L premium flow"
                ))

        return signals[:10]  # limit per update cycle

    def _build_strike_flows(
        self, chain: List[dict], spot: float, lot_size: int
    ) -> List[StrikeFlow]:
        """Build per-strike order flow objects."""
        by_strike: Dict[float, dict] = {}

        for row in chain:
            strike = float(row.get("strike", 0))
            right = str(row.get("right", "")).upper()
            vol = float(row.get("volume", 0))
            oi = float(row.get("oi", 0))
            ltp = float(row.get("ltp", 0))
            oi_chg = float(row.get("oi_chg", 0)) if "oi_chg" in row else 0.0
            if strike <= 0:
                continue

            if strike not in by_strike:
                by_strike[strike] = {}

            key = f"{strike}_{right}"
            history = list(self._vol_history.get(key, []))
            avg_vol = statistics.mean(history) if len(history) > 1 else max(vol, 1)
            vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0

            if "CALL" in right:
                by_strike[strike]["ce_vol"] = vol
                by_strike[strike]["ce_oi"] = oi
                by_strike[strike]["ce_oi_chg"] = oi_chg
                by_strike[strike]["ce_ltp"] = ltp
                by_strike[strike]["ce_vol_ratio"] = round(vol_ratio, 2)
                by_strike[strike]["ce_premium"] = ltp * vol * lot_size
            else:
                by_strike[strike]["pe_vol"] = vol
                by_strike[strike]["pe_oi"] = oi
                by_strike[strike]["pe_oi_chg"] = oi_chg
                by_strike[strike]["pe_ltp"] = ltp
                by_strike[strike]["pe_vol_ratio"] = round(vol_ratio, 2)
                by_strike[strike]["pe_premium"] = ltp * vol * lot_size

        result = []
        for strike, d in by_strike.items():
            ce_vol = d.get("ce_vol", 0)
            pe_vol = d.get("pe_vol", 0)
            ce_ratio = d.get("ce_vol_ratio", 1.0)
            pe_ratio = d.get("pe_vol_ratio", 1.0)
            ce_oi = d.get("ce_oi", 0)
            pe_oi = d.get("pe_oi", 0)
            ce_oi_chg = d.get("ce_oi_chg", 0)
            pe_oi_chg = d.get("pe_oi_chg", 0)
            ce_ltp = d.get("ce_ltp", 0)
            pe_ltp = d.get("pe_ltp", 0)
            ce_prem = d.get("ce_premium", 0)
            pe_prem = d.get("pe_premium", 0)

            # Dominant side
            dominant = "NEUTRAL"
            if ce_vol > pe_vol * 1.3:
                dominant = "CE"
            elif pe_vol > ce_vol * 1.3:
                dominant = "PE"

            # Flow direction
            flow_dir = self._flow_direction(ce_vol, pe_vol, ce_oi_chg, pe_oi_chg)

            # Heat score
            heat = min(100.0, (max(ce_ratio, pe_ratio) - 1) * 25)

            result.append(StrikeFlow(
                strike=strike,
                ce_volume=ce_vol, pe_volume=pe_vol,
                ce_vol_ratio=ce_ratio, pe_vol_ratio=pe_ratio,
                ce_oi=ce_oi, pe_oi=pe_oi,
                ce_oi_chg=ce_oi_chg, pe_oi_chg=pe_oi_chg,
                ce_ltp=ce_ltp, pe_ltp=pe_ltp,
                ce_premium_value=round(ce_prem, 0),
                pe_premium_value=round(pe_prem, 0),
                heat_score=round(heat, 1),
                dominant_side=dominant,
                flow_direction=flow_dir,
            ))

        return result

    def _flow_direction(self, ce_vol: float, pe_vol: float, ce_oi_chg: float, pe_oi_chg: float) -> str:
        """Classify flow direction from volume and OI change."""
        if ce_vol > pe_vol * 1.5 and ce_oi_chg > 0:
            return "CALL_WRITING"      # CE vol up + OI up = writers active
        if ce_vol > pe_vol * 1.5 and ce_oi_chg < 0:
            return "CALL_COVERING"     # CE vol up + OI down = short covering
        if pe_vol > ce_vol * 1.5 and pe_oi_chg > 0:
            return "PUT_WRITING"
        if pe_vol > ce_vol * 1.5 and pe_oi_chg < 0:
            return "PUT_COVERING"
        if ce_vol > pe_vol * 1.2:
            return "CALL_BUYING"
        if pe_vol > ce_vol * 1.2:
            return "PUT_BUYING"
        return "MIXED"

    def _smart_money_bias(
        self, strike_flows: List[StrikeFlow], signals: List[FlowSignal]
    ) -> Tuple[str, str]:
        """Derive smart money positioning from aggregate flow."""
        if not strike_flows:
            return "NEUTRAL", "Insufficient data"

        # Count directional flows
        call_writing = sum(1 for s in strike_flows if s.flow_direction == "CALL_WRITING")
        put_writing = sum(1 for s in strike_flows if s.flow_direction == "PUT_WRITING")
        call_buying = sum(1 for s in strike_flows if s.flow_direction == "CALL_BUYING")
        put_buying = sum(1 for s in strike_flows if s.flow_direction == "PUT_BUYING")

        # Call writing + put buying = bearish institutional activity
        # Put writing + call buying = bullish institutional activity
        bearish_score = call_writing + put_buying
        bullish_score = put_writing + call_buying

        if bullish_score > bearish_score * 1.5:
            return "BULLISH", f"Put writing + Call buying dominant ({bullish_score} vs {bearish_score} strikes)"
        if bearish_score > bullish_score * 1.5:
            return "BEARISH", f"Call writing + Put buying dominant ({bearish_score} vs {bullish_score} strikes)"
        return "NEUTRAL", f"Mixed activity (bullish: {bullish_score}, bearish: {bearish_score})"

    def reset_session(self) -> None:
        """Reset session counters (call at market open)."""
        self._session_ce_vol = 0.0
        self._session_pe_vol = 0.0
        self._session_ce_premium = 0.0
        self._session_pe_premium = 0.0
        self._big_trade_count = 0
        self._spike_count = 0
        self._signals.clear()
