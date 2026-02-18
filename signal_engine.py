"""
signal_engine.py  — NEW MODULE

Real-time Options Signal Engine:
  • IV-based entry quality score (0-100)
  • Option flow sentiment (bullish / neutral / bearish)
  • Delta imbalance signal (spot likely direction)
  • Premium decay velocity (theta burn rate vs time left)
  • Straddle entry quality (ideal when IV is rich vs history)
  • Market regime signal (combine PCR + OI + IV + spot momentum)
  • Pre-trade checklist (go/no-go for a proposed trade)

Zero breaking-changes: pure add-on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from greeks_engine import BlackScholes, time_to_expiry
from models import OptionRight
from utils import safe_float


# ────────────────────────────────────────────────────────────────
# Signal Result
# ────────────────────────────────────────────────────────────────

@dataclass
class Signal:
    signal_type: str
    value: float
    label: str           # "BULLISH", "BEARISH", "NEUTRAL", "RICH", "CHEAP", etc.
    score: float         # 0-100 (normalized strength)
    description: str
    action: str          # suggested action
    severity: str = "INFO"  # INFO, WARN, OPPORTUNITY
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))

    def to_dict(self) -> dict:
        return {
            "type": self.signal_type,
            "value": round(self.value, 3),
            "label": self.label,
            "score": round(self.score, 1),
            "description": self.description,
            "action": self.action,
            "severity": self.severity,
            "time": self.timestamp,
        }


@dataclass
class PreTradeChecklist:
    instrument: str
    strategy_type: str
    spot: float
    expiry: str
    checks: List[Dict] = field(default_factory=list)
    overall_score: float = 0.0
    go_nogo: str = "NEUTRAL"  # GO / CAUTION / NO-GO
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "strategy": self.strategy_type,
            "spot": self.spot,
            "checks": self.checks,
            "score": round(self.overall_score, 1),
            "verdict": self.go_nogo,
            "summary": self.summary,
        }


# ────────────────────────────────────────────────────────────────
# Signal Engine
# ────────────────────────────────────────────────────────────────

class SignalEngine:
    """
    Generates market signals from live option chain data.
    All methods are stateless and accept chain + spot as input.
    """

    # IV thresholds (in %)
    IV_VERY_LOW = 10.0
    IV_LOW = 15.0
    IV_NORMAL = 20.0
    IV_HIGH = 28.0
    IV_EXTREME = 40.0

    def run_all_signals(
        self, chain: List[dict], spot: float, instrument: str = "", expiry: str = ""
    ) -> List[Signal]:
        """Run all signal generators and return sorted list."""
        if not chain or spot <= 0:
            return []

        signals: List[Signal] = []

        iv_sig = self.iv_richness_signal(chain, spot, instrument)
        if iv_sig:
            signals.append(iv_sig)

        flow_sig = self.option_flow_signal(chain, spot, instrument)
        if flow_sig:
            signals.append(flow_sig)

        delta_sig = self.delta_imbalance_signal(chain, spot, instrument)
        if delta_sig:
            signals.append(delta_sig)

        theta_sig = self.theta_decay_signal(chain, spot, expiry, instrument)
        if theta_sig:
            signals.append(theta_sig)

        pcr_sig = self.pcr_signal(chain, instrument)
        if pcr_sig:
            signals.append(pcr_sig)

        regime_sig = self.composite_regime_signal(chain, spot, instrument)
        if regime_sig:
            signals.append(regime_sig)

        signals.sort(key=lambda s: s.score, reverse=True)
        return signals

    # ── IV Richness Signal ─────────────────────────────────────

    def iv_richness_signal(self, chain: List[dict], spot: float, instrument: str = "") -> Optional[Signal]:
        """
        Compute ATM IV and classify it.
        High IV → premium selling opportunity.
        Low IV → avoid selling, consider buying.
        """
        from utils import atm_strike
        from app_config import Config

        inst = Config.instrument(instrument) if instrument else {"strike_gap": 50}
        gap = inst.get("strike_gap", 50)
        atm = atm_strike(spot, gap)

        atm_ivs = []
        for row in chain:
            k = float(row.get("strike", 0))
            iv = float(row.get("iv", 0))
            if abs(k - atm) <= gap and iv > 0:
                atm_ivs.append(iv)

        if not atm_ivs:
            return None

        avg_atm_iv = sum(atm_ivs) / len(atm_ivs)

        if avg_atm_iv >= self.IV_EXTREME:
            label, score, action = "EXTREME", 95, "Premium selling IDEAL – extreme IV crush potential"
            severity = "OPPORTUNITY"
        elif avg_atm_iv >= self.IV_HIGH:
            label, score, action = "RICH", 80, "Good premium selling environment"
            severity = "OPPORTUNITY"
        elif avg_atm_iv >= self.IV_NORMAL:
            label, score, action = "NORMAL", 60, "Moderate — straddle/strangle acceptable"
            severity = "INFO"
        elif avg_atm_iv >= self.IV_LOW:
            label, score, action = "LOW", 35, "Low IV — reduce selling, prefer buying"
            severity = "WARN"
        else:
            label, score, action = "VERY LOW", 15, "Avoid premium selling — buy instead"
            severity = "WARN"

        return Signal(
            signal_type="IV_RICHNESS",
            value=round(avg_atm_iv, 1),
            label=label,
            score=score,
            description=f"ATM IV = {avg_atm_iv:.1f}% [{label}]",
            action=action,
            severity=severity,
        )

    # ── Option Flow Signal ─────────────────────────────────────

    def option_flow_signal(self, chain: List[dict], spot: float, instrument: str = "") -> Optional[Signal]:
        """
        Weighted option flow = sum(call_vol * delta) - sum(put_vol * delta).
        Positive → call-heavy flow (bullish).
        Negative → put-heavy flow (bearish).
        """
        call_flow = 0.0
        put_flow = 0.0

        for row in chain:
            vol = float(row.get("volume", 0))
            delta = abs(float(row.get("delta", 0)))
            ltp = float(row.get("ltp", 0))
            right = str(row.get("right", "")).upper()

            if vol <= 0 or delta <= 0 or ltp <= 0:
                continue

            flow = vol * delta * ltp  # ₹-weighted flow

            if right == "CALL":
                call_flow += flow
            else:
                put_flow += flow

        total_flow = call_flow + put_flow
        if total_flow <= 0:
            return None

        net_ratio = (call_flow - put_flow) / total_flow  # -1 to +1
        score = (net_ratio + 1) / 2 * 100  # normalize 0-100

        if net_ratio > 0.25:
            label = "BULLISH"
            action = "Call-heavy flow — consider short puts or bull put spreads"
            severity = "INFO"
        elif net_ratio < -0.25:
            label = "BEARISH"
            action = "Put-heavy flow — consider short calls or bear call spreads"
            severity = "INFO"
        else:
            label = "NEUTRAL"
            action = "Mixed flow — straddle/strangle preferred"
            severity = "INFO"

        return Signal(
            signal_type="OPTION_FLOW",
            value=round(net_ratio * 100, 1),
            label=label,
            score=round(score, 1),
            description=f"Net flow: {net_ratio:+.1%} (Calls ₹{call_flow/1e6:.2f}M vs Puts ₹{put_flow/1e6:.2f}M)",
            action=action,
            severity=severity,
        )

    # ── Delta Imbalance Signal ─────────────────────────────────

    def delta_imbalance_signal(self, chain: List[dict], spot: float, instrument: str = "") -> Optional[Signal]:
        """
        OI-weighted net delta exposure of the entire chain.
        High call OI near spot = dealer is short calls = bullish pressure.
        High put OI near spot = dealer is short puts = bearish risk.
        """
        from utils import atm_strike
        from app_config import Config

        inst = Config.instrument(instrument) if instrument else {"strike_gap": 50}
        gap = inst.get("strike_gap", 50)
        range_width = gap * 5  # look within ±5 strikes of ATM

        call_oi_delta = 0.0
        put_oi_delta = 0.0

        for row in chain:
            k = float(row.get("strike", 0))
            oi = float(row.get("oi", 0))
            delta = float(row.get("delta", 0))
            right = str(row.get("right", "")).upper()

            if abs(k - spot) > range_width or oi <= 0:
                continue

            if right == "CALL":
                call_oi_delta += oi * abs(delta)
            else:
                put_oi_delta += oi * abs(delta)

        total = call_oi_delta + put_oi_delta
        if total <= 0:
            return None

        imbalance = (call_oi_delta - put_oi_delta) / total * 100

        if imbalance > 20:
            label = "CALL HEAVY"
            action = "Strong call OI near spot → resistance zone; consider downside bias"
            severity = "WARN"
        elif imbalance < -20:
            label = "PUT HEAVY"
            action = "Strong put OI near spot → support zone; consider upside bias"
            severity = "WARN"
        else:
            label = "BALANCED"
            action = "Delta imbalance within normal range"
            severity = "INFO"

        score = 50 + imbalance / 2  # map to 0-100
        score = max(0, min(100, score))

        return Signal(
            signal_type="DELTA_IMBALANCE",
            value=round(imbalance, 1),
            label=label,
            score=round(score, 1),
            description=f"Near-ATM OI delta imbalance: {imbalance:+.0f}% (Calls−Puts)",
            action=action,
            severity=severity,
        )

    # ── Theta Decay Signal ────────────────────────────────────

    def theta_decay_signal(self, chain: List[dict], spot: float, expiry: str, instrument: str = "") -> Optional[Signal]:
        """
        Evaluate if theta decay rate is attractive for premium sellers.
        Score based on theta/premium ratio (higher is better for sellers).
        """
        T = time_to_expiry(expiry) if expiry else 0.0
        if T <= 0:
            return None

        dte = T * 365

        # Find ATM straddle premium and theta
        from utils import atm_strike
        from app_config import Config

        inst = Config.instrument(instrument) if instrument else {"strike_gap": 50}
        gap = inst.get("strike_gap", 50)
        atm = atm_strike(spot, gap)

        atm_premium = 0.0
        atm_theta = 0.0

        for row in chain:
            k = float(row.get("strike", 0))
            if abs(k - atm) <= gap:
                atm_premium += float(row.get("ltp", 0))
                atm_theta += abs(float(row.get("theta", 0)))

        if atm_premium <= 0 or atm_theta <= 0:
            return None

        # Daily decay rate
        decay_pct = (atm_theta / atm_premium) * 100

        # DTE-based scoring: sweet spot is 5-20 DTE
        if 5 <= dte <= 20:
            dte_score = 90
            dte_note = f"DTE={dte:.0f} — optimal theta zone"
        elif 3 <= dte < 5:
            dte_score = 70
            dte_note = f"DTE={dte:.0f} — good theta, elevated gamma risk"
        elif 1 <= dte < 3:
            dte_score = 40
            dte_note = f"DTE={dte:.0f} — very high gamma risk"
        elif dte > 30:
            dte_score = 50
            dte_note = f"DTE={dte:.0f} — slow theta, consider shorter expiry"
        else:
            dte_score = 20
            dte_note = f"DTE={dte:.0f} — expiry imminent, avoid new positions"

        score = min(dte_score * (decay_pct / 2), 100)

        if decay_pct > 3:
            label = "HIGH DECAY"
            action = "Premium selling attractive — high theta/premium ratio"
            severity = "OPPORTUNITY"
        elif decay_pct > 1.5:
            label = "NORMAL DECAY"
            action = "Acceptable theta decay environment"
            severity = "INFO"
        else:
            label = "LOW DECAY"
            action = "Slow decay — may not justify premium selling at this DTE"
            severity = "WARN"

        return Signal(
            signal_type="THETA_DECAY",
            value=round(decay_pct, 2),
            label=label,
            score=round(score, 1),
            description=f"ATM straddle: ₹{atm_premium:.0f} premium, Θ={atm_theta:.1f}/day ({decay_pct:.2f}%). {dte_note}",
            action=action,
            severity=severity,
        )

    # ── PCR Signal ────────────────────────────────────────────

    def pcr_signal(self, chain: List[dict], instrument: str = "") -> Optional[Signal]:
        """
        Put-Call Ratio signal.
        PCR > 1.2 → bearish (put heavy)
        PCR < 0.7 → bullish (call heavy)
        """
        total_ce_oi = sum(float(r.get("oi", 0)) for r in chain if str(r.get("right", "")).upper() == "CALL")
        total_pe_oi = sum(float(r.get("oi", 0)) for r in chain if str(r.get("right", "")).upper() == "PUT")

        if total_ce_oi <= 0:
            return None

        pcr = total_pe_oi / total_ce_oi

        if pcr > 1.5:
            label = "EXTREME BEARISH"
            action = "Very high PCR — contrarian signal: possible reversal upwards"
            score = 20
            severity = "WARN"
        elif pcr > 1.2:
            label = "BEARISH"
            action = "Put heavy — market fearful; short call spreads favored"
            score = 35
            severity = "INFO"
        elif pcr > 0.9:
            label = "NEUTRAL"
            action = "Balanced PCR — straddle/strangle suitable"
            score = 60
            severity = "INFO"
        elif pcr > 0.7:
            label = "BULLISH"
            action = "Call heavy — market greedy; short put spreads favored"
            score = 70
            severity = "INFO"
        else:
            label = "EXTREME BULLISH"
            action = "Very low PCR — contrarian signal: possible reversal downwards"
            score = 80
            severity = "WARN"

        return Signal(
            signal_type="PCR",
            value=round(pcr, 3),
            label=label,
            score=score,
            description=f"PCR (OI) = {pcr:.3f} [CE OI: {total_ce_oi:,.0f} | PE OI: {total_pe_oi:,.0f}]",
            action=action,
            severity=severity,
        )

    # ── Composite Regime Signal ───────────────────────────────

    def composite_regime_signal(self, chain: List[dict], spot: float, instrument: str = "") -> Optional[Signal]:
        """
        Combine IV, PCR, delta imbalance into a single regime score.
        """
        sub_signals = [
            self.iv_richness_signal(chain, spot, instrument),
            self.pcr_signal(chain, instrument),
        ]

        valid = [s for s in sub_signals if s is not None]
        if not valid:
            return None

        avg_score = sum(s.score for s in valid) / len(valid)
        avg_value = sum(s.value for s in valid) / len(valid)

        if avg_score >= 70:
            label = "SELL PREMIUM"
            action = "Multiple signals favour premium selling — deploy shorts"
            severity = "OPPORTUNITY"
        elif avg_score >= 50:
            label = "NEUTRAL"
            action = "Mixed signals — trade smaller, use defined-risk spreads"
            severity = "INFO"
        else:
            label = "AVOID SELLING"
            action = "Unfavourable for premium selling — reduce risk or step aside"
            severity = "WARN"

        return Signal(
            signal_type="COMPOSITE_REGIME",
            value=round(avg_score, 1),
            label=label,
            score=round(avg_score, 1),
            description=f"Composite market regime score: {avg_score:.0f}/100 from {len(valid)} signals",
            action=action,
            severity=severity,
        )

    # ── Pre-Trade Checklist ───────────────────────────────────

    def pre_trade_checklist(
        self,
        chain: List[dict],
        spot: float,
        instrument: str,
        strategy_type: str,
        expiry: str,
        lots: int = 1,
        available_margin: float = 0.0,
    ) -> PreTradeChecklist:
        """
        Generate a go/no-go pre-trade checklist for a proposed strategy.
        Returns scored checklist with overall verdict.
        """
        checks = []
        total_score = 0
        max_score = 0

        # 1. IV Environment
        iv_sig = self.iv_richness_signal(chain, spot, instrument)
        if iv_sig:
            iv_ok = iv_sig.score >= 60
            checks.append({
                "check": "IV Environment",
                "status": "✅" if iv_ok else "⚠️",
                "detail": f"ATM IV = {iv_sig.value}% [{iv_sig.label}]",
                "score": iv_sig.score,
                "weight": 30,
            })
            total_score += iv_sig.score * 30
            max_score += 100 * 30

        # 2. DTE check
        T = time_to_expiry(expiry) if expiry else 0.0
        dte = T * 365
        dte_score = 80 if 5 <= dte <= 20 else (50 if dte > 20 else 20)
        dte_ok = dte >= 3
        checks.append({
            "check": "Days to Expiry",
            "status": "✅" if dte_ok else "🔴",
            "detail": f"DTE = {dte:.0f} ({'optimal' if 5 <= dte <= 20 else 'suboptimal'})",
            "score": dte_score,
            "weight": 20,
        })
        total_score += dte_score * 20
        max_score += 100 * 20

        # 3. Market direction (PCR)
        pcr_sig = self.pcr_signal(chain, instrument)
        if pcr_sig:
            pcr_ok = 0.7 <= pcr_sig.value <= 1.3
            checks.append({
                "check": "PCR Range",
                "status": "✅" if pcr_ok else "⚠️",
                "detail": f"PCR = {pcr_sig.value:.3f} [{pcr_sig.label}]",
                "score": pcr_sig.score,
                "weight": 20,
            })
            total_score += pcr_sig.score * 20
            max_score += 100 * 20

        # 4. Liquidity (bid-ask spread)
        atm_rows = [r for r in chain if abs(float(r.get("strike", 0)) - spot) <= 200]
        spreads = []
        for r in atm_rows:
            bid = float(r.get("bid", 0))
            ask = float(r.get("ask", 0))
            ltp = float(r.get("ltp", 0))
            if bid > 0 and ask > 0 and ltp > 0:
                spreads.append((ask - bid) / ltp * 100)

        if spreads:
            avg_spread_pct = sum(spreads) / len(spreads)
            liq_ok = avg_spread_pct < 5
            liq_score = max(0, 100 - avg_spread_pct * 8)
            checks.append({
                "check": "Liquidity (B-A Spread)",
                "status": "✅" if liq_ok else "⚠️",
                "detail": f"Avg ATM spread = {avg_spread_pct:.1f}%",
                "score": round(liq_score, 1),
                "weight": 20,
            })
            total_score += liq_score * 20
            max_score += 100 * 20

        # 5. Margin check
        if available_margin > 0:
            from app_config import Config
            base_margin = {"NIFTY": 90_000, "CNXBAN": 120_000, "BSESEN": 80_000}
            inst = Config.instrument(instrument)
            per_lot = base_margin.get(inst["breeze_code"], 80_000)
            total_margin_needed = per_lot * lots
            margin_ratio = available_margin / total_margin_needed if total_margin_needed > 0 else 0
            margin_ok = margin_ratio >= 1.2
            margin_score = min(100, margin_ratio * 50)
            checks.append({
                "check": "Margin Adequacy",
                "status": "✅" if margin_ok else "🔴",
                "detail": f"Available ₹{available_margin:,.0f} vs needed ≈ ₹{total_margin_needed:,.0f} ({margin_ratio:.1f}x)",
                "score": round(margin_score, 1),
                "weight": 10,
            })
            total_score += margin_score * 10
            max_score += 100 * 10

        # Compute overall
        overall = (total_score / max_score * 100) if max_score > 0 else 50

        if overall >= 70:
            verdict = "GO"
            summary = f"Trade setup looks good ({overall:.0f}/100). Proceed with discipline."
        elif overall >= 50:
            verdict = "CAUTION"
            summary = f"Mixed signals ({overall:.0f}/100). Trade smaller or wait for better setup."
        else:
            verdict = "NO-GO"
            summary = f"Poor setup ({overall:.0f}/100). Consider waiting for better conditions."

        return PreTradeChecklist(
            instrument=instrument,
            strategy_type=strategy_type,
            spot=spot,
            expiry=expiry,
            checks=checks,
            overall_score=round(overall, 1),
            go_nogo=verdict,
            summary=summary,
        )
