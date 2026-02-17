# ═══════════════════════════════════════════════════════════════
# FILE: analytics.py  (NEW — the brain Sensibull has and we didn't)
# ═══════════════════════════════════════════════════════════════
"""
Market analytics computed from option chain and historical data:
  • Max Pain — strike where option writers lose least
  • PCR — Put-Call Ratio by OI and volume
  • IV Percentile & Rank — current IV vs historical context
  • Expected Move — ATM straddle implied range
  • Probability of Profit — per-strategy PoP
  • IV Skew — smile/skew across strikes
  • Support/Resistance from OI — highest OI CE/PE strikes
"""

import math
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from scipy.stats import norm

from app_config import Config
from greeks_engine import BlackScholes, time_to_expiry
from models import Leg, Strategy, OptionRight, OrderSide, LegStatus
from utils import LOG, safe_float


@dataclass
class MarketAnalytics:
    """Snapshot of all analytics for an instrument."""
    max_pain: float = 0.0
    max_pain_data: List[Dict] = field(default_factory=list)
    pcr_oi: float = 0.0
    pcr_volume: float = 0.0
    total_ce_oi: float = 0.0
    total_pe_oi: float = 0.0
    highest_ce_oi_strike: float = 0.0  # resistance
    highest_pe_oi_strike: float = 0.0  # support
    expected_move: float = 0.0
    expected_move_pct: float = 0.0
    atm_iv: float = 0.0
    iv_percentile: float = 0.0
    iv_rank: float = 0.0
    iv_skew: List[Dict] = field(default_factory=list)
    spot: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class AnalyticsEngine:
    """Computes all analytics from chain data + historical data."""

    def __init__(self, session, state):
        self.session = session
        self.state = state
        self._hist_cache: Dict[str, List[Dict]] = {}
        self._hist_cache_time: Dict[str, float] = {}
        self._iv_history: Dict[str, List[float]] = {}

    def compute_all(self, chain_data: List[dict], spot: float,
                    instrument: str, expiry: str) -> MarketAnalytics:
        """Single entry point — compute everything from one chain snapshot."""
        a = MarketAnalytics(spot=spot)
        if not chain_data or spot <= 0:
            return a

        # Parse chain into CE/PE maps
        ce_map, pe_map = self._parse_chain(chain_data)
        all_strikes = sorted(set(list(ce_map.keys()) + list(pe_map.keys())))

        if not all_strikes:
            return a

        # ── Max Pain ─────────────────────────────────────────
        a.max_pain, a.max_pain_data = self._compute_max_pain(
            all_strikes, ce_map, pe_map
        )

        # ── PCR ──────────────────────────────────────────────
        a.total_ce_oi = sum(v.get("oi", 0) for v in ce_map.values())
        a.total_pe_oi = sum(v.get("oi", 0) for v in pe_map.values())
        a.pcr_oi = round(a.total_pe_oi / a.total_ce_oi, 3) if a.total_ce_oi > 0 else 0

        total_ce_vol = sum(v.get("volume", 0) for v in ce_map.values())
        total_pe_vol = sum(v.get("volume", 0) for v in pe_map.values())
        a.pcr_volume = round(total_pe_vol / total_ce_vol, 3) if total_ce_vol > 0 else 0

        # ── OI Support/Resistance ────────────────────────────
        if ce_map:
            a.highest_ce_oi_strike = max(ce_map, key=lambda k: ce_map[k].get("oi", 0))
        if pe_map:
            a.highest_pe_oi_strike = max(pe_map, key=lambda k: pe_map[k].get("oi", 0))

        # ── ATM IV + Expected Move ───────────────────────────
        gap = Config.strike_gap(instrument)
        from utils import atm_strike
        atm = atm_strike(spot, gap)

        T = time_to_expiry(expiry)
        atm_ce_ltp = ce_map.get(atm, {}).get("ltp", 0)
        atm_pe_ltp = pe_map.get(atm, {}).get("ltp", 0)

        if atm_ce_ltp > 0:
            a.atm_iv = BlackScholes.implied_vol(
                atm_ce_ltp, spot, atm, T, Config.RISK_FREE_RATE, OptionRight.CALL
            )
        elif atm_pe_ltp > 0:
            a.atm_iv = BlackScholes.implied_vol(
                atm_pe_ltp, spot, atm, T, Config.RISK_FREE_RATE, OptionRight.PUT
            )

        straddle_price = atm_ce_ltp + atm_pe_ltp
        a.expected_move = round(straddle_price, 2)
        a.expected_move_pct = round(straddle_price / spot * 100, 2) if spot > 0 else 0

        # ── IV Skew ──────────────────────────────────────────
        a.iv_skew = self._compute_iv_skew(ce_map, pe_map, all_strikes, spot, T)

        # ── IV Percentile & Rank ─────────────────────────────
        a.iv_percentile, a.iv_rank = self._compute_iv_context(
            instrument, a.atm_iv
        )

        # Track IV over time for this session
        self._track_iv(instrument, a.atm_iv)

        return a

    def _parse_chain(self, chain: List[dict]) -> Tuple[Dict, Dict]:
        ce, pe = {}, {}
        for item in chain:
            s = item.get("strike", 0)
            if s <= 0:
                continue
            entry = {
                "ltp": item.get("ltp", 0), "bid": item.get("bid", 0),
                "ask": item.get("ask", 0), "oi": item.get("oi", 0),
                "volume": item.get("volume", 0), "iv": item.get("iv", 0),
                "delta": item.get("delta", 0),
            }
            if item.get("right", "").upper() == "CALL":
                ce[s] = entry
            else:
                pe[s] = entry
        return ce, pe

    def _compute_max_pain(self, strikes, ce_map, pe_map):
        """
        Standard max pain algorithm:
        For each potential expiry price P:
          pain = Σ(max(P-K,0)*CE_OI[K]) + Σ(max(K-P,0)*PE_OI[K])
        Max pain = P with minimum total pain.
        """
        pain_data = []
        min_pain = float("inf")
        mp_strike = strikes[len(strikes) // 2] if strikes else 0

        for P in strikes:
            total = 0.0
            for K in strikes:
                ce_oi = ce_map.get(K, {}).get("oi", 0)
                pe_oi = pe_map.get(K, {}).get("oi", 0)
                if P > K and ce_oi > 0:
                    total += (P - K) * ce_oi
                if P < K and pe_oi > 0:
                    total += (K - P) * pe_oi
            pain_data.append({"strike": P, "pain": round(total / 1e6, 2)})
            if total < min_pain:
                min_pain = total
                mp_strike = P

        return mp_strike, pain_data

    def _compute_iv_skew(self, ce_map, pe_map, strikes, spot, T):
        skew = []
        r = Config.RISK_FREE_RATE
        for K in strikes:
            entry = {"strike": K}
            if K in ce_map and ce_map[K]["ltp"] > 0:
                iv = BlackScholes.implied_vol(
                    ce_map[K]["ltp"], spot, K, T, r, OptionRight.CALL
                )
                entry["ce_iv"] = round(iv * 100, 2)
            else:
                entry["ce_iv"] = 0
            if K in pe_map and pe_map[K]["ltp"] > 0:
                iv = BlackScholes.implied_vol(
                    pe_map[K]["ltp"], spot, K, T, r, OptionRight.PUT
                )
                entry["pe_iv"] = round(iv * 100, 2)
            else:
                entry["pe_iv"] = 0
            skew.append(entry)
        return skew

    def _compute_iv_context(self, instrument: str, current_iv: float):
        """
        IV Percentile and Rank using historical realized vol as proxy.
        Professional platforms use actual historical IV — we approximate
        with close-to-close realized vol from underlying OHLC.
        """
        if current_iv <= 0:
            return 0.0, 0.0

        hist = self._get_historical_data(instrument)
        if not hist or len(hist) < 30:
            return 50.0, 0.5

        # Compute 20-day rolling realized vol
        closes = [safe_float(d.get("close", 0)) for d in hist if safe_float(d.get("close", 0)) > 0]
        if len(closes) < 30:
            return 50.0, 0.5

        log_returns = [math.log(closes[i] / closes[i - 1])
                       for i in range(1, len(closes)) if closes[i - 1] > 0]

        window = 20
        rolling_vols = []
        for i in range(window, len(log_returns)):
            chunk = log_returns[i - window:i]
            rv = math.sqrt(252 * sum(r ** 2 for r in chunk) / len(chunk))
            rolling_vols.append(rv)

        if not rolling_vols:
            return 50.0, 0.5

        # IV Percentile: % of days where realized vol < current IV
        below = sum(1 for v in rolling_vols if v < current_iv)
        iv_pct = round(below / len(rolling_vols) * 100, 1)

        # IV Rank: (current - min) / (max - min)
        mn, mx = min(rolling_vols), max(rolling_vols)
        iv_rank = round((current_iv - mn) / (mx - mn), 3) if mx > mn else 0.5

        return iv_pct, iv_rank

    def _get_historical_data(self, instrument: str) -> List[Dict]:
        """Fetch and cache 252 days of underlying OHLC."""
        import time as _time
        inst = Config.instrument(instrument)
        cache_key = inst["breeze_code"]

        # Check cache (refresh every 24 hours)
        if (cache_key in self._hist_cache
                and _time.time() - self._hist_cache_time.get(cache_key, 0) < 86400):
            return self._hist_cache[cache_key]

        try:
            to_dt = datetime.now()
            from_dt = to_dt - timedelta(days=365)
            cash_code = inst.get("cash_code", inst["breeze_code"])
            cash_exc = inst.get("cash_exchange", "NSE")

            r = self.session.breeze.get_historical_data(
                interval="1day",
                from_date=from_dt.strftime("%Y-%m-%dT06:00:00.000Z"),
                to_date=to_dt.strftime("%Y-%m-%dT06:00:00.000Z"),
                stock_code=cash_code,
                exchange_code=cash_exc,
                product_type="cash",
            )
            if r and r.get("Status") == 200 and r.get("Success"):
                data = r["Success"]
                self._hist_cache[cache_key] = data
                self._hist_cache_time[cache_key] = _time.time()
                return data
        except Exception as e:
            LOG.error(f"Historical data fetch: {e}")
        return self._hist_cache.get(cache_key, [])

    def _track_iv(self, instrument: str, iv: float):
        if iv <= 0:
            return
        key = instrument
        if key not in self._iv_history:
            self._iv_history[key] = []
        self._iv_history[key].append(iv)
        if len(self._iv_history[key]) > 5000:
            self._iv_history[key] = self._iv_history[key][-2500:]

    def get_iv_history(self, instrument: str) -> List[float]:
        return self._iv_history.get(instrument, [])

    # ── Probability of Profit ────────────────────────────────

    @staticmethod
    def probability_of_profit(strategy: Strategy, spot: float) -> float:
        """
        Compute PoP for a strategy using log-normal distribution.
        For short straddle/strangle: PoP = P(lower_be < S_T < upper_be)
        """
        active = [l for l in strategy.legs if l.status in (LegStatus.ACTIVE, LegStatus.ENTERING)]
        if not active or spot <= 0:
            return 0.0

        # Compute breakevens
        total_premium = sum(l.entry_price * l.quantity for l in active if l.side == OrderSide.SELL)
        total_premium -= sum(l.entry_price * l.quantity for l in active if l.side == OrderSide.BUY)

        if total_premium <= 0:
            return 0.0

        per_unit_premium = total_premium / active[0].quantity if active[0].quantity > 0 else 0
        sell_strikes = [l.strike_price for l in active if l.side == OrderSide.SELL]
        buy_strikes = [l.strike_price for l in active if l.side == OrderSide.BUY]

        if not sell_strikes:
            return 0.0

        # For straddle: both strikes same
        # For strangle: CE strike > PE strike
        ce_legs = [l for l in active if l.right == OptionRight.CALL and l.side == OrderSide.SELL]
        pe_legs = [l for l in active if l.right == OptionRight.PUT and l.side == OrderSide.SELL]

        if ce_legs and pe_legs:
            ce_strike = ce_legs[0].strike_price
            pe_strike = pe_legs[0].strike_price
            ce_prem = ce_legs[0].entry_price
            pe_prem = pe_legs[0].entry_price

            # Account for bought wings (iron condor)
            buy_ce = [l for l in active if l.right == OptionRight.CALL and l.side == OrderSide.BUY]
            buy_pe = [l for l in active if l.right == OptionRight.PUT and l.side == OrderSide.BUY]
            net_ce = ce_prem - (buy_ce[0].entry_price if buy_ce else 0)
            net_pe = pe_prem - (buy_pe[0].entry_price if buy_pe else 0)

            upper_be = ce_strike + net_ce + net_pe
            lower_be = pe_strike - net_ce - net_pe
        else:
            return 50.0

        # Use ATM IV for the distribution
        T = time_to_expiry(active[0].expiry_date)
        if T <= 0:
            return 50.0

        iv = active[0].greeks.iv if active[0].greeks.iv > 0 else 0.15
        sigma_sqrt_t = iv * math.sqrt(T)

        if sigma_sqrt_t <= 0:
            return 50.0

        d_upper = (math.log(upper_be / spot)) / sigma_sqrt_t
        d_lower = (math.log(lower_be / spot)) / sigma_sqrt_t

        pop = (norm.cdf(d_upper) - norm.cdf(d_lower)) * 100
        return round(max(0, min(100, pop)), 1)
