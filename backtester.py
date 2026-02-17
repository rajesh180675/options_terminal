# ═══════════════════════════════════════════════════════════════
# FILE: backtester.py  (NEW)
# ═══════════════════════════════════════════════════════════════
"""
Historical strategy backtester.
  • Simulates straddle/strangle selling over date range
  • Uses historical underlying prices + BS model
  • Tracks SL events, P&L per trade, drawdown curve
  • Works with mock data or Breeze historical API
"""

import math
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from greeks_engine import BlackScholes
from models import OptionRight
from app_config import Config
from utils import LOG, safe_float, atm_strike


@dataclass
class BacktestTrade:
    entry_date: str
    exit_date: str
    entry_spot: float
    exit_spot: float
    ce_strike: float
    pe_strike: float
    ce_entry_price: float
    pe_entry_price: float
    ce_exit_price: float
    pe_exit_price: float
    pnl: float
    exit_reason: str  # "expiry", "sl_hit", "target"
    dte_at_entry: int


@dataclass
class BacktestResult:
    params: Dict = field(default_factory=dict)
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    total_pnl: float = 0
    win_rate: float = 0
    profit_factor: float = 0
    max_drawdown: float = 0
    total_trades: int = 0
    avg_pnl: float = 0
    sharpe: float = 0


class Backtester:
    """
    Backtest options selling strategies using:
      1. Historical underlying prices (from API or synthetic)
      2. Black-Scholes model for option pricing
      3. Configurable SL, entry DTE, delta target
    """

    def __init__(self, session_manager=None):
        self.session = session_manager

    def run(
        self,
        instrument: str = "NIFTY",
        strategy: str = "strangle",  # "straddle" or "strangle"
        start_date: str = "",
        end_date: str = "",
        entry_dte: int = 5,          # enter N days before expiry
        target_delta: float = 0.15,  # for strangle
        sl_pct: float = 50.0,
        iv_assumption: float = 0.15,
        lots: int = 1,
    ) -> BacktestResult:
        """Run the backtest and return results."""

        inst = Config.instrument(instrument)
        lot_size = inst["lot_size"]
        gap = inst["strike_gap"]
        quantity = lot_size * lots
        r = Config.RISK_FREE_RATE

        # Generate date range
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Get historical prices
        prices = self._get_historical_prices(
            inst["breeze_code"], inst["exchange"], start_date, end_date
        )

        if len(prices) < 30:
            LOG.warning("Insufficient historical data for backtest")
            return BacktestResult(params={"error": "Insufficient data"})

        # Generate expiry dates
        expiry_weekday = inst["expiry_weekday"]
        expiries = self._generate_expiries(start_date, end_date, expiry_weekday)

        trades = []
        equity_curve = [0.0]

        for expiry in expiries:
            entry_date = expiry - timedelta(days=entry_dte)

            # Find closest trading day prices
            entry_idx = self._find_closest_date(prices, entry_date)
            expiry_idx = self._find_closest_date(prices, expiry)

            if entry_idx is None or expiry_idx is None:
                continue
            if entry_idx >= expiry_idx:
                continue

            entry_spot = prices[entry_idx]["close"]
            T_entry = entry_dte / 365.0
            iv = iv_assumption

            # Determine strikes
            if strategy == "straddle":
                ce_strike = atm_strike(entry_spot, gap)
                pe_strike = ce_strike
            else:  # strangle
                ce_strike, pe_strike = self._find_delta_strikes_bs(
                    entry_spot, gap, T_entry, r, iv, target_delta
                )

            # Price options at entry
            ce_entry = BlackScholes.price(entry_spot, ce_strike, T_entry, r, iv, OptionRight.CALL)
            pe_entry = BlackScholes.price(entry_spot, pe_strike, T_entry, r, iv, OptionRight.PUT)

            total_premium = (ce_entry + pe_entry) * quantity
            sl_threshold_ce = ce_entry * (1 + sl_pct / 100)
            sl_threshold_pe = pe_entry * (1 + sl_pct / 100)

            # Simulate daily from entry to expiry
            exit_reason = "expiry"
            exit_spot = prices[expiry_idx]["close"]
            ce_exit = max(exit_spot - ce_strike, 0)
            pe_exit = max(pe_strike - exit_spot, 0)

            for day_idx in range(entry_idx + 1, expiry_idx + 1):
                day_spot = prices[day_idx]["close"]
                days_left = (expiry_idx - day_idx)
                T_now = max(days_left / 365.0, 1e-6)

                day_ce = BlackScholes.price(day_spot, ce_strike, T_now, r, iv, OptionRight.CALL)
                day_pe = BlackScholes.price(day_spot, pe_strike, T_now, r, iv, OptionRight.PUT)

                if day_ce >= sl_threshold_ce or day_pe >= sl_threshold_pe:
                    exit_reason = "sl_hit"
                    ce_exit = day_ce
                    pe_exit = day_pe
                    exit_spot = day_spot
                    break

            pnl = ((ce_entry - ce_exit) + (pe_entry - pe_exit)) * quantity

            trades.append(BacktestTrade(
                entry_date=prices[entry_idx]["date"],
                exit_date=prices[min(expiry_idx, len(prices)-1)]["date"],
                entry_spot=round(entry_spot, 2),
                exit_spot=round(exit_spot, 2),
                ce_strike=ce_strike, pe_strike=pe_strike,
                ce_entry_price=round(ce_entry, 2),
                pe_entry_price=round(pe_entry, 2),
                ce_exit_price=round(ce_exit, 2),
                pe_exit_price=round(pe_exit, 2),
                pnl=round(pnl, 2),
                exit_reason=exit_reason,
                dte_at_entry=entry_dte,
            ))

            running = equity_curve[-1] + pnl
            equity_curve.append(round(running, 2))

        # Compute statistics
        return self._compute_stats(trades, equity_curve, {
            "instrument": instrument, "strategy": strategy,
            "entry_dte": entry_dte, "target_delta": target_delta,
            "sl_pct": sl_pct, "iv": iv_assumption, "lots": lots,
            "period": f"{start_date} to {end_date}",
        })

    def _get_historical_prices(self, breeze_code, exchange, start, end):
        """Get daily OHLC. Uses API if live, synthetic if mock."""
        if self.session and Config.is_live():
            try:
                r = self.session.breeze.get_historical_data(
                    interval="1day", from_date=f"{start}T06:00:00.000Z",
                    to_date=f"{end}T06:00:00.000Z",
                    stock_code=breeze_code,
                    exchange_code="NSE" if exchange == "NFO" else "BSE",
                    product_type="cash",
                )
                if r and r.get("Success"):
                    return [
                        {"date": d.get("datetime", "")[:10],
                         "close": safe_float(d.get("close", 0))}
                        for d in r["Success"] if safe_float(d.get("close", 0)) > 0
                    ]
            except Exception as e:
                LOG.error(f"Historical data fetch: {e}")

        # Synthetic data fallback
        return self._generate_synthetic(breeze_code, start, end)

    def _generate_synthetic(self, code, start, end):
        """Generate GBM synthetic daily prices."""
        spots = {"NIFTY": 22000, "CNXBAN": 48000, "BSESEN": 72000}
        S = spots.get(code, 22000)
        dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")

        prices = []
        vol = 0.15
        while dt <= end_dt:
            if dt.weekday() < 5:  # Trading days only
                daily_ret = random.gauss(0.0003, vol / math.sqrt(252))
                S *= math.exp(daily_ret)
                prices.append({"date": dt.strftime("%Y-%m-%d"), "close": round(S, 2)})
            dt += timedelta(days=1)
        return prices

    def _generate_expiries(self, start, end, weekday):
        """Generate weekly expiry dates."""
        dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        expiries = []
        while dt <= end_dt:
            days_ahead = weekday - dt.weekday()
            if days_ahead < 0: days_ahead += 7
            exp = dt + timedelta(days=days_ahead)
            if exp not in expiries and exp <= end_dt:
                expiries.append(exp)
            dt += timedelta(days=7)
        return expiries

    def _find_closest_date(self, prices, target_date):
        target_str = target_date.strftime("%Y-%m-%d") if isinstance(target_date, datetime) else str(target_date)
        for i, p in enumerate(prices):
            if p["date"] >= target_str:
                return i
        return None

    def _find_delta_strikes_bs(self, spot, gap, T, r, iv, target_delta):
        """Find OTM CE and PE strikes closest to target delta."""
        best_ce, best_ce_d = spot, 99
        best_pe, best_pe_d = spot, 99

        for offset in range(1, 20):
            ce_s = atm_strike(spot, gap) + offset * gap
            pe_s = atm_strike(spot, gap) - offset * gap

            cd = abs(abs(BlackScholes.delta(spot, ce_s, T, r, iv, OptionRight.CALL)) - target_delta)
            pd = abs(abs(BlackScholes.delta(spot, pe_s, T, r, iv, OptionRight.PUT)) - target_delta)

            if cd < best_ce_d: best_ce_d, best_ce = cd, ce_s
            if pd < best_pe_d: best_pe_d, best_pe = pd, pe_s

        return best_ce, best_pe

    def _compute_stats(self, trades, equity_curve, params):
        if not trades:
            return BacktestResult(params=params)

        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # Max drawdown
        peak = 0
        max_dd = 0
        for v in equity_curve:
            if v > peak: peak = v
            dd = peak - v
            if dd > max_dd: max_dd = dd

        # Sharpe (annualized, assuming weekly trades)
        if len(pnls) > 1:
            mean_pnl = np.mean(pnls)
            std_pnl = np.std(pnls)
            sharpe = (mean_pnl / max(std_pnl, 1)) * math.sqrt(52)
        else:
            sharpe = 0

        return BacktestResult(
            params=params, trades=trades,
            equity_curve=equity_curve,
            total_pnl=round(sum(pnls), 2),
            win_rate=round(len(wins) / max(len(pnls), 1) * 100, 1),
            profit_factor=round(
                sum(wins) / max(abs(sum(losses)), 1), 2
            ) if losses else 99.0,
            max_drawdown=round(max_dd, 2),
            total_trades=len(trades),
            avg_pnl=round(np.mean(pnls), 2) if pnls else 0,
            sharpe=round(sharpe, 2),
        )
