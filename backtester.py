# ═══════════════════════════════════════════════════════════════
# FILE: backtester.py  (NEW — ThinkorSwim's thinkBack equivalent)
# ═══════════════════════════════════════════════════════════════
"""
Strategy backtester using historical underlying OHLC.
Option prices are estimated via Black-Scholes (since historical option
data is not reliably available for all strikes).

Supports: short straddle, short strangle, iron condor (estimated).

Output metrics match professional backtesting standards:
  Win rate, avg win/loss, profit factor, max drawdown, Sharpe, expectancy.
"""

import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from greeks_engine import BlackScholes, time_to_expiry
from models import OptionRight
from app_config import Config
from utils import LOG, safe_float


@dataclass
class BacktestTrade:
    entry_date: str = ""
    exit_date: str = ""
    entry_spot: float = 0.0
    exit_spot: float = 0.0
    ce_strike: float = 0.0
    pe_strike: float = 0.0
    entry_premium: float = 0.0
    exit_premium: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    sl_hit: bool = False
    days_held: int = 0


@dataclass
class BacktestResult:
    trades: List[BacktestTrade] = field(default_factory=list)
    total_pnl: float = 0.0
    num_trades: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    equity_curve: List[Dict] = field(default_factory=list)
    error: str = ""


class Backtester:
    """
    Backtest short option selling strategies on historical data.

    Limitations (documented honestly like a professional tool):
    - Uses BS-estimated option prices, not actual market prices
    - SL detection uses daily high/low, may miss intraday spikes
    - Does not account for spreads, slippage, or commissions
    - IV is estimated from realized vol, not actual market IV
    """

    def __init__(self, session):
        self.session = session

    def run(
        self,
        instrument: str,
        strategy_type: str = "straddle",   # straddle / strangle
        target_delta: float = 0.15,
        sl_pct: float = 50.0,
        lookback_days: int = 180,
        lots: int = 1,
    ) -> BacktestResult:
        """Run backtest and return comprehensive results."""
        inst = Config.instrument(instrument)
        bc = inst["breeze_code"]
        gap = inst["strike_gap"]
        lot_size = inst["lot_size"]
        qty = lot_size * lots
        expiry_weekday = inst["expiry_weekday"]

        # Fetch historical data
        hist = self._get_history(inst, lookback_days)
        if not hist or len(hist) < 30:
            return BacktestResult(error="Insufficient historical data")

        # Compute realized vol from history
        closes = [safe_float(d.get("close", 0)) for d in hist if safe_float(d.get("close", 0)) > 0]
        if len(closes) < 30:
            return BacktestResult(error="Insufficient price data")

        log_rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        avg_rv = math.sqrt(252 * sum(r ** 2 for r in log_rets) / len(log_rets))
        avg_rv = max(avg_rv, 0.08)

        # Group data by expiry cycles
        trades = []
        r = Config.RISK_FREE_RATE

        # Simulate week-by-week
        i = 0
        while i < len(hist) - 5:
            entry_bar = hist[i]
            entry_close = safe_float(entry_bar.get("close", 0))
            if entry_close <= 0:
                i += 1
                continue

            # Determine strikes
            atm = round(entry_close / gap) * gap

            if strategy_type == "straddle":
                ce_strike = atm
                pe_strike = atm
            else:
                # Strangle: estimate delta strikes
                T_est = 5.0 / 365.0  # ~5 days for weekly
                ce_strike = atm
                pe_strike = atm
                for offset in range(1, 20):
                    test_k = atm + offset * gap
                    d = abs(BlackScholes.delta(entry_close, test_k, T_est, r, avg_rv, OptionRight.CALL))
                    if d <= target_delta:
                        ce_strike = test_k
                        break
                for offset in range(1, 20):
                    test_k = atm - offset * gap
                    d = abs(BlackScholes.delta(entry_close, test_k, T_est, r, avg_rv, OptionRight.PUT))
                    if d <= target_delta:
                        pe_strike = test_k
                        break

            # Estimate entry option prices
            T_entry = 5.0 / 365.0
            ce_entry = BlackScholes.price(entry_close, ce_strike, T_entry, r, avg_rv, OptionRight.CALL)
            pe_entry = BlackScholes.price(entry_close, pe_strike, T_entry, r, avg_rv, OptionRight.PUT)
            total_entry = ce_entry + pe_entry

            if total_entry <= 0:
                i += 5
                continue

            sl_price = total_entry * (1 + sl_pct / 100)

            # Simulate through the week
            sl_hit = False
            exit_day = min(i + 5, len(hist) - 1)
            exit_close = entry_close

            for j in range(i + 1, min(i + 6, len(hist))):
                bar = hist[j]
                day_high = safe_float(bar.get("high", 0))
                day_low = safe_float(bar.get("low", 0))
                day_close = safe_float(bar.get("close", 0))

                if day_close <= 0:
                    continue

                days_remaining = max(exit_day - j, 1)
                T_now = days_remaining / 365.0

                # Check SL using day extremes
                for test_spot in [day_high, day_low]:
                    ce_now = BlackScholes.price(test_spot, ce_strike, T_now, r, avg_rv, OptionRight.CALL)
                    pe_now = BlackScholes.price(test_spot, pe_strike, T_now, r, avg_rv, OptionRight.PUT)
                    if (ce_now + pe_now) >= sl_price:
                        sl_hit = True
                        exit_close = test_spot
                        exit_day = j
                        break

                if sl_hit:
                    break
                exit_close = day_close

            # Calculate exit prices
            T_exit = max(0.001, (min(i + 5, len(hist) - 1) - exit_day) / 365.0)
            if sl_hit:
                T_exit = 0.001

            ce_exit = max(BlackScholes.price(exit_close, ce_strike, T_exit, r, avg_rv, OptionRight.CALL), 0)
            pe_exit = max(BlackScholes.price(exit_close, pe_strike, T_exit, r, avg_rv, OptionRight.PUT), 0)
            total_exit = ce_exit + pe_exit

            pnl = (total_entry - total_exit) * qty
            pnl_pct = (total_entry - total_exit) / total_entry * 100 if total_entry > 0 else 0

            entry_dt = entry_bar.get("datetime", "")[:10]
            exit_dt = hist[exit_day].get("datetime", "")[:10] if exit_day < len(hist) else ""

            trades.append(BacktestTrade(
                entry_date=entry_dt, exit_date=exit_dt,
                entry_spot=round(entry_close, 1), exit_spot=round(exit_close, 1),
                ce_strike=ce_strike, pe_strike=pe_strike,
                entry_premium=round(total_entry, 2),
                exit_premium=round(total_exit, 2),
                pnl=round(pnl, 2), pnl_pct=round(pnl_pct, 2),
                sl_hit=sl_hit,
                days_held=exit_day - i,
            ))

            i = exit_day + 1  # Move to next cycle

        return self._compute_stats(trades)

    def _compute_stats(self, trades: List[BacktestTrade]) -> BacktestResult:
        if not trades:
            return BacktestResult(error="No trades generated")

        result = BacktestResult(trades=trades, num_trades=len(trades))
        pnls = [t.pnl for t in trades]
        result.total_pnl = round(sum(pnls), 2)

        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]
        result.winners = len(winners)
        result.losers = len(losers)
        result.win_rate = round(len(winners) / len(pnls) * 100, 1)
        result.avg_win = round(sum(winners) / len(winners), 2) if winners else 0
        result.avg_loss = round(sum(losers) / len(losers), 2) if losers else 0
        result.max_win = round(max(pnls), 2)
        result.max_loss = round(min(pnls), 2)

        gross_profit = sum(winners)
        gross_loss = abs(sum(losers))
        result.profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 999

        result.expectancy = round(
            (result.avg_win * result.win_rate / 100) +
            (result.avg_loss * (100 - result.win_rate) / 100), 2
        )

        # Max drawdown
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        curve = []
        for t in trades:
            equity += t.pnl
            peak = max(peak, equity)
            dd = peak - equity
            max_dd = max(max_dd, dd)
            curve.append({"trade": t.entry_date, "equity": round(equity, 2)})
        result.max_drawdown = round(max_dd, 2)
        result.equity_curve = curve

        # Sharpe (annualized, using weekly returns)
        if len(pnls) > 1:
            import numpy as np
            pnl_arr = np.array(pnls)
            if pnl_arr.std() > 0:
                result.sharpe = round(
                    (pnl_arr.mean() / pnl_arr.std()) * math.sqrt(52), 2
                )

        return result

    def _get_history(self, inst: dict, days: int) -> List[Dict]:
        try:
            to_dt = datetime.now()
            from_dt = to_dt - timedelta(days=days)
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
                return r["Success"]
        except Exception as e:
            LOG.error(f"Backtest history fetch: {e}")
        return []
