"""
performance_report.py  — NEW MODULE

Performance Analytics & Report Generator
  • Equity curve with drawdown overlay
  • Rolling win rate / profit factor
  • Sharpe / Sortino / Calmar ratios
  • Monthly P&L heatmap (calendar)
  • Trade duration analysis
  • Strategy-type breakdown
  • Best / worst trades analysis
  • CSV / JSON export

Zero breaking-changes: pure add-on.
"""

from __future__ import annotations

import math
import json
import csv
import io
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

from utils import LOG


# ────────────────────────────────────────────────────────────────
# Data types
# ────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    trade_id: str
    strategy_type: str
    instrument: str
    entry_date: str
    exit_date: str
    entry_spot: float
    exit_spot: float
    lots: int
    pnl: float
    max_loss_hit: bool
    duration_minutes: float
    exit_reason: str
    premium_collected: float = 0.0


@dataclass
class PerformanceMetrics:
    # Basic
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # P&L
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_pnl: float = 0.0
    expectancy: float = 0.0

    # Risk
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_consecutive_losses: int = 0
    max_runup: float = 0.0

    # Ratios
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0

    # Duration
    avg_duration_minutes: float = 0.0
    avg_win_duration: float = 0.0
    avg_loss_duration: float = 0.0

    # Streaks
    current_streak: int = 0
    best_streak: int = 0
    worst_streak: int = 0

    # Premium analysis (for short sellers)
    total_premium_collected: float = 0.0
    avg_premium_kept_pct: float = 0.0
    sl_hit_rate: float = 0.0


# ────────────────────────────────────────────────────────────────
# Performance Report Engine
# ────────────────────────────────────────────────────────────────

class PerformanceReport:
    """
    Computes comprehensive performance analytics from trade records.

    Usage:
        report = PerformanceReport()
        metrics = report.compute(trades)
        equity_curve = report.equity_curve(trades)
        monthly_grid = report.monthly_pnl_heatmap(trades)
    """

    def __init__(self, risk_free_rate: float = 0.07):
        self.r = risk_free_rate

    # ── Primary Metrics ───────────────────────────────────────

    def compute(self, trades: List[TradeRecord]) -> PerformanceMetrics:
        """Full metrics computation from trade list."""
        m = PerformanceMetrics()
        if not trades:
            return m

        m.total_trades = len(trades)
        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        m.winning_trades = len(wins)
        m.losing_trades = len(losses)
        m.win_rate = (m.winning_trades / m.total_trades * 100) if m.total_trades > 0 else 0

        m.total_pnl = sum(pnls)
        m.gross_profit = sum(wins)
        m.gross_loss = abs(sum(losses))
        m.profit_factor = (m.gross_profit / m.gross_loss) if m.gross_loss > 0 else float("inf")

        m.avg_win = sum(wins) / len(wins) if wins else 0
        m.avg_loss = abs(sum(losses) / len(losses)) if losses else 0
        m.avg_pnl = m.total_pnl / m.total_trades
        m.expectancy = (m.win_rate / 100 * m.avg_win) - ((1 - m.win_rate / 100) * m.avg_loss)

        # Drawdown
        m.max_drawdown, m.max_drawdown_pct, m.max_runup = self._compute_drawdown(pnls)

        # Streaks
        m.max_consecutive_losses, m.best_streak, m.worst_streak, m.current_streak = self._compute_streaks(pnls)

        # Risk ratios (using daily P&L as return series)
        m.sharpe, m.sortino = self._compute_ratios(pnls, m.total_pnl)

        # Calmar = CAGR / max_drawdown
        if m.max_drawdown != 0 and m.total_pnl > 0:
            m.calmar = abs(m.total_pnl / m.max_drawdown)

        # Duration
        durations = [t.duration_minutes for t in trades if t.duration_minutes > 0]
        win_dur = [t.duration_minutes for t in trades if t.pnl > 0 and t.duration_minutes > 0]
        loss_dur = [t.duration_minutes for t in trades if t.pnl <= 0 and t.duration_minutes > 0]

        m.avg_duration_minutes = sum(durations) / len(durations) if durations else 0
        m.avg_win_duration = sum(win_dur) / len(win_dur) if win_dur else 0
        m.avg_loss_duration = sum(loss_dur) / len(loss_dur) if loss_dur else 0

        # Premium analysis
        premiums = [t.premium_collected for t in trades if t.premium_collected > 0]
        if premiums:
            m.total_premium_collected = sum(premiums)
            kept = [pnl / prem for pnl, prem in zip(pnls, [t.premium_collected for t in trades]) if prem > 0]
            m.avg_premium_kept_pct = (sum(kept) / len(kept) * 100) if kept else 0

        sl_hits = sum(1 for t in trades if t.max_loss_hit)
        m.sl_hit_rate = (sl_hits / m.total_trades * 100) if m.total_trades > 0 else 0

        return m

    # ── Equity Curve ──────────────────────────────────────────

    def equity_curve(self, trades: List[TradeRecord]) -> List[dict]:
        """
        Cumulative P&L curve with drawdown.
        Returns [{date, cumulative_pnl, drawdown, trade_pnl}, ...]
        """
        if not trades:
            return []

        sorted_trades = sorted(trades, key=lambda t: t.exit_date or t.entry_date)

        curve = []
        running = 0.0
        peak = 0.0

        for t in sorted_trades:
            running += t.pnl
            peak = max(peak, running)
            drawdown = running - peak

            curve.append({
                "date": (t.exit_date or t.entry_date)[:10],
                "cumulative_pnl": round(running, 0),
                "drawdown": round(drawdown, 0),
                "trade_pnl": round(t.pnl, 0),
                "trade_id": t.trade_id,
                "strategy_type": t.strategy_type,
            })

        return curve

    # ── Monthly P&L Heatmap ───────────────────────────────────

    def monthly_pnl_heatmap(self, trades: List[TradeRecord]) -> dict:
        """
        Monthly P&L calendar grid.
        Returns {years: [int], months: [str], grid: [[pnl|None]], totals_by_year: dict}
        """
        if not trades:
            return {}

        # Group by month
        monthly: Dict[Tuple[int, int], float] = {}
        for t in trades:
            date_str = (t.exit_date or t.entry_date)[:7]  # YYYY-MM
            try:
                dt = datetime.strptime(date_str, "%Y-%m")
                key = (dt.year, dt.month)
                monthly[key] = monthly.get(key, 0.0) + t.pnl
            except Exception:
                continue

        if not monthly:
            return {}

        years = sorted(set(k[0] for k in monthly))
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        grid = []
        totals_by_year = {}

        for year in years:
            row = []
            year_total = 0.0
            for month_idx in range(1, 13):
                val = monthly.get((year, month_idx))
                row.append(round(val, 0) if val is not None else None)
                if val is not None:
                    year_total += val
            grid.append(row)
            totals_by_year[year] = round(year_total, 0)

        return {
            "years": years,
            "months": months,
            "grid": grid,        # grid[year_idx][month_idx]
            "totals_by_year": totals_by_year,
        }

    # ── Strategy Breakdown ────────────────────────────────────

    def strategy_breakdown(self, trades: List[TradeRecord]) -> List[dict]:
        """Per-strategy-type performance summary."""
        by_type: Dict[str, List[TradeRecord]] = {}
        for t in trades:
            by_type.setdefault(t.strategy_type, []).append(t)

        result = []
        for stype, type_trades in by_type.items():
            m = self.compute(type_trades)
            result.append({
                "strategy_type": stype,
                "trades": m.total_trades,
                "win_rate": round(m.win_rate, 1),
                "total_pnl": round(m.total_pnl, 0),
                "profit_factor": round(m.profit_factor, 2),
                "avg_pnl": round(m.avg_pnl, 0),
                "max_drawdown": round(m.max_drawdown, 0),
                "sharpe": round(m.sharpe, 2),
            })

        result.sort(key=lambda x: x["total_pnl"], reverse=True)
        return result

    # ── Best / Worst Trades ───────────────────────────────────

    def best_worst(self, trades: List[TradeRecord], n: int = 5) -> dict:
        """Return best and worst N trades."""
        sorted_by_pnl = sorted(trades, key=lambda t: t.pnl, reverse=True)
        return {
            "best": [self._trade_to_dict(t) for t in sorted_by_pnl[:n]],
            "worst": [self._trade_to_dict(t) for t in sorted_by_pnl[-n:]],
        }

    # ── Rolling Analytics ─────────────────────────────────────

    def rolling_win_rate(self, trades: List[TradeRecord], window: int = 10) -> List[dict]:
        """Rolling N-trade win rate for trend analysis."""
        if len(trades) < window:
            return []

        sorted_trades = sorted(trades, key=lambda t: t.exit_date or "")
        result = []

        for i in range(window - 1, len(sorted_trades)):
            window_trades = sorted_trades[i - window + 1:i + 1]
            wins = sum(1 for t in window_trades if t.pnl > 0)
            wr = wins / window * 100
            result.append({
                "date": (sorted_trades[i].exit_date or "")[:10],
                "rolling_win_rate": round(wr, 1),
                "rolling_pnl": round(sum(t.pnl for t in window_trades), 0),
            })

        return result

    def duration_analysis(self, trades: List[TradeRecord]) -> dict:
        """Trade duration distribution and P&L by duration bucket."""
        if not trades:
            return {}

        buckets = {
            "< 1h": (0, 60),
            "1h-4h": (60, 240),
            "4h-1d": (240, 1440),
            "1d-3d": (1440, 4320),
            "> 3d": (4320, float("inf")),
        }

        result = {}
        for label, (lo, hi) in buckets.items():
            bucket_trades = [t for t in trades if lo <= t.duration_minutes < hi]
            if bucket_trades:
                pnls = [t.pnl for t in bucket_trades]
                result[label] = {
                    "count": len(bucket_trades),
                    "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1),
                    "avg_pnl": round(sum(pnls) / len(pnls), 0),
                    "total_pnl": round(sum(pnls), 0),
                }

        return result

    # ── Export ────────────────────────────────────────────────

    def to_csv(self, trades: List[TradeRecord]) -> str:
        """Export trade list to CSV string."""
        output = io.StringIO()
        if not trades:
            return ""

        fieldnames = [
            "trade_id", "strategy_type", "instrument", "entry_date", "exit_date",
            "entry_spot", "exit_spot", "lots", "pnl", "premium_collected",
            "exit_reason", "duration_minutes"
        ]

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for t in trades:
            writer.writerow({
                "trade_id": t.trade_id,
                "strategy_type": t.strategy_type,
                "instrument": t.instrument,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "entry_spot": t.entry_spot,
                "exit_spot": t.exit_spot,
                "lots": t.lots,
                "pnl": round(t.pnl, 2),
                "premium_collected": round(t.premium_collected, 2),
                "exit_reason": t.exit_reason,
                "duration_minutes": round(t.duration_minutes, 0),
            })

        return output.getvalue()

    def to_json(self, metrics: PerformanceMetrics, trades: List[TradeRecord]) -> str:
        """Export full report as JSON."""
        from dataclasses import asdict
        report = {
            "generated_at": datetime.now().isoformat(),
            "metrics": asdict(metrics),
            "equity_curve": self.equity_curve(trades),
            "monthly_pnl": self.monthly_pnl_heatmap(trades),
            "strategy_breakdown": self.strategy_breakdown(trades),
            "best_worst": self.best_worst(trades, 3),
        }
        return json.dumps(report, indent=2, default=str)

    # ── Private ───────────────────────────────────────────────

    def _compute_drawdown(self, pnls: List[float]) -> Tuple[float, float, float]:
        """Max drawdown (absolute, %), max runup."""
        running = 0.0
        peak = 0.0
        max_dd = 0.0
        max_runup = 0.0
        initial = 500_000.0  # assumed starting capital for pct

        for pnl in pnls:
            running += pnl
            peak = max(peak, running)
            dd = running - peak
            max_dd = min(max_dd, dd)
            max_runup = max(max_runup, running)

        max_dd_pct = (max_dd / (initial + max_runup)) * 100 if (initial + max_runup) > 0 else 0
        return max_dd, max_dd_pct, max_runup

    def _compute_streaks(self, pnls: List[float]) -> Tuple[int, int, int, int]:
        """Returns (max_consecutive_losses, best_streak, worst_streak, current_streak)."""
        max_loss_streak = 0
        best = 0
        worst = 0
        current = 0
        curr_loss = 0

        for pnl in pnls:
            if pnl > 0:
                current = current + 1 if current > 0 else 1
                best = max(best, current)
                curr_loss = 0
            else:
                current = current - 1 if current < 0 else -1
                worst = min(worst, current)
                curr_loss += 1
                max_loss_streak = max(max_loss_streak, curr_loss)

        return max_loss_streak, best, abs(worst), current

    def _compute_ratios(self, pnls: List[float], total_pnl: float) -> Tuple[float, float]:
        """Sharpe and Sortino ratios (simplified, no benchmark)."""
        if len(pnls) < 3:
            return 0.0, 0.0

        n = len(pnls)
        mean = sum(pnls) / n
        variance = sum((p - mean) ** 2 for p in pnls) / (n - 1)
        std = math.sqrt(variance) if variance > 0 else 1

        # Daily risk-free return assumption
        daily_rf = 0.0

        sharpe = (mean - daily_rf) / std if std > 0 else 0
        sharpe_annual = sharpe * math.sqrt(252)

        # Sortino: only downside deviation
        downside_pnls = [p for p in pnls if p < 0]
        if downside_pnls:
            ds_var = sum(p**2 for p in downside_pnls) / len(downside_pnls)
            ds_std = math.sqrt(ds_var)
            sortino = (mean - daily_rf) / ds_std if ds_std > 0 else 0
            sortino_annual = sortino * math.sqrt(252)
        else:
            sortino_annual = float("inf")

        return round(sharpe_annual, 2), round(sortino_annual, 2)

    def _trade_to_dict(self, t: TradeRecord) -> dict:
        return {
            "trade_id": t.trade_id,
            "type": t.strategy_type,
            "instrument": t.instrument,
            "pnl": round(t.pnl, 0),
            "entry_date": t.entry_date[:10],
            "exit_date": t.exit_date[:10],
            "duration_h": round(t.duration_minutes / 60, 1),
            "exit_reason": t.exit_reason,
        }
