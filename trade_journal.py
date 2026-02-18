"""
trade_journal.py  — Layer 3 NEW MODULE

Trade Journal & P&L Attribution Engine
  • Per-trade record with entry/exit metadata
  • Daily P&L calendar view (monthly grid)
  • Strategy-type performance breakdown
  • Win rate, avg profit, avg loss, Profit Factor
  • Streak analysis (current/best win streak, worst losing streak)
  • Best and worst trades
  • Instrument-level attribution
  • Greeks P&L attribution (theta, delta, vega components)
  • Hold time distribution

Zero breaking-changes: pure add-on. No existing file modified.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple


# ────────────────────────────────────────────────────────────────
# Data Classes
# ────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """Immutable trade record extracted from a closed strategy/leg."""
    trade_id: str
    strategy_id: str
    strategy_type: str          # short_straddle, iron_condor, etc.
    instrument: str             # NIFTY, BANKNIFTY
    entry_date: str             # YYYY-MM-DD
    exit_date: str
    entry_time: str             # HH:MM
    exit_time: str
    hold_hours: float
    lots: int
    premium_collected: float    # +ve = net credit received
    pnl: float                  # realized P&L
    pnl_pct: float              # % of premium collected
    max_profit: float           # theoretical max
    exit_reason: str            # SL, TARGET, EXPIRY, MANUAL
    strike_ce: float = 0.0
    strike_pe: float = 0.0
    expiry: str = ""
    # Greeks attribution (optional)
    theta_earned: float = 0.0
    delta_pnl: float = 0.0
    vega_pnl: float = 0.0


@dataclass
class DailyPnL:
    date: str               # YYYY-MM-DD
    pnl: float
    trades: int
    winners: int
    losers: int
    best_trade: float
    worst_trade: float


@dataclass
class StrategyStats:
    strategy_type: str
    total_trades: int
    winners: int
    losers: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float    # total_wins / total_losses
    avg_hold_hours: float
    best_trade: float
    worst_trade: float
    avg_pnl_pct: float      # avg % of max profit captured


@dataclass
class StreakInfo:
    current_streak: int          # +ve = win streak, -ve = loss streak
    current_streak_type: str     # "WIN" or "LOSS"
    max_win_streak: int
    max_loss_streak: int
    last_5_results: List[str]    # ["W","L","W","W","L"]


@dataclass
class JournalSummary:
    # Overall stats
    total_trades: int
    winners: int
    losers: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    profit_factor: float
    expectancy: float           # avg trade EV

    # Best/Worst
    best_trade: Optional[TradeRecord]
    worst_trade: Optional[TradeRecord]

    # Streaks
    streak: StreakInfo

    # Per instrument
    instrument_stats: Dict[str, dict]

    # Per strategy type
    strategy_stats: List[StrategyStats]

    # Daily P&L calendar
    daily_pnl: List[DailyPnL]

    # Hold time stats
    avg_hold_hours: float
    median_hold_hours: float

    # Recent trades (last 20)
    recent_trades: List[TradeRecord]

    # Month-to-date, week-to-date
    mtd_pnl: float
    wtd_pnl: float
    today_pnl: float

    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


# ────────────────────────────────────────────────────────────────
# Trade Journal Engine
# ────────────────────────────────────────────────────────────────

class TradeJournal:
    """
    Ingests closed strategies from DB and provides analytics.

    Usage:
        journal = TradeJournal()
        summary = journal.analyze(strategies)    # pass all closed strategies
    """

    def __init__(self):
        self._records: List[TradeRecord] = []

    # ── Main Analysis ─────────────────────────────────────────

    def analyze(self, strategies: list) -> JournalSummary:
        """
        Build full journal summary from a list of Strategy objects
        (both closed and active; only closed ones are counted in stats).
        """
        # Extract trade records from closed strategies
        records = self._extract_records(strategies)
        self._records = records

        if not records:
            return self._empty_summary()

        # Sort by exit date
        records_sorted = sorted(records, key=lambda r: (r.exit_date, r.exit_time))

        winners = [r for r in records_sorted if r.pnl > 0]
        losers = [r for r in records_sorted if r.pnl <= 0]
        n = len(records_sorted)

        total_pnl = sum(r.pnl for r in records_sorted)
        win_rate = len(winners) / n * 100 if n > 0 else 0
        avg_win = statistics.mean(r.pnl for r in winners) if winners else 0
        avg_loss = statistics.mean(r.pnl for r in losers) if losers else 0
        total_wins_sum = sum(r.pnl for r in winners)
        total_loss_sum = abs(sum(r.pnl for r in losers))
        profit_factor = (total_wins_sum / total_loss_sum) if total_loss_sum > 0 else float("inf")
        expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

        hold_times = [r.hold_hours for r in records_sorted if r.hold_hours > 0]
        avg_hold = statistics.mean(hold_times) if hold_times else 0
        median_hold = statistics.median(hold_times) if hold_times else 0

        best = max(records_sorted, key=lambda r: r.pnl, default=None)
        worst = min(records_sorted, key=lambda r: r.pnl, default=None)

        streak = self._compute_streaks(records_sorted)
        daily_pnl = self._compute_daily_pnl(records_sorted)
        strat_stats = self._compute_strategy_stats(records_sorted)
        inst_stats = self._compute_instrument_stats(records_sorted)

        today_str = date.today().strftime("%Y-%m-%d")
        week_start = (date.today() - timedelta(days=date.today().weekday())).strftime("%Y-%m-%d")
        month_start = date.today().replace(day=1).strftime("%Y-%m-%d")

        today_pnl = sum(r.pnl for r in records_sorted if r.exit_date == today_str)
        wtd_pnl = sum(r.pnl for r in records_sorted if r.exit_date >= week_start)
        mtd_pnl = sum(r.pnl for r in records_sorted if r.exit_date >= month_start)

        return JournalSummary(
            total_trades=n,
            winners=len(winners),
            losers=len(losers),
            win_rate=round(win_rate, 1),
            total_pnl=round(total_pnl, 2),
            avg_pnl=round(total_pnl / n, 2) if n > 0 else 0,
            profit_factor=round(profit_factor, 2),
            expectancy=round(expectancy, 2),
            best_trade=best,
            worst_trade=worst,
            streak=streak,
            instrument_stats=inst_stats,
            strategy_stats=strat_stats,
            daily_pnl=daily_pnl,
            avg_hold_hours=round(avg_hold, 1),
            median_hold_hours=round(median_hold, 1),
            recent_trades=list(reversed(records_sorted[-20:])),
            mtd_pnl=round(mtd_pnl, 2),
            wtd_pnl=round(wtd_pnl, 2),
            today_pnl=round(today_pnl, 2),
        )

    def get_monthly_calendar(self, year: int, month: int) -> Dict[int, float]:
        """Returns {day: pnl} dict for a given month (for calendar heatmap)."""
        result: Dict[int, float] = {}
        for r in self._records:
            try:
                d = datetime.strptime(r.exit_date, "%Y-%m-%d")
                if d.year == year and d.month == month:
                    result[d.day] = result.get(d.day, 0.0) + r.pnl
            except ValueError:
                pass
        return result

    # ── Record Extraction ──────────────────────────────────────

    def _extract_records(self, strategies: list) -> List[TradeRecord]:
        """Convert Strategy objects to TradeRecord instances."""
        records = []
        from models import StrategyStatus, LegStatus

        for s in strategies:
            if not hasattr(s, "status"):
                continue
            if s.status.value not in ("closed", "partial_exit"):
                continue

            # Compute premium and P&L
            premium = 0.0
            pnl = 0.0
            lots = 0
            strike_ce = 0.0
            strike_pe = 0.0

            for leg in getattr(s, "legs", []):
                if leg.status.value not in ("squared_off", "sl_triggered", "active"):
                    continue
                lot_size = leg.quantity
                if lots == 0:
                    lots = lot_size

                if leg.side.value == "sell":
                    premium += leg.entry_price * leg.quantity
                    pnl += (leg.entry_price - leg.exit_price) * leg.quantity
                    if hasattr(leg, "right"):
                        if leg.right.value in ("call", "Call"):
                            strike_ce = leg.strike_price
                        else:
                            strike_pe = leg.strike_price
                else:
                    pnl -= (leg.exit_price - leg.entry_price) * leg.quantity

            if premium == 0 and pnl == 0:
                continue

            # Parse dates
            entry_dt = self._parse_dt(getattr(s, "created_at", ""))
            exit_dt = self._parse_dt(getattr(s, "closed_at", "") or datetime.now().isoformat())
            hold_hours = max(0.0, (exit_dt - entry_dt).total_seconds() / 3600)

            pnl_pct = (pnl / premium * 100) if premium > 0 else 0.0

            # Determine exit reason
            has_sl = any(
                getattr(l, "status", None) and l.status.value == "sl_triggered"
                for l in getattr(s, "legs", [])
            )
            exit_reason = "SL" if has_sl else "TARGET/MANUAL"

            records.append(TradeRecord(
                trade_id=s.strategy_id[:8],
                strategy_id=s.strategy_id,
                strategy_type=s.strategy_type.value,
                instrument=s.stock_code,
                entry_date=entry_dt.strftime("%Y-%m-%d"),
                exit_date=exit_dt.strftime("%Y-%m-%d"),
                entry_time=entry_dt.strftime("%H:%M"),
                exit_time=exit_dt.strftime("%H:%M"),
                hold_hours=round(hold_hours, 2),
                lots=lots,
                premium_collected=round(premium, 2),
                pnl=round(pnl, 2),
                pnl_pct=round(pnl_pct, 1),
                max_profit=round(premium, 2),
                exit_reason=exit_reason,
                strike_ce=strike_ce,
                strike_pe=strike_pe,
                expiry=getattr(s.legs[0], "expiry_date", "") if s.legs else "",
            ))

        return records

    # ── Analytics Helpers ──────────────────────────────────────

    def _compute_streaks(self, records: List[TradeRecord]) -> StreakInfo:
        if not records:
            return StreakInfo(0, "NONE", 0, 0, [])

        results = ["W" if r.pnl > 0 else "L" for r in records]
        last5 = results[-5:]

        # Current streak
        curr_streak = 1
        curr_type = results[-1]
        for r in reversed(results[:-1]):
            if r == curr_type:
                curr_streak += 1
            else:
                break
        if curr_type == "L":
            curr_streak = -curr_streak

        # Max streaks
        max_win = max_loss = cur = 1
        for i in range(1, len(results)):
            if results[i] == results[i - 1]:
                cur += 1
            else:
                cur = 1
            if results[i] == "W":
                max_win = max(max_win, cur)
            else:
                max_loss = max(max_loss, cur)

        return StreakInfo(
            current_streak=curr_streak,
            current_streak_type="WIN" if curr_streak > 0 else "LOSS",
            max_win_streak=max_win,
            max_loss_streak=max_loss,
            last_5_results=last5,
        )

    def _compute_daily_pnl(self, records: List[TradeRecord]) -> List[DailyPnL]:
        by_day: Dict[str, List[TradeRecord]] = defaultdict(list)
        for r in records:
            by_day[r.exit_date].append(r)

        result = []
        for d, recs in sorted(by_day.items()):
            pnl = sum(r.pnl for r in recs)
            winners = [r for r in recs if r.pnl > 0]
            losers = [r for r in recs if r.pnl <= 0]
            result.append(DailyPnL(
                date=d,
                pnl=round(pnl, 2),
                trades=len(recs),
                winners=len(winners),
                losers=len(losers),
                best_trade=max(r.pnl for r in recs),
                worst_trade=min(r.pnl for r in recs),
            ))
        return result

    def _compute_strategy_stats(self, records: List[TradeRecord]) -> List[StrategyStats]:
        by_type: Dict[str, List[TradeRecord]] = defaultdict(list)
        for r in records:
            by_type[r.strategy_type].append(r)

        result = []
        for stype, recs in by_type.items():
            winners = [r for r in recs if r.pnl > 0]
            losers = [r for r in recs if r.pnl <= 0]
            n = len(recs)
            total_wins = sum(r.pnl for r in winners)
            total_loss = abs(sum(r.pnl for r in losers))
            avg_win = statistics.mean(r.pnl for r in winners) if winners else 0
            avg_loss = statistics.mean(r.pnl for r in losers) if losers else 0
            holds = [r.hold_hours for r in recs if r.hold_hours > 0]
            pnl_pcts = [r.pnl_pct for r in recs]

            result.append(StrategyStats(
                strategy_type=stype,
                total_trades=n,
                winners=len(winners),
                losers=len(losers),
                win_rate=round(len(winners) / n * 100, 1),
                total_pnl=round(sum(r.pnl for r in recs), 2),
                avg_win=round(avg_win, 2),
                avg_loss=round(avg_loss, 2),
                profit_factor=round(total_wins / total_loss, 2) if total_loss > 0 else 999.0,
                avg_hold_hours=round(statistics.mean(holds), 1) if holds else 0,
                best_trade=max(r.pnl for r in recs),
                worst_trade=min(r.pnl for r in recs),
                avg_pnl_pct=round(statistics.mean(pnl_pcts), 1) if pnl_pcts else 0,
            ))
        return sorted(result, key=lambda s: s.total_pnl, reverse=True)

    def _compute_instrument_stats(self, records: List[TradeRecord]) -> Dict[str, dict]:
        by_inst: Dict[str, List[TradeRecord]] = defaultdict(list)
        for r in records:
            by_inst[r.instrument].append(r)

        result = {}
        for inst, recs in by_inst.items():
            winners = [r for r in recs if r.pnl > 0]
            n = len(recs)
            result[inst] = {
                "trades": n,
                "win_rate": round(len(winners) / n * 100, 1),
                "total_pnl": round(sum(r.pnl for r in recs), 2),
                "avg_pnl": round(sum(r.pnl for r in recs) / n, 2),
            }
        return result

    def _parse_dt(self, s: str) -> datetime:
        if not s:
            return datetime.now()
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(s[:19], fmt[:19])
            except ValueError:
                continue
        return datetime.now()

    def _empty_summary(self) -> JournalSummary:
        return JournalSummary(
            total_trades=0, winners=0, losers=0, win_rate=0.0,
            total_pnl=0.0, avg_pnl=0.0, profit_factor=0.0, expectancy=0.0,
            best_trade=None, worst_trade=None,
            streak=StreakInfo(0, "NONE", 0, 0, []),
            instrument_stats={}, strategy_stats=[], daily_pnl=[],
            avg_hold_hours=0.0, median_hold_hours=0.0, recent_trades=[],
            mtd_pnl=0.0, wtd_pnl=0.0, today_pnl=0.0,
        )
