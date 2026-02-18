"""
pages/04_Performance.py  — NEW PAGE

Performance Analytics Report
  • Equity curve with drawdown
  • Monthly P&L heatmap (calendar grid)
  • Strategy breakdown table
  • Rolling win rate / profit factor
  • Trade duration analysis
  • Best / worst trades
  • Position sizing recommendations
  • Export (CSV / JSON)
"""

import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Performance | Options Terminal",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Performance Analytics")


@st.cache_resource
def boot_engine():
    from trading_engine import TradingEngine
    eng = TradingEngine()
    eng.start()
    return eng

engine = boot_engine()

try:
    from performance_report import PerformanceReport, TradeRecord
    from position_sizer import PositionSizer
    from regime_detector import RegimeDetector
    from app_config import Config, INSTRUMENTS
    from utils import breeze_expiry, next_weekly_expiry
    ok = True
except Exception as e:
    st.error(f"Module load error: {e}")
    ok = False

if not ok:
    st.stop()

perf_engine = PerformanceReport(risk_free_rate=Config.RISK_FREE_RATE)
sizer = PositionSizer()
regime_detector = RegimeDetector()

# ── Load trade data from journal ──────────────────────────────────
trades: list[TradeRecord] = []

if getattr(engine, "journal", None):
    try:
        journal_entries = engine.journal.get_entries(500)
        for e in journal_entries:
            t = TradeRecord(
                trade_id=getattr(e, "strategy_id", ""),
                strategy_type=getattr(e, "strategy_type", ""),
                instrument=getattr(e, "stock_code", "NIFTY"),
                entry_date=getattr(e, "entry_date", ""),
                exit_date=getattr(e, "exit_date", ""),
                entry_spot=float(getattr(e, "entry_spot", 0)),
                exit_spot=float(getattr(e, "exit_spot", 0)),
                lots=int(getattr(e, "lots_count", 1)),
                pnl=float(getattr(e, "pnl", 0)),
                max_loss_hit=str(getattr(e, "exit_reason", "")).lower() in ("sl_hit", "panic"),
                duration_minutes=float(getattr(e, "duration_minutes", 0)),
                exit_reason=str(getattr(e, "exit_reason", "")),
                premium_collected=float(getattr(e, "premium_collected", 0)),
            )
            trades.append(t)
    except Exception as ex:
        st.warning(f"Journal load error: {ex}")

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    if trades:
        all_types = list(set(t.strategy_type for t in trades))
        sel_types = st.multiselect("Strategy Types", all_types, default=all_types)
        all_insts = list(set(t.instrument for t in trades))
        sel_insts = st.multiselect("Instruments", all_insts, default=all_insts)

        min_date = min((t.exit_date or t.entry_date)[:10] for t in trades if (t.exit_date or t.entry_date))
        max_date = max((t.exit_date or t.entry_date)[:10] for t in trades if (t.exit_date or t.entry_date))

        date_range = st.date_input(
            "Date Range",
            value=(datetime.strptime(min_date, "%Y-%m-%d") if min_date else datetime.now(),
                   datetime.now()),
        )

        trades = [
            t for t in trades
            if t.strategy_type in sel_types
            and t.instrument in sel_insts
        ]

    st.divider()
    st.subheader("Position Sizer")
    sizer_inst = st.selectbox("Instrument", list(INSTRUMENTS.keys()), key="sizer_inst")
    sizer_capital = st.number_input("Available Capital (₹)", value=500_000, step=50_000)
    sizer_iv = st.slider("Current ATM IV%", 5, 50, 15, key="sizer_iv") / 100


# ── Demo data if no journal ───────────────────────────────────────
if not trades:
    st.info("No journal entries found. Showing demo data to preview analytics.")
    import random
    random.seed(42)
    demo_trades = []
    from datetime import timedelta
    base = datetime(2025, 1, 1)
    running_spot = 24000.0

    for i in range(80):
        dt = base + timedelta(days=i * 4.5)
        spot_move = random.gauss(0, 200)
        running_spot += spot_move
        pnl = random.gauss(1200, 3000)
        premium = random.uniform(2000, 8000)
        demo_trades.append(TradeRecord(
            trade_id=f"demo_{i:04d}",
            strategy_type=random.choice(["short_straddle", "short_strangle", "iron_condor"]),
            instrument=random.choice(["NIFTY", "BANKNIFTY"]),
            entry_date=dt.strftime("%Y-%m-%d"),
            exit_date=(dt + timedelta(days=random.randint(1, 7))).strftime("%Y-%m-%d"),
            entry_spot=running_spot,
            exit_spot=running_spot + spot_move * 0.5,
            lots=random.randint(1, 3),
            pnl=round(pnl, 0),
            max_loss_hit=(pnl < -3000),
            duration_minutes=random.uniform(60, 10000),
            exit_reason=random.choice(["sl_hit", "expiry", "profit_target", "manual"]),
            premium_collected=round(premium, 0),
        ))

    trades = demo_trades

# ── Compute metrics ───────────────────────────────────────────────
metrics = perf_engine.compute(trades)

# ── Summary cards ─────────────────────────────────────────────────
st.subheader("Overall Performance")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Trades", metrics.total_trades)
c2.metric("Win Rate", f"{metrics.win_rate:.1f}%",
          delta=f"{metrics.win_rate-50:.1f}% vs 50%")
c3.metric("Total P&L", f"₹{metrics.total_pnl:+,.0f}")
c4.metric("Profit Factor", f"{metrics.profit_factor:.2f}",
          delta="Good" if metrics.profit_factor > 1.5 else "Needs improvement")
c5.metric("Sharpe", f"{metrics.sharpe:.2f}")
c6.metric("Max Drawdown", f"₹{metrics.max_drawdown:,.0f}")

c7, c8, c9, c10, c11, c12 = st.columns(6)
c7.metric("Avg Win", f"₹{metrics.avg_win:,.0f}")
c8.metric("Avg Loss", f"₹{metrics.avg_loss:,.0f}")
c9.metric("Expectancy", f"₹{metrics.expectancy:+,.0f}")
c10.metric("Sortino", f"{metrics.sortino:.2f}")
c11.metric("SL Hit Rate", f"{metrics.sl_hit_rate:.1f}%")
c12.metric("Best Streak", f"{metrics.best_streak} wins")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────
t_curve, t_monthly, t_breakdown, t_rolling, t_duration, t_trades, t_sizer, t_regime = st.tabs([
    "Equity Curve", "Monthly P&L", "Strategy Breakdown",
    "Rolling", "Duration", "Trade Log", "Position Sizer", "Regime"
])

# ── Tab: Equity Curve ─────────────────────────────────────────────
with t_curve:
    st.subheader("Equity Curve & Drawdown")
    curve = perf_engine.equity_curve(trades)
    if curve:
        curve_df = pd.DataFrame(curve).set_index("date")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.line_chart(curve_df[["cumulative_pnl"]], height=280, color=["#00ff88"])
            st.caption("Cumulative P&L")

            st.line_chart(curve_df[["drawdown"]], height=180, color=["#ff4444"])
            st.caption("Drawdown (₹)")

        with col2:
            st.metric("Peak P&L", f"₹{metrics.max_runup:+,.0f}")
            st.metric("Max Drawdown", f"₹{metrics.max_drawdown:,.0f}")
            st.metric("Recovery", "Yes" if metrics.total_pnl > metrics.max_drawdown else "In DD")
            st.metric("Calmar Ratio", f"{metrics.calmar:.2f}")


# ── Tab: Monthly P&L ──────────────────────────────────────────────
with t_monthly:
    st.subheader("Monthly P&L Calendar")
    monthly = perf_engine.monthly_pnl_heatmap(trades)

    if monthly and monthly.get("grid"):
        years = monthly["years"]
        months = monthly["months"]
        grid = monthly["grid"]
        totals = monthly["totals_by_year"]

        for y_idx, year in enumerate(years):
            st.markdown(f"**{year}** — Total: ₹{totals.get(year, 0):+,.0f}")
            row_data = {}
            for m_idx, month in enumerate(months):
                val = grid[y_idx][m_idx]
                row_data[month] = val

            month_df = pd.DataFrame([row_data])

            def color_monthly(val):
                if val is None:
                    return "background-color: #1a1a2e; color: #555"
                if val > 5000:
                    return "background-color: #0d4d2b; color: #00ff88; font-weight: bold"
                if val > 0:
                    return "background-color: #0d3d1a; color: #00cc66"
                if val > -5000:
                    return "background-color: #3d0d0d; color: #ff8888"
                return "background-color: #4d0d0d; color: #ff4444; font-weight: bold"

            st.dataframe(
                month_df.style.applymap(color_monthly).format("{:+,.0f}", na_rep="—"),
                use_container_width=True,
                hide_index=True,
                height=70,
            )


# ── Tab: Strategy Breakdown ───────────────────────────────────────
with t_breakdown:
    st.subheader("Performance by Strategy Type")
    breakdown = perf_engine.strategy_breakdown(trades)
    if breakdown:
        bdf = pd.DataFrame(breakdown)
        st.dataframe(bdf, use_container_width=True, hide_index=True)

        best_type = max(breakdown, key=lambda x: x["total_pnl"])["strategy_type"]
        st.success(f"Best performing strategy: **{best_type.replace('_', ' ').upper()}**")

        # P&L bar chart
        bdf_chart = bdf.set_index("strategy_type")[["total_pnl"]]
        st.bar_chart(bdf_chart, height=220)


# ── Tab: Rolling Analytics ────────────────────────────────────────
with t_rolling:
    st.subheader("Rolling Analytics")
    window = st.slider("Rolling Window (trades)", 5, 30, 10)
    rolling = perf_engine.rolling_win_rate(trades, window)
    if rolling:
        rdf = pd.DataFrame(rolling).set_index("date")
        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(rdf[["rolling_win_rate"]], height=220)
            st.caption(f"Rolling {window}-trade Win Rate (%)")
        with col2:
            st.line_chart(rdf[["rolling_pnl"]], height=220)
            st.caption(f"Rolling {window}-trade P&L (₹)")


# ── Tab: Duration Analysis ────────────────────────────────────────
with t_duration:
    st.subheader("Trade Duration Analysis")
    duration = perf_engine.duration_analysis(trades)
    if duration:
        ddf = pd.DataFrame(duration).T
        ddf.index.name = "Duration Bucket"
        st.dataframe(ddf, use_container_width=True)

        # Stats
        st.metric("Avg Hold Time", f"{metrics.avg_duration_minutes/60:.1f} hours")
        col1, col2 = st.columns(2)
        col1.metric("Avg Win Duration", f"{metrics.avg_win_duration/60:.1f} hours")
        col2.metric("Avg Loss Duration", f"{metrics.avg_loss_duration/60:.1f} hours")


# ── Tab: Trade Log ────────────────────────────────────────────────
with t_trades:
    st.subheader("Trade Log")

    best_worst = perf_engine.best_worst(trades, 5)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("🏆 **Best Trades**")
        st.dataframe(pd.DataFrame(best_worst["best"]), use_container_width=True, hide_index=True)
    with col2:
        st.markdown("💀 **Worst Trades**")
        st.dataframe(pd.DataFrame(best_worst["worst"]), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("All Trades")

    all_df = pd.DataFrame([{
        "date": (t.exit_date or t.entry_date)[:10],
        "type": t.strategy_type,
        "instrument": t.instrument,
        "pnl": round(t.pnl, 0),
        "premium": round(t.premium_collected, 0),
        "exit": t.exit_reason,
        "dur_h": round(t.duration_minutes / 60, 1),
    } for t in trades])

    def color_pnl_row(row):
        if row.get("pnl", 0) > 0:
            return ["background-color: #0d4d2b; color: #00ff88"] * len(row)
        return ["background-color: #4d0d0d; color: #ff4444"] * len(row)

    st.dataframe(
        all_df.style.apply(color_pnl_row, axis=1),
        use_container_width=True, hide_index=True, height=420
    )

    # Export
    csv_data = perf_engine.to_csv(trades)
    if csv_data:
        st.download_button(
            "📥 Download CSV",
            data=csv_data,
            file_name=f"trades_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )


# ── Tab: Position Sizer ───────────────────────────────────────────
with t_sizer:
    st.subheader("🎯 Position Sizing Recommendations")

    inst = INSTRUMENTS.get(sizer_inst, INSTRUMENTS["NIFTY"])
    state = engine.state
    spot_for_sizer = state.get_spot(inst["breeze_code"])
    if spot_for_sizer <= 0:
        spot_for_sizer = 24000.0

    exp_dt = next_weekly_expiry(sizer_inst)
    exp_str = breeze_expiry(exp_dt)

    # Use journal win rate if available
    wr_from_journal = metrics.win_rate / 100 if metrics.total_trades > 5 else 0.60
    avg_win_from_journal = metrics.avg_win / max(metrics.avg_win + metrics.avg_loss, 1) if metrics.avg_win > 0 else 0.5

    strategy_type = st.selectbox("Strategy Type", ["straddle", "strangle", "iron_condor"], key="sizer_type")

    col1, col2 = st.columns(2)
    with col1:
        win_rate_input = st.slider("Win Rate %", 30, 90, int(wr_from_journal * 100)) / 100
        used_capital = st.number_input("Already Deployed (₹)", value=0, step=10_000)
    with col2:
        avg_win_pct = st.slider("Avg Win as % of Premium", 10, 100, 50) / 100
        avg_loss_pct = st.slider("Avg Loss as % of Premium", 50, 200, 100) / 100

    sizing = sizer.size_for_premium(
        instrument=sizer_inst,
        spot=spot_for_sizer,
        atm_iv=sizer_iv,
        expiry=exp_str,
        strategy_type=strategy_type,
        available_capital=float(sizer_capital),
        used_capital=float(used_capital),
        win_rate=win_rate_input,
        avg_win_pct=avg_win_pct,
        avg_loss_pct=avg_loss_pct,
    )

    # Recommended
    st.markdown(f"### Recommended: **{sizing.recommended_lots} Lots**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("By Margin", f"{sizing.max_lots_margin} lots")
    c2.metric("By Kelly", f"{sizing.max_lots_kelly} lots")
    c3.metric("By Risk %", f"{sizing.max_lots_risk} lots")
    c4.metric("Portfolio Heat", f"{sizing.portfolio_heat_pct:.1f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Margin Required", f"₹{sizing.margin_required:,.0f}")
    c6.metric("Premium (est.)", f"₹{sizing.premium_collected:,.0f}")
    c7.metric("Max Loss", f"₹{sizing.max_loss_estimate:,.0f}")
    c8.metric("Risk/Reward", f"{sizing.risk_reward:.2f}x")

    # EV
    ev = sizer.expected_value(sizing.premium_collected, sizing.max_loss_estimate, win_rate_input)
    if ev:
        st.divider()
        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Expected Value", f"₹{ev['expected_value']:+,.0f}")
        ec2.metric("EV as % of Premium", f"{ev['ev_pct_of_premium']:.1f}%")
        ec3.metric("Break-even Win Rate", f"{ev['break_even_win_rate']:.1f}%")

    # Notes
    if sizing.notes:
        with st.expander("Sizing Notes"):
            for note in sizing.notes:
                st.write(f"• {note}")

    # ROM
    st.divider()
    st.subheader("Return on Margin")
    col1, col2 = st.columns(2)
    dte_for_rom = st.slider("Days in Trade", 1, 30, 5)
    rom = sizer.compute_roi(sizing.premium_collected, sizing.margin_required, dte_for_rom)
    if rom:
        rc1, rc2 = st.columns(2)
        rc1.metric("ROM (trade)", f"{rom['rom_trade_pct']:.2f}%")
        rc2.metric("ROM (annualized)", f"{rom['rom_annualized_pct']:.1f}%")


# ── Tab: Market Regime ────────────────────────────────────────────
with t_regime:
    st.subheader("🌡️ Market Regime Analysis")

    regime_inst = st.selectbox("Instrument", list(INSTRUMENTS.keys()), key="regime_inst")
    regime_inst_cfg = INSTRUMENTS[regime_inst]
    regime_spot = engine.state.get_spot(regime_inst_cfg["breeze_code"])

    # Try to get current chain IV
    try:
        from utils import breeze_expiry, next_weekly_expiry
        exp_dt2 = next_weekly_expiry(regime_inst)
        exp_str2 = breeze_expiry(exp_dt2)
        chain2 = engine.get_chain(regime_inst, exp_str2)
        from utils import atm_strike
        if chain2 and regime_spot > 0:
            atm = atm_strike(regime_spot, regime_inst_cfg["strike_gap"])
            atm_rows = [r for r in chain2 if r["strike"] == atm]
            if atm_rows:
                current_iv = sum(r.get("iv", 0) for r in atm_rows) / len(atm_rows) / 100
            else:
                current_iv = 0.15
        else:
            current_iv = 0.15
    except Exception:
        current_iv = 0.15

    st.metric("Detected ATM IV", f"{current_iv*100:.1f}%")
    manual_iv = st.slider("Override IV %", 5, 60, int(current_iv * 100)) / 100

    regime = regime_detector.analyze(
        current_iv=manual_iv,
        spot_history=None,   # Would need historical data from API
        iv_history=None,
    )

    result = regime.to_dict()

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Vol Regime", result["vol_regime"].replace("_", " ").upper())
    r2.metric("IV State", result["iv_state"].replace("_", " ").upper())
    r3.metric("Term Structure", result["ts_shape"].upper())
    r4.metric("Confidence", f"{result['confidence']*100:.0f}%")

    if result["preferred"]:
        st.success("✅ **Preferred Strategies:** " + ", ".join(s.replace("_", " ").upper() for s in result["preferred"]))
    if result["avoid"]:
        st.error("❌ **Avoid:** " + ", ".join(s.replace("_", " ").upper() for s in result["avoid"]))

    if result["notes"]:
        with st.expander("Regime Notes"):
            for note in result["notes"]:
                st.write(f"• {note}")

    st.info(
        "💡 **Tip:** Connect historical spot data and IV history for more accurate regime detection. "
        "Currently using IV-only classification."
    )
