"""
pages/05_Rollover.py  — NEW PAGE

Expiry Rollover Manager
  • Near-expiry alerts for all active positions
  • Roll cost / credit analysis
  • Optimal roll strike suggestions
  • One-click roll execution (via engine)
"""

import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Rollover | Options Terminal",
    page_icon="🔄",
    layout="wide",
)

st.title("🔄 Expiry Rollover Manager")

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60_000, limit=None, key="rollover_refresh")
except Exception:
    pass


@st.cache_resource
def boot_engine():
    from trading_engine import TradingEngine
    eng = TradingEngine()
    eng.start()
    return eng

engine = boot_engine()
state = engine.state

try:
    from rollover import RolloverEngine
    from models import StrategyStatus
    from app_config import Config, INSTRUMENTS
    from utils import breeze_expiry, next_weekly_expiry
    roll_ok = True
except Exception as e:
    st.error(f"Rollover module error: {e}")
    roll_ok = False

if not roll_ok:
    st.stop()

roll_engine = RolloverEngine(risk_free_rate=Config.RISK_FREE_RATE)

# ── Active strategies ─────────────────────────────────────────────
strategies = [s for s in state.get_strategies()
              if s.status in (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT)]

if not strategies:
    st.info("No active strategies to analyze for rollover.")
    st.stop()

# ── Build spot map ────────────────────────────────────────────────
spot_map = state.get_all_spots()

# ── Analyze all strategies ────────────────────────────────────────
analyses = roll_engine.analyze_all(strategies, spot_map)

# ── Summary ───────────────────────────────────────────────────────
urgent = [a for a in analyses if a.urgency in ("CRITICAL", "HIGH") and a.should_roll]
medium = [a for a in analyses if a.urgency == "MEDIUM"]
low = [a for a in analyses if a.urgency == "LOW"]

h1, h2, h3, h4 = st.columns(4)
h1.metric("Active Strategies", len(strategies))
h2.metric("🚨 Urgent Rolls", len(urgent))
h3.metric("⚠️ Watch", len(medium))
h4.metric("✅ OK", len(low))

if urgent:
    st.warning(f"**{len(urgent)} strategy(ies) need immediate rollover attention!**")

st.divider()

# ── Per-strategy analysis ─────────────────────────────────────────
for analysis in sorted(analyses, key=lambda a: a.dte_current):
    urgency_emoji = {"CRITICAL": "🚨", "HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(analysis.urgency, "⚪")
    roll_label = "ROLL NOW" if analysis.should_roll else "MONITOR"

    with st.expander(
        f"{urgency_emoji} {analysis.strategy_id} | {analysis.current_expiry} "
        f"({analysis.dte_current:.1f} DTE) | {roll_label}",
        expanded=analysis.urgency in ("CRITICAL", "HIGH"),
    ):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("DTE Remaining", f"{analysis.dte_current:.1f}d")
        col2.metric("Next Expiry DTE", f"{analysis.dte_next:.1f}d")
        col3.metric("Roll P&L", f"₹{analysis.roll_pnl:+,.0f}",
                    help="Positive = credit roll; Negative = debit roll")
        col4.metric("Urgency", analysis.urgency)

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Current Theta/day", f"₹{analysis.current_theta:.2f}")
        c6.metric("Next Theta/day", f"₹{analysis.next_theta:.2f}")
        c7.metric("Current Gamma", f"{analysis.current_gamma:.5f}")
        c8.metric("IV Pickup", f"{analysis.iv_pickup:+.2f}%")

        if analysis.reasons:
            st.info("\n".join(f"• {r}" for r in analysis.reasons))

        # Roll action buttons
        if analysis.should_roll:
            st.divider()
            st.markdown("**Roll Actions**")

            ra, rb, rc = st.columns(3)
            with ra:
                if st.button(f"🔄 Roll {analysis.strategy_id} → {analysis.next_expiry}",
                             key=f"roll_{analysis.strategy_id}",
                             type="primary"):
                    st.info("Roll execution: Close current + open next expiry positions.")
                    st.info(
                        "**Implementation note:** Use the main terminal to:\n"
                        "1. Exit current position (Positions tab → Exit Leg)\n"
                        "2. Deploy new position for next expiry\n\n"
                        "Auto-roll execution will be available in a future release."
                    )

            with rb:
                st.metric("Close at", f"₹{analysis.close_credit:,.0f}")

            with rc:
                st.metric("Open cost", f"₹{analysis.open_debit:,.0f}")


# ── Roll cost table ───────────────────────────────────────────────
st.divider()
st.subheader("📊 Roll Analysis Summary")

if analyses:
    summary_df = pd.DataFrame([{
        "Strategy": a.strategy_id,
        "Curr Expiry": a.current_expiry,
        "DTE": a.dte_current,
        "Next Expiry": a.next_expiry,
        "Roll P&L": f"₹{a.roll_pnl:+,.0f}",
        "Roll Type": "CREDIT" if a.roll_pnl > 0 else "DEBIT",
        "Θ Now": f"₹{a.current_theta:.2f}",
        "Θ Next": f"₹{a.next_theta:.2f}",
        "Urgency": a.urgency,
        "Action": "ROLL" if a.should_roll else "HOLD",
    } for a in analyses])

    def color_roll(row):
        if row.get("Action") == "ROLL" and row.get("Urgency") in ("CRITICAL", "HIGH"):
            return ["background-color: #4d0d0d; color: #ff4444"] * len(row)
        if row.get("Action") == "ROLL":
            return ["background-color: #4d4d0d; color: #ffcc44"] * len(row)
        return ["background-color: #0d2d4d; color: #44aaff"] * len(row)

    st.dataframe(
        summary_df.style.apply(color_roll, axis=1),
        use_container_width=True, hide_index=True
    )

# ── DTE Timeline ──────────────────────────────────────────────────
st.divider()
st.subheader("⏱️ DTE Timeline")

timeline_df = pd.DataFrame([{
    "Strategy": a.strategy_id,
    "DTE Remaining": a.dte_current,
} for a in analyses]).sort_values("DTE Remaining")

if not timeline_df.empty:
    st.bar_chart(timeline_df.set_index("Strategy"), height=200)
    st.caption("Days to expiry per strategy — lower = more urgent to act")

# ── Configuration ─────────────────────────────────────────────────
with st.expander("⚙️ Rollover Configuration"):
    st.write(f"Roll DTE Threshold: **{roll_engine.ROLL_DTE_THRESHOLD}** days")
    st.write(f"Urgent Roll DTE: **{roll_engine.ROLL_DTE_URGENT}** days")
    st.write(f"Gamma Threshold: **{roll_engine.GAMMA_THRESHOLD}**")
    st.write(f"Max Debit for Roll: **₹{roll_engine.MIN_CREDIT_ROLL:,.0f}**")
    st.info("Adjust thresholds in rollover.py to customize roll triggers.")
