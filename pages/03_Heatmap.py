"""
pages/03_Heatmap.py  — NEW PAGE

P&L Heatmap Explorer
  • Spot × Time heatmap for active strategies
  • Spot × IV what-if analysis
  • Theta bleed simulation
  • Per-leg decomposition payoff
  • Scenario summary table
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Heatmap | Options Terminal",
    page_icon="🌡️",
    layout="wide",
)

st.title("🌡️ P&L Heatmap Explorer")


@st.cache_resource
def boot_engine():
    from trading_engine import TradingEngine
    eng = TradingEngine()
    eng.start()
    return eng

engine = boot_engine()
state = engine.state

try:
    from heatmap_engine import HeatmapEngine
    from models import StrategyStatus
    from app_config import Config, INSTRUMENTS
    hm_ok = True
except Exception as e:
    st.error(f"Heatmap module error: {e}")
    hm_ok = False

if not hm_ok:
    st.stop()

hm_engine = HeatmapEngine(risk_free_rate=Config.RISK_FREE_RATE)

# ── Strategy selector ─────────────────────────────────────────────
strategies = [s for s in state.get_strategies() if s.status == StrategyStatus.ACTIVE]

if not strategies:
    st.info("No active strategies. Deploy a strategy from the main terminal first.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("Strategy")
    labels = [f"{s.strategy_id} | {s.strategy_type.value.replace('_', ' ').upper()}" for s in strategies]
    chosen = st.selectbox("Select strategy", labels)
    s = strategies[labels.index(chosen)]

    st.divider()
    st.subheader("Heatmap Settings")
    spot_range = st.slider("Spot Range ±%", 2, 20, 8)
    spot_steps = st.slider("Spot Steps", 5, 20, 11)
    dte_steps = st.slider("DTE Steps", 3, 12, 7)

    st.divider()
    st.subheader("IV Stress")
    current_iv = st.slider("Current IV %", 5, 60, 15) / 100
    iv_stress_range = st.slider("IV Stress Range ±%", 10, 80, 40)

    st.divider()
    theta_days = st.slider("Theta Bleed Days", 3, 20, 7)

# ── Spot price ────────────────────────────────────────────────────
# Get from first active leg's stock_code
breeze_code = s.stock_code
spot = state.get_spot(breeze_code)

if spot <= 0:
    # Try to estimate from strategy entry
    for leg in s.legs:
        if leg.entry_price > 0 and leg.strike_price > 0:
            spot = leg.strike_price  # rough estimate
            break

if spot <= 0:
    st.warning("Could not determine spot price. Using 24000 as fallback.")
    spot = 24000.0

# ── Header ────────────────────────────────────────────────────────
h1, h2, h3, h4 = st.columns(4)
h1.metric("Strategy", s.strategy_type.value.replace("_", " ").upper())
h2.metric("Spot", f"₹{spot:,.0f}")
h3.metric("Active Legs", sum(1 for l in s.legs if l.status.value == "active"))
h4.metric("Current P&L", f"₹{s.compute_total_pnl():+,.0f}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────
t_main, t_iv, t_theta, t_decomp, t_scenario = st.tabs([
    "Spot × Time", "Spot × IV", "Theta Bleed", "Leg Decomp", "Scenarios"
])

# ── Helper: render heatmap ────────────────────────────────────────
def render_heatmap(data: dict, height: int = 400):
    if not data:
        st.info("No data for heatmap")
        return

    z = data.get("z_matrix", [])
    x_labels = data.get("x_labels", [])
    y_labels = data.get("y_labels", [])

    if not z or not x_labels or not y_labels:
        return

    df = pd.DataFrame(z, index=y_labels, columns=x_labels)

    max_pnl = data.get("max_profit", 1)
    min_pnl = data.get("max_loss", -1)
    abs_max = max(abs(max_pnl), abs(min_pnl), 1)

    def color_pnl(val):
        if val is None:
            return "background-color: #1a1a2e; color: #444"
        pct = val / abs_max
        if val > 0:
            intensity = min(int(pct * 200), 200)
            return f"background-color: rgb(0,{100+intensity},50); color: white; font-weight: bold"
        elif val < 0:
            intensity = min(int(abs(pct) * 200), 200)
            return f"background-color: rgb({100+intensity},0,0); color: white; font-weight: bold"
        return "background-color: #333; color: white"

    styled = df.style.applymap(color_pnl).format("{:.0f}", na_rep="—")
    st.dataframe(styled, use_container_width=True, height=height)

    # Stats row
    max_p = data.get("max_profit", 0)
    max_l = data.get("max_loss", 0)
    be_u = data.get("breakeven_upper", 0)
    be_l = data.get("breakeven_lower", 0)

    sc = st.columns(4)
    sc[0].metric("Max Profit", f"₹{max_p:+,.0f}")
    sc[1].metric("Max Loss", f"₹{max_l:+,.0f}")
    if be_u:
        sc[2].metric("Breakeven Upper", f"₹{be_u:,.0f}")
    if be_l:
        sc[3].metric("Breakeven Lower", f"₹{be_l:,.0f}")


# ── Tab: Spot × Time ─────────────────────────────────────────────
with t_main:
    st.subheader(f"P&L: Spot × DTE — {s.strategy_id}")
    st.caption("Green = profit, Red = loss. Each cell = estimated P&L at that spot/DTE.")

    with st.spinner("Computing heatmap..."):
        hm_data = hm_engine.spot_vs_time(
            s, spot,
            spot_range_pct=float(spot_range),
            spot_steps=spot_steps,
            dte_steps=dte_steps,
        )

    render_heatmap(hm_data, height=420)


# ── Tab: Spot × IV ───────────────────────────────────────────────
with t_iv:
    st.subheader("P&L: Spot × IV (Vega Stress Test)")
    st.caption("Shows how P&L changes as both spot AND volatility move simultaneously.")

    with st.spinner("Computing IV stress heatmap..."):
        iv_data = hm_engine.spot_vs_iv(
            s, spot,
            current_iv=current_iv,
            spot_range_pct=float(spot_range),
            iv_range_pct=float(iv_stress_range),
            spot_steps=min(spot_steps, 9),
            iv_steps=7,
        )

    render_heatmap(iv_data, height=360)

    if current_iv > 0:
        st.info(
            f"📌 Current IV: {current_iv*100:.0f}% | "
            f"Range tested: {max(1, current_iv*(1-iv_stress_range/100))*100:.0f}% – "
            f"{current_iv*(1+iv_stress_range/100)*100:.0f}%"
        )


# ── Tab: Theta Bleed ─────────────────────────────────────────────
with t_theta:
    st.subheader(f"Theta Bleed — Next {theta_days} Days")
    st.caption("Cumulative P&L over time at different spot levels (assuming constant IV).")

    with st.spinner("Computing theta simulation..."):
        theta_data = hm_engine.theta_bleed(s, spot, days=theta_days)

    render_heatmap(theta_data, height=320)

    # Theta summary
    net_greeks = s.net_greeks
    st.divider()
    gc = st.columns(4)
    gc[0].metric("Net Δ", f"{net_greeks.delta:+.1f}")
    gc[1].metric("Net Γ", f"{net_greeks.gamma:+.4f}")
    gc[2].metric("Net Θ/day", f"₹{net_greeks.theta:+.1f}")
    gc[3].metric("Net V", f"{net_greeks.vega:+.1f}")

    if net_greeks.theta > 0:
        st.success(f"✅ Positive theta: earning approximately ₹{net_greeks.theta:.0f}/day from time decay")
    else:
        st.warning(f"⚠️ Negative theta: losing approximately ₹{abs(net_greeks.theta):.0f}/day from time decay")


# ── Tab: Leg Decomposition ────────────────────────────────────────
with t_decomp:
    st.subheader("Per-Leg Payoff at Expiry")
    st.caption("Individual leg contribution to total P&L at different spot levels.")

    decomp = hm_engine.leg_decomposition(s, spot, spot_range_pct=float(spot_range), steps=spot_steps)

    if not decomp:
        st.info("No active legs with entry prices.")
    else:
        # Combined chart
        chart_data = {}
        for leg_data in decomp:
            name = leg_data["leg_name"]
            chart_data[name] = leg_data["pnl_values"]

        if chart_data:
            x = decomp[0]["x_labels"] if decomp else []
            chart_df = pd.DataFrame(chart_data, index=x)

            # Add total
            chart_df["TOTAL"] = chart_df.sum(axis=1)

            st.line_chart(chart_df, height=300)
            st.caption("Dashed = total strategy P&L (sum of all legs)")

        # Leg details
        st.divider()
        for leg_data in decomp:
            with st.expander(f"📋 {leg_data['leg_name']} ({leg_data['side']} {leg_data['right']})"):
                lc = st.columns(3)
                lc[0].metric("Entry Price", f"₹{leg_data['entry_price']:.2f}")
                lc[1].metric("Strike", f"₹{leg_data['strike']:,}")
                lc[2].metric("SL Price", f"₹{leg_data['sl_price']:.2f}")

                leg_df = pd.DataFrame({
                    "Spot": leg_data["x_labels"],
                    "P&L": leg_data["pnl_values"],
                })
                st.dataframe(leg_df, use_container_width=True, hide_index=True, height=180)


# ── Tab: Scenarios ────────────────────────────────────────────────
with t_scenario:
    st.subheader("Quick Scenario Analysis")
    st.caption("P&L in key spot/time scenarios — snapshot view for quick review.")

    scenario_range = st.slider("Scenario Range ±%", 2, 15, 5, key="sc_range")

    scenarios = hm_engine.compute_scenario_stats(s, spot, spot_range_pct=float(scenario_range))

    if scenarios:
        rows = []
        for label, vals in scenarios.items():
            rows.append({
                "Scenario": label,
                "Spot": f"₹{vals['spot']:,.0f}",
                "P&L Today": f"₹{vals['pnl_today']:+,.0f}",
                "P&L at Expiry": f"₹{vals['pnl_expiry']:+,.0f}",
            })

        sc_df = pd.DataFrame(rows)

        def color_rows(row):
            pnl_str = row.get("P&L at Expiry", "₹0")
            pnl = float(pnl_str.replace("₹", "").replace(",", "").replace("+", ""))
            if pnl > 0:
                return ["background-color: #0d4d2b; color: #00ff88"] * len(row)
            if pnl < 0:
                return ["background-color: #4d0d0d; color: #ff4444"] * len(row)
            return [""] * len(row)

        st.dataframe(sc_df.style.apply(color_rows, axis=1),
                     use_container_width=True, hide_index=True)

        # Visual bar
        st.divider()
        st.markdown("**P&L at Expiry by Scenario**")
        bar_data = {r["Scenario"]: float(r["P&L at Expiry"].replace("₹", "").replace(",", "").replace("+", ""))
                    for r in rows}
        bar_df = pd.DataFrame.from_dict(bar_data, orient="index", columns=["P&L"])
        st.bar_chart(bar_df, height=250)
