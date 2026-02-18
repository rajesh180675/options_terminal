"""
pages/01_IV_Surface.py  — NEW PAGE

Volatility Surface Explorer
  • Multi-expiry IV surface (moneyness × DTE)
  • Term structure (ATM IV across expiries)
  • Skew metrics (25Δ risk reversal, butterfly)
  • IV Percentile / Rank display

This page is independent and non-breaking.
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="IV Surface | Options Terminal",
    page_icon="🌊",
    layout="wide",
)

st.title("🌊 Volatility Surface Explorer")

# ── Bootstrap engine (shared with main.py) ──────────────────────
@st.cache_resource
def boot_engine():
    from trading_engine import TradingEngine
    eng = TradingEngine()
    eng.start()
    return eng

engine = boot_engine()
state = engine.state

# ── Imports ─────────────────────────────────────────────────────
try:
    from iv_surface import IVSurfaceEngine
    from app_config import Config, INSTRUMENTS
    from utils import breeze_expiry, next_weekly_expiry, atm_strike
    iv_ok = True
except Exception as e:
    st.error(f"IV Surface module error: {e}")
    iv_ok = False

if not iv_ok:
    st.stop()

surf_engine = IVSurfaceEngine(risk_free_rate=Config.RISK_FREE_RATE)

# ── Sidebar controls ─────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    inst_name = st.selectbox("Instrument", list(INSTRUMENTS.keys()))
    inst = Config.instrument(inst_name)

    from datetime import datetime
    exp_dt = next_weekly_expiry(inst_name)
    exp_str = breeze_expiry(exp_dt)
    st.caption(f"Expiry: {exp_dt.strftime('%d-%b-%Y')}")

    show_calls = st.checkbox("Include Calls", value=True)
    show_puts = st.checkbox("Include Puts", value=True)
    st.divider()
    st.markdown("**Note:** Surface builds from live chain data. Refresh to update.")

# ── Load chain ───────────────────────────────────────────────────
try:
    chain = engine.get_chain(inst_name, exp_str)
except Exception as e:
    chain = []
    st.warning(f"Chain load error: {e}")

breeze_code = inst["breeze_code"]
spot = state.get_spot(breeze_code)

if not chain or spot <= 0:
    st.info("Waiting for chain data and spot price...")
    st.stop()

# Filter by right
filtered_chain = chain
if not show_calls:
    filtered_chain = [r for r in chain if r.get("right") != "CALL"]
if not show_puts:
    filtered_chain = [r for r in chain if r.get("right") != "PUT"]

# ── Build surface ────────────────────────────────────────────────
surface_pts = surf_engine.build(filtered_chain, spot)

# ── Layout ───────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Spot", f"₹{spot:,.0f}")
col2.metric("Chain Rows", len(chain))
col3.metric("Surface Points", len(surface_pts))

st.divider()

# ── Tab layout ───────────────────────────────────────────────────
t_skew, t_term, t_surface, t_grid = st.tabs(["Skew", "Term Structure", "Surface", "Raw Grid"])

# ── Tab: Skew ────────────────────────────────────────────────────
with t_skew:
    st.subheader("Volatility Skew Analysis")

    skew = surf_engine.skew_metrics(chain, spot, exp_str)
    if skew:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ATM IV", f"{skew['atm_iv']:.1f}%")
        c2.metric("25Δ RR", f"{skew['rr25']:+.2f}%",
                  help="Risk Reversal: OTM Call IV - OTM Put IV. Negative = put skew (fear)")
        c3.metric("25Δ BF", f"{skew['bf25']:.2f}%",
                  help="Butterfly: avg OTM wing IV - ATM IV. Higher = fatter tails")
        c4.metric("Skew Slope", f"{skew['skew_slope']:.3f}",
                  help="IV vs Moneyness slope. Negative = put skew dominates")

        c5, c6, c7 = st.columns(3)
        c5.metric("25Δ Call IV", f"{skew['call_25d_iv']:.1f}%")
        c6.metric("25Δ Put IV", f"{skew['put_25d_iv']:.1f}%")
        c7.metric("Put Skew Premium", f"{skew['put_skew']:+.2f}%")

    # Skew chart: IV vs Strike
    calls = [r for r in chain if r.get("right") == "CALL" and r.get("iv", 0) > 0]
    puts = [r for r in chain if r.get("right") == "PUT" and r.get("iv", 0) > 0]

    if calls or puts:
        skew_df_data = {}
        if calls:
            ce_df = pd.DataFrame({"strike": [r["strike"] for r in calls],
                                   "CE IV": [r["iv"] for r in calls]}).set_index("strike")
            skew_df_data["CE IV"] = ce_df["CE IV"]
        if puts:
            pe_df = pd.DataFrame({"strike": [r["strike"] for r in puts],
                                   "PE IV": [r["iv"] for r in puts]}).set_index("strike")
            skew_df_data["PE IV"] = pe_df["PE IV"]

        if skew_df_data:
            skew_chart = pd.DataFrame(skew_df_data)
            st.line_chart(skew_chart, height=280, color=["#00aaff", "#ff6644"])

    # ATM highlight line
    if spot > 0:
        atm = atm_strike(spot, inst["strike_gap"])
        st.caption(f"ATM Strike: {int(atm)} | Spot: ₹{spot:,.0f}")


# ── Tab: Term Structure ──────────────────────────────────────────
with t_term:
    st.subheader("IV Term Structure")

    term = surf_engine.term_structure(chain, spot)
    if term:
        ts_df = pd.DataFrame(term)
        c1, c2 = st.columns([2, 1])

        with c1:
            chart_data = ts_df.set_index("dte")[["atm_iv", "call_iv", "put_iv"]]
            st.line_chart(chart_data, height=280,
                         color=["#ffffff", "#00aaff", "#ff6644"])
            st.caption("ATM IV (white), Call IV (blue), Put IV (red) vs DTE")

        with c2:
            st.dataframe(ts_df, use_container_width=True, hide_index=True)

        # Shape classification
        from iv_surface import _estimate_strike_gap
        if len(term) >= 2:
            near_iv = term[0]["atm_iv"]
            far_iv = term[-1]["atm_iv"]
            slope = (far_iv - near_iv) / far_iv if far_iv > 0 else 0

            if slope > 0.05:
                shape_msg = "📈 **Contango** (normal structure — near < far)"
                shape_color = "green"
            elif slope < -0.05:
                shape_msg = "📉 **Backwardation** (stressed — near > far, roll down opportunity)"
                shape_color = "red"
            else:
                shape_msg = "➡️ **Flat** term structure"
                shape_color = "blue"

            st.info(shape_msg)
    else:
        st.info("Term structure requires multi-expiry chain data. Currently showing single expiry.")

        # Show single expiry IV summary
        ivs = [r["iv"] for r in chain if r.get("iv", 0) > 0]
        if ivs:
            st.metric("Average Chain IV", f"{sum(ivs)/len(ivs):.1f}%")
            st.metric("IV Range", f"{min(ivs):.1f}% – {max(ivs):.1f}%")


# ── Tab: Surface ─────────────────────────────────────────────────
with t_surface:
    st.subheader("IV Surface Grid (Moneyness × DTE)")

    if surface_pts:
        grid_data = surf_engine.surface_to_grid(surface_pts, moneyness_bins=9, dte_bins=5)

        if grid_data:
            z = grid_data["z_matrix"]
            y_labels = grid_data["y_labels"]
            x_labels = grid_data["x_labels"]

            # Create DataFrame for display
            surf_df = pd.DataFrame(z, index=y_labels, columns=x_labels)

            # Color-scale display (higher IV = warmer color via Streamlit style)
            def color_iv(val):
                if val is None:
                    return "background-color: #1a1a2e; color: #444"
                if val > 25:
                    return "background-color: #4d0d0d; color: #ff6644; font-weight: bold"
                if val > 20:
                    return "background-color: #4d3d0d; color: #ffaa44"
                if val > 15:
                    return "background-color: #0d4d0d; color: #44ff88"
                return "background-color: #0d2d4d; color: #44aaff"

            styled = surf_df.style.applymap(color_iv)
            st.dataframe(styled, use_container_width=True)
            st.caption("IV% by Moneyness (rows) vs DTE (columns)")
        else:
            st.info("Not enough data points to build surface grid.")
    else:
        st.info("Build surface by loading chain data.")


# ── Tab: Raw Grid ─────────────────────────────────────────────────
with t_grid:
    st.subheader("Surface Data Points")

    if surface_pts:
        pts_df = pd.DataFrame([{
            "strike": int(p.strike),
            "right": p.right,
            "expiry": p.expiry,
            "moneyness_%": round(p.moneyness * 100, 2),
            "dte_days": p.dte_days,
            "iv_%": round(p.iv * 100, 2),
            "ltp": p.ltp,
        } for p in surface_pts])

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            right_filter = st.multiselect("Right", ["CALL", "PUT"], default=["CALL", "PUT"])
        with col2:
            iv_min = st.slider("Min IV%", 0, 50, 5)

        filtered = pts_df[
            pts_df["right"].isin(right_filter) &
            (pts_df["iv_%"] >= iv_min)
        ]

        st.dataframe(filtered, use_container_width=True, hide_index=True, height=450)
        st.caption(f"Showing {len(filtered)} of {len(pts_df)} surface points")
    else:
        st.info("No surface points computed.")
