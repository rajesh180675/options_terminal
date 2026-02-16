"""
main.py — Streamlit-based Options Trading Terminal.
Run with:  streamlit run main.py
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from config import CFG, INSTRUMENTS
from strategy_engine import StrategyParams
from trading_engine import TradingEngine

# ================================================================
# Page config
# ================================================================
st.set_page_config(
    page_title="Options Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# Custom CSS
# ================================================================
st.markdown("""
<style>
    .stMetric { border: 1px solid #333; border-radius: 8px; padding: 10px; }
    .profit { background-color: #0d4d0d !important; border-radius: 8px; padding: 15px; }
    .loss { background-color: #4d0d0d !important; border-radius: 8px; padding: 15px; }
    div[data-testid="stMetric"] label { font-size: 0.9rem !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important; font-weight: bold;
    }
    .log-entry { font-family: monospace; font-size: 0.75rem; line-height: 1.3; }
    .log-error { color: #ff4444; }
    .log-warn { color: #ffaa00; }
    .log-info { color: #aaaaaa; }
    .log-critical { color: #ff0000; font-weight: bold; }
    .kill-btn button { background-color: #ff0000 !important; color: white !important;
                       font-weight: bold !important; font-size: 1.2rem !important; }
</style>
""", unsafe_allow_html=True)


# ================================================================
# Engine initialisation (once per session)
# ================================================================
def get_engine() -> TradingEngine:
    if "engine" not in st.session_state:
        st.session_state.engine = TradingEngine()
    return st.session_state.engine


engine = get_engine()


# ================================================================
# Sidebar — Connection & Strategy Controls
# ================================================================
with st.sidebar:
    st.title("⚙️ Control Panel")

    # --- Connection ---
    st.subheader("Connection")
    col_status, col_btn = st.columns([2, 1])
    with col_status:
        if engine.is_running:
            st.success("🟢 Connected")
        else:
            st.error("🔴 Disconnected")
    with col_btn:
        if not engine.is_running:
            if st.button("Connect", use_container_width=True):
                with st.spinner("Connecting..."):
                    ok = engine.start()
                if ok:
                    st.rerun()
                else:
                    st.error("Connection failed. Check logs.")
        else:
            if st.button("Disconnect", use_container_width=True):
                engine.stop()
                st.rerun()

    st.divider()

    # --- Strategy Deployment ---
    st.subheader("Deploy Strategy")

    instrument = st.selectbox("Instrument",
                              list(INSTRUMENTS.keys()),
                              index=0)

    strategy_type = st.selectbox("Strategy",
                                 ["strangle", "straddle"],
                                 index=0)

    target_delta = st.slider("Target Delta",
                             min_value=0.05, max_value=0.40,
                             value=0.15, step=0.01,
                             help="Absolute delta for OTM strike selection",
                             disabled=(strategy_type == "straddle"))

    lots = st.number_input("Lots", min_value=1, max_value=50, value=1)

    sl_mult = st.number_input("SL Multiplier",
                              min_value=1.0, max_value=5.0,
                              value=CFG.default_sl_multiplier,
                              step=0.1)

    expiry_dates = engine.get_expiry_dates(instrument) if engine.is_running else []
    expiry = st.selectbox("Expiry Date",
                          expiry_dates if expiry_dates else ["Connect first"],
                          index=0)

    if st.button("🚀 Deploy Strategy",
                 use_container_width=True,
                 disabled=not engine.is_running or expiry == "Connect first",
                 type="primary"):
        params = StrategyParams(
            instrument=instrument,
            strategy_type=strategy_type,
            target_delta=target_delta,
            lots=lots,
            sl_multiplier=sl_mult,
            expiry_date=expiry,
        )
        with st.spinner("Deploying..."):
            sid = engine.deploy_strategy(params)
        if sid:
            st.success(f"Strategy {sid} deployed!")
        else:
            st.error("Deployment failed. Check logs.")
        st.rerun()

    st.divider()

    # --- Kill Switch ---
    st.subheader("Emergency")
    kill_col = st.container()
    with kill_col:
        st.markdown('<div class="kill-btn">', unsafe_allow_html=True)
        if st.button("🚨 PANIC EXIT — CLOSE ALL",
                     use_container_width=True,
                     disabled=not engine.is_running):
            engine.panic_exit()
            st.warning("Kill switch activated!")
            time.sleep(2)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ================================================================
# Main Dashboard
# ================================================================

# --- Header: P&L ---
st.title("📊 Options Trading Terminal")


@st.fragment(run_every=1)
def render_pnl_header():
    """Real-time P&L display — updates every second."""
    mtm = engine.state.get_total_mtm()
    ws_ok = engine.state.is_ws_connected()
    tick_age = engine.state.get_last_tick_age()

    cols = st.columns([2, 1, 1, 1])
    with cols[0]:
        color = "normal" if mtm >= 0 else "inverse"
        st.metric("Total MTM P&L", f"₹{mtm:,.2f}",
                  delta=f"{'🟢' if mtm >= 0 else '🔴'}")
    with cols[1]:
        st.metric("Mode", CFG.trading_mode.upper())
    with cols[2]:
        st.metric("WebSocket", "🟢 Live" if ws_ok else "🔴 Down")
    with cols[3]:
        st.metric("Tick Age", f"{tick_age:.0f}s")


render_pnl_header()

st.divider()

# --- Active Strategies ---
st.subheader("📋 Active Strategies")


@st.fragment(run_every=2)
def render_strategies():
    strategies = engine.state.get_all_strategies()

    if not strategies:
        st.info("No active strategies. Deploy one from the sidebar.")
        return

    for sid, strat in strategies.items():
        status_emoji = {
            "active": "🟢",
            "initializing": "⏳",
            "entering": "⏳",
            "exiting": "🟡",
            "closed": "⚫",
            "error": "🔴",
        }.get(strat.get("status", ""), "❓")

        with st.expander(
            f"{status_emoji} {strat.get('name', '').upper()} | "
            f"{sid} | "
            f"CE={strat.get('ce_strike', 'N/A')} / "
            f"PE={strat.get('pe_strike', 'N/A')} | "
            f"MTM: ₹{strat.get('current_mtm', 0):,.0f}",
            expanded=(strat.get("status") == "active"),
        ):
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("CE Strike", strat.get("ce_strike", "N/A"))
                st.metric("CE Entry", f"₹{strat.get('ce_entry_price', 0):.2f}")
                ce_ltp = engine.state.get_ltp(
                    strat.get("stock_code", ""),
                    strat.get("ce_strike", 0), "call"
                )
                st.metric("CE LTP", f"₹{ce_ltp:.2f}" if ce_ltp > 0 else "—")
                st.metric("CE SL", f"₹{strat.get('ce_sl_price', 0):.2f}")

            with c2:
                st.metric("PE Strike", strat.get("pe_strike", "N/A"))
                st.metric("PE Entry", f"₹{strat.get('pe_entry_price', 0):.2f}")
                pe_ltp = engine.state.get_ltp(
                    strat.get("stock_code", ""),
                    strat.get("pe_strike", 0), "put"
                )
                st.metric("PE LTP", f"₹{pe_ltp:.2f}" if pe_ltp > 0 else "—")
                st.metric("PE SL", f"₹{strat.get('pe_sl_price', 0):.2f}")

            with c3:
                st.metric("Status", strat.get("status", "").upper())
                st.metric("Lots", strat.get("lots", 1))
                st.metric("Premium", f"₹{strat.get('total_premium', 0):,.0f}")
                st.metric("MTM", f"₹{strat.get('current_mtm', 0):,.0f}")

            with c4:
                # Greeks display
                if strat.get("status") == "active":
                    ce_g = engine.state.get_greeks(
                        strat.get("stock_code", ""),
                        strat.get("ce_strike", 0), "call"
                    )
                    pe_g = engine.state.get_greeks(
                        strat.get("stock_code", ""),
                        strat.get("pe_strike", 0), "put"
                    )
                    st.caption("**Net Greeks**")
                    net_delta = ce_g.delta + pe_g.delta
                    net_theta = ce_g.theta + pe_g.theta
                    st.metric("Net Δ", f"{net_delta:.4f}")
                    st.metric("Net Θ", f"₹{net_theta:.2f}/day")
                    st.metric("CE IV", f"{ce_g.iv:.1%}")
                    st.metric("PE IV", f"{pe_g.iv:.1%}")

            # Exit buttons
            if strat.get("status") in ("active", "entering"):
                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    if st.button(f"Exit (Chase) — {sid}",
                                 key=f"exit_{sid}"):
                        engine.exit_strategy(sid, use_market=False)
                        st.toast(f"Exiting {sid}...")
                        time.sleep(1)
                        st.rerun()
                with bcol2:
                    if st.button(f"Exit (Market) — {sid}",
                                 key=f"mkt_exit_{sid}",
                                 type="primary"):
                        engine.exit_strategy(sid, use_market=True)
                        st.toast(f"Market exit {sid}...")
                        time.sleep(1)
                        st.rerun()


render_strategies()

st.divider()

# --- Option Chain Viewer ---
st.subheader("🔗 Option Chain")


@st.fragment(run_every=5)
def render_option_chain():
    if not engine.is_running:
        st.info("Connect to view option chain.")
        return

    oc_inst = st.selectbox("Chain Instrument",
                           list(INSTRUMENTS.keys()),
                           key="oc_instrument",
                           index=0)
    oc_expiries = engine.get_expiry_dates(oc_inst)
    oc_expiry = st.selectbox("Chain Expiry",
                             oc_expiries if oc_expiries else ["N/A"],
                             key="oc_expiry")

    if oc_expiry == "N/A":
        return

    if st.button("Refresh Chain", key="refresh_chain"):
        chain = engine.refresh_option_chain(oc_inst, oc_expiry)
        if chain:
            st.session_state["_chain_data"] = chain
            st.session_state["_chain_inst"] = oc_inst
            st.session_state["_chain_expiry"] = oc_expiry

    chain = st.session_state.get("_chain_data", [])
    if not chain:
        st.info("Click 'Refresh Chain' to load data.")
        return

    spec = CFG.get_instrument(oc_inst)
    spot = engine.state.get_spot(spec.stock_code)

    # Parse into DataFrame
    ce_data = {}
    pe_data = {}
    for rec in chain:
        try:
            strike = float(rec.get("strike_price", 0))
            right_raw = rec.get("right", "").lower()
            ltp = float(rec.get("ltp", 0))
            bid = float(rec.get("best_bid_price", 0))
            ask = float(rec.get("best_offer_price", 0))
            oi = rec.get("open_interest", "0")

            row = {
                "LTP": round(ltp, 2),
                "Bid": round(bid, 2),
                "Ask": round(ask, 2),
                "OI": oi,
            }

            # Compute delta
            tte = engine.greeks_engine.time_to_expiry_years(oc_expiry)
            if ltp > 0 and spot > 0 and tte > 0:
                right = "call" if "call" in right_raw else "put"
                greeks = engine.greeks_engine.compute_for_strike(
                    spot, strike, tte, ltp, right
                )
                row["Δ"] = round(greeks.delta, 4)
                row["IV"] = f"{greeks.iv:.1%}"
                row["Θ"] = round(greeks.theta, 2)

            if "call" in right_raw:
                ce_data[strike] = row
            else:
                pe_data[strike] = row
        except (ValueError, TypeError):
            continue

    all_strikes = sorted(set(list(ce_data.keys()) + list(pe_data.keys())))

    # Build combined table
    rows = []
    for strike in all_strikes:
        ce = ce_data.get(strike, {})
        pe = pe_data.get(strike, {})
        rows.append({
            "CE Δ": ce.get("Δ", ""),
            "CE IV": ce.get("IV", ""),
            "CE Θ": ce.get("Θ", ""),
            "CE OI": ce.get("OI", ""),
            "CE Bid": ce.get("Bid", ""),
            "CE LTP": ce.get("LTP", ""),
            "CE Ask": ce.get("Ask", ""),
            "STRIKE": int(strike),
            "PE Bid": pe.get("Bid", ""),
            "PE LTP": pe.get("LTP", ""),
            "PE Ask": pe.get("Ask", ""),
            "PE OI": pe.get("OI", ""),
            "PE Θ": pe.get("Θ", ""),
            "PE IV": pe.get("IV", ""),
            "PE Δ": pe.get("Δ", ""),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        # Highlight ATM
        atm_strike = min(all_strikes, key=lambda s: abs(s - spot)) if spot > 0 else 0
        st.caption(f"Spot: {spot:.2f} | ATM: {int(atm_strike)}")
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            height=500,
        )
    else:
        st.warning("No chain data parsed.")


render_option_chain()

st.divider()

# --- Positions Table ---
st.subheader("📊 Open Positions")


@st.fragment(run_every=2)
def render_positions():
    positions = engine.state.get_open_positions()
    if not positions:
        st.info("No open positions.")
        return

    rows = []
    for pid, pos in positions.items():
        stock_code = pos.get("stock_code", "")
        strike = pos.get("strike_price", 0)
        right = pos.get("right_type", "")
        current = engine.state.get_ltp(stock_code, strike, right)
        greeks = engine.state.get_greeks(stock_code, strike, right)

        rows.append({
            "ID": pid[:8],
            "Strike": int(strike),
            "Type": right.upper()[:1] + "E",
            "Qty": pos.get("quantity", 0),
            "Entry": pos.get("entry_price", 0),
            "LTP": round(current, 2) if current > 0 else "—",
            "SL": pos.get("sl_price", 0),
            "P&L": f"₹{pos.get('pnl', 0):,.0f}",
            "Δ": round(greeks.delta, 4),
            "IV": f"{greeks.iv:.1%}",
            "Status": pos.get("status", "").upper(),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


render_positions()

st.divider()

# --- Live Logs ---
st.subheader("📝 Agent Logs")


@st.fragment(run_every=1)
def render_logs():
    logs = engine.state.get_logs(100)

    if not logs:
        st.info("No log entries yet.")
        return

    log_html = []
    for entry in reversed(logs):
        level = entry.get("level", "INFO").upper()
        css_class = {
            "ERROR": "log-error",
            "WARN": "log-warn",
            "WARNING": "log-warn",
            "CRITICAL": "log-critical",
        }.get(level, "log-info")

        ts = entry.get("ts", "")
        source = entry.get("source", "")
        msg = entry.get("message", "")

        log_html.append(
            f'<div class="log-entry {css_class}">'
            f'[{ts}] [{level:8s}] [{source:10s}] {msg}'
            f'</div>'
        )

    st.markdown("\n".join(log_html), unsafe_allow_html=True)


render_logs()

# --- Footer ---
st.divider()
st.caption(
    f"Options Terminal v1.0 | Mode: {CFG.trading_mode} | "
    f"DB: {CFG.db_path} | "
    f"Rate Limit: {CFG.api_rate_limit}/{CFG.api_rate_period}s"
)
