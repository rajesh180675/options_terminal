# ═══════════════════════════════════════════════════════════════
# FILE: main.py  (Streamlit dashboard — entry point)
# ═══════════════════════════════════════════════════════════════
"""
streamlit run main.py

Layout:
  ┌──────────────────────────────────────────────────────┐
  │  HEADER: Total MTM   │  Status indicators            │
  ├──────────┬───────────┴──────────────────────────────┤
  │ SIDEBAR  │  BODY                                     │
  │ Config   │  Tab 1: Option Chain (CE / PE side-by-side)│
  │ Deploy   │  Tab 2: Active Positions                   │
  │ Controls │  Tab 3: Order & System Logs                │
  └──────────┴───────────────────────────────────────────┘
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime

st.set_page_config(
    page_title="Options Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auto-refresh (1 second) ─────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=1000, limit=None, key="terminal_refresh")
except ImportError:
    pass  # Will need manual refresh without the package

from trading_engine import TradingEngine
from config import Config
from utils import breeze_date, next_weekly_expiry
from models import LegStatus, StrategyStatus


# ═══════════════════════════════════════════════════════════════
# ENGINE SINGLETON (initialised once per Streamlit session)
# ═══════════════════════════════════════════════════════════════

def get_engine() -> TradingEngine:
    if "engine" not in st.session_state:
        engine = TradingEngine()
        success = engine.start()
        st.session_state.engine = engine
        st.session_state.engine_started = success
    return st.session_state.engine


engine = get_engine()
state = engine.state


# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .pnl-positive {
        background-color: #0d4d2b; color: #00ff88;
        padding: 12px 20px; border-radius: 8px;
        font-size: 28px; font-weight: bold; text-align: center;
    }
    .pnl-negative {
        background-color: #4d0d0d; color: #ff4444;
        padding: 12px 20px; border-radius: 8px;
        font-size: 28px; font-weight: bold; text-align: center;
    }
    .status-badge {
        padding: 4px 12px; border-radius: 12px;
        font-size: 13px; font-weight: 600;
        display: inline-block; margin: 2px;
    }
    .badge-green { background-color: #0d4d2b; color: #00ff88; }
    .badge-red { background-color: #4d0d0d; color: #ff4444; }
    .badge-yellow { background-color: #4d4d0d; color: #ffff44; }
    .log-container {
        font-family: 'Courier New', monospace;
        font-size: 12px; background: #1a1a2e;
        padding: 10px; border-radius: 6px;
        max-height: 400px; overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════

total_mtm = state.get_total_mtm()
pnl_class = "pnl-positive" if total_mtm >= 0 else "pnl-negative"
pnl_sign = "+" if total_mtm >= 0 else ""

col_h1, col_h2, col_h3, col_h4 = st.columns([3, 1, 1, 1])

with col_h1:
    st.markdown(
        f'<div class="{pnl_class}">Total MTM: ₹{pnl_sign}{total_mtm:,.2f}</div>',
        unsafe_allow_html=True,
    )

with col_h2:
    conn_color = "badge-green" if state.connected else "badge-red"
    conn_text = "API ✓" if state.connected else "API ✗"
    st.markdown(f'<span class="status-badge {conn_color}">{conn_text}</span>',
                unsafe_allow_html=True)

with col_h3:
    ws_color = "badge-green" if state.ws_connected else "badge-red"
    ws_text = "WS ✓" if state.ws_connected else "WS ✗"
    st.markdown(f'<span class="status-badge {ws_color}">{ws_text}</span>',
                unsafe_allow_html=True)

with col_h4:
    spot = state.get_spot(Config.DEFAULT_STOCK)
    st.markdown(
        f'<span class="status-badge badge-yellow">'
        f'{Config.DEFAULT_STOCK}: {spot:,.1f}</span>',
        unsafe_allow_html=True,
    )

st.divider()


# ═══════════════════════════════════════════════════════════════
# SIDEBAR — Configuration & Controls
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Strategy Deployer")

    stock_code = st.selectbox("Underlying", ["NIFTY", "BANKNIFTY"], index=0)
    breeze_stock = "CNXBAN" if stock_code == "BANKNIFTY" else "NIFTY"
    expiry_dt = next_weekly_expiry(breeze_stock)
    expiry_str = breeze_date(expiry_dt)
    st.text(f"Expiry: {expiry_dt.strftime('%d-%b-%Y')}")

    lots = st.number_input("Lots", min_value=1, max_value=50, value=1)
    sl_pct = st.slider("SL %", min_value=10, max_value=200, value=int(Config.SL_PERCENTAGE))

    st.subheader("Short Straddle")
    if st.button("🔴 Deploy Straddle", use_container_width=True, type="primary"):
        with st.spinner("Deploying Straddle…"):
            result = engine.deploy_straddle(breeze_stock, expiry_str, lots, float(sl_pct))
        if result:
            st.success(f"Straddle deployed: {result.strategy_id}")
        else:
            st.error("Straddle deployment failed")

    st.subheader("Short Strangle")
    target_delta = st.slider("Target Delta", 0.05, 0.40, 0.15, 0.01)
    if st.button("🔴 Deploy Strangle", use_container_width=True, type="primary"):
        with st.spinner("Deploying Strangle…"):
            result = engine.deploy_strangle(
                breeze_stock, target_delta, expiry_str, lots, float(sl_pct)
            )
        if result:
            st.success(f"Strangle deployed: {result.strategy_id}")
        else:
            st.error("Strangle deployment failed")

    st.divider()
    st.subheader("🚨 Kill Switch")
    if st.button("⚠️ PANIC EXIT — Close All", use_container_width=True,
                 type="primary"):
        engine.trigger_panic_exit()
        st.error("PANIC EXIT TRIGGERED")

    st.divider()
    st.caption(f"Mode: **{Config.TRADING_MODE.upper()}**")
    st.caption(f"Rate limit remaining: {__import__('rate_limiter').get_rate_limiter().remaining}")
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")


# ═══════════════════════════════════════════════════════════════
# MAIN BODY — Tabs
# ═══════════════════════════════════════════════════════════════

tab_chain, tab_positions, tab_logs = st.tabs([
    "📈 Option Chain", "📋 Active Positions", "📝 Logs"
])


# ── Tab 1: Option Chain ─────────────────────────────────────

with tab_chain:
    st.subheader(f"Option Chain — {stock_code}")

    chain_placeholder = st.empty()

    try:
        chain_data = engine.get_chain_with_greeks(breeze_stock, expiry_str)
    except Exception as e:
        chain_data = []
        st.warning(f"Chain fetch error: {e}")

    if chain_data:
        df = pd.DataFrame(chain_data)

        # Split into CE and PE
        df_ce = df[df["right"] == "CALL"].rename(columns={
            "ltp": "CE LTP", "bid": "CE Bid", "ask": "CE Ask",
            "iv": "CE IV%", "delta": "CE Δ", "theta": "CE Θ",
            "vega": "CE V", "oi": "CE OI",
        })[["strike", "CE OI", "CE IV%", "CE Δ", "CE Θ", "CE V", "CE Bid", "CE LTP", "CE Ask"]]

        df_pe = df[df["right"] == "PUT"].rename(columns={
            "ltp": "PE LTP", "bid": "PE Bid", "ask": "PE Ask",
            "iv": "PE IV%", "delta": "PE Δ", "theta": "PE Θ",
            "vega": "PE V", "oi": "PE OI",
        })[["strike", "PE Bid", "PE LTP", "PE Ask", "PE Δ", "PE Θ", "PE V", "PE IV%", "PE OI"]]

        merged = pd.merge(df_ce, df_pe, on="strike", how="outer").sort_values("strike")
        merged = merged.fillna(0)

        with chain_placeholder.container():
            st.dataframe(
                merged,
                use_container_width=True,
                height=500,
                hide_index=True,
            )
    else:
        chain_placeholder.info("No chain data available. Engine may still be connecting.")


# ── Tab 2: Active Positions ─────────────────────────────────

with tab_positions:
    st.subheader("Active Strategies & Legs")

    strategies = state.get_strategies()

    if not strategies:
        st.info("No active strategies. Deploy one from the sidebar.")
    else:
        for strategy in strategies:
            status_emoji = {
                StrategyStatus.ACTIVE: "🟢",
                StrategyStatus.DEPLOYING: "🟡",
                StrategyStatus.PARTIAL_EXIT: "🟠",
                StrategyStatus.CLOSED: "⚫",
                StrategyStatus.ERROR: "🔴",
            }.get(strategy.status, "⚪")

            pnl = strategy.compute_total_pnl()
            pnl_color = "green" if pnl >= 0 else "red"

            with st.expander(
                f"{status_emoji} {strategy.strategy_type.value.upper()} "
                f"[{strategy.strategy_id}] — "
                f"P&L: :{pnl_color}[₹{pnl:+,.2f}]",
                expanded=(strategy.status == StrategyStatus.ACTIVE),
            ):
                leg_rows = []
                for leg in strategy.legs:
                    leg_rows.append({
                        "Leg": leg.leg_id[:6],
                        "Type": f"{leg.right.value.upper()[0]}E",
                        "Strike": leg.strike_price,
                        "Entry": leg.entry_price,
                        "LTP": leg.current_price,
                        "SL": leg.sl_price,
                        "P&L": round(leg.pnl, 2),
                        "Δ": round(leg.greeks.delta, 4),
                        "Θ": round(leg.greeks.theta, 2),
                        "IV%": round(leg.greeks.iv * 100, 1),
                        "Status": leg.status.value,
                    })
                if leg_rows:
                    leg_df = pd.DataFrame(leg_rows)

                    def color_pnl(val):
                        if isinstance(val, (int, float)):
                            return "color: #00ff88" if val >= 0 else "color: #ff4444"
                        return ""

                    def color_status(val):
                        colors = {
                            "active": "background-color: #0d4d2b; color: #00ff88",
                            "squared_off": "background-color: #333; color: #aaa",
                            "sl_triggered": "background-color: #4d0d0d; color: #ff4444",
                            "error": "background-color: #4d0d0d; color: #ff4444",
                            "entering": "background-color: #4d4d0d; color: #ffff44",
                        }
                        return colors.get(val, "")

                    styled = leg_df.style.applymap(
                        color_pnl, subset=["P&L"]
                    ).applymap(
                        color_status, subset=["Status"]
                    )
                    st.dataframe(styled, use_container_width=True, hide_index=True)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Strategy P&L", f"₹{pnl:+,.2f}")
                with col_b:
                    st.metric("Status", strategy.status.value.upper())


# ── Tab 3: Logs ──────────────────────────────────────────────

with tab_logs:
    st.subheader("System Logs")

    log_col1, log_col2 = st.columns([3, 2])

    with log_col1:
        st.markdown("**Agent Thoughts & Events**")
        logs = state.get_logs(100)

        if logs:
            log_lines = []
            for entry in reversed(logs):
                level_color = {
                    "INFO": "#88ccff",
                    "WARN": "#ffcc44",
                    "ERROR": "#ff4444",
                    "CRIT": "#ff0000",
                }.get(entry.level, "#cccccc")
                log_lines.append(
                    f'<span style="color:{level_color}">[{entry.timestamp}] '
                    f'{entry.level:5s}</span> │ '
                    f'<span style="color:#aaa">{entry.source:12s}</span> │ '
                    f'{entry.message}'
                )
            html = '<div class="log-container">' + "<br>".join(log_lines) + "</div>"
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("No log entries yet.")

    with log_col2:
        st.markdown("**Order History**")
        try:
            order_logs = engine.db.get_recent_order_logs(30)
            if order_logs:
                ol_df = pd.DataFrame(order_logs)
                display_cols = [c for c in ["timestamp", "action", "status", "price",
                                            "quantity", "order_id", "message"]
                                if c in ol_df.columns]
                st.dataframe(ol_df[display_cols], use_container_width=True,
                             hide_index=True, height=400)
            else:
                st.info("No orders yet.")
        except Exception as e:
            st.warning(f"Order log error: {e}")


# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════

st.divider()
footer_cols = st.columns(4)
with footer_cols[0]:
    st.caption(f"Mode: {Config.TRADING_MODE.upper()}")
with footer_cols[1]:
    st.caption(f"DB: {Config.DB_PATH}")
with footer_cols[2]:
    active_count = sum(1 for s in strategies if s.status == StrategyStatus.ACTIVE)
    st.caption(f"Active strategies: {active_count}")
with footer_cols[3]:
    if state.panic_triggered:
        st.markdown("🚨 **PANIC MODE**")
    else:
        st.caption(f"Engine: {'Running' if state.engine_running else 'Stopped'}")
