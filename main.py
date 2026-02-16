# ═══════════════════════════════════════════════════════════════
# FILE: main.py  (Streamlit Dashboard — fully rewritten)
# ═══════════════════════════════════════════════════════════════
"""
streamlit run main.py

NEW: Tab 4 — Limit Order Screen with strike selector and price input.
FIXED: Engine singleton, autorefresh, Streamlit secrets integration.
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

# ── Auto-refresh ─────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=1500, limit=None, key="terminal_refresh")
except ImportError:
    pass

from config import Config
from utils import breeze_expiry_format, next_weekly_expiry
from models import LegStatus, StrategyStatus, OptionRight


# ═══════════════════════════════════════════════════════════════
# ENGINE SINGLETON — guaranteed ONE engine per Streamlit session
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def create_engine():
    from trading_engine import TradingEngine
    engine = TradingEngine()
    success = engine.start()
    return engine, success


engine, engine_ok = create_engine()
state = engine.state


# ═══════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .pnl-positive {
        background: linear-gradient(135deg, #0d4d2b, #1a6b3f);
        color: #00ff88; padding: 14px 24px; border-radius: 10px;
        font-size: 30px; font-weight: bold; text-align: center;
        border: 1px solid #00ff8833;
    }
    .pnl-negative {
        background: linear-gradient(135deg, #4d0d0d, #6b1a1a);
        color: #ff4444; padding: 14px 24px; border-radius: 10px;
        font-size: 30px; font-weight: bold; text-align: center;
        border: 1px solid #ff444433;
    }
    .badge { padding: 4px 14px; border-radius: 14px;
             font-size: 13px; font-weight: 600; display: inline-block; }
    .badge-g { background: #0d4d2b; color: #00ff88; }
    .badge-r { background: #4d0d0d; color: #ff4444; }
    .badge-y { background: #4d4d0d; color: #ffff44; }
    .log-box {
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 11.5px; background: #0d1117;
        padding: 12px; border-radius: 8px; max-height: 420px;
        overflow-y: auto; border: 1px solid #21262d;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════

total_mtm = state.get_total_mtm()
pnl_cls = "pnl-positive" if total_mtm >= 0 else "pnl-negative"
pnl_sign = "+" if total_mtm >= 0 else ""

hcol1, hcol2, hcol3, hcol4, hcol5 = st.columns([3, 1, 1, 1, 1])

with hcol1:
    st.markdown(
        f'<div class="{pnl_cls}">Total MTM: ₹{pnl_sign}{total_mtm:,.2f}</div>',
        unsafe_allow_html=True,
    )
with hcol2:
    c = "badge-g" if state.connected else "badge-r"
    st.markdown(f'<span class="badge {c}">API {"✓" if state.connected else "✗"}</span>',
                unsafe_allow_html=True)
with hcol3:
    c = "badge-g" if state.ws_connected else "badge-r"
    st.markdown(f'<span class="badge {c}">WS {"✓" if state.ws_connected else "✗"}</span>',
                unsafe_allow_html=True)
with hcol4:
    breeze_stock = Config.breeze_code(Config.DEFAULT_STOCK)
    spot = state.get_spot(breeze_stock)
    st.markdown(
        f'<span class="badge badge-y">{Config.DEFAULT_STOCK}: ₹{spot:,.1f}</span>',
        unsafe_allow_html=True,
    )
with hcol5:
    mode_c = "badge-r" if Config.is_live() else "badge-y"
    st.markdown(
        f'<span class="badge {mode_c}">{Config.TRADING_MODE.upper()}</span>',
        unsafe_allow_html=True,
    )

st.divider()


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Strategy Deployer")

    stock_sel = st.selectbox("Underlying", ["NIFTY", "BANKNIFTY"], index=0)
    breeze_stock = Config.breeze_code(stock_sel)
    expiry_dt = next_weekly_expiry(breeze_stock)
    expiry_str = breeze_expiry_format(expiry_dt)
    st.caption(f"📅 Expiry: **{expiry_dt.strftime('%d-%b-%Y')}**")
    st.caption(f"🔑 Breeze code: `{breeze_stock}`")

    lots = st.number_input("Lots", 1, 50, 1)
    sl_pct = st.slider("SL %", 10, 200, int(Config.SL_PERCENTAGE))

    st.subheader("Short Straddle")
    if st.button("🔴 Deploy Straddle", use_container_width=True, type="primary"):
        with st.spinner("Deploying…"):
            r = engine.deploy_straddle(breeze_stock, expiry_str, lots, float(sl_pct))
        if r:
            st.success(f"Straddle: {r.strategy_id}")
        else:
            st.error("Failed")

    st.subheader("Short Strangle")
    target_delta = st.slider("Target Delta", 0.05, 0.40, 0.15, 0.01)
    if st.button("🔴 Deploy Strangle", use_container_width=True, type="primary"):
        with st.spinner("Deploying…"):
            r = engine.deploy_strangle(
                breeze_stock, target_delta, expiry_str, lots, float(sl_pct)
            )
        if r:
            st.success(f"Strangle: {r.strategy_id}")
        else:
            st.error("Failed")

    st.divider()
    st.subheader("🚨 Kill Switch")
    confirm = st.checkbox("I understand this will close ALL positions")
    if st.button("⚠️ PANIC EXIT", use_container_width=True,
                 type="primary", disabled=not confirm):
        engine.trigger_panic_exit()
        st.error("🚨 PANIC EXIT TRIGGERED")

    st.divider()
    with st.expander("📋 Config"):
        st.json(Config.dump())


# ═══════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════

tab_chain, tab_limit, tab_positions, tab_logs = st.tabs([
    "📈 Option Chain", "📝 Limit Orders", "📋 Positions", "📝 Logs"
])


# ── Tab 1: Option Chain ─────────────────────────────────────

with tab_chain:
    st.subheader(f"Option Chain — {stock_sel}")
    chain_box = st.empty()

    try:
        chain_data = engine.get_chain_with_greeks(breeze_stock, expiry_str)
    except Exception as e:
        chain_data = []
        st.warning(f"Chain error: {e}")

    if chain_data:
        df = pd.DataFrame(chain_data)
        df_ce = df[df["right"] == "CALL"].copy()
        df_pe = df[df["right"] == "PUT"].copy()

        df_ce = df_ce.rename(columns={
            "ltp": "CE LTP", "bid": "CE Bid", "ask": "CE Ask",
            "iv": "CE IV%", "delta": "CE Δ", "theta": "CE Θ",
            "vega": "CE V", "oi": "CE OI", "volume": "CE Vol",
        })[["strike", "CE OI", "CE Vol", "CE IV%", "CE Δ",
            "CE Θ", "CE V", "CE Bid", "CE LTP", "CE Ask"]]

        df_pe = df_pe.rename(columns={
            "ltp": "PE LTP", "bid": "PE Bid", "ask": "PE Ask",
            "iv": "PE IV%", "delta": "PE Δ", "theta": "PE Θ",
            "vega": "PE V", "oi": "PE OI", "volume": "PE Vol",
        })[["strike", "PE Bid", "PE LTP", "PE Ask",
            "PE Δ", "PE Θ", "PE V", "PE IV%", "PE Vol", "PE OI"]]

        merged = pd.merge(df_ce, df_pe, on="strike", how="outer").sort_values("strike")
        merged = merged.fillna(0)

        # Highlight ATM
        if spot > 0:
            gap = Config.strike_gap(breeze_stock)
            from utils import atm_strike as calc_atm
            atm_val = calc_atm(spot, gap)

            def highlight_atm(row):
                if row["strike"] == atm_val:
                    return ["background-color: #1a3a5c"] * len(row)
                return [""] * len(row)

            styled = merged.style.apply(highlight_atm, axis=1)
            chain_box.dataframe(styled, use_container_width=True, height=500, hide_index=True)
        else:
            chain_box.dataframe(merged, use_container_width=True, height=500, hide_index=True)
    else:
        chain_box.info("Loading chain…")


# ── Tab 2: Limit Order Screen ───────────────────────────────

with tab_limit:
    st.subheader("📝 Manual Limit Order — Sell")
    st.caption("Select a strike, enter your limit price, and sell.")

    lcol1, lcol2, lcol3 = st.columns(3)

    with lcol1:
        # Build strike options from chain
        if chain_data:
            all_strikes = sorted(set(item["strike"] for item in chain_data))
        else:
            gap = Config.strike_gap(breeze_stock)
            center = int(spot) if spot > 0 else 24000
            all_strikes = [center + i * gap for i in range(-10, 11)]

        selected_strike = st.selectbox(
            "Strike Price",
            all_strikes,
            index=len(all_strikes) // 2 if all_strikes else 0,
        )

    with lcol2:
        selected_right = st.radio("Option Type", ["CALL", "PUT"], horizontal=True)
        opt_right = OptionRight.CALL if selected_right == "CALL" else OptionRight.PUT

    with lcol3:
        limit_lots = st.number_input("Lots (Limit)", 1, 50, 1, key="limit_lots")
        limit_sl_pct = st.slider("SL % (Limit)", 10, 200,
                                  int(Config.SL_PERCENTAGE), key="limit_sl")

    # Show current prices for the selected strike
    if chain_data:
        matching = [
            c for c in chain_data
            if c["strike"] == selected_strike and c["right"] == selected_right
        ]
        if matching:
            m = matching[0]
            pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns(5)
            pcol1.metric("Bid", f"₹{m['bid']:.2f}")
            pcol2.metric("LTP", f"₹{m['ltp']:.2f}")
            pcol3.metric("Ask", f"₹{m['ask']:.2f}")
            pcol4.metric("IV%", f"{m['iv']:.1f}")
            pcol5.metric("Delta", f"{m['delta']:.4f}")

            default_price = m["bid"] if m["bid"] > 0 else m["ltp"]
        else:
            default_price = 100.0
    else:
        default_price = 100.0

    limit_price = st.number_input(
        "Limit Price (₹)", min_value=0.05, value=float(round(default_price, 2)),
        step=0.05, format="%.2f", key="limit_price_input",
    )

    lot_sz = Config.lot_size(breeze_stock)
    total_qty = lot_sz * limit_lots
    notional = limit_price * total_qty

    st.info(
        f"**Order Preview:** SELL {total_qty} × "
        f"{breeze_stock} {int(selected_strike)} {selected_right} "
        f"@ ₹{limit_price:.2f}  |  "
        f"Premium = ₹{notional:,.2f}  |  "
        f"SL = ₹{round(limit_price * (1 + limit_sl_pct / 100), 2):,.2f}"
    )

    if st.button("📤 Place Limit Sell Order", use_container_width=True, type="primary"):
        with st.spinner("Placing limit order…"):
            r = engine.deploy_limit_sell(
                stock_code=breeze_stock,
                strike=float(selected_strike),
                right=opt_right,
                expiry=expiry_str,
                lots=limit_lots,
                limit_price=limit_price,
                sl_pct=float(limit_sl_pct),
            )
        if r:
            st.success(f"Order placed: {r.strategy_id}")
        else:
            st.error("Order failed — check logs")


# ── Tab 3: Active Positions ─────────────────────────────────

with tab_positions:
    st.subheader("Active Strategies & Legs")
    strategies = state.get_strategies()

    if not strategies:
        st.info("No active strategies.")
    else:
        for strat in strategies:
            emoji = {
                StrategyStatus.ACTIVE: "🟢",
                StrategyStatus.DEPLOYING: "🟡",
                StrategyStatus.PARTIAL_EXIT: "🟠",
                StrategyStatus.CLOSED: "⚫",
                StrategyStatus.ERROR: "🔴",
            }.get(strat.status, "⚪")

            pnl = strat.compute_total_pnl()
            pnl_c = "green" if pnl >= 0 else "red"

            with st.expander(
                f"{emoji} {strat.strategy_type.value.replace('_', ' ').upper()} "
                f"[{strat.strategy_id}] — "
                f"P&L: :{pnl_c}[₹{pnl:+,.2f}]",
                expanded=(strat.status == StrategyStatus.ACTIVE),
            ):
                rows = []
                for leg in strat.legs:
                    rows.append({
                        "ID": leg.leg_id[:6],
                        "Type": f"{leg.right.value.upper()[:1]}E",
                        "Strike": int(leg.strike_price),
                        "Entry": round(leg.entry_price, 2),
                        "LTP": round(leg.current_price, 2),
                        "SL": round(leg.sl_price, 2),
                        "P&L": round(leg.pnl, 2),
                        "Δ": round(leg.greeks.delta, 4),
                        "Θ": round(leg.greeks.theta, 2),
                        "IV%": round(leg.greeks.iv * 100, 1),
                        "Status": leg.status.value,
                    })

                if rows:
                    df_legs = pd.DataFrame(rows)

                    def _color_pnl(v):
                        if isinstance(v, (int, float)):
                            return "color: #00ff88" if v >= 0 else "color: #ff4444"
                        return ""

                    def _color_status(v):
                        m = {
                            "active": "background: #0d4d2b; color: #00ff88",
                            "squared_off": "background: #333; color: #999",
                            "sl_triggered": "background: #4d0d0d; color: #ff4444",
                            "error": "background: #4d0d0d; color: #ff4444",
                            "entering": "background: #4d4d0d; color: #ffff44",
                        }
                        return m.get(v, "")

                    styled = df_legs.style.applymap(
                        _color_pnl, subset=["P&L"]
                    ).applymap(
                        _color_status, subset=["Status"]
                    )
                    st.dataframe(styled, use_container_width=True, hide_index=True)

                mc1, mc2 = st.columns(2)
                mc1.metric("Strategy P&L", f"₹{pnl:+,.2f}")
                mc2.metric("Status", strat.status.value.upper())


# ── Tab 4: Logs ──────────────────────────────────────────────

with tab_logs:
    st.subheader("System Logs")
    lcol1, lcol2 = st.columns([3, 2])

    with lcol1:
        st.markdown("**Agent Thoughts & Events**")
        logs = state.get_logs(150)
        if logs:
            lines = []
            for e in reversed(logs):
                color = {
                    "INFO": "#88ccff", "WARN": "#ffcc44",
                    "ERROR": "#ff4444", "CRIT": "#ff0000",
                }.get(e.level, "#cccccc")
                lines.append(
                    f'<span style="color:{color}">[{e.timestamp}] '
                    f'{e.level:5s}</span> │ '
                    f'<span style="color:#888">{e.source:12s}</span> │ '
                    f'{e.message}'
                )
            st.markdown(
                '<div class="log-box">' + "<br>".join(lines) + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No logs yet.")

    with lcol2:
        st.markdown("**Order History**")
        try:
            ologs = engine.db.get_recent_order_logs(30)
            if ologs:
                odf = pd.DataFrame(ologs)
                show = [c for c in ["timestamp", "action", "status", "price",
                                     "quantity", "order_id", "message"]
                        if c in odf.columns]
                st.dataframe(odf[show], use_container_width=True,
                             hide_index=True, height=400)
            else:
                st.info("No orders.")
        except Exception as e:
            st.warning(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════

st.divider()
fc = st.columns(5)
fc[0].caption(f"Mode: **{Config.TRADING_MODE.upper()}**")
fc[1].caption(f"DB: `{Config.DB_PATH}`")

try:
    from rate_limiter import get_rate_limiter
    fc[2].caption(f"Rate: {get_rate_limiter().remaining}/{Config.API_RATE_LIMIT}")
except Exception:
    fc[2].caption("Rate: N/A")

active_n = sum(1 for s in strategies if s.status == StrategyStatus.ACTIVE)
fc[3].caption(f"Active: {active_n}")

if state.panic_triggered:
    fc[4].markdown("🚨 **PANIC**")
else:
    fc[4].caption(f"Engine: {'✓' if state.engine_running else '✗'}")
