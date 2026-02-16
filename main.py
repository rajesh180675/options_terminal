# ═══════════════════════════════════════════════════════════════
# FILE: main.py  (Streamlit Dashboard)
# ═══════════════════════════════════════════════════════════════
"""
streamlit run main.py

Tabs:
  1. Option Chain (CE / PE with Greeks)
  2. Limit Order Screen (manual sell)
  3. Active Positions (with trailing SL status)
  4. Portfolio Dashboard (net Greeks, MTM chart)
  5. Logs
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime

st.set_page_config(page_title="Options Terminal", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=1500, limit=None, key="refresh")
except ImportError:
    pass

from app_config import Config, INSTRUMENTS
from utils import breeze_expiry, next_weekly_expiry, is_expiry_day
from models import LegStatus, StrategyStatus, OptionRight


# ── Engine singleton ─────────────────────────────────────────

@st.cache_resource
def boot_engine():
    from trading_engine import TradingEngine
    eng = TradingEngine()
    ok = eng.start()
    return eng, ok


engine, engine_ok = boot_engine()
state = engine.state


# ── CSS ──────────────────────────────────────────────────────

st.markdown("""<style>
.pnl-pos{background:linear-gradient(135deg,#0d4d2b,#1a6b3f);color:#00ff88;
  padding:14px 24px;border-radius:10px;font-size:28px;font-weight:bold;text-align:center}
.pnl-neg{background:linear-gradient(135deg,#4d0d0d,#6b1a1a);color:#ff4444;
  padding:14px 24px;border-radius:10px;font-size:28px;font-weight:bold;text-align:center}
.badge{padding:4px 12px;border-radius:12px;font-size:12px;font-weight:600;display:inline-block}
.bg{background:#0d4d2b;color:#00ff88}.br{background:#4d0d0d;color:#ff4444}
.by{background:#4d4d0d;color:#ffff44}.bb{background:#0d2d4d;color:#44aaff}
.logbox{font-family:'Courier New',monospace;font-size:11px;background:#0d1117;
  padding:10px;border-radius:6px;max-height:400px;overflow-y:auto;border:1px solid #21262d}
</style>""", unsafe_allow_html=True)


# ── HEADER ───────────────────────────────────────────────────

mtm = state.get_total_mtm()
pc = "pnl-pos" if mtm >= 0 else "pnl-neg"
ps = "+" if mtm >= 0 else ""

hc = st.columns([3, 1, 1, 1, 1, 1])
hc[0].markdown(f'<div class="{pc}">MTM: ₹{ps}{mtm:,.2f}</div>', unsafe_allow_html=True)
hc[1].markdown(f'<span class="badge {"bg" if state.connected else "br"}">API {"✓" if state.connected else "✗"}</span>', unsafe_allow_html=True)
hc[2].markdown(f'<span class="badge {"bg" if state.ws_connected else "br"}">WS {"✓" if state.ws_connected else "✗"}</span>', unsafe_allow_html=True)

# Show spot for default stock
dc = Config.breeze_code(Config.DEFAULT_STOCK)
spot = state.get_spot(dc)
hc[3].markdown(f'<span class="badge by">{Config.DEFAULT_STOCK}: ₹{spot:,.1f}</span>', unsafe_allow_html=True)

# Portfolio Greeks
pg = state.get_portfolio_greeks()
hc[4].markdown(f'<span class="badge bb">Δ {pg.delta:+.1f} Θ {pg.theta:+.1f}</span>', unsafe_allow_html=True)

mode_c = "br" if Config.is_live() else "by"
hc[5].markdown(f'<span class="badge {mode_c}">{Config.TRADING_MODE.upper()}</span>', unsafe_allow_html=True)

st.divider()


# ── SIDEBAR ──────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Deploy")
    inst_name = st.selectbox("Instrument", list(INSTRUMENTS.keys()))
    inst = Config.instrument(inst_name)
    bc = inst["breeze_code"]
    exc = inst["exchange"]
    exp_dt = next_weekly_expiry(inst_name)
    exp_str = breeze_expiry(exp_dt)

    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    st.caption(f"📅 Expiry: **{exp_dt.strftime('%d-%b-%Y')} "
               f"({weekday_names[inst['expiry_weekday']]})**")
    st.caption(f"🏢 Exchange: `{exc}` | Code: `{bc}` | Lot: {inst['lot_size']}")

    if is_expiry_day(inst_name):
        st.warning("⚠️ TODAY IS EXPIRY DAY")

    lots = st.number_input("Lots", 1, 50, 1)
    sl_pct = st.slider("SL %", 10, 200, int(Config.SL_PERCENTAGE))

    st.subheader("Short Straddle")
    if st.button("🔴 Deploy Straddle", use_container_width=True, type="primary"):
        with st.spinner("Deploying…"):
            r = engine.deploy_straddle(inst_name, exp_str, lots, float(sl_pct))
        st.success(f"OK: {r.strategy_id}") if r else st.error("Failed")

    st.subheader("Short Strangle")
    tdelta = st.slider("Target Delta", 0.05, 0.40, 0.15, 0.01)
    if st.button("🔴 Deploy Strangle", use_container_width=True, type="primary"):
        with st.spinner("Deploying…"):
            r = engine.deploy_strangle(inst_name, tdelta, exp_str, lots, float(sl_pct))
        st.success(f"OK: {r.strategy_id}") if r else st.error("Failed")

    st.divider()
    st.subheader("🚨 Kill Switch")
    confirm = st.checkbox("Confirm close ALL positions")
    if st.button("⚠️ PANIC EXIT", use_container_width=True, type="primary", disabled=not confirm):
        engine.trigger_panic()
        st.error("🚨 PANIC TRIGGERED")

    st.divider()
    with st.expander("📋 Config"):
        st.json(Config.dump())

    try:
        from rate_limiter import get_rate_limiter
        st.caption(f"Rate: {get_rate_limiter().remaining}/{Config.API_RATE_LIMIT}")
    except Exception:
        pass


# ── TABS ─────────────────────────────────────────────────────

tab_chain, tab_limit, tab_pos, tab_dash, tab_logs = st.tabs([
    "📈 Chain", "📝 Limit Order", "📋 Positions", "📊 Dashboard", "📝 Logs"
])

# ── Tab 1: Chain ─────────────────────────────────────────────

with tab_chain:
    st.subheader(f"Option Chain — {inst_name} ({exc})")
    try:
        chain = engine.get_chain(inst_name, exp_str)
    except Exception as e:
        chain = []; st.warning(f"Error: {e}")

    if chain:
        df = pd.DataFrame(chain)
        dfc = df[df["right"] == "CALL"].rename(columns={
            "ltp": "CE LTP", "bid": "CE Bid", "ask": "CE Ask",
            "iv": "CE IV%", "delta": "CE Δ", "theta": "CE Θ",
            "vega": "CE V", "oi": "CE OI"
        })[["strike", "CE OI", "CE IV%", "CE Δ", "CE Θ", "CE V", "CE Bid", "CE LTP", "CE Ask"]]
        dfp = df[df["right"] == "PUT"].rename(columns={
            "ltp": "PE LTP", "bid": "PE Bid", "ask": "PE Ask",
            "iv": "PE IV%", "delta": "PE Δ", "theta": "PE Θ",
            "vega": "PE V", "oi": "PE OI"
        })[["strike", "PE Bid", "PE LTP", "PE Ask", "PE Δ", "PE Θ", "PE V", "PE IV%", "PE OI"]]

        merged = pd.merge(dfc, dfp, on="strike", how="outer").sort_values("strike").fillna(0)

        if spot > 0:
            from utils import atm_strike
            atm_v = atm_strike(spot, inst["strike_gap"])
            def hl(row):
                return ["background-color:#1a3a5c"] * len(row) if row["strike"] == atm_v else [""] * len(row)
            st.dataframe(merged.style.apply(hl, axis=1),
                         use_container_width=True, height=500, hide_index=True)
        else:
            st.dataframe(merged, use_container_width=True, height=500, hide_index=True)
    else:
        st.info("Loading…")


# ── Tab 2: Limit Order ──────────────────────────────────────

with tab_limit:
    st.subheader(f"📝 Manual Limit Sell — {inst_name}")

    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        if chain:
            strikes = sorted(set(c["strike"] for c in chain))
        else:
            gap = inst["strike_gap"]
            ctr = int(spot) if spot > 0 else 24000
            strikes = [ctr + i * gap for i in range(-10, 11)]
        sel_strike = st.selectbox("Strike", strikes,
                                  index=len(strikes)//2 if strikes else 0)
    with lc2:
        sel_right_str = st.radio("Type", ["CALL", "PUT"], horizontal=True)
        sel_right = OptionRight.CALL if sel_right_str == "CALL" else OptionRight.PUT
    with lc3:
        l_lots = st.number_input("Lots", 1, 50, 1, key="ll")
        l_sl = st.slider("SL%", 10, 200, int(Config.SL_PERCENTAGE), key="lsl")

    # Show current prices
    default_px = 100.0
    if chain:
        match = [c for c in chain if c["strike"] == sel_strike and c["right"] == sel_right_str]
        if match:
            m = match[0]
            mc = st.columns(5)
            mc[0].metric("Bid", f"₹{m['bid']:.2f}")
            mc[1].metric("LTP", f"₹{m['ltp']:.2f}")
            mc[2].metric("Ask", f"₹{m['ask']:.2f}")
            mc[3].metric("IV%", f"{m['iv']:.1f}")
            mc[4].metric("Delta", f"{m['delta']:.4f}")
            default_px = m["bid"] if m["bid"] > 0 else m["ltp"]

    limit_px = st.number_input("Limit Price ₹", 0.05, 50000.0,
                               float(round(default_px, 2)), 0.05,
                               format="%.2f", key="lpx")

    qty = inst["lot_size"] * l_lots
    prem = limit_px * qty
    sl_val = round(limit_px * (1 + l_sl / 100), 2)

    st.info(f"**SELL {qty} × {bc} {sel_strike} {sel_right_str}** "
            f"@ ₹{limit_px:.2f} | Premium ₹{prem:,.0f} | SL ₹{sl_val}")

    if st.button("📤 Place Limit Sell", use_container_width=True, type="primary"):
        with st.spinner("Placing…"):
            r = engine.deploy_limit_sell(
                inst_name, float(sel_strike), sel_right,
                exp_str, l_lots, limit_px, float(l_sl))
        st.success(f"OK: {r.strategy_id}") if r else st.error("Failed")


# ── Tab 3: Positions ─────────────────────────────────────────

with tab_pos:
    st.subheader("Active Strategies")
    strategies = state.get_strategies()
    if not strategies:
        st.info("No active strategies.")
    else:
        for s in strategies:
            emoji = {"active": "🟢", "deploying": "🟡", "partial_exit": "🟠",
                     "closed": "⚫", "error": "🔴"}.get(s.status.value, "⚪")
            pnl = s.compute_total_pnl()
            pc = "green" if pnl >= 0 else "red"
            with st.expander(
                f"{emoji} {s.strategy_type.value.replace('_', ' ').upper()} "
                f"[{s.strategy_id}] — :{pc}[₹{pnl:+,.2f}]",
                expanded=(s.status == StrategyStatus.ACTIVE)):

                rows = []
                for leg in s.legs:
                    trail_info = "🔄" if leg.trailing_active else ""
                    rows.append({
                        "Leg": leg.leg_id[:6],
                        "Type": f"{leg.right.value[0].upper()}E",
                        "Strike": int(leg.strike_price),
                        "Entry": round(leg.entry_price, 2),
                        "LTP": round(leg.current_price, 2),
                        "SL": f"{round(leg.sl_price, 2)} {trail_info}",
                        "Low": round(leg.lowest_price, 2),
                        "P&L": round(leg.pnl, 2),
                        "Δ": round(leg.greeks.delta, 4),
                        "Θ": round(leg.greeks.theta, 2),
                        "IV%": round(leg.greeks.iv * 100, 1),
                        "Status": leg.status.value,
                    })
                if rows:
                    ldf = pd.DataFrame(rows)
                    def cpnl(v):
                        return "color:#00ff88" if isinstance(v, (int, float)) and v >= 0 else "color:#ff4444" if isinstance(v, (int, float)) else ""
                    def cst(v):
                        m = {"active": "background:#0d4d2b;color:#00ff88",
                             "squared_off": "background:#333;color:#999",
                             "sl_triggered": "background:#4d0d0d;color:#ff4444",
                             "error": "background:#4d0d0d;color:#ff4444",
                             "entering": "background:#4d4d0d;color:#ffff44"}
                        return m.get(v, "")
                    styled = ldf.style.applymap(cpnl, subset=["P&L"]).applymap(cst, subset=["Status"])
                    st.dataframe(styled, use_container_width=True, hide_index=True)

                # Net Greeks for this strategy
                ng = s.net_greeks
                gc = st.columns(4)
                gc[0].metric("Net Δ", f"{ng.delta:+.2f}")
                gc[1].metric("Net Γ", f"{ng.gamma:+.4f}")
                gc[2].metric("Net Θ", f"₹{ng.theta:+.2f}")
                gc[3].metric("Net V", f"{ng.vega:+.2f}")


# ── Tab 4: Dashboard ─────────────────────────────────────────

with tab_dash:
    st.subheader("📊 Portfolio Dashboard")

    dc1, dc2 = st.columns(2)

    with dc1:
        st.markdown("**Portfolio Greeks**")
        pg = state.get_portfolio_greeks()
        gc = st.columns(4)
        gc[0].metric("Net Delta", f"{pg.delta:+.2f}",
                     help="Directional exposure. 0 = neutral.")
        gc[1].metric("Net Gamma", f"{pg.gamma:+.4f}",
                     help="Rate of delta change.")
        gc[2].metric("Net Theta", f"₹{pg.theta:+.2f}/day",
                     help="Daily time decay (positive = earning).")
        gc[3].metric("Net Vega", f"{pg.vega:+.2f}",
                     help="Volatility exposure.")

        # Delta alert
        if abs(pg.delta) > 50:
            st.warning(f"⚠️ Portfolio delta ({pg.delta:+.1f}) is high. Consider adjusting.")

    with dc2:
        st.markdown("**Intraday MTM**")
        hist = state.get_mtm_history()
        if hist and len(hist) > 1:
            mdf = pd.DataFrame(hist)
            st.line_chart(mdf.set_index("time")["mtm"], height=250)
        else:
            st.info("MTM chart will appear after a few seconds of data.")

    st.divider()

    # Config summary
    st.markdown("**Active Configuration**")
    cc = st.columns(4)
    cc[0].markdown(f"**SL:** {Config.SL_PERCENTAGE}%")
    cc[1].markdown(f"**Trail:** {'ON' if Config.TRAIL_ENABLED else 'OFF'} "
                   f"(after {Config.TRAIL_ACTIVATION_PCT}% profit, trail {Config.TRAIL_SL_PCT}%)")
    cc[2].markdown(f"**Auto-exit:** {'ON' if Config.AUTO_EXIT_ENABLED else 'OFF'} "
                   f"at {Config.AUTO_EXIT_HOUR}:{Config.AUTO_EXIT_MINUTE:02d}")
    cc[3].markdown(f"**Global Max Loss:** ₹{Config.GLOBAL_MAX_LOSS:,.0f}")


# ── Tab 5: Logs ──────────────────────────────────────────────

with tab_logs:
    st.subheader("System Logs")
    lc1, lc2 = st.columns([3, 2])

    with lc1:
        logs = state.get_logs(150)
        if logs:
            lines = []
            for e in reversed(logs):
                clr = {"INFO": "#88ccff", "WARN": "#ffcc44",
                       "ERROR": "#ff4444", "CRIT": "#ff0000"}.get(e.level, "#ccc")
                lines.append(
                    f'<span style="color:{clr}">[{e.timestamp}] {e.level:5s}</span>'
                    f' │ <span style="color:#888">{e.source:8s}</span> │ {e.message}')
            st.markdown('<div class="logbox">' + "<br>".join(lines) + '</div>',
                        unsafe_allow_html=True)
        else:
            st.info("No logs.")

    with lc2:
        st.markdown("**Order History**")
        try:
            ol = engine.db.get_recent_logs(30)
            if ol:
                odf = pd.DataFrame(ol)
                show = [c for c in ["timestamp", "action", "status", "price",
                                     "quantity", "order_id", "message"] if c in odf.columns]
                st.dataframe(odf[show], use_container_width=True, hide_index=True, height=400)
            else:
                st.info("No orders.")
        except Exception as e:
            st.warning(str(e))


# ── Footer ───────────────────────────────────────────────────

st.divider()
fc = st.columns(5)
fc[0].caption(f"Mode: **{Config.TRADING_MODE.upper()}**")
fc[1].caption(f"DB: `{Config.DB_PATH}`")
active_n = sum(1 for s in strategies if s.status == StrategyStatus.ACTIVE)
fc[2].caption(f"Active: {active_n}")
fc[3].caption(f"Time: {datetime.now().strftime('%H:%M:%S')}")
fc[4].markdown("🚨 **PANIC**" if state.panic_triggered else
               f"Engine: {'✓' if state.engine_running else '✗'}")
