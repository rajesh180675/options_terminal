# ═══════════════════════════════════════════════════════════════
# FILE: main.py  (COMPLETE REWRITE — 6 professional tabs)
# ═══════════════════════════════════════════════════════════════
"""
streamlit run main.py

Tabs:
  1. Chain — Option chain with Greeks
  2. Limit Order — Manual sell with margin preview
  3. Positions — Strategies + broker sync + EXIT BUTTONS
  4. Broker — Order book + Trade book + Funds + Reconciliation
  5. Dashboard — MTM chart, portfolio Greeks
  6. Logs — Agent thoughts + order history
"""

import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Options Terminal", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=1500, limit=None, key="r")
except ImportError:
    pass

from app_config import Config, INSTRUMENTS
from utils import breeze_expiry, next_weekly_expiry, is_expiry_day, atm_strike
from models import LegStatus, StrategyStatus, OptionRight


@st.cache_resource
def boot():
    from trading_engine import TradingEngine
    e = TradingEngine()
    return e, e.start()

engine, ok = boot()
state = engine.state

# ── CSS ──────────────────────────────────────────────────────

st.markdown("""<style>
.pnl-pos{background:linear-gradient(135deg,#0d4d2b,#1a6b3f);color:#00ff88;
  padding:12px 20px;border-radius:10px;font-size:26px;font-weight:bold;text-align:center}
.pnl-neg{background:linear-gradient(135deg,#4d0d0d,#6b1a1a);color:#ff4444;
  padding:12px 20px;border-radius:10px;font-size:26px;font-weight:bold;text-align:center}
.b{padding:3px 10px;border-radius:10px;font-size:11px;font-weight:600;display:inline-block}
.bg{background:#0d4d2b;color:#00ff88}.br{background:#4d0d0d;color:#ff4444}
.by{background:#4d4d0d;color:#ffff44}.bb{background:#0d2d4d;color:#44aaff}
.lb{font-family:monospace;font-size:11px;background:#0d1117;padding:8px;
  border-radius:6px;max-height:380px;overflow-y:auto;border:1px solid #21262d}
</style>""", unsafe_allow_html=True)


# ── HEADER ───────────────────────────────────────────────────

mtm = state.get_total_mtm()
hc = st.columns([3, 1, 1, 1, 1, 1])
pc = "pnl-pos" if mtm >= 0 else "pnl-neg"
hc[0].markdown(f'<div class="{pc}">MTM: ₹{"+" if mtm>=0 else ""}{mtm:,.2f}</div>', unsafe_allow_html=True)
hc[1].markdown(f'<span class="b {"bg" if state.connected else "br"}">API {"✓" if state.connected else "✗"}</span>', unsafe_allow_html=True)
hc[2].markdown(f'<span class="b {"bg" if state.ws_connected else "br"}">WS {"✓" if state.ws_connected else "✗"}</span>', unsafe_allow_html=True)

dc = Config.breeze_code(Config.DEFAULT_STOCK)
spot = state.get_spot(dc)
hc[3].markdown(f'<span class="b by">{Config.DEFAULT_STOCK}: ₹{spot:,.1f}</span>', unsafe_allow_html=True)

# Funds in header
funds = engine.broker.get_funds()
margin_txt = f"₹{funds.free_margin:,.0f}" if funds else "N/A"
hc[4].markdown(f'<span class="b bb">Margin: {margin_txt}</span>', unsafe_allow_html=True)

hc[5].markdown(f'<span class="b {"br" if Config.is_live() else "by"}">{Config.TRADING_MODE.upper()}</span>', unsafe_allow_html=True)
st.divider()


# ── SIDEBAR ──────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Deploy")
    inst_name = st.selectbox("Instrument", list(INSTRUMENTS.keys()))
    inst = Config.instrument(inst_name)
    bc = inst["breeze_code"]; exc = inst["exchange"]
    exp_dt = next_weekly_expiry(inst_name)
    exp_str = breeze_expiry(exp_dt)
    wd = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    st.caption(f"📅 {exp_dt.strftime('%d-%b-%Y')} ({wd[inst['expiry_weekday']]})")
    st.caption(f"🔑 {exc}/{bc} lot={inst['lot_size']}")
    if is_expiry_day(inst_name): st.warning("⚠️ EXPIRY DAY")

    lots = st.number_input("Lots", 1, 50, 1)
    sl_pct = st.slider("SL%", 10, 200, int(Config.SL_PERCENTAGE))

    st.subheader("Straddle")
    if st.button("🔴 Deploy Straddle", use_container_width=True, type="primary"):
        with st.spinner("..."): r = engine.deploy_straddle(inst_name, exp_str, lots, float(sl_pct))
        st.success(f"OK: {r.strategy_id}") if r else st.error("Failed — check margin/logs")

    st.subheader("Strangle")
    td = st.slider("Target Δ", 0.05, 0.40, 0.15, 0.01)
    if st.button("🔴 Deploy Strangle", use_container_width=True, type="primary"):
        with st.spinner("..."): r = engine.deploy_strangle(inst_name, td, exp_str, lots, float(sl_pct))
        st.success(f"OK: {r.strategy_id}") if r else st.error("Failed")

    st.divider()
    st.subheader("🚨 Kill Switch")
    if st.button("⚠️ PANIC EXIT", use_container_width=True, type="primary",
                 disabled=not st.checkbox("Confirm close ALL")):
        engine.trigger_panic(); st.error("🚨 PANIC")

    st.divider()
    with st.expander("Config"): st.json(Config.dump())
    try:
        from rate_limiter import get_rate_limiter
        st.caption(f"Rate: {get_rate_limiter().remaining}/{Config.API_RATE_LIMIT}")
    except: pass


# ── TABS ─────────────────────────────────────────────────────

t1, t2, t3, t4, t5, t6 = st.tabs([
    "📈 Chain", "📝 Limit", "📋 Positions",
    "🏦 Broker", "📊 Dashboard", "📝 Logs",
])


# ── TAB 1: CHAIN ────────────────────────────────────────────

with t1:
    st.subheader(f"Option Chain — {inst_name} ({exc})")
    try: chain_data = engine.get_chain(inst_name, exp_str)
    except: chain_data = []

    if chain_data:
        df = pd.DataFrame(chain_data)
        dfc = df[df["right"]=="CALL"].rename(columns={
            "ltp":"CE","bid":"CE Bid","ask":"CE Ask","iv":"CE IV%",
            "delta":"CE Δ","theta":"CE Θ","oi":"CE OI"
        })[["strike","CE OI","CE IV%","CE Δ","CE Θ","CE Bid","CE","CE Ask"]]
        dfp = df[df["right"]=="PUT"].rename(columns={
            "ltp":"PE","bid":"PE Bid","ask":"PE Ask","iv":"PE IV%",
            "delta":"PE Δ","theta":"PE Θ","oi":"PE OI"
        })[["strike","PE Bid","PE","PE Ask","PE Δ","PE Θ","PE IV%","PE OI"]]
        m = pd.merge(dfc, dfp, on="strike", how="outer").sort_values("strike").fillna(0)
        if spot > 0:
            atm_v = atm_strike(spot, inst["strike_gap"])
            def hl(row):
                return ["background-color:#1a3a5c"]*len(row) if row["strike"]==atm_v else [""]*len(row)
            st.dataframe(m.style.apply(hl, axis=1), use_container_width=True, height=500, hide_index=True)
        else:
            st.dataframe(m, use_container_width=True, height=500, hide_index=True)
    else:
        st.info("Loading...")


# ── TAB 2: LIMIT ORDER ──────────────────────────────────────

with t2:
    st.subheader(f"📝 Limit Sell — {inst_name}")
    c1,c2,c3 = st.columns(3)
    with c1:
        if chain_data: strikes = sorted(set(c["strike"] for c in chain_data))
        else:
            g=inst["strike_gap"]; ct=int(spot) if spot>0 else 24000
            strikes=[ct+i*g for i in range(-10,11)]
        sel_s = st.selectbox("Strike", strikes, index=len(strikes)//2)
    with c2:
        sr = st.radio("Type", ["CALL","PUT"], horizontal=True)
        opt_r = OptionRight.CALL if sr=="CALL" else OptionRight.PUT
    with c3:
        ll = st.number_input("Lots", 1, 50, 1, key="ll")
        lsl = st.slider("SL%", 10, 200, int(Config.SL_PERCENTAGE), key="lsl")

    dpx = 100.0
    if chain_data:
        mm = [c for c in chain_data if c["strike"]==sel_s and c["right"]==sr]
        if mm:
            mc=st.columns(5)
            mc[0].metric("Bid", f"₹{mm[0]['bid']:.2f}")
            mc[1].metric("LTP", f"₹{mm[0]['ltp']:.2f}")
            mc[2].metric("Ask", f"₹{mm[0]['ask']:.2f}")
            mc[3].metric("IV%", f"{mm[0]['iv']:.1f}")
            mc[4].metric("Δ", f"{mm[0]['delta']:.4f}")
            dpx = mm[0]["bid"] if mm[0]["bid"]>0 else mm[0]["ltp"]

    lpx = st.number_input("Limit ₹", 0.05, 50000.0, float(round(dpx,2)), 0.05, format="%.2f", key="lpx")
    qty = inst["lot_size"]*ll
    st.info(f"**SELL {qty} × {bc} {sel_s} {sr}** @ ₹{lpx:.2f} | Premium ₹{lpx*qty:,.0f} | SL ₹{round(lpx*(1+lsl/100),2)}")

    # Margin check display
    ok_m, req_m, msg_m = engine.broker.check_margin(bc, exc, float(sel_s), opt_r.value, exp_str, qty, lpx, "sell")
    if ok_m:
        st.success(f"✅ Margin: {msg_m}")
    else:
        st.error(f"❌ Margin: {msg_m}")

    if st.button("📤 Place Limit Sell", use_container_width=True, type="primary", disabled=not ok_m):
        with st.spinner("Placing..."):
            r = engine.deploy_limit_sell(inst_name, float(sel_s), opt_r, exp_str, ll, lpx, float(lsl))
        st.success(f"OK: {r.strategy_id}") if r else st.error("Failed")


# ── TAB 3: POSITIONS + EXIT BUTTONS ─────────────────────────

with t3:
    st.subheader("Positions & Exit")
    strategies = state.get_strategies()
    if not strategies:
        st.info("No active strategies.")
    else:
        for s in strategies:
            em = {"active":"🟢","deploying":"🟡","partial_exit":"🟠","closed":"⚫","error":"🔴"}.get(s.status.value,"⚪")
            pnl = s.compute_total_pnl()
            with st.expander(f"{em} {s.strategy_type.value.replace('_',' ').upper()} [{s.strategy_id}] — {'green' if pnl>=0 else 'red'}[₹{pnl:+,.2f}]", expanded=(s.status==StrategyStatus.ACTIVE)):
                for leg in s.legs:
                    lc = st.columns([1,1,1,1,1,1,1,1,2])
                    lc[0].write(f"**{leg.right.value[0].upper()}E**")
                    lc[1].write(f"K={int(leg.strike_price)}")
                    lc[2].write(f"Entry=₹{leg.entry_price:.2f}")
                    lc[3].write(f"LTP=₹{leg.current_price:.2f}")
                    lc[4].write(f"SL=₹{leg.sl_price:.2f}{'🔄' if leg.trailing_active else ''}")
                    lc[5].write(f"P&L=₹{leg.pnl:+,.0f}")
                    lc[6].write(f"Δ={leg.greeks.delta:.3f}")
                    lc[7].write(f"**{leg.status.value}**")
                    with lc[8]:
                        if leg.status == LegStatus.ACTIVE:
                            if st.button(f"🔴 Exit", key=f"exit_{leg.leg_id}", type="primary"):
                                ok = engine.exit_leg(leg.leg_id)
                                st.success("Exited") if ok else st.error("Failed")
                                st.rerun()

                ng = s.net_greeks
                gc = st.columns(4)
                gc[0].metric("Net Δ", f"{ng.delta:+.1f}")
                gc[1].metric("Net Γ", f"{ng.gamma:+.4f}")
                gc[2].metric("Net Θ", f"₹{ng.theta:+.1f}")
                gc[3].metric("Net V", f"{ng.vega:+.1f}")

    # Broker reconciliation
    recon = engine.broker.get_recon_issues()
    if recon:
        st.divider()
        st.subheader("⚠️ Reconciliation Issues")
        for issue in recon:
            st.warning(issue)


# ── TAB 4: BROKER DATA ──────────────────────────────────────

with t4:
    st.subheader("🏦 Broker Data")
    bt1, bt2, bt3, bt4 = st.tabs(["💰 Funds", "📋 Orders", "📊 Trades", "📍 Positions"])

    with bt1:
        st.markdown("**Account Funds**")
        funds = engine.broker.get_funds()
        if funds:
            fc = st.columns(4)
            fc[0].metric("Allocated", f"₹{funds.allocated:,.0f}")
            fc[1].metric("Utilized", f"₹{funds.utilized:,.0f}")
            fc[2].metric("Available", f"₹{funds.free_margin:,.0f}")
            fc[3].metric("Blocked", f"₹{funds.blocked:,.0f}")
        else:
            st.info("Funds data loading...")

        if st.button("🔄 Refresh Funds"):
            engine.broker.fetch_funds()
            st.rerun()

        # Customer details
        cust = engine.broker.get_customer_details()
        if cust:
            with st.expander("Customer Details"):
                st.json(cust)

    with bt2:
        st.markdown("**Today's Orders**")
        if st.button("🔄 Refresh Orders"):
            engine.broker.fetch_orders()
            engine.broker.fetch_orders("BFO")
            st.rerun()

        orders = engine.broker.get_orders()
        if orders:
            odf = pd.DataFrame([{
                "ID": o.order_id[-8:] if len(o.order_id) > 8 else o.order_id,
                "Stock": o.stock_code, "Strike": int(o.strike_price) if o.strike_price else "",
                "Type": o.right[:1].upper() if o.right else "",
                "Action": o.action, "OrdType": o.order_type,
                "Qty": o.quantity, "Price": o.price,
                "AvgPx": o.avg_price, "Status": o.status,
                "Time": o.order_time,
            } for o in orders])

            def color_ord_status(v):
                if v == "Executed": return "background:#0d4d2b;color:#00ff88"
                if v in ("Cancelled","Rejected"): return "background:#4d0d0d;color:#ff4444"
                if v == "Pending": return "background:#4d4d0d;color:#ffff44"
                return ""

            st.dataframe(odf.style.applymap(color_ord_status, subset=["Status"]),
                         use_container_width=True, hide_index=True, height=400)

            # Cancel/Modify pending orders
            pending = [o for o in orders if o.status == "Pending"]
            if pending:
                st.subheader("Pending Order Actions")
                for o in pending:
                    pc = st.columns([3, 1, 1])
                    pc[0].write(f"{o.order_id[-8:]}: {o.stock_code} {int(o.strike_price)} {o.action} @ ₹{o.price}")
                    if pc[1].button("❌ Cancel", key=f"canc_{o.order_id}"):
                        engine.broker.cancel_order_from_ui(o.order_id, o.exchange_code)
                        st.rerun()
                    new_px = pc[2].number_input("New₹", value=o.price, step=0.05, key=f"mod_{o.order_id}", label_visibility="collapsed")
                    if new_px != o.price:
                        if st.button(f"✏️ Modify", key=f"modbtn_{o.order_id}"):
                            engine.broker.modify_order_from_ui(o.order_id, new_px, o.exchange_code)
                            st.rerun()
        else:
            st.info("No orders today.")

    with bt3:
        st.markdown("**Today's Trades**")
        if st.button("🔄 Refresh Trades"):
            engine.broker.fetch_trades()
            engine.broker.fetch_trades("BFO")
            st.rerun()

        trades = engine.broker.get_trades()
        if trades:
            tdf = pd.DataFrame([{
                "TradeID": t.trade_id, "OrderID": t.order_id[-8:] if len(t.order_id)>8 else t.order_id,
                "Stock": t.stock_code, "Strike": int(t.strike_price) if t.strike_price else "",
                "Type": t.right[:1].upper() if t.right else "",
                "Action": t.action, "Qty": t.quantity,
                "Price": t.trade_price, "Time": t.trade_time,
            } for t in trades])
            st.dataframe(tdf, use_container_width=True, hide_index=True, height=400)

            # Realized P&L from trades
            realized = 0.0
            for t in trades:
                if t.action.lower() == "sell":
                    realized += t.trade_price * t.quantity
                else:
                    realized -= t.trade_price * t.quantity
            st.metric("Realized P&L (from trades)", f"₹{realized:+,.2f}")
        else:
            st.info("No trades today.")

    with bt4:
        st.markdown("**Broker Positions (live from API)**")
        if st.button("🔄 Refresh Positions"):
            engine.broker.fetch_positions()
            st.rerun()

        bpos = engine.broker.get_positions()
        if bpos:
            pdf = pd.DataFrame([{
                "Stock": p.stock_code, "Strike": int(p.strike_price),
                "Type": p.right[:1].upper() if p.right else "",
                "Qty": p.quantity, "Avg": p.avg_price,
                "LTP": p.ltp,
                "P&L": round((p.avg_price - p.ltp) * abs(p.quantity) if p.quantity < 0
                       else (p.ltp - p.avg_price) * p.quantity, 2),
                "Exchange": p.exchange_code,
            } for p in bpos])
            st.dataframe(pdf, use_container_width=True, hide_index=True)
        else:
            st.info("No broker positions.")


# ── TAB 5: DASHBOARD ────────────────────────────────────────

with t5:
    st.subheader("📊 Dashboard")
    d1, d2 = st.columns(2)

    with d1:
        st.markdown("**Portfolio Greeks**")
        pg = state.get_portfolio_greeks()
        gc = st.columns(4)
        gc[0].metric("Net Δ", f"{pg.delta:+.1f}", help="0 = neutral")
        gc[1].metric("Net Γ", f"{pg.gamma:+.4f}")
        gc[2].metric("Net Θ", f"₹{pg.theta:+.1f}/day")
        gc[3].metric("Net V", f"{pg.vega:+.1f}")
        if abs(pg.delta) > 50:
            st.warning(f"⚠️ Delta {pg.delta:+.1f} — consider adjusting")

    with d2:
        st.markdown("**Intraday MTM**")
        hist = state.get_mtm_history()
        if hist and len(hist) > 2:
            mdf = pd.DataFrame(hist)
            st.line_chart(mdf.set_index("time")["mtm"], height=200)
        else:
            st.info("Collecting data...")

    st.divider()
    cc = st.columns(4)
    cc[0].markdown(f"**SL:** {Config.SL_PERCENTAGE}%")
    cc[1].markdown(f"**Trail:** {'ON' if Config.TRAIL_ENABLED else 'OFF'} ({Config.TRAIL_ACTIVATION_PCT}%→{Config.TRAIL_SL_PCT}%)")
    cc[2].markdown(f"**Auto-exit:** {Config.AUTO_EXIT_HOUR}:{Config.AUTO_EXIT_MINUTE:02d}")
    cc[3].markdown(f"**Max Loss:** ₹{Config.GLOBAL_MAX_LOSS:,.0f}")

    # Strategy history
    st.divider()
    st.markdown("**Strategy History (last 50)**")
    try:
        all_s = engine.db.get_all_strategies()
        if all_s:
            hdf = pd.DataFrame([{
                "ID": s.strategy_id, "Type": s.strategy_type.value,
                "Stock": s.stock_code, "P&L": round(s.total_pnl, 2),
                "Status": s.status.value, "Created": s.created_at[:19] if s.created_at else "",
                "Closed": s.closed_at[:19] if s.closed_at else "",
            } for s in all_s])
            st.dataframe(hdf, use_container_width=True, hide_index=True)
    except: pass


# ── TAB 6: LOGS ─────────────────────────────────────────────

with t6:
    st.subheader("Logs")
    l1, l2 = st.columns([3, 2])
    with l1:
        logs = state.get_logs(150)
        if logs:
            lines = []
            for e in reversed(logs):
                c = {"INFO":"#88ccff","WARN":"#ffcc44","ERROR":"#ff4444","CRIT":"#ff0000"}.get(e.level,"#ccc")
                lines.append(f'<span style="color:{c}">[{e.timestamp}] {e.level:5s}</span> | {e.source:8s} | {e.message}')
            st.markdown('<div class="lb">'+"<br>".join(lines)+'</div>', unsafe_allow_html=True)
        else:
            st.info("No logs.")
    with l2:
        st.markdown("**Order Log**")
        try:
            ol = engine.db.get_recent_logs(30)
            if ol:
                st.dataframe(pd.DataFrame(ol)[[c for c in ["timestamp","action","status","price","quantity","order_id","message"] if c in pd.DataFrame(ol).columns]], use_container_width=True, hide_index=True, height=380)
            else: st.info("No orders.")
        except: pass


# ── FOOTER ───────────────────────────────────────────────────

st.divider()
fc = st.columns(5)
fc[0].caption(f"Mode: **{Config.TRADING_MODE.upper()}**")
fc[1].caption(f"DB: `{Config.DB_PATH}`")
fc[2].caption(f"Active: {sum(1 for s in strategies if s.status==StrategyStatus.ACTIVE)}")
fc[3].caption(f"{datetime.now().strftime('%H:%M:%S')}")
fc[4].markdown("🚨 **PANIC**" if state.panic_triggered else f"Engine: {'✓' if state.engine_running else '✗'}")
