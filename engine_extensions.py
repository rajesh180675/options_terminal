"""
engine_extensions.py  — NEW MODULE

Extends TradingEngine with new strategy types and analytics modules
WITHOUT modifying trading_engine.py.

Usage (called once in main.py after boot_engine()):
    from engine_extensions import extend_engine
    extend_engine(engine)

This adds:
  engine.quick              → QuickStrategyEngine
  engine.dashboard          → PortfolioDashboard
  engine.risk_matrix        → RiskMatrix
  engine.signals            → SignalEngine
  engine.calculator         → OptionsCalculator

Zero breaking-changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trading_engine import TradingEngine


def extend_engine(engine: "TradingEngine") -> None:
    """Attach new module instances to an existing TradingEngine."""

    # Quick Strategies
    try:
        from quick_strategies import QuickStrategyEngine
        engine.quick = QuickStrategyEngine(
            session=engine.session,
            order_mgr=engine.omgr,
            db=engine.db,
            state=engine.state,
        )
    except Exception as e:
        engine.quick = None
        engine.state.add_log("WARN", "Ext", f"QuickStrategyEngine unavailable: {e}")

    # Portfolio Dashboard
    try:
        from portfolio_dashboard import PortfolioDashboard
        engine.dashboard = PortfolioDashboard()
    except Exception as e:
        engine.dashboard = None
        engine.state.add_log("WARN", "Ext", f"PortfolioDashboard unavailable: {e}")

    # Risk Matrix
    try:
        from risk_matrix import RiskMatrix
        engine.risk_matrix_engine = RiskMatrix()
    except Exception as e:
        engine.risk_matrix_engine = None
        engine.state.add_log("WARN", "Ext", f"RiskMatrix unavailable: {e}")

    # Signal Engine
    try:
        from signal_engine import SignalEngine
        engine.signals = SignalEngine()
    except Exception as e:
        engine.signals = None
        engine.state.add_log("WARN", "Ext", f"SignalEngine unavailable: {e}")

    # Options Calculator
    try:
        from options_calculator import OptionsCalculator
        engine.calculator = OptionsCalculator()
    except Exception as e:
        engine.calculator = None
        engine.state.add_log("WARN", "Ext", f"OptionsCalculator unavailable: {e}")

    # ── Layer 2 Additions ─────────────────────────────────────

    # MTM Live Analytics
    try:
        from mtm_live import MTMLiveEngine
        engine.mtm_live = MTMLiveEngine()
    except Exception as e:
        engine.mtm_live = None
        engine.state.add_log("WARN", "Ext", f"MTMLiveEngine unavailable: {e}")

    # IV Rank Engine
    try:
        from iv_rank import IVRankEngine
        engine.iv_rank = IVRankEngine()
    except Exception as e:
        engine.iv_rank = None
        engine.state.add_log("WARN", "Ext", f"IVRankEngine unavailable: {e}")

    # Theta Projector
    try:
        from theta_projector import ThetaProjector
        engine.theta_proj = ThetaProjector()
    except Exception as e:
        engine.theta_proj = None
        engine.state.add_log("WARN", "Ext", f"ThetaProjector unavailable: {e}")

    # Delta Hedger
    try:
        from delta_hedger import DeltaHedger
        engine.delta_hedger = DeltaHedger()
    except Exception as e:
        engine.delta_hedger = None
        engine.state.add_log("WARN", "Ext", f"DeltaHedger unavailable: {e}")

    # GEX Engine
    try:
        from gex import GEXEngine
        engine.gex = GEXEngine()
    except Exception as e:
        engine.gex = None
        engine.state.add_log("WARN", "Ext", f"GEXEngine unavailable: {e}")

    # Calendar Strategies
    try:
        from calendar_strategies import CalendarStrategyEngine
        engine.calendar = CalendarStrategyEngine(
            session=engine.session,
            order_mgr=engine.omgr,
            db=engine.db,
            state=engine.state,
        )
    except Exception as e:
        engine.calendar = None
        engine.state.add_log("WARN", "Ext", f"CalendarStrategyEngine unavailable: {e}")

    # ── Layer 2.5 — Previously orphaned modules (now wired) ───

    # Rollover Engine
    try:
        from rollover import RolloverEngine
        engine.rollover = RolloverEngine(risk_free_rate=0.07)
    except Exception as e:
        engine.rollover = None
        engine.state.add_log("WARN", "Ext", f"RolloverEngine unavailable: {e}")

    # Regime Detector
    try:
        from regime_detector import RegimeDetector
        engine.regime = RegimeDetector()
    except Exception as e:
        engine.regime = None
        engine.state.add_log("WARN", "Ext", f"RegimeDetector unavailable: {e}")

    # Position Sizer
    try:
        from position_sizer import PositionSizer
        engine.sizer = PositionSizer()
    except Exception as e:
        engine.sizer = None
        engine.state.add_log("WARN", "Ext", f"PositionSizer unavailable: {e}")

    # Heatmap Engine
    try:
        from heatmap_engine import HeatmapEngine
        engine.heatmap = HeatmapEngine(risk_free_rate=0.07)
    except Exception as e:
        engine.heatmap = None
        engine.state.add_log("WARN", "Ext", f"HeatmapEngine unavailable: {e}")

    # IV Surface Engine
    try:
        from iv_surface import IVSurfaceEngine
        engine.iv_surface = IVSurfaceEngine(risk_free_rate=0.07)
    except Exception as e:
        engine.iv_surface = None
        engine.state.add_log("WARN", "Ext", f"IVSurfaceEngine unavailable: {e}")

    # Analytics (module-level functions, wrap in container)
    try:
        import analytics as _analytics_mod
        engine.analytics_mod = _analytics_mod
    except Exception as e:
        engine.analytics_mod = None
        engine.state.add_log("WARN", "Ext", f"analytics module unavailable: {e}")

    # ── Layer 3 New Modules ────────────────────────────────────

    # OI Analytics Engine
    try:
        from oi_analytics import OIAnalyticsEngine
        engine.oi_analytics = OIAnalyticsEngine()
    except Exception as e:
        engine.oi_analytics = None
        engine.state.add_log("WARN", "Ext", f"OIAnalyticsEngine unavailable: {e}")

    # Trade Journal
    try:
        from trade_journal import TradeJournal
        engine.trade_journal = TradeJournal()
    except Exception as e:
        engine.trade_journal = None
        engine.state.add_log("WARN", "Ext", f"TradeJournal unavailable: {e}")

    # Strategy Monitor
    try:
        from strategy_monitor import StrategyMonitor
        engine.strategy_monitor = StrategyMonitor()
    except Exception as e:
        engine.strategy_monitor = None
        engine.state.add_log("WARN", "Ext", f"StrategyMonitor unavailable: {e}")

    # Payoff Builder
    try:
        from payoff_builder import PayoffBuilder
        engine.payoff = PayoffBuilder(risk_free_rate=0.07)
    except Exception as e:
        engine.payoff = None
        engine.state.add_log("WARN", "Ext", f"PayoffBuilder unavailable: {e}")

    # Margin Ledger
    try:
        from margin_ledger import MarginLedger
        engine.margin_ledger = MarginLedger()
    except Exception as e:
        engine.margin_ledger = None
        engine.state.add_log("WARN", "Ext", f"MarginLedger unavailable: {e}")

    # Order Flow Tracker
    try:
        from order_flow_tracker import OrderFlowTracker
        engine.flow_tracker = OrderFlowTracker()
    except Exception as e:
        engine.flow_tracker = None
        engine.state.add_log("WARN", "Ext", f"OrderFlowTracker unavailable: {e}")

    # Notifier / Alert hook into RMS
    try:
        from notifier import Notifier
        engine.notifier = Notifier()
        # Hook notifier into RMS alerts if available
        if hasattr(engine, "rms") and engine.rms is not None:
            engine.rms.alerts = engine.notifier
    except Exception as e:
        engine.notifier = None
        engine.state.add_log("WARN", "Ext", f"Notifier unavailable: {e}")

    engine.state.add_log("INFO", "Ext", "Engine extensions loaded (Layer 1 + Layer 2 + Layer 3)")
