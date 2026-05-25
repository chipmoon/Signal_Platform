"""
AI Forecast Page
================
Real stock price prediction using the trading system's AI engine.
Fetches live data, trains model, and displays predictions.
"""

from __future__ import annotations

import calendar
import random
import textwrap
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from loguru import logger

from src.config import AIConfig, WyckoffConfig, SmcConfig, StochasticConfig
from src.strategies.seasonality import SeasonalityFilter
from src.plugins import registry
from src.strategies.ai_predictor import AIPredictor
from src.strategies.wyckoff_analyzer import WyckoffAnalyzer
from src.strategies.volume_profile import VolumeProfile
from src.strategies.smc_analyzer import SmcAnalyzer
from src.strategies.momentum import StochasticOscillator
from src.strategies.quant_money_flow import QuantMoneyFlowAnalyzer
from src.strategies.live_geo_osint import LiveGeoOsintEngine
from src.risk_manager import calculate_var
from src.llm_advisor import advisor
# ── Phase 1-4: Advanced Analysis Modules ──────────────────────────────────────
try:
    from src.analytics.mtf_confluence import compute_mtf_confluence
    _MTF_AVAILABLE = True
except ImportError:
    _MTF_AVAILABLE = False
try:
    from src.strategies.elliott_wave import ElliottWaveAnalyzer
    _EW_AVAILABLE = True
except ImportError:
    _EW_AVAILABLE = False
try:
    from src.analytics.fundamental_score import get_fundamental_dict
    _FUND_AVAILABLE = True
except ImportError:
    _FUND_AVAILABLE = False

# Mozyfin integration (optional - graceful fallback if key missing)
try:
    from src.mozyfin_client import MozyfinClient as _MozyfinClient
    _MOZYFIN_AVAILABLE = True
except ImportError:
    _MOZYFIN_AVAILABLE = False

# Gemini 2.5 Flash fallback (Google AI Studio - free 1,500 req/day)
try:
    from src.gemini_client import GeminiClient as _GeminiClient
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False


def _get_mozyfin_client():
    """Get Mozyfin client, returns None if API key not configured."""
    if not _MOZYFIN_AVAILABLE:
        return None
    try:
        return _MozyfinClient()
    except Exception:
        return None


def _get_gemini_client():
    """Get Gemini 2.5 Flash client, returns None if API key not configured."""
    if not _GEMINI_AVAILABLE:
        return None
    try:
        return _GeminiClient()
    except Exception:
        return None


def _get_best_ai_analyst(mozy_client=None):
    """
    Smart AI analyst selector with automatic fallback:
      1. Mozyfin AI  — if configured + credits remaining
      2. Gemini 2.5 Flash (Google) — free fallback, 1,500 req/day
    Returns: (client, provider 'mozyfin'|'gemini'|'none', usage_dict)
    """
    if mozy_client:
        try:
            usage = mozy_client.get_usage()
            if usage.get("credits_used", 0) < usage.get("credits_cap", 50):
                return mozy_client, "mozyfin", usage
        except Exception:
            pass

    gemini = _get_gemini_client()
    if gemini:
        return gemini, "gemini", {}

    return None, "none", {}


# ── Internal Helper Functions ──────────────────────────────────

def _render_html(html_str: str):
    """Ultimate cleanup for Streamlit HTML rendering.
    Uses st.html() (Streamlit 1.31+) to bypass Markdown processing entirely.
    """
    if not html_str: return
    # Remove all leading/trailing whitespace from each line and join into one line
    cleaned = "".join([line.strip() for line in html_str.splitlines() if line.strip()])
    try:
        # Dedicated HTML rendering method (bypasses Markdown)
        st.html(cleaned)
    except AttributeError:
        # Fallback for older Streamlit versions
        st.markdown(cleaned, unsafe_allow_html=True)


def _fetch_price_data(symbol: str, market: str, days: int = 500) -> pd.DataFrame:
    """Fetch historical price data for a symbol."""
    provider = registry.get(market)
    if not provider:
        return pd.DataFrame()

    end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    df = provider.get_price_data(symbol, start, end)
    if df is None or df.empty:
        return pd.DataFrame()
        
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        
    return df


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators as features."""
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    # Technical Indicators
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
    
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()

    df["Volatility"] = close.pct_change().rolling(20).std()
    
    # Seasonality / SMA
    df["SMA_50"] = close.rolling(50).mean()
    df["SMA_200"] = close.rolling(200).mean()

    return df.dropna(subset=["RSI_14", "ATR_14", "SMA_50"]).fillna(0)


def _run_prediction(df: pd.DataFrame, horizons: list[int]) -> dict:
    """Run probabilistic AI forecasting."""
    if df.empty: return {}
    
    config = AIConfig(horizons=tuple(horizons))
    predictor = AIPredictor(config)
    predictor.train(df)
    
    df_result = predictor.generate_signals(df)
    if df_result.empty: return {}

    last_row = df_result.iloc[-1]
    hist_price = last_row["Close"]  # Historical close (for model training)
    market_phase_info = predictor.detect_market_phase(df)
    
    results = {}
    label_map = {1: "🟢 BULLISH", -1: "🔴 BEARISH", 0: "🟡 NEUTRAL"}
    
    for h in horizons:
        price_median = last_row.get(f"ai_target_price_{h}d", hist_price)
        ret_median = (price_median - hist_price) / hist_price if hist_price > 0 else 0
        bias_code = last_row.get("ai_bias", 0) if h == 1 else (1 if ret_median > 0.01 else (-1 if ret_median < -0.01 else 0))
            
        results[h] = {
            "predicted_price": price_median,
            "predicted_lower": last_row.get(f"ai_price_lower_{h}d", hist_price),
            "predicted_upper": last_row.get(f"ai_price_upper_{h}d", hist_price),
            "predicted_return": ret_median * 100,
            "bias": label_map.get(bias_code, "🟡 NEUTRAL"),
            "train_r2": predictor.train_metrics.get("r2", 0.0),
            "test_r2": predictor.train_metrics.get("cv_r2_mean", 0.0),
            "importance": predictor.feature_importance,
            "current_price": hist_price,
            "confidence": last_row.get("ai_confidence", 0.0),
            "market_phase": market_phase_info
        }
    return results


def _apply_institutional_calibration(
    predictions: dict,
    wyckoff_state: dict,
    smc_state: dict,
    qmf_state: dict,
) -> dict:
    """
    Calibrate raw AI outputs with institutional context layers:
    - Wyckoff structure score/phase
    - SMC structure score/trend
    - Quant Money Flow score/signal
    - Stochastic overbought/oversold regime
    """
    if not predictions:
        return predictions

    w_score = float(wyckoff_state.get("score", 0.0))
    w_phase = str(wyckoff_state.get("phase", "")).upper()
    smc_score = float(smc_state.get("smc_score", 0.0))
    smc_trend = str(smc_state.get("trend", ""))
    qmf_score = float(qmf_state.get("score", 0.0))
    qmf_signal = int(qmf_state.get("signal", 0))
    stoch = smc_state.get("stoch", {}) if isinstance(smc_state, dict) else {}
    stoch_k = float(stoch.get("k", 50.0))
    stoch_d = float(stoch.get("d", 50.0))

    # Structural prior: signed directional prior in [-1, 1]
    prior = np.clip((0.45 * w_score) + (0.35 * smc_score) + (0.20 * qmf_score), -1.0, 1.0)

    # Regime penalty when short-term oscillator is exhausted
    overbought = stoch_k > 85 and stoch_d > 85
    oversold = stoch_k < 15 and stoch_d < 15

    for h in [1, 5, 21, 63]:
        if h not in predictions:
            continue
        p = predictions[h]
        current_price = float(p.get("current_price", 0.0))
        if current_price <= 0:
            continue

        raw_ret = float(p.get("predicted_return", 0.0)) / 100.0
        horizon_scale = {1: 0.25, 5: 0.45, 21: 0.70, 63: 0.90}.get(h, 0.5)
        calibrated_ret = raw_ret + (prior * 0.015 * horizon_scale)

        # Short-term exhaustion handling (mean-reversion penalty)
        if overbought and calibrated_ret > 0:
            calibrated_ret *= 0.70 if h <= 5 else 0.85
        if oversold and calibrated_ret < 0:
            calibrated_ret *= 0.70 if h <= 5 else 0.85

        # Trend-consistency nudge
        if "BULL" in smc_trend.upper() and calibrated_ret > 0:
            calibrated_ret *= 1.05
        if "BEAR" in smc_trend.upper() and calibrated_ret < 0:
            calibrated_ret *= 1.05

        calibrated_price = current_price * (1.0 + calibrated_ret)

        # Uncertainty band: widen in ranging/uncertain regimes
        band_pct = abs(float(p.get("predicted_upper", current_price)) - float(p.get("predicted_lower", current_price))) / max(current_price, 1e-9)
        if "RANGE" in w_phase or abs(smc_score) < 0.12:
            band_pct *= 1.15
        if overbought or oversold:
            band_pct *= 1.10
        band_pct = float(np.clip(band_pct, 0.01, 0.22))

        p["predicted_price"] = float(calibrated_price)
        p["predicted_return"] = float(calibrated_ret * 100.0)
        p["predicted_lower"] = float(calibrated_price * (1.0 - band_pct / 2.0))
        p["predicted_upper"] = float(calibrated_price * (1.0 + band_pct / 2.0))

        base_conf = float(p.get("confidence", 0.0))
        conf_boost = (0.08 if np.sign(raw_ret) == np.sign(prior) else -0.08) * min(abs(prior), 1.0)
        if overbought or oversold:
            conf_boost -= 0.03
        if qmf_signal == 0:
            conf_boost -= 0.02
        p["confidence"] = float(np.clip(base_conf + conf_boost, 0.0, 1.0))

        p["institutional_prior"] = float(prior)
        p["institutional_note"] = (
            f"WY={w_score:+.2f} | SMC={smc_score:+.2f} | QMF={qmf_score:+.2f} | "
            f"StochK={stoch_k:.1f}"
        )
    return predictions


def _render_forecast_basis(predictions: dict):
    """Show compact rationale for calibrated forecasts."""
    st.markdown("### Forecast Basis")
    for h, label in [(1, "1D"), (5, "1W"), (21, "1M"), (63, "3M")]:
        if h not in predictions:
            continue
        p = predictions[h]
        note = p.get("institutional_note", "N/A")
        prior = float(p.get("institutional_prior", 0.0))
        st.caption(f"{label}: prior={prior:+.2f} | {note}")


def _render_geo_gate_banner(geo_gate: dict) -> None:
    """Display active geo-execution regime synced from Geo-OSINT Lab."""
    if not geo_gate or geo_gate.get("action", "NORMAL") == "NORMAL":
        return

    action = str(geo_gate.get("action", "NORMAL"))
    risk = float(geo_gate.get("geo_risk_score", 0.0))
    oil_mv = float(geo_gate.get("expected_oil_move_1d_pct", 0.0))
    reason = str(geo_gate.get("reason", "Geo gate active"))
    multiplier = float(geo_gate.get("position_multiplier", 1.0))

    color = "#f6ad55" if action == "REDUCE_SIZE" else "#f56565"
    st.markdown(
        f"""
        <div style="background: rgba(26,26,46,0.92); border:1px solid {color}; border-radius:10px; padding:10px 14px; margin-bottom:14px;">
            <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; letter-spacing:1px;">Geo Execution Gate</div>
            <div style="font-size:0.95rem; color:#fff;"><b>{action}</b> | Size Multiplier: <b>{multiplier:.2f}</b></div>
            <div style="font-size:0.8rem; color:#cbd5e1;">Geo Risk: {risk:.2f} | Expected Oil Move(1D): {oil_mv:+.2f}% | {reason}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _apply_auto_veto(predictions: dict, smc_state: dict, qmf_state: dict) -> dict:
    """Hard risk gate when momentum and flow are strongly contradictory."""
    if not predictions:
        return {"active": False, "reason": "N/A", "severity": "LOW"}

    p1 = predictions.get(1, {})
    stoch = smc_state.get("stoch", {}) if isinstance(smc_state, dict) else {}
    stoch_k = float(stoch.get("k", 50.0))
    stoch_d = float(stoch.get("d", 50.0))
    qmf_signal = int(qmf_state.get("signal", 0))
    qmf_score = float(qmf_state.get("score", 0.0))
    smc_score = float(smc_state.get("smc_score", 0.0))
    ret_1d = float(p1.get("predicted_return", 0.0))

    long_veto = (stoch_k > 88 and stoch_d > 85 and qmf_signal < 0 and ret_1d > 0)
    short_veto = (stoch_k < 12 and stoch_d < 15 and qmf_signal > 0 and ret_1d < 0)
    conflict_veto = abs(smc_score) < 0.08 and abs(qmf_score) < 0.10 and abs(ret_1d) > 1.2

    active = bool(long_veto or short_veto or conflict_veto)
    reason = "No major contradiction"
    if long_veto:
        reason = "Overbought + outflow: veto chasing long entries."
    elif short_veto:
        reason = "Oversold + inflow: veto forcing short entries."
    elif conflict_veto:
        reason = "Low structural conviction vs high return forecast."

    if active:
        for h in [1, 5]:
            if h in predictions:
                predictions[h]["confidence"] = float(np.clip(predictions[h].get("confidence", 0.0) * 0.72, 0.0, 1.0))
                predictions[h]["bias"] = "🟡 NEUTRAL"

    return {
        "active": active,
        "reason": reason,
        "severity": "HIGH" if active else "LOW",
    }


def _normalize_geo_gate(gate: dict | None) -> dict:
    """Normalize gate shape and clamp values."""
    if not isinstance(gate, dict):
        return {}

    action = str(gate.get("action", "NORMAL")).upper().strip()
    if action not in {"NORMAL", "REDUCE_SIZE", "PAUSE_NEW_LONGS"}:
        action = "NORMAL"
    multiplier = float(np.clip(gate.get("position_multiplier", 1.0), 0.0, 1.0))
    return {
        "action": action,
        "position_multiplier": multiplier,
        "reason": str(gate.get("reason", "Geo gate active")),
        "geo_risk_score": float(gate.get("geo_risk_score", 0.0)),
        "expected_oil_move_1d_pct": float(gate.get("expected_oil_move_1d_pct", 0.0)),
    }


def _sync_live_geo_gate(symbol: str, market: str, force_refresh: bool = False) -> tuple[dict, dict]:
    """Sync live OSINT gate to session with short TTL cache."""
    cache_key = f"live_geo_state::{market}::{symbol}"
    ts_key = f"{cache_key}::ts"
    now = datetime.now()
    last_ts = st.session_state.get(ts_key)
    state = st.session_state.get(cache_key)
    is_fresh = bool(last_ts and isinstance(state, dict) and (now - last_ts).total_seconds() < 180)

    if force_refresh or not is_fresh:
        try:
            state = LiveGeoOsintEngine().build_live_state(symbol=symbol, market=market)
            st.session_state[cache_key] = state
            st.session_state[ts_key] = now
        except Exception as exc:
            logger.warning(f"Live geo sync failed for {symbol} ({market}): {exc}")
            state = state if isinstance(state, dict) else {}

    gate = _normalize_geo_gate(state.get("gate", {}) if isinstance(state, dict) else {})
    osint = state.get("osint", {}) if isinstance(state, dict) else {}
    causal = state.get("causal", {}) if isinstance(state, dict) else {}
    status = state.get("status", {}) if isinstance(state, dict) else {}

    if gate:
        st.session_state["geo_gate_state"] = gate
    if isinstance(osint, dict):
        st.session_state["geo_osint_state"] = osint
    if isinstance(causal, dict):
        st.session_state["geo_causal_state"] = causal

    return gate, status if isinstance(status, dict) else {}


def _apply_geo_execution_gate(predictions: dict, veto_state: dict, geo_gate: dict) -> tuple[dict, dict]:
    """Apply Geo-OSINT execution regime to forecast confidence and veto behavior."""
    if not geo_gate or geo_gate.get("action", "NORMAL") == "NORMAL":
        return predictions, veto_state

    action = geo_gate.get("action", "NORMAL")
    pos_mul = float(np.clip(geo_gate.get("position_multiplier", 1.0), 0.0, 1.0))
    conf_mul = 0.65 + 0.35 * pos_mul

    for h in [1, 5, 21, 63]:
        if h not in predictions:
            continue
        p = predictions[h]
        base_conf = float(p.get("confidence", 0.0))
        p["confidence"] = float(np.clip(base_conf * conf_mul, 0.0, 1.0))

        if action == "PAUSE_NEW_LONGS" and float(p.get("predicted_return", 0.0)) > 0:
            # Risk-first: suppress bullish edge under extreme geo stress.
            p["predicted_return"] = 0.0
            p["predicted_price"] = float(p.get("current_price", p.get("predicted_price", 0.0)))
            p["bias"] = "🟡 NEUTRAL"
        elif action == "REDUCE_SIZE" and float(p.get("predicted_return", 0.0)) > 0:
            adj_ret = float(p.get("predicted_return", 0.0)) * 0.8
            curr = float(p.get("current_price", p.get("predicted_price", 0.0)))
            p["predicted_return"] = adj_ret
            if curr > 0:
                p["predicted_price"] = curr * (1.0 + adj_ret / 100.0)

        p["geo_gate_action"] = action

    if action == "PAUSE_NEW_LONGS":
        veto_state["active"] = True
        veto_state["severity"] = "HIGH"
        veto_state["reason"] = f"{veto_state.get('reason', 'N/A')} | GEO_GATE_PAUSE_NEW_LONGS"
    elif action == "REDUCE_SIZE":
        veto_state["severity"] = "MEDIUM" if veto_state.get("active") else "LOW"
        veto_state["reason"] = f"{veto_state.get('reason', 'N/A')} | GEO_GATE_REDUCE_SIZE"

    return predictions, veto_state


def _estimate_var_gate(df_train: pd.DataFrame, max_var_pct: float = 0.03) -> dict:
    """Estimate 1-day VaR from recent close-to-close path and produce gate decision."""
    if df_train.empty or "Close" not in df_train.columns or len(df_train) < 40:
        return {"blocked": False, "var_pct": 0.0, "max_var_pct": max_var_pct}

    rets = df_train["Close"].astype(float).pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if rets.empty:
        return {"blocked": False, "var_pct": 0.0, "max_var_pct": max_var_pct}

    base = 100_000.0
    equity_curve = (1.0 + rets).cumprod() * base
    stats = calculate_var(equity_curve.tolist(), confidence=0.95, lookback=252)
    var_pct = float(stats.get("var_pct", 0.0))
    return {"blocked": bool(var_pct > max_var_pct), "var_pct": var_pct, "max_var_pct": max_var_pct}


def _estimate_kelly_cap(predictions: dict, df_train: pd.DataFrame) -> float:
    """Estimate a conservative half-Kelly cap from confidence and ATR proxy."""
    if 1 not in predictions or df_train.empty:
        return 0.025

    p1 = predictions[1]
    conf = float(np.clip(p1.get("confidence", 0.5), 0.05, 0.95))
    exp_ret_pct = abs(float(p1.get("predicted_return", 0.0)))
    tr = pd.concat(
        [
            (df_train["High"] - df_train["Low"]).abs(),
            (df_train["High"] - df_train["Close"].shift()).abs(),
            (df_train["Low"] - df_train["Close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else 0.0
    price = float(df_train["Close"].iloc[-1])
    atr_pct = (atr / price) * 100.0 if price > 0 else 1.0
    rr = float(np.clip(exp_ret_pct / max(atr_pct, 0.25), 0.3, 3.0))
    q = 1.0 - conf
    kelly = conf - (q / rr)
    half_kelly = float(np.clip(kelly * 0.5, 0.003, 0.025))
    return half_kelly


def _compute_dynamic_position_sizing(
    predictions: dict,
    wyckoff_state: dict,
    smc_state: dict,
    qmf_state: dict,
    veto_state: dict,
    ai_confidence_weight: float = 1.0,
    kelly_cap: float | None = None,
    geo_multiplier: float = 1.0,
) -> dict:
    """Risk budget and core/satellite split based on confidence and institutional prior."""
    p1 = predictions.get(1, {})
    p63 = predictions.get(63, {})
    conf = float(p1.get("confidence", 0.5))
    conf = float(np.clip(0.5 + (conf - 0.5) * ai_confidence_weight, 0.0, 1.0))
    prior = float(p63.get("institutional_prior", p1.get("institutional_prior", 0.0)))
    qmf_score = float(qmf_state.get("score", 0.0))
    smc_score = float(smc_state.get("smc_score", 0.0))
    w_score = float(wyckoff_state.get("score", 0.0))

    inst_conviction = float(np.clip(0.4 * prior + 0.25 * qmf_score + 0.2 * smc_score + 0.15 * w_score, -1.0, 1.0))
    base_risk = 0.006 + 0.018 * conf
    conviction_mul = 1.0 + (0.55 * inst_conviction)
    if veto_state.get("active"):
        conviction_mul *= 0.55
    risk_pct = float(np.clip(base_risk * conviction_mul, 0.003, 0.025))
    risk_pct *= float(np.clip(geo_multiplier, 0.0, 1.0))
    if kelly_cap is not None:
        risk_pct = float(min(risk_pct, max(float(kelly_cap), 0.003)))
    risk_pct = float(np.clip(risk_pct, 0.003, 0.025))

    phase = str(wyckoff_state.get("phase", "")).upper()
    is_accum = any(x in phase for x in ["PHASE B", "PHASE C", "PHASE D", "ACCUM"])
    if is_accum and inst_conviction > 0:
        core_alloc = int(np.clip(55 + 25 * inst_conviction, 40, 80))
    else:
        core_alloc = int(np.clip(35 + 20 * inst_conviction, 20, 65))
    tactical_alloc = 100 - core_alloc

    return {
        "risk_pct": risk_pct,
        "core_alloc": core_alloc,
        "tactical_alloc": tactical_alloc,
        "inst_conviction": inst_conviction,
        "ai_conf_weight": float(ai_confidence_weight),
        "kelly_cap": float(kelly_cap) if kelly_cap is not None else None,
        "geo_multiplier": float(np.clip(geo_multiplier, 0.0, 1.0)),
    }


def _run_ab_backtest(df_full: pd.DataFrame, market: str, lookback_months: int = 12) -> pd.DataFrame:
    """A/B backtest: raw forecast vs calibrated forecast."""
    if df_full.empty or len(df_full) < 260:
        return pd.DataFrame()

    cutoff = df_full["Date"].max() - pd.DateOffset(months=lookback_months)
    idx_candidates = [i for i in range(120, len(df_full) - 64, 21) if df_full.iloc[i]["Date"] >= cutoff]
    rows = []

    for i in idx_candidates:
        train_raw = df_full.iloc[: i + 1].copy()
        feat = _compute_features(train_raw.copy())
        raw_pred = _run_prediction(feat, [1, 5, 21, 63])
        if not raw_pred:
            continue

        wa = WyckoffAnalyzer(WyckoffConfig())
        wy_state = wa.analyze_current_state(train_raw)
        smc_state = SmcAnalyzer(SmcConfig()).get_current_state(train_raw)
        qmf_df = QuantMoneyFlowAnalyzer().generate_signals(train_raw)
        q_last = qmf_df.iloc[-1] if not qmf_df.empty else {}
        q_state = {
            "score": float(q_last.get("qmf_score", 0.0)),
            "signal": int(q_last.get("qmf_signal", 0)),
        }
        cal_pred = _apply_institutional_calibration(deepcopy(raw_pred), wy_state, smc_state, q_state)

        for h in [1, 5, 21, 63]:
            if h not in raw_pred or i + h >= len(df_full):
                continue
            actual = float(df_full.iloc[i + h]["Close"])
            pr = float(raw_pred[h]["predicted_price"])
            pc = float(cal_pred[h]["predicted_price"])
            curr = float(raw_pred[h].get("current_price", train_raw["Close"].iloc[-1]))
            dir_actual = np.sign(actual - curr)
            rows.append({
                "h": h,
                "mae_raw": abs(pr - actual),
                "mae_cal": abs(pc - actual),
                "mape_raw": abs(pr - actual) / max(actual, 1e-9),
                "mape_cal": abs(pc - actual) / max(actual, 1e-9),
                "hit_raw": 1 if np.sign(pr - curr) == dir_actual else 0,
                "hit_cal": 1 if np.sign(pc - curr) == dir_actual else 0,
            })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = df.groupby("h", as_index=False).agg(
        mae_raw=("mae_raw", "mean"),
        mae_cal=("mae_cal", "mean"),
        mape_raw=("mape_raw", "mean"),
        mape_cal=("mape_cal", "mean"),
        hit_raw=("hit_raw", "mean"),
        hit_cal=("hit_cal", "mean"),
    )
    out["mae_improve_pct"] = (out["mae_raw"] - out["mae_cal"]) / out["mae_raw"].replace(0, np.nan) * 100
    out["mape_improve_pct"] = (out["mape_raw"] - out["mape_cal"]) / out["mape_raw"].replace(0, np.nan) * 100
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
    return out


def _render_upgrade_governance(
    backtest_df: pd.DataFrame,
    sizing: dict,
    veto_state: dict,
    geo_gate: dict | None = None,
    var_gate: dict | None = None,
):
    """Render A/B quality + risk governance panels."""
    st.markdown("### Forecast Governance")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Risk Budget / Trade", f"{sizing.get('risk_pct', 0.0) * 100:.2f}%")
        st.caption(f"Inst conviction: {sizing.get('inst_conviction', 0.0):+.2f}")
    with c2:
        st.metric("Strategic Core", f"{int(sizing.get('core_alloc', 30))}%")
        st.caption(f"Tactical: {int(sizing.get('tactical_alloc', 70))}%")
    with c3:
        gate = "ACTIVE" if veto_state.get("active") else "CLEAR"
        st.metric("Auto Veto Gate", gate)
        st.caption(veto_state.get("reason", "N/A"))

    if geo_gate and geo_gate.get("action", "NORMAL") != "NORMAL":
        st.caption(
            "Geo Gate: "
            f"{geo_gate.get('action')} | mult={float(geo_gate.get('position_multiplier', 1.0)):.2f} | "
            f"risk={float(geo_gate.get('geo_risk_score', 0.0)):.2f}"
        )
    if var_gate:
        st.caption(
            "VaR Gate: "
            f"{'BLOCKED' if var_gate.get('blocked') else 'CLEAR'} | "
            f"1D VaR={float(var_gate.get('var_pct', 0.0)):.2%} (limit {float(var_gate.get('max_var_pct', 0.03)):.2%})"
        )
    if sizing.get("kelly_cap") is not None:
        st.caption(f"Kelly Cap (half-kelly proxy): {float(sizing.get('kelly_cap', 0.0)):.2%}")

    st.markdown("### A/B Backtest (Raw vs Calibrated)")
    if backtest_df.empty:
        st.info("Not enough data to run backtest yet.")
        return
    label_map = {1: "1D", 5: "1W", 21: "1M", 63: "3M"}
    show = backtest_df.copy()
    show["horizon"] = show["h"].map(label_map)
    show["hit_raw"] = (show["hit_raw"] * 100).round(1)
    show["hit_cal"] = (show["hit_cal"] * 100).round(1)
    show = show[["horizon", "mae_raw", "mae_cal", "mape_raw", "mape_cal", "hit_raw", "hit_cal", "mae_improve_pct", "mape_improve_pct"]]
    st.dataframe(show, width="stretch")


def _fetch_realtime_quote(symbol: str, market: str):
    """Fetch real-time price quote with 5-minute session caching."""
    cache_key = f"rt_quote_{symbol}"
    cache_ts_key = f"rt_quote_ts_{symbol}"

    # Check session cache (refresh every 5 minutes)
    cached_quote = st.session_state.get(cache_key)
    cached_ts = st.session_state.get(cache_ts_key)
    if cached_quote and cached_ts:
        age = (datetime.now() - cached_ts).total_seconds()
        if age < 60:  # 1 minute (Faster than 5m for better user experience)
            return cached_quote

    provider = registry.get(market)
    if not provider:
        return None

    try:
        quote = provider.get_realtime_quote(symbol)
        if quote:
            st.session_state[cache_key] = quote
            st.session_state[cache_ts_key] = datetime.now()
        return quote
    except Exception as e:
        logger.warning(f"Realtime quote failed for {symbol}: {e}")
        return cached_quote  # Return stale cache if fresh fetch fails


# ── UI Components ──────────────────────────────────────────────

def _render_volume_profile(df: pd.DataFrame):
    """Render Price vs Volume Horizontal Profile (from institutional_flow)."""
    from src.strategies.volume_profile import VolumeProfile
    vp = VolumeProfile()
    analysis = vp.analyze(df)
    
    if analysis["status"] != "SUCCESS":
        st.warning("Insufficient data for Volume Profile.")
        return

    hist = analysis["histogram"]
    prices = [h["price"] for h in hist]
    volumes = [h["volume"] for h in hist]
    colors = ["#FF5252" if h["is_poc"] else "#667eea" if h["in_va"] else "#334155" for h in hist]

    fig = go.Figure(go.Bar(
        y=prices, x=volumes,
        orientation='h',
        marker_color=colors,
        name="Volume at Price"
    ))
    
    fig.add_hline(y=analysis["poc"], line_dash="dash", line_color="#FF5252", annotation_text="POC")
    fig.add_hline(y=analysis["vah"], line_dash="dot", line_color="#00E676", annotation_text="VAH")
    fig.add_hline(y=analysis["val"], line_dash="dot", line_color="#00E676", annotation_text="VAL")

    fig.update_layout(
        template="plotly_dark",
        title="Volume Profile (Institutional Accumulation zones)",
        xaxis_title="Volume",
        yaxis_title="Price",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, width="stretch")

def _render_asset_selector():
    """Render the stock search and hot picks section."""
    st.markdown("### Stock Selector")
    col1, col2 = st.columns([3, 2])
    with col1:
        search_q = st.text_input("Change target asset", placeholder="e.g. GOLD, NVDA, BTC...", label_visibility="collapsed")
        if search_q:
            hits = registry.search_all(search_q, limit=10)
            if hits:
                for hit in hits:
                    if st.button(f"🚀 Predict {hit.symbol} ({hit.name})", key=f"ai_hit_{hit.symbol}"):
                        st.session_state["global_symbol"] = hit.symbol
                        st.session_state["global_market"] = hit.market
                        st.rerun()
    with col2:
        st.markdown("#### ⚡ Hot Picks")
        hp_cols = st.columns(4)
        for idx, s in enumerate(["GOLD", "NVDA", "VCB", "BTC-USD"]):
            if hp_cols[idx % 4].button(s, key=f"ai_hp_{s}"):
                hits = registry.search_all(s, limit=1)
                if hits:
                    st.session_state["global_symbol"] = hits[0].symbol
                    st.session_state["global_market"] = hits[0].market
                    st.rerun()


def _render_wyckoff_panel(df: pd.DataFrame):
    """Render the Wyckoff Analysis panel."""
    with st.spinner("Analyzing Wyckoff..."):
        try:
            wyckoff = WyckoffAnalyzer(WyckoffConfig())
            w = wyckoff.analyze_current_state(df)
            
            score_color = "#48bb78" if w['score'] > 0.2 else "#f56565" if w['score'] < -0.2 else "#f6ad55"
            score_bar = int((w['score'] + 1.0) / 2.0 * 100)
            
            _render_html(f"""
            <div class="wyckoff-card">
              <div style="color:#90cdf4; font-size:0.85rem; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:8px; margin-bottom:12px;">WYCKOFF INTELLIGENCE</div>
              <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                <span>Phase: <b>{w['phase']}</b></span>
                <span>Event: <b>{w['vsa_label']}</b></span>
              </div>
              <div style="background:#2d3748; height:8px; border-radius:4px; margin-bottom:5px;">
                <div style="background:{score_color}; height:100%; border-radius:4px; width:{score_bar}%;"></div>
              </div>
              <div style="display:flex; justify-content:space-between; color:#94a3b8; font-size:0.75rem;">
                <span>Score: {w['score']:+.2f}</span>
                <span>Trend: {'Bullish' if w['score']>0 else 'Bearish'}</span>
              </div>
            </div>
            """)
            return w
        except Exception as e:
            logger.debug(f"Wyckoff UI error: {e}")
            return {}


def _render_execution_panel(df: pd.DataFrame, smc_state: dict, wyckoff_state: dict, predictions: dict, scalp_intel: dict = None):
    """Render the Unified Professional Execution Panel (Strategic Hub Roadmap)."""
    p1 = predictions.get(1, {})
    p63 = predictions.get(63, {})
    daily_bias = p1.get("bias", "🟡 NEUTRAL")
    lt_bias = p63.get("bias", "🟡 NEUTRAL")
    
    # MTF CONFLUENCE LOGIC:
    # 1. Start with H4 Structure (HTF)
    h4_health = scalp_intel.get("health", "") if scalp_intel else ""
    h4_is_bullish = any(x in h4_health for x in ["STRONG BUY", "BULL FLAG", "BULLISH"])
    h4_is_bearish = any(x in h4_health for x in ["BEARISH SYNC", "BEAR PENNANT", "BEARISH"])
    
    # 2. Refine with H1 Momentum/Bias (LTF)
    # is_bullish = ("BULLISH" in daily_bias) or ("BULLISH" in lt_bias and "BEARISH" not in daily_bias)
    is_bullish = h4_is_bullish and ("BULLISH" in daily_bias)
    is_bearish = h4_is_bearish and ("BEARISH" in daily_bias)
    
    # 3. Handle 'Weak' or 'No Sync' states strictly for Optimization
    is_weak = ("REBOUND" in h4_health) or (h4_is_bullish != ("BULLISH" in daily_bias))
    
    setup_label = "CONFLUENCE LONG" if (is_bullish and not is_weak) else "CONFLUENCE SHORT" if (is_bearish and not is_weak) else "WAIT / NO SYNC"
    setup_color = "#00E676" if is_bullish else "#FF5252" if is_bearish else "#f6ad55"
    if is_weak: setup_color = "#f6ad55" # Yellow for caution

    with st.spinner("Engineering Trade Execution..."):
        try:
            # Use real-time price if available in session state
            rt_quote = st.session_state.get(f"rt_quote_{st.session_state.get('global_symbol', '')}")
            curr_price = rt_quote.price if rt_quote else df["Close"].iloc[-1]
            atr_d = df["ATR_14"].iloc[-1] if "ATR_14" in df.columns else (curr_price * 0.02)

            # 1. Macro Entry Calculation
            vp = VolumeProfile()
            vpa = vp.analyze(df)
            poc, vah, val = vpa.get("poc", curr_price), vpa.get("vah", curr_price*1.05), vpa.get("val", curr_price*0.95)

            macro_entry = poc
            if is_bullish:
                bull_obs = [ob for ob in smc_state.get('bull_obs', []) if ob['top'] < curr_price]
                macro_entry = max(bull_obs, key=lambda x: x['top'])['top'] if bull_obs else val
            elif is_bearish:
                bear_obs = [ob for ob in smc_state.get('bear_obs', []) if ob['bottom'] > curr_price]
                macro_entry = min(bear_obs, key=lambda x: x['bottom'])['bottom'] if bear_obs else vah

            # 2. Adaptive Logic: Switch to Intraday if Macro is too far (>5%)
            dist_macro = abs(curr_price - macro_entry) / curr_price
            use_intraday = (scalp_intel is not None) and (dist_macro > 0.05)
            
            # STRICT MTF: Explicitly show the transition
            duality_note = f"MTF SYNC: H4 ({'UP' if h4_is_bullish else 'DOWN' if h4_is_bearish else 'RANGE'}) ➔ H1 ({'UP' if 'BULLISH' in daily_bias else 'DOWN' if 'BEARISH' in daily_bias else 'NEUT'})"
            if is_weak: duality_note = "⚠ SYNC FAILURE: HTF/LTF Divergence"
            
            entry_price = macro_entry
            tp_price = 0.0
            sl_price = 0.0
            plan_type = "MACRO PLAN"
            ready_color = "#94a3b8"
            ready_status = "WAITING"

            if use_intraday:
                entry_price = scalp_intel["step1"]
                tp_price = scalp_intel["step2"]
                plan_type = "INTRADAY PLAN"
                
                df_1h = scalp_intel.get("df_1h")
                # FAULT TOLERANCE: Protect against NaN in rolling std
                if df_1h is not None and len(df_1h) >= 14:
                    atr_h = df_1h["Close"].rolling(14).std().iloc[-1]
                    if np.isnan(atr_h): atr_h = atr_d * 0.15 # Fallback to 15% of daily vol
                else:
                    atr_h = atr_d * 0.15
                
                sl_price = entry_price - (atr_h * 2.0) if is_bullish else entry_price + (atr_h * 2.0)
            else:
                if is_bullish:
                    struct_low = macro_entry * 0.98
                    bull_obs = [ob for ob in smc_state.get('bull_obs', []) if ob['top'] < curr_price]
                    if bull_obs: struct_low = min(ob['bottom'] for ob in bull_obs)
                    sl_price = struct_low - (atr_d * 0.5)
                    tp_price = vah if vah > curr_price else (curr_price * 1.10)
                else:
                    struct_high = macro_entry * 1.02
                    bear_obs = [ob for ob in smc_state.get('bear_obs', []) if ob['bottom'] > curr_price]
                    if bear_obs: struct_high = max(ob['top'] for ob in bear_obs)
                    sl_price = struct_high + (atr_d * 0.5)
                    tp_price = val if val < curr_price else (curr_price * 0.90)

            # Metrics
            risk = abs(entry_price - sl_price)
            reward = abs(tp_price - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            dist_to_entry = abs(curr_price - entry_price) / entry_price
            
            # 3-Level Status
            ready_status = "READY" if dist_to_entry < 0.005 else "ALMOST" if dist_to_entry < 0.015 else "WAITING"
            
            # 4th Level: 🔥 TRIGGER / PYRAMID (Option B)
            conf = p1.get("confidence", 0)
            smc_score = abs(smc_state.get("smc_score", 0))
            is_trigger = (ready_status == "READY") and (conf > 0.7 or smc_score > 0.4)
            is_pyramid = scalp_intel.get("pattern") in ["BULL FLAG", "BEAR PENNANT"] if scalp_intel else False
            
            if is_trigger:
                ready_status = "🔥 TRIGGER"
                ready_color = "#FFD700" 
            elif is_pyramid:
                ready_status = "🔥 PYRAMID"
                ready_color = "#b794f4" # Purple for Flag pattern
            else:
                ready_color = "#00E676" if ready_status == "READY" else "#f6ad55" if ready_status == "ALMOST" else "#94a3b8"

            strat_desc = "Focusing on H1 Range" if use_intraday else f"Structure: {wyckoff_state.get('phase', 'Accumulation')}"

            # Pre-compute real-time change display
            live_indicator = "LIVE" if rt_quote and rt_quote.source != "historical_fallback" else "EOD"
            if rt_quote:
                chg_color = "#00E676" if rt_quote.change >= 0 else "#FF5252"
                change_html = f'<div style="font-size:0.7rem; color:{chg_color}; font-weight:700;">{rt_quote.change:+.2f}%</div>'
            else:
                change_html = ""

            # UI Construction
            ui_html = (
                f'<div class="rhs-card" style="border-top: 4px solid {"#ffd700" if is_trigger else ("#00E676" if is_bullish else "#FF5252")}; padding-top: 20px;">'
                f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;">'
                f'<span style="font-weight:800; font-size:1rem; letter-spacing:1px; color:#fff;">{plan_type}</span>'
                f'<span style="color:{ready_color}; font-size:0.75rem; font-weight:800; border:1px solid {ready_color}44; padding:3px 10px; border-radius:4px; background:{ready_color}11;">{ready_status}</span>'
                f'</div>'
                f'<div style="font-size:0.6rem; color:#94a3b8; margin-bottom:15px; text-transform:uppercase; letter-spacing:0.5px;">{duality_note}</div>'
                
                # Signal Info
                f'<div style="background:rgba(255,255,255,0.03); border-radius:12px; border:1px solid rgba(255,255,255,0.05); margin-bottom:15px; overflow:hidden;">'
                f'<div style="text-align:center; padding:10px 15px; background:rgba(255,255,255,0.02); border-bottom:1px solid rgba(255,255,255,0.05); font-weight:700; color:{setup_color}; font-size:0.85rem;">{setup_label} SETUP ALERT</div>'
                
                # MAIN PRICE INFO — Real-time quote
                f'<div style="padding:15px; display:flex; flex-direction:column; gap:12px; text-align:center;">'
                f'  <div>'
                f'    <div style="font-size:0.65rem; color:#94a3b8; text-transform:uppercase; letter-spacing:1px;">Market Price'
                f'      <span style="color:#667eea; font-size:0.5rem; margin-left:4px;">{live_indicator}</span>'
                f'    </div>'
                f'    <div style="font-size:1.6rem; font-weight:900; color:#fff; font-family:monospace;">${curr_price:,.2f}</div>'
                f'    {change_html}'
                f'  </div>'
                f'  <div style="border-top:1px dashed rgba(255,255,255,0.1); padding-top:12px;">'
                f'    <div style="font-size:0.65rem; color:#f6ad55; text-transform:uppercase; letter-spacing:1px; font-weight:700;">Risk : Reward Ratio</div>'
                f'    <div style="font-size:1.4rem; font-weight:900; color:#f6ad55;">1 : {rr_ratio:.1f}</div>'
                f'  </div>'
                f'</div>'
                f'</div>'
            )
            
            # ➔ INSTITUTIONAL ROADMAP (Strategic Pathway)
            s0 = scalp_intel.get("step0") if scalp_intel else None
            # Core Alignment: Phase 1 is Step 1 (Entry/Structure), Phase 2 is Step 2 (Target/Trend)
            s1 = scalp_intel["step1"] if scalp_intel else entry_price
            s2 = scalp_intel["step2"] if scalp_intel else tp_price
            
            health = scalp_intel.get("health", "ACTIVE") if scalp_intel else "STABLE"
            health_color = "#00E676" if any(x in health for x in ["READY", "STRONG", "ACTIVE"]) else "#f6ad55"
            road_label = "➔ STRATEGIC HUB ROADMAP"
            
            ui_html += (
                f'<div style="margin-bottom:12px; display:flex; justify-content:space-between; align-items:center;">'
                f'  <span style="color:#68d391; font-size:0.75rem; font-weight:700; letter-spacing:0.5px;">{road_label}</span>'
                f'  <span style="color:{health_color}; font-size:0.6rem; font-weight:900; background:{health_color}11; padding:2px 8px; border-radius:4px; border:1px solid {health_color}22;">{health}</span>'
                f'</div>'
            )

            # Conditional Step 0 - Early Entry
            if s0:
                ui_html += (
                    f'<div style="background:rgba(255,215,0,0.05); border:1px solid rgba(255,215,0,0.2); border-radius:8px; padding:10px; margin-bottom:10px; display:flex; justify-content:space-between; align-items:center;">'
                    f'  <div style="color:#ffd700; font-size:0.65rem; font-weight:800; letter-spacing:1px;">⚡ PHASE 0: EARLY (SL to Entry @70%)</div>'
                    f'  <div style="font-size:1.1rem; font-weight:900; color:#fff; font-family:monospace;">${s0:,.2f}</div>'
                    f'</div>'
                )

            ui_html += (
                f'<div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-bottom:15px;">'
                f'  <div style="background:linear-gradient(to bottom, rgba(102,126,234,0.1), rgba(102,126,234,0.02)); border:1px solid rgba(102,126,234,0.2); border-radius:8px; padding:12px 5px; text-align:center;">'
                f'    <div style="font-size:0.6rem; color:#667eea; margin-bottom:4px; font-weight:800; text-transform:uppercase;">Phase 1: Struct</div>'
                f'    <div style="font-size:1rem; font-weight:900; color:#fff; font-family:monospace;">${s1:,.2f}</div>'
                f'  </div>'
                f'  <div style="background:linear-gradient(to bottom, rgba(183,148,244,0.1), rgba(183,148,244,0.02)); border:1px solid rgba(183,148,244,0.2); border-radius:8px; padding:12px 5px; text-align:center;">'
                f'    <div style="font-size:0.6rem; color:#b794f4; margin-bottom:4px; font-weight:800; text-transform:uppercase;">Phase 2: Goal</div>'
                f'    <div style="font-size:1rem; font-weight:900; color:#fff; font-family:monospace;">${s2:,.2f}</div>'
                f'  </div>'
                f'</div>'
                f'<div style="font-size:0.6rem; color:#94a3b8; margin: -5px 0 15px 0; display:flex; justify-content:space-between; opacity:0.8;">'
                f'  <span>BIAS: <b style="color:#fff">{scalp_intel.get("bias", daily_bias) if scalp_intel else daily_bias}</b></span>'
                f'  <span>CONFIDENCE: <b style="color:#00E676;">MASTER</b></span>'
                f'</div>'
            )

            # SL/TP Execution Blocks (Safe Fallback)
            safe_scalp = scalp_intel if scalp_intel else {}
            
            ui_html += (
                f'<div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-bottom:15px;">'
                f'  <div style="background:linear-gradient(to bottom, rgba(255,82,82,0.1), rgba(255,82,82,0.02)); border:1px solid rgba(255,82,82,0.15); border-radius:8px; padding:12px 5px; text-align:center;">'
                f'    <div style="font-size:0.65rem; color:#FF5252; font-weight:800; margin-bottom:4px; letter-spacing:0.5px;">STOP LOSS</div>'
                f'    <div style="font-size:1rem; font-weight:900; color:#fff; font-family:monospace;">${sl_price:,.2f}</div>'
                f'  </div>'
                f'  <div style="background:linear-gradient(to bottom, rgba(0,230,118,0.1), rgba(0,230,118,0.02)); border:1px solid rgba(0,230,118,0.15); border-radius:8px; padding:12px 5px; text-align:center;">'
                f'    <div style="font-size:0.65rem; color:#00E676; font-weight:800; margin-bottom:4px; letter-spacing:0.5px;">TAKE PROFIT (MAX)</div>'
                f'    <div style="font-size:1rem; font-weight:900; color:#fff; font-family:monospace;">${tp_price:,.2f}</div>'
                f'  </div>'
                f'</div>'
                f'<div style="font-size:0.7rem; color:#94a3b8; font-style:italic; text-align:center; opacity:0.7;">STRICT MTF: HTF {h4_health} ➔ LTF {daily_bias}</div>'
                f'</div>'
            )
            _render_html(ui_html)
            
            # Key Levels
            _render_html(f"""
            <div style="display:flex; justify-content:space-around; background:rgba(0,0,0,0.2); padding:10px; border-radius:8px; margin-top:-10px; border:1px solid rgba(255,255,255,0.05);">
                <div style="text-align:center;"><small style="color:#94a3b8">VAL</small><br><b>${val:,.1f}</b></div>
                <div style="text-align:center;"><small style="color:#94a3b8">POC</small><br><b style="color:#667eea">${poc:,.1f}</b></div>
                <div style="text-align:center;"><small style="color:#94a3b8">VAH</small><br><b>${vah:,.1f}</b></div>
            </div>
            """)

        except Exception as e:
            logger.error(f"Execution engine error: {e}")


def _render_trade_plan_panel(df: pd.DataFrame, smc_state: dict, predictions: dict):
    """Render the detailed Professional Trade Plan dashboard (matching user request)."""
    p1 = predictions.get(1, {})
    daily_bias = p1.get("bias", "🟡 NEUTRAL")
    lt_bias = predictions.get(63, {}).get("bias", "🟡 NEUTRAL")
def _render_trade_plan_panel(df: pd.DataFrame, smc_state: dict, predictions: dict, scalp_intel: dict = None):
    """
    Render a high-fidelity trade plan dashboard.
    SYNCED: Now uses scalp_intel (recent structure) to avoid 'Zombie OBs' at old prices.
    """
    try:
        # Use latest price from real-time quote or fallback to last close
        symbol = st.session_state.get("global_symbol", "ASSET")
        rt_quote = st.session_state.get(f"rt_quote_{symbol}")
        curr_price = rt_quote.price if rt_quote else df["Close"].iloc[-1]
        atr_d = df["ATR_14"].iloc[-1] if "ATR_14" in df.columns else (curr_price * 0.02)
        
        p1 = predictions.get(1, {})
        # SYNC: If we have scalp_intel, use its bias to drive the Trade Plan (Unified Vision)
        effective_bias = scalp_intel.get("bias", p1.get("bias", "")) if scalp_intel else p1.get("bias", "")
        
        is_bullish = "BULLISH" in str(effective_bias).upper()
        setup_label = "BULLISH" if is_bullish else "BEARISH"
        setup_color = "#00E676" if is_bullish else "#FF5252"
        setup_icon = "↗" if is_bullish else "↘"

        # ➔ SYNC LOGIC: Use Intel SMC (Recent) instead of global smc_state (Long-term)
        # This prevents picking OBs from $10 when price is $36
        target_smc = scalp_intel.get("smc_state", smc_state) if scalp_intel else smc_state
        
        bull_obs = target_smc.get('bull_obs', [])
        bear_obs = target_smc.get('bear_obs', [])
        relevant_zones = bull_obs if is_bullish else bear_obs
        
        # Primary: SMC Order Blocks (Must be within 20% of curr_price to be valid)
        target_zone = None
        if relevant_zones:
            target_zone = min(relevant_zones, key=lambda x: abs(x['top'] - curr_price))
            # Safety check: if the "closest" zone is more than 30% away, it's a 'Zombie' zone
            if abs(target_zone['top'] - curr_price) / curr_price > 0.3:
                target_zone = None

        if target_zone:
            range_text = f"{target_zone['bottom']:,.2f} - {target_zone['top']:,.2f}"
            entry_price = target_zone['top'] if is_bullish else target_zone['bottom']
            sl_price = target_zone['bottom'] - (atr_d * 0.2) if is_bullish else target_zone['top'] + (atr_d * 0.2)
            range_source = "SMC OB"
        else:
            # Fallback 1: Recent FVG
            fvgs = target_smc.get('bull_fvgs' if is_bullish else 'bear_fvgs', [])
            target_fvg = min(fvgs, key=lambda x: abs(x['top'] - curr_price)) if fvgs else None
            
            if target_fvg and abs(target_fvg['top'] - curr_price) / curr_price < 0.2:
                range_text = f"{target_fvg['bottom']:,.2f} - {target_fvg['top']:,.2f}"
                entry_price = (target_fvg['bottom'] + target_fvg['top']) / 2
                sl_price = target_fvg['bottom'] - (atr_d * 0.2) if is_bullish else target_fvg['top'] + (atr_d * 0.2)
                range_source = "FVG GAP"
            else:
                # Fallback 2: Strategic Hub Phases (Best for BSR where 1H data is erratic)
                s0 = scalp_intel.get("step0") if scalp_intel else None
                entry_price = s0 if s0 else curr_price
                sl_price = entry_price * (0.975 if is_bullish else 1.025)
                range_text = f"AUTO: {entry_price:,.2f}"
                range_source = "PRECISION ENTRY" # Updated for strict mode

        risk = abs(entry_price - sl_price)
        if risk == 0: risk = curr_price * 0.015
        
        tp1 = entry_price + (risk * 1.5) if is_bullish else entry_price - (risk * 1.5)
        tp2 = entry_price + (risk * 2.5) if is_bullish else entry_price - (risk * 2.5)
        tp3 = entry_price + (risk * 4.0) if is_bullish else entry_price - (risk * 4.0)
        
        rr_val = abs(tp2 - entry_price) / risk if risk > 0 else 1.0
        unfilled_fvg = len(target_smc.get('bull_fvgs', [])) + len(target_smc.get('bear_fvgs', []))

        ui_html = f"""
        <div style="background: rgba(18, 18, 30, 0.95); border: 1px solid rgba(102, 126, 234, 0.3); border-top: 4px solid {setup_color}; border-radius: 12px; padding: 20px; font-family: 'Inter', sans-serif; margin-bottom: 20px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
                <div style="display:flex; align-items:center; gap:10px;">
                    <span style="font-weight:800; font-size:1rem; color:#fff; letter-spacing:0.5px;">Fair Value Gap</span>
                </div>
                <span style="font-size:0.8rem; color:#94a3b8; font-weight:600;">{symbol}</span>
            </div>
            
            <div style="font-size:0.9rem; color:#fff; margin-bottom:18px; font-weight:600; display:flex; align-items:center; gap:8px;">
                <span style="color:#94a3b8;">Unfilled:</span>
                <span style="background:rgba(255,255,255,0.1); padding:2px 8px; border-radius:4px; font-weight:800; color:#fff;">{unfilled_fvg}</span>
            </div>
            
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); padding:15px; border-radius:10px; margin-bottom:18px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <span style="color:{setup_color}; font-weight:900; font-size:1.1rem; letter-spacing:1px;">{setup_icon} {setup_label}</span>
                    <span style="color:#f6ad55; font-weight:800; font-size:0.75rem; text-transform:uppercase; background:rgba(246,173,85,0.1); padding:3px 10px; border-radius:4px; border:1px solid rgba(246,173,85,0.3);">{range_source}</span>
                </div>
                <div style="font-size:0.85rem; color:#94a3b8; margin-bottom:15px; background:rgba(0,0,0,0.2); padding:8px; border-radius:6px; text-align:center;">Range: <b style="color:#fff">{range_text}</b></div>
                
                <div style="display:flex; flex-direction:column; gap:10px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="color:#94a3b8; font-size:0.85rem;">Entry:</span>
                        <span style="color:#fff; font-weight:800; font-size:1rem; font-family:monospace;">{entry_price:,.2f}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="color:#f56565; font-size:0.85rem;">Stop Loss:</span>
                        <span style="color:#f56565; font-weight:800; font-size:1rem; font-family:monospace;">{sl_price:,.2f}</span>
                    </div>
                    <div style="height:1px; background:rgba(255,255,255,0.05); margin:5px 0;"></div>
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="color:#48bb78; font-size:0.85rem; font-weight:600;">TP1 (1.5R):</span>
                        <span style="color:#48bb78; font-weight:800; font-size:1rem; font-family:monospace;">{tp1:,.2f}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="color:#48bb78; font-size:0.85rem; font-weight:600;">TP2 (2.5R):</span>
                        <span style="color:#48bb78; font-weight:800; font-size:1rem; font-family:monospace;">{tp2:,.2f}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="color:#48bb78; font-size:0.85rem; font-weight:600;">TP3 (4R):</span>
                        <span style="color:#48bb78; font-weight:800; font-size:1rem; font-family:monospace;">{tp3:,.2f}</span>
                    </div>
                </div>
            </div>
            
            <div style="border-top:1px dashed rgba(255,255,255,0.1); padding-top:15px; display:flex; justify-content:space-between; align-items:center;">
                <span style="color:#94a3b8; font-size:0.85rem; font-weight:700;">Risk:Reward Ratio</span>
                <span style="color:#f6ad55; font-weight:900; font-size:1.1rem; background:rgba(246,173,85,0.1); padding:2px 12px; border-radius:6px;">1 : {rr_val:.1f}</span>
            </div>
            <div style="font-size:0.65rem; color:#4a5568; margin-top:12px; text-align:left; letter-spacing:0.5px;">{datetime.now().strftime('%m/%d/%Y, %I:%M:%S %p')}</div>
        </div>
        """
        _render_html(ui_html)
    except Exception as e:
        logger.error(f"Trade plan panel error: {e}")



def _render_smc_panel(df: pd.DataFrame):
    """Render the Smart Money Concepts panel."""
    with st.spinner("Scanning SMC Zones..."):
        try:
            from src.strategies.smc_analyzer import SmcAnalyzer
            from src.config import SmcConfig
            smc = SmcAnalyzer(SmcConfig())
            s = smc.get_current_state(df)
            
            struct_color = "#48bb78" if s['structure'] == "Bullish" else "#f56565" if s['structure'] == "Bearish" else "#f6ad55"
            
            def _get_liq_info(pct):
                if pct < 1.0: return "#FF3D00", "🔥 SWEEP"  # Critical Risk
                if pct < 3.0: return "#FFEA00", "⚠ ATTRAC" # Warning
                if pct < 7.0: return "#00E676", "✅ SAFE"   # Safe/Buffer
                return "#00B0FF", " NEUT"                 # Neutral/Far
            
            b_color, b_label = _get_liq_info(s['near_buy_liq_pct'])
            s_color, s_label = _get_liq_info(s['near_sell_liq_pct'])

            _render_html(f"""
            <div class="smc-card">
              <div style="color:#b794f4; font-size:0.85rem; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:8px; margin-bottom:12px;">SMC ANALYSIS</div>
              <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
                <span>Structure: <b style="color:{struct_color}">{s['structure']}</b></span>
                <span>SMC Score: <b>{s['smc_score']:+.2f}</b></span>
              </div>
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <div style="font-size:0.8rem; color:#94a3b8;">Near Buy Liq: <b style="color:#fff">{s['near_buy_liq_pct']:.1f}%</b> (${s.get('near_buy_liq_price', 0):,.0f})</div>
              </div>
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="font-size:0.8rem; color:#94a3b8;">Near Sell Liq: <b style="color:#fff">{s['near_sell_liq_pct']:.1f}%</b> (${s.get('near_sell_liq_price', 0):,.0f})</div>
              </div>
            </div>
            """)
            return s
        except Exception as e:
            logger.debug(f"SMC UI error: {e}")
            return {}


def _render_stochastic_panel(smc_state: dict):
    """Render the Stochastic Momentum Panel (SEPARATED)."""
    try:
        stoch = smc_state.get("stoch", {})
        k_val = stoch.get("k", 50.0)
        d_val = stoch.get("d", 50.0)
        status = stoch.get("status", "NEUTRAL")
        crossover = stoch.get("crossover", False)
        crossunder = stoch.get("crossunder", False)

        # Status colors
        if status == "OVERBOUGHT":
            status_color = "#FF5252"
            status_bg = "rgba(255,82,82,0.1)"
            status_icon = "🔴"
        elif status == "OVERSOLD":
            status_color = "#00E676"
            status_bg = "rgba(0,230,118,0.1)"
            status_icon = "🟢"
        else:
            status_color = "#f6ad55"
            status_bg = "rgba(246,173,85,0.1)"
            status_icon = "🟡"

        # Crossover indicator
        cross_html = ""
        if crossover:
            cross_html = '<div style="color:#00E676; font-size:0.75rem; font-weight:800; margin-top:8px;">⚡ %K × %D BULLISH CROSSOVER</div>'
        elif crossunder:
            cross_html = '<div style="color:#FF5252; font-size:0.75rem; font-weight:800; margin-top:8px;">⚡ %K × %D BEARISH CROSSUNDER</div>'

        # K gauge bar (0-100)
        k_bar_pct = min(max(k_val, 0), 100)
        k_bar_color = "#FF5252" if k_val >= 80 else "#00E676" if k_val <= 20 else "#667eea"

        _render_html(f"""
        <div style="background: rgba(26, 26, 46, 0.95); border: 1px solid rgba(183, 148, 244, 0.3);
                    border-radius: 12px; padding: 20px; margin-bottom: 20px;">
            <div style="display:flex; justify-content:space-between; align-items:center;
                        border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:8px; margin-bottom:15px;">
                <span style="color:#b794f4; font-size:0.85rem; font-weight:800; letter-spacing:1px;">📈 STOCHASTIC MOMENTUM</span>
                <span style="background:{status_bg}; color:{status_color}; font-size:0.7rem; font-weight:800;
                            padding:3px 10px; border-radius:4px; border:1px solid {status_color}44;">
                    {status_icon} {status}
                </span>
            </div>

            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px; margin-bottom:12px;">
                <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05);
                            border-radius:8px; padding:10px; text-align:center;">
                    <div style="font-size:0.65rem; color:#94a3b8; margin-bottom:4px;">%K (Fast)</div>
                    <div style="font-size:1.3rem; font-weight:900; color:#fff; font-family:monospace;">{k_val:.1f}</div>
                </div>
                <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05);
                            border-radius:8px; padding:10px; text-align:center;">
                    <div style="font-size:0.65rem; color:#94a3b8; margin-bottom:4px;">%D (Signal)</div>
                    <div style="font-size:1.3rem; font-weight:900; color:#fff; font-family:monospace;">{d_val:.1f}</div>
                </div>
            </div>

            <div style="position:relative; margin-bottom:8px;">
                <div style="background:#2d3748; height:8px; border-radius:4px; position:relative;">
                    <div style="position:absolute; left:0; width:20%; height:100%; background:rgba(0,230,118,0.15); border-radius:4px 0 0 4px;"></div>
                    <div style="position:absolute; right:0; width:20%; height:100%; background:rgba(255,82,82,0.15); border-radius:0 4px 4px 0;"></div>
                    <div style="background:{k_bar_color}; height:100%; border-radius:4px; width:{k_bar_pct}%;
                                transition:width 0.3s;"></div>
                </div>
                <div style="display:flex; justify-content:space-between; color:#64748b; font-size:0.6rem; margin-top:3px;">
                    <span>0 (Oversold)</span><span>50</span><span>100 (Overbought)</span>
                </div>
            </div>
            {cross_html}
        </div>
        """)
    except Exception as e:
        logger.debug(f"Stochastic panel error: {e}")


def _render_entry_confirmation_table(smc_state: dict):
    """Render the Entry Confirmation Table (SEPARATED)."""
    try:
        entries = smc_state.get("entry_confirmation", [])
        curr_price = smc_state.get("current_price", 0)

        ui_html = """
        <div style="background: rgba(18, 18, 32, 0.95); border: 1px solid rgba(0, 212, 255, 0.3);
                    border-radius: 12px; padding: 15px; margin-bottom: 20px;">
            <div style="font-size: 0.85rem; color: #00d4ff; font-weight: 800; letter-spacing: 1px;
                        margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px;">
                🎯 ENTRY CONFIRMATION TABLE
            </div>
            <table style="width: 100%; border-collapse: collapse; font-size: 0.8rem;">
                <thead>
                    <tr style="color: #64748b; border-bottom: 1px solid rgba(255,255,255,0.08); text-align: left;">
                        <th style="padding: 8px 4px;">ZONE</th>
                        <th style="padding: 8px 4px;">TYPE</th>
                        <th style="padding: 8px 4px;">STOCH</th>
                        <th style="padding: 8px 4px; text-align: right;">ACTION</th>
                    </tr>
                </thead>
                <tbody>
        """

        if not entries:
            ui_html += '<tr><td colspan="4" style="padding:12px; text-align:center; color:#4a5568;">No active zones</td></tr>'
        else:
            for e in entries:
                zone_text = f"{e.get('zone_bottom', 0):,.1f} - {e.get('zone_top', 0):,.1f}"
                e_type = e.get("type", "")
                direction = e.get("direction", "")
                stoch_label = e.get("stoch_label", "⚠ N/A")
                action = e.get("action", "")

                type_color = "#667eea" if e_type == "OB" else "#b794f4"
                dir_icon = "↗" if direction == "Bullish" else "↘"
                action_bg = "rgba(0,230,118,0.1)" if "BUY" in action else "rgba(255,82,82,0.1)" if "SELL" in action else "rgba(246,173,85,0.1)"

                ui_html += f"""
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                    <td style="padding:8px 4px; color:#fff; font-family:monospace; font-size:0.75rem;">{zone_text}</td>
                    <td style="padding:8px 4px;">
                        <span style="color:{type_color}; font-weight:700;">{dir_icon} {e_type}</span>
                    </td>
                    <td style="padding:8px 4px; font-size:0.75rem;">{stoch_label}</td>
                    <td style="padding:8px 4px; text-align:right;">
                        <span style="background:{action_bg}; padding:3px 8px; border-radius:4px; font-weight:800; font-size:0.75rem;">{action}</span>
                    </td>
                </tr>
                """

        stoch_k = smc_state.get("stoch", {}).get("k", 50)
        ui_html += f"""
                </tbody>
            </table>
            <div style="font-size: 0.65rem; color: #4a5568; margin-top: 10px; font-style: italic;
                        display:flex; justify-content:space-between;">
                <span>*Stochastic %K = {stoch_k:.1f}</span>
                <span>Zones filtered by momentum</span>
            </div>
        </div>
        """
        _render_html(ui_html)
    except Exception as e:
        logger.debug(f"Entry confirmation table error: {e}")


def _render_smart_entry_scanner(df: pd.DataFrame, smc_state: dict, wyckoff_state: dict):
    """Unified Smart Entry Scanner with 5-factor Confluence Scoring.

    Replaces both 'Liquidity Black Holes' and 'Entry Confirmation Table'
    with a single, non-contradictory view.
    """
    try:
        curr_price = float(smc_state.get("current_price", df["Close"].iloc[-1]))
        stoch = smc_state.get("stoch", {})
        stoch_k = stoch.get("k", 50.0)
        stoch_status = stoch.get("status", "NEUTRAL")

        # ── 1. Collect all FVG/OB zones ───────────────────────────────
        zones: list[dict] = []
        for f in smc_state.get("bull_fvgs", []):
            zones.append({"top": f["top"], "bottom": f["bottom"], "type": "FVG", "dir": "BULL"})
        for f in smc_state.get("bear_fvgs", []):
            zones.append({"top": f["top"], "bottom": f["bottom"], "type": "FVG", "dir": "BEAR"})
        for ob in smc_state.get("bull_obs", []):
            zones.append({"top": ob["top"], "bottom": ob["bottom"], "type": "OB", "dir": "BULL"})
        for ob in smc_state.get("bear_obs", []):
            zones.append({"top": ob["top"], "bottom": ob["bottom"], "type": "OB", "dir": "BEAR"})

        # ── 2. Filter zombie zones (>20% from current price) ─────────
        filtered = []
        for z in zones:
            mid = (z["top"] + z["bottom"]) / 2
            dist = abs(mid - curr_price) / curr_price if curr_price > 0 else 1
            if dist <= 0.20:
                z["proximity"] = round(dist * 100, 1)
                filtered.append(z)

        # ── 3. Fibonacci levels (recent 120-bar swing) ────────────────
        recent = df.tail(120)
        swing_high = float(recent["High"].max())
        swing_low = float(recent["Low"].min())
        swing_diff = swing_high - swing_low
        fib_levels = [
            swing_high - 0.236 * swing_diff,
            swing_high - 0.382 * swing_diff,
            swing_high - 0.500 * swing_diff,
            swing_high - 0.618 * swing_diff,
        ] if swing_diff > 0 else []

        # ── 4. Volume Profile ─────────────────────────────────────────
        try:
            vp = VolumeProfile()
            vpa = vp.analyze(df)
            vp_levels = [vpa.get("poc", 0), vpa.get("vah", 0), vpa.get("val", 0)]
            vp_levels = [v for v in vp_levels if v > 0]
        except Exception:
            vp_levels = []

        # ── 5. Wyckoff phase ──────────────────────────────────────────
        w_phase = str(wyckoff_state.get("phase", "")).upper()
        wyckoff_bullish = any(x in w_phase for x in ["PHASE B", "PHASE C", "PHASE D", "ACCUMULATION", "MARKUP"])

        # ── 6. Score each zone ────────────────────────────────────────
        for z in filtered:
            score = 0
            factors = []
            mid = (z["top"] + z["bottom"]) / 2
            zone_width = z["top"] - z["bottom"]
            margin = max(zone_width * 0.5, curr_price * 0.01)  # tolerance

            # Factor 1: FVG/OB exists (+1, always true)
            score += 1
            factors.append(z["type"])

            # Factor 2: Stochastic confirmation
            if z["dir"] == "BULL" and stoch_status != "OVERBOUGHT":
                score += 1
                factors.append("STOCH")
            elif z["dir"] == "BEAR" and stoch_status != "OVERSOLD":
                score += 1
                factors.append("STOCH")

            # Factor 3: Volume Profile overlap
            for vl in vp_levels:
                if z["bottom"] - margin <= vl <= z["top"] + margin:
                    score += 1
                    factors.append("VOL")
                    break

            # Factor 4: Fibonacci overlap
            for fl in fib_levels:
                if z["bottom"] - margin <= fl <= z["top"] + margin:
                    score += 1
                    factors.append("FIB")
                    break

            # Factor 5: Wyckoff phase alignment
            if z["dir"] == "BULL" and wyckoff_bullish:
                score += 1
                factors.append("WYK")
            elif z["dir"] == "BEAR" and not wyckoff_bullish:
                score += 1
                factors.append("WYK")

            # Factor 6: IDM / Sweep (Institutional Liquidity Grab)
            idm_bull = smc_state.get("idm_bull", False)
            idm_bear = smc_state.get("idm_bear", False)
            sweep_bull = smc_state.get("sweep_bull", False)
            sweep_bear = smc_state.get("sweep_bear", False)

            if z["dir"] == "BULL" and (idm_bull or sweep_bull):
                score += 1
                factors.append("IDM")
            elif z["dir"] == "BEAR" and (idm_bear or sweep_bear):
                score += 1
                factors.append("IDM")

            z["score"] = score
            z["factors"] = factors

            # Action label
            is_bull = z["dir"] == "BULL"
            if score >= 4:
                z["action"] = "🟢 STRONG BUY" if is_bull else "🔴 STRONG SELL"
                z["action_bg"] = "rgba(0,230,118,0.18)" if is_bull else "rgba(255,82,82,0.18)"
            elif score >= 3:
                z["action"] = "🟢 BUY" if is_bull else "🔴 SELL"
                z["action_bg"] = "rgba(0,230,118,0.10)" if is_bull else "rgba(255,82,82,0.10)"
            elif score >= 2:
                z["action"] = "🟡 WATCH"
                z["action_bg"] = "rgba(246,173,85,0.10)"
            else:
                z["action"] = " SKIP"
                z["action_bg"] = "rgba(148,163,184,0.08)"

        # Sort: score DESC → proximity ASC
        filtered.sort(key=lambda x: (-x["score"], x["proximity"]))
        display = filtered[:8]

        # ── 7. Render ─────────────────────────────────────────────────
        stoch_badge_color = "#FF5252" if stoch_status == "OVERBOUGHT" else "#00E676" if stoch_status == "OVERSOLD" else "#f6ad55"

        ui = f"""
        <div style="background:rgba(18,18,32,0.95); border:1px solid rgba(0,212,255,0.3);
                    border-top:4px solid #00d4ff; border-radius:12px; padding:20px; margin-bottom:20px;">
            <div style="display:flex; justify-content:space-between; align-items:center;
                        margin-bottom:15px;">
                <span style="color:#00d4ff; font-size:0.95rem; font-weight:800; letter-spacing:1px;">
                    🎯 SMART ENTRY SCANNER</span>
                <div style="display:flex; gap:8px; align-items:center;">
                    <span style="color:#94a3b8; font-size:0.65rem;">Stoch %K={stoch_k:.0f}</span>
                    <span style="background:{stoch_badge_color}22; color:{stoch_badge_color};
                                font-size:0.6rem; font-weight:800; padding:2px 8px;
                                border-radius:4px; border:1px solid {stoch_badge_color}44;">
                        {stoch_status}</span>
                </div>
            </div>
            <div style="color:#94a3b8; font-size:0.7rem; margin-bottom:12px;
                        border-bottom:1px solid rgba(255,255,255,0.08); padding-bottom:8px;">
                Current: <b style="color:#fff; font-family:monospace;">${curr_price:,.2f}</b>
                &nbsp;|&nbsp; Confluence: FVG · STOCH · VOL · FIB · WYK · IDM/SW
            </div>
            
            {f'''<div style="background:rgba(0,212,255,0.1); border:1px solid rgba(0,212,255,0.3); 
                             border-radius:6px; padding:6px 12px; margin-bottom:12px; 
                             font-size:0.75rem; color:#00e676; font-weight:800;">
                    ⚡ LIQUIDITY GRAB DETECTED: {'Bullish IDM/Sweep' if smc_state.get('idm_bull') or smc_state.get('sweep_bull') else 'Bearish IDM/Sweep'}
                 </div>''' if any([smc_state.get('idm_bull'), smc_state.get('idm_bear'), smc_state.get('sweep_bull'), smc_state.get('sweep_bear')]) else ''}
            <table style="width:100%; border-collapse:collapse; font-size:0.8rem;">
                <thead>
                    <tr style="color:#64748b; border-bottom:1px solid rgba(255,255,255,0.08); text-align:left;">
                        <th style="padding:8px 4px; width:5%;">#</th>
                        <th style="padding:8px 4px;">ZONE</th>
                        <th style="padding:8px 4px;">TYPE</th>
                        <th style="padding:8px 4px; text-align:center;">SCORE</th>
                        <th style="padding:8px 4px;">FACTORS</th>
                        <th style="padding:8px 4px;">PROX.</th>
                        <th style="padding:8px 4px; text-align:right;">ACTION</th>
                    </tr>
                </thead>
                <tbody>
        """

        if not display:
            ui += '<tr><td colspan="7" style="padding:15px; text-align:center; color:#4a5568;">No valid zones near current price</td></tr>'
        else:
            for idx, z in enumerate(display):
                is_top = idx == 0 and z["score"] >= 3
                row_bg = "rgba(0,212,255,0.06)" if is_top else "transparent"
                star = "" if is_top else f"{idx+1}"
                dir_icon = "↗" if z["dir"] == "BULL" else "↘"
                dir_color = "#00E676" if z["dir"] == "BULL" else "#FF5252"
                score_color = "#00E676" if z["score"] >= 4 else "#f6ad55" if z["score"] >= 3 else "#94a3b8"

                # Factor badges
                factor_html = ""
                all_factors = ["FVG", "OB", "STOCH", "VOL", "FIB", "WYK", "IDM"]
                for af in all_factors:
                    if af in z["factors"]:
                        factor_html += f'<span style="color:#00E676; font-size:0.65rem;">✓</span>'
                    elif af == z["type"]:  # skip, already counted above
                        continue
                    else:
                        factor_html += f'<span style="color:#4a5568; font-size:0.65rem;">✗</span>'
                    factor_html += " "

                ui += f"""
                <tr style="border-bottom:1px solid rgba(255,255,255,0.04); background:{row_bg};">
                    <td style="padding:8px 4px; color:#00d4ff; font-weight:800; font-size:0.85rem;">{star}</td>
                    <td style="padding:8px 4px; color:#fff; font-family:monospace; font-size:0.75rem;">
                        {z['bottom']:,.1f} - {z['top']:,.1f}</td>
                    <td style="padding:8px 4px;">
                        <span style="color:{dir_color}; font-weight:700; font-size:0.75rem;">{dir_icon} {z['dir']} {z['type']}</span></td>
                    <td style="padding:8px 4px; text-align:center;">
                        <span style="color:{score_color}; font-weight:900; font-size:1rem;">{z['score']}</span>
                        <span style="color:#64748b; font-size:0.65rem;">/5</span></td>
                    <td style="padding:8px 4px; font-size:0.7rem;">{factor_html}</td>
                    <td style="padding:8px 4px; color:#94a3b8; font-size:0.75rem;">{z['proximity']:.1f}%</td>
                    <td style="padding:8px 4px; text-align:right;">
                        <span style="background:{z['action_bg']}; padding:3px 10px; border-radius:4px;
                                    font-weight:800; font-size:0.75rem;">{z['action']}</span></td>
                </tr>
                """

        # TOP PICK callout
        top_pick_html = ""
        if display and display[0]["score"] >= 3:
            tp = display[0]
            tp_dir = "BUY" if tp["dir"] == "BULL" else "SELL"
            tp_color = "#00E676" if tp["dir"] == "BULL" else "#FF5252"
            stoch_hint = "Stoch < 30" if tp["dir"] == "BULL" else "Stoch > 70"
            top_pick_html = f"""
                <div style="background:rgba(0,212,255,0.08); border:1px dashed rgba(0,212,255,0.3);
                            border-radius:8px; padding:10px 15px; margin-top:12px;
                            display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="color:#00d4ff; font-weight:800; font-size:0.8rem;">🔑 TOP PICK:</span>
                        <span style="color:#fff; font-weight:700; font-size:0.8rem;">
                            {tp_dir} ${tp['bottom']:,.1f}-{tp['top']:,.1f}</span>
                    </div>
                    <span style="color:{tp_color}; font-size:0.7rem; font-weight:700;">
                        Score {tp['score']}/5 · {tp['proximity']:.1f}% away</span>
                </div>
            """

        ui += f"""
                </tbody>
            </table>
            {top_pick_html}
            <div style="font-size:0.6rem; color:#4a5568; margin-top:10px; display:flex;
                        justify-content:space-between; font-style:italic;">
                <span>Zones within 20% of price · Scored by confluence</span>
                <span>≥ 3/5 = Actionable</span>
            </div>
        </div>
        """
        _render_html(ui)
    except Exception as e:
        logger.error(f"Smart Entry Scanner error: {e}")


def _render_seasonality_panel(symbol: str, market: str):
    """Render the Seasonality context panel."""
    with st.spinner("Analyzing Seasonality..."):
        try:
            from src.strategies.seasonality import SeasonalityFilter
            sf = SeasonalityFilter()
            s = sf.get_current_season(symbol, market, datetime.now())
            
            _render_html(f"""
            <div class="season-card">
              <div style="color:#68d391; font-size:0.85rem; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:8px; margin-bottom:12px;">SEASONAL CONTEXT</div>
              <div style="font-weight:700;">{s['label']}</div>
              <div style="font-size:0.8rem; color:#94a3b8; margin-top:5px;">Historical Multiplier: <b>{s['weight']:.1f}x</b></div>
            </div>
            """)
        except Exception as e:
             logger.debug(f"Seasonality UI error: {e}")


def _calculate_intraday_intel(symbol: str, market: str, predictions: dict):
    """Calculate the Scalp Roadmap data without rendering UI (for merged plan)."""
    try:
        p1 = predictions.get(1, {})
        p63 = predictions.get(63, {})
        daily_bias = p1.get("bias", "🟡 NEUTRAL")
        long_term_bullish = "BULLISH" in p63.get("bias", "")
        provider = registry.get(market)
        if not provider: return None
        
        # Fetch last 7 days of 1H data
        end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        df_1h = provider.get_price_data(symbol, start, end, interval="1h")
        
        # ➔ CRITICAL FALLBACK for VN Stocks (BSR etc) that might lack 1H data
        if df_1h is None or df_1h.empty:
            logger.warning(f"1H data unavailable for {symbol}, falling back to daily for phases.")
            # Fetch more daily data to simulate structure
            start_daily = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            df_1h = provider.get_price_data(symbol, start_daily, end, interval="1d")
            if df_1h.empty: return None

        df_4h = df_1h.iloc[::4].copy() if len(df_1h) > 4 else df_1h.copy()
        predictor = AIPredictor()
        scalp = predictor.analyze_multi_tf(df_1h, df_4h)
        pattern = scalp.get("pattern", "None")
        
        from src.strategies.smc_analyzer import SmcAnalyzer
        from src.config import SmcConfig
        smc_h = SmcAnalyzer(SmcConfig())
        s_h = smc_h.get_current_state(df_1h)
        
        curr_price = df_1h["Close"].iloc[-1]
        # ADVANCED CONSENSUS: Use long-term trend to override neutral daily bias
        is_bullish = ("BULLISH" in daily_bias) or (long_term_bullish and "BEARISH" not in daily_bias)
        is_bearish = ("BEARISH" in daily_bias) or (not long_term_bullish and "BULLISH" not in daily_bias)
        
        # Phase 1 & 2 (Structure & Goal)
        if is_bullish:
            # Force Buy Steps: Step 1 is Structural Entry, Step 2 is AI Price Target
            bull_obs = s_h.get("bull_obs", [])
            step1_price = bull_obs[0]["top"] if bull_obs else (curr_price * 0.99)
            step2_price = p63.get("predicted_price", curr_price * 1.10)
        else:
            # Short Alignment
            bear_obs = s_h.get("bear_obs", [])
            step1_price = bear_obs[0]["bottom"] if bear_obs else (curr_price * 1.01)
            step2_price = p63.get("predicted_price", curr_price * 0.90)

        # ➔ NEW: Phase 0 (Immediate Entry) for Daily Trade
        # We look for a breakout of the last 4 hours to catch the move towards Step 1/2 immediately
        step0_price = None
        if is_bullish:
            # Entry on break of recent 4H High
            step0_price = df_1h["High"].rolling(4).max().iloc[-1]
            if step0_price < curr_price: step0_price = curr_price # Already breaking out
        else:
            # Entry on break of recent 4H Low
            step0_price = df_1h["Low"].rolling(4).min().iloc[-1]
            if step0_price > curr_price: step0_price = curr_price # Already breaking down

        return {
            "step0": step0_price,
            "step1": step1_price,
            "step2": step2_price,
            "bias": scalp["bias"],
            "health": scalp["health"],
            "pattern": pattern,
            "df_1h": df_1h
        }
    except Exception as e:
        logger.debug(f"Intel calculation error: {e}")
        return None


def _render_intel_card(symbol: str, p: dict):
    """Render the primary AI Intelligence card."""
    phase = p["market_phase"]
    confidence_pct = int(p["confidence"] * 100)
    bias_label = p["bias"].split(" ")[1] if " " in p["bias"] else p["bias"]
    
    _render_html(f"""
    <div class="intel-card">
        <div class="intel-header">{symbol} AI INTELLIGENCE</div>
        <div class="phase-row">
            <span>Market Phase:</span>
            <span style="color:{phase['color']}; font-weight:800;">{phase['phase']}</span>
        </div>
        <div class="conf-row">
            <div class="conf-label"><span>AI Confidence Index:</span><span>{confidence_pct}%</span></div>
            <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{confidence_pct}%"></div></div>
        </div>
        <div class="bias-box">
            <div class="bias-row"><span>Daily Bias:</span><span style="color:{'#48bb78' if 'BULLISH' in p['bias'] else '#f56565'}; font-weight:800;">{bias_label}</span></div>
            <div class="target-row"><span>T+1 Price Target:</span><span style="font-weight:800; font-size:1.2rem;">${p['predicted_price']:,.2f}</span></div>
        </div>
    </div>
    """)

def _render_money_flow_panel(qmf_state: dict):
    """Render Quant Money Flow confirmation panel."""
    score = float(qmf_state.get("score", 0.0))
    signal = int(qmf_state.get("signal", 0))
    mfi = float(qmf_state.get("mfi", 50.0))
    cmf = float(qmf_state.get("cmf", 0.0))
    reason = str(qmf_state.get("reason", "QMF_NEUTRAL"))
    anomaly_score = float(qmf_state.get("anomaly_score", 0.0))
    anomaly_flag = int(qmf_state.get("anomaly_flag", 0))
    anomaly_reason = str(qmf_state.get("anomaly_reason", "QMF_FLOW_NORMAL"))
    label = "INFLOW CONFIRMED" if signal > 0 else "OUTFLOW WARNING" if signal < 0 else "NEUTRAL FLOW"
    color = "#00E676" if signal > 0 else "#FF5252" if signal < 0 else "#f6ad55"
    width = int((score + 1.0) / 2.0 * 100)
    anomaly_color = "#FF5252" if anomaly_flag else "#4a5568"
    anomaly_pct = int(anomaly_score * 100)
    _render_html(f"""
    <div class="smc-card">
        <div style="font-size:0.85rem; color:#90cdf4; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:8px; margin-bottom:12px;">
            QUANT MONEY FLOW
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
            <span>State:</span><span style="font-weight:800; color:{color};">{label}</span>
        </div>
        <div style="background:#2d3748; height:8px; border-radius:4px; margin-bottom:8px;">
            <div style="background:{color}; height:100%; border-radius:4px; width:{width}%;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; color:#94a3b8; font-size:0.75rem;">
            <span>MFI: {mfi:.1f}</span>
            <span>CMF: {cmf:+.3f}</span>
            <span>Score: {score:+.2f}</span>
        </div>
        <div style="margin-top:8px; color:#94a3b8; font-size:0.72rem;">{reason}</div>
        <div style="margin-top:8px; color:{anomaly_color}; font-size:0.72rem;">
            Flow Anomaly: {anomaly_pct}% | {anomaly_reason}
        </div>
    </div>
    """)


def _compute_intraday_flow_shock(symbol: str, market: str) -> dict:
    """
    Intraday anomaly detector (15m preferred, fallback 1h).
    Detects early distribution before daily close.
    """
    provider = registry.get(market)
    if not provider:
        return {"available": False, "reason": "NO_PROVIDER"}

    end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

    df_i = pd.DataFrame()
    used_interval = "15m"
    try:
        df_i = provider.get_price_data(symbol, start, end, interval="15m")
    except Exception:
        df_i = pd.DataFrame()

    if df_i is None or df_i.empty or len(df_i) < 20:
        used_interval = "1h"
        try:
            df_i = provider.get_price_data(symbol, start, end, interval="1h")
        except Exception:
            df_i = pd.DataFrame()

    if df_i is None or df_i.empty or len(df_i) < 20:
        return {"available": False, "reason": "NO_INTRADAY_DATA"}

    close = df_i["Close"].astype(float)
    open_ = df_i["Open"].astype(float)
    vol = df_i["Volume"].astype(float).fillna(0.0)
    ret = close.pct_change().fillna(0.0)
    vol_ma = vol.rolling(20).mean().replace(0, np.nan)
    vol_std = vol.rolling(20).std().replace(0, np.nan)
    vol_z = ((vol - vol_ma) / vol_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_std = ret.rolling(20).std().replace(0, np.nan)
    ret_z = (ret / ret_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Intraday distribution signature:
    # large red candles + volume spike + lower-close clustering
    candle_body = (close - open_) / open_.replace(0, np.nan)
    red_pressure = np.tanh(np.maximum(-candle_body.fillna(0.0), 0.0) * 40.0)
    vol_pressure = np.tanh(np.maximum(vol_z, 0.0) / 2.2)
    ret_pressure = np.tanh(np.maximum(-ret_z, 0.0) / 2.0)

    shock = (0.35 * red_pressure + 0.35 * vol_pressure + 0.30 * ret_pressure).clip(0.0, 1.0)
    shock_now = float(shock.iloc[-1])
    shock_ema = float(shock.ewm(span=5, adjust=False).mean().iloc[-1])

    # Foreign/institution flow (optional if provider exposes columns)
    flow_cols_buy = ["foreign_buy_volume", "buy_foreign_qty", "foreigner_buy_volume"]
    flow_cols_sell = ["foreign_sell_volume", "sell_foreign_qty", "foreigner_sell_volume"]
    buy_col = next((c for c in flow_cols_buy if c in df_i.columns), None)
    sell_col = next((c for c in flow_cols_sell if c in df_i.columns), None)
    netflow = 0.0
    if buy_col and sell_col:
        netflow = float(df_i[buy_col].fillna(0.0).iloc[-1] - df_i[sell_col].fillna(0.0).iloc[-1])

    severity = "LOW"
    if shock_ema >= 0.70:
        severity = "HIGH"
    elif shock_ema >= 0.50:
        severity = "MEDIUM"

    return {
        "available": True,
        "interval": used_interval,
        "shock_now": shock_now,
        "shock_ema": shock_ema,
        "severity": severity,
        "netflow": netflow,
    }


def _render_intraday_flow_shock_panel(shock_state: dict):
    """Render intraday anomaly warning panel."""
    st.markdown("### Intraday Flow Shock Alert")
    if not shock_state.get("available"):
        st.info(f"Intraday alert unavailable: {shock_state.get('reason', 'N/A')}")
        return

    sev = shock_state.get("severity", "LOW")
    color = "#FF5252" if sev == "HIGH" else "#f6ad55" if sev == "MEDIUM" else "#00E676"
    st.metric("Severity", sev)
    c1, c2, c3 = st.columns(3)
    c1.metric("Shock (Now)", f"{shock_state.get('shock_now', 0.0) * 100:.1f}%")
    c2.metric("Shock (EMA5)", f"{shock_state.get('shock_ema', 0.0) * 100:.1f}%")
    c3.metric("Interval", str(shock_state.get("interval", "N/A")))
    st.markdown(
        f"<div style='padding:8px 10px;border:1px solid {color}55;border-radius:8px;color:{color};'>"
        f"Netflow proxy: {shock_state.get('netflow', 0.0):,.0f} | "
        f"Rule: high shock => reduce intraday exposure / tighten stop.</div>",
        unsafe_allow_html=True,
    )


def _render_hybrid_portfolio(symbol: str, curr_price: float, predictions: dict, wyckoff_state: dict):
    """Render the Hybrid Strategic Core + Tactical Satellite portfolio breakdown."""
    st.markdown("### Hybrid Portfolio Allocation (v6.0)")
    
    p63 = predictions.get(63, {})
    upside = p63.get("predicted_return", 0)
    
    # Logic for allocation
    phase = wyckoff_state.get("phase", "UNKNOWN")
    is_accumulation = any(x in str(phase).upper() for x in ["PHASE B", "PHASE C", "PHASE D"])
    
    # Alert Logic (Option B)
    if is_accumulation:
        st.toast(f"ALERT: Core Entry Available for {symbol}!", icon="💼")
        st.success(f"STRATEGIC ALERT: Market is in {phase}. Accumulation zone detected - Good for Long-term Core building.")
    elif upside > 15:
        st.toast(f"ALERT: High Conviction Markup for {symbol}!", icon="🚀")
        st.info(f"CONVICTION ALERT: AI expects {upside:.1f}% upside. Strategic markup identified.")

    sizing = predictions.get("_position_sizing", {})
    if sizing:
        core_alloc = int(sizing.get("core_alloc", 30))
        tactical_alloc = int(sizing.get("tactical_alloc", 70))
    else:
        core_alloc = 70 if is_accumulation or upside > 15 else 30
        tactical_alloc = 100 - core_alloc
    
    col1, col2 = st.columns(2)
    with col1:
        _render_html(f"""
        <div style="background:rgba(102,126,234,0.1); border:1px solid rgba(102,126,234,0.3); border-radius:12px; padding:15px; text-align:center;">
            <div style="color:#667eea; font-size:0.7rem; font-weight:800; text-transform:uppercase; letter-spacing:1px;">Strategic Core</div>
            <div style="font-size:2rem; font-weight:900; color:#fff; margin:5px 0;">{core_alloc}%</div>
            <div style="font-size:0.6rem; color:#94a3b8;">Long-term Value Component</div>
        </div>
        """)
    with col2:
        _render_html(f"""
        <div style="background:rgba(183,148,244,0.1); border:1px solid rgba(183,148,244,0.3); border-radius:12px; padding:15px; text-align:center;">
            <div style="color:#b794f4; font-size:0.7rem; font-weight:800; text-transform:uppercase; letter-spacing:1px;">Tactical Satellite</div>
            <div style="font-size:2rem; font-weight:900; color:#fff; margin:5px 0;">{tactical_alloc}%</div>
            <div style="font-size:0.6rem; color:#94a3b8;">Short-term ROI Component</div>
        </div>
        """)


def _render_liquidity_void_panel(smc_state: dict, current_price: float):
    """Render a detailed list of Unfilled Liquidity Gaps (FVGs)."""
    bull_fvgs = smc_state.get('bull_fvgs', [])
    bear_fvgs = smc_state.get('bear_fvgs', [])
    
    all_voids = []
    for f in bull_fvgs:
        dist = abs((f['top'] + f['bottom'])/2 - current_price) / current_price * 100
        all_voids.append({'type': 'BULL', 'top': f['top'], 'bottom': f['bottom'], 'dist': dist, 'color': '#00E676'})
    for f in bear_fvgs:
        dist = abs((f['top'] + f['bottom'])/2 - current_price) / current_price * 100
        all_voids.append({'type': 'BEAR', 'top': f['top'], 'bottom': f['bottom'], 'dist': dist, 'color': '#FF5252'})
        
    # Sort by proximity
    all_voids = sorted(all_voids, key=lambda x: x['dist'])[:6] # Top 6 closest

    ui_html = """
    <div style="background: rgba(18, 18, 32, 0.8); border: 1px solid rgba(102, 126, 234, 0.2); border-radius: 12px; padding: 15px; margin-bottom: 20px;">
        <div style="font-size: 0.9rem; color: #94a3b8; font-weight: 800; letter-spacing: 1px; margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 5px;">
            LIQUIDITY BLACK HOLES (UNFILLED GAPS)
        </div>
        <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
            <thead>
                <tr style="color: #64748b; border-bottom: 1px solid rgba(255,255,255,0.05); text-align: left;">
                    <th style="padding: 8px;">TYPE</th>
                    <th style="padding: 8px;">RANGE</th>
                    <th style="padding: 8px; text-align: right;">PROXIMITY</th>
                </tr>
            </thead>
            <tbody>
    """
    
    if not all_voids:
        ui_html += '<tr><td colspan="3" style="padding: 10px; text-align: center; color: #4a5568;">No major voids detected</td></tr>'
    else:
        for v in all_voids:
            ui_html += f"""
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                    <td style="padding: 8px 5px; color: {v['color']}; font-weight: 800;">{v['type']}</td>
                    <td style="padding: 8px 5px; color: #fff; font-family: monospace;">${v['bottom']:,.1f} - {v['top']:,.1f}</td>
                    <td style="padding: 8px 5px; text-align: right; color: #94a3b8;">{v['dist']:.1f}%</td>
                </tr>
            """
            
    ui_html += """
            </tbody>
        </table>
        <div style="font-size: 0.7rem; color: #4a5568; margin-top: 10px; font-style: italic;">
            *Price is magnetically pulled to fill these imbalances.
        </div>
    </div>
    """
    _render_html(ui_html)


def _render_reality_check(predictions: dict, df_outcome: pd.DataFrame):
    """Reality-check forecasts with tolerance KPI (<2%) across available horizons."""
    if df_outcome.empty:
        return

    st.markdown("### AI Reality Check")
    eval_rows = []
    for h in [1, 5, 21, 63]:
        if h not in predictions or len(df_outcome) < h:
            continue
        pred_price = float(predictions[h].get("predicted_price", np.nan))
        actual_price = float(df_outcome.iloc[h - 1]["Close"])
        curr_price = float(predictions[h].get("current_price", np.nan))
        if not np.isfinite(pred_price) or not np.isfinite(actual_price):
            continue
        abs_err = abs(pred_price - actual_price)
        ape = abs_err / max(actual_price, 1e-9)
        direction_hit = 0
        if np.isfinite(curr_price):
            direction_hit = 1 if np.sign(pred_price - curr_price) == np.sign(actual_price - curr_price) else 0
        tol_gap_pct = (ape - 0.02) * 100.0
        eval_rows.append(
            {
                "horizon": h,
                "pred": pred_price,
                "actual": actual_price,
                "abs_err": abs_err,
                "ape": ape,
                "hit_2pct": 1 if ape <= 0.02 else 0,
                "tol_gap_pct": tol_gap_pct,
                "dir_hit": direction_hit,
            }
        )

    if not eval_rows:
        st.info("Not enough forward bars for reality-check KPIs.")
        return

    eval_df = pd.DataFrame(eval_rows)
    tolerance_2pct = float(eval_df["hit_2pct"].mean() * 100.0)
    mape = float(eval_df["ape"].mean() * 100.0)
    tol_gap_avg = float(eval_df["tol_gap_pct"].mean())
    mae = float(eval_df["abs_err"].mean())
    dir_hit = float(eval_df["dir_hit"].mean() * 100.0)

    first = eval_df.iloc[0]
    first_ape_pct = float(first["ape"] * 100.0)
    first_tol_gap = float(first["tol_gap_pct"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AI Forecast", f"${first['pred']:,.2f}")
    c2.metric("Actual Close", f"${first['actual']:,.2f}")
    c3.metric("Tolerance <2%", f"{first_ape_pct:.2f}%", delta=f"{first_tol_gap:+.2f}% vs 2%")
    c4.metric("MAPE", f"{mape:.2f}%")

    c5, c6 = st.columns(2)
    c5.metric("MAE", f"${mae:,.2f}")
    c6.metric("Direction Hit", f"{dir_hit:.1f}%")

    eval_df["horizon"] = eval_df["horizon"].map({1: "1D", 5: "1W", 21: "1M", 63: "3M"})
    show = eval_df[["horizon", "pred", "actual", "abs_err", "ape", "hit_2pct", "dir_hit"]].copy()
    show["ape"] = (show["ape"] * 100).round(2)
    show["hit_2pct"] = (show["hit_2pct"] * 100).round(1)
    show["dir_hit"] = (show["dir_hit"] * 100).round(1)
    show["tol_gap_pct"] = eval_df["tol_gap_pct"].round(2)
    show = show[["horizon", "pred", "actual", "abs_err", "ape", "tol_gap_pct", "hit_2pct", "dir_hit"]]
    st.dataframe(show, width="stretch")


def _render_entry_playbook(df: pd.DataFrame, live_price: float, predictions: dict, sizing: dict, veto_state: dict):
    """Render 3-mode entry playbook with zone, SL, and position sizing."""
    if df.empty or live_price <= 0:
        return

    st.markdown("### Entry Playbook")
    p1 = predictions.get(1, {})
    p21 = predictions.get(21, {})
    p63 = predictions.get(63, {})
    conf = float(p1.get("confidence", 0.0))
    up_1m = float(p21.get("predicted_return", 0.0))
    up_3m = float(p63.get("predicted_return", 0.0))
    veto = bool(veto_state.get("active", False))

    # Robust ATR estimate (14)
    tr = pd.concat(
        [
            (df["High"] - df["Low"]).abs(),
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else float(live_price * 0.02)
    if not np.isfinite(atr) or atr <= 0:
        atr = float(live_price * 0.02)

    base_risk_pct = float(sizing.get("risk_pct", 0.01)) * 100.0
    bias = "LONG" if (up_1m + up_3m) >= 0 else "DEFENSIVE"

    modes = [
        {"mode": "Aggressive", "zone_atr": 0.35, "sl_atr": 1.20, "size_mul": 1.25},
        {"mode": "Balanced", "zone_atr": 0.65, "sl_atr": 1.70, "size_mul": 1.00},
        {"mode": "Conservative", "zone_atr": 1.10, "sl_atr": 2.20, "size_mul": 0.70},
    ]
    rows = []
    for m in modes:
        zone_low = live_price - (m["zone_atr"] * atr)
        zone_high = live_price + (m["zone_atr"] * atr * 0.35)
        sl = live_price - (m["sl_atr"] * atr)
        risk_pct = base_risk_pct * m["size_mul"] * (0.70 if veto else 1.00) * (0.75 + 0.5 * conf)
        risk_pct = float(np.clip(risk_pct, 0.20, 4.00))
        rows.append(
            {
                "Mode": m["mode"],
                "Bias": bias,
                "Entry Zone": f"{zone_low:,.2f} - {zone_high:,.2f}",
                "SL": f"{sl:,.2f}",
                "Position Size": f"{risk_pct:.2f}% risk/trade",
                "Note": "Reduce size: veto active" if veto else ("Momentum setup" if m["mode"] == "Aggressive" else "Risk-balanced"),
            }
        )

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        f"Inputs: live={live_price:,.2f} | ATR14={atr:,.2f} | conf={conf*100:.1f}% | 1M={up_1m:+.2f}% | 3M={up_3m:+.2f}%"
    )


def render():
    """Main render function for AI Forecast page."""
    st.title("AI Institutional Forecast")
    st.markdown("Professional Tier v5.0 — **Strategic Execution Hub**")
    
    # Custom CSS
    _render_html("""
    <style>
    .rhs-card, .wyckoff-card, .smc-card, .season-card, .intel-card {
        background: rgba(26, 26, 46, 0.95);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px; padding: 20px; margin-bottom: 20px;
    }
    .intel-header { font-size: 0.9rem; color: #94a3b8; letter-spacing: 1px; margin-bottom: 15px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 5px; }
    .phase-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
    .conf-row { margin-bottom: 20px; }
    .conf-bar-bg { background: #2d3748; height: 8px; border-radius: 4px; width: 100%; }
    .conf-bar-fill { background: linear-gradient(90deg, #4fd1c5, #38b2ac); height: 100%; border-radius: 4px; }
    .bias-box { background: rgba(45, 55, 72, 0.5); border-radius: 8px; padding: 15px; }
    .bias-row, .target-row { display: flex; justify-content: space-between; }
    </style>
    """)

    _render_asset_selector()
    symbol = st.session_state.get("global_symbol", "")
    market = st.session_state.get("global_market", "")
    if not symbol:
        st.info("Pick a symbol to run the AI engine.")
        return

    with st.expander("Execution Gate Controls", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            use_var_gate = st.checkbox("Enable VaR Gate", value=True, key="ai_vargate_enabled")
        with c2:
            use_kelly_cap = st.checkbox("Enable Kelly Cap", value=True, key="ai_kellycap_enabled")
        with c3:
            ai_confidence_weight = st.slider(
                "AI Confidence Weight",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.05,
                key="ai_conf_weight_slider",
            )
        with c4:
            use_live_geo_gate = st.checkbox("Enable Live OSINT Gate", value=True, key="ai_live_geo_gate")
            force_geo_refresh = st.button("Refresh OSINT", key="ai_geo_refresh")

    geo_gate_state = {}
    geo_status = {}
    if use_live_geo_gate:
        geo_gate_state, geo_status = _sync_live_geo_gate(symbol, market, force_refresh=force_geo_refresh)
    else:
        geo_gate_state = _normalize_geo_gate(st.session_state.get("geo_gate_state", {}))

    # Time-Travel Check
    with st.expander("AI Time-Travel (Point-in-Time Verification)", expanded=False):
        ref_date = st.date_input("Forecast Reference Date", value=datetime.now().date(), max_value=datetime.now().date())
    
    # ── Fetch & Prepare ──
    with st.spinner("Engine Warmup..."):
        df_full = _fetch_price_data(symbol, market)
        if df_full.empty:
            st.error("No data available.")
            return
            
        ref_dt = pd.to_datetime(ref_date)
        df_train = df_full[df_full["Date"] <= ref_dt].copy()
        df_outcome = df_full[df_full["Date"] > ref_dt].copy()

    if len(df_train) < 100:
        st.warning("Insufficient history before reference date.")
        return

    # ── Run Analysis ──
    df_feat = _compute_features(df_train)
    raw_predictions = _run_prediction(df_feat, [1, 5, 21, 63])

    if not raw_predictions:
        st.error("AI engine failure.")
        return
    predictions = deepcopy(raw_predictions)

    # Pull necessary states from analyzers (Using top-level imports)
    wa = WyckoffAnalyzer(WyckoffConfig())
    wyckoff_state = wa.analyze_current_state(df_train)
    
    smc_core = SmcAnalyzer(SmcConfig())
    smc_state = smc_core.get_current_state(df_train)
    qmf_df = QuantMoneyFlowAnalyzer().generate_signals(df_train)
    qmf_last = qmf_df.iloc[-1] if not qmf_df.empty else {}

    # ── Phase 1: Multi-Timeframe Confluence ──────────────────────────────────
    mtf_state = {}
    if _MTF_AVAILABLE:
        try:
            mtf_state = compute_mtf_confluence(symbol, market, df_daily=df_train)
        except Exception as _e:
            logger.debug(f"MTF confluence failed: {_e}")

    # ── Phase 2: Elliott Wave (Weekly strategic context) ─────────────────────
    ew_state = {}
    if _EW_AVAILABLE:
        try:
            import yfinance as _yf
            _yf_sym = f"{symbol}.VN" if market == "VN" else symbol
            _df_wk = _yf.Ticker(_yf_sym).history(period="2y", interval="1wk", auto_adjust=True)
            
            # Fallback for VN stocks if yfinance is blocked/empty
            if _df_wk.empty and market == "VN":
                from pathlib import Path
                import pandas as pd
                _cache_p = Path(__file__).resolve().parents[1] / ".cache" / "prices" / f"{symbol}_VN.parquet"
                if _cache_p.exists():
                    _df_daily = pd.read_parquet(_cache_p, engine="pyarrow")
                    if not _df_daily.empty:
                        _col_map = {c: c.capitalize() for c in _df_daily.columns}
                        _col_map.update({"Adj close": "Close", "Adj Close": "Close"})
                        _df_daily = _df_daily.rename(columns=_col_map)
                        if "Date" in _df_daily.columns:
                            _df_daily["Date"] = pd.to_datetime(_df_daily["Date"])
                            _df_daily = _df_daily.sort_values("Date")
                            _df_daily.set_index("Date", inplace=True)
                            _df_wk = _df_daily.resample("W").agg({
                                "Open": "first",
                                "High": "max",
                                "Low": "min",
                                "Close": "last",
                                "Volume": "sum"
                            }).dropna().reset_index()
                            print(f"Elliott Wave: Resampled {symbol} daily cache to weekly ({len(_df_wk)} rows)")

            if not _df_wk.empty:
                _df_wk = _df_wk.reset_index()
                _df_wk.columns = [str(c) for c in _df_wk.columns]
                ew_state = ElliottWaveAnalyzer().get_current_state(_df_wk)
        except Exception as _e:
            logger.debug(f"Elliott Wave failed: {_e}")

    # ── Phase 3: Fundamental Quality Score ──────────────────────────────────
    fund_state = {}
    if _FUND_AVAILABLE:
        try:
            _fund_key = f"fund_{symbol}"
            if _fund_key not in st.session_state or not st.session_state[_fund_key] or st.session_state[_fund_key].get("data_coverage", 0) == 0:
                st.session_state[_fund_key] = get_fundamental_dict(symbol, market)
            fund_state = st.session_state[_fund_key]
        except Exception as _e:
            logger.debug(f"Fundamental score failed: {_e}")
    qmf_state = {
        "score": float(qmf_last.get("qmf_score", 0.0)),
        "signal": int(qmf_last.get("qmf_signal", 0)),
        "reason": str(qmf_last.get("qmf_reason", "QMF_NEUTRAL")),
        "mfi": float(qmf_last.get("QMF_MFI", 50.0)),
        "cmf": float(qmf_last.get("QMF_CMF", 0.0)),
        "anomaly_score": float(qmf_last.get("qmf_anomaly_score", 0.0)),
        "anomaly_flag": int(qmf_last.get("qmf_anomaly_flag", 0)),
        "anomaly_reason": str(qmf_last.get("qmf_anomaly_reason", "QMF_FLOW_NORMAL")),
    }
    # Reliability adjustment: modest confidence calibration from money-flow confirmation
    for h in [1, 5, 21, 63]:
        if h in predictions:
            base_conf = float(predictions[h].get("confidence", 0.0))
            calibrated = np.clip(base_conf + (qmf_state["score"] * 0.08), 0.0, 1.0)
            predictions[h]["confidence"] = float(calibrated)
            predictions[h]["money_flow_score"] = qmf_state["score"]
            predictions[h]["money_flow_signal"] = qmf_state["signal"]
            predictions[h]["money_flow_reason"] = qmf_state["reason"]
    predictions = _apply_institutional_calibration(
        predictions=predictions,
        wyckoff_state=wyckoff_state,
        smc_state=smc_state,
        qmf_state=qmf_state,
    )
    shock_state = _compute_intraday_flow_shock(symbol, market)
    veto_state = _apply_auto_veto(predictions, smc_state, qmf_state)
    if qmf_state.get("anomaly_flag", 0) == 1:
        veto_state["active"] = True
        veto_state["reason"] = f"{veto_state.get('reason', 'N/A')} | {qmf_state.get('anomaly_reason', 'QMF_DISTRIBUTION_ANOMALY')}"
    if shock_state.get("available") and shock_state.get("severity") == "HIGH":
        veto_state["active"] = True
        veto_state["reason"] = f"{veto_state.get('reason', 'N/A')} | INTRADAY_FLOW_SHOCK_HIGH"

    predictions, veto_state = _apply_geo_execution_gate(predictions, veto_state, geo_gate_state)
    geo_multiplier = float(geo_gate_state.get("position_multiplier", 1.0)) if geo_gate_state else 1.0

    var_gate = _estimate_var_gate(df_train, max_var_pct=0.03) if use_var_gate else {"blocked": False, "var_pct": 0.0, "max_var_pct": 0.03}
    if var_gate.get("blocked"):
        veto_state["active"] = True
        veto_state["severity"] = "HIGH"
        veto_state["reason"] = f"{veto_state.get('reason', 'N/A')} | VAR_GATE_BLOCKED({var_gate['var_pct']:.2%}>{var_gate['max_var_pct']:.2%})"

    kelly_cap = _estimate_kelly_cap(predictions, df_train) if use_kelly_cap else None
    sizing_state = _compute_dynamic_position_sizing(
        predictions,
        wyckoff_state,
        smc_state,
        qmf_state,
        veto_state,
        ai_confidence_weight=ai_confidence_weight,
        kelly_cap=kelly_cap,
        geo_multiplier=geo_multiplier,
    )
    predictions["_veto"] = veto_state
    predictions["_position_sizing"] = sizing_state
    predictions["_raw_baseline"] = raw_predictions
    predictions["_geo_gate"] = geo_gate_state
    predictions["_geo_status"] = geo_status
    predictions["_var_gate"] = var_gate
    predictions["_kelly_cap"] = kelly_cap
    with st.spinner("Running A/B forecast validation..."):
        backtest_df = _run_ab_backtest(df_full, market, lookback_months=12)
    
    scalp_intel = _calculate_intraday_intel(symbol, market, predictions)

    # ── Fetch Real-Time Quote ──
    rt_quote = _fetch_realtime_quote(symbol, market)
    if rt_quote:
        # Store for use by child components
        st.session_state[f"rt_quote_{symbol}"] = rt_quote
        live_price = rt_quote.price
        chg_sign = "+" if rt_quote.change >= 0 else ""
        chg_color = "#00E676" if rt_quote.change >= 0 else "#FF5252"
        source_label = rt_quote.source.replace("_", " ").title()
        ts_display = datetime.fromisoformat(rt_quote.timestamp).strftime("%H:%M:%S")
    else:
        live_price = df_train["Close"].iloc[-1]
        source_label = "Historical Close"
        ts_display = "N/A"

    # ── Live Price Banner ──
    if rt_quote:
        _render_html(f"""
        <div style="background:linear-gradient(135deg, rgba(102,126,234,0.15), rgba(0,230,118,0.08));
                    border:1px solid rgba(102,126,234,0.3); border-radius:12px;
                    padding:12px 20px; margin-bottom:20px; display:flex;
                    justify-content:space-between; align-items:center;">
            <div>
                <span style="color:#94a3b8; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;">
                    {symbol} Live Price</span>
                <span style="color:#667eea; font-size:0.5rem; margin-left:6px;">LATEST: {source_label}</span>
            </div>
            <div style="display:flex; align-items:center; gap:15px;">
                <span style="font-size:1.3rem; font-weight:900; color:#fff; font-family:monospace;">
                    ${live_price:,.2f}</span>
                <span style="color:{chg_color}; font-weight:700; font-size:0.85rem;">
                    {chg_sign}{rt_quote.change:.2f}%</span>
                <span style="color:#64748b; font-size:0.6rem;">
                    Updated {ts_display}</span>
            </div>
        </div>
        """)
    _render_geo_gate_banner(geo_gate_state)
    if geo_status:
        stale = bool(geo_status.get("stale", False))
        msg = (
            f"Live OSINT feed | events={int(geo_status.get('events_count', 0))} | "
            f"freshness={float(geo_status.get('freshness_minutes', -1.0)):.1f} min"
        )
        if stale:
            st.warning(f"{msg} | status=STALE -> defensive gate enforced")
        else:
            st.caption(f"{msg} | status=FRESH")

    # -- Mozyfin Market Indices Banner --
    mozy = _get_mozyfin_client()
    if mozy:
        try:
            indices = mozy.get_market_indices()
            vn_indices = [i for i in indices if i.get("market_id") == "VN"]
            if vn_indices:
                idx_cols = st.columns(len(vn_indices[:5]))
                for col, idx in zip(idx_cols, vn_indices[:5]):
                    chg = float(idx.get("change_percent", 0))
                    val = float(idx.get("current_value", 0))
                    arrow = chr(0x25b2) if chg >= 0 else chr(0x25bc)
                    col.metric(
                        label=f"{idx.get('symbol', '')} (Mozyfin)",
                        value=f"{val:,.2f}",
                        delta=f"{arrow} {abs(chg):.2f}%",
                        delta_color="normal" if chg >= 0 else "inverse",
                    )
                st.caption("Mozyfin live index data")
        except Exception as _e:
            logger.debug(f"Mozyfin indices banner: {_e}")

    # -- Render UI Panes --
    col_l, col_r = st.columns([2, 1], gap="medium")
    
    with col_l:
        tab_chart, tab_vp, tab_ai = st.tabs(["🚀 AI Forecast", "📊 Inst. Flow (VP)", "🤖 AI Analysis"])

        # ── AI Analysis Tab (Unified: Gemini/Mozyfin + Senate Debate) ──────
        with tab_ai:
            clean_sym = symbol.replace(".VN", "").replace(".TW", "").replace(".TWO", "")
            _mkt = "TW" if ".TW" in symbol or ".TWO" in symbol or (symbol.replace(".","").isdigit() and len(symbol) <= 6) else "VN"

            # ── Section 1: AI Stock Analysis ──────────────────────────────
            st.markdown("#### 🤖 Phân tích AI Chuyên sâu")
            mozy2 = _get_mozyfin_client()
            analyst, provider, usage = _get_best_ai_analyst(mozy2)

            if provider == "mozyfin":
                used = usage.get("credits_used", 0)
                cap = usage.get("credits_cap", 50)
                st.caption(
                    f"🟢 **Mozyfin AI** | Credits còn: **{cap - used}/{cap}** "
                    f"| Gemini tự động thay thế khi hết"
                )
            elif provider == "gemini":
                st.caption(
                    "🔵 **Gemini 2.5 Flash** (Google AI) — Miễn phí 1,500 req/ngày"
                )
            else:
                st.warning("Chưa cấu hình AI. Cần MOZYFIN_API_KEY hoặc GOOGLE_API_KEY.")

            if analyst:
                ai_key = f"ai_analysis_{clean_sym}"
                if ai_key not in st.session_state:
                    st.session_state[ai_key] = ""
                    st.session_state[f"{ai_key}_prov"] = ""

                btn_label = (
                    f"🔍 Phân tích {clean_sym} bằng Mozyfin AI"
                    if provider == "mozyfin"
                    else f"🤖 Phân tích {clean_sym} bằng Gemini 2.5 Flash"
                )
                spin_msg = (
                    "Mozyfin AI đang phân tích... (~30-60 giây)"
                    if provider == "mozyfin"
                    else "Gemini 2.5 Flash đang phân tích... (~10 giây)"
                )

                if st.button(btn_label, key=f"ai_btn_{clean_sym}"):
                    with st.spinner(spin_msg):
                        try:
                            if provider == "gemini":
                                result = analyst.analyze_stock(clean_sym, market=_mkt)
                            else:
                                result = analyst.analyze_stock(clean_sym)
                        except TypeError:
                            result = analyst.analyze_stock(clean_sym)
                        st.session_state[ai_key] = result
                        st.session_state[f"{ai_key}_prov"] = provider
                        st.rerun()

                saved = st.session_state.get(ai_key, "")
                saved_prov = st.session_state.get(f"{ai_key}_prov", provider)
                if saved:
                    prov_label = (
                        "MOZYFIN AI ANALYST"
                        if saved_prov == "mozyfin"
                        else "GEMINI 2.5 FLASH — GOOGLE AI"
                    )
                    st.markdown("---")
                    _render_html(
                        '<div style="background:rgba(26,26,46,0.95);'
                        'border:1px solid rgba(102,126,234,0.3);'
                        'border-radius:12px;padding:20px;line-height:1.8;color:#e2e8f0;">'
                        f'<div style="color:#90cdf4;font-size:0.75rem;'
                        f'letter-spacing:1px;margin-bottom:12px;">'
                        f'{prov_label} — {clean_sym}</div>'
                        + saved.replace("\n", "<br>")
                        + "</div>"
                    )
                else:
                    st.info("Bấm nút để nhận phân tích AI chuyên sâu.")

            # ── Section 2: Mozyfin Entity Info + News (VN only) ──────────
            if mozy2 and _mkt == "VN":
                entity_data = {}
                news_data = []
                try:
                    entity_data = mozy2.search_entity(clean_sym)
                    news_data = mozy2.get_news(clean_sym, limit=4)
                except Exception as _e:
                    logger.debug(f"Mozyfin entity/news: {_e}")

                if entity_data:
                    st.markdown("---")
                    st.markdown("#### 🏢 Thông tin doanh nghiệp (Mozyfin)")
                    e1, e2, e3 = st.columns(3)
                    price = entity_data.get("current_price", 0)
                    mcap = entity_data.get("market_cap", 0)
                    e1.metric("Tên", entity_data.get("local_short_name") or entity_data.get("short_name", "-"))
                    e2.metric("Giá", f"{price:,.0f}" if price else "-")
                    e3.metric("Vốn hóa", f"{mcap/1e12:.1f}T VND" if mcap else "-")
                    profile = entity_data.get("profile", "")
                    if profile:
                        with st.expander("Giới thiệu doanh nghiệp", expanded=False):
                            st.write(profile[:800] + ("..." if len(profile) > 800 else ""))

                if news_data:
                    st.markdown("#### 📰 Tin tức mới nhất")
                    for article in news_data:
                        title = article.get("title", "")
                        snip = article.get("content", "")[:200]
                        st.markdown(f"**{title}**")
                        if snip:
                            st.caption(snip + "...")
                        st.divider()

            # ── Section 3: Senate Debate — 5 Members + Volume Intelligence ──
            st.markdown("---")
            st.markdown("#### ⚖️ AI Senate Debate — 5 Thành viên")
            st.caption(
                "🐂 Bull · 🐻 Bear · 🤖 Quant · 💰 Smart Money · 📰 Macro — "
                "tranh luận đa chiều với dữ liệu volume định lượng."
            )

            # Compute volume intelligence (Phase 1 — always available)
            try:
                from src.analytics.volume_intelligence import compute_volume_intelligence
                _vol_data = compute_volume_intelligence(df_train)
            except Exception as _ve:
                logger.debug(f"Volume intelligence error: {_ve}")
                _vol_data = {}

            # VN investor flow (Phase 2 — local only, graceful fallback)
            _flow_data = {}
            if _mkt == "VN":
                try:
                    from src.analytics.vn_investor_flow import get_vn_flow_intel
                    _flow_data = get_vn_flow_intel(clean_sym, df_train)
                except Exception as _fe:
                    logger.debug(f"VN flow error: {_fe}")

            # ── Volume Intelligence Dashboard ─────────────────────────────
            if _vol_data:
                sm_color = _vol_data.get("smart_money_color", "#94a3b8")
                sm_signal = _vol_data.get("smart_money_signal", "N/A")
                _render_html(f"""
                <div style="background:rgba(15,23,42,0.8);border:1px solid rgba(99,102,241,0.3);
                     border-radius:10px;padding:16px;margin-bottom:12px;">
                  <div style="color:#7c3aed;font-size:0.7rem;letter-spacing:1px;margin-bottom:10px;">
                    📊 VOLUME INTELLIGENCE
                  </div>
                  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:10px;">
                    <div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:10px;">
                      <div style="color:#94a3b8;font-size:0.65rem;">OBV Trend</div>
                      <div style="color:#e2e8f0;font-size:0.85rem;font-weight:600;">{_vol_data.get('obv_trend','N/A')}</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:10px;">
                      <div style="color:#94a3b8;font-size:0.65rem;">CMF(14)</div>
                      <div style="color:#e2e8f0;font-size:0.85rem;font-weight:600;">
                        {_vol_data.get('cmf_14',0):.3f} — {_vol_data.get('cmf_signal','N/A')}
                      </div>
                    </div>
                    <div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:10px;">
                      <div style="color:#94a3b8;font-size:0.65rem;">VWAP Deviation</div>
                      <div style="color:#e2e8f0;font-size:0.85rem;font-weight:600;">
                        {_vol_data.get('vwap_deviation',0):+.2f}% — {_vol_data.get('vwap_signal','N/A')}
                      </div>
                    </div>
                    <div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:10px;">
                      <div style="color:#94a3b8;font-size:0.65rem;">Block Trades</div>
                      <div style="color:#e2e8f0;font-size:0.85rem;font-weight:600;">
                        {_vol_data.get('block_ratio_pct',0):.0f}% — {_vol_data.get('block_direction','N/A')}
                      </div>
                    </div>
                    <div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:10px;">
                      <div style="color:#94a3b8;font-size:0.65rem;">Volume 5D vs Avg</div>
                      <div style="color:#e2e8f0;font-size:0.85rem;font-weight:600;">
                        {_vol_data.get('vol_ratio_5d_pct',100):.0f}% — {_vol_data.get('volume_delta_signal','N/A')}
                      </div>
                    </div>
                    <div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:10px;">
                      <div style="color:#94a3b8;font-size:0.65rem;">Dòng tiền (Flow)</div>
                      <div style="color:#e2e8f0;font-size:0.85rem;font-weight:600;">
                        {_flow_data.get('net_flow', 'N/A (Cloud)')}
                      </div>
                    </div>
                  </div>
                  <div style="background:rgba(255,255,255,0.06);border-radius:8px;padding:10px;
                       border-left:3px solid {sm_color};">
                    <span style="color:{sm_color};font-weight:700;font-size:0.9rem;">{sm_signal}</span>
                  </div>
                </div>
                """)

            # ── Senate Button ─────────────────────────────────────────────
            if st.button("⚖ Triệu tập Hội đồng 5 Thành viên", key="senate_btn"):
                tech_data = {
                    "price":        live_price,
                    "forecast_1d":  predictions[1]["predicted_price"],
                    "smc_trend":    smc_state.get("trend", "N/A"),
                    "wyckoff_phase": wyckoff_state.get("phase", "N/A"),
                    "ml_confidence": int(predictions[1]["confidence"] * 100),
                }
                with st.spinner("5 chuyên gia đang tranh luận... (~15 giây)"):
                    debate_transcript = advisor.get_senate_debate(
                        symbol, tech_data,
                        volume_data=_vol_data or None,
                        flow_data=_flow_data or None,
                    )
                    st.markdown(f"""
                    <div style="background:rgba(26,26,46,0.9);border:1px solid rgba(102,126,234,0.3);
                         border-radius:12px;padding:25px;line-height:1.8;color:#e2e8f0;
                         white-space:pre-wrap;">
                        {debate_transcript}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(
                    "Bấm nút để nghe 🐂 Bull, 🐻 Bear, 🤖 Quant, "
                    "💰 Smart Money và 📰 Macro tranh luận với dữ liệu định lượng."
                )

        with tab_chart:
            st.markdown("### 📈 Price & Forecast Band")

            chart_ctrls = st.columns([1, 1, 1, 1])
            with chart_ctrls[0]:
                show_ma = st.checkbox("MA (50/200)", value=False, key="chart_ma")
            with chart_ctrls[1]:
                show_fibo = st.checkbox("Fibonacci", value=False, key="chart_fibo")
            with chart_ctrls[2]:
                show_smc = st.checkbox("SMC Zones", value=False, key="chart_smc")
            with chart_ctrls[3]:
                st.markdown('<div style="text-align:right; color:#94a3b8; font-size:0.75rem; padding-top:10px;">PRO VIEW</div>', unsafe_allow_html=True)

            fig = go.Figure()

            # Base Price: Candlestick when SMC on, Line otherwise
            if show_smc:
                fig.add_trace(go.Candlestick(
                    x=df_train["Date"],
                    open=df_train["Open"],
                    high=df_train["High"],
                    low=df_train["Low"],
                    close=df_train["Close"],
                    name="Price",
                    increasing_line_color="#22c55e",
                    decreasing_line_color="#ef4444",
                    increasing_fillcolor="rgba(34,197,94,0.7)",
                    decreasing_fillcolor="rgba(239,68,68,0.7)",
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=df_train["Date"], y=df_train["Close"],
                    name="History", line=dict(color="#00d4ff", width=2)
                ))
            
            # 1. Moving Averages Overlay
            if show_ma:
                ma50 = df_train["Close"].rolling(50).mean()
                ma200 = df_train["Close"].rolling(200).mean()
                fig.add_trace(go.Scatter(x=df_train["Date"], y=ma50, name="MA 50", line=dict(color="#f6ad55", width=1, dash="dot")))
                fig.add_trace(go.Scatter(x=df_train["Date"], y=ma200, name="MA 200", line=dict(color="#f56565", width=1.5)))
                
            # 2. Fibonacci Retracement Overlay (Recent Swing)
            if show_fibo:
                recent_data = df_train.tail(120)
                swing_high = recent_data["High"].max()
                swing_low = recent_data["Low"].min()
                swing_diff = swing_high - swing_low
                
                fibo_levels = {
                    "0.0 (High)": swing_high,
                    "0.236": swing_high - 0.236 * swing_diff,
                    "0.382": swing_high - 0.382 * swing_diff,
                    "0.5 (Gold)": swing_high - 0.5 * swing_diff,
                    "0.618 (Gold)": swing_high - 0.618 * swing_diff,
                    "1.0 (Low)": swing_low
                }
                
                fibo_start_date = recent_data["Date"].iloc[0]
                fibo_end_date = df_train["Date"].iloc[-1] + timedelta(days=5)
                
                for label, level in fibo_levels.items():
                    is_gold = "(Gold)" in label
                    fig.add_trace(go.Scatter(
                        x=[fibo_start_date, fibo_end_date],
                        y=[level, level],
                        name=label,
                        line=dict(color="#b794f4" if not is_gold else "#ffd700", 
                                 width=1.5 if is_gold else 0.8,
                                 dash="solid" if is_gold else "dash"),
                        mode="lines"
                    ))

            # 3. SMC Zones Overlay (Order Blocks, FVGs, Liquidity Pools)
            if show_smc:
                _x0 = df_train["Date"].iloc[max(0, len(df_train) - 80)]
                _x1 = df_train["Date"].iloc[-1] + timedelta(days=6)

                # ── Bullish Order Blocks (Demand / Support zones)
                for _ob in smc_state.get("bull_obs", []):
                    _label = f"SUP {_ob['bottom']:,.2f}–{_ob['top']:,.2f}"
                    fig.add_hrect(
                        y0=_ob["bottom"], y1=_ob["top"],
                        x0=_x0, x1=_x1,
                        fillcolor="rgba(34,197,94,0.10)",
                        line=dict(color="rgba(34,197,94,0.65)", width=1.2),
                        annotation_text=f"🟢 {_label}",
                        annotation_position="top right",
                        annotation_font=dict(color="#4ade80", size=11, family="monospace"),
                    )

                # ── Bearish Order Blocks (Supply / Resistance zones)
                for _ob in smc_state.get("bear_obs", []):
                    _label = f"RES {_ob['bottom']:,.2f}–{_ob['top']:,.2f}"
                    fig.add_hrect(
                        y0=_ob["bottom"], y1=_ob["top"],
                        x0=_x0, x1=_x1,
                        fillcolor="rgba(239,68,68,0.10)",
                        line=dict(color="rgba(239,68,68,0.65)", width=1.2),
                        annotation_text=f"🔴 {_label}",
                        annotation_position="bottom right",
                        annotation_font=dict(color="#f87171", size=11, family="monospace"),
                    )

                # ── Bullish FVG (Gap Up — unmitigated draw zones, blue)
                for _fvg in smc_state.get("bull_fvgs", []):
                    _opa = max(0.06, 0.18 * (1 - _fvg["filled_pct"]))
                    fig.add_hrect(
                        y0=_fvg["bottom"], y1=_fvg["top"],
                        x0=_x0, x1=_x1,
                        fillcolor=f"rgba(59,130,246,{_opa:.2f})",
                        line=dict(color="rgba(96,165,250,0.55)", width=1, dash="dot"),
                        annotation_text=f"FVG↑ {_fvg['bottom']:,.2f}",
                        annotation_position="top left",
                        annotation_font=dict(color="#93c5fd", size=10),
                    )

                # ── Bearish FVG (Gap Down, orange)
                for _fvg in smc_state.get("bear_fvgs", []):
                    _opa = max(0.06, 0.18 * (1 - _fvg["filled_pct"]))
                    fig.add_hrect(
                        y0=_fvg["bottom"], y1=_fvg["top"],
                        x0=_x0, x1=_x1,
                        fillcolor=f"rgba(249,115,22,{_opa:.2f})",
                        line=dict(color="rgba(251,146,60,0.55)", width=1, dash="dot"),
                        annotation_text=f"FVG↓ {_fvg['top']:,.2f}",
                        annotation_position="bottom left",
                        annotation_font=dict(color="#fdba74", size=10),
                    )

                # ── Buy-side Liquidity Pools (Equal Highs — purple dashed)
                for _pool in smc_state.get("buy_liq", []):
                    fig.add_hline(
                        y=_pool["level"],
                        line=dict(color="rgba(168,85,247,0.75)", width=1.5, dash="dot"),
                        annotation_text=f"🟣 BSL {_pool['level']:,.2f} (×{_pool['touches']})",
                        annotation_position="top left",
                        annotation_font=dict(color="#c084fc", size=10),
                    )

                # ── Sell-side Liquidity Pools (Equal Lows — yellow dashed)
                for _pool in smc_state.get("sell_liq", []):
                    fig.add_hline(
                        y=_pool["level"],
                        line=dict(color="rgba(234,179,8,0.75)", width=1.5, dash="dot"),
                        annotation_text=f"🟡 SSL {_pool['level']:,.2f} (×{_pool['touches']})",
                        annotation_position="bottom left",
                        annotation_font=dict(color="#fde68a", size=10),
                    )

            # AI Target
            if 1 in predictions:
                p1 = predictions[1]
                next_d = df_train["Date"].iloc[-1] + timedelta(days=1)
                fig.add_trace(go.Scatter(
                    x=[df_train["Date"].iloc[-1], next_d], 
                    y=[df_train["Close"].iloc[-1], p1["predicted_price"]], 
                    name="AI Target", 
                    line=dict(color="#00E676", dash="dash", width=3)
                ))
            
            # Add real-time price marker on chart
            if rt_quote:
                fig.add_hline(y=live_price, line_dash="dot", line_color="#667eea",
                             annotation_text=f"Live: ${live_price:,.2f}",
                             annotation_font_color="#667eea",
                             annotation_position="top right")

            fig.update_layout(
                template="plotly_dark", 
                height=500, 
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            st.plotly_chart(fig, width="stretch")

            # SMC Legend
            if show_smc:
                st.markdown(
                    '<div style="font-size:0.72rem;color:#64748b;padding:4px 0;">'
                    '&nbsp;&nbsp;'
                    '<span style="color:#4ade80;">🟢 SUP</span> = Bullish OB (Demand)&nbsp;&nbsp;'
                    '<span style="color:#f87171;">🔴 RES</span> = Bearish OB (Supply)&nbsp;&nbsp;'
                    '<span style="color:#93c5fd;">🔵 FVG↑</span> = Bullish Gap&nbsp;&nbsp;'
                    '<span style="color:#fdba74;">🟠 FVG↓</span> = Bearish Gap&nbsp;&nbsp;'
                    '<span style="color:#c084fc;">🟣 BSL</span> = Buy-side Liquidity&nbsp;&nbsp;'
                    '<span style="color:#fde68a;">🟡 SSL</span> = Sell-side Liquidity'
                    '</div>',
                    unsafe_allow_html=True,
                )

        with tab_vp:
            st.markdown("###  Volume Profile Analysis")
            _render_volume_profile(df_train)


        # NEW: Trade Plan Dashboard (SYNCED with Intel)
        _render_trade_plan_panel(df_train, smc_state, predictions, scalp_intel)

        # UPGRADED: Smart Entry Scanner (replaces Liquidity Black Holes + Entry Table)
        _render_smart_entry_scanner(df_train, smc_state, wyckoff_state)

        # ── Phase 1: Multi-Timeframe Confluence Panel ────────────────────────
        _render_mtf_panel(mtf_state)

        # ── Phase 2: Elliott Wave Strategic Context Panel ────────────────────
        _render_elliott_wave_panel(ew_state)

        # ── Phase 3+4: Fundamental Score + Composite Signal Matrix ──────────
        _render_composite_score_panel(smc_state, wyckoff_state, mtf_state, ew_state, fund_state)

        # INTEGRATION: Hybrid Portfolio (Moved down)
        _render_hybrid_portfolio(symbol, live_price, predictions, wyckoff_state)

        if ref_date < datetime.now().date():
            _render_reality_check(predictions, df_outcome)

    with col_r:
        _render_intel_card(symbol, predictions[1])
        
        # RESTORED: Execution Plan (Phases 0, 1, 2)
        _render_execution_panel(df_train, smc_state, wyckoff_state, predictions, scalp_intel)
        
        # 2. Context Panels
        _render_html("<br>")
        _render_smc_panel(df_train)
        
        # Stochastic Momentum Panel
        _render_stochastic_panel(smc_state)
        _render_money_flow_panel(qmf_state)
        _render_intraday_flow_shock_panel(shock_state)
        
        _render_wyckoff_panel(df_train)
        _render_seasonality_panel(symbol, market)

    # ── Forecast Metrics ──
    st.markdown("---")
    st.markdown("### Probabilistic AI Forecast")
    p_cols = st.columns(4)
    h_labels = {1: "1 Day", 5: "1 Week", 21: "1 Month", 63: "3 Months"}
    for idx, h in enumerate([1, 5, 21, 63]):
        p = predictions[h]
        p_cols[idx].metric(h_labels[h], f"${p['predicted_price']:,.2f}", delta=f"{p['predicted_return']:+.2f}%")
        p_cols[idx].markdown(f"<small>{p['bias']}</small>", unsafe_allow_html=True)
    _render_forecast_basis(predictions)
    _render_upgrade_governance(
        backtest_df,
        sizing_state,
        veto_state,
        geo_gate=geo_gate_state,
        var_gate=var_gate,
    )
    _render_entry_playbook(df_train, live_price, predictions, sizing_state, veto_state)

    # ── Data Freshness Footer ──
    _render_html(f"""
    <div style="text-align:center; color:#64748b; font-size:0.6rem; margin-top:10px; padding:8px;
                border-top:1px solid rgba(255,255,255,0.05);">
        Historical Data: {df_train['Date'].iloc[-1].strftime('%Y-%m-%d')} |
        Live Price: {'${:,.2f} via {}'.format(live_price, source_label) if rt_quote else 'N/A'} |
        Last Refresh: {ts_display}
    </div>
    """)



# ═════════════════════════════════════════════════════
# Phase 1-4: Rendering Functions
# ═════════════════════════════════════════════════════

def _render_mtf_panel(mtf_state: dict) -> None:
    if not mtf_state:
        return
    label = mtf_state.get("confluence_label", "Unknown")
    agreement = mtf_state.get("agreement", "")
    bg_color = ("rgba(34,197,94,0.08)" if "Bullish" in label
                else "rgba(239,68,68,0.08)" if "Bearish" in label
                else "rgba(148,163,184,0.08)")
    border_color = ("#22c55e" if "Bullish" in label
                   else "#ef4444" if "Bearish" in label else "#64748b")
    with st.expander(f"📡 Multi-Timeframe Confluence — {label}", expanded=True):
        st.markdown(
            f'<div style="background:{bg_color};border-left:3px solid {border_color};padding:12px;border-radius:6px;margin-bottom:8px;">'
            f'<div style="font-size:1.1rem;font-weight:700;color:{border_color};">{label}</div>'
            f'<div style="font-size:0.8rem;color:#94a3b8;margin-top:4px;">{agreement}</div></div>',
            unsafe_allow_html=True)
        cols = st.columns(3)
        for col, (name, key) in zip(cols, [("Weekly","weekly"),("Daily","daily"),("4H","4h")]):
            tf = mtf_state.get(key, {})
            bias = tf.get("bias", "Unknown")
            icon = "🟢" if "Bullish" in bias else ("🔴" if "Bearish" in bias else "⚪")
            col.metric(label=f"{icon} {name}", value=bias,
                       help=f"Structure: {tf.get('structure','?')} | Wyckoff: {tf.get('wyckoff_phase','?')}")


def _render_elliott_wave_panel(ew_state: dict) -> None:
    if not ew_state:
        return
    wave = ew_state.get("current_wave", "?")
    bias = ew_state.get("bias", "Neutral")
    pattern = ew_state.get("pattern", "Unknown")
    target = ew_state.get("target_price", 0)
    target_label = ew_state.get("target_label", "")
    invalidation = ew_state.get("invalidation", 0)
    confidence = ew_state.get("confidence", 0)
    notes = ew_state.get("notes", "")
    violations = ew_state.get("rules_violated", [])
    icon = "🟢" if "Bullish" in bias else ("🔴" if "Bearish" in bias else "⚪")
    conf_color = "#22c55e" if confidence >= 60 else ("#f59e0b" if confidence >= 40 else "#ef4444")
    with st.expander(f"🌊 Elliott Wave — Wave {wave} ({pattern})", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div style="background:rgba(102,126,234,0.08);border-left:3px solid #667eea;padding:12px;border-radius:6px;">'
                f'<div style="font-size:0.75rem;color:#94a3b8;">Current Wave (Weekly)</div>'
                f'<div style="font-size:2rem;font-weight:900;color:#667eea;">{icon} Wave {wave}</div>'
                f'<div style="font-size:0.85rem;color:#e2e8f0;">{pattern}</div>'
                f'<div style="font-size:0.75rem;color:#94a3b8;margin-top:4px;">{notes}</div></div>',
                unsafe_allow_html=True)
        with c2:
            if target > 0:
                inv_html = (f'<div style="font-size:0.7rem;color:#ef4444;margin-top:6px;">Invalidation: {invalidation:,.2f}</div>'
                            if invalidation > 0 else '')
                st.markdown(
                    f'<div style="background:rgba(34,197,94,0.08);border-left:3px solid #22c55e;padding:12px;border-radius:6px;">'
                    f'<div style="font-size:0.75rem;color:#94a3b8;">Fibonacci Target</div>'
                    f'<div style="font-size:1.3rem;font-weight:700;color:#22c55e;">{target:,.2f}</div>'
                    f'<div style="font-size:0.7rem;color:#94a3b8;">{target_label}</div>'
                    + inv_html + f'</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:0.75rem;color:{conf_color};margin-top:8px;">Confidence: {confidence}% | Strategic context only</div>',
                    unsafe_allow_html=True)
        for v in violations:
            st.warning(f"⚠️ {v}")


def _render_composite_score_panel(smc_state, wyckoff_state, mtf_state, ew_state, fund_state):
    with st.expander("📊 Composite Signal Score Matrix", expanded=True):
        if fund_state:
            fs_score = fund_state.get("total_score", 0)
            fs_grade = fund_state.get("grade", "N/A")
            fs_label = fund_state.get("label", "")
            coverage = fund_state.get("data_coverage", 0)
            scores_detail = fund_state.get("scores", {})
            gc = "#22c55e" if fs_score >= 70 else ("#f59e0b" if fs_score >= 50 else "#ef4444")
            st.markdown("#### 🏦 Fundamental Quality Score")
            fc1, fc2 = st.columns([1, 2])
            with fc1:
                st.markdown(
                    f'<div style="text-align:center;background:rgba(0,0,0,0.3);border:2px solid {gc};border-radius:12px;padding:16px;">'
                    f'<div style="font-size:3rem;font-weight:900;color:{gc};">{fs_grade}</div>'
                    f'<div style="font-size:1.5rem;font-weight:700;color:{gc};">{fs_score}/100</div>'
                    f'<div style="font-size:0.75rem;color:#94a3b8;">{fs_label}</div>'
                    f'<div style="font-size:0.65rem;color:#475569;">Data: {coverage}% coverage</div></div>',
                    unsafe_allow_html=True)
            with fc2:
                for metric, data in scores_detail.items():
                    pts = data.get("pts", 0); maxp = data.get("max", 1); raw = data.get("raw", "N/A")
                    pct = pts / maxp if maxp > 0 else 0
                    bc = "#22c55e" if pct >= 0.7 else ("#f59e0b" if pct >= 0.4 else "#ef4444")
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
                        f'<div style="width:90px;font-size:0.7rem;color:#94a3b8;">{metric}</div>'
                        f'<div style="flex:1;background:rgba(255,255,255,0.05);border-radius:3px;height:8px;">'
                        f'<div style="width:{pct*100:.0f}%;background:{bc};height:8px;border-radius:3px;"></div></div>'
                        f'<div style="width:60px;font-size:0.7rem;color:#e2e8f0;text-align:right;">{raw}</div>'
                        f'<div style="width:40px;font-size:0.7rem;color:{bc};text-align:right;">{pts}/{maxp}</div></div>',
                        unsafe_allow_html=True)
            for cav in fund_state.get("caveats", [])[:2]:
                st.caption(f"⚠️ {cav}")
            st.markdown("---")

        st.markdown("#### 🎯 Composite Signal Score")
        factors = []
        smc_sig = smc_state.get("signal", 0)
        smc_raw = float(smc_state.get("smc_score", 0.0))
        smc_pts = min(10, max(0, int((smc_raw + 1) * 5))) if smc_sig != 0 else 5
        factors.append({"factor": "SMC Structure", "weight": 20, "score": smc_pts, "max": 10,
                        "signal": "🟢 BUY" if smc_sig==1 else ("🔴 SELL" if smc_sig==-1 else "⚪ NEUTRAL")})
        wy_bias = str(wyckoff_state.get("bias", "Neutral"))
        wy_pts = 8 if any(x in wy_bias for x in ["Bullish","Markup"]) else (2 if any(x in wy_bias for x in ["Bearish","Markdown"]) else 5)
        factors.append({"factor": "Wyckoff Phase", "weight": 15, "score": wy_pts, "max": 10,
                        "signal": wyckoff_state.get("phase", "?")})
        if mtf_state:
            mtf_raw = mtf_state.get("confluence_score", 0)
            mtf_pts = min(10, max(0, int((mtf_raw + 9) / 18 * 10)))
            factors.append({"factor": "MTF Confluence", "weight": 20, "score": mtf_pts, "max": 10,
                            "signal": mtf_state.get("confluence_label", "?")})
        if ew_state:
            ew_conf = ew_state.get("confidence", 0)
            ew_bias_s = ew_state.get("bias", "Neutral")
            ew_pts = int(ew_conf/10) if "Bullish" in ew_bias_s else (10-int(ew_conf/10) if "Bearish" in ew_bias_s else 5)
            factors.append({"factor": "Elliott Wave", "weight": 15, "score": min(10,max(0,ew_pts)), "max": 10,
                            "signal": f"Wave {ew_state.get('current_wave','?')} ({ew_bias_s})"})
        if fund_state:
            factors.append({"factor": "Fundamental", "weight": 15,
                            "score": int(fund_state.get("total_score", 50)/10), "max": 10,
                            "signal": f"Grade {fund_state.get('grade','N/A')}"})
        factors.append({"factor": "Volume/Smart Money", "weight": 10,
                        "score": min(10,max(0,int((float(smc_state.get("smc_score",0))*0.5+0.5)*10))),
                        "max": 10, "signal": "OBV/CMF/VWAP"})
        total_w = sum(f["weight"] for f in factors)
        composite = sum(f["score"]/f["max"]*f["weight"] for f in factors)/total_w*100 if total_w else 50
        if composite >= 70:   cl, cc = "STRONG BUY", "#22c55e"
        elif composite >= 55: cl, cc = "BUY", "#86efac"
        elif composite <= 30: cl, cc = "STRONG SELL", "#ef4444"
        elif composite <= 45: cl, cc = "SELL", "#fca5a5"
        else:                 cl, cc = "NEUTRAL", "#94a3b8"
        n_high = sum(1 for f in factors if f["score"]/f["max"] >= 0.7)
        conviction = "HIGH" if n_high >= len(factors)*0.6 else ("MEDIUM" if n_high >= len(factors)*0.4 else "LOW")
        st.markdown(
            f'<div style="background:rgba(0,0,0,0.4);border:2px solid {cc};border-radius:12px;padding:16px;margin-bottom:16px;text-align:center;">'
            f'<div style="font-size:0.75rem;color:#94a3b8;">COMPOSITE SIGNAL</div>'
            f'<div style="font-size:2.5rem;font-weight:900;color:{cc};">{cl}</div>'
            f'<div style="font-size:1.2rem;font-weight:700;color:{cc};">{composite:.0f}/100</div>'
            f'<div style="font-size:0.75rem;color:#64748b;margin-top:4px;">Conviction: {conviction}</div></div>',
            unsafe_allow_html=True)
        for f in factors:
            pct = f["score"] / f["max"]
            bc = "#22c55e" if pct >= 0.7 else ("#f59e0b" if pct >= 0.4 else "#ef4444")
            st.markdown(
                f'<div style="display:grid;grid-template-columns:130px 50px 1fr 80px;gap:4px;'
                f'padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04);align-items:center;">'
                f'<span style="font-size:0.7rem;color:#e2e8f0;">{f["factor"]}</span>'
                f'<span style="font-size:0.7rem;color:#64748b;">{f["weight"]}%</span>'
                f'<span style="font-size:0.65rem;color:#94a3b8;">{f["signal"]}</span>'
                f'<div style="display:flex;align-items:center;gap:3px;">'
                f'<div style="flex:1;background:rgba(255,255,255,0.05);border-radius:2px;height:6px;">'
                f'<div style="width:{int(pct*100)}%;background:{bc};height:6px;border-radius:2px;"></div></div>'
                f'<span style="font-size:0.65rem;color:{bc};width:24px;">{f["score"]}/{f["max"]}</span>'
                f'</div></div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.6rem;color:#475569;margin-top:8px;text-align:right;">'
            'SMC 20% · MTF 20% · Wyckoff 15% · Elliott 15% · Fundamental 15% · Volume 10%</div>',
            unsafe_allow_html=True)
