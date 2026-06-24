"""Pure decision rules for the SMC daily entry radar."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.strategies.volume_profile import VolumeProfile

MAX_ZONE_AGE_BARS = 40
MAX_ZONE_WIDTH_PCT = 0.03
MAX_ZONE_WIDTH_ATR = 2.0
MAX_FILLED_PCT = 0.85
MAX_TACTICAL_RISK_PCT = 0.065
NEAR_ZONE_PCT = 0.02
READY_DISTANCE_PCT = 0.005


def build_execution_gate(
    predictions: dict,
    scalp_intel: dict | None,
    smc_state: dict,
) -> dict[str, bool]:
    """Create one shared long-execution gate for Radar and Execution UI."""
    daily_bias = str(predictions.get(1, {}).get("bias", "")).upper()
    health = str((scalp_intel or {}).get("health", "")).upper()
    daily_bullish = "BULLISH" in daily_bias
    h4_bullish = any(token in health for token in ("STRONG BUY", "BULL FLAG", "BULLISH"))
    weak_rebound = "REBOUND" in health
    h1_actionable = bool(
        scalp_intel
        and scalp_intel.get("available")
        and scalp_intel.get("actionable")
    )
    mtf_confirmed = daily_bullish and h4_bullish and not weak_rebound
    stoch = smc_state.get("stoch", {})
    bullish_trigger = bool(
        int(smc_state.get("signal", 0)) > 0
        and (
            stoch.get("crossover")
            or smc_state.get("bos_bull")
            or smc_state.get("sweep_bull")
            or smc_state.get("idm_bull")
        )
    )
    return {
        "daily_bullish": daily_bullish,
        "h4_bullish": h4_bullish,
        "h1_actionable": h1_actionable,
        "mtf_confirmed": mtf_confirmed,
        "bullish_trigger": bullish_trigger,
        "buy_triggered": mtf_confirmed and h1_actionable and bullish_trigger,
    }


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    true_range = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    value = float(true_range.rolling(period, min_periods=period).mean().iloc[-1])
    if not np.isfinite(value) or value <= 0:
        value = float(close.iloc[-1]) * 0.02
    return value


def _age_bars(zone: dict[str, Any], df: pd.DataFrame) -> int:
    try:
        candle_idx = int(zone.get("candle_idx"))
        return max(0, len(df) - 1 - candle_idx)
    except (TypeError, ValueError):
        pass
    formed_at = zone.get("formed_at")
    if formed_at is None or "Date" not in df.columns:
        return 0
    formed = pd.to_datetime(formed_at, errors="coerce")
    dates = pd.to_datetime(df["Date"], errors="coerce")
    return int((dates > formed).sum()) if pd.notna(formed) else 0


def build_smc_entry_candidates(
    df: pd.DataFrame,
    smc_state: dict,
    wyckoff_state: dict,
    *,
    decision_price: float | None = None,
    execution_gate: dict | None = None,
) -> list[dict]:
    """Rank fresh tactical zones without turning a zone touch into a BUY."""
    if df.empty:
        return []
    current = float(
        decision_price
        if decision_price is not None
        else smc_state.get("current_price", df["Close"].iloc[-1])
    )
    if not np.isfinite(current) or current <= 0:
        return []

    gate = execution_gate or {}
    atr = _atr(df)
    stoch = smc_state.get("stoch", {})
    stoch_k = float(stoch.get("k", 50.0))
    stoch_status = str(stoch.get("status", "NEUTRAL")).upper()
    zones: list[dict] = []
    for zone_type, source in (
        ("FVG", smc_state.get("bull_fvgs", [])),
        ("OB", smc_state.get("bull_obs", [])),
    ):
        for zone in source:
            zones.append({**zone, "type": zone_type})

    recent = df.tail(120)
    swing_high = float(recent["High"].max())
    swing_low = float(recent["Low"].min())
    swing_diff = max(swing_high - swing_low, 0.0)
    fib_levels = [
        swing_high - ratio * swing_diff for ratio in (0.382, 0.5, 0.618)
    ] if swing_diff > 0 else []
    try:
        profile = VolumeProfile().analyze(df)
        vp_levels = [
            float(value)
            for value in (profile.get("poc"), profile.get("vah"), profile.get("val"))
            if value and np.isfinite(float(value))
        ]
    except Exception:
        vp_levels = []

    phase = str(wyckoff_state.get("phase", "")).upper()
    wyckoff_ok = any(
        token in phase
        for token in ("PHASE B", "PHASE C", "PHASE D", "ACCUMULATION", "MARKUP")
    )
    candidates: list[dict] = []
    for zone in zones:
        top = float(zone.get("top", 0.0))
        source_bottom = float(zone.get("bottom", 0.0))
        if not (0 < source_bottom < top):
            continue
        max_tactical_width = min(current * MAX_ZONE_WIDTH_PCT, atr * MAX_ZONE_WIDTH_ATR)
        bottom = max(source_bottom, top - max_tactical_width)
        width = top - bottom
        width_pct = width / current
        width_atr = width / atr
        proximal_slice = bottom > source_bottom
        age = _age_bars(zone, df)
        filled_pct = float(zone.get("filled_pct", 0.0))
        if (
            age > MAX_ZONE_AGE_BARS
            or filled_pct >= MAX_FILLED_PCT
        ):
            continue

        if current > top:
            distance = (current - top) / current
        elif current < bottom:
            distance = (bottom - current) / current
        else:
            distance = 0.0
        if distance > 0.18:
            continue
        inside = bottom <= current <= top

        score = 1
        factors = [zone["type"]]
        if stoch_status != "OVERBOUGHT":
            score += 1
            factors.append("STOCH_OK")
        if stoch_k <= 35:
            score += 1
            factors.append("OVERSOLD")
        if any(bottom - width <= level <= top + width for level in vp_levels):
            score += 1
            factors.append("VOL_PROFILE")
        if any(bottom - width <= level <= top + width for level in fib_levels):
            score += 1
            factors.append("FIB")
        if wyckoff_ok:
            score += 1
            factors.append("WYCKOFF")
        if smc_state.get("sweep_bull") or smc_state.get("idm_bull"):
            score += 1
            factors.append("LIQ_SWEEP")
        if int(smc_state.get("signal", 0)) > 0:
            score += 1
            factors.append("SMC_SIGNAL")
        if proximal_slice:
            factors.append("PROXIMAL_SLICE")

        stop_buffer = max(width * 0.20, atr * 0.35, current * 0.0075)
        stop = max(0.01, bottom - stop_buffer)
        risk = top - stop
        if risk <= 0 or risk / top > MAX_TACTICAL_RISK_PCT:
            continue
        target = top + 2.0 * risk
        rr = (target - top) / risk
        near_touch = inside or distance <= READY_DISTANCE_PCT
        if near_touch and score >= 5 and gate.get("buy_triggered"):
            status = "BUY_TRIGGERED"
        elif (
            near_touch
            and score >= 5
            and gate.get("mtf_confirmed")
            and gate.get("h1_actionable")
        ):
            status = "SETUP_READY"
        elif inside and score >= 4:
            status = "ZONE_TOUCH"
        elif distance <= NEAR_ZONE_PCT and score >= 4:
            status = "NEAR_ZONE"
        elif score >= 4:
            status = "WATCH"
        else:
            status = "WAIT"

        candidates.append({
            "status": status,
            "type": zone["type"],
            "timeframe": "1D",
            "entry_low": bottom,
            "entry_high": top,
            "source_low": source_bottom,
            "stop": stop,
            "target": target,
            "rr": rr,
            "score": min(score, 8),
            "distance_pct": distance * 100.0,
            "inside": inside,
            "age_bars": age,
            "filled_pct": filled_pct,
            "width_atr": width_atr,
            "factors": factors,
        })

    rank = {
        "BUY_TRIGGERED": 0,
        "SETUP_READY": 1,
        "ZONE_TOUCH": 2,
        "NEAR_ZONE": 3,
        "WATCH": 4,
        "WAIT": 5,
    }
    candidates.sort(
        key=lambda item: (
            rank.get(item["status"], 9),
            -item["score"],
            item["distance_pct"],
            item["age_bars"],
        )
    )
    return candidates
