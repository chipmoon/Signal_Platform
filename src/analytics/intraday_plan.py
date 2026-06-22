"""Pure, testable intraday execution-plan rules."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.vn_price import vn_price_scale_is_consistent

MIN_INTRADAY_STOP_PCT = 0.015
MAX_INTRADAY_STOP_PCT = 0.04
MAX_PHASE0_DISTANCE_PCT = 0.03
MAX_INTRADAY_ZONE_DISTANCE_PCT = 0.03
MAX_INTRADAY_TARGET_PCT = 0.06
MIN_RISK_REWARD = 1.5


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    true_range = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    value = float(true_range.rolling(period).mean().iloc[-1])
    return value if np.isfinite(value) and value > 0 else float(close.iloc[-1]) * 0.01


def validate_intraday_bars(
    df: pd.DataFrame | None,
    *,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """Reject daily fallbacks, stale bars, scale jumps and implausible H1 gaps."""
    if df is None or df.empty:
        return False, "H1 data unavailable"
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        return False, "H1 columns incomplete"
    if len(df) < 14:
        return False, "Need at least 14 H1 bars"
    if not vn_price_scale_is_consistent(df):
        return False, "Mixed price units"

    dates = pd.to_datetime(df["Date"], errors="coerce")
    if dates.isna().all():
        return False, "Invalid H1 timestamps"
    has_intraday_time = bool((dates.dt.normalize() != dates).any())
    has_multiple_bars_per_day = bool(dates.dt.normalize().duplicated().any())
    if not (has_intraday_time and has_multiple_bars_per_day):
        return False, "Daily fallback is not an intraday feed"

    close = pd.to_numeric(df["Close"], errors="coerce")
    if close.isna().any() or (close <= 0).any():
        return False, "Invalid H1 prices"
    if close.pct_change().abs().dropna().max() > 0.15:
        return False, "Implausible H1 price jump"

    reference = pd.Timestamp(now or datetime.now()).tz_localize(None)
    latest = dates.max()
    if getattr(latest, "tzinfo", None) is not None:
        latest = latest.tz_localize(None)
    if reference - latest > pd.Timedelta(days=4):
        return False, "H1 data is stale"
    return True, "OK"


def select_nearby_zone(
    zones: list[dict[str, Any]],
    current_price: float,
    *,
    side: str,
    max_distance_pct: float = MAX_INTRADAY_ZONE_DISTANCE_PCT,
) -> dict[str, Any] | None:
    """Select the closest valid zone; old distant order blocks are ignored."""
    if current_price <= 0:
        return None
    candidates: list[tuple[float, dict[str, Any]]] = []
    for zone in zones or []:
        try:
            top = float(zone["top"])
            bottom = float(zone["bottom"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 < bottom <= top):
            continue
        anchor = top if side == "long" else bottom
        correct_side = anchor < current_price if side == "long" else anchor > current_price
        distance = abs(anchor - current_price) / current_price
        if correct_side and distance <= max_distance_pct:
            candidates.append((distance, zone))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def _bounded_stop(entry: float, candidate: float, *, side: str) -> float:
    if side == "long":
        return float(np.clip(
            candidate,
            entry * (1.0 - MAX_INTRADAY_STOP_PCT),
            entry * (1.0 - MIN_INTRADAY_STOP_PCT),
        ))
    return float(np.clip(
        candidate,
        entry * (1.0 + MIN_INTRADAY_STOP_PCT),
        entry * (1.0 + MAX_INTRADAY_STOP_PCT),
    ))


def validate_trade_plan(plan: dict[str, Any]) -> tuple[bool, str]:
    """Enforce direction, stop distance, target ordering and minimum R:R."""
    try:
        side = str(plan["side"])
        entry = float(plan["entry"])
        stop = float(plan["stop"])
        target = float(plan["target"])
    except (KeyError, TypeError, ValueError):
        return False, "Plan prices incomplete"
    if not all(np.isfinite(value) and value > 0 for value in (entry, stop, target)):
        return False, "Plan prices invalid"
    if side == "long" and not (stop < entry < target):
        return False, "Long plan ordering invalid"
    if side == "short" and not (target < entry < stop):
        return False, "Short plan ordering invalid"
    risk = abs(entry - stop)
    reward = abs(target - entry)
    stop_pct = risk / entry
    if not MIN_INTRADAY_STOP_PCT - 1e-9 <= stop_pct <= MAX_INTRADAY_STOP_PCT + 1e-9:
        return False, "Intraday stop outside 1.5%-4.0%"
    if reward / risk < MIN_RISK_REWARD - 1e-9:
        return False, "Risk:reward below 1.5"
    return True, "OK"


def build_intraday_plan(
    df: pd.DataFrame,
    *,
    side: str,
    bull_obs: list[dict[str, Any]] | None = None,
    bear_obs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a bounded H1-only plan; no long-horizon forecast is consumed."""
    if side not in {"long", "short"}:
        return {"actionable": False, "reason": "Directional MTF confirmation required"}
    valid, reason = validate_intraday_bars(df)
    if not valid:
        return {"actionable": False, "reason": reason}

    current = float(pd.to_numeric(df["Close"], errors="coerce").iloc[-1])
    atr_h = _atr(df)
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    previous_high = float(high.shift(1).rolling(4).max().iloc[-1])
    previous_low = float(low.shift(1).rolling(4).min().iloc[-1])

    if side == "long":
        zone = select_nearby_zone(bull_obs or [], current, side=side)
        entry = float(zone["top"]) if zone else current * 0.995
        phase0 = max(current, previous_high)
        if (phase0 - current) / current > MAX_PHASE0_DISTANCE_PCT:
            phase0 = None
        raw_stop = float(zone["bottom"]) - 0.2 * atr_h if zone else entry - 1.5 * atr_h
        stop = _bounded_stop(entry, raw_stop, side=side)
        risk = entry - stop
        recent_structure = float(high.shift(1).tail(20).max())
        target = max(recent_structure, current + 2.0 * atr_h, entry + MIN_RISK_REWARD * risk)
        target = min(target, entry * (1.0 + MAX_INTRADAY_TARGET_PCT))
        break_even_trigger = entry + 0.70 * (target - entry)
        if current >= break_even_trigger:
            stop = max(stop, entry)
    else:
        zone = select_nearby_zone(bear_obs or [], current, side=side)
        entry = float(zone["bottom"]) if zone else current * 1.005
        phase0 = min(current, previous_low)
        if (current - phase0) / current > MAX_PHASE0_DISTANCE_PCT:
            phase0 = None
        raw_stop = float(zone["top"]) + 0.2 * atr_h if zone else entry + 1.5 * atr_h
        stop = _bounded_stop(entry, raw_stop, side=side)
        risk = stop - entry
        recent_structure = float(low.shift(1).tail(20).min())
        target = min(recent_structure, current - 2.0 * atr_h, entry - MIN_RISK_REWARD * risk)
        target = max(target, entry * (1.0 - MAX_INTRADAY_TARGET_PCT))
        break_even_trigger = entry - 0.70 * (entry - target)
        if current <= break_even_trigger:
            stop = min(stop, entry)

    plan = {
        "actionable": True,
        "reason": "OK",
        "side": side,
        "current": current,
        "phase0": phase0,
        "entry": entry,
        "stop": stop,
        "target": target,
        "atr_h": atr_h,
        "break_even_trigger": break_even_trigger,
        "break_even_active": stop == entry,
        "zone": zone,
    }
    valid, reason = validate_trade_plan(plan)
    # A break-even stop is valid after the 70% trigger even though risk is zero.
    if plan["break_even_active"]:
        valid, reason = True, "Break-even protected"
    plan["actionable"] = valid
    plan["reason"] = reason
    return plan
