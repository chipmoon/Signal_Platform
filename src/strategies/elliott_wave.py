"""
Elliott Wave Engine (Practical Implementation)
===============================================
Automated wave counting using Zigzag pivot detection + Fibonacci validation.

Design Philosophy:
    - Elliott Wave on WEEKLY timeframe = Strategic Context only
    - NOT used for entries (that's Wyckoff + SMC's job)
    - Simplified but robust: Zigzag → 5-wave impulse / 3-wave correction
    - Output: current wave position + Fibonacci targets

Wave Rules enforced:
    Rule 1: Wave 2 never retraces more than 100% of Wave 1
    Rule 2: Wave 3 is never the shortest impulse wave
    Rule 3: Wave 4 never overlaps Wave 1 price territory
    Guideline: Wave 3 is often 1.618x Wave 1; Wave 5 ≈ Wave 1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class WavePoint:
    """A single pivot point in the wave count."""
    idx: int
    date: object
    price: float
    wave_label: str      # "1", "2", "3", "4", "5", "A", "B", "C"
    wave_type: Literal["high", "low"]


@dataclass
class ElliottWaveState:
    """Current Elliott Wave analysis result."""
    # Current position
    current_wave: str = "Unknown"          # "1","2","3","4","5","A","B","C"
    wave_degree: str = "Intermediate"      # "Primary","Intermediate","Minor"
    pattern: str = "Unknown"              # "Impulse", "Correction ABC"
    bias: str = "Neutral"                  # "Bullish", "Bearish", "Neutral"

    # Wave points identified
    wave_points: list[WavePoint] = field(default_factory=list)

    # Fibonacci targets
    target_price: float = 0.0             # Next wave target
    target_label: str = ""                # "Wave 3 target (1.618x)"
    invalidation: float = 0.0            # Level that invalidates current count

    # Confidence
    confidence: int = 0                   # 0-100
    rules_violated: list[str] = field(default_factory=list)
    notes: str = ""


# ─── Zigzag Pivot Detector ────────────────────────────────────────────────────

def _find_zigzag_pivots(
    df: pd.DataFrame,
    threshold_pct: float = 0.05,
) -> list[tuple[int, float, str]]:
    """
    Find significant swing highs and lows using a percentage threshold filter.

    Returns list of (index, price, 'H'|'L') for significant pivots.
    Filters out noise by requiring each swing to be > threshold_pct from previous pivot.
    """
    if len(df) < 10:
        return []

    close = df["High"].values  # Use High for swing highs
    low_arr = df["Low"].values
    n = len(df)

    # Initial pivot detection using local extremes
    pivots: list[tuple[int, float, str]] = []
    last_dir: str | None = None
    last_price = close[0]
    last_idx = 0

    for i in range(1, n):
        curr_h = close[i]
        curr_l = low_arr[i]

        # Check for new high
        if curr_h > last_price * (1 + threshold_pct):
            if last_dir == "H":
                # Update last high to higher high
                pivots[-1] = (i, curr_h, "H")
            else:
                if last_dir == "L":
                    pass  # Already recorded low, add new high
                pivots.append((i, curr_h, "H"))
                last_dir = "H"
            last_price = curr_h
            last_idx = i

        elif curr_l < last_price * (1 - threshold_pct):
            if last_dir == "L":
                # Update last low to lower low
                pivots[-1] = (i, curr_l, "L")
            else:
                pivots.append((i, curr_l, "L"))
                last_dir = "L"
            last_price = curr_l
            last_idx = i

    return pivots


def _find_significant_pivots(
    df: pd.DataFrame,
    n_pivots: int = 12,
    threshold_pct: float = 0.08,
) -> list[tuple[int, object, float, str]]:
    """
    Returns the most significant (index, date, price, type) pivot points.
    Used as the raw material for wave counting.
    """
    pivots_raw = _find_zigzag_pivots(df, threshold_pct)
    if not pivots_raw:
        return []

    dates = df.index if "Date" not in df.columns else df["Date"].values

    result = []
    for idx, price, ptype in pivots_raw[-n_pivots:]:
        date = dates[idx] if idx < len(dates) else None
        result.append((idx, date, price, ptype))

    return result


# ─── Wave Counter ─────────────────────────────────────────────────────────────

def _label_impulse_waves(pivots: list) -> list[WavePoint]:
    """
    Attempt to label a series of pivots as a 5-wave impulse.
    pivots: [(idx, date, price, type), ...]
    Starting low → needs pattern: L H L H L H (6 pivots = 5 waves)
    """
    wave_labels = ["0", "1", "2", "3", "4", "5"]
    points = []

    for i, (idx, date, price, ptype) in enumerate(pivots[:6]):
        label = wave_labels[i] if i < len(wave_labels) else f"?{i}"
        wtype = "low" if ptype == "L" else "high"
        points.append(WavePoint(idx=idx, date=date, price=price,
                                wave_label=label, wave_type=wtype))
    return points


def _validate_impulse(waves: list[WavePoint]) -> tuple[bool, list[str]]:
    """
    Validate Elliott Wave Rules 1, 2, 3 on a 5-wave impulse.
    Returns (is_valid, list_of_violations).
    """
    if len(waves) < 6:
        return False, ["Insufficient pivots"]

    prices = [w.price for w in waves]
    violations = []

    # Wave price deltas
    w1 = prices[1] - prices[0]  # Wave 1 magnitude
    w2_retrace = prices[1] - prices[2]  # Wave 2 retracement
    w3 = prices[3] - prices[2]  # Wave 3 magnitude
    w4_retrace = prices[3] - prices[4]  # Wave 4 retracement
    w5 = prices[5] - prices[4]  # Wave 5 magnitude

    # Rule 1: Wave 2 cannot retrace > 100% of Wave 1
    if w1 > 0 and w2_retrace / w1 > 1.0:
        violations.append("Rule 1: Wave 2 retraced > 100% of Wave 1")

    # Rule 2: Wave 3 is never the shortest
    wave_mags = [abs(w1), abs(w3), abs(w5)]
    if abs(w3) == min(wave_mags):
        violations.append("Rule 2: Wave 3 is shortest (invalid)")

    # Rule 3: Wave 4 does not overlap Wave 1 territory
    if prices[4] < prices[1]:
        violations.append("Rule 3: Wave 4 overlaps Wave 1 territory")

    return len(violations) == 0, violations


def _compute_wave_targets(waves: list[WavePoint]) -> tuple[float, str, float]:
    """
    Compute Fibonacci targets for next wave based on existing waves.
    Returns (target_price, target_label, invalidation_level)
    """
    if len(waves) < 3:
        return 0.0, "", 0.0

    prices = [w.price for w in waves]
    n = len(prices)
    current_wave = waves[-1].wave_label

    if current_wave == "2":
        # Projecting Wave 3: typically 1.618x Wave 1
        w1_size = prices[1] - prices[0]
        target = prices[2] + 1.618 * w1_size
        return target, "Wave 3 target (Fib 1.618× Wave 1)", prices[0]

    elif current_wave == "4":
        # Projecting Wave 5: typically equal to Wave 1
        w1_size = prices[1] - prices[0]
        w3_size = prices[3] - prices[2]
        # Wave 5 often = Wave 1, or 0.618x Wave 3
        target = prices[4] + max(w1_size, 0.618 * w3_size)
        return target, "Wave 5 target (≈ Wave 1 or 0.618× Wave 3)", prices[2]

    elif current_wave == "3":
        # Currently in Wave 3 — target is 1.618x extension
        w1_size = prices[1] - prices[0]
        target = prices[2] + 1.618 * w1_size
        return target, "Wave 3 completion (Fib 1.618× Wave 1)", prices[2]

    elif current_wave == "5":
        # Wave 5 complete — expect ABC correction
        # Target: pullback to Wave 4 area (38.2%-61.8% of full move)
        full_move = prices[-1] - prices[0]
        target = prices[-1] - 0.382 * full_move
        return target, "ABC Correction target (Fib 0.382 retracement)", prices[-1]

    return 0.0, "", 0.0


# ─── Main: Elliott Wave Analyzer ─────────────────────────────────────────────

class ElliottWaveAnalyzer:
    """
    Practical Elliott Wave analyzer.

    Usage:
        ewa = ElliottWaveAnalyzer()
        state = ewa.analyze(df_weekly)
    """

    def __init__(self, threshold_pct: float = 0.08) -> None:
        self.threshold_pct = threshold_pct

    def analyze(self, df: pd.DataFrame) -> ElliottWaveState:
        """
        Run Elliott Wave analysis on a weekly DataFrame.
        Returns ElliottWaveState with current wave position and targets.
        """
        state = ElliottWaveState()

        if df.empty or len(df) < 20:
            state.notes = "Insufficient data for Elliott Wave analysis"
            return state

        # Ensure standard columns
        required = {"High", "Low", "Close"}
        if not required.issubset(df.columns):
            state.notes = "Missing OHLCV columns"
            return state

        # Find significant pivots
        pivots = _find_significant_pivots(
            df, n_pivots=14, threshold_pct=self.threshold_pct
        )

        if len(pivots) < 4:
            state.notes = "Not enough pivot points found"
            state.confidence = 10
            return state

        # Determine overall trend direction from first and last pivot
        first_price = pivots[0][2]
        last_price = pivots[-1][2]
        is_uptrend = last_price > first_price

        # Try to find a valid 5-wave impulse in the most recent pivots
        best_waves: list[WavePoint] = []
        best_score = -1
        best_start = 0

        # Slide window to find best impulse count
        start_type = "L" if is_uptrend else "H"

        for start_i in range(len(pivots) - 5):
            if pivots[start_i][3] != start_type:
                continue

            candidate_pivots = pivots[start_i: start_i + 6]
            # Must alternate H/L
            types = [p[3] for p in candidate_pivots]
            expected = ["L","H","L","H","L","H"] if is_uptrend else ["H","L","H","L","H","L"]
            if types != expected[:len(types)]:
                continue

            waves = _label_impulse_waves(candidate_pivots)
            is_valid, violations = _validate_impulse(waves)
            score = len(waves) - len(violations) * 2
            if score > best_score:
                best_score = score
                best_waves = waves
                best_start = start_i
                state.rules_violated = violations

        if not best_waves:
            # Fallback: label whatever pivots we have as ABC correction
            state.pattern = "Correction ABC"
            state.current_wave = "B" if len(pivots) % 2 == 0 else "C"
            state.bias = "Bearish" if is_uptrend else "Bullish"
            state.notes = "No valid impulse found — likely in correction"
            state.confidence = 30
            return state

        # Determine current wave from last labeled point
        state.wave_points = best_waves
        state.current_wave = best_waves[-1].wave_label
        state.pattern = "Impulse (5-wave)"

        # Bias from current wave
        bullish_waves = {"1", "3", "5"}
        bearish_correction_waves = {"2", "4"}
        if is_uptrend:
            if state.current_wave in bullish_waves:
                state.bias = "Bullish"
            elif state.current_wave in bearish_correction_waves:
                state.bias = "Neutral — Correction in Uptrend"
            else:
                state.bias = "Neutral"
        else:
            if state.current_wave in bullish_waves:
                state.bias = "Bearish"
            else:
                state.bias = "Bearish — Correction"

        # Fibonacci targets
        target, label, invalidation = _compute_wave_targets(best_waves)
        state.target_price = round(target, 2)
        state.target_label = label
        state.invalidation = round(invalidation, 2)

        # Confidence: more pivots + fewer violations = higher confidence
        n_valid = len(best_waves)
        n_viols = len(state.rules_violated)
        state.confidence = max(10, min(90, n_valid * 12 - n_viols * 20))

        # Degree: approximate from timespan
        if len(df) > 100:
            state.wave_degree = "Primary"
        elif len(df) > 50:
            state.wave_degree = "Intermediate"
        else:
            state.wave_degree = "Minor"

        state.notes = f"Identified Wave {state.current_wave} of {state.wave_degree} degree"
        if state.rules_violated:
            state.notes += f" (⚠️ {len(state.rules_violated)} rule violation(s))"

        logger.debug(f"ElliottWave: Wave {state.current_wave} | Target {state.target_price:.2f} | Confidence {state.confidence}%")
        return state

    def get_current_state(self, df: pd.DataFrame) -> dict:
        """Return serializable dict for UI and Senate debate."""
        state = self.analyze(df)
        return {
            "current_wave": state.current_wave,
            "wave_degree": state.wave_degree,
            "pattern": state.pattern,
            "bias": state.bias,
            "target_price": state.target_price,
            "target_label": state.target_label,
            "invalidation": state.invalidation,
            "confidence": state.confidence,
            "rules_violated": state.rules_violated,
            "notes": state.notes,
            "wave_points": [
                {"label": wp.wave_label, "price": wp.price,
                 "date": str(wp.date)[:10] if wp.date is not None else ""}
                for wp in state.wave_points
            ],
        }
