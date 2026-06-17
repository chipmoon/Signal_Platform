"""
SMC Analyzer — Smart Money Concepts Engine
==========================================
Implements modern institutional trading concepts that complement Wyckoff:

Concepts:
    1. Order Blocks (OB) — Institutional supply/demand zones
       - Bullish OB: Last bearish candle before a strong rally
       - Bearish OB: Last bullish candle before a strong selloff

    2. Fair Value Gap (FVG) / Imbalance
       - 3-candle pattern: candle 2 body doesn't overlap with candle 1 & 3 wicks
       - Unmitigated FVGs are high-probability draw zones for price

    3. Liquidity Pools
       - Equal Highs → Buy-side Liquidity (target for smart money sell stops)
       - Equal Lows  → Sell-side Liquidity (target for smart money buy stops)

    4. Market Structure
       - BOS (Break of Structure): Continuation confirmation
       - ChoCH (Change of Character): Reversal first signal

Integration:
    - Feeds smc_signal (-1/0/1) into Signal Combiner
    - Feeds scalar scores into AIPredictor._prepare_features()
    - Displays SMC zones overlay on ai_forecast.py chart

Conforms to ``TradingStrategy`` Protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd
from loguru import logger

from src.config import SmcConfig, StochasticConfig
from src.strategies.momentum import StochasticOscillator


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

class OrderBlock(NamedTuple):
    """Represents an unmitigated Order Block zone."""
    direction: str        # "Bullish" or "Bearish"
    top: float            # Upper boundary of OB
    bottom: float         # Lower boundary of OB
    candle_idx: int       # Index in DataFrame
    date: object          # Date for display
    strength: float       # Move strength after OB (0-1 normalized)
    mitigated: bool       # Has price returned to this zone?


class FairValueGap(NamedTuple):
    """Represents an unmitigated Fair Value Gap (imbalance)."""
    direction: str        # "Bullish" (gap up) or "Bearish" (gap down)
    top: float            # Upper boundary of gap
    bottom: float         # Lower boundary of gap
    candle_idx: int       # Index of middle candle
    date: object
    filled_pct: float     # 0.0 = unfilled, 1.0 = fully filled


class LiquidityPool(NamedTuple):
    """Represents a Liquidity Pool (cluster of stops)."""
    side: str             # "Buy-side" (equal highs) or "Sell-side" (equal lows)
    level: float          # Price level
    touches: int          # How many times price reached this level
    candle_idx: int
    date: object
    swept: bool           # Has this pool been swept?
    is_sweep_event: bool = False  # True if current candle is performing a sweep


class Inducement(NamedTuple):
    """Represents a Smart Money Inducement (IDM) level."""
    type: str             # "Bullish" or "Bearish"
    level: float
    candle_idx: int
    date: object
    taken: bool           # Has this IDM been taken (Inducement successful)?


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Swing Highs / Lows
# ─────────────────────────────────────────────────────────────────────────────

def _find_swing_highs(high: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    """Return boolean Series where True = swing high."""
    n = len(high)
    result = pd.Series(False, index=high.index)
    arr = high.values
    for i in range(left, n - right):
        window = arr[i - left: i + right + 1]
        if arr[i] == window.max() and arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            result.iloc[i] = True
    return result


def _find_swing_lows(low: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    """Return boolean Series where True = swing low."""
    n = len(low)
    result = pd.Series(False, index=low.index)
    arr = low.values
    for i in range(left, n - right):
        window = arr[i - left: i + right + 1]
        if arr[i] == window.min() and arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            result.iloc[i] = True
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Task 1A: Order Block Detector
# ─────────────────────────────────────────────────────────────────────────────

def detect_order_blocks(
    df: pd.DataFrame, cfg: SmcConfig
) -> tuple[list[OrderBlock], list[OrderBlock]]:
    """
    Detect Bullish and Bearish Order Blocks.

    Bullish OB = Last consecutive bearish candle(s) before a strong up-move.
    Rule:
        1. Find a strong bullish move: close > prev_close * (1 + threshold)
        2. Look back 1-3 bars to find the last down-bar → That is the OB
        3. OB zone = [low of that candle, high of that candle]
        4. Mark as mitigated if price has since returned to that zone

    Bearish OB = mirror logic.
    """
    if len(df) < 20:
        return [], []

    close = df["Close"].values
    open_ = df["Open"].values
    high = df["High"].values
    low = df["Low"].values
    dates = df["Date"].values if "Date" in df.columns else df.index.values

    atr_raw = pd.Series(
        np.maximum(high - low,
                   np.maximum(np.abs(high - np.roll(close, 1)),
                              np.abs(low - np.roll(close, 1))))
    )
    atr = atr_raw.rolling(14, min_periods=1).mean().values

    bull_obs: list[OrderBlock] = []
    bear_obs: list[OrderBlock] = []
    threshold = cfg.ob_impulse_multiplier  # e.g. 1.5x ATR for qualifying move

    for i in range(5, len(df) - 1):
        # ── Bullish OB: Strong up-candle preceded by down-candle(s)
        move_up = close[i] - close[i - 1]
        if move_up >= threshold * atr[i]:
            # Find last bearish candle in prior 1-cfg.ob_lookback bars
            for j in range(i - 1, max(i - cfg.ob_lookback - 1, 0), -1):
                if close[j] < open_[j]:  # Down-bar = bearish candle
                    ob_top = high[j]
                    ob_bottom = low[j]
                    strength = min(move_up / (atr[i] * threshold), 1.0)
                    # Check mitigation: did price return to this zone after OB?
                    subsequent_lows = low[j + 1: i + 1]
                    mitigated = bool(np.any(subsequent_lows <= ob_top))
                    bull_obs.append(OrderBlock(
                        direction="Bullish",
                        top=float(ob_top),
                        bottom=float(ob_bottom),
                        candle_idx=j,
                        date=dates[j],
                        strength=float(strength),
                        mitigated=mitigated,
                    ))
                    break   # Only the last bearish candle

        # ── Bearish OB: Strong down-candle preceded by up-candle(s)
        move_down = close[i - 1] - close[i]
        if move_down >= threshold * atr[i]:
            for j in range(i - 1, max(i - cfg.ob_lookback - 1, 0), -1):
                if close[j] > open_[j]:   # Up-bar = bullish candle
                    ob_top = high[j]
                    ob_bottom = low[j]
                    strength = min(move_down / (atr[i] * threshold), 1.0)
                    subsequent_highs = high[j + 1: i + 1]
                    mitigated = bool(np.any(subsequent_highs >= ob_bottom))
                    bear_obs.append(OrderBlock(
                        direction="Bearish",
                        top=float(ob_top),
                        bottom=float(ob_bottom),
                        candle_idx=j,
                        date=dates[j],
                        strength=float(strength),
                        mitigated=mitigated,
                    ))
                    break

    # Keep only unmitigated OBs + most recent N
    unmitigated_bull = [ob for ob in bull_obs if not ob.mitigated][-cfg.ob_max_zones:]
    unmitigated_bear = [ob for ob in bear_obs if not ob.mitigated][-cfg.ob_max_zones:]

    logger.debug(
        f"Order Blocks: {len(unmitigated_bull)} bullish, {len(unmitigated_bear)} bearish (unmitigated)"
    )
    return unmitigated_bull, unmitigated_bear


# ─────────────────────────────────────────────────────────────────────────────
# Task 1B: Fair Value Gap Detector
# ─────────────────────────────────────────────────────────────────────────────

def detect_fair_value_gaps(
    df: pd.DataFrame, cfg: SmcConfig
) -> tuple[list[FairValueGap], list[FairValueGap]]:
    """
    Detect Bullish and Bearish Fair Value Gaps (imbalances).

    Bullish FVG (gap up):
        Candle[i-2].high < Candle[i].low
        → Gap between top of candle 1 and bottom of candle 3

    Bearish FVG (gap down):
        Candle[i-2].low > Candle[i].high
        → Gap between bottom of candle 1 and top of candle 3

    Filled % = how much of the gap has price retraced into.
    """
    if len(df) < 5:
        return [], []

    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    dates = df["Date"].values if "Date" in df.columns else df.index.values

    bull_fvgs: list[FairValueGap] = []
    bear_fvgs: list[FairValueGap] = []

    # Minimum FVG size as % of price
    for i in range(2, len(df)):
        # Bullish FVG: gap up
        if low[i] > high[i - 2]:
            gap_top = low[i]
            gap_bottom = high[i - 2]
            gap_size_pct = (gap_top - gap_bottom) / close[i]
            if gap_size_pct < cfg.fvg_min_size_pct:
                continue
            # Check fill: subsequent candles entering the gap
            filled = 0.0
            for k in range(i + 1, len(df)):
                if low[k] <= gap_top:
                    entry = min(low[k], gap_top)
                    filled = 1.0 - max(0.0, (entry - gap_bottom) / (gap_top - gap_bottom))
                    break
            bull_fvgs.append(FairValueGap(
                direction="Bullish",
                top=float(gap_top),
                bottom=float(gap_bottom),
                candle_idx=i - 1,
                date=dates[i - 1],
                filled_pct=float(min(filled, 1.0)),
            ))

        # Bearish FVG: gap down
        elif high[i] < low[i - 2]:
            gap_top = low[i - 2]
            gap_bottom = high[i]
            gap_size_pct = (gap_top - gap_bottom) / close[i]
            if gap_size_pct < cfg.fvg_min_size_pct:
                continue
            filled = 0.0
            for k in range(i + 1, len(df)):
                if high[k] >= gap_bottom:
                    entry = max(high[k], gap_bottom)
                    filled = 1.0 - max(0.0, (gap_top - entry) / (gap_top - gap_bottom))
                    break
            bear_fvgs.append(FairValueGap(
                direction="Bearish",
                top=float(gap_top),
                bottom=float(gap_bottom),
                candle_idx=i - 1,
                date=dates[i - 1],
                filled_pct=float(min(filled, 1.0)),
            ))

    # Keep only unfilled FVGs (< 90% filled) — most recent N
    unfilled_bull = [f for f in bull_fvgs if f.filled_pct < cfg.fvg_fill_threshold][-cfg.fvg_max_zones:]
    unfilled_bear = [f for f in bear_fvgs if f.filled_pct < cfg.fvg_fill_threshold][-cfg.fvg_max_zones:]

    logger.debug(f"FVGs: {len(unfilled_bull)} bullish, {len(unfilled_bear)} bearish (unfilled)")
    return unfilled_bull, unfilled_bear


# ─────────────────────────────────────────────────────────────────────────────
# Task 1C: Liquidity Pool Detector
# ─────────────────────────────────────────────────────────────────────────────

def detect_liquidity_pools(
    df: pd.DataFrame, cfg: SmcConfig
) -> tuple[list[LiquidityPool], list[LiquidityPool]]:
    """
    Detect Liquidity Pools (clusters of stops).

    Buy-side Liquidity = Equal Highs (retail buy stops sitting above)
    Sell-side Liquidity = Equal Lows (retail sell stops sitting below)

    Logic: Swing points within cfg.liq_tolerance% of each other = "equal"
    Multiple tests = higher liquidity concentration.
    """
    if len(df) < 20:
        return [], []

    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    dates = df["Date"].values if "Date" in df.columns else df.index.values

    swing_highs_mask = _find_swing_highs(high, left=3, right=3)
    swing_lows_mask = _find_swing_lows(low, left=3, right=3)

    swing_h_prices = high[swing_highs_mask].values
    swing_h_idx = np.where(swing_highs_mask.values)[0]
    swing_l_prices = low[swing_lows_mask].values
    swing_l_idx = np.where(swing_lows_mask.values)[0]

    tol = cfg.liq_tolerance_pct / 100.0

    buy_side: list[LiquidityPool] = []
    sell_side: list[LiquidityPool] = []

    # ── Buy-side (Equal Highs)
    processed = set()
    for i, (price_i, idx_i) in enumerate(zip(swing_h_prices, swing_h_idx)):
        if i in processed:
            continue
        cluster_prices = [price_i]
        cluster_idxs = [idx_i]
        for j, (price_j, idx_j) in enumerate(zip(swing_h_prices, swing_h_idx)):
            if j <= i or j in processed:
                continue
            if abs(price_j - price_i) / price_i <= tol:
                cluster_prices.append(price_j)
                cluster_idxs.append(idx_j)
                processed.add(j)
        if len(cluster_prices) >= 2:
            level = float(np.mean(cluster_prices))
            last_idx = max(cluster_idxs)
            # Check if swept (price has closed above this level since)
            swept = bool((close.values[last_idx + 1:] > level).any()) if last_idx + 1 < len(close) else False
            
            # Detect Sweep Event (Price was above level, but closed below)
            is_sweep = False
            curr_high = high.iloc[-1]
            curr_close = close.iloc[-1]
            if curr_high > level > curr_close:
                is_sweep = True

            buy_side.append(LiquidityPool(
                side="Buy-side",
                level=level,
                touches=len(cluster_prices),
                candle_idx=last_idx,
                date=dates[last_idx],
                swept=swept,
                is_sweep_event=is_sweep
            ))

    # ── Sell-side (Equal Lows)
    processed = set()
    for i, (price_i, idx_i) in enumerate(zip(swing_l_prices, swing_l_idx)):
        if i in processed:
            continue
        cluster_prices = [price_i]
        cluster_idxs = [idx_i]
        for j, (price_j, idx_j) in enumerate(zip(swing_l_prices, swing_l_idx)):
            if j <= i or j in processed:
                continue
            if abs(price_j - price_i) / price_i <= tol:
                cluster_prices.append(price_j)
                cluster_idxs.append(idx_j)
                processed.add(j)
        if len(cluster_prices) >= 2:
            level = float(np.mean(cluster_prices))
            last_idx = max(cluster_idxs)
            # Check if swept (closed below)
            swept = bool((close.values[last_idx + 1:] < level).any()) if last_idx + 1 < len(close) else False
            
            # ── TASK 1C-2: Detect Sweep Event (Price was below, now above)
            is_sweep = False
            if last_idx + 1 < len(close):
                curr_low = low.iloc[-1]
                curr_close = close.iloc[-1]
                if curr_low < level < curr_close:
                    is_sweep = True

            sell_side.append(LiquidityPool(
                side="Sell-side",
                level=level,
                touches=len(cluster_prices),
                candle_idx=last_idx,
                date=dates[last_idx],
                swept=swept,
                is_sweep_event=is_sweep
            ))

    # Keep most recent unswept pools
    buy_unswept = [p for p in buy_side if not p.swept][-cfg.liq_max_pools:]
    sell_unswept = [p for p in sell_side if not p.swept][-cfg.liq_max_pools:]

    logger.debug(f"Liquidity Pools: {len(buy_unswept)} buy-side, {len(sell_unswept)} sell-side")
    return buy_unswept, sell_unswept


# ─────────────────────────────────────────────────────────────────────────────
# Task 1D: Market Structure (BOS / ChoCH)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_market_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Break of Structure (BOS) and Change of Character (ChoCH).

    BOS = price breaks a prior swing high (bullish continuation) or
          price breaks a prior swing low (bearish continuation)
    ChoCH = price breaks structure AGAINST the previous trend direction
            → First signal of trend reversal

    Adds columns:
        smc_bos_bull   — bool: Bullish BOS (broke prior swing high)
        smc_bos_bear   — bool: Bearish BOS (broke prior swing low)
        smc_choch      — bool: Change of Character event
        smc_structure  — str: "Bullish", "Bearish", "Ranging"
    """
    result = df.copy()
    n = len(result)

    high = result["High"].values
    low = result["Low"].values
    close = result["Close"].values

    sw_highs = _find_swing_highs(result["High"], left=3, right=3)
    sw_lows = _find_swing_lows(result["Low"], left=3, right=3)

    bos_bull = np.zeros(n, dtype=bool)
    bos_bear = np.zeros(n, dtype=bool)
    choch = np.zeros(n, dtype=bool)

    last_sh = None  # last swing high price
    last_sl = None  # last swing low price
    prev_break_dir = None  # "bull" or "bear"

    for i in range(1, n):
        if sw_highs.iloc[i]:
            last_sh = high[i]
        if sw_lows.iloc[i]:
            last_sl = low[i]

        # ── BOS/ChoCH Logic ───────────────────
        if last_sh is not None and close[i] > last_sh:
            if prev_break_dir == "bear":
                choch[i] = True
            else:
                bos_bull[i] = True
            prev_break_dir = "bull"
            last_sh = None

        elif last_sl is not None and close[i] < last_sl:
            if prev_break_dir == "bull":
                choch[i] = True
            else:
                bos_bear[i] = True
            prev_break_dir = "bear"
            last_sl = None

    result["smc_bos_bull"] = bos_bull
    result["smc_bos_bear"] = bos_bear
    result["smc_choch"] = choch

    # ── Task 1D-2: Inducement (IDM) & Liquidity Sweep Detection ──
    idm_bull = np.zeros(n, dtype=bool)
    idm_bear = np.zeros(n, dtype=bool)
    sweep_bull = np.zeros(n, dtype=bool)
    sweep_bear = np.zeros(n, dtype=bool)
    
    # Track the last structural swing point
    # Bull trend -> we look for the last minor low to be taken (IDM)
    # Bear trend -> we look for the last minor high to be taken (IDM)
    last_minor_low_val = None
    last_minor_high_val = None
    last_sw_high = None
    last_sw_low = None
    
    for i in range(5, n):
        if sw_lows.iloc[i-1]:
            last_minor_low_val = low[i-1]
            last_sw_low = low[i-1]
        if sw_highs.iloc[i-1]:
            last_minor_high_val = high[i-1]
            last_sw_high = high[i-1]
            
        # ── IDM Logic (Inducement Taken)
        if prev_break_dir == "bull" and last_minor_low_val is not None:
            # Bullish: price must grab the inducement (last minor low) to confirm liquidity grab
            if low[i] < last_minor_low_val and close[i] > last_minor_low_val:
                # This is a SWEEP of the IDM (Very strong Bullish)
                idm_bull[i] = True
                sweep_bull[i] = True
                last_minor_low_val = None
            elif low[i] < last_minor_low_val:
                # Just taking IDM (standard)
                idm_bull[i] = True
                last_minor_low_val = None
                
        elif prev_break_dir == "bear" and last_minor_high_val is not None:
            # Bearish: price must grab the inducement (last minor high)
            if high[i] > last_minor_high_val and close[i] < last_minor_high_val:
                # This is a SWEEP of the IDM (Very strong Bearish)
                idm_bear[i] = True
                sweep_bear[i] = True
                last_minor_high_val = None
            elif high[i] > last_minor_high_val:
                idm_bear[i] = True
                last_minor_high_val = None

        # ── Generic Liquidity Sweep (Major Swing Sweep)
        # Price pierces a major swing but closes back inside
        if last_sw_high is not None and high[i] > last_sw_high and close[i] < last_sw_high:
            sweep_bear[i] = True # Distribution/Sweep High
        if last_sw_low is not None and low[i] < last_sw_low and close[i] > last_sw_low:
            sweep_bull[i] = True # Accumulation/Sweep Low

    result["smc_idm_bull"] = idm_bull
    result["smc_idm_bear"] = idm_bear
    result["smc_sweep_bull"] = sweep_bull
    result["smc_sweep_bear"] = sweep_bear

    # Structural bias: rolling 20-bar window
    bull_count = pd.Series(bos_bull.astype(int), index=result.index).rolling(20, min_periods=1).sum()
    bear_count = pd.Series(bos_bear.astype(int), index=result.index).rolling(20, min_periods=1).sum()
    structure = pd.Series("Ranging", index=result.index)
    structure[bull_count > bear_count] = "Bullish"
    structure[bear_count > bull_count] = "Bearish"
    result["smc_structure"] = structure

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SMC Composite Score per Bar
# ─────────────────────────────────────────────────────────────────────────────

def _compute_smc_scores(
    df: pd.DataFrame,
    bull_obs: list[OrderBlock],
    bear_obs: list[OrderBlock],
    bull_fvgs: list[FairValueGap],
    bear_fvgs: list[FairValueGap],
    buy_liq: list[LiquidityPool],
    sell_liq: list[LiquidityPool],
    cfg: SmcConfig,
) -> pd.Series:
    """
    Compute a scalar SMC score per bar (-1.0 to +1.0).

    Logic:
    - Price inside a Bullish OB zone → +0.3
    - Price inside a Bullish FVG → +0.2 (unfilled gap = draw)
    - Price above nearest Buy-side Liq pool → -0.1 (pool swept, continuation possible)
    - BOS Bullish → +0.2
    - ChoCH → ±0.15 (invert current bias)
    """
    close = df["Close"].values
    n = len(df)
    score = np.zeros(n)

    # Pre-extract levels for vectorized comparison
    for ob in bull_obs:
        mask = (close >= ob.bottom) & (close <= ob.top)
        score[mask] += 0.30 * ob.strength

    for ob in bear_obs:
        mask = (close >= ob.bottom) & (close <= ob.top)
        score[mask] -= 0.30 * ob.strength

    for fvg in bull_fvgs:
        unfill_weight = 1.0 - fvg.filled_pct
        mask = (close >= fvg.bottom) & (close <= fvg.top)
        score[mask] += 0.20 * unfill_weight

    for fvg in bear_fvgs:
        unfill_weight = 1.0 - fvg.filled_pct
        mask = (close >= fvg.bottom) & (close <= fvg.top)
        score[mask] -= 0.20 * unfill_weight

    # BOS contribution
    if "smc_bos_bull" in df.columns:
        score[df["smc_bos_bull"].values] += 0.20
        score[df["smc_bos_bear"].values] -= 0.20
        score[df["smc_choch"].values] *= -0.5  # Invert signal on ChoCH

    # Inducement (IDM) contribution
    if "smc_idm_bull" in df.columns:
        score[df["smc_idm_bull"].values] += 0.25
        score[df["smc_idm_bear"].values] -= 0.25
    
    # Sweep contribution
    if "smc_sweep_bull" in df.columns:
        score[df["smc_sweep_bull"].values] += 0.35
        score[df["smc_sweep_bear"].values] -= 0.35

    return pd.Series(score, index=df.index).rolling(5, min_periods=1).mean().clip(-1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Main: SmcAnalyzer Strategy
# ─────────────────────────────────────────────────────────────────────────────

class SmcAnalyzer:
    """
    Smart Money Concepts (SMC) Analyzer.

    Detects institutional footprints: Order Blocks, Fair Value Gaps,
    Liquidity Pools, and Market Structure events (BOS/ChoCH).

    Conforms to ``TradingStrategy`` Protocol.

    Output columns:
        smc_signal          — int (-1, 0, 1)
        smc_signal_reason   — str
        smc_score           — float -1.0 to +1.0 composite score
        smc_structure       — str "Bullish" / "Bearish" / "Ranging"
        smc_bos_bull        — bool
        smc_bos_bear        — bool
        smc_choch           — bool
        smc_in_bull_ob      — bool: price inside bullish OB
        smc_in_bear_ob      — bool: price inside bearish OB
        smc_in_bull_fvg     — bool: price inside bullish FVG (unmitigated)
        smc_in_bear_fvg     — bool: price inside bearish FVG
        smc_near_buy_liq    — float: distance to nearest buy-side liq pool (%)
        smc_near_sell_liq   — float: distance to nearest sell-side liq pool (%)
    """

    def __init__(self, config: SmcConfig | None = None, stoch_config: StochasticConfig | None = None) -> None:
        self.cfg = config or SmcConfig()
        self._stoch = StochasticOscillator(stoch_config)
        # Cached analysis results for UI display
        self._bull_obs: list[OrderBlock] = []
        self._bear_obs: list[OrderBlock] = []
        self._bull_fvgs: list[FairValueGap] = []
        self._bear_fvgs: list[FairValueGap] = []
        self._buy_liq: list[LiquidityPool] = []
        self._sell_liq: list[LiquidityPool] = []

    def generate_signals(self, data: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        """Run full SMC analysis and generate signals."""
        required = {"High", "Low", "Close", "Open"}
        if not required.issubset(data.columns):
            logger.warning("SmcAnalyzer: missing OHLC columns.")
            data = data.copy()
            data["smc_signal"] = 0
            data["smc_signal_reason"] = "MISSING_DATA"
            return data

        if len(data) < 30:
            data = data.copy()
            data["smc_signal"] = 0
            data["smc_signal_reason"] = "INSUFFICIENT_DATA"
            return data

        df = data.copy()

        # ── Detect all SMC zones ─────────────────────────────────────
        bull_obs, bear_obs = detect_order_blocks(df, self.cfg)
        bull_fvgs, bear_fvgs = detect_fair_value_gaps(df, self.cfg)
        buy_liq, sell_liq = detect_liquidity_pools(df, self.cfg)
        df = _detect_market_structure(df)

        # Cache for UI display
        self._bull_obs = bull_obs
        self._bear_obs = bear_obs
        self._bull_fvgs = bull_fvgs
        self._bear_fvgs = bear_fvgs
        self._buy_liq = buy_liq
        self._sell_liq = sell_liq

        # ── Per-bar zone membership ──────────────────────────────────
        close = df["Close"].values
        n = len(df)

        # OB membership
        in_bull_ob = np.zeros(n, dtype=bool)
        in_bear_ob = np.zeros(n, dtype=bool)
        for ob in bull_obs:
            in_bull_ob |= (close >= ob.bottom) & (close <= ob.top)
        for ob in bear_obs:
            in_bear_ob |= (close >= ob.bottom) & (close <= ob.top)
        df["smc_in_bull_ob"] = in_bull_ob
        df["smc_in_bear_ob"] = in_bear_ob

        # FVG membership
        in_bull_fvg = np.zeros(n, dtype=bool)
        in_bear_fvg = np.zeros(n, dtype=bool)
        for fvg in bull_fvgs:
            in_bull_fvg |= (close >= fvg.bottom) & (close <= fvg.top)
        for fvg in bear_fvgs:
            in_bear_fvg |= (close >= fvg.bottom) & (close <= fvg.top)
        df["smc_in_bull_fvg"] = in_bull_fvg
        df["smc_in_bear_fvg"] = in_bear_fvg

        # Liquidity proximity (% distance from current close)
        curr_close = close[-1] if n > 0 else 1.0
        
        near_buy_lvl = 0.0
        if buy_liq:
            closest_buy = min(buy_liq, key=lambda p: abs(p.level - curr_close))
            near_buy_lvl = closest_buy.level
            df["smc_near_buy_liq"] = abs(near_buy_lvl - curr_close) / curr_close
        else:
            df["smc_near_buy_liq"] = 1.0
            
        near_sell_lvl = 0.0
        if sell_liq:
            closest_sell = min(sell_liq, key=lambda p: abs(p.level - curr_close))
            near_sell_lvl = closest_sell.level
            df["smc_near_sell_liq"] = abs(near_sell_lvl - curr_close) / curr_close
        else:
            df["smc_near_sell_liq"] = 1.0
            
        # Store metadata for get_current_state
        self._near_buy_lvl = near_buy_lvl
        self._near_sell_lvl = near_sell_lvl

        # ── Stochastic Momentum Layer ─────────────────────────────
        df = self._stoch.generate_signals(df)

        # ── Composite Score ──────────────────────────────────────────
        df["smc_score"] = _compute_smc_scores(
            df, bull_obs, bear_obs, bull_fvgs, bear_fvgs, buy_liq, sell_liq, self.cfg
        )

        # ── Signal Generation ────────────────────────────────────────
        df["smc_signal"] = 0
        df["smc_signal_reason"] = ""
        df["smc_entry_signal"] = 0
        df["smc_entry_reason"] = ""
        df["smc_entry_zone_top"] = np.nan
        df["smc_entry_zone_bottom"] = np.nan
        df["smc_entry_type"] = ""
        df["smc_entry_quality"] = 0
        df["smc_entry_grade"] = ""
        df["smc_entry_factors"] = ""
        df["smc_entry_retest_no"] = 0
        df["smc_entry_tactical_signal"] = 0

        score = df["smc_score"]
        struct = df["smc_structure"]

        # ── Stochastic Filter Masks ───────────────────────────────
        stoch_not_ob = df["stoch_status"] != "OVERBOUGHT"  # Veto buy if overbought
        stoch_not_os = df["stoch_status"] != "OVERSOLD"    # Veto sell if oversold

        pos = np.arange(n)
        open_arr = df["Open"].to_numpy(dtype=float)
        high_arr = df["High"].to_numpy(dtype=float)
        low_arr = df["Low"].to_numpy(dtype=float)
        close_arr = df["Close"].to_numpy(dtype=float)
        bullish_close = close_arr >= open_arr
        candle_range = np.maximum(high_arr - low_arr, 1e-9)
        close_pos = (close_arr - low_arr) / candle_range
        stoch_k_arr = df["stoch_k"].to_numpy(dtype=float) if "stoch_k" in df.columns else np.full(n, 50.0)
        vol_arr = df["Volume"].to_numpy(dtype=float) if "Volume" in df.columns else np.ones(n)
        vol_ma = pd.Series(vol_arr).rolling(20, min_periods=1).mean().to_numpy()
        ret_10 = pd.Series(close_arr).pct_change(10).fillna(0.0).to_numpy()
        stoch_ok_arr = stoch_not_ob.to_numpy(dtype=bool)

        def _mark_entry(mask: np.ndarray, reason: str, zone_top: float, zone_bottom: float, zone_type: str) -> None:
            if not bool(mask.any()):
                return
            unused = df["smc_entry_signal"].to_numpy(dtype=int) == 0
            raw_pos = np.flatnonzero(mask & unused)
            if len(raw_pos) == 0:
                return
            selected_pos = []
            last_pos = -999
            for p in raw_pos:
                if p - last_pos >= 2:
                    selected_pos.append(p)
                    last_pos = p
            idx = df.index[selected_pos]
            df.loc[idx, "smc_entry_signal"] = 1
            df.loc[idx, "smc_entry_reason"] = reason
            df.loc[idx, "smc_entry_zone_top"] = float(zone_top)
            df.loc[idx, "smc_entry_zone_bottom"] = float(zone_bottom)
            df.loc[idx, "smc_entry_type"] = zone_type
            zone_mid = (float(zone_top) + float(zone_bottom)) / 2.0
            zone_width_pct = (float(zone_top) - float(zone_bottom)) / np.maximum(close_arr[selected_pos], 1e-9)
            distance_above_zone = np.maximum(close_arr[selected_pos] - float(zone_top), 0.0) / np.maximum(close_arr[selected_pos], 1e-9)
            quality: list[int] = []
            grades: list[str] = []
            factors_list: list[str] = []
            retest_nos: list[int] = []
            tactical_flags: list[int] = []
            for local_i, p in enumerate(selected_pos):
                q = 0
                factors = []
                retest_no = local_i + 1
                retest_nos.append(retest_no)
                if retest_no == 1:
                    q += 2
                    factors.append("FIRST_RETEST")
                elif retest_no == 2:
                    q += 1
                    factors.append("SECOND_RETEST")
                else:
                    q -= 1
                    factors.append("LATE_RETEST")
                if close_arr[p] >= float(zone_top):
                    q += 2
                    factors.append("RECLAIM_TOP")
                elif close_arr[p] >= zone_mid:
                    q += 1
                    factors.append("MID_RECLAIM")
                if bullish_close[p]:
                    q += 2
                    factors.append("BULL_CLOSE")
                if close_pos[p] >= 0.55:
                    q += 1
                    factors.append("STRONG_CLOSE")
                if (np.minimum(open_arr[p], close_arr[p]) - low_arr[p]) >= (0.30 * candle_range[p]):
                    q += 1
                    factors.append("WICK_REJECT")
                if stoch_k_arr[p] <= 40:
                    q += 2
                    factors.append("STOCH_LOW")
                elif stoch_k_arr[p] <= 70:
                    q += 1
                    factors.append("STOCH_OK")
                if vol_arr[p] >= 0.80 * max(vol_ma[p], 1.0):
                    q += 1
                    factors.append("VOL_OK")
                if zone_width_pct[local_i] >= 0.005:
                    q += 1
                    factors.append("WIDE_FVG")
                if distance_above_zone[local_i] <= 0.03:
                    q += 1
                    factors.append("NEAR_ZONE")
                elif distance_above_zone[local_i] > 0.08:
                    q -= 2
                    factors.append("CHASE")
                if stoch_k_arr[p] >= 75:
                    q -= 1
                    factors.append("STOCH_HIGH")
                if ret_10[p] >= 0.30:
                    q -= 2
                    factors.append("HOT_RUN")
                elif ret_10[p] >= 0.20:
                    q -= 1
                    factors.append("RUN_EXTENDED")
                q = int(np.clip(q, 0, 10))
                grade = "BEST" if q >= 7 else "GOOD" if q >= 5 else "RAW"
                tactical_ok = (
                    q >= 7
                    and retest_no <= 2
                    and distance_above_zone[local_i] <= 0.04
                    and stoch_k_arr[p] <= 72
                    and ret_10[p] < 0.25
                )
                quality.append(q)
                grades.append(grade)
                factors_list.append(",".join(factors))
                tactical_flags.append(1 if tactical_ok else 0)
            df.loc[idx, "smc_entry_quality"] = quality
            df.loc[idx, "smc_entry_grade"] = grades
            df.loc[idx, "smc_entry_factors"] = factors_list
            df.loc[idx, "smc_entry_retest_no"] = retest_nos
            df.loc[idx, "smc_entry_tactical_signal"] = tactical_flags

        for i in range(2, n):
            if low_arr[i] <= high_arr[i - 2]:
                continue
            zone_top = float(low_arr[i])
            zone_bottom = float(high_arr[i - 2])
            gap_size_pct = (zone_top - zone_bottom) / max(float(close_arr[i]), 1e-9)
            if gap_size_pct < self.cfg.fvg_min_size_pct:
                continue
            after_zone_confirmed = pos > i
            wick_touch = (low_arr <= zone_top) & (high_arr >= zone_bottom)
            reclaim = close_arr >= zone_bottom
            reclaim_top = close_arr >= zone_top
            lower_wick_reject = (np.minimum(open_arr, close_arr) - low_arr) >= (0.30 * candle_range)
            loose_confirm = bullish_close | reclaim_top | lower_wick_reject
            fvg_retest = after_zone_confirmed & wick_touch & reclaim & loose_confirm & stoch_ok_arr
            _mark_entry(
                fvg_retest,
                "BULL_FVG_RETEST: Wick retest/reclaim of bullish FVG",
                zone_top,
                zone_bottom,
                "FVG",
            )

        for ob in bull_obs:
            after_zone_confirmed = pos > int(ob.candle_idx)
            wick_touch = (low_arr <= float(ob.top)) & (high_arr >= float(ob.bottom))
            reclaim = close_arr >= float(ob.bottom)
            reclaim_top = close_arr >= float(ob.top)
            lower_wick_reject = (np.minimum(open_arr, close_arr) - low_arr) >= (0.30 * candle_range)
            loose_confirm = bullish_close | reclaim_top | lower_wick_reject
            ob_retest = after_zone_confirmed & wick_touch & reclaim & loose_confirm & stoch_ok_arr
            _mark_entry(
                ob_retest,
                "BULL_OB_RETEST: Wick retest/reclaim of bullish order block",
                ob.top,
                ob.bottom,
                "OB",
            )

        # Strong bullish: in bull OB + bullish structure + positive score
        # + Stochastic NOT overbought (momentum filter)
        buy_mask = (
            df["smc_in_bull_ob"]
            & (struct == "Bullish")
            & (score >= self.cfg.signal_score_threshold)
            & stoch_not_ob
        )
        df.loc[buy_mask, "smc_signal"] = 1
        df.loc[buy_mask, "smc_signal_reason"] = "BULL_OB: Price in Bullish Order Block with Bullish Structure (Stoch ✅)"

        # BOS bullish confirmation + Stochastic filter
        bos_buy = (
            df["smc_bos_bull"]
            & (score >= 0)
            & (df["smc_signal"] == 0)
            & stoch_not_ob
        )
        df.loc[bos_buy, "smc_signal"] = 1
        df.loc[bos_buy, "smc_signal_reason"] = "BOS_BULL: Break of Structure confirmed — continuation (Stoch ✅)"

        # FVG bullish fill + Stochastic filter
        fvg_buy = (
            (df["smc_entry_signal"] == 1)
            & (struct != "Bearish")
            & (df["smc_signal"] == 0)
        )
        df.loc[fvg_buy, "smc_signal"] = 1
        df.loc[fvg_buy, "smc_signal_reason"] = "BULL_FVG: Price drawn to unfilled Bullish FVG (Stoch ✅)"

        df.loc[fvg_buy, "smc_signal_reason"] = df.loc[fvg_buy, "smc_entry_reason"]

        # Bear signals + Stochastic filter (veto sell when oversold)
        sell_mask = (
            df["smc_in_bear_ob"]
            & (struct == "Bearish")
            & (score <= -self.cfg.signal_score_threshold)
            & stoch_not_os
        )
        df.loc[sell_mask, "smc_signal"] = -1
        df.loc[sell_mask, "smc_signal_reason"] = "BEAR_OB: Price in Bearish Order Block — supply zone (Stoch ✅)"

        bos_sell = (
            df["smc_bos_bear"]
            & (score <= 0)
            & (df["smc_signal"] == 0)
            & stoch_not_os
        )
        df.loc[bos_sell, "smc_signal"] = -1
        df.loc[bos_sell, "smc_signal_reason"] = "BOS_BEAR: Bearish Break of Structure — continuation down (Stoch ✅)"

        # ChoCH: override to 0, wait for new setup
        df.loc[df["smc_choch"], "smc_signal"] = 0
        df.loc[df["smc_choch"], "smc_signal_reason"] = "CHOCH: Change of Character — awaiting new structure"

        # ── TASK 2-B: Inducement (IDM) Reversal Signal
        if "smc_idm_bull" in df.columns:
            idm_buy = df["smc_idm_bull"] & (df["smc_signal"] == 0)
            df.loc[idm_buy, "smc_signal"] = 1
            df.loc[idm_buy, "smc_signal_reason"] = "IDM_BULL: Inducement taken (Liquidity Grab) — Bullish confirmation"
            
            idm_sell = df["smc_idm_bear"] & (df["smc_signal"] == 0)
            df.loc[idm_sell, "smc_signal"] = -1
            df.loc[idm_sell, "smc_signal_reason"] = "IDM_BEAR: Inducement taken (Liquidity Grab) — Bearish confirmation"

        # ── TASK 2-C: Liquidity Sweep (Live only for the last bar)
        if buy_liq and any(p.is_sweep_event for p in buy_liq):
             df.loc[df.index[-1], "smc_signal"] = -1
             df.loc[df.index[-1], "smc_signal_reason"] = "LIQ_SWEEP_BEAR: Highs swept — institutional sell-side pressure"
             
        if sell_liq and any(p.is_sweep_event for p in sell_liq):
             df.loc[df.index[-1], "smc_signal"] = 1
             df.loc[df.index[-1], "smc_signal_reason"] = "LIQ_SWEEP_BULL: Lows swept — institutional buy-side pressure"

        n_buy = int((df["smc_signal"] == 1).sum())
        n_sell = int((df["smc_signal"] == -1).sum())
        n_choch = int(df["smc_choch"].sum())
        logger.info(
            f"SmcAnalyzer: {n_buy} BUY / {n_sell} SELL | "
            f"{len(bull_obs)} Bull OBs | {len(bear_obs)} Bear OBs | "
            f"{len(bull_fvgs)} Bull FVGs | {len(bear_fvgs)} Bear FVGs | "
            f"{n_choch} ChoCH events"
        )
        return df

    def get_current_state(self, data: pd.DataFrame) -> dict:
        """Return a snapshot dict for UI display."""
        if data.empty or len(data) < 30:
            return {
                "smc_score": 0.0, "signal": 0, "structure": "UNKNOWN",
                "bull_obs": [], "bear_obs": [], "bull_fvgs": [], "bear_fvgs": [],
                "buy_liq": [], "sell_liq": [], "current_price": 0.0,
                "stoch": {"k": 50.0, "d": 50.0, "status": "NEUTRAL"},
                "entry_confirmation": [],
            }

        df = self.generate_signals(data)
        last = df.iloc[-1]
        close = last["Close"]

        # ── Stochastic state ───────────────────────────────────────
        stoch_state = self._stoch.get_current_state(data)

        # ── Entry Confirmation Table ───────────────────────────────
        zones = []
        for fvg in self._bull_fvgs:
            zones.append({"zone_top": fvg.top, "zone_bottom": fvg.bottom,
                          "type": "FVG", "direction": "Bullish",
                          "filled_pct": fvg.filled_pct})
        for fvg in self._bear_fvgs:
            zones.append({"zone_top": fvg.top, "zone_bottom": fvg.bottom,
                          "type": "FVG", "direction": "Bearish",
                          "filled_pct": fvg.filled_pct})
        for ob in self._bull_obs:
            zones.append({"zone_top": ob.top, "zone_bottom": ob.bottom,
                          "type": "OB", "direction": "Bullish",
                          "strength": ob.strength})
        for ob in self._bear_obs:
            zones.append({"zone_top": ob.top, "zone_bottom": ob.bottom,
                          "type": "OB", "direction": "Bearish",
                          "strength": ob.strength})

        entry_confirmation = self._stoch.evaluate_entry(data, zones)

        return {
            "smc_score": round(float(last.get("smc_score", 0.0)), 3),
            "signal": int(last.get("smc_signal", 0)),
            "signal_reason": str(last.get("smc_signal_reason", "")),
            "structure": str(last.get("smc_structure", "Ranging")),
            "bos_bull": bool(last.get("smc_bos_bull", False)),
            "bos_bear": bool(last.get("smc_bos_bear", False)),
            "choch": bool(last.get("smc_choch", False)),
            "in_bull_ob": bool(last.get("smc_in_bull_ob", False)),
            "in_bear_ob": bool(last.get("smc_in_bear_ob", False)),
            "in_bull_fvg": bool(last.get("smc_in_bull_fvg", False)),
            "in_bear_fvg": bool(last.get("smc_in_bear_fvg", False)),
            "near_buy_liq_pct": round(float(last.get("smc_near_buy_liq", 1.0)) * 100, 2),
            "near_sell_liq_pct": round(float(last.get("smc_near_sell_liq", 1.0)) * 100, 2),
            "near_buy_liq_price": round(getattr(self, "_near_buy_lvl", 0.0), 2),
            "near_sell_liq_price": round(getattr(self, "_near_sell_lvl", 0.0), 2),
            # Zone lists for chart overlay
            "bull_obs": [{"top": ob.top, "bottom": ob.bottom, "strength": ob.strength} for ob in self._bull_obs],
            "bear_obs": [{"top": ob.top, "bottom": ob.bottom, "strength": ob.strength} for ob in self._bear_obs],
            "bull_fvgs": [{"top": f.top, "bottom": f.bottom, "filled_pct": f.filled_pct} for f in self._bull_fvgs],
            "bear_fvgs": [{"top": f.top, "bottom": f.bottom, "filled_pct": f.filled_pct} for f in self._bear_fvgs],
            "buy_liq": [{"level": p.level, "touches": p.touches} for p in self._buy_liq],
            "sell_liq": [{"level": p.level, "touches": p.touches} for p in self._sell_liq],
            "current_price": round(float(close), 4),
            # Stochastic data (separated panel)
            "stoch": stoch_state,
            # Institutional Events
            "idm_bull": bool(last.get("smc_idm_bull", False)),
            "idm_bear": bool(last.get("smc_idm_bear", False)),
            "sweep_bull": bool(last.get("smc_sweep_bull", False)),
            "sweep_bear": bool(last.get("smc_sweep_bear", False)),
            # Entry Confirmation Table data (separated panel)
            "entry_confirmation": entry_confirmation,
        }
