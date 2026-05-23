"""
Wyckoff Analyzer — Triple Filter Strategy
==========================================
Implements the full Wyckoff Methodology as a quantitative signal engine.

Architecture (Triple Filter):
    Layer 1 (Context):   Trading Range detection + Phase assignment
                         (Accumulation A/B/C/D/E, Distribution, Markup, Markdown)
    Layer 2 (Effort):    VSA Engine — No Supply, Stopping Volume, Shakeout, SOS/SOW
                         + Effort vs. Result ratio (Weis Wave inspired)
    Layer 3 (Trigger):   Spring event, LPS (Last Point of Support), UTAD detector

Integration:
    - Feeds `wyckoff_signal` (-1/0/1) into the Signal Combiner
    - Feeds scalar scores into AIPredictor._prepare_features()
    - Displays Wyckoff labels on ai_forecast.py Streamlit view

Conforms to ``TradingStrategy`` Protocol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.config import WyckoffConfig


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes (output structures)
# ─────────────────────────────────────────────────────────────────────────────

class TradingRange:
    """Represents a detected Wyckoff Trading Range (consolidation zone)."""

    def __init__(
        self,
        upper: float,
        lower: float,
        start_idx: int,
        end_idx: int,
        width_days: int,
    ) -> None:
        self.upper = upper          # Creek / Resistance level
        self.lower = lower          # Ice / Support level
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.width_days = width_days
        self.midpoint = (upper + lower) / 2
        self.range_pct = (upper - lower) / lower if lower > 0 else 0.0

    def position_score(self, price: float) -> float:
        """Return 0.0 (at lower) to 1.0 (at upper) position within range."""
        if self.upper == self.lower:
            return 0.5
        return (price - self.lower) / (self.upper - self.lower)

    def projected_target(self, multiplier: float = 1.0) -> float:
        """Point & Figure inspired target: upper + range_width * multiplier."""
        return self.upper + (self.upper - self.lower) * multiplier

    def __repr__(self) -> str:
        return (
            f"TradingRange(upper={self.upper:.2f}, lower={self.lower:.2f}, "
            f"width={self.width_days}d, range={self.range_pct:.1%})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: ATR
# ─────────────────────────────────────────────────────────────────────────────

def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range vectorized."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: Trading Range Detector
# ─────────────────────────────────────────────────────────────────────────────

def _detect_trading_range(
    df: pd.DataFrame, cfg: WyckoffConfig, atr: pd.Series
) -> TradingRange | None:
    """
    Detect the most recent Trading Range (TR) in the price series.

    A TR is identified when:
    - Price oscillates between a stable upper (Creek) and lower (Ice) boundary
    - ATR is contracting (< cfg.tr_atr_ratio * long-term ATR)
    - Price touches both boundaries at least cfg.tr_min_touches times
    """
    if len(df) < cfg.tr_lookback + 20:
        return None

    window = df.tail(cfg.tr_lookback).copy()
    window_atr = atr.tail(cfg.tr_lookback)
    long_term_atr = atr.tail(cfg.tr_lookback * 3).mean()

    # ATR contraction check — TR requires low volatility regime
    current_atr_avg = window_atr.mean()
    if long_term_atr > 0 and current_atr_avg > (cfg.tr_atr_ratio * long_term_atr):
        return None  # Volatility too high — not a TR

    # Define upper (Creek) and lower (Ice) using rolling percentiles
    upper = window["High"].quantile(0.85)
    lower = window["Low"].quantile(0.15)

    # Verify minimum touches on each boundary (±0.5 * current ATR tolerance)
    tolerance = current_atr_avg * 0.5
    upper_touches = ((window["High"] >= upper - tolerance) & (window["High"] <= upper + tolerance)).sum()
    lower_touches = ((window["Low"] <= lower + tolerance) & (window["Low"] >= lower - tolerance)).sum()

    if upper_touches < cfg.tr_min_touches or lower_touches < cfg.tr_min_touches:
        return None  # Not enough boundary tests

    start_idx = len(df) - cfg.tr_lookback
    end_idx = len(df) - 1

    tr = TradingRange(
        upper=float(upper),
        lower=float(lower),
        start_idx=start_idx,
        end_idx=end_idx,
        width_days=cfg.tr_lookback,
    )
    logger.debug(f"Trading Range detected: {tr}")
    return tr


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: Wyckoff Phase Classifier
# ─────────────────────────────────────────────────────────────────────────────

def _classify_wyckoff_phase(
    df: pd.DataFrame,
    tr: TradingRange | None,
    ma_200: pd.Series,
    cfg: WyckoffConfig,
) -> pd.Series:
    """
    Assign a Wyckoff phase label to each row.

    Phase mapping:
        0 = Unknown / Markup-Trend (above TR, above MA200 → Markup)
        1 = Accumulation Phase A (initial stopping of downtrend)
        2 = Accumulation Phase B (ranging, building cause)
        3 = Accumulation Phase C (Spring — shakeout below support)
        4 = Accumulation Phase D (SOS — markup within TR)
        5 = Accumulation Phase E (markup escape)
        6 = Distribution Phase (ranging at highs)
        7 = Markdown (downtrend)
    """
    phases = pd.Series(0, index=df.index, dtype=int, name="wyckoff_phase")
    phase_labels = pd.Series("UNKNOWN", index=df.index, dtype=str, name="wyckoff_phase_label")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # MA200 trend context
    above_ma200 = close > ma_200.fillna(close)
    below_ma200 = ~above_ma200

    if tr is None:
        # No TR — classify by trend alone
        phases[above_ma200] = 0
        phase_labels[above_ma200] = "📈 MARKUP"
        phases[below_ma200] = 7
        phase_labels[below_ma200] = "📉 MARKDOWN"
        return phase_labels

    # Within TR window — classify by price position
    tr_window = df.index[tr.start_idx:tr.end_idx + 1]
    if len(tr_window) == 0:
        return phase_labels

    tr_mask = df.index.isin(tr_window)
    pos = close.apply(tr.position_score)

    # Phase B — normal ranging within TR (middle zone)
    phase_b = tr_mask & (pos >= 0.2) & (pos <= 0.80)
    phases[phase_b] = 2
    phase_labels[phase_b] = "📦 PHASE B (Accumulation)"

    # Phase D — SOS: upper 80–100% of TR on strong move
    phase_d = tr_mask & (pos > 0.80)
    phases[phase_d] = 4
    phase_labels[phase_d] = "🚀 PHASE D (SOS Markup)"

    # Phase C — Spring zone: below TR lower boundary
    spring_zone = low < (tr.lower * (1 - cfg.spring_penetration))
    phase_c = tr_mask & spring_zone & below_ma200
    phases[phase_c] = 3
    phase_labels[phase_c] = "🌀 PHASE C (Spring Zone)"

    # Phase E — escape above TR
    phase_e = (close > tr.upper) & below_ma200.shift(10, fill_value=True)
    phases[phase_e] = 5
    phase_labels[phase_e] = "⚡ PHASE E (Markup Escape)"

    # Distribution — ranging at highs (same TR logic but above MA200)
    dist_mask = tr_mask & above_ma200
    phases[dist_mask] = 6
    phase_labels[dist_mask] = "📉 DISTRIBUTION"

    return phase_labels


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: VSA Engine (Volume Spread Analysis)
# ─────────────────────────────────────────────────────────────────────────────

def _run_vsa_engine(df: pd.DataFrame, cfg: WyckoffConfig) -> pd.DataFrame:
    """
    Compute VSA labels for each bar.

    Output columns added:
        vsa_label           — Primary VSA event label (string)
        vsa_stopping_vol    — bool: Stopping Volume (climax bottom)
        vsa_no_supply       — bool: No Supply (low vol pullback = weak sellers)
        vsa_shakeout        — bool: Shakeout event (sharp drop + recovery)
        vsa_sos             — bool: Sign of Strength candle
        vsa_sow             — bool: Sign of Weakness candle
        effort_vs_result    — float: Effort (volume) vs Result (spread) ratio
        weis_wave_ratio     — float: Cumulative up-wave vol / down-wave vol
    """
    result = df.copy()
    close = result["Close"]
    high = result["High"]
    low = result["Low"]
    volume = result["Volume"]

    spread = (high - low).clip(lower=1e-6)
    vol_ma = volume.rolling(cfg.evr_lookback, min_periods=5).mean()
    vol_ratio = volume / vol_ma.replace(0, np.nan)
    spread_pct_rank = spread.rolling(cfg.evr_lookback).rank(pct=True) * 100

    is_up_bar = close > close.shift(1)
    is_down_bar = ~is_up_bar

    # ── Stopping Volume ────────────────────────────────────────────────────
    # High volume + narrow spread at range lows → smart money absorbing supply
    stopping_vol = (
        (vol_ratio >= cfg.vsa_stopping_vol_multiplier)
        & (spread_pct_rank <= cfg.vsa_stopping_spread_percentile)
        & is_down_bar
        & (close < close.rolling(20).mean())
    )
    result["vsa_stopping_vol"] = stopping_vol

    # ── No Supply ──────────────────────────────────────────────────────────
    # Very low volume on down bar → sellers exhausted (bullish)
    no_supply = (
        (vol_ratio <= cfg.vsa_no_supply_vol_ratio)
        & is_down_bar
        & (spread_pct_rank <= 50)
    )
    result["vsa_no_supply"] = no_supply

    # ── Shakeout (Spring validation) ───────────────────────────────────────
    # Sharp 1-2 bar drop followed by recovery with decreasing volume
    pct_change = close.pct_change()
    next_pct = close.pct_change().shift(-1)
    shakeout = (
        (pct_change <= cfg.vsa_shakeout_drop_pct)
        & (next_pct >= cfg.vsa_shakeout_recovery_pct)
        & (vol_ratio < 1.5)   # Not a panic selloff — controlled drop
    )
    result["vsa_shakeout"] = shakeout

    # ── Sign of Strength (SOS) ─────────────────────────────────────────────
    # Wide up-bar + high volume → demand overcoming supply
    sos = (
        (vol_ratio >= cfg.sos_volume_multiplier)
        & (spread_pct_rank >= cfg.sos_spread_percentile)
        & is_up_bar
        & (close > close.rolling(5).mean())
    )
    result["vsa_sos"] = sos

    # ── Sign of Weakness (SOW) ─────────────────────────────────────────────
    # Wide down-bar + high volume → supply overwhelming demand
    sow = (
        (vol_ratio >= cfg.sos_volume_multiplier)
        & (spread_pct_rank >= cfg.sos_spread_percentile)
        & is_down_bar
        & (close < close.rolling(5).mean())
    )
    result["vsa_sow"] = sow

    # ── Effort vs. Result ──────────────────────────────────────────────────
    # Low ratio = lots of effort (volume) with little result (spread) → absorption
    # High ratio = little effort with big result → ease of movement
    vol_norm = vol_ratio.fillna(1.0)
    spread_norm = (spread / spread.rolling(cfg.evr_lookback).mean().replace(0, 1.0)).fillna(1.0)
    result["effort_vs_result"] = (spread_norm / vol_norm).clip(0, 5)

    # ── Weis Wave Ratio ────────────────────────────────────────────────────
    # Compare cumulative volume on up-waves vs down-waves (rolling N bars)
    up_vol = np.where(is_up_bar, volume, 0.0)
    down_vol = np.where(is_down_bar, volume, 0.0)
    rolling_up = pd.Series(up_vol, index=result.index).rolling(cfg.evr_lookback, min_periods=5).sum()
    rolling_down = pd.Series(down_vol, index=result.index).rolling(cfg.evr_lookback, min_periods=5).sum()
    result["weis_wave_ratio"] = (rolling_up / rolling_down.replace(0, np.nan)).clip(0.1, 10.0)

    # ── Composite VSA Label ────────────────────────────────────────────────
    vsa_label = pd.Series("⚪ Normal", index=result.index, dtype=str)
    vsa_label[stopping_vol] = "🛑 Stopping Volume"
    vsa_label[no_supply] = "🌬️ No Supply"
    vsa_label[shakeout] = "⚡ Shakeout"
    vsa_label[sos] = "💚 Sign of Strength (SOS)"
    vsa_label[sow] = "🔴 Sign of Weakness (SOW)"
    result["vsa_label"] = vsa_label

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3: Spring Detector
# ─────────────────────────────────────────────────────────────────────────────

def _detect_spring(
    df: pd.DataFrame,
    tr: TradingRange | None,
    cfg: WyckoffConfig,
    atr: pd.Series,
) -> pd.Series:
    """
    Detect Spring events: false breakdowns below TR support with recovery.

    Rules:
    1. Low penetrates TR lower boundary by cfg.spring_penetration to cfg.spring_max_penetration
    2. Volume on the break is below average (smart money is NOT panicking)
    3. Close recovers back inside TR within cfg.spring_recovery_bars bars
    4. The recovered close is above TR lower (confirmed test)
    """
    spring = pd.Series(False, index=df.index, dtype=bool, name="spring_event")

    if tr is None:
        return spring

    vol_ma = df["Volume"].rolling(20, min_periods=5).mean()
    vol_ratio = df["Volume"] / vol_ma.replace(0, np.nan)

    tr_lower = tr.lower

    for i in range(len(df)):
        low_i = df["Low"].iloc[i]
        close_i = df["Close"].iloc[i]
        vol_i = vol_ratio.iloc[i] if not pd.isna(vol_ratio.iloc[i]) else 1.0

        # 1. Penetrates below TR lower
        pct_below = (tr_lower - low_i) / tr_lower if tr_lower > 0 else 0
        if not (cfg.spring_penetration <= pct_below <= cfg.spring_max_penetration):
            continue

        # 2. Low volume on the break (smart money not panicking)
        if vol_i >= cfg.spring_volume_ratio:
            continue

        # 3. Closes back inside TR (or at least above low)
        if close_i < tr_lower * 0.995:   # Close must be near or above Ice
            continue

        # 4. Verify recovery within N bars
        recovery_end = min(i + cfg.spring_recovery_bars, len(df) - 1)
        future_close = df["Close"].iloc[i + 1: recovery_end + 1]
        if future_close.empty or future_close.max() < tr_lower:
            continue

        spring.iloc[i] = True

    return spring


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3: LPS Detector (Last Point of Support)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_lps(
    df: pd.DataFrame,
    tr: TradingRange | None,
    cfg: WyckoffConfig,
    atr: pd.Series,
) -> pd.Series:
    """
    Detect LPS (Last Point of Support) events.

    LPS = A shallow, low-volume pullback after a SOS move.
    This is the optimal re-entry point in Phase D.

    Rules:
    1. Previous bar had a SOS (strong up-move)
    2. Current bar pulls back LESS than 1x ATR (shallow)
    3. Volume is below average (weak sellers)
    4. Price is still above TR midpoint
    """
    lps = pd.Series(False, index=df.index, dtype=bool, name="lps_event")

    if tr is None:
        return lps

    vol_ma = df["Volume"].rolling(20, min_periods=5).mean()
    vol_ratio = df["Volume"] / vol_ma.replace(0, np.nan)
    price_change = df["Close"].pct_change()
    is_pullback = price_change < 0

    for i in range(2, len(df)):
        close_i = df["Close"].iloc[i]
        atr_i = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0
        vol_ratio_i = vol_ratio.iloc[i] if not pd.isna(vol_ratio.iloc[i]) else 1.0

        # Must be above TR midpoint (Phase D territory)
        if close_i < tr.midpoint:
            continue

        # Must be a pullback bar
        if not is_pullback.iloc[i]:
            continue

        # Pullback depth must be shallow: < 1x ATR
        pullback_depth = df["Close"].iloc[i - 1] - close_i
        if atr_i > 0 and pullback_depth > (cfg.lps_pullback_atr_mult * atr_i):
            continue

        # Volume must be below average (weak sellers)
        if vol_ratio_i >= cfg.lps_volume_ratio:
            continue

        # Previous 2 bars should have had up-momentum (SOS context)
        prior_returns = price_change.iloc[max(0, i - 3):i]
        if prior_returns.sum() <= 0:
            continue

        lps.iloc[i] = True

    return lps


# ─────────────────────────────────────────────────────────────────────────────
# R:R Calculator
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_rr(
    entry: float,
    stop: float,
    tr: TradingRange | None,
    target: float | None = None,
) -> dict[str, float]:
    """
    Calculate Risk/Reward ratio based on Wyckoff targets.

    - Stop = Spring low (or ATR-based below Ice)
    - Target = TR upper (Creek) or projected target (2x range width)
    - R:R < 1:2 → reject trade
    """
    if entry <= 0 or stop <= 0 or entry <= stop:
        return {"rr": 0.0, "target": 0.0, "risk": 0.0, "reward": 0.0, "valid": False}

    risk = entry - stop
    if risk <= 0:
        return {"rr": 0.0, "target": 0.0, "risk": 0.0, "reward": 0.0, "valid": False}

    if target is None and tr is not None:
        # Phase 1 target: Creek (TR upper)
        # Phase 2 target: Projected (TR upper + range width)
        target = tr.projected_target(multiplier=1.0)
    elif target is None:
        target = entry * 1.05  # Fallback: 5% target

    reward = target - entry
    rr = reward / risk if risk > 0 else 0.0

    return {
        "rr": round(rr, 2),
        "target": round(target, 4),
        "risk": round(risk, 4),
        "reward": round(reward, 4),
        "valid": rr >= 2.0,  # Only valid if R:R >= 1:2
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main: WyckoffAnalyzer Strategy
# ─────────────────────────────────────────────────────────────────────────────

class WyckoffAnalyzer:
    """
    Wyckoff+ Triple Filter Strategy.

    Implements the full Wyckoff methodology with modern SMC enhancements.
    Conforms to ``TradingStrategy`` Protocol.

    Output columns (appended to DataFrame):
        wyckoff_signal         — int (-1, 0, 1)
        wyckoff_signal_reason  — str explanation
        wyckoff_phase          — str phase label
        wyckoff_tr_upper       — float Creek level
        wyckoff_tr_lower       — float Ice level
        wyckoff_tr_position    — float 0-1 position within range
        wyckoff_spring         — bool Spring event
        wyckoff_lps            — bool LPS event
        wyckoff_rr             — float Risk:Reward ratio
        wyckoff_target         — float projected price target
        wyckoff_stop           — float suggested stop loss
        vsa_label              — str VSA event label
        vsa_sos                — bool Sign of Strength
        vsa_sow                — bool Sign of Weakness
        vsa_no_supply          — bool No Supply
        vsa_stopping_vol       — bool Stopping Volume
        vsa_shakeout           — bool Shakeout
        effort_vs_result       — float Effort vs Result ratio
        weis_wave_ratio        — float cumulative up/down volume ratio
        wyckoff_score          — float composite score -1.0 to +1.0 (for AI features)
    """

    def __init__(self, config: WyckoffConfig | None = None) -> None:
        self.cfg = config or WyckoffConfig()

    def generate_signals(self, data: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        """
        Run the full Wyckoff Triple Filter and generate trading signals.

        Args:
            data: DataFrame with columns: Date, Open, High, Low, Close, Volume

        Returns:
            DataFrame with Wyckoff signal columns appended.
        """
        required = {"High", "Low", "Close", "Volume"}
        if not required.issubset(data.columns):
            missing = required - set(data.columns)
            logger.warning(f"WyckoffAnalyzer: missing columns {missing}, returning neutral.")
            data = data.copy()
            data["wyckoff_signal"] = 0
            data["wyckoff_signal_reason"] = "MISSING_DATA"
            return data

        if len(data) < 50:
            logger.warning("WyckoffAnalyzer: Not enough data (need 50+ bars).")
            data = data.copy()
            data["wyckoff_signal"] = 0
            data["wyckoff_signal_reason"] = "INSUFFICIENT_DATA"
            return data

        df = data.copy()

        # ── Layer 0: Baseline Indicators ─────────────────────────────────
        atr = _compute_atr(df, period=14)
        ma_200 = df["Close"].rolling(self.cfg.ma_long_period, min_periods=50).mean()

        # ── Layer 1: Trading Range + Phase ───────────────────────────────
        tr = _detect_trading_range(df, self.cfg, atr)
        phase_labels = _classify_wyckoff_phase(df, tr, ma_200, self.cfg)
        df["wyckoff_phase"] = phase_labels

        # Store TR levels
        df["wyckoff_tr_upper"] = tr.upper if tr else np.nan
        df["wyckoff_tr_lower"] = tr.lower if tr else np.nan
        df["wyckoff_tr_position"] = (
            df["Close"].apply(tr.position_score) if tr else 0.5
        )

        # ── Layer 2: VSA Engine ──────────────────────────────────────────
        df = _run_vsa_engine(df, self.cfg)

        # ── Layer 3: Event Detection ─────────────────────────────────────
        spring = _detect_spring(df, tr, self.cfg, atr)
        lps = _detect_lps(df, tr, self.cfg, atr)
        df["wyckoff_spring"] = spring
        df["wyckoff_lps"] = lps

        # ── Composite Wyckoff Score ───────────────────────────────────────
        # Scores each bar from -1.0 (full distribution) to +1.0 (prime accumulation)
        score = pd.Series(0.0, index=df.index)

        # Phase contribution
        phase_map = {
            "📦 PHASE B (Accumulation)": 0.1,
            "🌀 PHASE C (Spring Zone)": 0.3,
            "🚀 PHASE D (SOS Markup)": 0.4,
            "⚡ PHASE E (Markup Escape)": 0.2,
            "📈 MARKUP": 0.15,
            "📉 DISTRIBUTION": -0.3,
            "📉 MARKDOWN": -0.2,
        }
        for label, val in phase_map.items():
            score[df["wyckoff_phase"] == label] += val

        # VSA contribution
        score[df["vsa_sos"]] += 0.25
        score[df["vsa_sow"]] -= 0.25
        score[df["vsa_no_supply"]] += 0.15
        score[df["vsa_stopping_vol"]] += 0.10
        score[df["vsa_shakeout"]] += 0.10

        # Event contribution
        score[df["wyckoff_spring"]] += 0.30
        score[df["wyckoff_lps"]] += 0.20

        # Weis wave contribution
        if "weis_wave_ratio" in df.columns:
            wwr = df["weis_wave_ratio"].fillna(1.0)
            score += ((wwr - 1.0) * 0.1).clip(-0.2, 0.2)

        # Smooth and clip
        score = score.rolling(self.cfg.phase_score_smoothing, min_periods=1).mean().clip(-1.0, 1.0)
        df["wyckoff_score"] = score

        # ── Signal Generation ────────────────────────────────────────────
        df["wyckoff_signal"] = 0
        df["wyckoff_signal_reason"] = ""
        df["wyckoff_rr"] = 0.0
        df["wyckoff_target"] = np.nan
        df["wyckoff_stop"] = np.nan

        for i in range(len(df)):
            row_score = score.iloc[i]
            close_i = df["Close"].iloc[i]

            # Default stop: 1.5x ATR below current close
            atr_i = atr.iloc[i] if not pd.isna(atr.iloc[i]) else close_i * 0.02
            stop_i = close_i - 1.5 * atr_i

            # ── BUY Signals ──────────────────────────────────────────
            if spring.iloc[i]:
                # Best Wyckoff entry: Spring confirmed
                rr_result = _calculate_rr(close_i, stop_i, tr)
                df.at[df.index[i], "wyckoff_signal"] = 1
                df.at[df.index[i], "wyckoff_signal_reason"] = (
                    f"SPRING: Price tested below Ice and recovered. "
                    f"R:R={rr_result['rr']:.1f} | Target={rr_result['target']:.2f}"
                )
                df.at[df.index[i], "wyckoff_rr"] = rr_result["rr"]
                df.at[df.index[i], "wyckoff_target"] = rr_result["target"]
                df.at[df.index[i], "wyckoff_stop"] = stop_i

            elif lps.iloc[i]:
                # Phase D re-entry: LPS pullback
                rr_result = _calculate_rr(close_i, stop_i, tr)
                if rr_result["valid"]:   # Only if R:R ≥ 1:2
                    df.at[df.index[i], "wyckoff_signal"] = 1
                    df.at[df.index[i], "wyckoff_signal_reason"] = (
                        f"LPS: Last Point of Support — low-vol pullback in Phase D. "
                        f"R:R={rr_result['rr']:.1f} | Target={rr_result['target']:.2f}"
                    )
                    df.at[df.index[i], "wyckoff_rr"] = rr_result["rr"]
                    df.at[df.index[i], "wyckoff_target"] = rr_result["target"]
                    df.at[df.index[i], "wyckoff_stop"] = stop_i

            elif df["vsa_sos"].iloc[i] and row_score >= 0.4:
                rr_result = _calculate_rr(close_i, stop_i, tr)
                if rr_result["valid"]:
                    df.at[df.index[i], "wyckoff_signal"] = 1
                    df.at[df.index[i], "wyckoff_signal_reason"] = (
                        f"SOS: Sign of Strength — wide-spread up bar + high volume. "
                        f"Score={row_score:.2f} | R:R={rr_result['rr']:.1f}"
                    )
                    df.at[df.index[i], "wyckoff_rr"] = rr_result["rr"]
                    df.at[df.index[i], "wyckoff_target"] = rr_result["target"]
                    df.at[df.index[i], "wyckoff_stop"] = stop_i

            # ── SELL Signals ─────────────────────────────────────────
            elif df["vsa_sow"].iloc[i] and row_score <= -0.35:
                df.at[df.index[i], "wyckoff_signal"] = -1
                df.at[df.index[i], "wyckoff_signal_reason"] = (
                    f"SOW: Sign of Weakness — distribution detected. "
                    f"Score={row_score:.2f}"
                )
            elif row_score <= -0.5:
                df.at[df.index[i], "wyckoff_signal"] = -1
                df.at[df.index[i], "wyckoff_signal_reason"] = (
                    f"MARKDOWN: Composite Wyckoff score is strongly negative ({row_score:.2f})."
                )

        # Summary stats
        n_buy = int((df["wyckoff_signal"] == 1).sum())
        n_sell = int((df["wyckoff_signal"] == -1).sum())
        n_spring = int(spring.sum())
        n_lps = int(lps.sum())
        logger.info(
            f"WyckoffAnalyzer: {n_buy} BUY / {n_sell} SELL signals | "
            f"{n_spring} Springs | {n_lps} LPS events | TR={tr}"
        )

        return df

    def analyze_current_state(self, data: pd.DataFrame) -> dict:
        """
        Return a snapshot of current Wyckoff state for UI display.

        Returns a dict suitable for rendering in Streamlit ai_forecast.py.
        """
        if data.empty or len(data) < 50:
            return {"phase": "UNKNOWN", "score": 0.0, "signal": 0}

        df = self.generate_signals(data)
        last = df.iloc[-1]

        atr_14 = _compute_atr(data, 14).iloc[-1]
        close = last["Close"]
        spring_low = data["Low"].tail(60).min()
        stop = spring_low * 0.995

        # TR info
        atr = _compute_atr(data, 14)
        tr = _detect_trading_range(data, self.cfg, atr)
        rr_result = _calculate_rr(close, stop, tr) if tr else {"rr": 0.0, "valid": False, "target": 0.0}

        return {
            "phase": last.get("wyckoff_phase", "UNKNOWN"),
            "score": round(float(last.get("wyckoff_score", 0.0)), 3),
            "signal": int(last.get("wyckoff_signal", 0)),
            "signal_reason": str(last.get("wyckoff_signal_reason", "")),
            "vsa_label": str(last.get("vsa_label", "⚪ Normal")),
            "spring": bool(last.get("wyckoff_spring", False)),
            "lps": bool(last.get("wyckoff_lps", False)),
            "sos": bool(last.get("vsa_sos", False)),
            "sow": bool(last.get("vsa_sow", False)),
            "no_supply": bool(last.get("vsa_no_supply", False)),
            "effort_vs_result": round(float(last.get("effort_vs_result", 1.0)), 3),
            "weis_wave_ratio": round(float(last.get("weis_wave_ratio", 1.0)), 3),
            "tr_upper": tr.upper if tr else None,
            "tr_lower": tr.lower if tr else None,
            "tr_position": round(float(last.get("wyckoff_tr_position", 0.5)), 3),
            "tr_width_days": tr.width_days if tr else None,
            "rr": rr_result["rr"],
            "target": rr_result.get("target", 0.0),
            "stop": round(stop, 4),
            "valid_trade": rr_result.get("valid", False),
            "atr": round(float(atr_14), 4),
        }
