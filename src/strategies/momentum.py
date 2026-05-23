"""
Stochastic Oscillator — Momentum Confirmation Engine
=====================================================
Implements the Stochastic Oscillator (%K / %D) as a momentum filter
for FVG and Order Block entries.

Purpose:
    - Prevent "buying the top" when momentum is exhausted (overbought)
    - Confirm high-probability entries when price recovers from oversold
    - Provide timing confirmation for SMC zone entries

Output columns:
    stoch_k         — float 0-100: Fast Stochastic %K
    stoch_d         — float 0-100: Slow Stochastic %D (signal line)
    stoch_signal    — int (-1, 0, 1): Momentum signal
    stoch_status    — str: "OVERBOUGHT" / "OVERSOLD" / "NEUTRAL"
    stoch_crossover — bool: %K crossed above %D (bullish)
    stoch_crossunder— bool: %K crossed below %D (bearish)

Conforms to ``TradingStrategy`` Protocol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.config import StochasticConfig


class StochasticOscillator:
    """
    Stochastic Oscillator Engine.

    Calculates %K and %D lines, detects crossovers, and generates
    momentum signals for use as a confirmation filter.

    Conforms to ``TradingStrategy`` Protocol.
    """

    def __init__(self, config: StochasticConfig | None = None) -> None:
        self.cfg = config or StochasticConfig()

    def generate_signals(self, data: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator and generate momentum signals.

        Args:
            data: DataFrame with High, Low, Close columns.

        Returns:
            DataFrame with stochastic columns appended.
        """
        required = {"High", "Low", "Close"}
        if not required.issubset(data.columns):
            logger.warning("StochasticOscillator: missing HLC columns.")
            df = data.copy()
            df["stoch_k"] = 50.0
            df["stoch_d"] = 50.0
            df["stoch_signal"] = 0
            df["stoch_status"] = "NEUTRAL"
            df["stoch_crossover"] = False
            df["stoch_crossunder"] = False
            return df

        df = data.copy()
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        # ── Raw %K Calculation ────────────────────────────────────────
        # %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        lowest_low = low.rolling(self.cfg.k_period, min_periods=1).min()
        highest_high = high.rolling(self.cfg.k_period, min_periods=1).max()

        range_hl = (highest_high - lowest_low).replace(0, np.nan)
        raw_k = ((close - lowest_low) / range_hl * 100).fillna(50.0)

        # ── Slow %K (with slowing factor) ─────────────────────────────
        stoch_k = raw_k.rolling(self.cfg.slowing, min_periods=1).mean()

        # ── %D (Signal Line) ──────────────────────────────────────────
        stoch_d = stoch_k.rolling(self.cfg.d_period, min_periods=1).mean()

        df.loc[:, "stoch_k"] = stoch_k.clip(0, 100).values
        df.loc[:, "stoch_d"] = stoch_d.clip(0, 100).values

        # ── Status Classification ─────────────────────────────────────
        status = pd.Series("NEUTRAL", index=df.index, dtype=str)
        status[stoch_k >= self.cfg.overbought] = "OVERBOUGHT"
        status[stoch_k <= self.cfg.oversold] = "OVERSOLD"
        df.loc[:, "stoch_status"] = status.values

        # ── Crossover Detection ───────────────────────────────────────
        # Bullish crossover: %K crosses above %D
        prev_k = stoch_k.shift(1)
        prev_d = stoch_d.shift(1)

        crossover = (stoch_k > stoch_d) & (prev_k <= prev_d)
        crossunder = (stoch_k < stoch_d) & (prev_k >= prev_d)

        df.loc[:, "stoch_crossover"] = crossover.fillna(False).values
        df.loc[:, "stoch_crossunder"] = crossunder.fillna(False).values

        # ── Signal Generation ─────────────────────────────────────────
        # BUY:  %K crosses above %D in oversold zone (< 20)
        # SELL: %K crosses below %D in overbought zone (> 80)
        signal = pd.Series(0, index=df.index, dtype=int)

        buy_signal = crossover & (stoch_k <= self.cfg.oversold + 10)
        sell_signal = crossunder & (stoch_k >= self.cfg.overbought - 10)

        signal[buy_signal] = 1
        signal[sell_signal] = -1
        df.loc[:, "stoch_signal"] = signal.values

        # ── Signal Reason ─────────────────────────────────────────────
        reason = pd.Series("", index=df.index, dtype=str)
        reason[buy_signal] = "STOCH_BULL: %K crossed above %D in oversold zone — momentum recovery"
        reason[sell_signal] = "STOCH_BEAR: %K crossed below %D in overbought zone — momentum exhaustion"
        df.loc[:, "stoch_signal_reason"] = reason.values

        n_buy = int(buy_signal.sum())
        n_sell = int(sell_signal.sum())
        last_k = float(stoch_k.iloc[-1]) if len(stoch_k) > 0 else 50.0
        last_status = status.iloc[-1] if len(status) > 0 else "NEUTRAL"

        logger.info(
            f"StochasticOscillator: {n_buy} BUY / {n_sell} SELL | "
            f"Current %K={last_k:.1f} | Status={last_status}"
        )
        return df

    def get_current_state(self, data: pd.DataFrame) -> dict:
        """Return snapshot dict for UI display."""
        if data.empty or len(data) < 20:
            return {
                "k": 50.0, "d": 50.0, "status": "NEUTRAL",
                "signal": 0, "crossover": False, "crossunder": False,
            }

        df = self.generate_signals(data)
        last = df.iloc[-1]

        return {
            "k": round(float(last.get("stoch_k", 50.0)), 1),
            "d": round(float(last.get("stoch_d", 50.0)), 1),
            "status": str(last.get("stoch_status", "NEUTRAL")),
            "signal": int(last.get("stoch_signal", 0)),
            "signal_reason": str(last.get("stoch_signal_reason", "")),
            "crossover": bool(last.get("stoch_crossover", False)),
            "crossunder": bool(last.get("stoch_crossunder", False)),
        }

    def evaluate_entry(
        self, data: pd.DataFrame, zones: list[dict]
    ) -> list[dict]:
        """
        Evaluate FVG/OB zones against Stochastic status for Entry Confirmation Table.

        Args:
            data: DataFrame with OHLC data (Stochastic will be calculated).
            zones: List of dicts with keys: zone_top, zone_bottom, type, direction.

        Returns:
            List of dicts with added: stoch_status, stoch_confirmed, action.
        """
        if not zones:
            return []

        df = self.generate_signals(data)
        last = df.iloc[-1]

        k_val = float(last.get("stoch_k", 50.0))
        status = str(last.get("stoch_status", "NEUTRAL"))
        crossover = bool(last.get("stoch_crossover", False))
        crossunder = bool(last.get("stoch_crossunder", False))

        results = []
        for zone in zones:
            entry = zone.copy()
            direction = zone.get("direction", "Bullish")

            if direction == "Bullish":
                if status == "OVERBOUGHT":
                    entry["stoch_confirmed"] = False
                    entry["stoch_label"] = "❌ Overbought"
                    entry["action"] = "⏸️ WAIT"
                elif status == "OVERSOLD" or crossover:
                    entry["stoch_confirmed"] = True
                    entry["stoch_label"] = "✅ Confirmed"
                    entry["action"] = "🟢 BUY"
                else:
                    entry["stoch_confirmed"] = True
                    entry["stoch_label"] = "⚠️ Neutral"
                    entry["action"] = "🟡 CAUTION"
            else:  # Bearish
                if status == "OVERSOLD":
                    entry["stoch_confirmed"] = False
                    entry["stoch_label"] = "❌ Oversold"
                    entry["action"] = "⏸️ WAIT"
                elif status == "OVERBOUGHT" or crossunder:
                    entry["stoch_confirmed"] = True
                    entry["stoch_label"] = "✅ Confirmed"
                    entry["action"] = "🔴 SELL"
                else:
                    entry["stoch_confirmed"] = True
                    entry["stoch_label"] = "⚠️ Neutral"
                    entry["action"] = "🟡 CAUTION"

            entry["stoch_k"] = k_val
            entry["stoch_status"] = status
            results.append(entry)

        return results
