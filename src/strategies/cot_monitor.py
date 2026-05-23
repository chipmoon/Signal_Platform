"""
Strategy 1: COT Report Monitor
===============================
Monitors weekly Commitment of Traders reports for manipulation signals.

Conforms to ``TradingStrategy`` Protocol.

Signals:
    1. Commercial Short Ratio > threshold + high volatility → FLUSH alert
    2. Short position change > N× average → ANOMALY alert
    3. Rapid short covering → Reversal BUY signal
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.config import ANNUALIZE_FACTOR, COTConfig


class COTMonitor:
    """Detects manipulation via commercial traders' COT positioning.

    Implements ``TradingStrategy`` Protocol.
    """

    def __init__(self, config: COTConfig | None = None) -> None:
        cfg = config or COTConfig()
        self.short_ratio_threshold = cfg.short_ratio_threshold
        self.anomaly_multiplier = cfg.anomaly_multiplier
        self.volatility_window = cfg.volatility_window
        self.change_lookback = cfg.change_lookback

    def generate_signals(self, data: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        """Analyze COT data and generate trading signals.

        Appends columns:
            ``cot_signal`` (-1/0/1), ``cot_signal_reason``,
            ``cot_short_ratio_alert``, ``cot_anomaly_alert``.
        """
        df = data.copy()

        if "Commercial_Short_Ratio" not in df.columns:
            logger.warning("COT data not found. Skipping COT signals.")
            df["cot_signal"] = 0
            df["cot_signal_reason"] = ""
            return df

        # Signal 1: Short Ratio exceeds threshold
        df["cot_short_ratio_alert"] = df["Commercial_Short_Ratio"] > self.short_ratio_threshold

        # Signal 2: Anomalous short position change
        short_abs = df["Short_Change"].abs()
        avg_change = short_abs.rolling(window=self.change_lookback, min_periods=1).mean()
        df["cot_anomaly_alert"] = short_abs > (self.anomaly_multiplier * avg_change)

        # Signal 3: Price volatility context
        returns = df["Close"].pct_change()
        vol = returns.rolling(window=self.volatility_window, min_periods=5).std()
        annualized_vol = vol * np.sqrt(ANNUALIZE_FACTOR)
        vol_ma = annualized_vol.rolling(window=60, min_periods=10).mean()
        high_vol = annualized_vol > vol_ma * 1.5

        # Signal 4: Multi-timeframe confirmation
        # Compare 4-week vs 12-week COT trend direction
        csr = df["Commercial_Short_Ratio"]
        csr_trend_4w = csr - csr.shift(4)   # ~1 month trend
        csr_trend_12w = csr - csr.shift(12)  # ~3 month trend
        df["cot_4w_trend"] = np.sign(csr_trend_4w.fillna(0))
        df["cot_12w_trend"] = np.sign(csr_trend_12w.fillna(0))
        # Both timeframes agree → stronger confirmation
        df["cot_mtf_agree"] = df["cot_4w_trend"] == df["cot_12w_trend"]

        # Combined signals (vectorized — no iterrows)
        df["cot_signal"] = 0
        df["cot_signal_reason"] = ""

        # SELL: High shorts + anomaly + volatile
        sell_strong = df["cot_short_ratio_alert"] & df["cot_anomaly_alert"] & high_vol
        df.loc[sell_strong, "cot_signal"] = -1
        df.loc[sell_strong, "cot_signal_reason"] = (
            "FLUSH_ALERT: Commercial Short spike + high volatility"
        )

        # SELL: High shorts + anomaly (moderate) — boosted by MTF confirmation
        sell_moderate = df["cot_short_ratio_alert"] & df["cot_anomaly_alert"] & ~sell_strong
        df.loc[sell_moderate, "cot_signal"] = -1
        df.loc[sell_moderate, "cot_signal_reason"] = (
            "COT_ANOMALY: Unusual commercial short buildup"
        )
        # When MTF disagrees on moderate sells, weaken the signal
        mtf_disagree = sell_moderate & ~df["cot_mtf_agree"]
        df.loc[mtf_disagree, "cot_signal"] = 0
        df.loc[mtf_disagree, "cot_signal_reason"] = ""

        # BUY: Commercial shorts covering rapidly + MTF confirmation
        buy_mask = (
            (df["Short_Change"] < 0)
            & (short_abs > self.anomaly_multiplier * avg_change)
            & (df["Commercial_Short_Ratio"] < self.short_ratio_threshold)
        )
        df.loc[buy_mask, "cot_signal"] = 1
        df.loc[buy_mask, "cot_signal_reason"] = (
            "COT_REVERSAL: Commercial traders covering shorts"
        )
        # MTF-confirmed buys get an enhanced reason
        mtf_buy = buy_mask & df["cot_mtf_agree"]
        df.loc[mtf_buy, "cot_signal_reason"] = (
            "COT_REVERSAL_MTF: Multi-timeframe confirmed short covering"
        )

        n_sell = (df["cot_signal"] == -1).sum()
        n_buy = (df["cot_signal"] == 1).sum()
        n_mtf = df["cot_mtf_agree"].sum()
        logger.info(
            f"COT Monitor: {n_sell} SELL, {n_buy} BUY signals "
            f"({n_mtf} multi-timeframe confirmations)."
        )
        return df

    def get_current_reading(self, data: pd.DataFrame) -> dict[str, str | int]:
        """Return the latest COT analysis snapshot."""
        if data.empty or "Commercial_Short_Ratio" not in data.columns:
            return {"status": "NO_DATA"}

        latest = data.iloc[-1]
        return {
            "date": str(latest.get("Date", "N/A")),
            "commercial_short_ratio": f"{latest.get('Commercial_Short_Ratio', 0):.2%}",
            "short_change_pct": f"{latest.get('Short_Change_Pct', 0):.1f}%",
            "signal": int(latest.get("cot_signal", 0)),
            "reason": str(latest.get("cot_signal_reason", "")),
            "alert_level": "HIGH" if latest.get("cot_anomaly_alert", False) else "NORMAL",
        }
