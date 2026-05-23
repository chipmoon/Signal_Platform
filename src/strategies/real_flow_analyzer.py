"""
Real Money Flow Analyzer
========================
Normalizes and scores institutional flow signals from real market data.

Design goals:
- No synthetic proxy in production.
- Fail-closed when required flow fields are missing.
- Provide deterministic, auditable flow factors for the signal combiner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class RealFlowAnalyzer:
    """Compute real flow score and discrete signal from market microstructure fields."""

    REQUIRED_FLOW_COLS = ("Foreign_Buy", "Foreign_Sell", "Block_Trade_Volume")

    def generate_signals(
        self,
        data: pd.DataFrame,
        *,
        fail_closed: bool = True,
    ) -> pd.DataFrame:
        """Append real flow features and signal columns.

        Output columns:
            - real_flow_available (bool)
            - foreign_net_flow_ratio
            - block_trade_ratio
            - session_rel_volume
            - intraday_vwap_dev
            - real_flow_score (-1..1)
            - real_flow_signal (-1/0/1)
            - real_flow_reason
        """
        df = data.copy()
        for col in (
            "foreign_net_flow_ratio",
            "block_trade_ratio",
            "session_rel_volume",
            "intraday_vwap_dev",
            "real_flow_score",
        ):
            df[col] = 0.0
        df["real_flow_signal"] = 0
        df["real_flow_reason"] = ""
        df["real_flow_available"] = False

        has_required = all(c in df.columns for c in self.REQUIRED_FLOW_COLS)
        if not has_required:
            if fail_closed:
                df["real_flow_reason"] = "MISSING_REAL_FLOW_COLUMNS"
                logger.warning(
                    "RealFlowAnalyzer: missing Foreign/Block flow columns. "
                    "Fail-closed mode active."
                )
                return df

        volume = pd.to_numeric(df.get("Volume", 0), errors="coerce").fillna(0.0)
        vol_ma20 = volume.rolling(20, min_periods=5).mean().replace(0, np.nan)
        df["session_rel_volume"] = (volume / vol_ma20).fillna(1.0).clip(0.0, 10.0)

        # VWAP deviation: prefer provided intraday VWAP; fallback to rolling typical-price VWAP.
        close = pd.to_numeric(df.get("Close", 0), errors="coerce")
        if "VWAP" in df.columns:
            vwap = pd.to_numeric(df["VWAP"], errors="coerce").replace(0, np.nan)
        else:
            typical = (
                pd.to_numeric(df.get("High", close), errors="coerce")
                + pd.to_numeric(df.get("Low", close), errors="coerce")
                + close
            ) / 3.0
            vwap = (
                (typical * volume).rolling(20, min_periods=5).sum()
                / volume.rolling(20, min_periods=5).sum().replace(0, np.nan)
            )
        df["intraday_vwap_dev"] = ((close - vwap) / vwap).fillna(0.0).clip(-0.2, 0.2)

        if has_required:
            foreign_buy = pd.to_numeric(df["Foreign_Buy"], errors="coerce").fillna(0.0)
            foreign_sell = pd.to_numeric(df["Foreign_Sell"], errors="coerce").fillna(0.0)
            block_vol = pd.to_numeric(df["Block_Trade_Volume"], errors="coerce").fillna(0.0)

            denom = volume.replace(0, np.nan)
            df["foreign_net_flow_ratio"] = ((foreign_buy - foreign_sell) / denom).fillna(0.0).clip(-2.0, 2.0)
            df["block_trade_ratio"] = (block_vol / denom).fillna(0.0).clip(0.0, 2.0)
            df["real_flow_available"] = True
        else:
            df["foreign_net_flow_ratio"] = 0.0
            df["block_trade_ratio"] = 0.0
            df["real_flow_available"] = False

        # Composite real flow score
        foreign_term = np.tanh(df["foreign_net_flow_ratio"] * 3.0) * 0.45
        block_term = np.tanh((df["block_trade_ratio"] - 0.12) * 3.0) * 0.25
        vol_term = np.tanh((df["session_rel_volume"] - 1.0) * 1.5) * 0.15
        vwap_term = np.tanh(df["intraday_vwap_dev"] * 12.0) * 0.15
        df["real_flow_score"] = (foreign_term + block_term + vol_term + vwap_term).clip(-1.0, 1.0)

        df.loc[df["real_flow_score"] > 0.20, "real_flow_signal"] = 1
        df.loc[df["real_flow_score"] < -0.20, "real_flow_signal"] = -1
        df.loc[df["real_flow_signal"] == 1, "real_flow_reason"] = "REAL_FLOW_BULLISH"
        df.loc[df["real_flow_signal"] == -1, "real_flow_reason"] = "REAL_FLOW_BEARISH"
        df.loc[df["real_flow_signal"] == 0, "real_flow_reason"] = "REAL_FLOW_NEUTRAL"

        return df
