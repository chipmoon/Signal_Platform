"""
Quant Money Flow Analyzer
=========================
Deterministic money-flow confirmation layer based on OHLCV:
- MFI (Money Flow Index)
- CMF (Chaikin Money Flow)
- OBV trend slope
- Accumulation/Distribution trend
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class QuantMoneyFlowAnalyzer:
    """Compute money-flow score and discrete confirmation signal."""

    def generate_signals(self, data: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        df = data.copy()
        required = {"High", "Low", "Close", "Volume"}
        if not required.issubset(df.columns):
            df["qmf_score"] = 0.0
            df["qmf_signal"] = 0
            df["qmf_reason"] = "MISSING_OHLCV"
            return df

        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        close = df["Close"].astype(float)
        vol = df["Volume"].astype(float).fillna(0.0)

        # MFI (14)
        tp = (high + low + close) / 3.0
        raw_flow = tp * vol
        pos_flow = raw_flow.where(tp > tp.shift(1), 0.0).rolling(14).sum()
        neg_flow = raw_flow.where(tp < tp.shift(1), 0.0).rolling(14).sum()
        money_ratio = pos_flow / neg_flow.replace(0, np.nan)
        mfi = 100 - (100 / (1 + money_ratio))
        df["QMF_MFI"] = mfi.fillna(50.0).clip(0, 100)

        # CMF (20)
        hl_range = (high - low).replace(0, np.nan)
        mfm = ((close - low) - (high - close)) / hl_range
        mfv = mfm.fillna(0.0) * vol
        cmf = mfv.rolling(20).sum() / vol.rolling(20).sum().replace(0, np.nan)
        df["QMF_CMF"] = cmf.fillna(0.0).clip(-1.0, 1.0)

        # OBV + trend slope
        direction = np.sign(close.diff()).fillna(0.0)
        obv = (direction * vol).cumsum()
        obv_slope = obv.diff(10).fillna(0.0)
        obv_norm = (obv_slope / vol.rolling(20).mean().replace(0, np.nan)).fillna(0.0)
        df["QMF_OBV"] = obv
        df["QMF_OBV_Slope"] = obv_slope

        # Acc/Dist line trend
        ad = (mfm.fillna(0.0) * vol).cumsum()
        ad_slope = ad.diff(10).fillna(0.0)
        ad_norm = (ad_slope / vol.rolling(20).mean().replace(0, np.nan)).fillna(0.0)
        df["QMF_AD"] = ad
        df["QMF_AD_Slope"] = ad_slope

        mfi_term = np.tanh((df["QMF_MFI"] - 50.0) / 20.0) * 0.35
        cmf_term = np.tanh(df["QMF_CMF"] * 3.0) * 0.30
        obv_term = np.tanh(obv_norm * 4.0) * 0.20
        ad_term = np.tanh(ad_norm * 4.0) * 0.15
        score = (mfi_term + cmf_term + obv_term + ad_term).clip(-1.0, 1.0)

        df["qmf_score"] = score
        df["qmf_signal"] = 0
        df.loc[df["qmf_score"] > 0.15, "qmf_signal"] = 1
        df.loc[df["qmf_score"] < -0.15, "qmf_signal"] = -1
        df["qmf_reason"] = "QMF_NEUTRAL"
        df.loc[df["qmf_signal"] == 1, "qmf_reason"] = "QMF_INFLOW_CONFIRMED"
        df.loc[df["qmf_signal"] == -1, "qmf_reason"] = "QMF_OUTFLOW_WARNING"

        # Early anomaly detection for unusual distribution / flow stress
        vol_ma20 = vol.rolling(20).mean().replace(0, np.nan)
        vol_std20 = vol.rolling(20).std().replace(0, np.nan)
        vol_z = ((vol - vol_ma20) / vol_std20).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ret_1d = close.pct_change().fillna(0.0)
        ret_vol20 = ret_1d.rolling(20).std().replace(0, np.nan)
        ret_z = (ret_1d / ret_vol20).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        cmf_delta = df["QMF_CMF"].diff(3).fillna(0.0)
        obv_delta = obv_slope.fillna(0.0)

        # Distribution anomaly score in [0, 1]
        sell_pressure = (
            0.35 * np.tanh(np.maximum(vol_z, 0.0) / 2.5)
            + 0.30 * np.tanh(np.maximum(-ret_z, 0.0) / 2.5)
            + 0.20 * np.tanh(np.maximum(-cmf_delta * 5.0, 0.0))
            + 0.15 * np.tanh(np.maximum(-obv_delta / vol_ma20.fillna(1.0), 0.0) * 3.0)
        )
        anomaly_score = sell_pressure.clip(0.0, 1.0)
        anomaly_flag = (anomaly_score >= 0.55).astype(int)
        anomaly_reason = np.where(
            anomaly_score >= 0.75,
            "QMF_DISTRIBUTION_ANOMALY_HIGH",
            np.where(anomaly_score >= 0.55, "QMF_DISTRIBUTION_ANOMALY_MEDIUM", "QMF_FLOW_NORMAL"),
        )

        df["qmf_anomaly_score"] = anomaly_score
        df["qmf_anomaly_flag"] = anomaly_flag
        df["qmf_anomaly_reason"] = anomaly_reason
        return df
