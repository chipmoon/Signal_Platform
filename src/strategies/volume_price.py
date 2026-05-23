"""
Strategy 2: Volume-Price Mismatch Detector
==========================================
Detects manipulation via abnormal volume-price divergences.
Uses Isolation Forest (ML) for anomaly detection.

Conforms to ``TradingStrategy`` Protocol.

Signals:
    1. Volume spike > 5× avg + price drop > 3% → MANIPULATION alert
    2. Isolation Forest flags unusual multi-feature patterns
    3. Volume spike + price increase → ACCUMULATION (BUY)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.config import VolumePriceConfig

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("scikit-learn not installed. ML anomaly detection disabled.")


# Feature columns for ML model
_ML_FEATURES = [
    "Volume_Ratio",
    "Volume_ZScore",
    "Price_Change",
    "Price_Volatility",
    "Price_Range",
    "VP_Divergence",
]


class VolumePriceDetector:
    """Detects manipulation using rule-based + ML approaches.

    Implements ``TradingStrategy`` Protocol.
    """

    def __init__(self, config: VolumePriceConfig | None = None) -> None:
        cfg = config or VolumePriceConfig()
        self.volume_spike_threshold = cfg.volume_spike_threshold
        self.price_drop_threshold = cfg.price_drop_threshold
        self.lookback_window = cfg.lookback_window
        self.ml_contamination = cfg.ml_contamination
        self.ml_n_estimators = cfg.ml_n_estimators
        self._model: object | None = None
        self._scaler: object | None = None

    # ── Feature Engineering ──────────────────────

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features for anomaly detection.

        All operations are vectorized (no loops).
        Includes OBV (On-Balance Volume) divergence detection.
        """
        result = df.copy()
        w = self.lookback_window

        # Volume features
        vol_ma = result["Volume"].rolling(window=w, min_periods=5).mean()
        vol_std = result["Volume"].rolling(window=w, min_periods=5).std()
        result["Volume_MA"] = vol_ma
        result["Volume_Ratio"] = result["Volume"] / vol_ma.replace(0, np.nan)
        result["Volume_ZScore"] = (result["Volume"] - vol_ma) / vol_std.replace(0, np.nan)

        # Price features
        result["Price_Change"] = result["Close"].pct_change()
        result["Price_Change_2d"] = result["Close"].pct_change(2)
        result["Price_Change_5d"] = result["Close"].pct_change(5)
        result["Price_Volatility"] = result["Price_Change"].rolling(window=w, min_periods=5).std()
        result["Price_Range"] = (result["High"] - result["Low"]) / result["Close"]

        # Volume-Price Divergence
        result["VP_Divergence"] = result["Volume_Ratio"] * result["Price_Change"].abs()
        result["VP_Divergence_Signed"] = result["Volume_Ratio"] * result["Price_Change"]

        # OBV (On-Balance Volume) — cumulative volume flow indicator
        price_direction = np.sign(result["Close"].diff().fillna(0))
        result["OBV"] = (result["Volume"] * price_direction).cumsum()

        # OBV divergence detection (rolling 20-day trends)
        obv_trend = result["OBV"].diff(w).fillna(0)
        price_trend = result["Close"].diff(w).fillna(0)
        # Hidden bullish: OBV rising + price falling = accumulation
        result["OBV_Bullish_Div"] = (obv_trend > 0) & (price_trend < 0)
        # Hidden bearish: OBV falling + price rising = distribution
        result["OBV_Bearish_Div"] = (obv_trend < 0) & (price_trend > 0)

        return result

    # ── ML Training ──────────────────────────────

    def train_ml_model(self, data: pd.DataFrame) -> None:
        """Train Isolation Forest on historical data to learn normal patterns."""
        if not ML_AVAILABLE:
            logger.warning("ML not available. Skipping model training.")
            return

        df = self._calculate_features(data).dropna(subset=_ML_FEATURES)
        X = df[_ML_FEATURES].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(
            contamination=self.ml_contamination,
            n_estimators=self.ml_n_estimators,
            max_samples="auto",
            random_state=42,
        )
        model.fit(X_scaled)

        self._model = model
        self._scaler = scaler
        logger.success(f"Isolation Forest trained on {len(X)} samples.")

    # ── Signal Generation ────────────────────────

    def generate_signals(self, data: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        """Generate signals from volume-price analysis.

        Appends columns:
            ``vp_signal`` (-1/0/1), ``vp_signal_reason``,
            ``vp_volume_spike``, ``vp_manipulation_alert``, ``vp_ml_anomaly``,
            ``OBV``, ``OBV_Bullish_Div``, ``OBV_Bearish_Div``.
        """
        df = self._calculate_features(data)

        # Rule-based detection
        df["vp_volume_spike"] = df["Volume_Ratio"] > self.volume_spike_threshold
        df["vp_price_drop"] = df["Price_Change"] < self.price_drop_threshold
        df["vp_manipulation_alert"] = df["vp_volume_spike"] & df["vp_price_drop"]

        # ML detection
        df["vp_ml_anomaly"] = False
        df["vp_ml_score"] = 0.0

        if self._model and self._scaler and ML_AVAILABLE:
            valid_mask = df[_ML_FEATURES].notna().all(axis=1)
            X_valid = df.loc[valid_mask, _ML_FEATURES].values

            if len(X_valid) > 0:
                X_scaled = self._scaler.transform(X_valid)
                preds = self._model.predict(X_scaled)
                scores = self._model.decision_function(X_scaled)
                df.loc[valid_mask, "vp_ml_anomaly"] = preds == -1
                df.loc[valid_mask, "vp_ml_score"] = scores

        # Combined signal
        df["vp_signal"] = 0
        df["vp_signal_reason"] = ""

        # SELL: Rule-based manipulation
        sell_rule = df["vp_manipulation_alert"]
        df.loc[sell_rule, "vp_signal"] = -1
        df.loc[sell_rule, "vp_signal_reason"] = "MANIPULATION_ALERT: Volume >5x avg + Price drop >3%"

        # SELL: ML anomaly with bearish divergence
        sell_ml = df["vp_ml_anomaly"] & (df["VP_Divergence_Signed"] < 0) & ~sell_rule
        df.loc[sell_ml, "vp_signal"] = -1
        df.loc[sell_ml, "vp_signal_reason"] = "ML_ANOMALY: Isolation Forest flagged bearish pattern"

        # SELL: OBV bearish divergence (distribution detected)
        sell_obv = df["OBV_Bearish_Div"] & ~sell_rule & ~sell_ml
        df.loc[sell_obv, "vp_signal"] = -1
        df.loc[sell_obv, "vp_signal_reason"] = "OBV_DIVERGENCE: Price rising but volume declining (distribution)"

        # BUY: Volume spike + price increase (accumulation)
        buy_mask = (
            df["vp_volume_spike"]
            & (df["Price_Change"] > abs(self.price_drop_threshold))
            & ~df["vp_ml_anomaly"]
        )
        df.loc[buy_mask, "vp_signal"] = 1
        df.loc[buy_mask, "vp_signal_reason"] = "ACCUMULATION: High volume buying detected"

        # BUY: OBV bullish divergence (hidden accumulation)
        buy_obv = df["OBV_Bullish_Div"] & ~buy_mask & (df["vp_signal"] == 0)
        df.loc[buy_obv, "vp_signal"] = 1
        df.loc[buy_obv, "vp_signal_reason"] = "OBV_ACCUMULATION: Price falling but smart money accumulating"

        n_manip = int(df["vp_manipulation_alert"].sum())
        n_ml = int(df["vp_ml_anomaly"].sum())
        n_obv_bull = int(df["OBV_Bullish_Div"].sum())
        n_obv_bear = int(df["OBV_Bearish_Div"].sum())
        logger.info(
            f"Volume-Price Detector: {n_manip} rule-based, {n_ml} ML anomalies, "
            f"OBV divergence: {n_obv_bull} bullish, {n_obv_bear} bearish."
        )
        return df
