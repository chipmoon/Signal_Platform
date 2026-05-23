"""
Tests for Trading Strategies
=============================
Tests each strategy's signal generation for correctness and expected outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import AIConfig, BankConfig, COTConfig, VolumePriceConfig
from src.data.fetcher import merge_price_and_cot
from src.strategies.ai_predictor import AIPredictor
from src.strategies.bank_participation import BankParticipationMonitor
from src.strategies.cot_monitor import COTMonitor
from src.strategies.volume_price import VolumePriceDetector


# ─── COT Monitor ─────────────────────────────────

class TestCOTMonitor:
    def test_generate_signals_columns(
        self, sample_ohlcv_data: pd.DataFrame, sample_cot_data: pd.DataFrame
    ) -> None:
        merged = merge_price_and_cot(sample_ohlcv_data, sample_cot_data)
        strategy = COTMonitor(config=COTConfig())
        result = strategy.generate_signals(merged)

        assert "cot_signal" in result.columns
        assert "cot_signal_reason" in result.columns
        assert "cot_short_ratio_alert" in result.columns
        assert "cot_anomaly_alert" in result.columns
        assert "cot_mtf_agree" in result.columns

    def test_signals_are_valid_values(
        self, sample_ohlcv_data: pd.DataFrame, sample_cot_data: pd.DataFrame
    ) -> None:
        merged = merge_price_and_cot(sample_ohlcv_data, sample_cot_data)
        strategy = COTMonitor(config=COTConfig())
        result = strategy.generate_signals(merged)
        assert set(result["cot_signal"].unique()).issubset({-1, 0, 1})

    def test_no_cot_data_returns_zeros(self, sample_ohlcv_data: pd.DataFrame) -> None:
        strategy = COTMonitor(config=COTConfig())
        result = strategy.generate_signals(sample_ohlcv_data)
        assert (result["cot_signal"] == 0).all()

    def test_anomaly_detection_with_injected_spike(
        self, sample_ohlcv_data: pd.DataFrame, sample_cot_data: pd.DataFrame
    ) -> None:
        """Inject a massive short spike and verify detection."""
        merged = merge_price_and_cot(sample_ohlcv_data, sample_cot_data)
        # Inject anomalous short change
        merged.loc[200:210, "Commercial_Short"] *= 3.0
        merged["Short_Change"] = merged["Commercial_Short"].diff()
        merged["Commercial_Short_Ratio"] = (
            merged["Commercial_Short"] / merged["Open_Interest"].replace(0, np.nan)
        )

        strategy = COTMonitor(config=COTConfig())
        result = strategy.generate_signals(merged)
        # Should detect at least one signal around the spike
        assert result["cot_anomaly_alert"].any()


# ─── Volume-Price Detector ───────────────────────

class TestVolumePriceDetector:
    def test_generate_signals_columns(self, sample_ohlcv_data: pd.DataFrame) -> None:
        strategy = VolumePriceDetector(config=VolumePriceConfig())
        strategy.train_ml_model(sample_ohlcv_data)
        result = strategy.generate_signals(sample_ohlcv_data)

        assert "vp_signal" in result.columns
        assert "vp_signal_reason" in result.columns
        assert "vp_volume_spike" in result.columns
        assert "vp_manipulation_alert" in result.columns
        assert "OBV" in result.columns
        assert "OBV_Bullish_Div" in result.columns
        assert "OBV_Bearish_Div" in result.columns

    def test_signals_are_valid(self, sample_ohlcv_data: pd.DataFrame) -> None:
        strategy = VolumePriceDetector(config=VolumePriceConfig())
        result = strategy.generate_signals(sample_ohlcv_data)
        assert set(result["vp_signal"].unique()).issubset({-1, 0, 1})

    def test_manipulation_detection_with_spike(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Inject a volume spike + price drop and verify detection."""
        df = sample_ohlcv_data.copy()
        idx = 150
        df.loc[idx, "Volume"] = df["Volume"].mean() * 10  # 10x volume
        df.loc[idx, "Close"] = df.loc[idx - 1, "Close"] * 0.95  # 5% drop

        strategy = VolumePriceDetector(config=VolumePriceConfig())
        result = strategy.generate_signals(df)
        assert result.loc[idx, "vp_manipulation_alert"]


# ─── Bank Participation ──────────────────────────

class TestBankParticipationMonitor:
    def test_generate_signals_columns(
        self, sample_ohlcv_data: pd.DataFrame, sample_usd_data: pd.DataFrame
    ) -> None:
        strategy = BankParticipationMonitor(config=BankConfig())
        result = strategy.generate_signals(sample_ohlcv_data, usd_data=sample_usd_data)

        assert "bank_signal" in result.columns
        assert "bank_signal_reason" in result.columns
        assert "Bank_Short_Concentration" in result.columns
        assert "market_regime" in result.columns

    def test_signals_are_valid(
        self, sample_ohlcv_data: pd.DataFrame, sample_usd_data: pd.DataFrame
    ) -> None:
        strategy = BankParticipationMonitor(config=BankConfig())
        result = strategy.generate_signals(sample_ohlcv_data, usd_data=sample_usd_data)
        assert set(result["bank_signal"].unique()).issubset({-1, 0, 1})

    def test_regime_detection(
        self, sample_ohlcv_data: pd.DataFrame, sample_usd_data: pd.DataFrame
    ) -> None:
        strategy = BankParticipationMonitor(config=BankConfig())
        result = strategy.generate_signals(sample_ohlcv_data, usd_data=sample_usd_data)
        assert set(result["market_regime"].unique()).issubset({"TRENDING", "RANGING"})

    def test_without_usd_data(self, sample_ohlcv_data: pd.DataFrame) -> None:
        strategy = BankParticipationMonitor(config=BankConfig())
        result = strategy.generate_signals(sample_ohlcv_data, usd_data=None)
        assert "bank_signal" in result.columns


# ─── AI Predictor ────────────────────────────────

class TestAIPredictor:
    def test_generate_signals_without_training(self, sample_ohlcv_data: pd.DataFrame) -> None:
        strategy = AIPredictor(config=AIConfig())
        result = strategy.generate_signals(sample_ohlcv_data)
        assert "ai_signal" in result.columns
        assert (result["ai_signal"] == 0).all()  # Not trained

    def test_train_and_predict(self, sample_ohlcv_data: pd.DataFrame) -> None:
        strategy = AIPredictor(config=AIConfig(horizons=(1,), max_iter=50))
        strategy.train(sample_ohlcv_data)
        result = strategy.generate_signals(sample_ohlcv_data)

        assert "ai_target_price_1d" in result.columns
        assert "ai_bias" in result.columns
        assert "ai_confidence" in result.columns

    def test_feature_importance_populated(self, sample_ohlcv_data: pd.DataFrame) -> None:
        strategy = AIPredictor(config=AIConfig(horizons=(1,), max_iter=50))
        strategy.train(sample_ohlcv_data)
        assert len(strategy.feature_importance) > 0

    def test_train_metrics_populated(self, sample_ohlcv_data: pd.DataFrame) -> None:
        strategy = AIPredictor(config=AIConfig(horizons=(1,), max_iter=50))
        strategy.train(sample_ohlcv_data)
        assert "r2" in strategy.train_metrics
        assert "mae" in strategy.train_metrics
