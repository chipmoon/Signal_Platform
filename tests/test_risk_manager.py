"""
Tests for Risk Manager
======================
Covers ATR, RSI, ADX, position sizing, trailing stop, and drawdown circuit breaker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk_manager import (
    DrawdownCircuitBreaker,
    PositionSizer,
    RiskConfig,
    TrailingStopManager,
    calculate_adx,
    calculate_atr,
    calculate_macd,
    calculate_rsi,
    calculate_var,
)


# ─── ATR ─────────────────────────────────────────

class TestCalculateATR:
    def test_atr_returns_series(self, sample_ohlcv_data: pd.DataFrame) -> None:
        atr = calculate_atr(sample_ohlcv_data, period=14)
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(sample_ohlcv_data)

    def test_atr_positive_values(self, sample_ohlcv_data: pd.DataFrame) -> None:
        atr = calculate_atr(sample_ohlcv_data, period=14)
        # After warmup, all ATR values should be positive
        assert (atr.iloc[14:] > 0).all()

    def test_atr_smaller_period_higher_sensitivity(self, sample_ohlcv_data: pd.DataFrame) -> None:
        atr_short = calculate_atr(sample_ohlcv_data, period=5)
        atr_long = calculate_atr(sample_ohlcv_data, period=50)
        # Short ATR should have higher variance (more sensitive)
        assert atr_short.std() >= atr_long.std() * 0.5


# ─── RSI ─────────────────────────────────────────

class TestCalculateRSI:
    def test_rsi_range(self, sample_ohlcv_data: pd.DataFrame) -> None:
        rsi = calculate_rsi(sample_ohlcv_data["Close"], period=14)
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

    def test_rsi_constant_price_returns_50(self) -> None:
        flat = pd.Series([100.0] * 100)
        rsi = calculate_rsi(flat, period=14)
        assert rsi.iloc[-1] == pytest.approx(50.0, abs=1.0)

    def test_rsi_rising_price_above_50(self) -> None:
        rising = pd.Series(np.linspace(100, 200, 100))
        rsi = calculate_rsi(rising, period=14)
        # Smooth uptrend should push RSI above or equal to 50
        assert rsi.iloc[-1] >= 50


# ─── ADX ─────────────────────────────────────────

class TestCalculateADX:
    def test_adx_returns_series(self, sample_ohlcv_data: pd.DataFrame) -> None:
        adx = calculate_adx(sample_ohlcv_data, period=14)
        assert isinstance(adx, pd.Series)
        assert len(adx) == len(sample_ohlcv_data)

    def test_adx_non_negative(self, sample_ohlcv_data: pd.DataFrame) -> None:
        adx = calculate_adx(sample_ohlcv_data, period=14)
        assert (adx >= 0).all()


# ─── MACD ────────────────────────────────────────

class TestCalculateMACD:
    def test_macd_returns_three_series(self, sample_ohlcv_data: pd.DataFrame) -> None:
        macd, signal, hist = calculate_macd(sample_ohlcv_data["Close"])
        assert len(macd) == len(sample_ohlcv_data)
        assert len(signal) == len(sample_ohlcv_data)
        assert len(hist) == len(sample_ohlcv_data)

    def test_histogram_is_macd_minus_signal(self, sample_ohlcv_data: pd.DataFrame) -> None:
        macd, signal, hist = calculate_macd(sample_ohlcv_data["Close"])
        expected = macd - signal
        pd.testing.assert_series_equal(hist, expected, check_names=False)


# ─── Trailing Stop ───────────────────────────────

class TestTrailingStopManager:
    def test_activate_sets_stop_below_entry(self) -> None:
        mgr = TrailingStopManager(RiskConfig())
        stop = mgr.activate(entry_price=100.0, atr=5.0)
        assert stop < 100.0  # Stop should be below entry
        assert stop == 100.0 - 3.0 * 5.0  # Default multiplier = 3.0

    def test_stop_ratchets_up_only(self) -> None:
        mgr = TrailingStopManager(RiskConfig())
        mgr.activate(entry_price=100.0, atr=5.0)
        initial_stop = mgr.stop_price

        mgr.update(current_price=110.0, atr=5.0)
        higher_stop = mgr.stop_price
        assert higher_stop > initial_stop

        # Price drops — stop should NOT move down
        mgr.update(current_price=105.0, atr=5.0)
        assert mgr.stop_price == higher_stop

    def test_trigger_on_price_at_stop(self) -> None:
        mgr = TrailingStopManager(RiskConfig())
        mgr.activate(entry_price=100.0, atr=5.0) # Stop at 100 - 15 = 85
        assert not mgr.is_triggered(95.0)
        assert mgr.is_triggered(84.0)  # Below stop at 85

    def test_deactivate_resets(self) -> None:
        mgr = TrailingStopManager(RiskConfig())
        mgr.activate(entry_price=100.0, atr=5.0)
        mgr.deactivate()
        assert mgr.stop_price == 0.0
        assert not mgr.is_triggered(50.0)


# ─── Position Sizer ──────────────────────────────

class TestPositionSizer:
    def test_volatility_size_respects_max(self) -> None:
        sizer = PositionSizer(RiskConfig(max_position_pct=0.25))
        shares = sizer.volatility_based_size(
            capital=100_000, entry_price=2000.0, atr=50.0, atr_multiplier=2.0
        )
        max_shares = 100_000 * 0.25 / 2000.0
        assert shares <= max_shares

    def test_volatility_size_zero_atr(self) -> None:
        sizer = PositionSizer(RiskConfig())
        shares = sizer.volatility_based_size(
            capital=100_000, entry_price=2000.0, atr=0.0, atr_multiplier=2.0
        )
        assert shares == 0.0

    def test_kelly_size_positive(self) -> None:
        sizer = PositionSizer(RiskConfig())
        kelly = sizer.kelly_size(
            capital=100_000, win_rate=0.6, avg_win=500.0, avg_loss=300.0
        )
        assert kelly > 0
        assert kelly <= 0.25  # max_position_pct


# ─── Drawdown Circuit Breaker ────────────────────

class TestDrawdownCircuitBreaker:
    def test_halt_on_large_drawdown(self) -> None:
        cb = DrawdownCircuitBreaker(RiskConfig(max_drawdown_pct=0.15))
        cb.reset(100_000)

        # Normal operation
        assert not cb.update(98_000)
        assert not cb.is_halted

        # Big drawdown
        assert cb.update(84_000)  # 16% drawdown > 15%
        assert cb.is_halted

    def test_resume_after_recovery(self) -> None:
        cb = DrawdownCircuitBreaker(
            RiskConfig(max_drawdown_pct=0.15, recovery_threshold=0.05)
        )
        cb.reset(100_000)
        cb.update(84_000)  # Trigger halt
        assert cb.is_halted

        # Recovery — equity back near peak
        cb.update(96_000)  # 4% drawdown < 5% threshold
        assert not cb.is_halted


# ─── VaR ─────────────────────────────────────────

class TestCalculateVaR:
    def test_var_empty_curve(self) -> None:
        result = calculate_var([100.0, 101.0])
        assert result["var_pct"] == 0.0

    def test_var_reasonable_output(self) -> None:
        np.random.seed(42)
        equity = [100_000 + i * 50 + np.random.normal(0, 500) for i in range(252)]
        result = calculate_var(equity)
        assert result["var_pct"] > 0
        assert result["cvar_pct"] > 0
        assert result["cvar_pct"] >= result["var_pct"]  # CVaR >= VaR
