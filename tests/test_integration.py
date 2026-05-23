"""
Integration Test — Full Pipeline
=================================
End-to-end test: synthetic data → strategies → combiner → backtest → result.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtesting.engine import BacktestEngine
from src.config import (
    AIConfig,
    BacktestConfig,
    BankConfig,
    COTConfig,
    CombinerConfig,
    VolumePriceConfig,
)
from src.data.fetcher import merge_price_and_cot
from src.strategies.ai_predictor import AIPredictor
from src.strategies.bank_participation import BankParticipationMonitor
from src.strategies.cot_monitor import COTMonitor
from src.strategies.volume_price import VolumePriceDetector

# Import combine_signals from the runner
import sys
sys.path.insert(0, ".")
from backtest_runner import combine_signals


class TestFullPipeline:
    """End-to-end integration test for the full trading pipeline."""

    def test_pipeline_produces_result(
        self,
        sample_ohlcv_data: pd.DataFrame,
        sample_cot_data: pd.DataFrame,
        sample_usd_data: pd.DataFrame,
    ) -> None:
        """Run the full pipeline and verify output."""
        # 1. Merge data
        merged = merge_price_and_cot(sample_ohlcv_data, sample_cot_data)

        # 2. Run strategies
        cot = COTMonitor(config=COTConfig())
        merged = cot.generate_signals(merged)

        vp = VolumePriceDetector(config=VolumePriceConfig())
        vp.train_ml_model(merged)
        merged = vp.generate_signals(merged)

        bank = BankParticipationMonitor(config=BankConfig())
        merged = bank.generate_signals(merged, usd_data=sample_usd_data)

        ai = AIPredictor(config=AIConfig(horizons=(1,), max_iter=30))
        ai.train(merged)
        merged = ai.generate_signals(merged)

        # 3. Combine signals
        merged = combine_signals(merged, cfg=CombinerConfig())

        # 4. Verify combined signal column exists
        assert "combined_signal" in merged.columns
        assert "signal_score" in merged.columns
        assert "RSI" in merged.columns
        assert "ADX" in merged.columns
        assert "BB_Squeeze" in merged.columns

        # 5. Run backtest
        engine = BacktestEngine(config=BacktestConfig(initial_capital=100_000))
        result_df = engine.run(merged, signal_column="combined_signal")

        # 6. Verify backtest output
        assert "equity" in result_df.columns
        assert "trade_marker" in result_df.columns
        assert "drawdown_pct" in result_df.columns

        # 7. Verify result metrics
        result = engine.get_result()
        assert result.final_capital > 0
        assert result.num_trades >= 0
        assert 0 <= result.win_rate <= 100
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.sortino_ratio, float)
        assert isinstance(result.calmar_ratio, float)
        assert isinstance(result.hit_rate, float)
        assert isinstance(result.precision_at_entry, float)
        assert isinstance(result.expectancy, float)
        assert isinstance(result.turnover_adjusted_sharpe, float)

    def test_pipeline_no_cot_data(
        self,
        sample_ohlcv_data: pd.DataFrame,
        sample_usd_data: pd.DataFrame,
    ) -> None:
        """Pipeline should work without COT data (graceful degradation)."""
        df = sample_ohlcv_data.copy()

        cot = COTMonitor(config=COTConfig())
        df = cot.generate_signals(df)

        vp = VolumePriceDetector(config=VolumePriceConfig())
        df = vp.generate_signals(df)

        bank = BankParticipationMonitor(config=BankConfig())
        df = bank.generate_signals(df, usd_data=sample_usd_data)

        df = combine_signals(df, cfg=CombinerConfig())

        engine = BacktestEngine(config=BacktestConfig())
        result_df = engine.run(df)
        assert "equity" in result_df.columns

    def test_backtest_result_fields_complete(
        self,
        sample_ohlcv_data: pd.DataFrame,
        sample_cot_data: pd.DataFrame,
        sample_usd_data: pd.DataFrame,
    ) -> None:
        """Verify all BacktestResult fields are populated."""
        merged = merge_price_and_cot(sample_ohlcv_data, sample_cot_data)

        cot = COTMonitor(config=COTConfig())
        merged = cot.generate_signals(merged)

        vp = VolumePriceDetector(config=VolumePriceConfig())
        merged = vp.generate_signals(merged)

        bank = BankParticipationMonitor(config=BankConfig())
        merged = bank.generate_signals(merged, usd_data=sample_usd_data)

        merged = combine_signals(merged, cfg=CombinerConfig())

        engine = BacktestEngine(config=BacktestConfig())
        engine.run(merged)
        result = engine.get_result()

        # Check all fields exist
        assert hasattr(result, "total_return")
        assert hasattr(result, "total_pnl")
        assert hasattr(result, "num_trades")
        assert hasattr(result, "win_rate")
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "sortino_ratio")
        assert hasattr(result, "calmar_ratio")
        assert hasattr(result, "max_drawdown")
        assert hasattr(result, "profit_factor")
        assert hasattr(result, "hit_rate")
        assert hasattr(result, "precision_at_entry")
        assert hasattr(result, "expectancy")
        assert hasattr(result, "turnover")
        assert hasattr(result, "turnover_adjusted_sharpe")
        assert hasattr(result, "trades")
        assert hasattr(result, "equity_curve")
