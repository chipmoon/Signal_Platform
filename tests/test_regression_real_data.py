from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pytest

from backtest_runner import combine_signals
from src.backtesting.engine import BacktestEngine
from src.config import BacktestConfig, CombinerConfig
from src.plugins import registry
from src.strategies.ai_predictor import AIPredictor
from src.strategies.real_flow_analyzer import RealFlowAnalyzer


pytestmark = pytest.mark.real_data


def _requires_real_data() -> bool:
    return os.getenv("RUN_REAL_DATA_TESTS", "0") == "1"


@pytest.mark.skipif(not _requires_real_data(), reason="Set RUN_REAL_DATA_TESTS=1 to enable real-data regression tests.")
@pytest.mark.parametrize(
    "market_id,symbol",
    [
        ("VN", "HAG.VN"),
        ("US", "AAPL"),
        ("COMMODITY", "GC=F"),
    ],
)
def test_real_data_market_regression(market_id: str, symbol: str) -> None:
    provider = registry.get(market_id)
    if provider is None:
        pytest.skip(f"Provider {market_id} not available.")

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
    df = provider.get_price_data(symbol, start_date, end_date)
    if df is None or df.empty:
        pytest.skip(f"No data for {symbol} ({market_id}).")

    # Real-flow layer should be deterministic and fail-closed without synthetic proxies.
    df = RealFlowAnalyzer().generate_signals(df, fail_closed=True)
    ai = AIPredictor()
    ai.train(df)
    df = ai.generate_signals(df)
    df = combine_signals(df, CombinerConfig(require_real_flow=False))

    oos = df.iloc[-252:].copy() if len(df) > 252 else df.copy()
    engine = BacktestEngine(config=BacktestConfig(initial_capital=100_000), market_id=market_id)
    engine.run(oos, signal_column="combined_signal")
    result = engine.get_result()

    metrics = [
        result.total_return,
        result.sharpe_ratio,
        result.sortino_ratio,
        result.calmar_ratio,
        result.max_drawdown,
        result.hit_rate,
        result.precision_at_entry,
        result.turnover_adjusted_sharpe,
    ]
    assert all(np.isfinite(x) for x in metrics)
