from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.real_flow_analyzer import RealFlowAnalyzer


def test_real_flow_fail_closed_when_missing_columns(sample_ohlcv_data: pd.DataFrame) -> None:
    df = sample_ohlcv_data.copy()
    result = RealFlowAnalyzer().generate_signals(df, fail_closed=True)
    assert "real_flow_available" in result.columns
    assert not result["real_flow_available"].any()
    assert (result["real_flow_signal"] == 0).all()


def test_real_flow_generates_signal_with_real_columns(sample_ohlcv_data: pd.DataFrame) -> None:
    df = sample_ohlcv_data.copy()
    n = len(df)
    np.random.seed(42)
    df["Foreign_Buy"] = df["Volume"] * np.random.uniform(0.05, 0.20, n)
    df["Foreign_Sell"] = df["Volume"] * np.random.uniform(0.03, 0.18, n)
    df["Block_Trade_Volume"] = df["Volume"] * np.random.uniform(0.05, 0.25, n)
    result = RealFlowAnalyzer().generate_signals(df, fail_closed=True)

    assert result["real_flow_available"].all()
    assert "real_flow_score" in result.columns
    assert set(result["real_flow_signal"].unique()).issubset({-1, 0, 1})
