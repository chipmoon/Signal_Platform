from __future__ import annotations

import pandas as pd

from src.strategies.quant_money_flow import QuantMoneyFlowAnalyzer


def test_quant_money_flow_columns(sample_ohlcv_data: pd.DataFrame) -> None:
    analyzer = QuantMoneyFlowAnalyzer()
    result = analyzer.generate_signals(sample_ohlcv_data)
    assert "QMF_MFI" in result.columns
    assert "QMF_CMF" in result.columns
    assert "qmf_score" in result.columns
    assert "qmf_signal" in result.columns
    assert "qmf_reason" in result.columns


def test_quant_money_flow_signal_range(sample_ohlcv_data: pd.DataFrame) -> None:
    analyzer = QuantMoneyFlowAnalyzer()
    result = analyzer.generate_signals(sample_ohlcv_data)
    assert set(result["qmf_signal"].dropna().unique()).issubset({-1, 0, 1})
    assert result["qmf_score"].between(-1.0, 1.0).all()

