from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.smc_radar import (
    build_execution_gate,
    build_smc_entry_candidates,
)
from src.config import SmcConfig
from src.strategies.smc_analyzer import detect_fair_value_gaps
from src.vn_price import canonicalize_vn_daily_bars


def test_vn_daily_canonicalization_deduplicates_and_drops_holidays():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2026-04-29 00:00",
                    "2026-04-29 07:00",
                    "2026-04-30 00:00",
                    "2026-05-04 07:00",
                ]
            ),
            "Open": [23200, 23200, 23600, 25250],
            "High": [23900, 23900, 23600, 25250],
            "Low": [23150, 23150, 23600, 24750],
            "Close": [23600, 23600, 23600, 25250],
            "Volume": [9691400, 9691400, 0, 8112800],
        }
    )

    clean = canonicalize_vn_daily_bars(df, symbol="BSR")

    assert clean["Date"].tolist() == [
        pd.Timestamp("2026-04-29"),
        pd.Timestamp("2026-05-04"),
    ]
    assert clean["Date"].dt.normalize().is_unique
    assert (clean["Volume"] > 0).all()


def test_fvg_fill_uses_deepest_later_retest():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=6, freq="D"),
            "Open": [99, 101, 111, 110, 106, 108],
            "High": [100, 103, 113, 112, 109, 110],
            "Low": [98, 100, 110, 109, 104, 107],
            "Close": [99, 102, 112, 110, 106, 109],
            "Volume": [1_000_000] * 6,
        }
    )

    bullish, _ = detect_fair_value_gaps(df, SmcConfig())
    target = next(gap for gap in bullish if gap.bottom == 100 and gap.top == 110)

    assert target.filled_pct == pytest.approx(0.60)


def _bsr_like_frame() -> pd.DataFrame:
    close = np.linspace(24500.0, 25550.0, 60)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2026-03-25", periods=60, freq="B"),
            "Open": close - 50,
            "High": close + 500,
            "Low": close - 500,
            "Close": close,
            "Volume": np.full(60, 8_000_000.0),
        }
    )


def _bsr_state() -> dict:
    return {
        "current_price": 25500.0,  # intentionally stale; decision price wins
        "signal": 1,
        "bos_bull": False,
        "sweep_bull": True,
        "idm_bull": False,
        "stoch": {
            "k": 12.0,
            "status": "OVERSOLD",
            "crossover": True,
        },
        "bull_fvgs": [
            {
                "bottom": 23600.0,
                "top": 25500.0,
                "filled_pct": 0.63,
                "candle_idx": 45,
            },
            {
                "bottom": 25250.0,
                "top": 26100.0,
                "filled_pct": 0.76,
                "candle_idx": 50,
            },
        ],
        "bull_obs": [],
    }


def test_radar_filters_wide_bsr_holiday_zone_and_uses_decision_price():
    candidates = build_smc_entry_candidates(
        _bsr_like_frame(),
        _bsr_state(),
        {"phase": "PHASE C"},
        decision_price=25550.0,
        execution_gate={},
    )

    assert candidates
    assert all(candidate["entry_low"] != 23600.0 for candidate in candidates)
    assert candidates[0]["source_low"] == 25250.0
    assert candidates[0]["entry_low"] > candidates[0]["source_low"]
    assert candidates[0]["inside"] is True
    assert candidates[0]["status"] == "ZONE_TOUCH"


def test_buy_trigger_requires_mtf_h1_and_bullish_trigger():
    state = _bsr_state()
    gate = build_execution_gate(
        {1: {"bias": "BULLISH"}},
        {"available": True, "actionable": True, "health": "BULLISH"},
        state,
    )
    candidates = build_smc_entry_candidates(
        _bsr_like_frame(),
        state,
        {"phase": "PHASE C"},
        decision_price=25550.0,
        execution_gate=gate,
    )

    assert gate["buy_triggered"] is True
    assert candidates[0]["status"] == "BUY_TRIGGERED"
