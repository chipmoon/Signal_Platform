from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pandas as pd

from src.analytics.intraday_plan import (
    MAX_INTRADAY_STOP_PCT,
    MAX_INTRADAY_TARGET_PCT,
    MIN_INTRADAY_STOP_PCT,
    build_intraday_plan,
    select_nearby_zone,
    validate_intraday_bars,
    validate_trade_plan,
)
from src.cache_manager import CacheManager
from src.vn_price import (
    normalize_vn_ohlcv,
    normalize_vn_price_value,
    vn_price_scale_is_consistent,
)


def _h1_frame(last_close: float = 26_400.0, rows: int = 40) -> pd.DataFrame:
    dates = pd.date_range(end=datetime.now().replace(minute=0, second=0, microsecond=0), periods=rows, freq="h")
    close = np.linspace(last_close * 0.985, last_close, rows)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close - 20.0,
            "High": close + 120.0,
            "Low": close - 120.0,
            "Close": close,
            "Volume": np.full(rows, 100_000.0),
        }
    )


def test_vn_prices_are_normalized_to_vnd() -> None:
    assert normalize_vn_price_value(26.4, symbol="BSR") == 26_400.0
    assert normalize_vn_price_value(26_400.0, symbol="BSR") == 26_400.0

    raw = pd.DataFrame(
        {
            "Open": [26.2, 26_300.0],
            "High": [26.5, 26_600.0],
            "Low": [26.0, 26_100.0],
            "Close": [26.4, 26_400.0],
        }
    )
    normalized = normalize_vn_ohlcv(raw, symbol="BSR")
    assert normalized["Close"].tolist() == [26_400.0, 26_400.0]
    assert vn_price_scale_is_consistent(normalized)


def test_vn_cache_boundary_reads_canonical_vnd(monkeypatch) -> None:
    manager = object.__new__(CacheManager)
    manager._meta = {}
    fake_path = Mock()
    fake_path.exists.return_value = True
    manager._price_path = lambda symbol, market: fake_path
    raw = _h1_frame()
    raw[["Open", "High", "Low", "Close"]] /= 1_000.0
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: raw.copy())
    cached = manager.get_cached_price_data("BSR", "VN", max_age_hours=1)
    assert cached is not None
    assert cached["Close"].iloc[-1] == 26_400.0


def test_daily_fallback_is_rejected_as_intraday() -> None:
    frame = _h1_frame()
    frame["Date"] = pd.date_range(end=datetime.now().date(), periods=len(frame), freq="D")
    valid, reason = validate_intraday_bars(frame)
    assert not valid
    assert "Daily fallback" in reason


def test_neutral_bias_never_creates_trade_prices() -> None:
    plan = build_intraday_plan(_h1_frame(), side="neutral")
    assert not plan["actionable"]
    assert "Directional" in plan["reason"]
    assert "entry" not in plan
    assert "stop" not in plan
    assert "target" not in plan


def test_zombie_bsr_order_block_is_ignored_and_stop_is_bounded() -> None:
    frame = _h1_frame()
    zombie = {"top": 16_377.71, "bottom": 15_758.51}
    nearby = {"top": 26_100.0, "bottom": 25_900.0}

    assert select_nearby_zone([zombie], 26_400.0, side="long") is None
    assert select_nearby_zone([zombie, nearby], 26_400.0, side="long") == nearby

    plan = build_intraday_plan(frame, side="long", bull_obs=[zombie])
    assert plan["actionable"], plan["reason"]
    stop_pct = (plan["entry"] - plan["stop"]) / plan["entry"]
    reward_risk = (plan["target"] - plan["entry"]) / (plan["entry"] - plan["stop"])
    assert MIN_INTRADAY_STOP_PCT <= stop_pct <= MAX_INTRADAY_STOP_PCT
    assert reward_risk >= 1.5
    assert plan["target"] <= plan["entry"] * (1.0 + MAX_INTRADAY_TARGET_PCT)
    assert plan["stop"] > 25_000.0


def test_phase0_uses_completed_h1_bars_and_is_not_far() -> None:
    frame = _h1_frame()
    previous_breakout = float(frame["High"].iloc[-5:-1].max())
    frame.loc[frame.index[-1], "High"] = frame["Close"].iloc[-1] * 1.12
    plan = build_intraday_plan(frame, side="long")
    assert plan["actionable"], plan["reason"]
    assert plan["phase0"] == max(float(frame["Close"].iloc[-1]), previous_breakout)
    assert (plan["phase0"] - plan["current"]) / plan["current"] <= 0.03


def test_trade_plan_invariants_reject_old_bsr_stop() -> None:
    valid, reason = validate_trade_plan(
        {
            "side": "long",
            "entry": 26_136.0,
            "stop": 15_303.16,
            "target": 32_296.71,
        }
    )
    assert not valid
    assert "stop outside" in reason
