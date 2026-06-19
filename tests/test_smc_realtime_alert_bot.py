from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from scripts import smc_realtime_alert_bot as bot


def _ohlcv(dates: pd.DatetimeIndex) -> pd.DataFrame:
    size = len(dates)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": [100.0] * size,
            "High": [102.0] * size,
            "Low": [99.0] * size,
            "Close": [101.0] * size,
            "Volume": [1000.0] * size,
        }
    )


def test_confirmed_candles_drops_forming_bar(monkeypatch):
    now = datetime(2026, 6, 19, 10, 17, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
    monkeypatch.setattr(bot, "_market_now", lambda market: now)
    df = _ohlcv(pd.to_datetime(["2026-06-19 09:45", "2026-06-19 10:00", "2026-06-19 10:15"]))

    result = bot._confirmed_candles(df, "VN", "15m")

    assert result["Date"].tolist() == [pd.Timestamp("2026-06-19 09:45"), pd.Timestamp("2026-06-19 10:00")]


def test_confirmed_candles_rejects_daily_cache():
    df = _ohlcv(pd.date_range("2026-06-01", periods=45, freq="D"))

    assert bot._confirmed_candles(df, "VN", "15m").empty


def test_scan_requires_tactical_even_when_quality_is_high(monkeypatch):
    now = datetime(2026, 6, 19, 14, 5, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
    dates = pd.date_range("2026-06-18 09:00", "2026-06-19 13:45", freq="15min")
    frame = _ohlcv(dates)

    class Provider:
        def get_price_data(self, symbol, start, end, interval):
            assert interval == "15m"
            return frame

    class Analyzer:
        tactical = 0

        def __init__(self, config):
            pass

        def generate_signals(self, data):
            result = data.copy()
            result["smc_entry_tactical_signal"] = self.tactical
            result["smc_entry_quality"] = 10
            return result

    monkeypatch.setattr(bot, "_market_now", lambda market: now)
    monkeypatch.setattr(bot.registry, "get", lambda market: Provider())
    monkeypatch.setattr(bot, "SmcAnalyzer", Analyzer)

    _, _, signal = bot._scan_symbol("BSR.VN", 45, "15m")
    assert signal is None

    Analyzer.tactical = 1
    _, _, signal = bot._scan_symbol("BSR.VN", 45, "15m")
    assert signal is not None
