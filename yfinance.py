"""Compatibility layer for yfinance.

Behavior:
1) If the real `yfinance` package exists in site-packages, delegate to it.
2) Otherwise provide a deterministic synthetic fallback used for tests/dev
   in constrained environments.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import importlib.machinery
import importlib.util
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd


def _load_real_yfinance() -> Any | None:
    here = pathlib.Path(__file__).resolve()
    here_dir = str(here.parent.resolve()).lower()
    search_paths = []
    for p in sys.path:
        try:
            rp = str(pathlib.Path(p).resolve()).lower()
        except Exception:
            rp = str(p).lower()
        if rp != here_dir:
            search_paths.append(p)

    spec = importlib.machinery.PathFinder.find_spec("yfinance", search_paths)
    if spec is None or spec.loader is None:
        return None
    origin = str(getattr(spec, "origin", "") or "").lower()
    if origin.endswith(str(here).lower()):
        return None

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


_real = _load_real_yfinance()

if _real is not None:
    globals().update(_real.__dict__)
else:
    def _parse_period(period: str) -> int:
        period = (period or "1mo").lower().strip()
        if period.endswith("d"):
            return max(int(period[:-1] or "1"), 2)
        if period.endswith("mo"):
            return max(int(period[:-2] or "1") * 21, 5)
        if period.endswith("y"):
            return max(int(period[:-1] or "1") * 252, 10)
        return 30


    def _make_ohlcv(
        *,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
        period: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        if start and end:
            try:
                idx = pd.date_range(start=start, end=end, freq="B")
            except Exception:
                idx = pd.date_range(end=datetime.now(), periods=30, freq="B")
        else:
            days = _parse_period(period or "1mo")
            idx = pd.date_range(end=datetime.now(), periods=days, freq="B")

        if len(idx) < 2:
            idx = pd.date_range(end=datetime.now(), periods=2, freq="B")

        seed = abs(hash((ticker, str(idx[0]), str(idx[-1]), interval))) % (2**32)
        rng = np.random.default_rng(seed)
        base = float(50 + (abs(hash(ticker)) % 300))
        rets = rng.normal(0.0002, 0.01, len(idx))
        close = base * np.cumprod(1 + rets)
        high = close * (1 + np.abs(rng.normal(0, 0.003, len(idx))))
        low = close * (1 - np.abs(rng.normal(0, 0.003, len(idx))))
        open_ = close * (1 + rng.normal(0, 0.0015, len(idx)))
        vol = rng.integers(50_000, 500_000, len(idx)).astype(float)

        df = pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": vol,
            },
            index=idx,
        )
        df.index.name = "Date"
        return df


    def download(
        tickers: str,
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
        period: str | None = None,
        progress: bool = False,
        auto_adjust: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        return _make_ohlcv(
            ticker=tickers,
            start=start,
            end=end,
            period=period,
            interval=interval,
        )


    class Ticker:
        def __init__(self, ticker: str):
            self.ticker = ticker
            self.info = {
                "shortName": ticker,
                "longName": ticker,
                "sector": "Unknown",
                "exchange": "SIM",
                "currency": "USD",
            }
            self.fast_info = {
                "lastPrice": float(100 + (abs(hash(ticker)) % 100)),
                "regularMarketPreviousClose": float(99 + (abs(hash(ticker)) % 100)),
                "dayHigh": float(101 + (abs(hash(ticker)) % 100)),
                "dayLow": float(98 + (abs(hash(ticker)) % 100)),
                "volume": float(100_000),
            }
            self.news = []

        def history(
            self,
            start: str | None = None,
            end: str | None = None,
            interval: str = "1d",
            period: str = "1mo",
            auto_adjust: bool = False,
            **kwargs: Any,
        ) -> pd.DataFrame:
            return _make_ohlcv(
                ticker=self.ticker,
                start=start,
                end=end,
                period=period,
                interval=interval,
            )


    def set_tz_cache_location(path: str) -> None:
        return None

