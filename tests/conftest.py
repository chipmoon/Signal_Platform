"""Shared test fixtures for the trading system."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate 252 days (1 year) of realistic OHLCV data for testing."""
    np.random.seed(42)
    n = 252
    dates = pd.date_range(end=datetime.now(), periods=n, freq="B")

    # Random walk price
    base_price = 2000.0
    returns = np.random.normal(0.0003, 0.015, n)
    prices = base_price * np.cumprod(1 + returns)

    high = prices * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_prices = prices * (1 + np.random.normal(0, 0.002, n))
    volume = np.random.randint(50_000, 200_000, n).astype(float)

    return pd.DataFrame({
        "Date": dates,
        "Open": open_prices,
        "High": high,
        "Low": low,
        "Close": prices,
        "Volume": volume,
    })


@pytest.fixture
def sample_cot_data() -> pd.DataFrame:
    """Generate 52 weeks (1 year) of synthetic COT data."""
    np.random.seed(42)
    target_n = 52
    dates = pd.date_range(end=datetime.now(), periods=target_n, freq="W-TUE")
    # Some pandas/python combinations can yield one less anchored period.
    n = len(dates)

    oi = np.random.randint(400_000, 600_000, n).astype(float)
    comm_short = (oi * np.random.uniform(0.20, 0.40, n)).astype(float)
    comm_long = (comm_short * np.random.uniform(0.8, 1.2, n)).astype(float)

    df = pd.DataFrame({
        "Date": dates,
        "Open_Interest": oi,
        "Commercial_Long": comm_long,
        "Commercial_Short": comm_short,
        "NonComm_Long": (oi * 0.20).astype(float),
        "NonComm_Short": (oi * 0.20).astype(float),
    })
    df["Commercial_Net"] = df["Commercial_Long"] - df["Commercial_Short"]
    df["Commercial_Short_Ratio"] = df["Commercial_Short"] / df["Open_Interest"]
    df["Short_Change"] = df["Commercial_Short"].diff()
    df["Short_Change_Pct"] = df["Commercial_Short"].pct_change() * 100
    return df


@pytest.fixture
def sample_usd_data() -> pd.DataFrame:
    """Generate USD index data aligned with OHLCV dates."""
    np.random.seed(123)
    n = 252
    dates = pd.date_range(end=datetime.now(), periods=n, freq="B")
    base = 104.0
    returns = np.random.normal(0.0001, 0.005, n)
    prices = base * np.cumprod(1 + returns)

    return pd.DataFrame({
        "Date": dates,
        "Open": prices * 0.999,
        "High": prices * 1.003,
        "Low": prices * 0.997,
        "Close": prices,
        "Volume": np.random.randint(10_000, 50_000, n).astype(float),
    })
