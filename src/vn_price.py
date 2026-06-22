"""Canonical VND/share normalization for Vietnamese equity market data."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

VN_PRICE_COLUMNS = ("Open", "High", "Low", "Close")
_THOUSAND_VND_CEILING = 1_000.0
_MIXED_SCALE_RATIO = 100.0


def normalize_vn_price_value(value: float, *, symbol: str = "") -> float:
    """Return one VN equity price in canonical VND/share."""
    price = float(value)
    if not np.isfinite(price) or price <= 0:
        return price
    if str(symbol).upper().replace(".VN", "") == "VNINDEX":
        return price
    return price * 1_000.0 if price < _THOUSAND_VND_CEILING else price


def normalize_vn_ohlcv(
    df: pd.DataFrame,
    *,
    symbol: str = "",
    price_columns: Iterable[str] = VN_PRICE_COLUMNS,
) -> pd.DataFrame:
    """Normalize VN OHLC columns to VND, including mixed-scale cache segments."""
    if df is None or df.empty:
        return df
    if str(symbol).upper().replace(".VN", "") == "VNINDEX":
        return df.copy()
    columns = [column for column in price_columns if column in df.columns]
    if "Close" not in columns:
        return df.copy()
    result = df.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    close = result["Close"]
    positive = close[(close > 0) & np.isfinite(close)]
    if positive.empty:
        return result
    median = float(positive.median())
    if median < _THOUSAND_VND_CEILING:
        row_mask = close.between(0, _THOUSAND_VND_CEILING, inclusive="neither")
    else:
        row_mask = close.between(
            0,
            min(_THOUSAND_VND_CEILING, median / _MIXED_SCALE_RATIO),
            inclusive="neither",
        )
    if row_mask.any():
        result.loc[row_mask, columns] = result.loc[row_mask, columns] * 1_000.0
    return result


def vn_price_scale_is_consistent(df: pd.DataFrame) -> bool:
    """Return False when adjacent closes contain a likely 1,000x unit jump."""
    if df is None or df.empty or "Close" not in df.columns:
        return False
    close = pd.to_numeric(df["Close"], errors="coerce")
    ratio = close / close.shift(1)
    finite = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    return bool(not finite.empty and finite.between(0.01, 100.0).all())
