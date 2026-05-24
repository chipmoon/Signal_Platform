"""
Volume Intelligence Module
==========================
Computes institutional vs retail flow proxies from OHLCV data.
Works for ALL markets (VN, TW, US) — no external API needed.

Key Indicators:
- OBV (On-Balance Volume): Accumulation/Distribution trend
- CMF (Chaikin Money Flow): Smart money pressure
- VWAP Deviation: Price vs volume-weighted fair value
- Block Trade Ratio: Proxy for institutional activity
- Volume Delta: Recent vs historical volume comparison
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def compute_volume_intelligence(df: pd.DataFrame) -> dict:
    """
    Compute full volume intelligence from OHLCV dataframe.

    Args:
        df: DataFrame with columns [Date, Open, High, Low, Close, Volume]

    Returns:
        dict with all volume indicators and human-readable signals
    """
    if df is None or df.empty or len(df) < 10:
        return _empty_result()

    try:
        df = df.copy()
        df = df.sort_values("Date").reset_index(drop=True)

        result = {}

        # ── 1. OBV (On-Balance Volume) ─────────────────────────────────
        result.update(_compute_obv(df))

        # ── 2. CMF (Chaikin Money Flow, 14-day) ───────────────────────
        result.update(_compute_cmf(df, period=14))

        # ── 3. VWAP Deviation ─────────────────────────────────────────
        result.update(_compute_vwap(df, period=20))

        # ── 4. Block Trade Ratio (Institutional Proxy) ────────────────
        result.update(_compute_block_trades(df, lookback=20))

        # ── 5. Volume Delta (Recent vs Historical) ────────────────────
        result.update(_compute_volume_delta(df))

        # ── 6. Composite Smart Money Signal ───────────────────────────
        result["smart_money_signal"] = _composite_signal(result)
        result["smart_money_color"]  = _signal_color(result["smart_money_signal"])

        logger.debug(f"Volume intelligence computed: {result['smart_money_signal']}")
        return result

    except Exception as e:
        logger.warning(f"Volume intelligence error: {e}")
        return _empty_result()


# ── Private Computation Functions ──────────────────────────────────────────────

def _compute_obv(df: pd.DataFrame) -> dict:
    """On-Balance Volume — cumulative buy/sell pressure."""
    close = df["Close"].values
    vol   = df["Volume"].values

    obv = np.zeros(len(df))
    for i in range(1, len(df)):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + vol[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - vol[i]
        else:
            obv[i] = obv[i - 1]

    # Trend: compare last 5 vs previous 5
    obv_recent = np.mean(obv[-5:])
    obv_prev   = np.mean(obv[-10:-5]) if len(obv) >= 10 else obv[0]
    obv_change_pct = ((obv_recent - obv_prev) / (abs(obv_prev) + 1e-9)) * 100

    if obv_change_pct > 5:
        trend = "Tích lũy mạnh ↑↑"
    elif obv_change_pct > 1:
        trend = "Tích lũy ↑"
    elif obv_change_pct < -5:
        trend = "Phân phối mạnh ↓↓"
    elif obv_change_pct < -1:
        trend = "Phân phối ↓"
    else:
        trend = "Trung lập →"

    return {
        "obv_current":    float(obv[-1]),
        "obv_change_pct": round(float(obv_change_pct), 1),
        "obv_trend":      trend,
        "obv_bullish":    obv_change_pct > 0,
    }


def _compute_cmf(df: pd.DataFrame, period: int = 14) -> dict:
    """Chaikin Money Flow — institutional buying/selling pressure."""
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    vol   = df["Volume"].values

    # Money Flow Multiplier
    denom = (high - low)
    denom = np.where(denom == 0, 1e-9, denom)
    mfm   = ((close - low) - (high - close)) / denom

    # Money Flow Volume
    mfv = mfm * vol

    # CMF = sum(MFV, period) / sum(Vol, period)
    n   = min(period, len(df))
    cmf = np.sum(mfv[-n:]) / (np.sum(vol[-n:]) + 1e-9)
    cmf = float(np.clip(cmf, -1.0, 1.0))

    if cmf > 0.20:
        signal = "Mua mạnh"
    elif cmf > 0.05:
        signal = "Mua"
    elif cmf < -0.20:
        signal = "Bán mạnh"
    elif cmf < -0.05:
        signal = "Bán"
    else:
        signal = "Trung lập"

    return {
        "cmf_14":      round(cmf, 3),
        "cmf_signal":  signal,
        "cmf_bullish": cmf > 0,
    }


def _compute_vwap(df: pd.DataFrame, period: int = 20) -> dict:
    """VWAP Deviation — is price above/below fair value?"""
    n     = min(period, len(df))
    sub   = df.tail(n)
    tp    = (sub["High"] + sub["Low"] + sub["Close"]) / 3  # Typical Price
    vol   = sub["Volume"]
    vwap  = (tp * vol).sum() / (vol.sum() + 1e-9)

    current_price = float(df["Close"].iloc[-1])
    deviation_pct = ((current_price - vwap) / vwap) * 100

    if deviation_pct > 3:
        signal = "Đắt so với VWAP"
    elif deviation_pct > 0.5:
        signal = "Trên VWAP"
    elif deviation_pct < -3:
        signal = "Rẻ so với VWAP"
    elif deviation_pct < -0.5:
        signal = "Dưới VWAP"
    else:
        signal = "Tại VWAP"

    return {
        "vwap_20":          round(float(vwap), 2),
        "vwap_deviation":   round(deviation_pct, 2),
        "vwap_signal":      signal,
        "price_above_vwap": deviation_pct >= 0,
    }


def _compute_block_trades(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Block Trade Ratio — proxy for institutional activity.

    Days with volume > 2x avg are likely block/institutional trades.
    If price closed UP on high-volume days → institutional buying.
    If price closed DOWN on high-volume days → institutional selling.
    """
    n      = min(lookback, len(df))
    recent = df.tail(n).copy()
    avg_vol = recent["Volume"].mean()

    block_days    = recent[recent["Volume"] > 2 * avg_vol]
    block_count   = len(block_days)
    block_ratio   = round((block_count / n) * 100, 1)

    if block_count > 0:
        buy_blocks  = (block_days["Close"] >= block_days["Open"]).sum()
        sell_blocks = block_count - buy_blocks
        block_direction = "Tổ chức MUA" if buy_blocks >= sell_blocks else "Tổ chức BÁN"
    else:
        buy_blocks  = 0
        sell_blocks = 0
        block_direction = "Không có block trade"

    # Last 5 days volume vs 20-day average
    recent_5d_avg = df["Volume"].tail(5).mean()
    vol_ratio_5d  = round((recent_5d_avg / (avg_vol + 1e-9)) * 100, 1)

    return {
        "block_ratio_pct":  block_ratio,
        "block_direction":  block_direction,
        "block_buy_days":   int(buy_blocks),
        "block_sell_days":  int(sell_blocks),
        "vol_ratio_5d_pct": vol_ratio_5d,    # e.g. 60% = vol down 40% vs avg
    }


def _compute_volume_delta(df: pd.DataFrame) -> dict:
    """Recent volume trend vs historical."""
    if len(df) < 25:
        return {"volume_delta_signal": "Không đủ data", "volume_trend": "N/A"}

    avg_5d  = df["Volume"].tail(5).mean()
    avg_20d = df["Volume"].tail(20).mean()
    delta   = ((avg_5d - avg_20d) / (avg_20d + 1e-9)) * 100

    if delta > 30:
        signal = "Volume tăng mạnh (+{:.0f}%)".format(delta)
    elif delta > 10:
        signal = "Volume tăng (+{:.0f}%)".format(delta)
    elif delta < -30:
        signal = "Volume giảm mạnh ({:.0f}%)".format(delta)
    elif delta < -10:
        signal = "Volume giảm ({:.0f}%)".format(delta)
    else:
        signal = "Volume ổn định ({:+.0f}%)".format(delta)

    return {
        "volume_delta_pct": round(float(delta), 1),
        "volume_delta_signal": signal,
        "volume_trend": "increasing" if delta > 0 else "decreasing",
    }


def _composite_signal(r: dict) -> str:
    """Combine all signals into a single Smart Money verdict."""
    score = 0

    # OBV: weight 2
    if r.get("obv_bullish"):
        score += 2

    # CMF: weight 3 (most reliable)
    cmf = r.get("cmf_14", 0)
    if cmf > 0.10:
        score += 3
    elif cmf > 0:
        score += 1
    elif cmf < -0.10:
        score -= 3
    elif cmf < 0:
        score -= 1

    # Block trade direction: weight 2
    bd = r.get("block_direction", "")
    if "MUA" in bd:
        score += 2
    elif "BÁN" in bd:
        score -= 2

    # Volume delta: weight 1
    vd = r.get("volume_delta_pct", 0)
    if vd > 10:
        score += 1
    elif vd < -10:
        score -= 1

    # Verdict
    if score >= 5:
        return "🟢 Smart Money TÍCH LŨY MẠNH"
    elif score >= 2:
        return "🟡 Smart Money Tích lũy"
    elif score <= -5:
        return "🔴 Smart Money PHÂN PHỐI MẠNH"
    elif score <= -2:
        return "🟠 Smart Money Phân phối"
    else:
        return "⚪ Smart Money Trung lập"


def _signal_color(signal: str) -> str:
    if "🟢" in signal:
        return "#22c55e"
    elif "🟡" in signal:
        return "#eab308"
    elif "🔴" in signal:
        return "#ef4444"
    elif "🟠" in signal:
        return "#f97316"
    return "#94a3b8"


def _empty_result() -> dict:
    return {
        "obv_trend": "N/A", "obv_change_pct": 0, "obv_bullish": False,
        "cmf_14": 0.0, "cmf_signal": "N/A", "cmf_bullish": False,
        "vwap_20": 0.0, "vwap_deviation": 0.0, "vwap_signal": "N/A",
        "block_ratio_pct": 0.0, "block_direction": "N/A",
        "block_buy_days": 0, "block_sell_days": 0,
        "vol_ratio_5d_pct": 100.0, "volume_delta_pct": 0.0,
        "volume_delta_signal": "N/A", "volume_trend": "N/A",
        "smart_money_signal": "⚪ Không đủ data",
        "smart_money_color": "#94a3b8",
    }
