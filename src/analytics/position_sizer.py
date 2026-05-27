"""
Position Sizer — src/analytics/position_sizer.py
=================================================
Tính khối lượng mua khuyến nghị dựa trên:
  - Rating từ nightly_scanner (Mid-term 6M + Short-term 1-3M)
  - ATR/Volatility để điều chỉnh tỷ trọng (risk cap)

Design goals:
  - Không over-engineer: Rating → Base Alloc → ATR Cap → Output
  - Fail-safe: nếu thiếu dữ liệu giá → trả về neutral/conservative defaults
  - No external dependencies ngoài pandas/numpy
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Base allocation map (% of portfolio) ─────────────────────────────────────

_BASE_ALLOC: dict[str, dict] = {
    # Mid-term 6M ratings
    "⭐⭐⭐ Strong Buy":  {"base_pct": 35.0, "stop_mult": 1.5},
    "⭐⭐ Watch":         {"base_pct": 15.0, "stop_mult": 1.2},
    "⭐ Neutral":         {"base_pct":  5.0, "stop_mult": 1.0},
    "⚠️ Avoid":          {"base_pct":  0.0, "stop_mult": 1.0},
    # Short-term 1-3M — override to smaller base for speculative plays
    "⭐⭐⭐ Strong Buy (Sóng ngắn)": {"base_pct": 25.0, "stop_mult": 1.5},
    "⭐⭐ Buy (Sóng ngắn)":          {"base_pct": 15.0, "stop_mult": 1.2},
    "Watch (Đang tích lũy)":         {"base_pct":  8.0, "stop_mult": 1.0},
    "Neutral (Trung lập)":           {"base_pct":  0.0, "stop_mult": 1.0},
    "Avoid (Giảm mạnh)":             {"base_pct":  0.0, "stop_mult": 1.0},
}

# ── Volatility risk tiers (annualized %) ─────────────────────────────────────
_VOL_HIGH   = 80.0   # > 80% → rủi ro cao, cắt xuống 50% tỷ trọng
_VOL_MEDIUM = 60.0   # > 60% → rủi ro trung bình, cắt xuống 70% tỷ trọng

# ── Hard caps ─────────────────────────────────────────────────────────────────
_MAX_ALLOC_MIDTERM   = 40.0   # Tuyệt đối ≤ 40% cho mid-term
_MAX_ALLOC_SHORTTERM = 25.0   # Tuyệt đối ≤ 25% cho short-term speculation


def _compute_annualized_vol(df: pd.DataFrame) -> float:
    """Compute 20-day rolling annualized volatility (%)."""
    if df is None or "Close" not in df.columns or len(df) < 21:
        return 50.0  # conservative default
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close) < 21:
        return 50.0
    daily_std = close.pct_change().rolling(20).std().iloc[-1]
    if pd.isna(daily_std) or daily_std <= 0:
        return 50.0
    return float(daily_std * (252 ** 0.5) * 100)


def _compute_atr_pct(df: pd.DataFrame) -> float:
    """Compute ATR(14) as % of current price for stop-loss sizing."""
    if df is None or len(df) < 15:
        return 5.0  # conservative default 5%
    try:
        hi = pd.to_numeric(df["High"], errors="coerce")
        lo = pd.to_numeric(df["Low"],  errors="coerce")
        cl = pd.to_numeric(df["Close"], errors="coerce")
        cl_prev = cl.shift(1)
        tr = pd.concat([
            hi - lo,
            (hi - cl_prev).abs(),
            (lo - cl_prev).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean().iloc[-1]
        current = float(cl.iloc[-1])
        if pd.isna(atr14) or current <= 0:
            return 5.0
        return float(atr14 / current * 100)
    except Exception:
        return 5.0


def compute_position_size(
    symbol: str,
    current_price: float,
    mid_rating: str,
    short_rating: str,
    df_price: pd.DataFrame | None,
    portfolio_value_vnd: float = 1_000_000_000,
) -> dict:
    """
    Compute recommended position size for a stock.

    Parameters
    ----------
    symbol:               Ticker symbol (for output labeling only)
    current_price:        Current market price (VND or TWD)
    mid_rating:           Mid-term rating string from scan_symbol()
    short_rating:         Short-term rating string from scan_symbol()
    df_price:             Price DataFrame (OHLCV) for volatility/ATR calc
    portfolio_value_vnd:  Total investable capital in VND (default 1 billion)

    Returns
    -------
    dict with keys:
        recommended_allocation_pct  : float  (0–40%)
        recommended_vnd_million     : float  (VND in millions)
        recommended_shares          : int    (shares to buy)
        stop_loss_pct               : float  (% below entry)
        risk_level                  : str    ("Low" / "Medium" / "High")
        annualized_vol_pct          : float  (for transparency)
        sizing_basis                : str    ("Mid-term" / "Short-term" / "Avoid")
        notes                       : str    (human-readable explanation)
    """
    ann_vol = _compute_annualized_vol(df_price)
    atr_pct = _compute_atr_pct(df_price)

    # ── Determine sizing basis and base allocation ────────────────────────────
    # Prefer mid-term Strong Buy; fall back to short-term if ST is higher priority
    mid_cfg = _BASE_ALLOC.get(mid_rating, {"base_pct": 0.0, "stop_mult": 1.0})
    st_cfg  = _BASE_ALLOC.get(short_rating, {"base_pct": 0.0, "stop_mult": 1.0})

    if mid_cfg["base_pct"] >= st_cfg["base_pct"]:
        base_pct = mid_cfg["base_pct"]
        stop_mult = mid_cfg["stop_mult"]
        basis = "Mid-term"
        hard_cap = _MAX_ALLOC_MIDTERM
    else:
        base_pct = st_cfg["base_pct"]
        stop_mult = st_cfg["stop_mult"]
        basis = "Short-term"
        hard_cap = _MAX_ALLOC_SHORTTERM

    # ── ATR-based Volatility Risk Cap ─────────────────────────────────────────
    if ann_vol > _VOL_HIGH:
        vol_factor = 0.50   # Rủi ro cao: giảm còn 50%
        risk_level = "High"
        notes = f"Biến động {ann_vol:.0f}% > {_VOL_HIGH}% → giảm tỷ trọng còn 50% mức gốc"
    elif ann_vol > _VOL_MEDIUM:
        vol_factor = 0.70   # Rủi ro trung bình: giảm còn 70%
        risk_level = "Medium"
        notes = f"Biến động {ann_vol:.0f}% > {_VOL_MEDIUM}% → giảm tỷ trọng còn 70% mức gốc"
    else:
        vol_factor = 1.00   # Rủi ro thấp: tỷ trọng đầy đủ
        risk_level = "Low"
        notes = f"Biến động {ann_vol:.0f}% ≤ {_VOL_MEDIUM}% → tỷ trọng đầy đủ"

    # ── Compute final allocation ───────────────────────────────────────────────
    final_pct = min(base_pct * vol_factor, hard_cap)
    if base_pct == 0:
        final_pct = 0.0
        risk_level = "Avoid"
        basis = "Avoid"
        notes = "Rating AVOID/NEUTRAL — không khuyến nghị vào lệnh"

    # ── Stop loss sizing (ATR-based) ──────────────────────────────────────────
    stop_loss_pct = round(atr_pct * stop_mult * 1.5, 2)  # 1.5× ATR as buffer
    stop_loss_pct = max(3.0, min(stop_loss_pct, 15.0))    # clamp 3%–15%

    # ── Monetary output ───────────────────────────────────────────────────────
    recommended_vnd = portfolio_value_vnd * (final_pct / 100.0)
    if current_price > 0:
        recommended_shares = int(recommended_vnd / current_price)
        # Round down to lot size (100 shares for VN)
        lot_size = 100
        recommended_shares = (recommended_shares // lot_size) * lot_size
    else:
        recommended_shares = 0

    recommended_vnd_million = round(recommended_vnd / 1_000_000, 1)

    return {
        "symbol":                    symbol,
        "recommended_allocation_pct": round(final_pct, 1),
        "recommended_vnd_million":    recommended_vnd_million,
        "recommended_shares":         recommended_shares,
        "stop_loss_pct":              stop_loss_pct,
        "risk_level":                 risk_level,
        "annualized_vol_pct":         round(ann_vol, 1),
        "atr_pct":                    round(atr_pct, 2),
        "sizing_basis":               basis,
        "notes":                      notes,
    }
