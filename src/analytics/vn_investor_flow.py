"""
VN Investor Flow Module
=======================
Extracts buy/sell pressure data for Vietnamese stocks using vnstock.

Data Sources (in priority order):
1. vnstock price_depth (VCI) — bid/ask depth → supply/demand ratio
2. vnstock intraday (VCI) — tick-level buy/sell classification
3. Derived fallback from OHLCV if above blocked on cloud

Graceful degradation: always returns a result dict, never raises.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import pandas as pd
from loguru import logger


def get_vn_flow_intel(symbol: str, df_ohlcv: Optional[pd.DataFrame] = None) -> dict:
    """
    Get VN investor flow intelligence for a symbol.

    Tries vnstock sources first, falls back to OHLCV-derived estimates.

    Args:
        symbol: Clean VN symbol (e.g., 'BSR', 'VCB') — no .VN suffix
        df_ohlcv: Optional OHLCV dataframe for fallback computation

    Returns:
        dict with buy_pressure, sell_pressure, net_flow, etc.
    """
    clean = symbol.replace(".VN", "").replace(".TW", "").upper()

    # Try price_depth first (fastest, most reliable)
    result = _from_price_depth(clean)
    if result:
        return result

    # Try intraday tick data
    result = _from_intraday(clean)
    if result:
        return result

    # Final fallback: derive from OHLCV
    if df_ohlcv is not None and not df_ohlcv.empty:
        return _from_ohlcv_fallback(df_ohlcv)

    return _empty_result()


# ── Private Sources ─────────────────────────────────────────────────────────

def _from_price_depth(symbol: str) -> Optional[dict]:
    """
    Use vnstock price_depth to get bid/ask volume imbalance.
    BID volume >> ASK volume → buying pressure (smart money accumulating).
    """
    try:
        from vnstock import Vnstock
        stock = Vnstock().stock(symbol=symbol, source="VCI")
        depth = stock.quote.price_depth()

        if depth is None or depth.empty:
            return None

        # price_depth columns: bid_price, bid_vol, ask_price, ask_vol (top 3)
        bid_cols = [c for c in depth.columns if "bid" in c.lower() and "vol" in c.lower()]
        ask_cols = [c for c in depth.columns if "ask" in c.lower() and "vol" in c.lower()]

        if not bid_cols or not ask_cols:
            return None

        total_bid = float(depth[bid_cols].sum().sum())
        total_ask = float(depth[ask_cols].sum().sum())
        total     = total_bid + total_ask

        if total < 1:
            return None

        buy_pct  = round((total_bid / total) * 100, 1)
        sell_pct = round((total_ask / total) * 100, 1)
        imbalance = buy_pct - sell_pct  # +20 = strong buy pressure

        if imbalance > 15:
            net_flow = "Cầu vượt cung — Tích lũy"
        elif imbalance > 5:
            net_flow = "Nghiêng về mua"
        elif imbalance < -15:
            net_flow = "Cung vượt cầu — Phân phối"
        elif imbalance < -5:
            net_flow = "Nghiêng về bán"
        else:
            net_flow = "Cân bằng cung cầu"

        logger.debug(f"price_depth {symbol}: bid={buy_pct}% ask={sell_pct}%")
        return {
            "source":       "price_depth",
            "buy_pressure": buy_pct,
            "sell_pressure": sell_pct,
            "bid_vol":      int(total_bid),
            "ask_vol":      int(total_ask),
            "net_flow":     net_flow,
            "imbalance":    round(imbalance, 1),
            "retail_signal": _retail_signal(buy_pct, sell_pct),
        }
    except Exception as e:
        logger.debug(f"price_depth failed for {symbol}: {e}")
        return None


def _from_intraday(symbol: str) -> Optional[dict]:
    """
    Use vnstock intraday ticks to count buy-initiated vs sell-initiated trades.
    Aggregates last 100 ticks.
    """
    try:
        from vnstock import Vnstock
        stock  = Vnstock().stock(symbol=symbol, source="VCI")
        intra  = stock.quote.intraday(show_log=False)

        if intra is None or intra.empty:
            return None

        # Expected columns: time, price, volume, buy_sell_type (or 'match_type')
        type_col = None
        for c in ["buy_sell_type", "match_type", "type", "side"]:
            if c in intra.columns:
                type_col = c
                break

        if not type_col:
            return None

        recent = intra.tail(200)
        buy_vol  = recent[recent[type_col].str.upper().isin(["BU", "B", "BUY", "UP"])]["volume"].sum()
        sell_vol = recent[recent[type_col].str.upper().isin(["SD", "S", "SELL", "DOWN"])]["volume"].sum()
        total    = buy_vol + sell_vol

        if total < 1:
            return None

        buy_pct  = round((buy_vol / total) * 100, 1)
        sell_pct = round((sell_vol / total) * 100, 1)

        if buy_pct > sell_pct + 10:
            net_flow = "Lực mua chủ động cao"
        elif sell_pct > buy_pct + 10:
            net_flow = "Lực bán chủ động cao"
        else:
            net_flow = "Cân bằng lực mua/bán"

        logger.debug(f"intraday {symbol}: buy={buy_pct}% sell={sell_pct}%")
        return {
            "source":        "intraday",
            "buy_pressure":  buy_pct,
            "sell_pressure": sell_pct,
            "buy_vol":       int(buy_vol),
            "sell_vol":      int(sell_vol),
            "net_flow":      net_flow,
            "imbalance":     round(buy_pct - sell_pct, 1),
            "retail_signal": _retail_signal(buy_pct, sell_pct),
        }
    except Exception as e:
        logger.debug(f"intraday failed for {symbol}: {e}")
        return None


def _from_ohlcv_fallback(df: pd.DataFrame) -> dict:
    """
    Estimate buy/sell pressure from OHLCV candle body analysis.
    Close near High → buyers won the day (buy pressure).
    Close near Low  → sellers won (sell pressure).
    """
    recent = df.tail(10).copy()
    buy_pressure_days  = 0
    sell_pressure_days = 0

    for _, row in recent.iterrows():
        rng = row["High"] - row["Low"]
        if rng < 1e-9:
            continue
        close_pos = (row["Close"] - row["Low"]) / rng  # 0=low, 1=high
        if close_pos > 0.6:
            buy_pressure_days += 1
        elif close_pos < 0.4:
            sell_pressure_days += 1

    n        = len(recent)
    buy_pct  = round((buy_pressure_days / n) * 100, 1)
    sell_pct = round((sell_pressure_days / n) * 100, 1)

    if buy_pressure_days > sell_pressure_days + 2:
        net_flow = "Người mua kiểm soát (ước tính)"
    elif sell_pressure_days > buy_pressure_days + 2:
        net_flow = "Người bán kiểm soát (ước tính)"
    else:
        net_flow = "Cân bằng (ước tính từ giá)"

    return {
        "source":        "ohlcv_estimate",
        "buy_pressure":  buy_pct,
        "sell_pressure": sell_pct,
        "net_flow":      net_flow,
        "imbalance":     round(buy_pct - sell_pct, 1),
        "retail_signal": _retail_signal(buy_pct, sell_pct),
    }


def _retail_signal(buy_pct: float, sell_pct: float) -> str:
    """
    Interpret buy/sell imbalance in the context of retail vs smart money.
    Low overall volume + high sell% → retail panic selling (contrarian bullish).
    High overall volume + high buy% → institutional accumulation.
    """
    imbalance = buy_pct - sell_pct
    if imbalance > 20:
        return "Lực mua áp đảo — Cầu thể hiện rõ"
    elif imbalance > 8:
        return "Nghiêng về mua tích lũy"
    elif imbalance < -20:
        return "Lực bán áp đảo — Cung thể hiện rõ"
    elif imbalance < -8:
        return "Nghiêng về bán phân phối"
    return "Cân bằng cung cầu"


def _empty_result() -> dict:
    return {
        "source":        "unavailable",
        "buy_pressure":  50.0,
        "sell_pressure": 50.0,
        "net_flow":      "Không có dữ liệu flow",
        "imbalance":     0.0,
        "retail_signal": "N/A",
    }
