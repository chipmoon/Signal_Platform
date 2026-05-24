"""
Multi-Timeframe Confluence Engine
===================================
Analyzes SMC + Wyckoff across 3 timeframes (Weekly / Daily / 4H)
and produces a confluence score and alignment summary.

Philosophy (Karpathy-style):
    - Run existing analyzers on resampled OHLCV data
    - NO new algorithms — just orchestration across timeframes
    - Output: dict with per-TF bias + overall confluence strength
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from src.config import SmcConfig, WyckoffConfig
from src.strategies.smc_analyzer import SmcAnalyzer
from src.strategies.wyckoff_analyzer import WyckoffAnalyzer


# ─── Timeframe Definitions ────────────────────────────────────────────────────

TIMEFRAMES = {
    "Weekly":  {"yf_interval": "1wk",  "yf_period": "2y",  "min_bars": 30},
    "Daily":   {"yf_interval": "1d",   "yf_period": "1y",  "min_bars": 80},
    "4H":      {"yf_interval": "4h",   "yf_period": "60d", "min_bars": 40},
}


# ─── Helper: Normalize column names from yfinance ─────────────────────────────

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has standard OHLCV + Date columns."""
    if df.empty:
        return df
    df = df.copy()
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    # Rename to standard
    col_map = {c: c.capitalize() for c in df.columns}
    col_map.update({"Adj close": "Close", "Adj Close": "Close"})
    df = df.rename(columns=col_map)
    # Reset index to get Date as column
    if df.index.name in ("Date", "Datetime", "date", "datetime", None):
        df = df.reset_index()
    # Standardize Date column
    date_col = next((c for c in df.columns if "date" in c.lower() or "datetime" in c.lower()), None)
    if date_col and date_col != "Date":
        df = df.rename(columns={date_col: "Date"})
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


def _to_yf_ticker(symbol: str, market: str) -> str:
    """Convert internal symbol to Yahoo Finance ticker."""
    if market == "VN":
        return f"{symbol}.VN"
    if market in ("TW",):
        # Already has .TW or .TWO
        return symbol
    return symbol  # US stocks: symbol as-is


# ─── Fetch OHLCV for a given timeframe ────────────────────────────────────────

def _fetch_tf(yf_ticker: str, interval: str, period: str, symbol: str = "", market: str = "") -> pd.DataFrame:
    """Fetch OHLCV from Yahoo Finance for a given interval/period, with offline/cache fallback."""
    # Standardize symbol & market if not provided
    if not symbol and ".VN" in yf_ticker:
        symbol = yf_ticker.replace(".VN", "")
        market = "VN"

    try:
        ticker = yf.Ticker(yf_ticker)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if not df.empty:
            return _normalize_df(df)
    except Exception as e:
        logger.warning(f"MTF fetch failed [{yf_ticker} {interval}]: {e}")

    # Fallback for VN stocks when yfinance is blocked/fails
    if market == "VN" or ".VN" in yf_ticker:
        clean_sym = symbol or yf_ticker.replace(".VN", "")
        from pathlib import Path
        cache_path = Path(__file__).resolve().parents[2] / ".cache" / "prices" / f"{clean_sym}_VN.parquet"
        if cache_path.exists():
            try:
                daily_df = pd.read_parquet(cache_path, engine="pyarrow")
                if not daily_df.empty:
                    daily_df = _normalize_df(daily_df)
                    
                    if interval == "1wk":
                        # Resample daily to weekly
                        daily_df["Date"] = pd.to_datetime(daily_df["Date"])
                        daily_df = daily_df.sort_values("Date")
                        daily_df.set_index("Date", inplace=True)
                        weekly_df = daily_df.resample("W").agg({
                            "Open": "first",
                            "High": "max",
                            "Low": "min",
                            "Close": "last",
                            "Volume": "sum"
                        }).dropna().reset_index()
                        logger.info(f"MTF: Resampled {clean_sym} daily cache to weekly ({len(weekly_df)} rows)")
                        return weekly_df
                        
                    elif interval == "4h":
                        # No intraday cache, so we use daily data as a proxy fallback
                        logger.info(f"MTF: Using {clean_sym} daily cache as 4H proxy")
                        return daily_df
                        
                    elif interval == "1d":
                        return daily_df
            except Exception as ex:
                logger.error(f"Failed to load daily cache fallback for {clean_sym}: {ex}")

    return pd.DataFrame()


# ─── Analyze one timeframe ────────────────────────────────────────────────────

def _analyze_tf(df: pd.DataFrame, tf_name: str) -> dict:
    """
    Run SMC + Wyckoff on a single timeframe DataFrame.
    Returns a dict with bias, structure, wyckoff_phase, score.
    """
    required = {"Open", "High", "Low", "Close"}
    if df.empty or not required.issubset(df.columns):
        return {"tf": tf_name, "bias": "Unknown", "structure": "Unknown",
                "wyckoff_phase": "Unknown", "smc_score": 0.0, "confidence": 0}

    try:
        smc = SmcAnalyzer(SmcConfig())
        smc_state = smc.get_current_state(df)
        structure = smc_state.get("structure", "Ranging")
        smc_score = smc_state.get("smc_score", 0.0)
        signal = smc_state.get("signal", 0)
    except Exception as e:
        logger.debug(f"SMC failed on {tf_name}: {e}")
        structure, smc_score, signal = "Unknown", 0.0, 0

    try:
        wa = WyckoffAnalyzer(WyckoffConfig())
        wy_state = wa.analyze_current_state(df)
        wyckoff_phase = wy_state.get("phase", "Unknown")
        wyckoff_bias = wy_state.get("bias", "Neutral")
    except Exception as e:
        logger.debug(f"Wyckoff failed on {tf_name}: {e}")
        wyckoff_phase, wyckoff_bias = "Unknown", "Neutral"

    # Compute directional bias: +1 Bullish, -1 Bearish, 0 Neutral
    bull_score = 0
    if structure == "Bullish":
        bull_score += 1
    elif structure == "Bearish":
        bull_score -= 1
    if signal == 1:
        bull_score += 1
    elif signal == -1:
        bull_score -= 1
    if wyckoff_bias in ("Bullish", "Markup"):
        bull_score += 1
    elif wyckoff_bias in ("Bearish", "Markdown"):
        bull_score -= 1

    if bull_score >= 2:
        bias = "Bullish"
    elif bull_score <= -2:
        bias = "Bearish"
    elif bull_score == 1:
        bias = "Lean Bullish"
    elif bull_score == -1:
        bias = "Lean Bearish"
    else:
        bias = "Neutral"

    return {
        "tf": tf_name,
        "bias": bias,
        "structure": structure,
        "wyckoff_phase": wyckoff_phase,
        "wyckoff_bias": wyckoff_bias,
        "smc_score": round(smc_score, 3),
        "signal": signal,
        "bull_score": bull_score,
        "bull_obs": smc_state.get("bull_obs", []),
        "bear_obs": smc_state.get("bear_obs", []),
    }


# ─── Main: Multi-Timeframe Confluence ─────────────────────────────────────────

def compute_mtf_confluence(
    symbol: str,
    market: str,
    df_daily: pd.DataFrame | None = None,
) -> dict:
    """
    Compute Multi-Timeframe Confluence across Weekly / Daily / 4H.

    Args:
        symbol:   Internal symbol (e.g. 'BSR', '8096.TWO', 'NVDA')
        market:   Market ID ('VN', 'TW', 'US')
        df_daily: Optional pre-fetched daily DataFrame (reused to avoid extra API call)

    Returns:
        {
          'weekly': {...},   # TF analysis
          'daily': {...},
          '4h': {...},
          'confluence_score': float -3 to +3,
          'confluence_label': str,   # 'Strong Bullish', 'Neutral', etc.
          'alignment': int,          # 0-3: how many TFs agree
          'summary': str,
        }
    """
    yf_ticker = _to_yf_ticker(symbol, market)
    results = {}

    # ── Weekly ──────────────────────────────────────────
    df_weekly = _fetch_tf(yf_ticker, "1wk", "2y", symbol=symbol, market=market)
    results["Weekly"] = _analyze_tf(df_weekly, "Weekly")

    # ── Daily (reuse if provided) ────────────────────────
    if df_daily is not None and not df_daily.empty:
        df_d = _normalize_df(df_daily.copy())
        results["Daily"] = _analyze_tf(df_d, "Daily")
    else:
        df_d = _fetch_tf(yf_ticker, "1d", "1y", symbol=symbol, market=market)
        results["Daily"] = _analyze_tf(df_d, "Daily")

    # ── 4H ──────────────────────────────────────────────
    df_4h = _fetch_tf(yf_ticker, "4h", "60d", symbol=symbol, market=market)
    results["4H"] = _analyze_tf(df_4h, "4H")

    # ── Confluence Calculation ───────────────────────────
    scores = [results[tf]["bull_score"] for tf in ["Weekly", "Daily", "4H"]]
    confluence_score = sum(scores)  # range: -9 to +9 (3 TFs × max ±3 each)

    # Count aligned TFs (all same direction)
    bullish_tfs = sum(1 for s in scores if s > 0)
    bearish_tfs = sum(1 for s in scores if s < 0)
    alignment = max(bullish_tfs, bearish_tfs)  # 0-3

    if confluence_score >= 5:
        confluence_label = "🟢 Strong Bullish"
    elif confluence_score >= 2:
        confluence_label = "🟢 Lean Bullish"
    elif confluence_score <= -5:
        confluence_label = "🔴 Strong Bearish"
    elif confluence_score <= -2:
        confluence_label = "🔴 Lean Bearish"
    else:
        confluence_label = "⚪ Neutral / Mixed"

    # ── Summary Text ────────────────────────────────────
    lines = []
    for tf in ["Weekly", "Daily", "4H"]:
        r = results[tf]
        icon = "🟢" if r["bias"] in ("Bullish", "Lean Bullish") else (
               "🔴" if r["bias"] in ("Bearish", "Lean Bearish") else "⚪")
        lines.append(f"{icon} **{tf}**: {r['bias']} | Structure: {r['structure']} | Wyckoff: {r['wyckoff_phase']}")

    agreement = ""
    if alignment == 3:
        agreement = "✅ **All 3 timeframes aligned** — High Conviction Setup"
    elif alignment == 2:
        agreement = "⚠️ **2/3 timeframes aligned** — Moderate Conviction"
    else:
        agreement = "❌ **Timeframes conflicting** — Wait for clarity"

    return {
        "weekly": results["Weekly"],
        "daily": results["Daily"],
        "4h": results["4H"],
        "confluence_score": confluence_score,
        "confluence_label": confluence_label,
        "alignment": alignment,
        "tf_lines": lines,
        "agreement": agreement,
    }
