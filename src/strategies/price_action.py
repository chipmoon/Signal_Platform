import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger

from src.risk_manager import calculate_rsi, calculate_atr
from src.plugins import registry

class PriceActionEngine:
    """
    Price Action & Relative Strength Intelligence Engine.
    Detects market structure, trend strength, and index-relative performance.
    """

    @staticmethod
    def analyze(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Perform complete price action analysis on a price series.
        """
        if df.empty or len(df) < 20:
            return {}

        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        volume = df["Volume"].astype(float)

        analysis = {}

        # 1. Market Structure (HH/LL)
        analysis["structure"] = PriceActionEngine._detect_structure(high, low, close)
        
        # 2. Volume Momentum
        vol_avg = volume.rolling(20).mean().iloc[-1]
        vol_cur = volume.iloc[-1]
        vol_ratio = vol_cur / vol_avg if vol_avg > 0 else 1.0
        analysis["volume_status"] = "⚡ Climax" if vol_ratio > 2.0 else "🟢 Strong" if vol_ratio > 1.2 else "⚪ Normal"
        analysis["volume_ratio"] = vol_ratio

        # 3. RS Score (Relative Strength)
        if benchmark_df is not None and not benchmark_df.empty:
            analysis["rs_score"] = PriceActionEngine._calculate_rs(df, benchmark_df)
        else:
            analysis["rs_score"] = 0.0

        # 4. Trend Strength (Ad-hoc)
        rsi = calculate_rsi(close, period=14).iloc[-1]
        analysis["rsi"] = rsi
        analysis["momentum_bias"] = "Bullish" if rsi > 60 else "Bearish" if rsi < 40 else "Neutral"

        return analysis

    @staticmethod
    def _detect_structure(high: pd.Series, low: pd.Series, close: pd.Series) -> str:
        """Detect HH-HL / LH-LL sequence over last 20 bars."""
        last_5 = close.tail(5)
        last_20_high = high.tail(20).max()
        last_20_low = low.tail(20).min()
        
        current = close.iloc[-1]
        
        if current >= last_20_high * 0.98:
            return "🚀 HH-HL (Bullish)"
        elif current <= last_20_low * 1.02:
            return "📉 LH-LL (Bearish)"
        else:
            return "↔️ Sideways"

    @staticmethod
    def _calculate_rs(df: pd.DataFrame, benchmark_df: pd.DataFrame) -> float:
        """
        Calculate RS Score (Mansfield Relative Strength style).
        Compares asset return vs benchmark return over 6 months window.
        """
        try:
            # Align dates
            df = df.set_index("Date") if "Date" in df.columns else df
            bench = benchmark_df.set_index("Date") if "Date" in benchmark_df.columns else benchmark_df
            
            # Use last 120 trading days (~6 months)
            window = 120
            if len(df) < window or len(bench) < window:
                window = min(len(df), len(bench)) - 1

            asset_ret = (df["Close"].iloc[-1] / df["Close"].iloc[-window]) - 1
            bench_ret = (bench["Close"].iloc[-1] / bench["Close"].iloc[-window]) - 1
            
            # Simple Alpha: Asset Return - Bench Return
            rs_score = (asset_ret - bench_ret) * 100
            return round(rs_score, 2)
        except Exception as e:
            logger.debug(f"RS Calc failed: {e}")
            return 0.0
