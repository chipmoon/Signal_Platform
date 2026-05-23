"""
Market Regime Detector
=======================
Algorithmic classification of market states: Bull, Bear, and Sideways.
Helps adjust strategy aggression based on trend strength and volatility.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from src.risk_manager import calculate_adx

class RegimeDetector:
    """Classifies market environment based on price structure and momentum."""

    @staticmethod
    def identify(df: pd.DataFrame) -> dict:
        """
        Classify market into 3 main regimes:
        - 🐂 BULL: Price > SMA50 > SMA200 and ADX > 25
        - 🐻 BEAR: Price < SMA50 < SMA200 and ADX > 25
        - ↔️ SIDEWAYS: ADX < 20 (Mean Reversion regime)
        """
        if len(df) < 200:
            return {
                "regime": "UNKNOWN", 
                "icon": "⚪", 
                "adx": 0.0, 
                "status": "Insufficient data (need 200 bars)"
            }

        close = df['Close']
        sma50 = close.rolling(50).mean().iloc[-1]
        sma200 = close.rolling(200).mean().iloc[-1]
        last_price = close.iloc[-1]
        
        # Trend strength (ADX)
        adx_series = calculate_adx(df)
        last_adx = adx_series.iloc[-1]
        
        # Momentum (RSI)
        from src.risk_manager import calculate_rsi
        rsi = calculate_rsi(close, 14).iloc[-1]

        regime = "SIDEWAYS"
        icon = "↔️"
        desc = "Consolidation / Range-bound"

        if last_price > sma50 > sma200:
            if last_adx > 20:
                regime = "BULL"
                icon = "🐂"
                desc = "Strong Uptrend (Markup)"
            else:
                regime = "ACCUMULATION"
                icon = "📦"
                desc = "Potential Accumulation / Re-accumulation"
        elif last_price < sma50 < sma200:
            if last_adx > 20:
                regime = "BEAR"
                icon = "🐻"
                desc = "Strong Downtrend (Markdown)"
            else:
                regime = "DISTRIBUTION"
                icon = "📉"
                desc = "Potential Distribution / Redistribution"
        
        # Overextension check
        is_overextended = False
        if rsi > 75:
            is_overextended = True
            desc += " (⚠️ Overbought / Exhaustion Risk)"
        elif rsi < 25:
            is_overextended = True
            desc += " (⚠️ Oversold / Capitulation Risk)"

        return {
            "regime": regime,
            "icon": icon,
            "description": desc,
            "adx": round(last_adx, 2),
            "rsi": round(rsi, 2),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2),
            "is_overextended": is_overextended,
            "distance_to_sma200": round((last_price - sma200) / sma200 * 100, 2)
        }
