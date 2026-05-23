"""
Crash Shield Module
===================
Algorithmic cross-market panic detection for VN, TW, US, and Crypto.
Provides risk level (0-3) to protect portfolios from systemic crashes.
Cost: $0 (using free yfinance data).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime, timedelta

try:
    import yfinance as yf
except ImportError:
    yf = None

class CrashShield:
    """Inter-market panic detection system."""
    
    # Global indices and assets
    MARKETS = {
        "VN": "^VNINDEX",   # Vietnam
        "TW": "^TWII",      # Taiwan
        "US": "^GSPC",      # S&P 500
        "NAS": "^IXIC",     # Nasdaq
        "BTC": "BTC-USD",   # Crypto
        "GOLD": "GC=F",     # Safe Haven
        "VIX": "^VIX",      # Fear Index
        "DXY": "DX-Y.NYB"   # Dollar Index
    }

    @staticmethod
    def get_market_data() -> dict:
        """Fetch basic market metrics from free yfinance API."""
        if yf is None:
            logger.debug("CrashShield: yfinance unavailable, returning empty market data.")
            return {}
        data = {}
        for name, ticker in CrashShield.MARKETS.items():
            try:
                # Fetch 5 days to ensure we have at least 2 valid close prices
                df = yf.download(ticker, period="5d", interval="1d", progress=False)
                if not df.empty and len(df) >= 2:
                    current = df['Close'].iloc[-1]
                    prev = df['Close'].iloc[-2]
                    
                    # Handle multi-index columns if yfinance returns them
                    if isinstance(current, pd.Series):
                        current = current.iloc[0]
                    if isinstance(prev, pd.Series):
                        prev = prev.iloc[0]
                        
                    change_pct = (current - prev) / prev * 100
                    data[name] = {
                        "price": float(current),
                        "change_pct": float(change_pct),
                    }
            except Exception as e:
                logger.debug(f"CrashShield: Failed to fetch {name} ({ticker}): {e}")
        return data

    @staticmethod
    def evaluate_risk() -> dict:
        """
        Determine global risk level (0 to 3).
        
        Levels:
          0: ALL CLEAR      - Normal volatility
          1: CAUTION        - Elevated volatility, selective trading
          2: HIGH ALERT     - Systemic weakness detected, defensive posture
          3: CRASH PROTOCOL - Panic detected, halt new trades, exit weak ones
        """
        metrics = CrashShield.get_market_data()
        if not metrics:
            return {"level": 0, "status": "⚪ Data Unavailable", "data": {}}

        vix = metrics.get("VIX", {}).get("price", 0)
        
        # Count significant drops (> 2%)
        significant_drops = sum(
            1 for k, v in metrics.items() 
            if k not in ["VIX", "DXY", "GOLD"] and v["change_pct"] <= -2.0
        )
        
        # Count minor drops (> 1%)
        minor_drops = sum(
            1 for k, v in metrics.items() 
            if k not in ["VIX", "DXY", "GOLD"] and v["change_pct"] <= -1.0
        )

        level = 0
        status = "🟢 All Clear"
        
        # ── Logic ──
        if significant_drops >= 3 or (vix > 32):
            level = 3
            status = "🔴 CRASH PROTOCOL: Massive Sell-off"
        elif significant_drops >= 1 or minor_drops >= 3 or (vix > 25):
            level = 2
            status = "🟠 HIGH ALERT: Market Stress Detected"
        elif minor_drops >= 1 or (vix > 20):
            level = 1
            status = "🟡 CAUTION: Elevated Volatility"
            
        # Flight to safety signal (Stocks down, Gold up)
        equity_avg = np.mean([v["change_pct"] for k, v in metrics.items() if k in ["US", "VN", "TW"]])
        gold_up = metrics.get("GOLD", {}).get("change_pct", 0) > 0.5
        if equity_avg < -1.0 and gold_up:
            status += " (Flight to Safety Detected)"

        return {
            "level": level,
            "status": status,
            "data": metrics,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    @staticmethod
    def get_action_multiplier(level: int) -> float:
        """Returns a multiplier for position sizing based on risk level."""
        mapping = {
            0: 1.0,   # Full size
            1: 0.7,   # Reduce size (70%)
            2: 0.4,   # Aggressive reduction (40%)
            3: 0.0    # No new trades
        }
        return mapping.get(level, 0.0)
