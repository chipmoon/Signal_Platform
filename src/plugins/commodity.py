"""
Commodity Asset Provider Plugin
================================
Provides data for commodities (Gold, Silver, Oil, Gas) using Yahoo Finance.

Reuses the existing COT data fetching infrastructure.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
from loguru import logger

from .base import AssetInfo, AssetProvider


# Commodity symbol mappings
COMMODITY_SYMBOLS = {
    # Using Futures tickers as they are much more reliable on Yahoo Finance
    "GOLD": {"symbol": "GC=F", "name": "Gold Spot (Derived)", "sector": "Precious Metals"},
    "XAU": {"symbol": "GC=F", "name": "Gold Spot (Derived)", "sector": "Precious Metals"},
    "SILVER": {"symbol": "SI=F", "name": "Silver Spot (Derived)", "sector": "Precious Metals"},
    "XAG": {"symbol": "SI=F", "name": "Silver Spot (Derived)", "sector": "Precious Metals"},
    "CRUDE_OIL": {"symbol": "CL=F", "name": "Crude Oil Futures", "sector": "Energy"},
    "NATURAL_GAS": {"symbol": "NG=F", "name": "Natural Gas Futures", "sector": "Energy"},
    "COPPER": {"symbol": "HG=F", "name": "Copper Futures", "sector": "Industrial Metals"},
    "PLATINUM": {"symbol": "PL=F", "name": "Platinum Futures", "sector": "Precious Metals"},
    "PALLADIUM": {"symbol": "PA=F", "name": "Palladium Futures", "sector": "Precious Metals"},
}

PROXY_ENV_KEYS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]


@contextmanager
def _temporary_disable_broken_loopback_proxy():
    """Temporarily disable known-bad loopback proxy values (e.g. 127.0.0.1:9)."""
    backup: Dict[str, Optional[str]] = {}
    try:
        for key in PROXY_ENV_KEYS:
            value = os.environ.get(key)
            if not value:
                continue
            value_lower = value.lower()
            if "127.0.0.1:9" in value_lower or "localhost:9" in value_lower:
                backup[key] = value
                os.environ.pop(key, None)
        yield
    finally:
        for key, value in backup.items():
            if value is not None:
                os.environ[key] = value


class CommodityProvider(AssetProvider):
    """Asset provider for commodity futures via Yahoo Finance."""
    
    @property
    def market_id(self) -> str:
        return "COMMODITY"
    
    @property
    def market_name(self) -> str:
        return "Commodities & Futures"
    
    def search_assets(self, query: str, limit: int = 10) -> List[AssetInfo]:
        """
        Search commodities by symbol or name.
        
        Examples:
            "GOLD" → Gold Futures
            "silver" → Silver Futures
            "oil" → Crude Oil Futures
        """
        query_lower = query.lower()
        results = []
        
        for code, info in COMMODITY_SYMBOLS.items():
            # Match by code or name
            if (
                query_lower in code.lower()
                or query_lower in info["name"].lower()
                or query_lower in info["symbol"].lower()
            ):
                results.append(
                    AssetInfo(
                        symbol=info["symbol"],
                        name=info["name"],
                        market=self.market_id,
                        sector=info["sector"],
                        exchange="NYMEX/COMEX",
                        currency="USD",
                    )
                )
        
        return results[:limit]
    
    def get_asset_info(self, symbol: str) -> Optional[AssetInfo]:
        """
        Get commodity info by symbol.
        
        Args:
            symbol: Either commodity code (e.g., "GOLD") or Yahoo symbol (e.g., "GC=F")
        """
        # Check if it's a commodity code
        if symbol in COMMODITY_SYMBOLS:
            info = COMMODITY_SYMBOLS[symbol]
            return AssetInfo(
                symbol=info["symbol"],
                name=info["name"],
                market=self.market_id,
                sector=info["sector"],
                exchange="NYMEX/COMEX",
                currency="USD",
            )
        
        # Check if it's a direct Yahoo symbol
        for code, info in COMMODITY_SYMBOLS.items():
            if info["symbol"] == symbol:
                return AssetInfo(
                    symbol=info["symbol"],
                    name=info["name"],
                    market=self.market_id,
                    sector=info["sector"],
                    exchange="NYMEX/COMEX",
                    currency="USD",
                )
        
        return None
    
    def get_price_data(
        self,
        symbol: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch commodity price data with macro enrichment for Gold."""
        # Resolve commodity code to Yahoo symbol
        if symbol in COMMODITY_SYMBOLS:
            yahoo_symbol = COMMODITY_SYMBOLS[symbol]["symbol"]
        else:
            yahoo_symbol = symbol
        
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance not installed. Please run: pip install yfinance")
            
        try:
            logger.info(f"Fetching {yahoo_symbol} data from {start} to {end}")
            with _temporary_disable_broken_loopback_proxy():
                ticker = yf.Ticker(yahoo_symbol)
                df = ticker.history(start=start, end=end, auto_adjust=False)
            
            if df.empty:
                raise ValueError(f"No data available for {yahoo_symbol}")
            
            df = df.reset_index()
            
            # ── Gold Macro Enrichment ──────────────────────────────────
            if yahoo_symbol in ["GC=F", "GOLD", "XAUUSD=X"]:
                logger.info("Enriching Gold data with Macro Alphas (DXY, 10Y Yield)...")
                try:
                    # DXY (USD Index) and 10Y Yield
                    with _temporary_disable_broken_loopback_proxy():
                        dxy = yf.Ticker("DX-Y.NYB").history(start=start, end=end)[["Close"]].rename(columns={"Close": "DXY"})
                        yield10 = yf.Ticker("^TNX").history(start=start, end=end)[["Close"]].rename(columns={"Close": "US10Y"})
                    
                    macro = dxy.join(yield10, how="outer").ffill()
                    df = df.merge(macro, on="Date", how="left").ffill()
                    logger.success("Gold macro enrichment complete")
                except Exception as ex:
                    logger.warning(f"Macro enrichment failed: {ex}")

            logger.success(f"Fetched {len(df)} rows for {yahoo_symbol}")
            
            cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            for c in ["DXY", "US10Y"]:
                if c in df.columns:
                    cols.append(c)
            return df[cols]
            
        except Exception as e:
            logger.error(f"Failed to fetch {yahoo_symbol}: {e}")
            e_text = str(e).lower()
            if "127.0.0.1:9" in e_text or "could not connect to server" in e_text:
                raise ValueError(
                    f"Could not fetch data for {yahoo_symbol}: network/proxy is blocking Yahoo Finance. "
                    "Please disable invalid proxy settings and retry."
                )
            raise ValueError(f"Could not fetch data for {yahoo_symbol}: {e}")
    
    def get_fundamentals(self, symbol: str) -> Dict[str, float]:
        """
        Commodities don't have traditional fundamentals.
        
        Returns empty dict.
        """
        return {}
    
    def supports_fundamentals(self) -> bool:
        """Commodities don't have P/E, ROE, etc."""
        return False
    
    def supports_cot_data(self) -> bool:
        """Commodities support COT (Commitment of Traders) data."""
        return True
    
    def get_cot_code(self, symbol: str) -> Optional[str]:
        """
        Get CFTC commodity code for COT data.
        
        Args:
            symbol: Commodity symbol
            
        Returns:
            CFTC code or None
        """
        # Map Yahoo symbols/codes to CFTC codes
        cot_mapping = {
            "GOLD": "GOLD",
            "GC=F": "GOLD",
            "SILVER": "SILVER",
            "SI=F": "SILVER",
            "CRUDE_OIL": "CRUDE_OIL",
            "CL=F": "CRUDE_OIL",
            "NATURAL_GAS": "NATURAL_GAS",
            "NG=F": "NATURAL_GAS",
        }
        
        return cot_mapping.get(symbol)
    
    def list_all(self) -> List[AssetInfo]:
        """Get all available commodities."""
        return [
            AssetInfo(
                symbol=info["symbol"],
                name=info["name"],
                market=self.market_id,
                sector=info["sector"],
                exchange="NYMEX/COMEX",
                currency="USD",
            )
            for info in COMMODITY_SYMBOLS.values()
        ]


# Auto-register this provider
from . import registry
_commodity_provider = CommodityProvider()
registry.register(_commodity_provider)

logger.success(f"Registered {_commodity_provider.market_name} provider")
