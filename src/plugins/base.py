"""
Asset Provider Plugin System — Base Interface
==============================================
Defines the contract that all market data providers must implement.

This allows the system to support unlimited asset types (stocks, crypto, commodities)
through a unified interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class AssetInfo:
    """Metadata about a tradeable asset."""
    
    symbol: str                # Unique identifier (e.g., "VNM.VN", "2330.TW", "GC=F")
    name: str                  # Display name (e.g., "Vinamilk", "TSMC", "Gold")
    market: str                # Market identifier (e.g., "VN", "TW", "COMMODITY")
    sector: Optional[str] = None
    exchange: Optional[str] = None
    currency: str = "USD"      # Default currency


@dataclass
class RealtimeQuote:
    """Real-time price quote for an asset."""

    symbol: str
    price: float               # Latest price
    change: float = 0.0        # % change from previous close
    prev_close: float = 0.0    # Previous closing price
    volume: float = 0.0        # Current session volume
    high: float = 0.0          # Session high
    low: float = 0.0           # Session low
    timestamp: str = ""        # When this quote was fetched (ISO format)
    source: str = ""           # Data source (e.g., 'vnstock', 'yfinance', 'tvdatafeed')
    is_market_open: bool = False  # Whether the market is currently open


@dataclass
class AssetData:
    """Complete dataset for an asset including price and fundamentals."""
    
    info: AssetInfo
    price_data: pd.DataFrame   # OHLCV data
    fundamentals: Dict[str, float]  # P/E, ROE, EPS, etc.
    indicators: Dict[str, pd.Series]  # RSI, MACD, etc.
    volume_data: Optional[pd.DataFrame] = None
    metadata: Dict[str, any] = None


class AssetProvider(ABC):
    """
    Abstract base class for asset data providers.
    
    Each market (VN stocks, TW stocks, Commodities, Crypto, etc.)
    implements this interface as a plugin.
    """
    
    @property
    @abstractmethod
    def market_id(self) -> str:
        """Unique identifier for this market (e.g., 'VN', 'TW', 'COMMODITY')."""
        pass
    
    @property
    @abstractmethod
    def market_name(self) -> str:
        """Human-readable market name (e.g., 'Vietnam Stocks', 'Taiwan Stocks')."""
        pass
    
    @abstractmethod
    def search_assets(self, query: str, limit: int = 10) -> List[AssetInfo]:
        """
        Search for assets by symbol or name.
        
        Args:
            query: Search term (e.g., "VNM", "Vinamilk", "TSMC")
            limit: Maximum number of results
            
        Returns:
            List of matching assets
        """
        pass
    
    @abstractmethod
    def get_asset_info(self, symbol: str) -> Optional[AssetInfo]:
        """
        Get metadata for a specific asset.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            AssetInfo if found, None otherwise
        """
        pass
    
    @abstractmethod
    def get_price_data(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV price data for an asset.
        
        Args:
            symbol: Asset symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Data frequency ('1m', '5m', '15m', '1h', '4h', '1d', '1wk')
            
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
            
        Raises:
            ValueError: If symbol not found or data unavailable
        """
        pass
    
    @abstractmethod
    def get_fundamentals(self, symbol: str) -> Dict[str, float]:
        """
        Get fundamental data for an asset.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Dictionary of fundamental metrics:
            - eps: Earnings per share
            - pe: Price-to-earnings ratio
            - pb: Price-to-book ratio
            - roe: Return on equity (%)
            - bvps: Book value per share
            - dividend_yield: Dividend yield (%)
            
        Note:
            Returns empty dict if fundamentals not available (e.g., for commodities)
        """
        pass
    
    @abstractmethod
    def supports_fundamentals(self) -> bool:
        """Whether this provider supports fundamental data."""
        pass
    
    @abstractmethod
    def supports_cot_data(self) -> bool:
        """Whether this provider supports COT (Commitment of Traders) data."""
        pass

    def get_realtime_quote(self, symbol: str) -> Optional[RealtimeQuote]:
        """
        Get real-time price quote for an asset.

        This is a NON-CACHED, direct API call intended for displaying
        the current market price. Unlike get_price_data() which returns
        historical OHLCV candles, this returns the latest available price.

        Default implementation: fetches last 3 days of price data and
        returns the most recent Close. Subclasses should override for
        true intraday/real-time quotes.

        Args:
            symbol: Asset symbol

        Returns:
            RealtimeQuote or None if unavailable
        """
        try:
            end = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
            start = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
            df = self.get_price_data(symbol, start, end)

            if df is None or df.empty:
                return None

            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) >= 2 else last
            price = float(last['Close'])
            prev_close = float(prev['Close'])
            change = ((price - prev_close) / prev_close * 100) if prev_close > 0 else 0.0

            return RealtimeQuote(
                symbol=symbol,
                price=price,
                change=round(change, 2),
                prev_close=prev_close,
                volume=float(last.get('Volume', 0)),
                high=float(last.get('High', price)),
                low=float(last.get('Low', price)),
                timestamp=datetime.now().isoformat(),
                source='historical_fallback',
                is_market_open=True,
            )
        except Exception as e:
            from loguru import logger
            logger.warning(f"Realtime quote fallback failed for {symbol}: {e}")
            return None
    
    def get_full_data(
        self,
        symbol: str,
        start: str,
        end: str,
    ) -> AssetData:
        """
        Get complete dataset for an asset.
        
        This is a convenience method that combines all data sources.
        Subclasses can override for optimization.
        
        Args:
            symbol: Asset symbol
            start: Start date
            end: End date
            
        Returns:
            Complete AssetData object
        """
        info = self.get_asset_info(symbol)
        if not info:
            raise ValueError(f"Asset {symbol} not found in {self.market_id}")
        
        price_data = self.get_price_data(symbol, start, end)
        fundamentals = self.get_fundamentals(symbol) if self.supports_fundamentals() else {}
        
        return AssetData(
            info=info,
            price_data=price_data,
            fundamentals=fundamentals,
            indicators={},  # Will be calculated later
        )
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Check if a symbol is valid for this market.
        
        Args:
            symbol: Asset symbol to validate
            
        Returns:
            True if valid, False otherwise
        """
        return self.get_asset_info(symbol) is not None


class ProviderRegistry:
    """
    Central registry for all asset providers.
    
    Allows dynamic registration and lookup of providers by market ID.
    """
    
    def __init__(self):
        self._providers: Dict[str, AssetProvider] = {}
    
    def register(self, provider: AssetProvider) -> None:
        """Register a new provider."""
        self._providers[provider.market_id] = provider
    
    def get(self, market_id: str) -> Optional[AssetProvider]:
        """Get provider by market ID."""
        return self._providers.get(market_id)
    
    def list_markets(self) -> List[Dict[str, str]]:
        """List all registered markets."""
        return [
            {"id": provider.market_id, "name": provider.market_name}
            for provider in self._providers.values()
        ]
    
    def search_all(self, query: str, limit: int = 10) -> List[AssetInfo]:
        """Search across all markets with smart priority ordering.
        
        Priority: COMMODITY > TW > VN > US
        
        Key behaviors:
        - COMMODITY/TW/VN are always searched first
        - If priority markets find exact matches, US general search is skipped
        - US provider only adds results for symbols not already claimed
        - This ensures PAN → PAN.VN (not KO/BA from "Company" match)
        """
        # Search priority order: commodity first, then specific markets, then US
        priority_order = ["COMMODITY", "TW", "VN", "US"]
        ordered_providers = sorted(
            self._providers.values(),
            key=lambda p: priority_order.index(p.market_id) if p.market_id in priority_order else 99,
        )
        
        results = []
        seen_symbols = set()
        seen_base_symbols = set()  # To track 'BSR' when we see 'BSR.VN'
        priority_has_exact_match = False  # Track if VN/TW/COMMODITY found exact match
        
        # 1. Exact symbol matches first (case-insensitive)
        for provider in ordered_providers:
            info = provider.get_asset_info(query)
            if info and info.symbol not in seen_symbols:
                base = info.symbol.split('.')[0].upper()
                # Deduplication: If we have a market hit (VN/TW), ignore the generic US fallback
                if provider.market_id == "US" and base in seen_base_symbols:
                    continue
                # Also skip US if priority markets already found something
                if provider.market_id == "US" and priority_has_exact_match:
                    continue

                results.append(info)
                seen_symbols.add(info.symbol)
                seen_base_symbols.add(base)

                # Mark if a priority market found an exact match
                if provider.market_id != "US":
                    priority_has_exact_match = True

        # 2. General search — skip US if priority markets already have any results
        priority_has_results = any(r.market != "US" for r in results)
        for provider in ordered_providers:
            # If priority markets (TW/VN/COMMODITY) found anything, don't let US add noise
            if provider.market_id == "US" and priority_has_results:
                continue

            hits = provider.search_assets(query, limit=limit)
            for hit in hits:
                if hit.symbol not in seen_symbols:
                    base = hit.symbol.split('.')[0].upper()
                    if provider.market_id == "US" and base in seen_base_symbols:
                        continue

                    results.append(hit)
                    seen_symbols.add(hit.symbol)
                    seen_base_symbols.add(base)

        return results[:limit]


# Global registry instance
registry = ProviderRegistry()
