"""
Vietnam Stock Provider Plugin
==============================
Provides data for Vietnamese stocks using VNStock API.

Extracts from D:\Python_VS\stock\stock_vn_rs.py
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from .base import AssetInfo, AssetProvider

# Import vnstock with graceful fallback
try:
    from vnstock import Listing, Vnstock
    VNSTOCK_AVAILABLE = True
except ImportError:
    logger.warning("vnstock not installed. Run: pip install vnstock")
    VNSTOCK_AVAILABLE = False


class VietnamStockProvider(AssetProvider):
    """Asset provider for Vietnamese stocks via VNStock API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Vietnam stock provider.
        
        Args:
            api_key: VNStock API key (optional, can use environment variable)
        """
        if not VNSTOCK_AVAILABLE:
            raise ImportError("vnstock library not available. Install with: pip install vnstock")
        
        # Set API key if provided
        if api_key:
            os.environ['VNSTOCK_API_KEY'] = api_key
        elif 'VNSTOCK_API_KEY' not in os.environ:
            # Use default key from original file
            logger.warning("No VNSTOCK_API_KEY found, using default")
            os.environ['VNSTOCK_API_KEY'] = 'vnstock_ee8c180549c43fab65ea2396660d2051'
        
        self._stock_list: List[Dict[str, str]] = []
        self._industry_map: Dict[str, str] = {}
        self._initialized = False
    
    def _ensure_initialized(self) -> None:
        """Lazy load stock list on first use."""
        if self._initialized:
            return
        
        try:
            logger.info("Loading Vietnam stock list...")
            listing = Listing()
            df_listing = listing.symbols_by_exchange(lang='vi')
            
            # Filter to HOSE and HNX exchanges
            if 'exchange' in df_listing.columns:
                df_listing = df_listing[df_listing['exchange'].isin(['HOSE', 'HNX'])].copy()
            
            # Filter to stocks only (not derivatives)
            if 'type' in df_listing.columns:
                for stock_type in ['STOCK', 'Stock', 'stock']:
                    if (df_listing['type'] == stock_type).sum() > 0:
                        df_listing = df_listing[df_listing['type'] == stock_type].copy()
                        break
            
            # Get industry data
            try:
                df_industries = listing.symbols_by_industries(lang='vi')
                if 'industry_name' in df_industries.columns:
                    df_listing = df_listing.merge(
                        df_industries[['symbol', 'industry_name']],
                        on='symbol',
                        how='left'
                    )
                    self._industry_map = dict(zip(
                        df_listing['symbol'],
                        df_listing['industry_name'].fillna('Other')
                    ))
            except Exception as e:
                logger.warning(f"Could not load industry data: {e}")
                self._industry_map = {}
            
            # Build stock list
            for _, row in df_listing.iterrows():
                self._stock_list.append({
                    'symbol': row['symbol'],
                    'name': row.get('organ_short_name', row['symbol']),
                    'exchange': row.get('exchange', 'VN'),
                    'sector': self._industry_map.get(row['symbol'], 'Other'),
                })
            
            self._initialized = True
            logger.success(f"Loaded {len(self._stock_list)} Vietnam stocks")
            
        except Exception as e:
            logger.error(f"Failed to load stock list: {e}")
            self._initialized = True  # Don't retry on every call
    
    @property
    def market_id(self) -> str:
        return "VN"
    
    @property
    def market_name(self) -> str:
        return "Vietnam Stocks"
    
    def search_assets(self, query: str, limit: int = 10) -> List[AssetInfo]:
        """
        Search Vietnamese stocks by symbol or name.
        
        Examples:
            "VNM" → Vinamilk
            "vinamilk" → VNM
            "vcb" → Vietcombank
        """
        self._ensure_initialized()
        
        query_lower = query.lower()
        results = []
        
        for stock in self._stock_list:
            # Match by symbol or name
            if (
                query_lower in stock['symbol'].lower()
                or query_lower in stock['name'].lower()
            ):
                results.append(
                    AssetInfo(
                        symbol=f"{stock['symbol']}.VN",  # Add .VN suffix
                        name=stock['name'],
                        market=self.market_id,
                        sector=stock.get('sector'),
                        exchange=stock.get('exchange', 'HOSE'),
                        currency="VND",
                    )
                )
        
        return results[:limit]
    
    def get_asset_info(self, symbol: str) -> Optional[AssetInfo]:
        """
        Get Vietnam stock info by symbol.
        
        Args:
            symbol: Stock symbol (e.g., "VNM" or "VNM.VN")
        """
        self._ensure_initialized()
        
        # Normalize symbol (remove .VN suffix if present)
        base_symbol = symbol.replace('.VN', '').upper()
        
        for stock in self._stock_list:
            if stock['symbol'] == base_symbol:
                return AssetInfo(
                    symbol=f"{stock['symbol']}.VN",
                    name=stock['name'],
                    market=self.market_id,
                    sector=stock.get('sector'),
                    exchange=stock.get('exchange', 'HOSE'),
                    currency="VND",
                )
        
        return None
    
    def get_price_data(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = '1d',
    ) -> pd.DataFrame:
        """
        Fetch Vietnam stock price data.
        
        Args:
            symbol: Stock symbol (e.g., "VNM" or "VNM.VN")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Data frequency ('1d', '1h')
            
        Returns:
            DataFrame with OHLCV data
        """
        # Normalize symbol
        base_symbol = symbol.replace('.VN', '').upper()
        
        try:
            logger.info(f"Fetching {base_symbol} data ({interval}) from {start} to {end}")
            
            # vnstock uses '1D' or '1H'
            vn_interval = '1D' if interval == '1d' else '1H' if interval == '1h' else interval.upper()
            
            stock = Vnstock().stock(symbol=base_symbol, source='VCI')
            df = stock.quote.history(start=start, end=end, interval=vn_interval)
            
            if df is None or df.empty:
                raise ValueError(f"No data available for {base_symbol}")
            
            # VNStock returns lowercase column names
            # Rename to match standard format
            column_map = {
                'time': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            
            df = df.rename(columns=column_map)
            
            # Ensure required columns exist
            required = ["Date", "Open", "High", "Low", "Close", "Volume"]
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            
            logger.success(f"Fetched {len(df)} rows for {base_symbol}")
            return df[required]
            
        except Exception as e:
            logger.error(f"Failed to fetch {base_symbol}: {e}")
            raise ValueError(f"Could not fetch data for {base_symbol}: {e}")
    
    def get_fundamentals(self, symbol: str) -> Dict[str, float]:
        """
        Get fundamental data for Vietnam stock.
        
        Uses multiple fallback methods:
        1. financial_ratio() API
        2. overview() API
        3. Calculated from price data
        
        Returns:
            Dictionary with: eps, pe, pb, roe, bvps, dividend_yield
        """
        # Normalize symbol
        base_symbol = symbol.replace('.VN', '').upper()
        
        fundamentals = {
            'eps': 0.0,
            'pe': 0.0,
            'pb': 0.0,
            'roe': 0.0,
            'bvps': 0.0,
            'dividend_yield': 0.0,
        }
        
        try:
            stock = Vnstock().stock(symbol=base_symbol, source='VCI')
            
            # Method 1: Try financial_ratio() API
            try:
                ratios = stock.company.financial_ratio(period='quarter', lang='vi')
                
                if ratios is not None and not ratios.empty:
                    latest = ratios.iloc[0]
                    
                    # Extract metrics using flexible column matching
                    for col in ratios.columns:
                        col_lower = col.lower()
                        val = latest[col]
                        
                        if pd.notna(val) and val != 0:
                            if 'p/e' in col_lower or 'pe' == col_lower:
                                fundamentals['pe'] = float(val)
                            elif 'p/b' in col_lower or 'pb' == col_lower:
                                fundamentals['pb'] = float(val)
                            elif 'roe' in col_lower and 'pre' not in col_lower:
                                roe_val = float(val)
                                # ROE might be decimal (0.15) or percentage (15)
                                fundamentals['roe'] = roe_val * 100 if roe_val < 1 else roe_val
                            elif 'eps' in col_lower and 'beps' not in col_lower:
                                fundamentals['eps'] = float(val)
                            elif 'bvps' in col_lower or 'book_value' in col_lower:
                                fundamentals['bvps'] = float(val)
                            elif 'dividend' in col_lower and 'yield' in col_lower:
                                div_val = float(val)
                                fundamentals['dividend_yield'] = div_val * 100 if div_val < 1 else div_val
                
            except Exception as e:
                logger.debug(f"{base_symbol} - financial_ratio() failed: {e}")
            
            # Method 2: Try overview() API (backup)
            if fundamentals['eps'] == 0:
                try:
                    overview = stock.company.overview(lang='vi')
                    
                    if overview is not None and not overview.empty:
                        for col in overview.columns:
                            col_lower = col.lower()
                            val = overview[col].iloc[0] if len(overview) > 0 else None
                            
                            if val is not None and pd.notna(val):
                                if 'eps' in col_lower:
                                    fundamentals['eps'] = float(val)
                                elif 'pe' in col_lower or 'p/e' in col_lower:
                                    fundamentals['pe'] = float(val)
                                elif 'pb' in col_lower or 'p/b' in col_lower:
                                    fundamentals['pb'] = float(val)
                                elif 'roe' in col_lower:
                                    roe_val = float(val)
                                    fundamentals['roe'] = roe_val * 100 if roe_val < 1 else roe_val
                                elif 'bvps' in col_lower:
                                    fundamentals['bvps'] = float(val)
                
                except Exception as e:
                    logger.debug(f"{base_symbol} - overview() failed: {e}")
            
            # Method 3: Calculate P/E from current price if we have EPS
            if fundamentals['pe'] == 0 and fundamentals['eps'] > 0:
                try:
                    df = stock.quote.history(
                        start=(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'),
                        end=datetime.now().strftime('%Y-%m-%d'),
                        interval='1D'
                    )
                    if df is not None and len(df) > 0:
                        current_price = df.iloc[-1]['close']
                        fundamentals['pe'] = current_price / fundamentals['eps']
                except Exception as e:
                    logger.debug(f"{base_symbol} - price calculation failed: {e}")
            
            logger.info(f"{base_symbol} fundamentals: P/E={fundamentals['pe']:.2f}, EPS={fundamentals['eps']:.2f}, ROE={fundamentals['roe']:.2f}%")
            
        except Exception as e:
            logger.warning(f"Could not fetch fundamentals for {base_symbol}: {e}")
        
        return fundamentals
    
    def supports_fundamentals(self) -> bool:
        """Vietnam stocks have full fundamental data."""
        return True
    
    def supports_cot_data(self) -> bool:
        """Vietnam stocks don't have COT data (that's for commodities)."""
        return False
    
    def list_top_by_sector(self, sector: str, limit: int = 20) -> List[AssetInfo]:
        """
        List top stocks in a sector.
        
        Args:
            sector: Sector name (e.g., "Banking", "Technology")
            limit: Maximum number of stocks
            
        Returns:
            List of assets in the sector
        """
        self._ensure_initialized()
        
        results = []
        for stock in self._stock_list:
            if stock.get('sector', '').lower() == sector.lower():
                results.append(
                    AssetInfo(
                        symbol=f"{stock['symbol']}.VN",
                        name=stock['name'],
                        market=self.market_id,
                        sector=stock.get('sector'),
                        exchange=stock.get('exchange', 'HOSE'),
                        currency="VND",
                    )
                )
        
        return results[:limit]


# Auto-register this provider
from . import registry

try:
    _vn_provider = VietnamStockProvider()
    registry.register(_vn_provider)
    logger.success(f"Registered {_vn_provider.market_name} provider")
except Exception as e:
    logger.error(f"Failed to register Vietnam provider: {e}")
