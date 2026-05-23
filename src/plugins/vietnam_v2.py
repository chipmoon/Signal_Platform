"""
Vietnam Stock Provider Plugin v2.0
===================================
Enhanced with:
- Parquet caching for instant startup (no 30s wait)
- TvDatafeed fallback for reliable price data
- Graceful degradation when vnstock API fails

Architecture:
    1. Stock list: Cache â†’ vnstock API
    2. Price data: Cache â†’ vnstock â†’ TvDatafeed fallback
"""

from __future__ import annotations

import os
import threading
import concurrent.futures
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from .base import AssetInfo, AssetProvider
from ..cache_manager import cache

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
    """Temporarily disable loopback proxy values that break upstream API calls."""
    backup: Dict[str, Optional[str]] = {}
    
    # Streamlit Cloud uses /mount/src which might have permission issues for dotfiles. Use /tmp.
    is_streamlit = "STREAMLIT_SERVER_PORT" in os.environ or os.path.exists("/mount/src")
    project_root = "/tmp" if is_streamlit else str(Path(__file__).resolve().parents[2])
    
    try:
        # Force writable home for libraries that persist local state (e.g., vnstock)
        for home_key in ["HOME", "USERPROFILE"]:
            backup[home_key] = os.environ.get(home_key)
            os.environ[home_key] = project_root

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
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

# Import vnstock with graceful fallback
try:
    from vnstock import Listing, Vnstock
    VNSTOCK_AVAILABLE = True
except ImportError:
    logger.warning("vnstock not installed. Run: pip install vnstock")
    VNSTOCK_AVAILABLE = False

# Import tvDatafeed with graceful fallback
try:
    from tvDatafeed import TvDatafeed, Interval
    TVDATAFEED_AVAILABLE = True
except ImportError:
    logger.warning("tvDatafeed not installed â€” TvDatafeed features disabled.")
    TVDATAFEED_AVAILABLE = False


def _safe_err(e: Exception) -> str:
    """Return ASCII-safe error text to avoid charmap logging crashes."""
    try:
        return str(e).encode("ascii", errors="ignore").decode("ascii") or e.__class__.__name__
    except Exception:
        return e.__class__.__name__


class VietnamStockProviderV2(AssetProvider):
    """Enhanced Vietnam stock provider with caching and fallback."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Vietnam stock provider.

        Args:
            api_key: VNStock API key (optional, can use environment variable)
        """
        if not VNSTOCK_AVAILABLE:
            # Degrade gracefully â€” don't raise, just log and operate with empty list
            logger.warning("vnstock not available â€” Vietnam provider running in degraded mode (no VN stocks).")

        # Set API key
        if api_key:
            os.environ['VNSTOCK_API_KEY'] = api_key
        elif 'VNSTOCK_API_KEY' not in os.environ:
            logger.warning("No VNSTOCK_API_KEY found, using default")
            os.environ['VNSTOCK_API_KEY'] = 'vnstock_ee8c180549c43fab65ea2396660d2051'

        self._stock_list: List[Dict[str, str]] = []
        self._industry_map: Dict[str, str] = {}
        self._initialized = False
        self._vni_cache: Dict[str, pd.DataFrame] = {}
        self._vni_lock = threading.Lock()
        self._vni_fetch_failed = False

        # TvDatafeed instance (lazy init)
        self._tv: Optional[TvDatafeed] = None
        self._vnstock_source_timeout_sec = 4
        self._vnstock_max_sources = 2

    def _get_tv(self) -> Optional[TvDatafeed]:
        """Lazy initialize TvDatafeed."""
        if not TVDATAFEED_AVAILABLE:
            return None
        if self._tv is None:
            try:
                # Get from env vars first, fallback to hardcoded if not present
                tv_user = os.environ.get("TV_USERNAME", "tthieu27")
                tv_pass = os.environ.get("TV_PASSWORD", "tea2tesla")
                
                self._tv = TvDatafeed(username=tv_user, password=tv_pass)
                logger.info("TvDatafeed initialized with authentication")
            except Exception as e:
                logger.warning(f"TvDatafeed authenticated init failed: {e}. Trying anonymous...")
                try:
                    self._tv = TvDatafeed()
                    logger.info("TvDatafeed initialized anonymously")
                except Exception as ex:
                    logger.warning(f"TvDatafeed anonymous init failed: {ex}")
                    return None
        return self._tv

    def _ensure_initialized(self) -> None:
        """Lazy load stock list with cache."""
        if self._initialized:
            return
        
        # If vnstock is not available, we cannot initialize the stock list
        if not VNSTOCK_AVAILABLE:
            self._initialized = True
            return

        # Try cache first
        cached = cache.get_cached_stock_list("VN", max_age_hours=24)
        if cached and len(cached) > 0:
            self._stock_list = cached
            self._industry_map = {s['symbol']: s.get('sector', 'Other') for s in cached}
            self._initialized = True
            logger.success(f"Loaded {len(cached)} VN stocks from cache")
            return
        if cached is not None and len(cached) == 0:
            logger.warning("VN cache file exists but is empty, ignoring fresh cache")

        # Fallback to stale cache if available and non-empty
        stale_cached = cache.get_cached_stock_list("VN", max_age_hours=999999)
        if stale_cached and len(stale_cached) > 0:
            self._stock_list = stale_cached
            self._industry_map = {s['symbol']: s.get('sector', 'Other') for s in stale_cached}
            self._initialized = True
            logger.warning(f"Loaded {len(stale_cached)} VN stocks from stale cache")
            return

        # Fetch from API
        try:
            logger.info("Fetching Vietnam stock list from vnstock API...")
            with _temporary_disable_broken_loopback_proxy():
                listing = Listing()
                df_listing = listing.symbols_by_exchange(lang='vi')

            # Filter to HOSE and HNX
            if 'exchange' in df_listing.columns:
                df_listing = df_listing[df_listing['exchange'].isin(['HOSE', 'HNX'])].copy()

            # Filter to stocks only
            if 'type' in df_listing.columns:
                for stock_type in ['STOCK', 'Stock', 'stock']:
                    if (df_listing['type'] == stock_type).sum() > 0:
                        df_listing = df_listing[df_listing['type'] == stock_type].copy()
                        break

            # Deduplicate symbols before industry merge to prevent cartesian explosion
            df_listing = df_listing.drop_duplicates(subset=['symbol']).copy()

            # Get industry data
            try:
                with _temporary_disable_broken_loopback_proxy():
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

            # Cache it
            if self._stock_list:
                cache.cache_stock_list("VN", self._stock_list)

            self._initialized = True
            logger.success(f"Loaded {len(self._stock_list)} Vietnam stocks from API")

        except Exception as e:
            logger.error(f"Failed to load stock list: {e}")
            # Final fallback: derive symbol universe from local foundation dataset
            try:
                foundation_path = (
                    Path(__file__).resolve().parents[2] / "data" / "foundation_vn_3y.parquet"
                )
                if foundation_path.exists():
                    df = pd.read_parquet(foundation_path, columns=["Symbol"])
                    symbols = sorted({str(s).strip().upper() for s in df["Symbol"].dropna().tolist() if str(s).strip()})
                    if symbols:
                        self._stock_list = [
                            {"symbol": sym, "name": sym, "exchange": "VN", "sector": "Other"}
                            for sym in symbols
                        ]
                        self._industry_map = {s["symbol"]: "Other" for s in self._stock_list}
                        logger.warning(
                            f"Loaded {len(self._stock_list)} VN symbols from local foundation fallback"
                        )
            except Exception as fallback_err:
                logger.error(f"VN foundation fallback failed: {fallback_err}")
            self._initialized = True  # Don't retry on every call

    @property
    def market_id(self) -> str:
        return "VN"

    @property
    def market_name(self) -> str:
        return "Vietnam Stocks"

    def search_assets(self, query: str, limit: int = 10) -> List[AssetInfo]:
        """Search Vietnamese stocks by symbol or name."""
        self._ensure_initialized()

        query_lower = query.lower()
        results = []

        for stock in self._stock_list:
            full_symbol = f"{stock['symbol']}.VN".lower()
            if (
                query_lower in stock['symbol'].lower()
                or query_lower in full_symbol
                or query_lower in stock['name'].lower()
            ):
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

    def get_asset_info(self, symbol: str) -> Optional[AssetInfo]:
        """Get Vietnam stock info by symbol."""
        self._ensure_initialized()

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
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch Vietnam stock price data with signal enrichment:
        - Support for 1h, 1d intervals
        """
        base_symbol = symbol.replace('.VN', '').upper()
        
        # Internal mapping for vnstock
        interval_l = str(interval).lower()
        vn_interval = (
            '1D' if interval_l == '1d'
            else '1H' if interval_l == '1h'
            else '15m' if interval_l == '15m'
            else interval
        )

        # Try cache first
        cached = cache.get_cached_price_data(
            base_symbol, "VN", max_age_hours=24, start=start, end=end
        )
        if cached is not None and not cached.empty and "VNI" in cached.columns:
            logger.info(f"Using cached enriched data for {base_symbol}")
            return cached

        # Try primary fetch (vnstock)
        if VNSTOCK_AVAILABLE:
            try:
                # Handle 4h by fetching 1h and resampling
                fetch_interval = vn_interval
                resample_4h = False
                if vn_interval == '4h' or vn_interval == '4H':
                    fetch_interval = '1H'
                    resample_4h = True

                df = self._fetch_from_vnstock(base_symbol, start, end, fetch_interval)
                
                if df is not None and not df.empty:
                    if resample_4h:
                        logger.info(f"Resampling 1H to 4H for {base_symbol}")
                        df = df.set_index('Date').resample('4H').agg({
                            'Open': 'first',
                            'High': 'max',
                            'Low': 'min',
                            'Close': 'last',
                            'Volume': 'sum'
                        }).dropna().reset_index()

                    # â”€â”€ Vietnam Signal Enrichment (Only for Daily) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                    if interval_l == "1d" and base_symbol != "VNINDEX":
                        logger.info(f"Enriching {base_symbol} with Alpha Signals (VNI)...")
                        try:
                            # 1. VNI Index (Market Beta)
                            vni_cache_key = f"{start}:{end}"
                            vni = self._vni_cache.get(vni_cache_key)
                            if vni is None and not self._vni_fetch_failed:
                                # Prevent concurrent VNINDEX fetch stampede during threaded scans
                                with self._vni_lock:
                                    vni = self._vni_cache.get(vni_cache_key)
                                    if vni is None and not self._vni_fetch_failed:
                                        with _temporary_disable_broken_loopback_proxy():
                                            vni_ticker = Vnstock().stock(symbol='VNINDEX', source='VCI')
                                            vni = vni_ticker.quote.history(start=start, end=end, interval='1D')
                                        self._vni_cache[vni_cache_key] = vni
                            
                            if vni is not None and not vni.empty:
                                vni['Date'] = pd.to_datetime(vni['time']).dt.tz_localize(None)
                                vni = vni.rename(columns={'close': 'VNI'})[['Date', 'VNI']]
                                df = df.merge(vni, on="Date", how="left").ffill()
                        except Exception as ex:
                            self._vni_fetch_failed = True
                            logger.warning(f"VN signal enrichment failed: {_safe_err(ex)}")

                    # Cache and return
                    cache.cache_price_data(base_symbol, "VN", df)
                    return df
            except Exception as e:
                logger.warning(f"vnstock failed for {base_symbol}: {_safe_err(e)}")

        # Fallback to TvDatafeed (supports intraday better when available)
        if TVDATAFEED_AVAILABLE:
            try:
                df = self._fetch_from_tradingview(base_symbol, start, end, interval=interval_l)
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"TvDatafeed fallback failed for {base_symbol}: {_safe_err(e)}")

        # Intraday fallback to yfinance before giving up
        if interval_l != "1d":
            try:
                df = self._fetch_from_yfinance(base_symbol, start, end, interval_l)
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"yfinance intraday fallback failed for {base_symbol}: {_safe_err(e)}")

        # Last resort: stale cache
        stale_cache = cache.get_cached_price_data(
            base_symbol, "VN", max_age_hours=999999, start=start, end=end
        )
        if stale_cache is not None and not stale_cache.empty:
            return stale_cache

        # For intraday, if we reach here, just return empty instead of crashing
        if interval_l != "1d":
            logger.warning(f"Could not fetch intraday data for {base_symbol}, returning empty.")
            return pd.DataFrame()

        # Fallback to yfinance as ultimate resort (works on Streamlit Cloud)
        try:
            df = self._fetch_from_yfinance(base_symbol, start, end, interval)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning(f"yfinance fallback failed for {base_symbol}: {_safe_err(e)}")

        raise ValueError(f"Could not fetch data for {base_symbol} from any source")

    def _fetch_from_yfinance(self, symbol: str, start: str, end: str, interval: str) -> Optional[pd.DataFrame]:
        """Ultimate fallback using Yahoo Finance (adds .VN suffix)."""
        import yfinance as yf
        yf_symbol = f"{symbol}.VN"
        intr = str(interval).lower()
        if intr == "15m":
            yf_interval = "15m"
        elif intr in {"1h", "60m"}:
            yf_interval = "60m"
        elif intr == "4h":
            yf_interval = "60m"  # resample later if needed
        else:
            yf_interval = "1d"
        try:
            logger.info(f"Fetching {yf_symbol} from yfinance as ultimate fallback")
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(start=start, end=end, interval=yf_interval, auto_adjust=False)
            if df.empty:
                return None
            df = df.reset_index()
            if "Date" not in df.columns and "Datetime" in df.columns:
                df = df.rename(columns={"Datetime": "Date"})
            elif "Date" not in df.columns and "index" in df.columns:
                df = df.rename(columns={"index": "Date"})
                
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            required = ["Date", "Open", "High", "Low", "Close", "Volume"]
            if intr == "4h" and len(df) > 2:
                df = df.set_index("Date").resample("4H").agg({
                    "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
                }).dropna().reset_index()
            logger.success(f"yfinance: {len(df)} rows for {yf_symbol}")
            return df[required]
        except Exception as e:
            logger.debug(f"yfinance failed for {yf_symbol}: {_safe_err(e)}")
            return None

    def _fetch_from_vnstock(self, symbol: str, start: str, end: str, interval: str = '1D') -> Optional[pd.DataFrame]:
        """Fetch from vnstock API."""
        logger.info(f"Fetching {symbol} ({interval}) from vnstock ({start} to {end})")

        sources = ['VCI', 'TCBS', 'SSI', 'VND'][: self._vnstock_max_sources]
        df = None
        
        for source in sources:
            try:
                def _fetch_once() -> Optional[pd.DataFrame]:
                    with _temporary_disable_broken_loopback_proxy():
                        stock = Vnstock().stock(symbol=symbol, source=source)
                        return stock.quote.history(start=start, end=end, interval=interval)

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(_fetch_once)
                    df = fut.result(timeout=self._vnstock_source_timeout_sec)
                if df is not None and not df.empty:
                    logger.info(f"Successfully fetched {symbol} from {source}")
                    break
            except concurrent.futures.TimeoutError:
                logger.debug(f"vnstock source {source} timeout for {symbol}")
            except Exception as e:
                logger.debug(f"vnstock source {source} failed for {symbol}: {_safe_err(e)}")

        if df is None or df.empty:
            return None

        # Rename columns
        column_map = {
            'time': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        df = df.rename(columns=column_map)

        # Normalize optional real-flow columns when available from source
        optional_aliases = {
            "Foreign_Buy": [
                "foreign_buy_volume", "foreign_buy_value", "buy_foreign_qty", "foreigner_buy_volume"
            ],
            "Foreign_Sell": [
                "foreign_sell_volume", "foreign_sell_value", "sell_foreign_qty", "foreigner_sell_volume"
            ],
            "Block_Trade_Volume": [
                "deal_volume", "put_through_volume", "block_trade_volume", "pt_volume"
            ],
        }
        for target, aliases in optional_aliases.items():
            for src_col in aliases:
                if src_col in df.columns:
                    df[target] = pd.to_numeric(df[src_col], errors="coerce")
                    break
        
        # Ensure Date is timezone-naive datetime for reliable merging/analysis
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

        required = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        logger.success(f"vnstock: {len(df)} rows for {symbol}")
        optional = [c for c in ["Foreign_Buy", "Foreign_Sell", "Block_Trade_Volume"] if c in df.columns]
        return df[required + optional]

    def _fetch_from_tradingview(self, symbol: str, start: str, end: str, interval: str = "1d") -> Optional[pd.DataFrame]:
        """Fetch from TradingView via tvDatafeed."""
        tv = self._get_tv()
        if tv is None:
            return None

        logger.info(f"Fetching {symbol} from TradingView ({start} to {end})")

        # Calculate number of bars needed
        start_date = datetime.strptime(start, '%Y-%m-%d')
        end_date = datetime.strptime(end, '%Y-%m-%d')
        days = (end_date - start_date).days
        n_bars = min(days + 10, 5000)  # TradingView limit

        interval_l = str(interval).lower()
        tv_interval = Interval.in_daily
        if interval_l == "15m":
            tv_interval = Interval.in_15_minute
        elif interval_l in {"1h", "60m"}:
            tv_interval = Interval.in_1_hour

        df = None
        for ex in ["HOSE", "HNX", "UPCOM"]:
            try:
                df = tv.get_hist(symbol=symbol, exchange=ex, interval=tv_interval, n_bars=n_bars)
                if df is not None and not df.empty:
                    break
            except Exception:
                continue
        if df is None or df.empty:
            return None

        # TvDatafeed returns: datetime, symbol, open, high, low, close, volume
        df = df.reset_index()
        df = df.rename(columns={
            'datetime': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })

        # Filter by date range
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

        required = ["Date", "Open", "High", "Low", "Close", "Volume"]
        logger.success(f"TradingView: {len(df)} rows for {symbol} ({interval_l})")
        return df[required]

    def get_fundamentals(self, symbol: str) -> Dict[str, float]:
        """Get fundamental data for Vietnam stock."""
        base_symbol = symbol.replace('.VN', '').upper()

        fundamentals = {
            'eps': 0.0,
            'pe': 0.0,
            'pb': 0.0,
            'roe': 0.0,
            'bvps': 0.0,
            'dividend_yield': 0.0,
        }

        if not VNSTOCK_AVAILABLE:
            return fundamentals

        try:
            with _temporary_disable_broken_loopback_proxy():
                stock = Vnstock().stock(symbol=base_symbol, source='VCI')

            # Try financial_ratio() from stock.finance (vnstock v3 style)
            try:
                ratios = stock.finance.ratio(period='quarter', lang='vi')
                
                if ratios is not None and not ratios.empty:
                    # Flatten columns if multi-indexed
                    if isinstance(ratios.columns, pd.MultiIndex):
                        source_cols = [f"{a}_{b}" for a, b in ratios.columns]
                        ratios.columns = source_cols
                    
                    latest = ratios.iloc[0] if len(ratios) > 0 else None
                    if latest is not None:
                        mapping = {
                            'pe': ['Chá»‰ tiÃªu Ä‘á»‹nh giÃ¡_P/E', 'P/E'],
                            'pb': ['Chá»‰ tiÃªu Ä‘á»‹nh giÃ¡_P/B', 'P/B'],
                            'eps': ['Chá»‰ tiÃªu Ä‘á»‹nh giÃ¡_EPS (VND)', 'EPS (VND)'],
                            'roe': ['Chá»‰ tiÃªu kháº£ nÄƒng sinh lá»£i_ROE (%)', 'ROE (%)'],
                            'dividend_yield': ['Chá»‰ tiÃªu kháº£ nÄƒng sinh lá»£i_Tá»· suáº¥t cá»• tá»©c (%)', 'Dividend Yield'],
                            'bvps': ['Chá»‰ tiÃªu Ä‘á»‹nh giÃ¡_BVPS (VND)', 'BVPS (VND)']
                        }
                        
                        for key, possible_cols in mapping.items():
                            for col in possible_cols:
                                if col in ratios.columns:
                                    val = latest[col]
                                    if pd.notna(val) and val != 0:
                                        # Convert ROE and Div Yield to 100-base if they are decimals
                                        if key in ['roe', 'dividend_yield'] and abs(val) < 1:
                                            fundamentals[key] = float(val) * 100
                                        else:
                                            fundamentals[key] = float(val)
                                        break
            except Exception as e:
                logger.debug(f"{base_symbol} - v3 ratio fail: {e}")

            # Fallback to ratio_summary or older methods if needed
            if fundamentals['roe'] == 0:
                try:
                    summary = stock.company.ratio_summary()
                    if summary is not None and not summary.empty:
                        s_latest = summary.iloc[0]
                        if 'roe' in s_latest:
                            fundamentals['roe'] = float(s_latest['roe']) * 100 if s_latest['roe'] < 1 else float(s_latest['roe'])
                except: pass

            # Calculate P/E from price if we have EPS
            if fundamentals['pe'] == 0 and fundamentals['eps'] > 0:
                try:
                    q_df = stock.quote.history(start=(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'), interval='1D')
                    if q_df is not None and not q_df.empty:
                        current_price = q_df.iloc[-1]['close']
                        fundamentals['pe'] = current_price / fundamentals['eps']
                except Exception as e:
                    logger.debug(f"{base_symbol} - price calculation failed: {e}")

        except Exception as e:
            logger.warning(f"Could not fetch fundamentals for {base_symbol}: {e}")

        return fundamentals

    def get_realtime_quote(self, symbol: str):
        """
        Get real-time price quote for Vietnam stock.

        Uses 1H intraday data from vnstock (bypasses cache) to get
        the most current price during market hours.
        Falls back to daily data if intraday is unavailable.
        """
        from .base import RealtimeQuote

        base_symbol = symbol.replace('.VN', '').upper()
        now = datetime.now()

        # Vietnam market hours: 9:00 - 15:00 (UTC+7), Mon-Fri
        hour_utc7 = now.hour  # Assuming system runs in UTC+7/+8
        weekday = now.weekday()  # 0=Mon, 6=Sun
        is_market_open = (weekday < 5) and (9 <= hour_utc7 < 15)

        if VNSTOCK_AVAILABLE:
            try:
                # Strategy 1: Intraday 1H data (most current during market hours)
                with _temporary_disable_broken_loopback_proxy():
                    stock = Vnstock().stock(symbol=base_symbol, source='VCI')
                end_str = (now + timedelta(days=1)).strftime('%Y-%m-%d')
                start_str = (now - timedelta(days=3)).strftime('%Y-%m-%d')

                with _temporary_disable_broken_loopback_proxy():
                    df_1h = stock.quote.history(
                        start=start_str, end=end_str, interval='1H'
                    )

                if df_1h is not None and not df_1h.empty:
                    latest = df_1h.iloc[-1]
                    price = float(latest['close'])

                    # Get previous day's close for change calculation
                    with _temporary_disable_broken_loopback_proxy():
                        df_daily = stock.quote.history(
                            start=(now - timedelta(days=5)).strftime('%Y-%m-%d'),
                            end=end_str, interval='1D'
                        )
                    prev_close = price
                    if df_daily is not None and len(df_daily) >= 2:
                        prev_close = float(df_daily.iloc[-2]['close'])
                    elif df_daily is not None and len(df_daily) >= 1:
                        prev_close = float(df_daily.iloc[-1]['close'])

                    change = ((price - prev_close) / prev_close * 100) if prev_close > 0 else 0.0

                    return RealtimeQuote(
                        symbol=symbol,
                        price=price,
                        change=round(change, 2),
                        prev_close=prev_close,
                        volume=float(latest.get('volume', 0)),
                        high=float(latest.get('high', price)),
                        low=float(latest.get('low', price)),
                        timestamp=now.isoformat(),
                        source='vnstock_intraday',
                        is_market_open=is_market_open,
                    )

            except Exception as e:
                logger.debug(f"VN intraday quote failed for {base_symbol}: {e}")

        # Strategy 2: TvDatafeed real-time
        if TVDATAFEED_AVAILABLE:
            try:
                tv = self._get_tv()
                if tv:
                    df_tv = tv.get_hist(
                        symbol=base_symbol, exchange='HOSE',
                        interval=Interval.in_1_hour, n_bars=24
                    )
                    if df_tv is not None and not df_tv.empty:
                        df_tv = df_tv.reset_index()
                        latest = df_tv.iloc[-1]
                        price = float(latest['close'])
                        prev = df_tv.iloc[-2] if len(df_tv) >= 2 else latest
                        prev_close = float(prev['close'])
                        change = ((price - prev_close) / prev_close * 100) if prev_close > 0 else 0.0

                        return RealtimeQuote(
                            symbol=symbol,
                            price=price,
                            change=round(change, 2),
                            prev_close=prev_close,
                            volume=float(latest.get('volume', 0)),
                            high=float(latest.get('high', price)),
                            low=float(latest.get('low', price)),
                            timestamp=now.isoformat(),
                            source='tvdatafeed',
                            is_market_open=is_market_open,
                        )
            except Exception as e:
                logger.debug(f"TvDatafeed quote failed for {base_symbol}: {e}")

        # Strategy 3: Fall back to base implementation (daily data)
        return super().get_realtime_quote(symbol)

    def supports_fundamentals(self) -> bool:
        return True

    def supports_cot_data(self) -> bool:
        return False


# Auto-register this provider (replace old one)
from . import registry

try:
    _vn_provider = VietnamStockProviderV2()
    registry.register(_vn_provider)
    logger.success(f"Registered {_vn_provider.market_name} provider v2.0 (with cache + TvDatafeed)")
except Exception as e:
    logger.error(f"Failed to register Vietnam provider v2: {e} â€” VN market disabled, other markets still available.")
