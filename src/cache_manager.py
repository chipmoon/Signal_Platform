"""
Price Data Cache Manager
========================
Local Parquet-based caching for stock price data.
Eliminates slow API calls and provides offline capability.

Architecture:
    .cache/
    ├── stock_list_VN.parquet     # Vietnam stock list cache
    ├── stock_list_TW.parquet     # Taiwan stock list cache
    ├── prices/
    │   ├── VNM_VN.parquet        # Historical OHLCV per symbol
    │   ├── HPG_VN.parquet
    │   └── ...
    └── meta.json                  # Cache metadata (timestamps)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


_SCALE_JUMP_THRESHOLD = 2.0   # log-return magnitude signaling a unit change


def _sanitize_price_scale(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """
    Detect and repair unit-scale discontinuities in a price DataFrame.
    Repairs up to 2 break-points (3-segment data). Drops rows if hopelessly corrupt.
    Silent and fast — runs on every cache read/write to self-heal poisoned data.
    """
    if df.empty or "Close" not in df.columns:
        return df

    close = df["Close"].astype(float)
    log_ret = np.log(close / close.shift(1)).fillna(0)
    big = log_ret.abs()[log_ret.abs() >= _SCALE_JUMP_THRESHOLD]

    if big.empty:
        return df  # Clean — fast path

    # Collect scale-change breaks (ratio > 100x or < 0.005)
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    breaks = []
    for idx in big.index:
        prev = float(close.iloc[idx - 1]) if idx > 0 else 0
        if prev == 0:
            continue
        ratio = float(close.iloc[idx]) / prev
        if abs(ratio) > 100 or abs(ratio) < 0.005:
            breaks.append((int(idx), ratio))

    if not breaks:
        return df  # Jump exists but not a unit-scale issue (e.g. halt/delist)

    if len(breaks) > 2:
        # Hopelessly corrupt: truncate to the latest clean segment after last break
        last_idx = breaks[-1][0]
        logger.warning(f"CacheMgr [{label}]: {len(breaks)} scale breaks, keeping rows {last_idx}+")
        return df.iloc[last_idx:].reset_index(drop=True)

    df_out = df.copy()
    if len(breaks) == 1:
        idx, ratio = breaks[0]
        df_out.loc[:idx - 1, price_cols] = (
            df.loc[:idx - 1, price_cols].astype(float) * ratio
        )
    else:
        idx1, r1 = breaks[0]
        idx2, r2 = breaks[1]
        df_out.loc[idx1:idx2 - 1, price_cols] = (
            df.loc[idx1:idx2 - 1, price_cols].astype(float) * r2
        )
        df_out.loc[:idx1 - 1, price_cols] = (
            df.loc[:idx1 - 1, price_cols].astype(float) * r1 * r2
        )

    # Final sanity check
    new_log = np.log(df_out["Close"].astype(float) / df_out["Close"].astype(float).shift(1)).fillna(0)
    if new_log.abs().max() > _SCALE_JUMP_THRESHOLD:
        # Repair failed: keep only the latest segment
        last_idx = breaks[-1][0]
        logger.warning(f"CacheMgr [{label}]: repair failed, keeping rows {last_idx}+")
        return df.iloc[last_idx:].reset_index(drop=True)

    logger.debug(f"CacheMgr [{label}]: auto-repaired {len(breaks)} scale break(s)")
    return df_out


# ─── Default cache directory ───────────────────────────────────────
_DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".cache",
)


class CacheManager:
    """Manages local Parquet cache for price data and stock lists."""

    def __init__(self, cache_dir: str = _DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.prices_dir = self.cache_dir / "prices"
        self.meta_path = self.cache_dir / "meta.json"

        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.prices_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata
        self._meta = self._load_meta()

    # ─── Metadata ──────────────────────────────────────────────

    def _load_meta(self) -> Dict:
        """Load cache metadata."""
        if self.meta_path.exists():
            try:
                return json.loads(self.meta_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_meta(self) -> None:
        """Persist cache metadata."""
        self.meta_path.write_text(
            json.dumps(self._meta, indent=2, default=str),
            encoding="utf-8",
        )

    # ─── Stock List Cache ──────────────────────────────────────

    def cache_stock_list(
        self, market: str, stocks: List[Dict[str, str]]
    ) -> None:
        """Cache a market's stock list as Parquet."""
        key = f"stock_list_{market}"
        path = self.cache_dir / f"{key}.parquet"

        df = pd.DataFrame(stocks)
        df.to_parquet(path, index=False, engine="pyarrow")

        self._meta[key] = {
            "updated": datetime.now().isoformat(),
            "count": len(stocks),
        }
        self._save_meta()
        logger.info(f"Cached {len(stocks)} {market} stocks → {path.name}")

    def get_cached_stock_list(
        self, market: str, max_age_hours: int = 24
    ) -> Optional[List[Dict[str, str]]]:
        """
        Get cached stock list if it exists and isn't too old.

        Args:
            market: Market ID (e.g., 'VN')
            max_age_hours: Maximum cache age in hours

        Returns:
            List of stock dicts or None if cache is stale/missing
        """
        key = f"stock_list_{market}"
        path = self.cache_dir / f"{key}.parquet"

        if not path.exists():
            return None

        # Check age
        meta = self._meta.get(key, {})
        if meta:
            updated = datetime.fromisoformat(meta["updated"])
            age = datetime.now() - updated
            if age > timedelta(hours=max_age_hours):
                logger.info(f"{market} stock list cache expired ({age})")
                return None

        try:
            df = pd.read_parquet(path, engine="pyarrow")
            return df.to_dict("records")
        except Exception as e:
            logger.warning(f"Failed to read cache {path}: {e}")
            return None

    # ─── Price Data Cache ──────────────────────────────────────

    def _price_path(self, symbol: str, market: str) -> Path:
        """Get cache file path for a symbol's price data."""
        safe_name = symbol.replace(".", "_").replace("=", "_").replace("-", "_")
        return self.prices_dir / f"{safe_name}_{market}.parquet"

    def cache_price_data(
        self, symbol: str, market: str, df: pd.DataFrame
    ) -> None:
        """Cache price data for a symbol."""
        path = self._price_path(symbol, market)

        if df.empty:
            return

        # Merge with existing cache (append new data)
        existing = self.get_cached_price_data(symbol, market, max_age_hours=999999)
        if existing is not None and not existing.empty:
            # Combine and deduplicate by Date
            combined = pd.concat([existing, df], ignore_index=True)
            if "Date" in combined.columns:
                combined["Date"] = pd.to_datetime(combined["Date"])
                combined = combined.drop_duplicates(subset=["Date"], keep="last")
                combined = combined.sort_values("Date").reset_index(drop=True)
            df = combined

        # Always sanitize before writing to prevent scale-corrupt data persisting
        df = _sanitize_price_scale(df, label=f"{symbol}_{market}_write")

        df.to_parquet(path, index=False, engine="pyarrow")

        key = f"price_{symbol}_{market}"
        self._meta[key] = {
            "updated": datetime.now().isoformat(),
            "rows": len(df),
            "start": str(df["Date"].min()) if "Date" in df.columns else "",
            "end": str(df["Date"].max()) if "Date" in df.columns else "",
        }
        self._save_meta()

    def get_cached_price_data(
        self,
        symbol: str,
        market: str,
        max_age_hours: int = 24,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get cached price data for a symbol.

        Args:
            symbol: Stock symbol
            market: Market ID
            max_age_hours: Maximum cache age
            start: Optional start date filter (YYYY-MM-DD)
            end: Optional end date filter (YYYY-MM-DD)

        Returns:
            DataFrame or None if not cached/stale
        """
        path = self._price_path(symbol, market)

        if not path.exists():
            return None

        # Check age
        key = f"price_{symbol}_{market}"
        meta = self._meta.get(key, {})
        if meta:
            updated = datetime.fromisoformat(meta["updated"])
            age = datetime.now() - updated
            if age > timedelta(hours=max_age_hours):
                return None

        try:
            df = pd.read_parquet(path, engine="pyarrow")

            # Sanitize scale discontinuities on every read (self-healing)
            df = _sanitize_price_scale(df, label=f"{symbol}_{market}_read")

            # Apply date filters
            if "Date" in df.columns and (start or end):
                df["Date"] = pd.to_datetime(df["Date"])
                if start:
                    df = df[df["Date"] >= pd.Timestamp(start)]
                if end:
                    df = df[df["Date"] <= pd.Timestamp(end)]

            return df if not df.empty else None

        except Exception as e:
            logger.warning(f"Failed to read price cache {path}: {e}")
            return None

    # ─── Fundamental Data Cache ────────────────────────────────
    
    def cache_fundamentals(self, symbol: str, market: str, data: dict) -> None:
        """Cache fundamental data for a symbol as JSON."""
        fundamentals_dir = self.cache_dir / "fundamentals"
        fundamentals_dir.mkdir(parents=True, exist_ok=True)
        clean_symbol = symbol.replace(".VN", "").replace(".TW", "").replace(".TWO", "").upper()
        safe_name = clean_symbol.replace(".", "_").replace("=", "_").replace("-", "_")
        path = fundamentals_dir / f"{safe_name}_{market}.json"
        
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        
        key = f"fundamentals_{clean_symbol}_{market}"
        self._meta[key] = {
            "updated": datetime.now().isoformat(),
        }
        self._save_meta()
        logger.info(f"Cached fundamentals for {clean_symbol} ({market})")

    def get_cached_fundamentals(
        self, symbol: str, market: str, max_age_hours: int = 168
    ) -> Optional[dict]:
        """
        Get cached fundamental data if it exists and is fresh.
        Default max age is 7 days since fundamentals change slowly.
        """
        clean_symbol = symbol.replace(".VN", "").replace(".TW", "").replace(".TWO", "").upper()
        safe_name = clean_symbol.replace(".", "_").replace("=", "_").replace("-", "_")
        path = self.cache_dir / "fundamentals" / f"{safe_name}_{market}.json"
        
        if not path.exists():
            return None
            
        key = f"fundamentals_{clean_symbol}_{market}"
        meta = self._meta.get(key, {})
        if meta:
            try:
                updated = datetime.fromisoformat(meta["updated"])
                age = datetime.now() - updated
                if age > timedelta(hours=max_age_hours):
                    return None
            except Exception:
                pass
                
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to read fundamental cache {path}: {e}")
            return None

    # ─── Cache Stats ───────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        price_files = list(self.prices_dir.glob("*.parquet"))
        list_files = list(self.cache_dir.glob("stock_list_*.parquet"))

        total_size = sum(f.stat().st_size for f in price_files + list_files)

        return {
            "cache_dir": str(self.cache_dir),
            "stock_lists": len(list_files),
            "price_files": len(price_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "metadata_entries": len(self._meta),
        }

    def clear(self, market: Optional[str] = None) -> int:
        """
        Clear cache files.

        Args:
            market: If specified, only clear this market. Otherwise clear all.

        Returns:
            Number of files deleted
        """
        count = 0

        if market:
            # Clear specific market
            pattern = f"*_{market}.parquet"
            for f in self.prices_dir.glob(pattern):
                f.unlink()
                count += 1
            list_file = self.cache_dir / f"stock_list_{market}.parquet"
            if list_file.exists():
                list_file.unlink()
                count += 1
        else:
            # Clear all
            for f in self.prices_dir.glob("*.parquet"):
                f.unlink()
                count += 1
            for f in self.cache_dir.glob("stock_list_*.parquet"):
                f.unlink()
                count += 1

        # Reset metadata
        self._meta = {}
        self._save_meta()

        logger.info(f"Cleared {count} cache files")
        return count


# ─── Module-level singleton ───────────────────────────────────────
cache = CacheManager()
