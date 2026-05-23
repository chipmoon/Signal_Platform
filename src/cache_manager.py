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

import pandas as pd
from loguru import logger


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
