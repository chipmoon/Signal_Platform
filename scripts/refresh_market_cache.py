from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.plugins import registry


def refresh_market(
    market: str, days: int, chunk_size: int, pause_sec: int, max_symbols: int | None
) -> None:
    provider = registry.get(market)
    if not provider:
        raise ValueError(f"Provider not found: {market}")

    end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    symbols = provider.search_assets("", limit=5000)
    if market == "TW" and not symbols:
        tw_cache = Path(".cache") / "stock_list_TW.parquet"
        if tw_cache.exists():
            df = pd.read_parquet(tw_cache)
            from src.plugins.base import AssetInfo
            symbols = [
                AssetInfo(
                    symbol=row["symbol"],
                    name=str(row.get("name", row["symbol"])),
                    market="TW",
                    sector=str(row.get("sector", "Taiwan Stock")),
                    exchange=str(row.get("exchange", "TWSE/TPE")),
                    currency="TWD",
                )
                for _, row in df.iterrows()
                if isinstance(row.get("symbol", ""), str) and row["symbol"]
            ]
    if max_symbols is not None and max_symbols > 0:
        symbols = symbols[:max_symbols]
    total = len(symbols)
    ok = 0
    fail = 0

    logger.info(f"{market}: refreshing {total} symbols from {start} to {end}")
    for idx, asset in enumerate(symbols, 1):
        try:
            df = provider.get_price_data(asset.symbol, start, end)
            if df is not None and not df.empty:
                ok += 1
            else:
                fail += 1
        except Exception as ex:
            fail += 1
            logger.warning(f"{market}:{asset.symbol} failed: {ex}")

        if idx % chunk_size == 0:
            logger.info(f"{market}: progress {idx}/{total} | ok={ok} fail={fail} | pause {pause_sec}s")
            time.sleep(pause_sec)

    logger.success(f"{market}: done | ok={ok} fail={fail}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh cached market price data in chunks.")
    parser.add_argument("--markets", nargs="+", default=["VN", "TW"], help="Market IDs, e.g. VN TW")
    parser.add_argument("--days", type=int, default=120, help="Number of lookback days")
    parser.add_argument("--chunk-size", type=int, default=40, help="Symbols per chunk before pausing")
    parser.add_argument("--pause-sec", type=int, default=20, help="Pause seconds between chunks")
    parser.add_argument("--max-symbols", type=int, default=0, help="Optional cap for testing")
    args = parser.parse_args()

    # Disable broken local proxy defaults that block data providers
    for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        os.environ[k] = ""
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["HOME"] = os.getcwd()
    os.environ["USERPROFILE"] = os.getcwd()

    for market in args.markets:
        cap = args.max_symbols if args.max_symbols > 0 else None
        refresh_market(market.upper(), args.days, args.chunk_size, args.pause_sec, cap)


if __name__ == "__main__":
    main()
