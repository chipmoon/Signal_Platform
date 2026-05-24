"""
Nightly Full-Universe Pre-Scan Pipeline
========================================
Chay moi toi luc 21:00 (Taiwan UTC+8) tren may local:
  1. Fetch full VN universe tu vnstock (khong bi chan IP khi chay local)
  2. Chay AlphaScannerEngine (full Tier1 + Tier2 deep scan)
  3. Output top 20 VN candidates sorted by institutional_score
  4. Push ket qua JSON + OHLCV parquet cache len GitHub
  5. Streamlit Cloud doc JSON -> hien thi ngay, khong can goi API

Usage:
    python scripts/nightly_full_scan.py             # Scan + save, no push
    python scripts/nightly_full_scan.py --push      # Scan + save + git push
    python scripts/nightly_full_scan.py --top 20    # Top N results (default: 20)
    python scripts/nightly_full_scan.py --dry-run   # Test without saving

Setup Task Scheduler: chay setup_nightly_task.ps1 (Admin)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SCAN_RESULTS_PATH = DATA_DIR / "nightly_scan_results.json"
SCAN_META_PATH = DATA_DIR / "nightly_scan_meta.json"


def _fetch_full_vn_universe() -> list[str]:
    """
    Fetch the complete VN stock universe from vnstock API (local only).
    Falls back to extended hardcoded list if API fails.
    """
    try:
        from vnstock import Listing
        logger.info("Fetching full VN universe from vnstock...")
        listing = Listing()
        df = listing.symbols_by_exchange(lang='vi')

        # Filter to HOSE + HNX only
        if 'exchange' in df.columns:
            df = df[df['exchange'].isin(['HOSE', 'HNX'])].copy()

        # Filter to STOCK type only
        if 'type' in df.columns:
            for t in ['STOCK', 'Stock', 'stock']:
                if (df['type'] == t).sum() > 0:
                    df = df[df['type'] == t].copy()
                    break

        symbols = sorted(df['symbol'].dropna().unique().tolist())
        logger.success(f"Full VN universe: {len(symbols)} symbols from vnstock API")
        return symbols

    except Exception as e:
        logger.warning(f"vnstock API failed: {e} — using extended hardcoded list")
        # Import the extended list from alpha_scanner as fallback
        from src.strategies.alpha_scanner import VN_UNIVERSE_EXTENDED
        return VN_UNIVERSE_EXTENDED


def _run_full_scan(vn_symbols: list[str], top_n: int = 20) -> list[dict]:
    """
    Run AlphaScannerEngine with nightly-optimized config:
    - Shuffles symbols to avoid alphabetical bias (A-B dominating)
    - Overrides scan_total_timeout_sec 50s -> 1800s (30 min budget)
    - Increases worker counts for faster batch throughput
    """
    import random
    import src.strategies.alpha_scanner as scanner_mod

    original_extended = scanner_mod.VN_UNIVERSE_EXTENDED[:]
    original_config = scanner_mod.SCAN_CONFIG.copy()

    try:
        # Shuffle to cover all alphabet ranges equally
        shuffled = vn_symbols[:]
        random.shuffle(shuffled)
        scanner_mod.VN_UNIVERSE_EXTENDED = shuffled

        # Override timeouts for nightly batch (no UI waiting)
        scanner_mod.SCAN_CONFIG = {
            **original_config,
            "tier1_max_workers": 12,
            "tier1_max_workers_vn": 8,
            "tier1_timeout_sec": 600,
            "tier2_max_workers": 8,
            "tier2_max_workers_vn": 6,
            "tier2_timeout_sec": 900,
            "scan_total_timeout_sec": 1800,
            "tier1_top_n": 120,
            "wyckoff_top_n": 80,
            "tier2_top_n": top_n * 3,
        }

        logger.info(f"Running full scan: {len(vn_symbols)} VN symbols (shuffled, 30min budget)...")

        from src.strategies.alpha_scanner import AlphaScannerEngine

        start_time = datetime.now()

        def _progress(p: float):
            elapsed = int((datetime.now() - start_time).total_seconds())
            pct = int(p * 100)
            logger.info(f"  Scan progress: {pct}% | {elapsed}s elapsed")

        engine = AlphaScannerEngine(
            extended_universe=True,
            commodities=False,
            market_scope="VN",
        )
        results = engine.scan_universe(progress_callback=_progress)

        elapsed = int((datetime.now() - start_time).total_seconds())
        logger.success(f"Scan complete: {len(results)} candidates in {elapsed}s")

        results.sort(key=lambda x: x.get("institutional_score", -999), reverse=True)
        return results[:top_n]

    finally:
        scanner_mod.VN_UNIVERSE_EXTENDED = original_extended
        scanner_mod.SCAN_CONFIG = original_config

def _save_results(results: list[dict], top_n: int) -> None:
    """Save scan results as JSON for Streamlit Cloud to read."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Clean results for JSON serialization
    clean = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "_df"}
        # Convert numpy types to native Python
        for k, v in row.items():
            if hasattr(v, 'item'):
                row[k] = v.item()
            elif isinstance(v, float) and (v != v):  # NaN check
                row[k] = None
        clean.append(row)

    SCAN_RESULTS_PATH.write_text(
        json.dumps(clean, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8"
    )

    # Save metadata
    meta = {
        "generated_at": datetime.now().isoformat(),
        "generated_at_readable": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "top_n": top_n,
        "total_candidates": len(clean),
        "scan_date": datetime.now().strftime("%Y-%m-%d"),
        "symbols": [r.get("symbol", "") for r in clean],
    }
    SCAN_META_PATH.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    logger.success(f"Saved {len(clean)} results → {SCAN_RESULTS_PATH.name}")


def _git_push() -> bool:
    """Commit scan results and OHLCV cache, then push to GitHub."""
    try:
        # Stage results + cache
        files_to_add = [
            str(DATA_DIR),
            str(PROJECT_ROOT / ".cache"),
        ]
        subprocess.run(
            ["git", "add"] + files_to_add,
            cwd=PROJECT_ROOT, check=True, capture_output=True
        )

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        result = subprocess.run(
            ["git", "commit", "-m", f"data: nightly scan + cache update {timestamp}"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8"
        )
        if "nothing to commit" in result.stdout:
            logger.info("No changes to push")
            return True

        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=PROJECT_ROOT, check=True, capture_output=True
        )
        logger.success(f"Pushed to GitHub: {timestamp}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Git push failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Nightly full VN alpha scan")
    parser.add_argument("--push", action="store_true", help="Push results to GitHub")
    parser.add_argument("--top", type=int, default=20, help="Top N results to save (default: 20)")
    parser.add_argument("--dry-run", action="store_true", help="Run scan but don't save")
    parser.add_argument("--no-fetch", action="store_true", help="Use extended hardcoded list only")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"Nightly VN Alpha Scan — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Step 1: Get full universe
    if args.no_fetch:
        from src.strategies.alpha_scanner import VN_UNIVERSE_EXTENDED
        vn_symbols = VN_UNIVERSE_EXTENDED
        logger.info(f"Using hardcoded extended list: {len(vn_symbols)} symbols")
    else:
        vn_symbols = _fetch_full_vn_universe()

    # Also run nightly OHLCV cache for all symbols (feeds scanner's cache)
    logger.info("Pre-fetching OHLCV cache for all symbols...")
    try:
        import importlib
        cache_script = importlib.util.spec_from_file_location(
            "nightly_vn_cache",
            str(PROJECT_ROOT / "scripts" / "nightly_vn_cache.py")
        )
        cache_mod = importlib.util.module_from_spec(cache_script)
        # Override symbols with full list
        cache_mod_symbols = vn_symbols
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        from scripts.nightly_vn_cache import (
            fetch_with_vnstock, fetch_with_yfinance, save_cache,
            update_stock_list_cache
        )
        cache_dir = PROJECT_ROOT / ".cache"
        update_stock_list_cache(vn_symbols, cache_dir)

        ok, fail = 0, 0
        for sym in vn_symbols:
            df = fetch_with_vnstock(sym, start, end)
            if df is None or df.empty:
                df = fetch_with_yfinance(sym, start, end)
            if df is not None and not df.empty:
                save_cache(sym, df, cache_dir)
                ok += 1
            else:
                fail += 1
        logger.success(f"OHLCV cache: {ok} OK, {fail} failed")
    except Exception as e:
        logger.warning(f"OHLCV pre-fetch failed (scan will use existing cache): {e}")

    # Step 2: Run full alpha scan
    results = _run_full_scan(vn_symbols, top_n=args.top)

    if not results:
        logger.error("Scan returned no results!")
        return

    # Step 3: Save
    if not args.dry_run:
        _save_results(results, args.top)
    else:
        logger.info("[DRY RUN] Would save:")
        for i, r in enumerate(results[:5], 1):
            logger.info(f"  {i}. {r.get('symbol')} — {r.get('recommendation')} | inst={r.get('institutional_score', 0):.3f}")

    # Step 4: Push
    if args.push and not args.dry_run:
        _git_push()

    logger.success("Nightly scan complete!")


if __name__ == "__main__":
    main()
