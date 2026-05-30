"""
fix_parquet_scale.py - Detect and repair unit-scale discontinuities in cached parquet files.

Root cause: Some VN stock parquet files have a mid-stream crawl format change:
  - Older data: actual VND price (e.g. 93,000)
  - Newer data: price in VND thousands (e.g. 89.91 meaning 89,910 VND)
  - Or vice versa. Some files have 2 breaks (A->B->A format).
  - Some files are completely corrupt (alternating rows).

Fix strategy:
  1. Single break: rescale older segment to match newest segment.
  2. Two breaks: identify 3 segments, keep newest, rescale others.
  3. Many breaks (>2): delete cache file (corrupt, will be re-fetched).

Usage: venv/Scripts/python.exe scripts/fix_parquet_scale.py [--dry-run]
"""
import sys
import os
import warnings
import argparse

os.chdir("D:/Python_VS/trading_system")
sys.path.insert(0, "D:/Python_VS/trading_system")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

CACHE_DIR = Path(".cache/prices")
JUMP_THRESHOLD = 2.0    # log-return magnitude signaling a unit change


def _load_df(path: Path):
    try:
        df = pd.read_parquet(path)
        col_map = {c: c.capitalize() for c in df.columns}
        col_map.update({"Adj close": "Close", "Adj Close": "Close"})
        df = df.rename(columns=col_map)
        if "Date" not in df.columns or "Close" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  [LOAD ERROR] {path.stem}: {e}")
        return None


def _find_all_breaks(close) -> list[tuple[int, float]]:
    """Find all log-return discontinuities. Returns list of (idx, ratio)."""
    log_ret = np.log(close.astype(float) / close.astype(float).shift(1)).fillna(0)
    breaks = []
    big_idx = log_ret.abs()[log_ret.abs() >= JUMP_THRESHOLD].index.tolist()
    for idx in big_idx:
        prev = float(close.iloc[idx - 1])
        if prev == 0:
            continue
        ratio = float(close.iloc[idx]) / prev
        breaks.append((int(idx), ratio))
    return breaks


def _is_scale_ratio(ratio: float) -> bool:
    """Check if a ratio represents a unit-scale change (>100x or <0.005x)."""
    return abs(ratio) > 100 or abs(ratio) < 0.005


def _repair(df) -> tuple:
    """
    Detect and fix scale discontinuity. Returns (df_fixed, repaired, deleted, msg).
    """
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    close = df["Close"].astype(float)
    breaks = _find_all_breaks(close)

    # Filter to only valid scale-change breaks
    scale_breaks = [(idx, r) for idx, r in breaks if _is_scale_ratio(r)]

    if not scale_breaks:
        return df, False, False, "clean"

    n_breaks = len(scale_breaks)

    # Too many breaks: data is corrupt, delete the file
    if n_breaks > 2:
        return df, False, True, f"corrupt ({n_breaks} scale breaks), will delete cache"

    # Single break: rescale everything BEFORE the break to match the tail
    if n_breaks == 1:
        idx, ratio = scale_breaks[0]
        df_out = df.copy()
        df_out.loc[:idx - 1, price_cols] = (
            df.loc[:idx - 1, price_cols].astype(float) * ratio
        )
        new_log = np.log(df_out["Close"].astype(float) / df_out["Close"].astype(float).shift(1)).fillna(0)
        if new_log.abs().max() > JUMP_THRESHOLD:
            return df, False, True, "single-break repair failed, deleting"
        return df_out, True, False, f"1-break fix at row {idx} ratio={ratio:.4f}"

    # Two breaks: three segments A|B|C — keep C (latest), rescale A and B to match C
    # break1: A->B, break2: B->C
    # C scale: close.iloc[-1] is the reference
    # B scale: close.iloc[break2_idx-1] / ratio_B_to_C  -> B values in C units = B * ratio_B_to_C
    # A scale: A values in B units already; rescale to C units = A * ratio_A_to_B * ratio_B_to_C
    if n_breaks == 2:
        idx1, r1 = scale_breaks[0]   # A->B break
        idx2, r2 = scale_breaks[1]   # B->C break

        df_out = df.copy()
        # Segment B is rows idx1..idx2-1: rescale to C units (multiply by r2)
        df_out.loc[idx1:idx2 - 1, price_cols] = (
            df.loc[idx1:idx2 - 1, price_cols].astype(float) * r2
        )
        # Segment A is rows 0..idx1-1: rescale to B units first (multiply by r1),
        # then to C units (multiply by r2)
        df_out.loc[:idx1 - 1, price_cols] = (
            df.loc[:idx1 - 1, price_cols].astype(float) * r1 * r2
        )

        new_log = np.log(df_out["Close"].astype(float) / df_out["Close"].astype(float).shift(1)).fillna(0)
        if new_log.abs().max() > JUMP_THRESHOLD:
            return df, False, True, "2-break repair failed, deleting"
        return df_out, True, False, f"2-break fix at rows {idx1},{idx2}"

    return df, False, False, "unexpected"


def fix_all(dry_run: bool = False) -> dict:
    stats = {"total": 0, "fixed": 0, "clean": 0, "deleted": 0, "errors": 0}
    files = sorted(CACHE_DIR.glob("*.parquet"))
    print(f"Scanning {len(files)} parquet files...\n")

    for path in files:
        stats["total"] += 1
        df = _load_df(path)
        if df is None:
            stats["errors"] += 1
            continue

        df_fixed, repaired, should_delete, msg = _repair(df)

        if should_delete:
            stats["deleted"] += 1
            print(f"  DELETE: {path.stem} — {msg}")
            if not dry_run:
                path.unlink()
        elif repaired:
            stats["fixed"] += 1
            print(f"  FIXED:  {path.stem} — {msg}")
            if not dry_run:
                df_fixed.to_parquet(path, engine="pyarrow", index=False)
        else:
            stats["clean"] += 1

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    mode = "DRY RUN" if args.dry_run else "LIVE FIX"
    print(f"\n{'='*60}")
    print(f"  Parquet Scale Repair Tool [{mode}]")
    print(f"{'='*60}\n")

    s = fix_all(dry_run=args.dry_run)

    print(f"\n{'='*60}")
    print(f"  Total:   {s['total']}")
    print(f"  Fixed:   {s['fixed']}")
    print(f"  Deleted: {s['deleted']}  (corrupt, will re-fetch)")
    print(f"  Clean:   {s['clean']}")
    print(f"  Errors:  {s['errors']}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n  [DRY RUN] No files written. Re-run without --dry-run to apply.\n")
