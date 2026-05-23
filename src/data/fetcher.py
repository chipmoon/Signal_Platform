"""
Data Fetcher
=============
Fetches price data (Yahoo Finance), COT reports (CFTC), and USD Index.
Provides clean DataFrames ready for strategy consumption.

Follows:
- Python Data Science: vectorized ops, explicit dtypes, chained transforms
- Python Security: no hardcoded secrets, validated inputs
- Python Best Practices: EAFP, Google docstrings, typed returns
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from loguru import logger

try:
    import yfinance as yf
except ImportError:
    yf = None

from src.config import (
    CFTC_CURRENT_URL,
    CFTC_LEGACY_URL_TEMPLATE,
    MARKET_CODES,
)


# ─────────────────────────────────────────────
# 1. Price & Volume Data (Yahoo Finance)
# ─────────────────────────────────────────────

def fetch_price_data(
    ticker: str = "GC=F",
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance.

    Args:
        ticker: Yahoo Finance symbol (default: Gold Futures GC=F).
        start: Start date string ``YYYY-MM-DD``. Defaults to 3 years ago.
        end: End date string ``YYYY-MM-DD``. Defaults to today.
        interval: Data granularity (1d, 1h, etc.).

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume, Date.
    """
    end_date = end or (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = start or (datetime.now() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

    logger.info(f"Fetching price data: {ticker} from {start_date} to {end_date} ({interval})")

    if yf is None:
        logger.error("yfinance is not installed. Cannot fetch price data.")
        return pd.DataFrame()

    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)

        if data.empty:
            logger.error(f"No data returned for {ticker}")
            return pd.DataFrame()

        df = data.reset_index()

        # Flatten MultiIndex columns (yfinance sometimes returns these)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[1] == "" else col[0] for col in df.columns]

        # Ensure Date column
        if "Date" not in df.columns and "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        logger.success(f"Fetched {len(df)} rows of price data for {ticker}")
        return df

    except Exception as exc:
        logger.error(f"Failed to fetch price data for {ticker}: {exc}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# 2. COT Report Data (CFTC)
# ─────────────────────────────────────────────

def fetch_cot_data(
    commodity: str = "GOLD",
    years: int = 3,
    *,
    allow_synthetic: bool = False,
) -> pd.DataFrame:
    """Fetch COT (Commitments of Traders) data from CFTC.

    Args:
        commodity: Commodity name key (e.g. ``GOLD``, ``SILVER``).
        years: Number of years of history to fetch.

    Returns:
        DataFrame with COT positions by date.
    """
    market_code = MARKET_CODES.get(commodity.upper())
    if not market_code:
        supported = ", ".join(MARKET_CODES)
        logger.warning(f"Unknown commodity '{commodity}'. Supported: {supported}")
        if allow_synthetic:
            return _generate_synthetic_cot(commodity, years)
        return pd.DataFrame()

    current_year = datetime.now().year
    all_data: list[pd.DataFrame] = []

    for year in range(current_year - years + 1, current_year + 1):
        url = (
            CFTC_CURRENT_URL if year == current_year
            else CFTC_LEGACY_URL_TEMPLATE.format(year=year)
        )
        logger.debug(f"Downloading COT data for {year}: {url}")

        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as csv_file:
                    raw = pd.read_csv(csv_file)

            filtered = raw[
                raw["CFTC_Contract_Market_Code"].astype(str).str.strip() == market_code
            ].copy()

            if filtered.empty:
                logger.debug(f"No data for {commodity} in {year}")
                continue

            cot = pd.DataFrame({
                "Date": pd.to_datetime(filtered["As_of_Date_In_Form_YYMMDD"], format="%y%m%d"),
                "Open_Interest": filtered["Open_Interest_All"].astype(float),
                "Commercial_Long": filtered["Comm_Positions_Long_All"].astype(float),
                "Commercial_Short": filtered["Comm_Positions_Short_All"].astype(float),
                "NonComm_Long": filtered["NonComm_Positions_Long_All"].astype(float),
                "NonComm_Short": filtered["NonComm_Positions_Short_All"].astype(float),
            })
            all_data.append(cot)

        except requests.exceptions.HTTPError:
            logger.warning(f"Failed to fetch COT for {year}: HTTP {resp.status_code}")
        except Exception as exc:
            logger.warning(f"Error processing COT for {year}: {exc}")

    if not all_data:
        if allow_synthetic:
            logger.warning("No COT data fetched from CFTC. Generating synthetic COT data for backtesting.")
            return _generate_synthetic_cot(commodity, years)
        logger.error("No COT data fetched from CFTC. Fail-closed mode active (synthetic disabled).")
        return pd.DataFrame()

    result = (
        pd.concat(all_data, ignore_index=True)
        .sort_values("Date")
        .reset_index(drop=True)
        .assign(
            Commercial_Net=lambda x: x["Commercial_Long"] - x["Commercial_Short"],
            Commercial_Short_Ratio=lambda x: x["Commercial_Short"] / x["Open_Interest"].replace(0, np.nan),
            Short_Change=lambda x: x["Commercial_Short"].diff(),
            Short_Change_Pct=lambda x: x["Commercial_Short"].pct_change() * 100,
        )
    )

    logger.success(f"Fetched {len(result)} COT records for {commodity}")
    return result


def _generate_synthetic_cot(commodity: str, years: int) -> pd.DataFrame:
    """Generate realistic synthetic COT data for backtesting.

    Used as fallback when CFTC data is unavailable (network issues, rate limits).

    Args:
        commodity: Commodity name for logging.
        years: Number of years of data to generate.

    Returns:
        DataFrame matching the schema of real COT data.
    """
    logger.info(f"Generating {years} years of synthetic COT data for {commodity}...")

    weeks = years * 52
    np.random.seed(42)

    dates = pd.date_range(
        end=datetime.now(),
        periods=weeks,
        freq="W-TUE",
    )

    base_oi = 500_000
    oi = base_oi + np.cumsum(np.random.normal(0, 5_000, weeks))
    oi = np.clip(oi, base_oi * 0.5, base_oi * 2).astype(int)

    comm_short_ratio = 0.30 + 0.08 * np.sin(np.arange(weeks) * 2 * np.pi / 52)
    comm_short_ratio += np.random.normal(0, 0.02, weeks)
    comm_short = (oi * np.clip(comm_short_ratio, 0.15, 0.55)).astype(int)
    comm_long = (comm_short * np.random.uniform(0.7, 1.3, weeks)).astype(int)

    # Inject anomaly events
    num_anomalies = max(3, weeks // 30)
    anomaly_indices = np.random.choice(range(10, weeks - 5), size=num_anomalies, replace=False)
    for idx in anomaly_indices:
        spike = np.random.uniform(1.5, 2.5)
        comm_short[idx : idx + 3] = (comm_short[idx] * spike).astype(int)

    result = (
        pd.DataFrame({
            "Date": dates,
            "Open_Interest": oi,
            "Commercial_Long": comm_long,
            "Commercial_Short": comm_short,
            "NonComm_Long": (oi * np.random.uniform(0.15, 0.25, weeks)).astype(int),
            "NonComm_Short": (oi * np.random.uniform(0.15, 0.25, weeks)).astype(int),
        })
        .assign(
            Commercial_Net=lambda x: x["Commercial_Long"] - x["Commercial_Short"],
            Commercial_Short_Ratio=lambda x: x["Commercial_Short"] / x["Open_Interest"].replace(0, np.nan),
            Short_Change=lambda x: x["Commercial_Short"].diff(),
            Short_Change_Pct=lambda x: x["Commercial_Short"].pct_change() * 100,
        )
    )

    logger.success(f"Generated {len(result)} synthetic COT records with {num_anomalies} anomaly events.")
    return result


# ─────────────────────────────────────────────
# 3. USD Index Data
# ─────────────────────────────────────────────

def fetch_usd_index(
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch USD Index (DX-Y.NYB) from Yahoo Finance."""
    return fetch_price_data(ticker="DX-Y.NYB", start=start, end=end)


# ─────────────────────────────────────────────
# 4. Data Merger
# ─────────────────────────────────────────────

def merge_price_and_cot(
    price_df: pd.DataFrame,
    cot_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge daily price data with weekly COT data using forward-fill.

    Each trading day gets the most recent COT reading.

    Args:
        price_df: Daily OHLCV DataFrame with ``Date`` column.
        cot_df: Weekly COT DataFrame with ``Date`` column.

    Returns:
        Merged DataFrame sorted by date.
    """
    if cot_df.empty:
        logger.warning("COT data is empty. Returning price data only.")
        return price_df

    merged = pd.merge_asof(
        price_df.sort_values("Date"),
        cot_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    merged = merged.ffill().bfill()

    logger.info(f"Merged dataset: {len(merged)} rows with COT context.")
    return merged
