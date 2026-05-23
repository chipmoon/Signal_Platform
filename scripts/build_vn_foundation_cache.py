import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
from loguru import logger

# Add root directory to sys.path
sys.path.append(os.getcwd())

from src.plugins import registry
from src.cache_manager import cache

def build_foundation_cache():
    logger.info("Initializing Vietnam Foundation Data Collection...")
    
    # 1. Ensure VN Provider is registered
    provider = registry.get("VN")
    if not provider:
        logger.error("Vietnam Provider not found")
        return
        
    # 2. Get VN30 List (or use hardcoded if API fails)
    try:
        from vnstock import Listing
        vn30_symbols = Listing().symbols_by_group('VN30')['symbol'].tolist()
    except Exception as e:
        logger.warning(f"Failed to fetch VN30 list from API: {e}. Using fallback.")
        vn30_symbols = [
            "ACB", "BID", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG", "MBB", "MSN", 
            "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB", "TCB", "TPB", 
            "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE", "BVH", "BCM"
        ]
        
    logger.info(f"Targeting {len(vn30_symbols)} stocks for Foundation Model.")
    
    # 3. Fetch Index Data (VNI) for Relative Strength
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d")
    
    logger.info(f"Fetching VN-Index (VNI) for timeframe: {start_date} to {end_date}")
    try:
        vni_df = provider.get_price_data("VNINDEX", start_date, end_date)
        vni_df = vni_df.rename(columns={"Close": "VNI_Close"})[["Date", "VNI_Close"]]
        vni_df["Date"] = pd.to_datetime(vni_df["Date"])
    except Exception as e:
        logger.error(f"Critical Error: Failed to fetch VN-Index: {e}")
        return

    # 4. Fetch Each Stock and Build Big Data Matrix
    all_data = []
    
    for symbol in vn30_symbols:
        try:
            logger.info(f"Downloading {symbol}...")
            df = provider.get_price_data(symbol, start_date, end_date)
            if df.empty:
                continue
            
            df["Date"] = pd.to_datetime(df["Date"])
            df["Symbol"] = symbol
            
            # Merge with VNI for Relative Strength
            df = df.merge(vni_df, on="Date", how="inner")
            
            # Relative Strength Feature: Stock / Index
            df["Rel_Strength"] = df["Close"] / df["VNI_Close"]
            
            all_data.append(df)
            logger.success(f"Added {len(df)} rows for {symbol}")
            
        except Exception as e:
            logger.warning(f"Skipping {symbol} due to error: {e}")

    if not all_data:
        logger.error("No data collected!")
        return

    # 5. Consolidate and Save
    master_df = pd.concat(all_data, ignore_index=True)
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    output_path = "data/foundation_vn_3y.parquet"
    master_df.to_parquet(output_path, index=False)
    
    logger.success(f"--- BIG DATA COLLECTION COMPLETE ---")
    logger.info(f"Total Rows: {len(master_df)}")
    logger.info(f"Total Stocks: {master_df['Symbol'].nunique()}")
    logger.info(f"Saved to: {output_path}")

if __name__ == "__main__":
    build_foundation_cache()
