
import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.strategies.smc_analyzer import SmcAnalyzer
from src.config import SmcConfig

def get_smc_zones(symbol="4919.TW"):
    logger.info(f"Analyzing SMC zones for {symbol}")
    
    # Fetch data
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="1y", interval="1d")
    
    if df.empty:
        logger.error("No data found")
        return
    
    # Run SMC Analysis
    smc = SmcAnalyzer(SmcConfig())
    state = smc.get_current_state(df)
    
    curr_price = df['Close'].iloc[-1]
    logger.info(f"Current Price: {curr_price:.2f}")
    
    print(f"\n--- BUY ZONES (BULLISH) for {symbol} ---")
    
    # 1. Bullish Order Blocks (Demand Zones)
    bull_obs = state.get('bull_obs', [])
    if bull_obs:
        print("\n[Bullish Order Blocks - Demand]")
        # Sort by proximity to current price (below)
        below_price = [ob for ob in bull_obs if ob['top'] < curr_price]
        below_price = sorted(below_price, key=lambda x: x['top'], reverse=True)
        
        for i, ob in enumerate(below_price[:3]):
            print(f"{i+1}. Range: {ob['bottom']:.2f} - {ob['top']:.2f} (Dist: {((curr_price - ob['top'])/curr_price*100):.1f}%)")
    else:
        print("\nNo Bullish OBs found below current price.")
        
    # 2. Bullish Fair Value Gaps (Inbalances)
    bull_fvgs = state.get('bull_fvgs', [])
    if bull_fvgs:
        print("\n[Bullish Fair Value Gaps - Liquidity Voids]")
        below_fvg = [fvg for fvg in bull_fvgs if fvg['top'] < curr_price]
        below_fvg = sorted(below_fvg, key=lambda x: x['top'], reverse=True)
        
        for i, fvg in enumerate(below_fvg[:3]):
            print(f"{i+1}. Range: {fvg['bottom']:.2f} - {fvg['top']:.2f} (Dist: {((curr_price - fvg['top'])/curr_price*100):.1f}%)")
    else:
        print("\nNo Bullish FVGs found below current price.")

    # 3. Liquidity Voids (Gaps that need filling)
    print("\n[Strategic Context]")
    print(f"Price is currently trading above {len(below_price)} Bullish OBs.")
    print(f"Nearest Major Demand: {below_price[0]['top']:.2f} if found." if below_price else "No immediate demand zone nearby.")

if __name__ == "__main__":
    get_smc_zones("4919.TW")
