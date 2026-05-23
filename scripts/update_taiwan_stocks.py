
import requests
import pandas as pd
from io import StringIO
import sys
import os
import urllib3
from loguru import logger

# Add project root to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cache_manager import cache

def fetch_taiwan_stock_list():
    """Fetch all Taiwan stock symbols from TWSE and TPEx."""
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    # URLs for TWSE and TPEx listings (Big5 encoded)
    urls = [
        ("TWSE", "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"), # TWSE (Standard)
        ("TPEx", "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"), # TPEx (OTC)
    ]
    
    all_stocks = []
    
    for exchange, url in urls:
        try:
            logger.info(f"Fetching {exchange} listing...")
            r = requests.get(url, timeout=20, verify=False)
            r.encoding = 'Big5'
            
            # Simple HTML table parsing (these pages are basically one big table)
            # Each row has columns: [Code + Name, ISIN, Date, Market, Sector, CFI, Remarks]
            # Example: "1101  台泥" is in column 0.
            
            # Use pandas read_html to parse tables
            dfs = pd.read_html(StringIO(r.text))
            if not dfs: continue
            
            df = dfs[0]
            # Column 0: 有價證券代號及名稱
            # Column 4: 產業別 (Sector)
            
            import re
            for _, row in df.iterrows():
                val = str(row.iloc[0]).strip()
                # Most rows look like "1101  台泥" or "2330 台積電"
                # Use regex to find 4-6 digit code at the start
                match = re.match(r'^(\d{4})\s+(.+)$', val)
                if match:
                    code = match.group(1).strip()
                    name = match.group(2).strip()
                    sector = str(row.iloc[4]) if len(row) > 4 else "Other"
                    
                    # Store
                    suffix = ".TW" if exchange == "TWSE" else ".TWO"
                    all_stocks.append({
                        "symbol": f"{code}{suffix}",
                        "name": name,
                        "market": "TW",
                        "sector": sector,
                        "exchange": exchange,
                        "currency": "TWD"
                    })
            
            logger.success(f"Extracted {len(all_stocks)} {exchange} stocks.")
            
        except Exception as e:
            logger.error(f"Failed to fetch {exchange}: {e}")
            
    # Deduplicate and sort
    if all_stocks:
        dedup = {}
        for s in all_stocks:
            dedup[s["symbol"]] = s
        all_stocks = list(dedup.values())
        # Save to cache
        cache.cache_stock_list("TW", all_stocks)
        logger.success(f"Final Count: {len(all_stocks)} Taiwan stocks cached.")
        return True
    return False

if __name__ == "__main__":
    fetch_taiwan_stock_list()
