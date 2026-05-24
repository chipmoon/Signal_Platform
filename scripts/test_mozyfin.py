"""Test Mozyfin AI chat + stock data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
os.environ["MOZYFIN_API_KEY"] = "mozy_ak_live_6f35fadf98611eff445e1babe95dea47"

from src.mozyfin_client import MozyfinClient

client = MozyfinClient()

# Test 1: Credits check
print("=== Credits ===")
usage = client.get_usage()
print(f"  Used: {usage.get('credits_used')}/{usage.get('credits_cap')}")

# Test 2: Entity data (free - no credits)
print("\n=== VCB Entity (free) ===")
entity = client.search_entity("VCB")
print(f"  Name: {entity.get('short_name')}")
print(f"  Price: {entity.get('current_price'):,.0f}" if entity.get('current_price') else "  Price: N/A")
print(f"  Market Cap: {entity.get('market_cap', 0)/1e12:.1f}T VND" if entity.get('market_cap') else "")

# Test 3: Market indices (free)
print("\n=== Market Indices (free) ===")
indices = client.get_market_indices()
for idx in indices[:5]:
    chg = idx.get('change_percent', 0)
    print(f"  {idx.get('symbol'):<12} {idx.get('current_value'):.2f}  {chg:+.2f}%")

# Test 4: News (free)
print("\n=== VCB News (free) ===")
news = client.get_news("VCB", limit=2)
for n in news:
    print(f"  - {n.get('title', '')[:80]}")

# Test 5: AI Market Overview (uses 1 credit)
print("\n=== AI Market Overview (1 credit) ===")
print("Sending to AI... (may take 30-60s)")
overview = client.get_market_overview()
if overview:
    print(overview)
else:
    print("  (no response)")

# Test 6: AI Stock Analysis (uses 1 credit)
print("\n=== AI Stock Analysis: HPG (1 credit) ===")
print("Sending to AI...")
analysis = client.analyze_stock("HPG")
if analysis:
    print(analysis)
else:
    print("  (no response)")

# Final credits
print("\n=== Credits After ===")
usage2 = client.get_usage()
print(f"  Used: {usage2.get('credits_used')}/{usage2.get('credits_cap')}")
