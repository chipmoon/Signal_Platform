"""
Quick check: How many of the top 100 VN stocks yfinance can serve?
Run: python scripts/check_yfinance_vn_coverage.py
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Top ~120 VN stocks (HOSE + HNX blue-chips)
TOP_VN_STOCKS = [
    # HOSE - Top 60 by market cap
    "VCB", "BID", "CTG", "VIC", "GAS", "SAB", "VHM", "HPG", "MSN", "VNM",
    "TCB", "MBB", "ACB", "VPB", "STB", "FPT", "MWG", "REE", "PLX", "GVR",
    "PNJ", "VJC", "HDB", "LPB", "SSB", "EIB", "KDH", "NVL", "PDR", "DXG",
    "VRE", "POW", "NT2", "PPC", "BSR", "BCM", "VGI", "SHB", "TPB", "OCB",
    "BVH", "DHC", "DBC", "HAH", "PAN", "DPM", "DCM", "GMD", "TLG", "CSV",
    "VHC", "CMG", "EVF", "VPI", "AGR", "SCS", "ASM", "HAG", "TDM", "DGC",
    # HNX - Top 20
    "SHB", "VCS", "PVS", "CEO", "HUT", "NVB", "BVS", "TNG", "PGS", "THD",
    "SHS", "CMS", "VGS", "MBS", "HOR", "DTD", "TV2", "PVL", "L14", "KLF",
    # Additional liquid stocks
    "VCI", "SSI", "HCM", "VND", "BSI", "VDS", "ORS", "CTS", "APS", "FTS",
    "HAX", "VTO", "PVT", "GEX", "SIP", "IJC", "TDC", "PHR", "DPR", "TRC",
    "BMP", "AAA", "LSS", "NHS", "KHP", "BWE", "HDG", "GEG", "PC1", "HDC",
    "CTD", "HBC", "FCN", "VCG", "LCG", "C4G", "HHV", "VNE", "CII", "LIG",
]

# Remove duplicates while preserving order
seen = set()
UNIQUE_STOCKS = []
for s in TOP_VN_STOCKS:
    if s not in seen:
        seen.add(s)
        UNIQUE_STOCKS.append(s)

print(f"Testing {len(UNIQUE_STOCKS)} unique VN symbols on yfinance...\n")

end = datetime.now().strftime("%Y-%m-%d")
start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

ok = []
fail = []
empty = []

for sym in UNIQUE_STOCKS:
    yf_sym = f"{sym}.VN"
    try:
        ticker = yf.Ticker(yf_sym)
        df = ticker.history(start=start, end=end, interval="1d", auto_adjust=False)
        if df.empty:
            empty.append(sym)
        else:
            ok.append((sym, len(df)))
    except Exception as e:
        fail.append((sym, str(e)[:60]))

print(f"✅ SUCCESS ({len(ok)}):")
for sym, rows in sorted(ok):
    print(f"   {sym}.VN → {rows} rows")

print(f"\n⬜ EMPTY ({len(empty)}) — symbol exists but no data:")
for sym in empty:
    print(f"   {sym}.VN")

print(f"\n❌ FAILED ({len(fail)}):")
for sym, err in fail:
    print(f"   {sym}.VN → {err}")

print(f"\n📊 Summary: {len(ok)}/{len(UNIQUE_STOCKS)} available on yfinance")
print(f"   Coverage: {len(ok)/len(UNIQUE_STOCKS)*100:.0f}%")
