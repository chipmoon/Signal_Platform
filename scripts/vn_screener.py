r"""
Vietnamese Stock Screener — Option A (CLI Script)
=================================================
Vận hành bộ lọc Alpha Scanner trên toàn bộ sàn chứng khoán Việt Nam (VN_UNIVERSE_EXTENDED).
Xuất kết quả ra file Excel với đầy đủ thông số Intelligence & Trade Plan.

Sử dụng:
  - Chạy trực tiếp: .\venv\Scripts\python.exe scripts/vn_screener.py
"""

import os
import sys
import pandas as pd
from datetime import datetime
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategies.alpha_scanner import AlphaScannerEngine, SCAN_CONFIG
from src.plugins import registry

def calculate_trade_plan(asset_data: dict):
    """
    Calculate additional trade plan levels (SL, TP, Phase 0/1/2) 
    Replicates logic from views/ai_forecast.py
    """
    price = asset_data.get('last_close', 0)
    ai_target = asset_data.get('ai_target', price * 1.05)
    wy_target = asset_data.get('wyckoff_target', 0)
    
    # Final goal is usually the higher of AI or Wyckoff target
    goal_price = max(ai_target, wy_target) if wy_target > 0 else ai_target
    
    # Simple SL logic (using a default 3% for the macro view if ATR is not easy to fetch here)
    # In a full scan, we try to be conservative.
    bias = asset_data.get('ai_bias', 'Neutral')
    if bias == 'Bullish':
        sl = price * 0.96
        rr = (goal_price - price) / (price - sl) if price > sl else 0
    elif bias == 'Bearish':
        sl = price * 1.04
        # Inverse RR for short
        rr = (price - goal_price) / (sl - price) if sl > price else 0
    else:
        sl = price * 0.95
        rr = 0

    # Phase Roadmap (Approximation)
    phase0 = price # Current entry
    phase1 = price * 1.02 if bias == 'Bullish' else price * 0.98
    phase2 = goal_price
    
    return {
        "Stop Loss": round(sl, 2),
        "Take Profit": round(goal_price, 2),
        "R:R Ratio": round(rr, 2),
        "Phase 0 (Entry)": round(phase0, 2),
        "Phase 1 (Struct)": round(phase1, 2),
        "Phase 2 (Goal)": round(phase2, 2)
    }

def main():
    print("🚀 Bắt đầu quét cổ phiếu Việt Nam (VN_UNIVERSE_EXTENDED)...")
    logger.remove() # Remove default logger
    logger.add(sys.stderr, level="INFO")
    
    # 1. Khởi tạo Scanner
    # Chúng ta chỉ quét VN nên sẽ override task builder nếu cần, 
    # nhưng AlphaScannerEngine mặc định quét VN_CORE.
    # Chúng ta dùng extended_universe=True để lấy 100 mã.
    scanner = AlphaScannerEngine(extended_universe=True, commodities=False)
    
    # 2. Cấu hình Scanner (Tắt US/TW để chạy nhanh hơn cho đúng yêu cầu)
    # Override _build_tasks to ONLY VN
    def build_vn_only_tasks():
        tasks = []
        from src.strategies.alpha_scanner import VN_UNIVERSE_EXTENDED
        for sym in VN_UNIVERSE_EXTENDED:
            tasks.append({"symbol": f"{sym}.VN", "market": "VN", "benchmark": "VNINDEX"})
        return tasks
    
    scanner._build_tasks = build_vn_only_tasks
    
    # 3. Chạy quét
    results = scanner.scan_universe()
    
    if not results:
        print("❌ Không tìm thấy mã nào thỏa mãn bộ lọc RS/Wyckoff.")
        return

    # 4. Enrich & Format cho Excel
    enriched_data = []
    for r in results:
        plan = calculate_trade_plan(r)
        
        row = {
            "Mã CP": r['symbol'],
            "Tên": r['name'],
            "Ngành": r['sector'],
            "Giá đóng cửa": r['last_close'],
            "Sức mạnh (RS)": round(r['rs_score'], 2),
            "Cấu trúc": r['structure'],
            "Khối lượng": r['volume_status'],
            "AI Bias": r['ai_bias'],
            "AI Confidence": f"{r['ai_confidence']*100:.1f}%",
            "Mục tiêu AI": r['ai_target'],
            "Dòng tiền (SMC)": r['manipulation'],
            "Chu kỳ Wyckoff": r['wyckoff_phase'],
            "Score Wyckoff": round(r['wyckoff_score'], 2),
            "Tín hiệu VSA": r['wyckoff_vsa'],
            "R:R Ratio": plan['R:R Ratio'],
            "Stop Loss": plan['Stop Loss'],
            "Take Profit": plan['Take Profit'],
            "Entry (Phase 0)": plan['Phase 0 (Entry)'],
            "Struct (Phase 1)": plan['Phase 1 (Struct)'],
            "Goal (Phase 2)": plan['Phase 2 (Goal)']
        }
        enriched_data.append(row)

    # 5. Xuất Excel
    df = pd.DataFrame(enriched_data)
    
    # Sort by R:R and Confidence for user
    df = df.sort_values(by=["R:R Ratio", "Score Wyckoff"], ascending=False)
    
    filename = f"baocao_vn_stocks_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    df.to_excel(filename, index=False)
    
    print(f"\n✅ Đã quét xong {len(results)} mã cổ phiếu tiềm năng.")
    print(f"📊 Kết quả đã được lưu vào file: {filename}")
    print("💡 Anh có thể mở file này để xem tổng quát các mã có R:R tốt nhất.")

if __name__ == "__main__":
    main()
