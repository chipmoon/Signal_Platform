"""
Gemini Advisor Core (Hyper-Robust Browser-Like REST Edition)
============================================================
Status: EMERGENCY FIX - ISP BYPASS MODE
"""

from __future__ import annotations
import os
import requests
import json
import socket
from loguru import logger
from dotenv import load_dotenv

# Force IPv4 explicitly
orig_getaddrinfo = socket.getaddrinfo
def getaddrinfo_ipv4(host, port, family=0, type=0, proto=0, flags=0):
    return orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = getaddrinfo_ipv4

class GeminiAdvisor:
    """Hyper-robust REST bypass for ISP filtering and library 404 bugs."""
    
    def __init__(self) -> None:
        self._api_key = None
        self._initialized = False

    def _ensure_init(self):
        if not self._initialized:
            load_dotenv(override=True)
            self._api_key = os.getenv("GOOGLE_API_KEY")
            self._initialized = True

    def _direct_call(self, prompt: str, timeout: int = 15) -> str:
        self._ensure_init()
        if not self._api_key:
            return "⚠️ AI Offline: Thiếu GOOGLE_API_KEY trong .env"

        # Multi-Model Candidates — updated to 2025 models
        models = ["models/gemini-2.5-flash", "models/gemini-2.0-flash", "models/gemini-1.5-flash"]
        
        # Multi-Version endpoints
        versions = ["v1beta", "v1"]
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        
        last_err = ""
        for model in models:
            for ver in versions:
                url = f"https://generativelanguage.googleapis.com/{ver}/models/{model}:generateContent?key={self._api_key}"
                try:
                    logger.debug(f"Trying Gemini ({ver}/{model})...")
                    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
                    if response.status_code == 200:
                        data = response.json()
                        return data['candidates'][0]['content']['parts'][0]['text'].strip()
                    else:
                        last_err = f"{response.status_code}: {response.text[:100]}"
                except Exception as e:
                    last_err = str(e)
                    
        if "429" in last_err:
            return "🏛️ Hội đồng: Quá tải lượt gọi Free. Đợi 1 phút."
        return f"🏛️ Hội đồng: AI không phản hồi (Mạng/API Error: {last_err[:150]})"

    def get_senate_debate(self, symbol: str, tech_data: dict) -> str:
        prompt = f"""
        Bạn là Hội đồng AI Trading 3 chuyên gia: Bull(🐂), Bear(🐻), Quant(🤖).
        Phân tích mã {symbol} và tranh luận cực ngắn gọn (1-2 câu mỗi người).
        Dữ liệu: Giá {tech_data.get('price', 'N/A')}, Dự báo {tech_data.get('forecast_1d', 'N/A')}, Confidence {tech_data.get('ml_confidence', 'N/A')}%.
        Yêu cầu: Kết cấu hội thoại kịch tính. Trả về kết luận ⚖️ Consensus. Tiếng Việt.
        """
        return self._direct_call(prompt)

    def get_macro_summary(self, headlines: list[str]) -> str:
        headlines_str = "\n".join(headlines[:5])
        prompt = f"Tóm tắt 1 câu rủi ro vĩ mô từ: {headlines_str}"
        return self._direct_call(prompt)

# Singleton
advisor = GeminiAdvisor()
