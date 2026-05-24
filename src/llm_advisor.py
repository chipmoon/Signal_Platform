"""
Gemini Advisor Core — 5-Member Senate Edition
==============================================
Upgraded with Volume Intelligence + Investor Flow data.
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
    """5-Member AI Senate with Volume Intelligence."""

    def __init__(self) -> None:
        self._api_key = None
        self._initialized = False

    def _ensure_init(self):
        if not self._initialized:
            load_dotenv(override=True)
            self._api_key = os.getenv("GOOGLE_API_KEY")
            self._initialized = True

    def _direct_call(self, prompt: str, timeout: int = 30) -> str:
        self._ensure_init()
        if not self._api_key:
            return "⚠️ AI Offline: Thiếu GOOGLE_API_KEY"

        # No 'models/' prefix — URL template adds it
        models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        versions = ["v1beta", "v1"]

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        }
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        last_err = ""
        for model in models:
            for ver in versions:
                url = (
                    f"https://generativelanguage.googleapis.com/{ver}"
                    f"/models/{model}:generateContent?key={self._api_key}"
                )
                try:
                    logger.debug(f"Trying Gemini ({ver}/{model})...")
                    r = requests.post(
                        url, headers=headers, data=json.dumps(payload), timeout=timeout
                    )
                    if r.status_code == 200:
                        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    last_err = f"{r.status_code}: {r.text[:120]}"
                except Exception as e:
                    last_err = str(e)

        if "429" in last_err:
            return "🏛️ Hội đồng: Quá tải lượt gọi. Đợi 1 phút."
        return f"🏛️ Hội đồng: AI không phản hồi (Lỗi: {last_err[:150]})"

    def get_senate_debate(
        self,
        symbol: str,
        tech_data: dict,
        volume_data: dict | None = None,
        flow_data: dict | None = None,
    ) -> str:
        """
        5-Member AI Senate Debate with quantitative volume context.

        Members:
          🐂 Bull        — price & forecast optimist
          🐻 Bear        — risk & downside pessimist
          🤖 Quant       — SMC/Wyckoff/CMF technical analysis
          💰 Smart Money — institutional block trade & OBV flow
          📰 Macro       — volume trend, cung/cầu, sentiment
        """
        vd = volume_data or {}
        fd = flow_data or {}

        price_ctx = (
            f"Giá: {tech_data.get('price', 'N/A')} | "
            f"Dự báo 1D: {tech_data.get('forecast_1d', 'N/A')} | "
            f"ML Confidence: {tech_data.get('ml_confidence', 'N/A')}%"
        )
        tech_ctx = (
            f"SMC: {tech_data.get('smc_trend', 'N/A')} | "
            f"Wyckoff: {tech_data.get('wyckoff_phase', 'N/A')}"
        )
        vol_ctx = (
            f"OBV: {vd.get('obv_trend', 'N/A')} ({vd.get('obv_change_pct', 'N/A')}%) | "
            f"CMF(14): {vd.get('cmf_14', 'N/A')} [{vd.get('cmf_signal', 'N/A')}] | "
            f"VWAP lệch: {vd.get('vwap_deviation', 'N/A')}%"
        ) if vd else "Không có dữ liệu volume TA"
        block_ctx = (
            f"Block trade days: {vd.get('block_ratio_pct', 'N/A')}% | "
            f"{vd.get('block_direction', 'N/A')} | "
            f"Volume 5D: {vd.get('vol_ratio_5d_pct', 'N/A')}% vs avg"
        ) if vd else "N/A"
        flow_ctx = (
            f"Lực mua: {fd.get('buy_pressure', 'N/A')}% | "
            f"Lực bán: {fd.get('sell_pressure', 'N/A')}% | "
            f"{fd.get('net_flow', 'N/A')}"
        ) if fd else "Không có dữ liệu flow"
        smart_signal = vd.get("smart_money_signal", "Chưa tính")

        prompt = f"""Bạn là Hội đồng AI Trading 5 chuyên gia phân tích {symbol}.
Mỗi người nói đúng 1-2 câu, sắc bén, có số liệu cụ thể. KHÔNG giải thích dài dòng.

=== DỮ LIỆU ===
[Giá/Dự báo]   {price_ctx}
[Kỹ thuật]     {tech_ctx}
[Volume TA]    {vol_ctx}
[Block Trade]  {block_ctx}
[Dòng tiền]    {flow_ctx}
[Smart Signal] {smart_signal}

=== OUTPUT (giữ nguyên định dạng) ===

🐂 **Bull**: [Lập luận tích cực, trích dẫn số liệu]

🐻 **Bear**: [Rủi ro cụ thể, phản biện Bull]

🤖 **Quant**: [Đọc CMF/OBV/VWAP/Wyckoff, nêu tín hiệu kỹ thuật]

💰 **Smart Money**: [Nhận xét block trade và dòng tiền lớn, retail vs tổ chức]

📰 **Macro**: [Volume trend tổng thể, cung/cầu thị trường]

⚖️ **Consensus**: [MUA/BÁN/GIỮ — lý do 1 câu — Tin cậy: X/10]

Tranh luận thực sự, không đồng thuận ngay. Tiếng Việt."""

        return self._direct_call(prompt, timeout=30)

    def get_macro_summary(self, headlines: list[str]) -> str:
        headlines_str = "\n".join(headlines[:5])
        prompt = f"Tóm tắt 1 câu rủi ro vĩ mô từ: {headlines_str}"
        return self._direct_call(prompt)


# Singleton
advisor = GeminiAdvisor()
