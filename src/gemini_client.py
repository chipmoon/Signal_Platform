"""
Gemini AI Client (Google AI Studio)
=====================================
Fallback AI analyst khi Mozyfin credits hết.

Free tier: 1,500 requests/day (Gemini 2.0 Flash) — thực tế unlimited cho cá nhân.
REST API trực tiếp, không cần cài google-generativeai package.

Endpoint: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent
"""

from __future__ import annotations

import os
from typing import Optional

import requests
from loguru import logger

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
_MODEL = "models/gemini-2.5-flash"   # Best free model: reasoning + fast
_TIMEOUT = 60


class GeminiClient:
    """Lightweight Gemini API client via REST — no extra packages required."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or self._load_key()
        if not self.api_key:
            raise ValueError(
                "Google API key not found.\n"
                "Set GOOGLE_API_KEY in .streamlit/secrets.toml or env var."
            )

    @staticmethod
    def _load_key() -> Optional[str]:
        # 1. Env var
        key = os.environ.get("GOOGLE_API_KEY")
        if key:
            return key

        # 2. Streamlit Cloud secrets
        try:
            import streamlit as st
            key = st.secrets["GOOGLE_API_KEY"]
            if key:
                return str(key)
        except Exception:
            pass

        # 3. Local secrets.toml
        try:
            from pathlib import Path
            p = Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml"
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    if "GOOGLE_API_KEY" in line and "=" in line:
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if key:
                            return key
        except Exception:
            pass

        return None

    def is_available(self) -> bool:
        """Quick check if API key is configured."""
        return bool(self.api_key)

    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """
        Send a prompt to Gemini 2.0 Flash and return the response text.
        Uses REST API directly — no extra dependencies.
        """
        url = f"{_GEMINI_BASE}/{_MODEL}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 2048,
                "topP": 0.9,
            },
        }
        try:
            r = requests.post(url, json=payload, timeout=_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            # Extract text from response
            candidates = data.get("candidates", [])
            if not candidates:
                logger.warning(f"Gemini: no candidates in response")
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            return " ".join(p.get("text", "") for p in parts).strip()
        except Exception as e:
            logger.warning(f"Gemini generate() failed: {e}")
            return ""

    # ─── Market config ────────────────────────────────────────────────────────

    _MARKET_CONFIG = {
        "VN": {
            "name": "Việt Nam (HOSE/HNX)",
            "exchange": "sàn HOSE/HNX Việt Nam",
            "analyst_style": "SSI Research, VCBS, Yuanta Vietnam",
            "lang": "Vietnamese",
            "rec_labels": "Mua / Quan sát / Bán",
            "currency": "VND",
        },
        "TW": {
            "name": "Đài Loan (TWSE/TPEX)",
            "exchange": "台灣證券交易所 (TWSE) / 台灣證券商業同業公會 (TPEX)",
            "analyst_style": "元大投顧 (Yuanta), 凱基 (KGI), 富邦 (Fubon)",
            "lang": "Traditional Chinese",
            "rec_labels": "買進 / 觀察 / 賣出",
            "currency": "TWD",
        },
        "US": {
            "name": "United States (NYSE/NASDAQ)",
            "exchange": "NYSE / NASDAQ",
            "analyst_style": "Goldman Sachs, JPMorgan, Morgan Stanley",
            "lang": "English",
            "rec_labels": "Buy / Hold / Sell",
            "currency": "USD",
        },
    }

    def analyze_stock(self, symbol: str, market: str = "VN", context: str = "") -> str:
        """
        AI stock analysis — market-aware for VN / TW / US.
        market: 'VN' (Vietnam), 'TW' (Taiwan), 'US' (United States), or others.
        context: optional technical data from the trading system.
        """
        cfg = self._MARKET_CONFIG.get(market.upper(), self._MARKET_CONFIG["VN"])
        ctx_section = f"\n\nTechnical context:\n{context}" if context else ""

        if market.upper() == "TW":
            # Traditional Chinese prompt for Taiwan stocks
            prompt = (
                f"你是台灣頂級股票分析師，風格如{cfg['analyst_style']}。\n"
                f"請分析在{cfg['exchange']}上市的股票 {symbol}。{ctx_section}\n\n"
                f"請依以下三點作答：\n"
                f"**1. 短期價格趨勢（1-3個月）：**\n"
                f"**2. 主要風險：**\n"
                f"**3. 投資建議（{cfg['rec_labels']}）+ 理由：**\n\n"
                f"分析精簡專業，不超過250字，以繁體中文回答。"
            )
        elif market.upper() == "US":
            prompt = (
                f"You are a senior equity analyst at a top-tier US investment bank "
                f"(style: {cfg['analyst_style']}).\n"
                f"Analyze {symbol} listed on {cfg['exchange']}.{ctx_section}\n\n"
                f"Structure your response:\n"
                f"**1. Short-term price trend (1-3 months):**\n"
                f"**2. Key risks:**\n"
                f"**3. Recommendation ({cfg['rec_labels']}) + rationale:**\n\n"
                f"Concise professional analysis, max 250 words."
            )
        else:
            # Default: Vietnamese
            prompt = (
                f"Bạn là chuyên gia phân tích chứng khoán {cfg['name']} "
                f"cấp cao (phong cách {cfg['analyst_style']}).\n"
                f"Hãy phân tích cổ phiếu {symbol} trên {cfg['exchange']}.{ctx_section}\n\n"
                f"Trả lời theo 3 phần:\n"
                f"**1. Xu hướng giá ngắn hạn (1-3 tháng):**\n"
                f"**2. Rủi ro chính cần lưu ý:**\n"
                f"**3. Khuyến nghị ({cfg['rec_labels']}) + lý do:**\n\n"
                f"Phân tích ngắn gọn, chuyên nghiệp, tối đa 250 từ."
            )
        return self.generate(prompt)

    def get_market_overview(self, market: str = "VN") -> str:
        """Market overview — supports VN, TW, US."""
        if market.upper() == "TW":
            prompt = (
                "請以台灣頂級分析師的角度，簡述台灣股市（加權指數）現況：\n"
                "- 目前指數位置與短期趨勢？\n"
                "- 重要支撐/壓力區？\n"
                "- 資金流向哪些產業？\n"
                "- 本週投資策略建議？\n\n"
                "最多150字，以繁體中文回答，風格如早報研究報告。"
            )
        elif market.upper() == "US":
            prompt = (
                "As a senior US equity strategist, briefly summarize US market outlook:\n"
                "- Where is S&P 500 / NASDAQ currently, short-term trend?\n"
                "- Key support/resistance levels?\n"
                "- Which sectors are seeing capital inflows?\n"
                "- Strategic recommendation for this week?\n\n"
                "Max 150 words, professional style like Goldman morning note."
            )
        else:
            prompt = (
                "Bạn là chuyên gia phân tích vĩ mô thị trường chứng khoán Việt Nam.\n"
                "Tóm tắt tình hình thị trường hiện tại:\n"
                "- VN-Index đang ở vùng nào, xu hướng ngắn hạn?\n"
                "- Vùng hỗ trợ/kháng cự quan trọng nhất?\n"
                "- Dòng tiền tập trung vào ngành nào?\n"
                "- Khuyến nghị chiến lược cho nhà đầu tư tuần này?\n\n"
                "Tối đa 150 từ, phong cách báo cáo buổi sáng VCBS/SSI."
            )
        return self.generate(prompt)
