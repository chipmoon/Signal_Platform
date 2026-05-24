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

    # ─── High-level helpers ──────────────────────────────────────────────────

    def analyze_stock(self, symbol: str, context: str = "") -> str:
        """
        AI stock analysis for a VN ticker.
        Context: thêm dữ liệu kỹ thuật từ hệ thống để tăng chất lượng phân tích.
        """
        ctx_section = f"\n\nDữ liệu kỹ thuật:\n{context}" if context else ""
        prompt = (
            f"Bạn là chuyên gia phân tích chứng khoán Việt Nam cấp cao (phong cách SSI Research, VCBS).\n"
            f"Hãy phân tích cổ phiếu {symbol} trên sàn HOSE/HNX Việt Nam.{ctx_section}\n\n"
            f"Trả lời theo 3 phần:\n"
            f"**1. Xu hướng giá ngắn hạn (1-3 tháng):**\n"
            f"**2. Rủi ro chính cần lưu ý:**\n"
            f"**3. Khuyến nghị (Mua/Quan sát/Bán) + lý do:**\n\n"
            f"Phân tích ngắn gọn, chuyên nghiệp, tối đa 250 từ."
        )
        return self.generate(prompt)

    def get_market_overview(self) -> str:
        """Weekly VN market commentary."""
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
