"""
Mozyfin API Client (Antofin API v1.0)
======================================
Confirmed working endpoints:
  GET  /health
  GET  /api/v1/subscription/usage
  GET  /api/v1/market/exchange/entity?search={sym}
  GET  /api/v1/market/exchange/entity/{id}  (e.g. VCB.VN)
  GET  /api/v1/market/market-index
  GET  /api/v1/news?search={sym}&limit=N
  POST /api/v1/chat              → {data: {id}}
  POST /api/v1/chat/{id}/message → {data: {id, status, content}}
  GET  /api/v1/chat/messages/{msg_id}

Credits: 50/month (free tier). Use sparingly.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import requests
from loguru import logger

MOZYFIN_BASE = "https://api.mozyfin.com"
_TIMEOUT = 15
_POLL_INTERVAL = 2.5
_POLL_MAX_WAIT = 90


class MozyfinClient:
    """Antofin/Mozyfin API client — confirmed working v1.0.0."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or self._load_key()
        if not self.api_key:
            raise ValueError(
                "Mozyfin API key not found.\n"
                "Set MOZYFIN_API_KEY in .streamlit/secrets.toml or env var."
            )
        self._s = requests.Session()
        self._s.headers.update({
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
        })

    @staticmethod
    def _load_key() -> Optional[str]:
        key = os.environ.get("MOZYFIN_API_KEY")
        if key:
            return key
        try:
            import streamlit as st
            return st.secrets.get("MOZYFIN_API_KEY")
        except Exception:
            pass
        return None

    def _get(self, path: str, params: dict = None) -> dict:
        r = self._s.get(f"{MOZYFIN_BASE}{path}", params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict) -> dict:
        r = self._s.post(f"{MOZYFIN_BASE}{path}", json=body, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()

    # ─── Utilities ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Quick health check."""
        try:
            r = self._get("/health")
            return r.get("status") == "healthy"
        except Exception:
            return False

    def get_usage(self) -> dict:
        """Check remaining credits. Returns {credits_used, credits_cap}."""
        try:
            data = self._get("/api/v1/subscription/usage")
            return data.get("data", {})
        except Exception as e:
            logger.debug(f"Mozyfin usage: {e}")
            return {}

    # ─── Market Data ────────────────────────────────────────────────────────

    def search_entity(self, symbol: str) -> dict:
        """Search for a stock entity by ticker. Returns first match."""
        try:
            data = self._get("/api/v1/market/exchange/entity", {"search": symbol})
            items = data.get("data", [])
            if not items:
                return {}
            for item in items:
                if item.get("symbol", "").upper() == symbol.upper():
                    return item
            return items[0]
        except Exception as e:
            logger.debug(f"Mozyfin search {symbol}: {e}")
            return {}

    def get_entity(self, entity_id: str) -> dict:
        """Get full entity profile by ID (e.g. 'VCB.VN')."""
        try:
            data = self._get(f"/api/v1/market/exchange/entity/{entity_id}")
            return data.get("data", {})
        except Exception as e:
            logger.debug(f"Mozyfin entity {entity_id}: {e}")
            return {}

    def get_market_indices(self) -> list[dict]:
        """Get VNINDEX, HNX30, VN30 etc. with current value & change."""
        try:
            data = self._get("/api/v1/market/market-index")
            return data.get("data", [])
        except Exception as e:
            logger.debug(f"Mozyfin indices: {e}")
            return []

    def get_news(self, symbol: str = "", limit: int = 5) -> list[dict]:
        """Get recent news articles. Leave symbol empty for market-wide news."""
        try:
            params = {"limit": limit}
            if symbol:
                params["search"] = symbol
            data = self._get("/api/v1/news", params)
            return data.get("data", [])
        except Exception as e:
            logger.debug(f"Mozyfin news {symbol}: {e}")
            return []

    # ─── AI Chat ────────────────────────────────────────────────────────────

    def ask(self, prompt: str, max_wait: int = _POLL_MAX_WAIT) -> str:
        """
        Send a question to Mozyfin AI and return the response text.
        Uses 1 credit per call.

        Flow:
          POST /api/v1/chat → chat_id
          POST /api/v1/chat/{chat_id}/message → msg_id + immediate content
          GET  /api/v1/chat/messages/{msg_id} → poll until status done
        """
        try:
            # Step 1: Create session
            resp = self._post("/api/v1/chat", {"title": "trading-ai", "mode": "auto"})
            chat_id = resp.get("data", {}).get("id")
            if not chat_id:
                logger.warning(f"Mozyfin: no chat_id in response: {resp}")
                return ""

            # Step 2: Send message
            msg_resp = self._post(
                f"/api/v1/chat/{chat_id}/message",
                {"content": prompt}
            )
            msg_data = msg_resp.get("data", {})
            msg_id = msg_data.get("id")

            # If response is already complete
            if msg_data.get("content") and msg_data.get("status") not in ("thinking", "streaming", "pending"):
                return str(msg_data["content"])

            if not msg_id:
                logger.warning(f"Mozyfin: no msg_id in response: {msg_resp}")
                return ""

            # Step 3: Poll for completion
            deadline = time.time() + max_wait
            while time.time() < deadline:
                poll = self._get(f"/api/v1/chat/messages/{msg_id}")
                result = poll.get("data", poll)
                status = result.get("status", "")
                content = result.get("content", "")

                if status in ("done", "completed", "finished", "success"):
                    return str(content)
                if content and status not in ("thinking", "streaming", "pending", ""):
                    return str(content)
                if status == "error":
                    logger.warning(f"Mozyfin AI error: {result}")
                    return ""

                time.sleep(_POLL_INTERVAL)

            logger.warning(f"Mozyfin AI timeout ({max_wait}s): {prompt[:80]}")
            return ""

        except Exception as e:
            logger.warning(f"Mozyfin ask() exception: {e}")
            return ""

    # ─── High-level helpers ─────────────────────────────────────────────────

    def analyze_stock(self, symbol: str) -> str:
        """
        AI stock analysis in Vietnamese for a given ticker.
        Uses 1 credit.
        """
        prompt = (
            f"Phân tích cổ phiếu {symbol} trên sàn chứng khoán Việt Nam. "
            f"Bao gồm: (1) xu hướng giá ngắn hạn 1-3 tháng, "
            f"(2) các rủi ro chính cần lưu ý, "
            f"(3) khuyến nghị Mua/Quan sát/Bán với lý do cụ thể. "
            f"Ngắn gọn, tối đa 200 từ, phong cách chuyên gia phân tích."
        )
        return self.ask(prompt)

    def get_market_overview(self) -> str:
        """
        AI market commentary — VN-Index weekly summary in Vietnamese.
        Uses 1 credit.
        """
        prompt = (
            "Tóm tắt tình hình thị trường chứng khoán Việt Nam hiện tại. "
            "VN-Index đang ở vùng nào, xu hướng ngắn hạn ra sao? "
            "Vùng hỗ trợ/kháng cự quan trọng? Dòng tiền đang tập trung vào ngành nào? "
            "Tối đa 150 từ, súc tích như báo cáo buổi sáng của VCBS/SSI."
        )
        return self.ask(prompt)

    def get_stock_quick_info(self, symbol: str) -> dict:
        """
        Get entity data + news for a stock — NO credits used.
        Returns: {entity, news}
        """
        entity = self.search_entity(symbol)
        news = self.get_news(symbol, limit=3)
        return {"entity": entity, "news": news}
