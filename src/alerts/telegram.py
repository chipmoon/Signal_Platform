"""
Telegram Alert Sender
=====================
Sends trading signals, risk alerts, and daily reports via Telegram Bot API.

Usage:
    Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env file.
    SMC alerts can use TELEGRAM_SMC_BOT_TOKEN / TELEGRAM_SMC_CHAT_ID.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from loguru import logger


class TelegramAlertSender:
    """Sends formatted alerts to a Telegram chat via Bot API.

    All methods are no-ops if credentials are not configured,
    ensuring the system never crashes due to missing Telegram setup.
    """

    _BASE_URL = "https://api.telegram.org/bot{token}/{endpoint}"

    def __init__(self, channel: str = "default") -> None:
        try:
            from dotenv import load_dotenv
            project_env = Path(__file__).resolve().parents[2] / ".env"
            if project_env.exists():
                load_dotenv(project_env, override=False)
            load_dotenv(override=False)
        except Exception:
            # dotenv is optional at runtime; fallback to process environment.
            pass

        self.channel = str(channel or "default").lower()
        if self.channel == "smc":
            self.bot_token = os.getenv("TELEGRAM_SMC_BOT_TOKEN", "") or os.getenv("TELEGRAM_BOT_TOKEN", "")
            self.chat_id = os.getenv("TELEGRAM_SMC_CHAT_ID", "") or os.getenv("TELEGRAM_CHAT_ID", "")
        else:
            self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
            self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self._enabled = bool(self.bot_token and self.chat_id)
        self._session = requests.Session()
        # Ignore broken system proxy variables (common local issue: 127.0.0.1:9).
        self._session.trust_env = False

        if not self._enabled:
            logger.debug(f"Telegram alerts disabled for channel={self.channel} (no credentials in .env).")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def _request(self, endpoint: str, payload: dict[str, Any], timeout: int = 15) -> dict[str, Any]:
        if not self.bot_token:
            return {"ok": False, "result": []}
        url = self._BASE_URL.format(token=self.bot_token, endpoint=endpoint)
        resp = self._session.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _send(self, text: str, chat_id: str | None = None, reply_to_message_id: int | None = None) -> bool:
        """Send a message via Telegram Bot API.

        Returns True if sent successfully, False otherwise.
        """
        if not self._enabled:
            return False

        try:
            payload = {
                "chat_id": str(chat_id or self.chat_id),
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
            if reply_to_message_id is not None:
                payload["reply_to_message_id"] = int(reply_to_message_id)
            data = self._request("sendMessage", payload, timeout=10)
            if not bool(data.get("ok", False)):
                logger.warning(f"Telegram send failed (API not ok): {data}")
                return False
            return True
        except Exception as exc:
            logger.warning(f"Telegram send failed: {exc}")
            return False

    def send_text(
        self,
        text: str,
        chat_id: str | None = None,
        reply_to_message_id: int | None = None,
    ) -> bool:
        """Send a raw text/HTML message."""
        return self._send(text, chat_id=chat_id, reply_to_message_id=reply_to_message_id)

    def get_updates(self, offset: int | None = None, timeout: int = 25) -> list[dict]:
        """Read inbound updates via long polling."""
        if not self.bot_token:
            return []
        try:
            payload: dict[str, Any] = {
                "timeout": int(timeout),
                "allowed_updates": ["message"],
            }
            if offset is not None:
                payload["offset"] = int(offset)
            data = self._request("getUpdates", payload, timeout=timeout + 5)
            if bool(data.get("ok", False)):
                result = data.get("result", [])
                return result if isinstance(result, list) else []
            return []
        except Exception as exc:
            logger.warning(f"Telegram getUpdates failed: {exc}")
            return []

    def send_signal_alert(
        self,
        ticker: str,
        signal: int,
        price: float,
        reason: str,
    ) -> bool:
        """Send a trading signal alert.

        Args:
            ticker: Symbol (e.g. GC=F).
            signal: 1 (BUY), -1 (SELL), 0 (HOLD).
            price: Current price.
            reason: Signal reason string.
        """
        emoji = "🟢 BUY" if signal == 1 else "🔴 SELL" if signal == -1 else "⚪ HOLD"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")

        text = (
            f"<b>⚡ SIGNAL ALERT — {ticker}</b>\n\n"
            f"📊 Direction: <b>{emoji}</b>\n"
            f"💰 Price: <b>${price:,.2f}</b>\n"
            f"📝 Reason: {reason}\n"
            f"🕐 Time: {ts}"
        )
        return self._send(text)

    def send_risk_alert(
        self,
        alert_type: str,
        details: str,
        severity: str = "WARNING",
    ) -> bool:
        """Send a risk management alert.

        Args:
            alert_type: E.g. "DRAWDOWN", "CIRCUIT_BREAKER", "VAR_BREACH".
            details: Description of the risk event.
            severity: WARNING or CRITICAL.
        """
        emoji = "🚨" if severity == "CRITICAL" else "⚠️"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")

        text = (
            f"<b>{emoji} RISK ALERT — {alert_type}</b>\n\n"
            f"📋 Severity: <b>{severity}</b>\n"
            f"📝 Details: {details}\n"
            f"🕐 Time: {ts}"
        )
        return self._send(text)

    def send_daily_report(
        self,
        ticker: str,
        equity: float,
        daily_return: float,
        total_return: float,
        open_position: bool,
    ) -> bool:
        """Send end-of-day portfolio summary.

        Args:
            ticker: Active ticker.
            equity: Current portfolio equity.
            daily_return: Today's return percentage.
            total_return: Cumulative return percentage.
            open_position: Whether a position is currently open.
        """
        pos_emoji = "📈 LONG" if open_position else "💤 FLAT"
        ret_emoji = "🟢" if daily_return >= 0 else "🔴"
        ts = datetime.now().strftime("%Y-%m-%d")

        text = (
            f"<b>📊 DAILY REPORT — {ts}</b>\n\n"
            f"🏷 Ticker: <b>{ticker}</b>\n"
            f"💰 Equity: <b>${equity:,.2f}</b>\n"
            f"{ret_emoji} Daily: <b>{daily_return:+.2f}%</b>\n"
            f"📈 Total: <b>{total_return:+.2f}%</b>\n"
            f"📍 Position: <b>{pos_emoji}</b>"
        )
        return self._send(text)
