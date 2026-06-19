"""
Realtime SMC Telegram Alert Bot
===============================
Local polling monitor for confirmed tactical SMC BUY entries during market hours.

Default behavior:
  - Uses the top symbols from data/nightly_scan_results.json as a fast watchlist.
  - Sends alerts through TELEGRAM_SMC_BOT_TOKEN / TELEGRAM_SMC_CHAT_ID.
  - Deduplicates by symbol + date + SMC zone in data/smc_realtime_alert_state.json.

Examples:
  python scripts/smc_realtime_alert_bot.py --once
  python scripts/smc_realtime_alert_bot.py --interval-min 15 --candle-interval 1h --top 120
  python scripts/smc_realtime_alert_bot.py --symbols 8096.TWO,2330.TW,BSR.VN --once
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
try:
    from loguru import logger
except Exception:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("smc_realtime_alert_bot")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alerts.telegram import TelegramAlertSender
from src.plugins import registry
from src.strategies.smc_analyzer import SmcAnalyzer
from src.config import SmcConfig

NIGHTLY_JSON = PROJECT_ROOT / "data" / "nightly_scan_results.json"
STATE_PATH = PROJECT_ROOT / "data" / "smc_realtime_alert_state.json"


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        out = float(v)
        if out != out:
            return default
        return out
    except Exception:
        return default


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {"sent": {}}
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"sent": {}}
    except Exception:
        return {"sent": {}}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = datetime.now().isoformat()
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _infer_market(symbol: str) -> str:
    s = symbol.upper()
    if s.endswith(".VN"):
        return "VN"
    if s.endswith(".TW") or s.endswith(".TWO") or s.isdigit():
        return "TW"
    return "US"


def _previous_weekday(day):
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def _expected_session_date(market: str) -> pd.Timestamp:
    tz_name = "Asia/Ho_Chi_Minh" if market == "VN" else "Asia/Taipei"
    today = datetime.now(ZoneInfo(tz_name)).date()
    return pd.Timestamp(_previous_weekday(today))


def _market_now(market: str) -> datetime:
    tz_name = "Asia/Ho_Chi_Minh" if market == "VN" else "Asia/Taipei"
    return datetime.now(ZoneInfo(tz_name))


def _confirmed_candles(df: pd.DataFrame, market: str, interval: str) -> pd.DataFrame:
    """Keep closed bars and reject daily cache leaked into intraday scans."""
    if df is None or df.empty or "Date" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"])
    if out.empty:
        return out
    if str(interval).lower() == "1d":
        return out.sort_values("Date").drop_duplicates("Date", keep="last")

    dates = out["Date"].dt.tz_localize(None) if out["Date"].dt.tz is not None else out["Date"]
    has_time = bool((dates != dates.dt.normalize()).any())
    multiple_per_day = bool(dates.dt.normalize().duplicated().any())
    if not has_time and not multiple_per_day:
        logger.warning(f"Rejecting daily OHLCV returned for interval={interval}")
        return pd.DataFrame()

    minutes = {"15m": 15, "1h": 60}.get(str(interval).lower())
    if minutes is None:
        raise ValueError(f"Unsupported realtime candle interval: {interval}")
    cutoff = pd.Timestamp(_market_now(market)).tz_localize(None).floor(f"{minutes}min")
    out["Date"] = dates.values
    return out.loc[out["Date"] < cutoff].sort_values("Date").drop_duplicates("Date", keep="last")


def _load_symbols_from_nightly(top: int, scope: str) -> list[str]:
    scope = (scope or "VN_TW").upper()
    if not NIGHTLY_JSON.exists():
        return []
    try:
        rows = json.loads(NIGHTLY_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Could not read nightly results: {exc}")
        return []
    if not isinstance(rows, list):
        return []
    allowed = {"VN_TW": {"VN", "TW"}, "VN": {"VN"}, "TW": {"TW"}, "ALL": {"VN", "TW", "US"}}
    markets = allowed.get(scope, {"VN", "TW"})
    rows = [r for r in rows if str(r.get("market", "")).upper() in markets and r.get("symbol")]
    rows.sort(
        key=lambda r: (
            str(r.get("smc_entry_status", "")).upper() in {"READY", "NEAR"},
            _safe_float(r.get("institutional_score", 0.0)),
            _safe_float(r.get("pred_63d_ret", 0.0)),
        ),
        reverse=True,
    )
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        sym = str(row.get("symbol", "")).strip()
        if sym and sym not in seen:
            out.append(sym)
            seen.add(sym)
        if len(out) >= top:
            break
    return out


def _build_watchlist(args) -> list[str]:
    if args.symbols:
        items = [s.strip() for s in str(args.symbols).replace(";", ",").split(",")]
        return [s for s in items if s]
    symbols = _load_symbols_from_nightly(args.top, args.market_scope)
    if symbols:
        return symbols
    logger.warning("No nightly watchlist found; use --symbols or run nightly scan first.")
    return []


def _alert_key(symbol: str, latest: pd.Series) -> str:
    candle = pd.to_datetime(latest.get("Date"))
    date = candle.strftime("%Y-%m-%d")
    zbot = _safe_float(latest.get("smc_entry_zone_bottom", 0.0))
    ztop = _safe_float(latest.get("smc_entry_zone_top", 0.0))
    return f"{symbol.upper()}|{date}|{zbot:.2f}-{ztop:.2f}"


def _format_price(value: float, market: str) -> str:
    if market == "VN":
        return f"{value:,.0f} VND"
    if market == "TW":
        return f"{value:,.2f} TWD"
    return f"{value:,.2f}"


def _send_smc_alert(sender: TelegramAlertSender, symbol: str, market: str, latest: pd.Series) -> bool:
    close = _safe_float(latest.get("Close", 0.0))
    low = _safe_float(latest.get("Low", 0.0))
    zone_bottom = _safe_float(latest.get("smc_entry_zone_bottom", 0.0))
    zone_top = _safe_float(latest.get("smc_entry_zone_top", 0.0))
    quality = int(_safe_float(latest.get("smc_entry_quality", 0.0)))
    grade = html.escape(str(latest.get("smc_entry_grade", "BEST") or "BEST"))
    retest_no = int(_safe_float(latest.get("smc_entry_retest_no", 0.0)))
    factors = html.escape(str(latest.get("smc_entry_factors", "") or ""))
    reason = html.escape(str(latest.get("smc_entry_reason", "") or "SMC tactical entry"))
    candle = pd.to_datetime(latest.get("Date"))
    signal_time = candle.strftime("%Y-%m-%d %H:%M")
    sent_ts = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M Asia/Taipei")
    atr_stop = max(0.01, zone_bottom - max((zone_top - zone_bottom) * 0.35, close * 0.015))
    target = max(zone_top + (zone_top - atr_stop) * 2.0, close * 1.05)
    rr = (target - zone_top) / max(zone_top - atr_stop, 1e-9)

    text = (
        f"<b>🎯 SMC BUY ALERT — {html.escape(symbol)}</b>\n\n"
        f"Market: <b>{html.escape(market)}</b> | Closed candle: <b>{signal_time}</b>\n"
        f"Sent: <b>{sent_ts}</b>\n"
        f"Grade: <b>{grade}</b> | Quality: <b>{quality}/10</b> | Retest: <b>#{retest_no}</b>\n"
        f"Close: <b>{_format_price(close, market)}</b> | Low: <b>{_format_price(low, market)}</b>\n"
        f"Zone: <code>{_format_price(zone_bottom, market)} - {_format_price(zone_top, market)}</code>\n"
        f"Plan: <code>SL {_format_price(atr_stop, market)} | TP {_format_price(target, market)} | R {rr:.2f}</code>\n"
        f"Reason: <i>{reason}</i>\n"
        f"Factors: <i>{factors}</i>"
    )
    return sender.send_text(text)


def _send_status_message(sender: TelegramAlertSender, text: str) -> None:
    if sender.is_enabled:
        sender.send_text(text)


def _scan_symbol(
    symbol: str,
    lookback_days: int,
    candle_interval: str = "1h",
) -> tuple[str, str, pd.Series | None]:
    market = _infer_market(symbol)
    provider = registry.get(market)
    if not provider:
        return symbol, market, None
    # Daily providers such as yfinance treat end as exclusive; use tomorrow so
    # today's candle can be included when the upstream source has published it.
    market_now = _market_now(market)
    end = (market_now + timedelta(days=1)).strftime("%Y-%m-%d")
    start = (market_now - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    try:
        df = provider.get_price_data(symbol, start, end, interval=candle_interval)
        df = _confirmed_candles(df, market, candle_interval)
        if df is None or df.empty or len(df) < 40:
            return symbol, market, None
        out = SmcAnalyzer(SmcConfig()).generate_signals(df)
        latest = out.iloc[-1]
        latest_date = pd.to_datetime(latest.get("Date"), errors="coerce")
        expected_date = _expected_session_date(market)
        if pd.isna(latest_date) or latest_date.normalize() < expected_date:
            logger.info(
                f"Skipping stale SMC data [{symbol}]: latest={latest_date.date() if not pd.isna(latest_date) else 'N/A'} "
                f"expected={expected_date.date()}"
            )
            return symbol, market, None
        tactical = int(_safe_float(latest.get("smc_entry_tactical_signal", 0.0))) == 1
        if tactical:
            return symbol, market, latest
    except Exception as exc:
        logger.debug(f"SMC scan failed [{symbol}]: {exc}")
    return symbol, market, None


def _market_open_now(scope: str) -> bool:
    now_tw = datetime.now(ZoneInfo("Asia/Taipei"))
    if now_tw.weekday() >= 5:
        return False
    minutes = now_tw.hour * 60 + now_tw.minute
    scope = (scope or "VN_TW").upper()
    tw_open = (9 * 60) <= minutes <= (13 * 60 + 45)
    vn_minutes = minutes - 60
    vn_open = (9 * 60) <= vn_minutes <= (15 * 60)
    if scope == "TW":
        return tw_open
    if scope == "VN":
        return vn_open
    return tw_open or vn_open


def run_once(args) -> int:
    symbols = _build_watchlist(args)
    sender = TelegramAlertSender(channel="smc")
    if not sender.is_enabled:
        logger.warning("SMC Telegram bot is not configured.")
        return 0
    if not symbols:
        if args.notify_empty:
            _send_status_message(
                sender,
                "<b>SMC realtime scan</b>\nNo watchlist symbols found. Run nightly scan first or pass symbols manually.",
            )
        return 0

    state = _load_state()
    sent = state.setdefault("sent", {})
    count = 0
    logger.info(f"Scanning {len(symbols)} symbols for realtime SMC alerts...")
    for symbol in symbols:
        sym, market, latest = _scan_symbol(symbol, args.lookback_days, args.candle_interval)
        if latest is None:
            continue
        key = _alert_key(sym, latest)
        if sent.get(key) and not args.force:
            continue
        if _send_smc_alert(sender, sym, market, latest):
            sent[key] = datetime.now().isoformat()
            count += 1
            log_success = getattr(logger, "success", logger.info)
            log_success(f"SMC alert sent: {sym}")
    if count:
        _save_state(state)
    elif args.notify_empty:
        ts = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M Asia/Taipei")
        _send_status_message(
            sender,
            f"<b>SMC realtime scan complete</b>\nNo new SMC BUY alerts found.\n"
            f"Watchlist: <b>{len(symbols)}</b> symbols\nTime: <b>{ts}</b>",
        )
    logger.info(f"SMC realtime alerts sent: {count}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime SMC Telegram alert monitor")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols. Empty = use nightly top watchlist.")
    parser.add_argument("--top", type=int, default=120, help="Top nightly symbols to watch when --symbols is empty.")
    parser.add_argument("--market-scope", default="VN_TW", choices=["VN", "TW", "VN_TW", "ALL"])
    parser.add_argument("--interval-min", type=int, default=15, help="Polling interval when not --once.")
    parser.add_argument("--lookback-days", type=int, default=45)
    parser.add_argument(
        "--candle-interval",
        default="1h",
        choices=["15m", "1h", "1d"],
        help="OHLCV timeframe; polling frequency is configured separately.",
    )
    parser.add_argument("--once", action="store_true", help="Run one scan and exit.")
    parser.add_argument("--force", action="store_true", help="Send even if this signal key was already sent.")
    parser.add_argument("--ignore-market-hours", action="store_true", help="Run even outside VN/TW market hours.")
    parser.add_argument("--notify-empty", action="store_true", help="Send a Telegram status message when no alert is found.")
    args = parser.parse_args()

    if args.once:
        if args.ignore_market_hours or _market_open_now(args.market_scope):
            run_once(args)
        else:
            logger.info("Market closed; one-pass SMC scan skipped.")
            if args.notify_empty:
                sender = TelegramAlertSender(channel="smc")
                ts = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M Asia/Taipei")
                _send_status_message(sender, f"<b>SMC realtime scan skipped</b>\nMarket closed.\nTime: <b>{ts}</b>")
        return

    logger.info("Realtime SMC alert bot started.")
    while True:
        if args.ignore_market_hours or _market_open_now(args.market_scope):
            run_once(args)
        else:
            logger.info("Market closed; waiting.")
        time.sleep(max(60, int(args.interval_min) * 60))


if __name__ == "__main__":
    main()
