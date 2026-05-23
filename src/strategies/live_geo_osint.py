"""
Live Geo-OSINT ingestion for production runtime.

Pipeline:
    wire/symbol news -> normalized events -> causal score -> execution gate
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import re

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from src.strategies.geo_stack import GeoSignalPipeline
from src.strategies.sentiment_hub import SentimentHub


@dataclass(frozen=True)
class LiveGeoOsintConfig:
    lookback_hours: int = 72
    max_items_per_category: int = 8
    max_symbol_items: int = 8
    stale_after_minutes: int = 180
    min_events_to_trust: int = 2


class LiveGeoOsintEngine:
    """Production runtime engine for live geopolitical execution gate."""

    _TRUST_MAP = {
        "REUTERS": "A",
        "BLOOMBERG": "A",
        "FINANCIAL TIMES": "A",
        "WALL STREET JOURNAL": "A",
        "WSJ": "A",
        "CNBC": "B",
        "YAHOO": "B",
        "MARKETWATCH": "B",
        "INVESTING.COM": "C",
    }

    _RULES: list[tuple[re.Pattern[str], str, str, float]] = [
        (re.compile(r"(hormuz|strait|shipping|transit|tanker|vessel|chokepoint|port closure|route disruption)", re.I), "SHIPPING_DISRUPTION", "HIGH", 0.78),
        (re.compile(r"(missile|drone|strike|military|navy|airstrike|conflict|escalat|war)", re.I), "MILITARY_ESCALATION", "CRITICAL", 0.82),
        (re.compile(r"(sanction|embargo|restriction|blacklist|compliance action)", re.I), "SANCTIONS_ENFORCEMENT", "MEDIUM", 0.70),
        (re.compile(r"(insurance|war-risk premium|premium jumped|underwriting stress)", re.I), "INSURANCE_STRESS", "HIGH", 0.74),
        (re.compile(r"(pipeline|refinery|terminal|infrastructure|outage|explosion|sabotage)", re.I), "INFRA_ATTACK", "HIGH", 0.76),
        (re.compile(r"(ceasefire|de-escalat|reopen|diplomatic talks|truce|agreement)", re.I), "DIPLOMATIC_DEESCALATION", "MEDIUM", 0.68),
    ]

    def __init__(
        self,
        config: LiveGeoOsintConfig | None = None,
        hub: SentimentHub | None = None,
        pipeline: GeoSignalPipeline | None = None,
    ) -> None:
        self.config = config or LiveGeoOsintConfig()
        self.hub = hub or SentimentHub()
        self.pipeline = pipeline or GeoSignalPipeline()

    @staticmethod
    def _publisher_grade(name: str) -> str:
        if not name:
            return "C"
        up = name.upper()
        for k, grade in LiveGeoOsintEngine._TRUST_MAP.items():
            if k in up:
                return grade
        return "B"

    def _headline_to_event(self, headline: str, category: str, publisher: str) -> dict[str, Any] | None:
        text = str(headline or "").strip()
        if not text:
            return None

        event_type = ""
        severity = "MEDIUM"
        conf = 0.60
        for rule, e_type, sev, base_conf in self._RULES:
            if rule.search(text):
                event_type = e_type
                severity = sev
                conf = base_conf
                break
        if not event_type:
            return None

        if category == "Geopolitics":
            conf += 0.07
        elif category == "Energy":
            conf += 0.04

        grade = self._publisher_grade(publisher)
        if grade == "A":
            conf += 0.06
        elif grade == "D":
            conf -= 0.08

        region = "Hormuz" if re.search(r"(hormuz|iran|gulf|middle east)", text, flags=re.I) else "Global"
        return {
            "event_type": event_type,
            "severity": severity,
            "confidence": float(np.clip(conf, 0.20, 0.98)),
            "source_quality": grade,
            "region": region,
            "headline": text,
        }

    def _fetch_wire(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for cat in ("Geopolitics", "Energy", "Economy"):
            try:
                news = self.hub.get_macro_news(cat)[: self.config.max_items_per_category]
            except Exception as exc:
                logger.warning(f"Live OSINT wire fetch failed for {cat}: {exc}")
                news = []
            for n in news:
                rows.append(
                    {
                        "headline": n.get("title", ""),
                        "publisher": n.get("publisher", "Unknown"),
                        "event_ts": n.get("time"),
                        "category": cat,
                        "link": n.get("link", ""),
                        "proxy": n.get("proxy", ""),
                    }
                )
        return rows

    def _fetch_symbol_news(self, symbol: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not symbol:
            return rows
        try:
            news = yf.Ticker(symbol).news or []
        except Exception as exc:
            logger.debug(f"Symbol OSINT fetch failed for {symbol}: {exc}")
            news = []

        for n in news[: self.config.max_symbol_items]:
            ts_raw = n.get("providerPublishTime", n.get("publishTime"))
            if ts_raw:
                try:
                    ts = datetime.fromtimestamp(float(ts_raw), tz=timezone.utc).isoformat()
                except Exception:
                    ts = datetime.now(tz=timezone.utc).isoformat()
            else:
                ts = datetime.now(tz=timezone.utc).isoformat()
            rows.append(
                {
                    "headline": n.get("title", ""),
                    "publisher": n.get("publisher", n.get("source", "Unknown")),
                    "event_ts": ts,
                    "category": "Symbol",
                    "link": n.get("link", ""),
                    "proxy": symbol,
                }
            )
        return rows

    def fetch_events(self, symbol: str, market: str) -> pd.DataFrame:
        """Collect and normalize live OSINT events."""
        raw = self._fetch_wire() + self._fetch_symbol_news(symbol)
        if not raw:
            return pd.DataFrame(
                columns=["event_ts", "event_type", "severity", "confidence", "source_quality", "region", "headline"]
            )

        seen = set()
        rows: list[dict[str, Any]] = []
        cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(hours=self.config.lookback_hours)
        for item in raw:
            headline = str(item.get("headline", "")).strip()
            if not headline or headline in seen:
                continue
            seen.add(headline)

            evt = self._headline_to_event(headline, str(item.get("category", "")), str(item.get("publisher", "")))
            if not evt:
                continue
            evt["event_ts"] = item.get("event_ts")
            rows.append(evt)

        if not rows:
            return pd.DataFrame(
                columns=["event_ts", "event_type", "severity", "confidence", "source_quality", "region", "headline"]
            )

        events = self.pipeline.normalize_events(pd.DataFrame(rows))
        events = events[events["event_ts"] >= cutoff].copy()
        return events.reset_index(drop=True)

    def build_live_state(self, symbol: str, market: str) -> dict[str, Any]:
        """Return live gate/osint/causal bundle for runtime use."""
        events = self.fetch_events(symbol, market)
        osint = self.pipeline.score_osint(events)
        causal = self.pipeline.causal_score(osint)
        gate = self.pipeline.execution_gate(osint, causal)

        now = pd.Timestamp.now(tz=timezone.utc)
        if events.empty:
            latest_ts = None
            freshness_min = None
        else:
            latest_ts = events["event_ts"].max()
            freshness_min = float((now - latest_ts).total_seconds() / 60.0)

        stale = bool(
            events.empty
            or len(events) < self.config.min_events_to_trust
            or (freshness_min is not None and freshness_min > self.config.stale_after_minutes)
        )
        if stale:
            gate = {
                "action": "REDUCE_SIZE",
                "position_multiplier": 0.6,
                "reason": "OSINT feed stale/insufficient: defensive execution mode",
                "geo_risk_score": float(osint.get("geo_risk_score", 0.0)),
                "expected_oil_move_1d_pct": float(causal.get("expected_oil_move_1d_pct", 0.0)),
            }

        return {
            "events": events,
            "osint": osint,
            "causal": causal,
            "gate": gate,
            "status": {
                "stale": stale,
                "events_count": int(len(events)),
                "latest_event_ts": latest_ts.isoformat() if latest_ts is not None else "",
                "freshness_minutes": freshness_min if freshness_min is not None else -1.0,
            },
        }
