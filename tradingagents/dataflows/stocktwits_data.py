from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_API = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
_UA = "tradingagents-extended/0.1"


def fetch_stream_checked(ticker: str, limit: int = 30) -> tuple[list[dict], str | None]:
    url = _API.format(ticker=ticker.upper())
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except (HTTPError, URLError, json.JSONDecodeError, TimeoutError) as exc:
        logger.warning("StockTwits fetch failed for %s: %s", ticker, exc)
        return [], str(exc)
    return (data.get("messages", []) if isinstance(data, dict) else [])[:limit], None


def fetch_stream(ticker: str, limit: int = 30) -> list[dict]:
    messages, _ = fetch_stream_checked(ticker, limit=limit)
    return messages


def parse_sentiment(messages: list[dict]) -> dict[str, Any]:
    bullish = bearish = unlabeled = 0
    timestamps = []
    users = []

    for m in messages:
        entities = m.get("entities") or {}
        sentiment_obj = entities.get("sentiment") or {}
        sentiment = sentiment_obj.get("basic") if isinstance(sentiment_obj, dict) else None

        if sentiment == "Bullish":
            bullish += 1
        elif sentiment == "Bearish":
            bearish += 1
        else:
            unlabeled += 1

        created = m.get("created_at", "")
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            timestamps.append(dt.timestamp())
        except (ValueError, AttributeError):
            pass

        user = (m.get("user") or {}).get("username", "")
        if user:
            users.append(user)

    total = bullish + bearish + unlabeled
    return {
        "total_messages": total,
        "bullish": bullish,
        "bearish": bearish,
        "unlabeled": unlabeled,
        "bullish_ratio": round(bullish / total, 2) if total else 0,
        "bearish_ratio": round(bearish / total, 2) if total else 0,
        "timestamps": timestamps,
        "unique_users": len(set(users)),
        "messages": messages,
    }


def calculate_velocity(timestamps: list[float], window_hours: int = 6) -> float:
    now = time.time()
    recent = sum(1 for t in timestamps if now - t < window_hours * 3600)
    earlier = sum(
        1 for t in timestamps
        if window_hours * 3600 <= now - t < window_hours * 7200
    )
    recent_rate = recent / max(window_hours, 1)
    earlier_rate = earlier / max(window_hours, 1)
    if earlier_rate == 0:
        return float("inf") if recent_rate > 0 else 0.0
    return recent_rate / earlier_rate


def get_ticker_sentiment(ticker: str) -> dict[str, Any]:
    messages, error = fetch_stream_checked(ticker, limit=30)
    if not messages:
        return {
            "ticker": ticker,
            "status": "unavailable" if error else "available",
            "error": error,
            "total_messages": 0,
            "bullish_ratio": 0,
            "bearish_ratio": 0,
            "velocity": 0,
            "unique_users": 0,
        }
    parsed = parse_sentiment(messages)
    velocity = calculate_velocity(parsed["timestamps"])
    parsed["velocity"] = round(velocity, 2)
    parsed["ticker"] = ticker
    parsed["status"] = "available"
    parsed["error"] = None
    return parsed
