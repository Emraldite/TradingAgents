from __future__ import annotations

import logging
from typing import Any

from tradingagents.dataflows.stocktwits_data import (
    fetch_stream_checked,
    parse_sentiment,
    calculate_velocity,
)

logger = logging.getLogger(__name__)


def detect_manipulation(ticker: str) -> dict[str, Any]:
    messages, error = fetch_stream_checked(ticker, limit=30)
    if not messages:
        return {
            "ticker": ticker,
            "status": "unavailable" if error else "available",
            "error": error,
            "organic_score": None,
            "manipulation_risk": None,
            "recommendation": "UNKNOWN",
            "details": {},
        }

    parsed = parse_sentiment(messages)
    scores: dict[str, float] = {}
    details: dict[str, Any] = {}

    spike = _score_spike_velocity(parsed)
    scores["spike_velocity"] = spike
    details["spike_velocity"] = {
        "score": spike,
        "detail": _velocity_detail(parsed),
    }

    uniformity = _score_sentiment_uniformity(parsed)
    scores["sentiment_uniformity"] = uniformity
    details["sentiment_uniformity"] = {
        "score": uniformity,
        "detail": _uniformity_detail(parsed),
    }

    volume_divergence = _score_volume_divergence(parsed)
    scores["volume_divergence"] = volume_divergence
    details["volume_divergence"] = {
        "score": volume_divergence,
        "detail": _volume_detail(parsed),
    }

    total_manipulation = sum(scores.values())
    organic_score = max(0, 10 - total_manipulation)
    recommendation = "PASS" if total_manipulation <= 6 else "REJECT"

    return {
        "ticker": ticker,
        "status": "available",
        "error": None,
        "organic_score": round(organic_score, 1),
        "manipulation_risk": round(total_manipulation, 1),
        "recommendation": recommendation,
        "details": details,
    }


def _score_spike_velocity(parsed: dict) -> float:
    velocity = parsed.get("velocity", 0)
    if velocity == float("inf"):
        return 2.0 if parsed.get("total_messages", 0) > 10 else 0.0
    if velocity > 5:
        return 2.0
    if velocity > 2:
        return 1.0
    return 0.0


def _score_sentiment_uniformity(parsed: dict) -> float:
    total = parsed.get("total_messages", 0)
    if total < 5:
        return 0.0
    bullish_ratio = parsed.get("bullish_ratio", 0)
    bearish_ratio = parsed.get("bearish_ratio", 0)
    if bullish_ratio > 0.95 or bearish_ratio > 0.95:
        return 1.0
    return 0.0


def _score_volume_divergence(parsed: dict) -> float:
    total = parsed.get("total_messages", 0)
    unique = parsed.get("unique_users", 0)
    if total == 0:
        return 0.0
    if total > 15 and unique <= 3:
        return 2.0
    if total > 10 and unique <= 2:
        return 1.0
    return 0.0


def _velocity_detail(parsed: dict) -> str:
    v = parsed.get("velocity", 0)
    total = parsed.get("total_messages", 0)
    return f"velocity {v}x, {total} messages"


def _uniformity_detail(parsed: dict) -> str:
    return f"bullish {parsed.get('bullish_ratio', 0):.0%}, bearish {parsed.get('bearish_ratio', 0):.0%}"


def _volume_detail(parsed: dict) -> str:
    return f"{parsed.get('total_messages', 0)} msgs from {parsed.get('unique_users', 0)} users"
