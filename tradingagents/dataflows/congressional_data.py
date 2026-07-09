from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CACHE_DB = str(CACHE_DIR / "cache.db")
SCORED_TRADE_COLUMNS = [
    "ticker",
    "politician",
    "trade_type",
    "amount",
    "disclosure_date",
    "sources",
    "committees",
    "conviction_score",
    "score_breakdown",
]

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

CAPITOL_TRADES_URL = "https://www.capitoltrades.com/trades"
QUIVER_URL = "https://www.quiverquant.com/congresstrading"
_CAPITOL_TICKER_RE = re.compile(r"^[A-Z][A-Z./-]{0,9}(?::[A-Z]{2})?$")
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}$")

COMMITTEE_SECTOR_MAP: dict[str, list[str]] = {
    "Armed Services": ["LMT", "RTX", "NOC", "GD", "BA", "LHX", "HII"],
    "Energy and Natural Resources": ["XOM", "CVX", "COP", "SLB", "EOG", "OXY"],
    "Banking, Housing, Urban Affairs": ["JPM", "GS", "BAC", "MS", "C", "WFC"],
    "Health, Education, Labor": ["UNH", "JNJ", "PFE", "ABBV", "MRK", "CVS"],
    "Commerce, Science, Transportation": ["AMZN", "GOOGL", "META", "UPS", "FDX"],
    "Agriculture": ["ADM", "BG", "DE", "MON"],
}

POLITICIAN_COMMITTEE_MAP: dict[str, list[str]] = {
    "Nancy Pelosi": ["Finance"],
    "Dan Crenshaw": ["Armed Services", "Homeland Security"],
    "Josh Gottheimer": ["Finance"],
    "John Curtis": ["Energy and Natural Resources"],
    "Mark Green": ["Armed Services", "Homeland Security"],
    "Mike Garcia": ["Armed Services"],
    "Ro Khanna": ["Armed Services"],
    "Seth Moulton": ["Armed Services"],
    "Elissa Slotkin": ["Armed Services"],
    "Tommy Tuberville": ["Armed Services", "Banking, Housing, Urban Affairs"],
}


def _get_cache() -> sqlite3.Connection:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_DB)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS congressional_cache (
            key TEXT PRIMARY KEY,
            data TEXT,
            fetched_at REAL
        )"""
    )
    return conn


def _cached_or_fetch(cache_key: str, ttl_hours: float) -> list[dict] | None:
    conn = _get_cache()
    row = conn.execute(
        "SELECT data, fetched_at FROM congressional_cache WHERE key = ?", (cache_key,)
    ).fetchone()
    if row:
        data, fetched_at = row
        if time.time() - fetched_at < ttl_hours * 3600:
            return json.loads(data)
    return None


def _write_cache(cache_key: str, data: list[dict]) -> None:
    conn = _get_cache()
    conn.execute(
        "INSERT OR REPLACE INTO congressional_cache (key, data, fetched_at) VALUES (?, ?, ?)",
        (cache_key, json.dumps(data), time.time()),
    )
    conn.commit()


def _clean_ticker(raw: str) -> str:
    raw = raw.replace("N/A", "").replace("/", "").strip()
    m = re.search(r"\b[A-Z]{1,5}\b", raw)
    if m:
        ticker = m.group(0)
        if len(ticker) >= 1 and ticker not in ("A", "I", "AN", "AT", "IN", "ON", "TO", "BY", "OF", "FOR", "THE", "CITY", "STATE", "COUNTY", "CALIFORNIA"):
            return ticker
    return ""


def _clean_politician_name(raw: str) -> str:
    for party in ["Democrat", "Republican", "Independent"]:
        if party in raw:
            raw = raw.split(party)[0]
            break
    raw = re.sub(r"(House|Senate)", "", raw)
    raw = re.sub(r"[A-Z]{2}$", "", raw.strip())
    return raw.strip()


def _parse_capitol_amount(text: str) -> float:
    text = text.replace("$", "").replace(",", "").replace("\u2013", "-").replace("\u2014", "-").strip()
    parts = text.split("-")
    if len(parts) == 2:
        try:
            lower = _parse_single_amount(parts[0].strip())
            upper = _parse_single_amount(parts[1].strip())
            if upper > 0:
                return upper
            return lower
        except (ValueError, TypeError):
            pass
    return _parse_single_amount(text)


def _parse_single_amount(s: str) -> float:
    s = s.strip()
    if s.upper() == "N/A" or not s:
        return 0.0
    if "M" in s.upper():
        return float(s.upper().replace("M", "").strip()) * 1_000_000
    if "K" in s.upper():
        return float(s.upper().replace("K", "").strip()) * 1_000
    try:
        return float(s)
    except ValueError:
        return 0.0


def _format_trade_date(text: str) -> str:
    text = text.strip()
    for fmt in ("%d %b%Y", "%d %b %Y", "%Y-%m-%d", "%b %d %Y"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    return text


def _dedupe_records(records: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[tuple[str, str, str, str, float, str]] = set()
    for record in records:
        key = (
            str(record.get("ticker", "")),
            str(record.get("politician", "")),
            str(record.get("trade_type", "")),
            str(record.get("disclosure_date", "")),
            float(record.get("amount", 0.0) or 0.0),
            str(record.get("source", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _parse_capitol_trades_from_table(soup: BeautifulSoup) -> list[dict]:
    records = []
    rows = soup.select("table tbody tr")
    if not rows:
        return records

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 8:
            continue
        try:
            politician_raw = cells[0].get_text(strip=True)
            ticker_raw = cells[1].get_text(strip=True).upper()
            trade_date_text = cells[3].get_text(strip=True)
            trade_type = cells[6].get_text(strip=True)
            amount_text = cells[7].get_text(strip=True)
        except (IndexError, AttributeError):
            continue

        ticker = _clean_ticker(ticker_raw)
        if not ticker or len(ticker) > 6:
            continue

        politician = _clean_politician_name(politician_raw)
        records.append(
            {
                "ticker": ticker,
                "politician": politician,
                "trade_type": trade_type,
                "amount": _parse_capitol_amount(amount_text),
                "disclosure_date": _format_trade_date(trade_date_text),
                "source": "capitol_trades",
                "committees": POLITICIAN_COMMITTEE_MAP.get(politician, []),
            }
        )
    return records


def _parse_capitol_trades_from_cards(soup: BeautifulSoup) -> list[dict]:
    records = []
    tokens = [token.strip() for token in soup.stripped_strings if token.strip()]
    for i in range(len(tokens) - 13):
        party_line = tokens[i + 1]
        ticker_raw = tokens[i + 3].upper()
        trade_time = tokens[i + 4]
        if "House" not in party_line and "Senate" not in party_line:
            continue
        if not _CAPITOL_TICKER_RE.match(ticker_raw):
            continue
        if not _TIME_RE.match(trade_time):
            continue
        if tokens[i + 8].lower() != "days":
            continue

        politician = _clean_politician_name(tokens[i])
        ticker = _clean_ticker(ticker_raw)
        trade_type = tokens[i + 11].strip().lower()
        amount_text = tokens[i + 12]
        trade_date_text = _format_trade_date(f"{tokens[i + 6]} {tokens[i + 7]}")
        if not politician or not ticker:
            continue
        if trade_type not in {"buy", "sell", "exchange"}:
            continue

        records.append(
            {
                "ticker": ticker,
                "politician": politician,
                "trade_type": trade_type,
                "amount": _parse_capitol_amount(amount_text),
                "disclosure_date": trade_date_text,
                "source": "capitol_trades",
                "committees": POLITICIAN_COMMITTEE_MAP.get(politician, []),
            }
        )
    return records


def _parse_capitol_trades_html(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    records = _parse_capitol_trades_from_table(soup)
    if not records:
        records = _parse_capitol_trades_from_cards(soup)
    return _dedupe_records(records)


def _scrape_capitol_trades() -> list[dict]:
    try:
        resp = requests.get(CAPITOL_TRADES_URL, headers={"User-Agent": _UA}, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Capitol Trades scrape failed: %s", exc)
        return []

    records = _parse_capitol_trades_html(resp.text)
    if not records:
        logger.warning("No trade rows found on Capitol Trades")
    logger.info("Scraped %d trades from Capitol Trades", len(records))
    return records


def _parse_quiver_html(html: str) -> list[dict]:
    records = []
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="recentTradesTable")
    if table is None:
        for candidate in soup.find_all("table"):
            header_text = " ".join(candidate.stripped_strings).lower()
            if "politician" in header_text and "traded" in header_text:
                table = candidate
                break
    if table is None:
        return records
    rows = table.find("tbody")
    if not rows:
        return records
    rows = rows.find_all("tr")
    if not rows:
        return records

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 5:
            continue
        try:
            ticker = _clean_ticker(cells[0].get_text(strip=True).upper())
            trade_type = cells[1].get_text(strip=True)
            politician = _clean_politician_name(cells[2].get_text(strip=True))
            date_text = _format_trade_date(cells[4].get_text(strip=True))
            amount_text = cells[5].get_text(strip=True) if len(cells) > 5 else ""
        except (IndexError, AttributeError):
            continue

        if not ticker or not politician:
            continue
        records.append(
            {
                "ticker": ticker,
                "politician": politician,
                "trade_type": trade_type,
                "amount": _parse_amount(amount_text),
                "disclosure_date": date_text,
                "source": "quiver",
                "committees": POLITICIAN_COMMITTEE_MAP.get(politician, []),
            }
        )
    return _dedupe_records(records)


def _scrape_quiver() -> list[dict]:
    try:
        resp = requests.get(QUIVER_URL, headers={"User-Agent": _UA}, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Quiver scrape failed: %s", exc)
        return []

    records = _parse_quiver_html(resp.text)
    if not records:
        logger.warning("No trade table found on Quiver (likely layout or JS change)")
    logger.info("Scraped %d trades from Quiver", len(records))
    return records


def get_congressional_trades(
    lookback_days: int = 45,
    cache_hours: float = 6,
) -> pd.DataFrame:
    cache_key = f"congressional_trades_v2_{lookback_days}"
    cached = _cached_or_fetch(cache_key, cache_hours)
    if cached is not None:
        logger.info("Using cached congressional trades (%d records)", len(cached))
        return pd.DataFrame(cached)

    capitol = _scrape_capitol_trades()
    quiver = _scrape_quiver()

    all_records = capitol + quiver
    if not all_records:
        logger.warning("No congressional data from either source")
        return pd.DataFrame()

    cutoff = datetime.now() - timedelta(days=lookback_days)
    df = pd.DataFrame(all_records)
    df["disclosure_date"] = pd.to_datetime(df["disclosure_date"], errors="coerce")
    df = df[df["disclosure_date"] >= cutoff].copy()
    df["disclosure_date"] = df["disclosure_date"].dt.strftime("%Y-%m-%d")
    df = df.sort_values("disclosure_date", ascending=False).reset_index(drop=True)

    _write_cache(cache_key, df.to_dict("records"))
    logger.info(
        "Fetched %d congressional trades (Capitol: %d, Quiver: %d)",
        len(df), len(capitol), len(quiver),
    )
    return df


def _parse_amount(amount_str: str) -> float:
    if not amount_str:
        return 0.0
    cleaned = amount_str.replace("$", "").replace(",", "").strip()
    if cleaned.startswith("Over"):
        cleaned = cleaned.replace("Over", "").strip()
    if "M" in cleaned:
        return float(cleaned.replace("M", "").strip()) * 1_000_000
    if "K" in cleaned:
        return float(cleaned.replace("K", "").strip()) * 1_000
    cleaned = re.sub(r"[^0-9.]", "", cleaned)
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def score_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=SCORED_TRADE_COLUMNS)

    df["disclosure_date"] = pd.to_datetime(df["disclosure_date"], errors="coerce")
    cross_ref = df.groupby(["ticker", "politician", df["disclosure_date"].dt.date]).agg(
        sources=("source", lambda x: list(x)),
        trade_type=("trade_type", "first"),
        amount=("amount", "first"),
        committees=("committees", "first"),
    ).reset_index()

    rows = []
    for _, row in cross_ref.iterrows():
        score = 0
        breakdown: dict[str, int | float] = {}

        sources = row.get("sources", [])
        cross_bonus = min(len(set(sources)) * 2, 3)
        score += cross_bonus
        breakdown["cross_reference"] = cross_bonus

        committees = row.get("committees", [])
        ticker = row["ticker"]
        matched = [
            cmte for cmte in committees
            if ticker in COMMITTEE_SECTOR_MAP.get(cmte, [])
        ]
        committee_score = min(len(matched) * 2, 3)
        score += committee_score
        breakdown["committee_relevance"] = committee_score

        amount = row.get("amount", 0)
        if amount > 500_000:
            size_score = 2
        elif amount > 100_000:
            size_score = 1
        else:
            size_score = 0
        score += size_score
        breakdown["trade_size"] = size_score

        disclosure = row.get("disclosure_date")
        if pd.notna(disclosure):
            days_ago = (datetime.now() - pd.Timestamp(disclosure)).days
            if days_ago <= 7:
                recency_score = 2
            elif days_ago <= 30:
                recency_score = 1
            else:
                recency_score = 0
        else:
            recency_score = 0
        score += recency_score
        breakdown["recency"] = recency_score

        rows.append(
            {
                "ticker": ticker,
                "politician": row["politician"],
                "trade_type": row["trade_type"],
                "amount": amount,
                "disclosure_date": row["disclosure_date"],
                "sources": sources,
                "committees": committees,
                "conviction_score": score,
                "score_breakdown": breakdown,
            }
        )

    result = pd.DataFrame(rows, columns=SCORED_TRADE_COLUMNS)
    if result.empty:
        return result
    result = result.sort_values("conviction_score", ascending=False)
    return result


def get_conviction_watchlist(
    lookback_days: int = 45,
    min_score: int = 6,
    cache_hours: float = 6,
) -> pd.DataFrame:
    df = get_congressional_trades(
        lookback_days=lookback_days, cache_hours=cache_hours
    )
    df = score_trades(df)
    if df.empty or "disclosure_date" not in df.columns:
        logger.info("Congressional watchlist: 0 tickers above score %d", min_score)
        return pd.DataFrame(columns=SCORED_TRADE_COLUMNS)
    df["disclosure_date"] = pd.to_datetime(df["disclosure_date"], errors="coerce")
    watchlist = df[df["conviction_score"] >= min_score].copy()
    logger.info(
        "Congressional watchlist: %d tickers above score %d",
        watchlist["ticker"].nunique() if not watchlist.empty else 0,
        min_score,
    )
    return watchlist
