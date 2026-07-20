from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from tradingagents.dataflows.config import get_config

logger = logging.getLogger(__name__)

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

INSIDER_COLUMNS = [
    "ticker",
    "owner",
    "role",
    "transaction_code",
    "transaction_type",
    "transaction_date",
    "filing_date",
    "shares",
    "price",
    "value",
    "planned_10b5_1",
    "accession_number",
    "filing_url",
    "signal_score",
]

_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0
_MIN_REQUEST_INTERVAL_SECONDS = 0.15


def _cache_path() -> Path:
    config = get_config()
    return Path(config.get("data_cache_dir", "data")) / "sec_insider_cache.db"


def _connect_cache() -> sqlite3.Connection:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=15)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS sec_insider_cache (
            key TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            fetched_at REAL NOT NULL
        )"""
    )
    return conn


def _read_cache(
    key: str,
    *,
    success_ttl_hours: float,
    failure_ttl_hours: float = 1,
) -> dict[str, Any] | None:
    with _connect_cache() as conn:
        row = conn.execute(
            "SELECT payload, fetched_at FROM sec_insider_cache WHERE key=?",
            (key,),
        ).fetchone()
    if not row:
        return None
    try:
        payload = json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        return None
    ttl_hours = (
        failure_ttl_hours
        if payload.get("status") == "unavailable"
        else success_ttl_hours
    )
    if time.time() - float(row[1]) >= ttl_hours * 3600:
        return None
    return payload


def _write_cache(key: str, payload: dict[str, Any]) -> None:
    with _connect_cache() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO sec_insider_cache (key, payload, fetched_at)
               VALUES (?, ?, ?)""",
            (key, json.dumps(payload, default=str), time.time()),
        )


def _sec_user_agent() -> str:
    configured = str(get_config().get("sec_user_agent", "")).strip()
    if not configured:
        configured = os.getenv("TRADINGAGENTS_SEC_USER_AGENT", "").strip()
    if "@" not in configured or "example.com" in configured.lower():
        raise ValueError(
            "TRADINGAGENTS_SEC_USER_AGENT must identify the app and include a real contact email"
        )
    return configured


def _respect_sec_rate_limit() -> None:
    global _LAST_REQUEST_AT
    with _REQUEST_LOCK:
        elapsed = time.monotonic() - _LAST_REQUEST_AT
        if elapsed < _MIN_REQUEST_INTERVAL_SECONDS:
            time.sleep(_MIN_REQUEST_INTERVAL_SECONDS - elapsed)
        _LAST_REQUEST_AT = time.monotonic()


def _sec_get(url: str) -> requests.Response:
    _respect_sec_rate_limit()
    response = requests.get(
        url,
        headers={
            "User-Agent": _sec_user_agent(),
            "Accept-Encoding": "gzip, deflate",
        },
        timeout=20,
    )
    response.raise_for_status()
    return response


def _sec_get_json(url: str) -> dict[str, Any]:
    payload = _sec_get(url).json()
    if not isinstance(payload, dict):
        raise ValueError("SEC returned a non-object JSON response")
    return payload


def _strip_namespaces(root: ET.Element) -> None:
    for element in root.iter():
        if "}" in element.tag:
            element.tag = element.tag.split("}", 1)[1]


def _node_text(node: ET.Element, path: str) -> str:
    value = node.findtext(path)
    return str(value or "").strip()


def _to_float(raw: Any) -> float:
    try:
        return float(raw or 0)
    except (TypeError, ValueError):
        return 0.0


def _xml_flag(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def _reporting_owner(root: ET.Element) -> tuple[str, str]:
    names: list[str] = []
    roles: list[str] = []
    for reporting_owner in root.findall(".//reportingOwner"):
        name = _node_text(reporting_owner, "reportingOwnerId/rptOwnerName")
        relationship = reporting_owner.find("reportingOwnerRelationship")
        if name and name not in names:
            names.append(name)
        if relationship is None:
            continue
        title = _node_text(relationship, "officerTitle")
        if title:
            roles.append(title)
        if _xml_flag(_node_text(relationship, "isDirector")):
            roles.append("Director")
        if _xml_flag(_node_text(relationship, "isOfficer")) and not title:
            roles.append("Officer")
        if _xml_flag(_node_text(relationship, "isTenPercentOwner")):
            roles.append("10% Owner")
    unique_roles = list(dict.fromkeys(roles))
    return ", ".join(names) or "Unknown reporting owner", ", ".join(unique_roles) or "Insider"


def _transaction_score(record: dict[str, Any], as_of_date: date) -> int:
    value = float(record.get("value", 0) or 0)
    role = str(record.get("role", "")).lower()
    filing_date = _parse_date(record.get("filing_date"))
    recent = filing_date is not None and (as_of_date - filing_date).days <= 7

    if record.get("transaction_code") == "P":
        score = 3
        if value >= 1_000_000:
            score += 3
        elif value >= 100_000:
            score += 2
        elif value >= 10_000:
            score += 1
        if any(label in role for label in ("chief executive", "ceo", "chief financial", "cfo")):
            score += 2
        elif any(label in role for label in ("officer", "director", "president")):
            score += 1
        if recent:
            score += 1
        if record.get("planned_10b5_1"):
            score -= 2
        return max(1, min(score, 10))

    score = -1
    if value >= 1_000_000:
        score -= 1
    if record.get("planned_10b5_1"):
        score += 1
    return max(-3, min(score, 0))


def parse_form4_xml(
    xml_text: str,
    *,
    ticker: str,
    filing_date: str,
    accession_number: str,
    filing_url: str,
    as_of_date: date,
) -> list[dict[str, Any]]:
    """Parse only open-market non-derivative purchases and sales from Form 4."""
    root = ET.fromstring(xml_text)
    _strip_namespaces(root)
    owner, role = _reporting_owner(root)
    issuer_ticker = _node_text(root, ".//issuerTradingSymbol").upper() or ticker.upper()
    records: list[dict[str, Any]] = []

    for transaction in root.findall(".//nonDerivativeTransaction"):
        code = _node_text(transaction, "transactionCoding/transactionCode").upper()
        if code not in {"P", "S"}:
            continue
        transaction_date = _node_text(transaction, "transactionDate/value")
        parsed_transaction_date = _parse_date(transaction_date)
        if parsed_transaction_date is None or parsed_transaction_date > as_of_date:
            continue
        shares = _to_float(
            _node_text(transaction, "transactionAmounts/transactionShares/value")
        )
        price = _to_float(
            _node_text(
                transaction,
                "transactionAmounts/transactionPricePerShare/value",
            )
        )
        record = {
            "ticker": issuer_ticker,
            "owner": owner,
            "role": role,
            "transaction_code": code,
            "transaction_type": "purchase" if code == "P" else "sale",
            "transaction_date": parsed_transaction_date.isoformat(),
            "filing_date": filing_date,
            "shares": shares,
            "price": price,
            "value": shares * price if shares > 0 and price > 0 else 0.0,
            "planned_10b5_1": _xml_flag(
                _node_text(transaction, "transactionCoding/aff10b5One")
            ),
            "accession_number": accession_number,
            "filing_url": filing_url,
        }
        record["signal_score"] = _transaction_score(record, as_of_date)
        records.append(record)
    return records


def _parse_date(raw: Any) -> date | None:
    try:
        return datetime.strptime(str(raw), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _ticker_map() -> dict[str, dict[str, Any]]:
    cache_key = "sec_company_tickers_v1"
    cached = _read_cache(cache_key, success_ttl_hours=24)
    if cached is not None:
        if cached.get("status") != "ok":
            raise RuntimeError(str(cached.get("reason") or "SEC ticker map unavailable"))
        return dict(cached.get("records") or {})

    try:
        payload = _sec_get_json(SEC_TICKERS_URL)
        records = {
            str(item.get("ticker", "")).upper(): {
                "cik": int(item["cik_str"]),
                "title": str(item.get("title", "")),
            }
            for item in payload.values()
            if isinstance(item, dict) and item.get("ticker") and item.get("cik_str")
        }
    except Exception as exc:
        reason = f"SEC ticker map request failed: {type(exc).__name__}"
        _write_cache(cache_key, {"status": "unavailable", "reason": reason, "records": {}})
        raise RuntimeError(reason) from exc

    _write_cache(cache_key, {"status": "ok", "reason": "", "records": records})
    return records


def _normalize_sec_ticker(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "-")


def _recent_form4_filings(
    submissions: dict[str, Any],
    *,
    as_of_date: date,
    lookback_days: int,
    max_filings: int,
) -> list[dict[str, str]]:
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form") or []
    cutoff = as_of_date - timedelta(days=lookback_days)
    filings: list[dict[str, str]] = []
    for index, form in enumerate(forms):
        # Amended filings can repeat original transactions; exclude them rather
        # than risk double-counting a signal.
        if str(form).upper() != "4":
            continue
        try:
            filing_date = str(recent["filingDate"][index])
            parsed_date = _parse_date(filing_date)
            if parsed_date is None or not (cutoff <= parsed_date <= as_of_date):
                continue
            filings.append(
                {
                    "accession_number": str(recent["accessionNumber"][index]),
                    "filing_date": filing_date,
                    "primary_document": str(recent["primaryDocument"][index]),
                }
            )
        except (IndexError, KeyError, TypeError):
            continue
        if len(filings) >= max_filings:
            break
    return filings


def _fetch_activity_payload(
    ticker: str,
    *,
    as_of_date: date,
    lookback_days: int,
    max_filings: int,
) -> dict[str, Any]:
    normalized = _normalize_sec_ticker(ticker)
    config = get_config()
    cache_hours = float(config.get("insider_cache_hours", 12))
    cache_key = (
        f"sec_form4_v1:{normalized}:{as_of_date.isoformat()}:"
        f"{lookback_days}:{max_filings}"
    )
    cached = _read_cache(cache_key, success_ttl_hours=cache_hours)
    if cached is not None:
        return cached

    try:
        company = _ticker_map().get(normalized)
        if not company:
            payload = {
                "status": "unsupported",
                "reason": f"No SEC CIK mapping exists for {normalized}",
                "records": [],
            }
            _write_cache(cache_key, payload)
            return payload

        cik = int(company["cik"])
        submissions = _sec_get_json(SEC_SUBMISSIONS_URL.format(cik=cik))
        filings = _recent_form4_filings(
            submissions,
            as_of_date=as_of_date,
            lookback_days=lookback_days,
            max_filings=max_filings,
        )
        records: list[dict[str, Any]] = []
        filing_errors = 0
        for filing in filings:
            accession_compact = filing["accession_number"].replace("-", "")
            document_name = Path(filing["primary_document"]).name
            filing_url = (
                f"{SEC_ARCHIVES_BASE}/{cik}/{accession_compact}/{document_name}"
            )
            try:
                xml_text = _sec_get(filing_url).text
                records.extend(
                    parse_form4_xml(
                        xml_text,
                        ticker=normalized,
                        filing_date=filing["filing_date"],
                        accession_number=filing["accession_number"],
                        filing_url=filing_url,
                        as_of_date=as_of_date,
                    )
                )
            except Exception as exc:
                filing_errors += 1
                logger.warning(
                    "SEC Form 4 filing parse failed for %s (%s): %s",
                    normalized,
                    filing["accession_number"],
                    type(exc).__name__,
                )

        if filings and filing_errors == len(filings):
            payload = {
                "status": "unavailable",
                "reason": "Every eligible SEC Form 4 filing failed to load",
                "records": [],
            }
        else:
            payload = {
                "status": "ok" if records else "no_activity",
                "reason": "" if records else "No open-market Form 4 purchases or sales",
                "records": records,
            }
    except Exception as exc:
        payload = {
            "status": "unavailable",
            "reason": f"SEC insider data request failed: {type(exc).__name__}",
            "records": [],
        }

    _write_cache(cache_key, payload)
    return payload


def get_sec_insider_activity(
    ticker: str,
    *,
    as_of_date: str,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    config = get_config()
    try:
        _sec_user_agent()
    except ValueError as exc:
        frame = pd.DataFrame(columns=INSIDER_COLUMNS)
        frame.attrs.update(data_status="unavailable", data_reason=str(exc))
        logger.warning("SEC Form 4 data disabled: %s", exc)
        return frame
    days = int(
        lookback_days
        if lookback_days is not None
        else config.get("insider_lookback_days", 30)
    )
    max_filings = int(config.get("insider_max_filings", 20))
    parsed_as_of = _parse_date(as_of_date)
    if parsed_as_of is None:
        frame = pd.DataFrame(columns=INSIDER_COLUMNS)
        frame.attrs.update(data_status="unavailable", data_reason="Invalid analysis date")
        return frame
    if days < 1 or max_filings < 1:
        frame = pd.DataFrame(columns=INSIDER_COLUMNS)
        frame.attrs.update(
            data_status="unavailable",
            data_reason="Insider lookback and filing limit must be positive",
        )
        return frame

    payload = _fetch_activity_payload(
        ticker,
        as_of_date=parsed_as_of,
        lookback_days=days,
        max_filings=max_filings,
    )
    frame = pd.DataFrame(payload.get("records") or [], columns=INSIDER_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(
            ["filing_date", "transaction_date"], ascending=False
        ).reset_index(drop=True)
    frame.attrs.update(
        data_status=str(payload.get("status") or "unavailable"),
        data_reason=str(payload.get("reason") or ""),
    )
    log_method = logger.warning if frame.attrs["data_status"] == "unavailable" else logger.info
    log_method(
        "SEC Form 4 activity for %s: status=%s records=%d%s",
        ticker.upper(),
        frame.attrs["data_status"],
        len(frame),
        f" reason={frame.attrs['data_reason']}" if frame.attrs["data_reason"] else "",
    )
    return frame


def summarize_sec_insider_activity(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "purchase_count": 0,
            "sale_count": 0,
            "purchase_value": 0.0,
            "sale_value": 0.0,
            "unique_buyers": 0,
            "signal_score": 0,
        }
    purchases = frame[frame["transaction_code"] == "P"]
    sales = frame[frame["transaction_code"] == "S"]
    unique_buyers = int(purchases["owner"].nunique()) if not purchases.empty else 0
    score = int(frame["signal_score"].sum())
    if unique_buyers >= 2:
        score += 2
    return {
        "purchase_count": int(len(purchases)),
        "sale_count": int(len(sales)),
        "purchase_value": float(purchases["value"].sum()) if not purchases.empty else 0.0,
        "sale_value": float(sales["value"].sum()) if not sales.empty else 0.0,
        "unique_buyers": unique_buyers,
        "signal_score": max(-10, min(score, 10)),
    }
