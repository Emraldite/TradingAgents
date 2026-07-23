from __future__ import annotations

import csv
import hashlib
import io
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import requests


DEFAULT_HISTORY_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20%28Updated%29.csv"
)


@dataclass(frozen=True)
class MembershipInterval:
    ticker: str
    start_date: date
    end_date: date

    def contains(self, value: date) -> bool:
        return self.start_date <= value <= self.end_date


def fetch_membership_history(
    *,
    source_url: str = DEFAULT_HISTORY_URL,
    cache_path: str | Path | None = None,
    refresh: bool = False,
    timeout_seconds: int = 30,
) -> tuple[bytes, str]:
    """Fetch point-in-time membership snapshots, using an explicit local cache."""
    cache = Path(cache_path) if cache_path else None
    if cache and cache.exists() and not refresh:
        return cache.read_bytes(), str(cache.resolve())

    response = requests.get(source_url, timeout=timeout_seconds)
    response.raise_for_status()
    content = response.content
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(f"{cache.suffix}.tmp")
        temporary.write_bytes(content)
        temporary.replace(cache)
    return content, source_url


def parse_membership_history(content: bytes | str) -> list[MembershipInterval]:
    """Convert daily ``date,tickers`` snapshots into inclusive membership intervals."""
    text = content.decode("utf-8-sig") if isinstance(content, bytes) else content
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or not {"date", "tickers"}.issubset(reader.fieldnames):
        raise ValueError("membership history must contain date and tickers columns")

    snapshots: OrderedDict[date, set[str]] = OrderedDict()
    last_date: date | None = None
    for row in reader:
        snapshot_date = date.fromisoformat(row["date"].strip())
        if last_date and snapshot_date < last_date:
            raise ValueError("membership snapshots must be in ascending date order")
        members = {
            normalize_ticker(value)
            for value in row["tickers"].split(",")
            if value.strip()
        }
        if not members:
            raise ValueError(f"membership snapshot {snapshot_date} has no tickers")
        # Upstream occasionally publishes a corrected second row for the same date.
        # Treat the last row as the effective closing snapshot.
        snapshots[snapshot_date] = members
        last_date = snapshot_date

    active: dict[str, date] = {}
    intervals: list[MembershipInterval] = []
    previous_date: date | None = None
    for snapshot_date, members in snapshots.items():
        for ticker in sorted(set(active) - members):
            if previous_date is None:
                raise ValueError("membership history has an invalid first snapshot")
            intervals.append(MembershipInterval(ticker, active.pop(ticker), previous_date))
        for ticker in sorted(members - set(active)):
            active[ticker] = snapshot_date
        previous_date = snapshot_date

    if previous_date is None:
        raise ValueError("membership history is empty")
    for ticker, start_date in active.items():
        intervals.append(MembershipInterval(ticker, start_date, previous_date))
    return sorted(intervals, key=lambda item: (item.ticker, item.start_date))


def intervals_by_ticker(
    intervals: Iterable[MembershipInterval],
) -> dict[str, list[MembershipInterval]]:
    grouped: dict[str, list[MembershipInterval]] = defaultdict(list)
    for interval in intervals:
        grouped[interval.ticker].append(interval)
    return dict(grouped)


def tickers_overlapping(
    grouped: dict[str, list[MembershipInterval]],
    start_date: date,
    end_date: date,
) -> list[str]:
    return sorted(
        ticker
        for ticker, intervals in grouped.items()
        if any(item.start_date <= end_date and item.end_date >= start_date for item in intervals)
    )


def filter_samples_for_membership(
    samples: list[dict],
    intervals: list[MembershipInterval],
) -> list[dict]:
    return [
        sample
        for sample in samples
        if any(item.contains(date.fromisoformat(sample["sample_date"])) for item in intervals)
    ]


def normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()


def yahoo_ticker(ticker: str) -> str:
    """Translate class-share punctuation without changing canonical identity."""
    return normalize_ticker(ticker).replace(".", "-")


def content_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
