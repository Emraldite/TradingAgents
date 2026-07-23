from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


FEATURE_VERSION = "price-volume-v1"


def build_ml_feature_rows(
    stock: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    ticker: str,
) -> list[dict[str, Any]]:
    """Create point-in-time feature rows, including the latest unlabeled row."""
    prices = _prepare(stock, "stock")
    spy = _prepare(benchmark, "benchmark").rename(
        columns={column: f"benchmark_{column}" for column in ("Open", "Close")}
    )
    joined = prices.join(spy[["benchmark_Open", "benchmark_Close"]], how="inner")
    if joined.empty:
        return []

    raw = pd.DataFrame(index=joined.index)
    raw["return_5d"] = joined["Close"].pct_change(5)
    raw["return_20d"] = joined["Close"].pct_change(20)
    raw["return_60d"] = joined["Close"].pct_change(60)
    raw["volume_ratio_20d"] = joined["Volume"] / joined["Volume"].rolling(20).mean()
    raw["volatility_20d"] = joined["Close"].pct_change().rolling(20).std() * math.sqrt(252)
    raw["distance_sma20"] = joined["Close"] / joined["Close"].rolling(20).mean() - 1
    raw["relative_strength_20d"] = raw["return_20d"] - joined[
        "benchmark_Close"
    ].pct_change(20)
    features = raw.shift(1)

    rows: list[dict[str, Any]] = []
    for position, values in enumerate(features.itertuples(index=False, name=None)):
        if any(pd.isna(value) or not math.isfinite(float(value)) for value in values):
            continue
        rows.append(
            {
                "ticker": ticker.upper(),
                "sample_date": joined.index[position].date().isoformat(),
                "feature_version": FEATURE_VERSION,
                "features": {
                    name: float(value)
                    for name, value in zip(features.columns, values, strict=True)
                },
                "entry_price": float(joined["Open"].iloc[position]),
                "benchmark_entry_price": float(joined["benchmark_Open"].iloc[position]),
            }
        )
    return rows


def build_ml_samples(
    stock: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    ticker: str,
    horizon_days: int = 10,
) -> list[dict[str, Any]]:
    """Create point-in-time features and next-open forward-return labels."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")

    prices = _prepare(stock, "stock")
    spy = _prepare(benchmark, "benchmark").rename(
        columns={column: f"benchmark_{column}" for column in ("Open", "Close")}
    )
    joined = prices.join(spy[["benchmark_Open", "benchmark_Close"]], how="inner")
    feature_rows = build_ml_feature_rows(stock, benchmark, ticker=ticker)

    samples: list[dict[str, Any]] = []
    positions = {date.date().isoformat(): position for position, date in enumerate(joined.index)}
    for row in feature_rows:
        position = positions[row["sample_date"]]
        if position + horizon_days >= len(joined):
            continue
        exit_position = position + horizon_days
        entry_price = row["entry_price"]
        benchmark_entry = row["benchmark_entry_price"]
        if entry_price <= 0 or benchmark_entry <= 0:
            continue
        stock_return = float(joined["Close"].iloc[exit_position]) / entry_price - 1
        benchmark_return = (
            float(joined["benchmark_Close"].iloc[exit_position]) / benchmark_entry - 1
        )
        samples.append(
            {
                "ticker": row["ticker"],
                "sample_date": row["sample_date"],
                "label_date": joined.index[exit_position].date().isoformat(),
                "horizon_days": horizon_days,
                "feature_version": row["feature_version"],
                "features": row["features"],
                "stock_return_pct": stock_return * 100,
                "benchmark_return_pct": benchmark_return * 100,
                "alpha_pct": (stock_return - benchmark_return) * 100,
                "outperformed": stock_return > benchmark_return,
            }
        )
    return samples


class MLShadowLedger:
    """SQLite storage for offline ML samples; never used by trade execution."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_shadow_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    sample_date TEXT NOT NULL,
                    label_date TEXT NOT NULL,
                    horizon_days INTEGER NOT NULL,
                    feature_version TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    stock_return_pct REAL NOT NULL,
                    benchmark_return_pct REAL NOT NULL,
                    alpha_pct REAL NOT NULL,
                    outperformed INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(ticker, sample_date, horizon_days, feature_version)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_dataset_builds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    universe TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    horizon_days INTEGER NOT NULL,
                    attempted INTEGER NOT NULL,
                    succeeded INTEGER NOT NULL,
                    failed INTEGER NOT NULL,
                    inserted INTEGER NOT NULL,
                    details_json TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def record(self, samples: list[dict[str, Any]]) -> int:
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        inserted = 0
        with self._connect() as conn:
            for sample in samples:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO ml_shadow_samples (
                        ticker, sample_date, label_date, horizon_days, feature_version,
                        features_json, stock_return_pct, benchmark_return_pct, alpha_pct,
                        outperformed, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sample["ticker"],
                        sample["sample_date"],
                        sample["label_date"],
                        sample["horizon_days"],
                        sample["feature_version"],
                        json.dumps(sample["features"], sort_keys=True),
                        sample["stock_return_pct"],
                        sample["benchmark_return_pct"],
                        sample["alpha_pct"],
                        int(sample["outperformed"]),
                        created_at,
                    ),
                )
                inserted += int(cursor.rowcount > 0)
        return inserted

    def summary(self) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS samples, COUNT(DISTINCT ticker) AS tickers,
                       MIN(sample_date) AS start_date, MAX(label_date) AS end_date,
                       AVG(alpha_pct) AS avg_alpha_pct,
                       AVG(outperformed) AS outperform_rate
                FROM ml_shadow_samples
                """
            ).fetchone()
        return {
            "samples": int(row["samples"] or 0),
            "tickers": int(row["tickers"] or 0),
            "start_date": row["start_date"],
            "end_date": row["end_date"],
            "avg_alpha_pct": float(row["avg_alpha_pct"] or 0.0),
            "outperform_rate_pct": float(row["outperform_rate"] or 0.0) * 100,
        }

    def load_samples(
        self,
        *,
        horizon_days: int | None = None,
        feature_version: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if horizon_days is not None:
            clauses.append("horizon_days=?")
            values.append(horizon_days)
        if feature_version is not None:
            clauses.append("feature_version=?")
            values.append(feature_version)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM ml_shadow_samples{where} ORDER BY sample_date, ticker",
                values,
            ).fetchall()
        samples = []
        for row in rows:
            sample = dict(row)
            sample["features"] = json.loads(sample.pop("features_json"))
            sample["outperformed"] = bool(sample["outperformed"])
            samples.append(sample)
        return samples

    def record_build(self, build: dict[str, Any]) -> int:
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        known = {
            "universe", "source", "source_sha256", "start_date", "end_date",
            "horizon_days", "attempted", "succeeded", "failed", "inserted",
        }
        details = {key: value for key, value in build.items() if key not in known}
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO ml_dataset_builds (
                    created_at, universe, source, source_sha256, start_date, end_date,
                    horizon_days, attempted, succeeded, failed, inserted, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at, build["universe"], build["source"],
                    build["source_sha256"], build["start_date"], build["end_date"],
                    build["horizon_days"], build["attempted"], build["succeeded"],
                    build["failed"], build["inserted"], json.dumps(details, sort_keys=True),
                ),
            )
        return int(cursor.lastrowid)


def _prepare(data: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"Date", "Open", "Close"}
    if name == "stock":
        required.add("Volume")
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{name} data is missing: {', '.join(missing)}")
    frame = data.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    numeric = [column for column in ("Open", "Close", "Volume") if column in frame]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["Date", *numeric]).sort_values("Date")
    return frame.drop_duplicates("Date", keep="last").set_index("Date")
