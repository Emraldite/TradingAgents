from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURE_VERSION_V2 = "price-volume-v2"
REQUIRED_PRICE_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")


class HistoricalPriceCache:
    """Persistent audited OHLCV cache so feature revisions do not redownload prices."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS historical_prices (
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    source TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(ticker, date)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS price_download_attempts (
                    ticker TEXT PRIMARY KEY,
                    requested_start TEXT NOT NULL,
                    requested_end TEXT NOT NULL,
                    status TEXT NOT NULL,
                    rows INTEGER NOT NULL,
                    error TEXT,
                    attempted_at TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def reusable_attempt(self, ticker: str, start: str, end: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM price_download_attempts WHERE ticker=?",
                (ticker.upper(),),
            ).fetchone()
        if row is None:
            return None
        attempt = dict(row)
        if (
            attempt["status"] in {"success", "unavailable"}
            and attempt["requested_start"] <= start
            and attempt["requested_end"] >= end
        ):
            return attempt
        return None

    def record_download(
        self,
        ticker: str,
        data: pd.DataFrame,
        *,
        requested_start: str,
        requested_end: str,
        source: str = "yfinance",
        status: str = "success",
        error: str | None = None,
    ) -> int:
        symbol = ticker.upper()
        attempted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        frame = _prepare_prices(data, symbol) if not data.empty else data
        rows = []
        if not frame.empty:
            rows = [
                (
                    symbol,
                    index.date().isoformat(),
                    float(row.Open),
                    float(row.High),
                    float(row.Low),
                    float(row.Close),
                    float(row.Volume),
                    source,
                    attempted_at,
                )
                for index, row in frame.iterrows()
            ]
        with self._connect() as conn:
            if rows:
                conn.executemany(
                    """
                    INSERT INTO historical_prices (
                        ticker, date, open, high, low, close, volume, source, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticker, date) DO UPDATE SET
                        open=excluded.open, high=excluded.high, low=excluded.low,
                        close=excluded.close, volume=excluded.volume,
                        source=excluded.source, updated_at=excluded.updated_at
                    """,
                    rows,
                )
            conn.execute(
                """
                INSERT INTO price_download_attempts (
                    ticker, requested_start, requested_end, status, rows, error, attempted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    requested_start=excluded.requested_start,
                    requested_end=excluded.requested_end,
                    status=excluded.status,
                    rows=excluded.rows,
                    error=excluded.error,
                    attempted_at=excluded.attempted_at
                """,
                (
                    symbol,
                    requested_start,
                    requested_end,
                    status,
                    len(rows),
                    error,
                    attempted_at,
                ),
            )
        return len(rows)

    def load(self, ticker: str, *, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        clauses = ["ticker=?"]
        values: list[Any] = [ticker.upper()]
        if start:
            clauses.append("date>=?")
            values.append(start)
        if end:
            clauses.append("date<=?")
            values.append(end)
        query = f"""
            SELECT date AS Date, open AS Open, high AS High, low AS Low,
                   close AS Close, volume AS Volume
            FROM historical_prices
            WHERE {' AND '.join(clauses)}
            ORDER BY date
        """
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=values)

    def summary(self) -> dict[str, Any]:
        with self._connect() as conn:
            prices = conn.execute(
                """
                SELECT COUNT(*) AS rows, COUNT(DISTINCT ticker) AS tickers,
                       MIN(date) AS start_date, MAX(date) AS end_date
                FROM historical_prices
                """
            ).fetchone()
            attempts = conn.execute(
                """
                SELECT COUNT(*) AS attempts,
                       SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) AS successful,
                       SUM(CASE WHEN status!='success' THEN 1 ELSE 0 END) AS failed
                FROM price_download_attempts
                """
            ).fetchone()
        return {**dict(prices), **dict(attempts)}


def build_v2_samples(
    stock: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    ticker: str,
    horizon_days: int = 10,
) -> list[dict[str, Any]]:
    """Build rich prior-close features with next-open to future-open labels."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")
    prices = _prepare_prices(stock, ticker)
    spy = _prepare_prices(benchmark, "SPY").rename(
        columns={column: f"benchmark_{column}" for column in REQUIRED_PRICE_COLUMNS[1:]}
    )
    joined = prices.join(spy, how="inner")
    if joined.empty:
        return []

    close = joined["Close"]
    returns = close.pct_change()
    benchmark_close = joined["benchmark_Close"]
    benchmark_returns = benchmark_close.pct_change()
    raw = pd.DataFrame(index=joined.index)
    for period in (1, 2, 5, 10, 20, 60):
        raw[f"return_{period}d"] = close.pct_change(period)
    volume_mean_20 = joined["Volume"].rolling(20).mean()
    volume_std_20 = joined["Volume"].rolling(20).std()
    raw["volume_ratio_20d"] = joined["Volume"] / volume_mean_20
    raw["volume_zscore_20d"] = (joined["Volume"] - volume_mean_20) / volume_std_20
    raw["volume_acceleration"] = (
        joined["Volume"].rolling(5).mean() / volume_mean_20 - 1
    )
    for period in (10, 20, 60):
        raw[f"volatility_{period}d"] = (
            returns.rolling(period).std() * math.sqrt(252)
        )
    previous_close = close.shift(1)
    true_range = pd.concat(
        (
            joined["High"] - joined["Low"],
            (joined["High"] - previous_close).abs(),
            (joined["Low"] - previous_close).abs(),
        ),
        axis=1,
    ).max(axis=1)
    raw["atr_14d_pct"] = true_range.rolling(14).mean() / close
    raw["intraday_range_pct"] = (joined["High"] - joined["Low"]) / close
    raw["overnight_gap_pct"] = joined["Open"] / previous_close - 1
    for period in (20, 50, 200):
        average = close.rolling(period).mean()
        raw[f"distance_sma{period}"] = close / average - 1
        raw[f"sma{period}_slope_5d"] = average / average.shift(5) - 1
    for period in (20, 60, 252):
        raw[f"drawdown_{period}d"] = close / close.rolling(period).max() - 1
    for period in (5, 20, 60):
        raw[f"relative_strength_{period}d"] = (
            close.pct_change(period) - benchmark_close.pct_change(period)
        )
    raw["correlation_spy_60d"] = returns.rolling(60).corr(benchmark_returns)
    raw["beta_spy_60d"] = (
        returns.rolling(60).cov(benchmark_returns)
        / benchmark_returns.rolling(60).var()
    )
    for period in (5, 20, 60):
        raw[f"spy_return_{period}d"] = benchmark_close.pct_change(period)
    raw["spy_volatility_20d"] = benchmark_returns.rolling(20).std() * math.sqrt(252)
    benchmark_sma50 = benchmark_close.rolling(50).mean()
    raw["spy_distance_sma50"] = benchmark_close / benchmark_sma50 - 1
    raw["spy_drawdown_60d"] = (
        benchmark_close / benchmark_close.rolling(60).max() - 1
    )

    # A decision during sample date may only use completed bars through the prior close.
    features = raw.shift(1).replace((np.inf, -np.inf), np.nan)
    samples: list[dict[str, Any]] = []
    for position in range(len(joined)):
        entry_position = position + 1
        exit_position = entry_position + horizon_days
        if exit_position >= len(joined):
            continue
        values = features.iloc[position]
        if values.isna().any():
            continue
        entry_price = float(joined["Open"].iloc[entry_position])
        exit_price = float(joined["Open"].iloc[exit_position])
        benchmark_entry = float(joined["benchmark_Open"].iloc[entry_position])
        benchmark_exit = float(joined["benchmark_Open"].iloc[exit_position])
        if min(entry_price, exit_price, benchmark_entry, benchmark_exit) <= 0:
            continue
        stock_return = exit_price / entry_price - 1
        benchmark_return = benchmark_exit / benchmark_entry - 1
        samples.append(
            {
                "ticker": ticker.upper(),
                "sample_date": joined.index[position].date().isoformat(),
                "label_date": joined.index[exit_position].date().isoformat(),
                "horizon_days": horizon_days,
                "feature_version": FEATURE_VERSION_V2,
                "features": {name: float(values[name]) for name in features.columns},
                "stock_return_pct": stock_return * 100,
                "benchmark_return_pct": benchmark_return * 100,
                "alpha_pct": (stock_return - benchmark_return) * 100,
                "outperformed": stock_return > benchmark_return,
            }
        )
    return samples


def _prepare_prices(data: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_PRICE_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"{name} data is missing: {', '.join(missing)}")
    frame = data.loc[:, REQUIRED_PRICE_COLUMNS].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    numeric = list(REQUIRED_PRICE_COLUMNS[1:])
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=list(REQUIRED_PRICE_COLUMNS)).sort_values("Date")
    frame = frame.drop_duplicates("Date", keep="last")
    if (frame[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError(f"{name} contains nonpositive prices")
    if (frame["Volume"] < 0).any():
        raise ValueError(f"{name} contains negative volume")
    return frame.set_index("Date")
