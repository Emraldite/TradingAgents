from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")


def audit_market_data(data: pd.DataFrame) -> dict[str, Any]:
    """Check OHLCV data for errors that can invalidate a backtest or ML dataset."""
    issues: list[dict[str, Any]] = []
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        issues.append(
            {
                "severity": "error",
                "code": "missing_columns",
                "count": len(missing_columns),
                "message": f"Missing required columns: {', '.join(missing_columns)}",
            }
        )
        return _result(data, issues)

    frame = data.loc[:, REQUIRED_COLUMNS].copy()
    parsed_dates = pd.to_datetime(frame["Date"], errors="coerce")
    bad_dates = int(parsed_dates.isna().sum())
    if bad_dates:
        issues.append(_issue("error", "invalid_dates", bad_dates, "Rows have invalid dates."))

    valid_dates = parsed_dates.dropna()
    if not valid_dates.is_monotonic_increasing:
        issues.append(_issue("error", "unsorted_dates", 1, "Dates are not ascending."))

    duplicates = int(parsed_dates.duplicated(keep=False).sum())
    if duplicates:
        issues.append(
            _issue("error", "duplicate_dates", duplicates, "Rows share the same market date.")
        )

    numeric_columns = ("Open", "High", "Low", "Close", "Volume")
    numeric = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    missing_values = int(numeric.isna().sum().sum())
    if missing_values:
        issues.append(
            _issue("error", "missing_values", missing_values, "OHLCV values are missing or invalid.")
        )

    prices = numeric.loc[:, ("Open", "High", "Low", "Close")]
    nonpositive_prices = int((prices <= 0).sum().sum())
    if nonpositive_prices:
        issues.append(
            _issue("error", "nonpositive_prices", nonpositive_prices, "Prices must be positive.")
        )

    negative_volume = int((numeric["Volume"] < 0).sum())
    if negative_volume:
        issues.append(
            _issue("error", "negative_volume", negative_volume, "Volume cannot be negative.")
        )
    zero_volume = int((numeric["Volume"] == 0).sum())
    if zero_volume:
        issues.append(
            _issue("warning", "zero_volume", zero_volume, "Rows have zero trading volume.")
        )

    row_max = prices.max(axis=1)
    row_min = prices.min(axis=1)
    invalid_range = int(((numeric["High"] < row_max) | (numeric["Low"] > row_min)).sum())
    if invalid_range:
        issues.append(
            _issue(
                "error",
                "invalid_ohlc_range",
                invalid_range,
                "High/low does not contain the row's open and close.",
            )
        )

    return _result(frame.assign(Date=parsed_dates), issues)


def audit_feature_lookahead(
    data: pd.DataFrame,
    feature_builder: Callable[[pd.DataFrame], pd.DataFrame],
) -> dict[str, Any]:
    """Detect future-dependent features by comparing full and truncated calculations."""
    if len(data) < 4:
        return {
            "ok": False,
            "checks": 0,
            "issues": [
                _issue("error", "too_few_rows", len(data), "At least four rows are required.")
            ],
        }

    full = feature_builder(data.copy())
    source_columns = set(data.columns)
    feature_columns = [
        column for column in full.columns if column not in source_columns and column != "Date"
    ]
    if not feature_columns:
        return {
            "ok": False,
            "checks": 0,
            "issues": [
                _issue("error", "no_features", 0, "The builder did not add feature columns.")
            ],
        }

    cutoffs = sorted({max(2, len(data) // 4), len(data) // 2, (len(data) * 3) // 4, len(data) - 1})
    changed: set[str] = set()
    checks = 0
    for cutoff in cutoffs:
        prefix = feature_builder(data.iloc[:cutoff].copy())
        comparable_rows = min(len(prefix), len(full), cutoff)
        if comparable_rows == 0:
            continue
        for column in feature_columns:
            checks += 1
            left = prefix[column].iloc[:comparable_rows].reset_index(drop=True)
            right = full[column].iloc[:comparable_rows].reset_index(drop=True)
            if not left.equals(right):
                changed.add(str(column))

    issues = []
    if changed:
        issues.append(
            _issue(
                "error",
                "future_dependent_features",
                len(changed),
                f"Features changed when future rows were removed: {', '.join(sorted(changed))}",
            )
        )
    return {"ok": not issues, "checks": checks, "issues": issues}


def _issue(severity: str, code: str, count: int, message: str) -> dict[str, Any]:
    return {"severity": severity, "code": code, "count": count, "message": message}


def _result(data: pd.DataFrame, issues: list[dict[str, Any]]) -> dict[str, Any]:
    errors = sum(1 for issue in issues if issue["severity"] == "error")
    warnings = sum(1 for issue in issues if issue["severity"] == "warning")
    dates = pd.to_datetime(data["Date"], errors="coerce").dropna() if "Date" in data else []
    return {
        "ok": errors == 0,
        "rows": len(data),
        "start": dates.min().date().isoformat() if len(dates) else None,
        "end": dates.max().date().isoformat() if len(dates) else None,
        "errors": errors,
        "warnings": warnings,
        "issues": issues,
    }
