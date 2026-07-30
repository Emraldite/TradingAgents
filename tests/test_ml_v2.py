import pandas as pd

from backtests.ml_v2 import (
    FEATURE_VERSION_V2,
    HistoricalPriceCache,
    build_v2_samples,
)


def _prices(rows=400, multiplier=1.0):
    index = pd.Series(range(rows), dtype="float64")
    close = multiplier * (100 + index * 0.2 + (index % 7) * 0.03)
    open_price = close * (1 + ((index % 5) - 2) / 10_000)
    return pd.DataFrame(
        {
            "Date": pd.bdate_range("2020-01-01", periods=rows),
            "Open": open_price,
            "High": pd.concat((open_price, close), axis=1).max(axis=1) + 1,
            "Low": pd.concat((open_price, close), axis=1).min(axis=1) - 1,
            "Close": close,
            "Volume": 1_000_000 + index * 1_000,
        }
    )


def test_v2_uses_prior_close_features_and_next_open_labels():
    stock = _prices()
    benchmark = _prices(multiplier=0.8)

    samples = build_v2_samples(
        stock, benchmark, ticker="abc", horizon_days=5
    )

    assert samples
    first = samples[0]
    assert first["feature_version"] == FEATURE_VERSION_V2
    position = stock.index[stock["Date"] == pd.Timestamp(first["sample_date"])][0]
    expected_return = (
        stock.loc[position + 6, "Open"] / stock.loc[position + 1, "Open"] - 1
    ) * 100
    assert first["stock_return_pct"] == expected_return
    assert first["label_date"] == stock.loc[position + 6, "Date"].date().isoformat()
    assert {
        "return_1d",
        "overnight_gap_pct",
        "atr_14d_pct",
        "drawdown_252d",
        "spy_volatility_20d",
        "beta_spy_60d",
    } <= set(first["features"])


def test_future_mutation_does_not_change_earlier_v2_features():
    stock = _prices()
    benchmark = _prices(multiplier=0.8)
    original = build_v2_samples(stock, benchmark, ticker="ABC", horizon_days=5)
    stock.loc[len(stock) - 1, ["Close", "Volume"]] = [10_000, 99_000_000]
    changed = build_v2_samples(stock, benchmark, ticker="ABC", horizon_days=5)

    assert original[0]["features"] == changed[0]["features"]
    assert original[0]["alpha_pct"] == changed[0]["alpha_pct"]


def test_historical_price_cache_persists_rows_and_attempt_coverage(tmp_path):
    cache = HistoricalPriceCache(tmp_path / "prices.db")
    data = _prices(rows=20)

    assert (
        cache.record_download(
            "ABC",
            data,
            requested_start="2020-01-01",
            requested_end="2021-01-01",
        )
        == 20
    )
    loaded = cache.load("ABC")
    attempt = cache.reusable_attempt("ABC", "2020-02-01", "2020-12-01")

    assert len(loaded) == 20
    assert attempt is not None
    assert attempt["status"] == "success"
    assert cache.summary()["tickers"] == 1


def test_cache_remembers_unavailable_ticker_without_fake_prices(tmp_path):
    cache = HistoricalPriceCache(tmp_path / "prices.db")
    cache.record_download(
        "OLD",
        pd.DataFrame(),
        requested_start="2010-01-01",
        requested_end="2026-01-01",
        status="unavailable",
        error="no history",
    )

    assert cache.load("OLD").empty
    assert cache.reusable_attempt("OLD", "2011-01-01", "2025-01-01")["status"] == (
        "unavailable"
    )


def test_cache_retries_transient_download_errors(tmp_path):
    cache = HistoricalPriceCache(tmp_path / "prices.db")
    cache.record_download(
        "ABC",
        pd.DataFrame(),
        requested_start="2010-01-01",
        requested_end="2026-01-01",
        status="download_error",
        error="temporary rate limit",
    )

    assert cache.reusable_attempt("ABC", "2011-01-01", "2025-01-01") is None
