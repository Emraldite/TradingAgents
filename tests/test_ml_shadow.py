import pandas as pd

from backtests.ml_shadow import MLShadowLedger, build_ml_feature_rows, build_ml_samples


def _prices(multiplier: float = 1.0, rows: int = 100) -> pd.DataFrame:
    close = pd.Series([multiplier * (100 + index * 0.25) for index in range(rows)])
    return pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=rows, freq="D"),
            "Open": close - 0.1,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": [1_000_000 + index * 1_000 for index in range(rows)],
        }
    )


def test_ml_samples_use_prior_data_and_have_forward_labels():
    stock = _prices()
    samples = build_ml_samples(stock, _prices(0.8), ticker="abc", horizon_days=5)

    assert samples
    first = samples[0]
    assert first["ticker"] == "ABC"
    assert first["sample_date"] < first["label_date"]
    assert set(first["features"]) == {
        "return_5d",
        "return_20d",
        "return_60d",
        "volume_ratio_20d",
        "volatility_20d",
        "distance_sma20",
        "relative_strength_20d",
    }


def test_future_price_change_does_not_change_earlier_features():
    stock = _prices()
    original = build_ml_samples(stock, _prices(0.8), ticker="ABC", horizon_days=5)
    stock.loc[len(stock) - 1, "Close"] = 10_000
    changed = build_ml_samples(stock, _prices(0.8), ticker="ABC", horizon_days=5)

    assert original[0]["features"] == changed[0]["features"]


def test_feature_rows_include_dates_that_do_not_have_future_labels_yet():
    stock = _prices()
    features = build_ml_feature_rows(stock, _prices(0.8), ticker="ABC")
    samples = build_ml_samples(stock, _prices(0.8), ticker="ABC", horizon_days=5)

    assert features[-1]["sample_date"] > samples[-1]["sample_date"]
    assert features[-1]["feature_version"] == samples[-1]["feature_version"]


def test_shadow_ledger_is_idempotent(tmp_path):
    samples = build_ml_samples(_prices(), _prices(0.8), ticker="ABC", horizon_days=5)
    ledger = MLShadowLedger(tmp_path / "shadow.db")

    assert ledger.record(samples) == len(samples)
    assert ledger.record(samples) == 0
    assert ledger.summary()["samples"] == len(samples)
    assert len(ledger.load_samples(horizon_days=5)) == len(samples)


def test_shadow_ledger_records_dataset_provenance(tmp_path):
    ledger = MLShadowLedger(tmp_path / "shadow.db")
    build_id = ledger.record_build(
        {
            "universe": "sp500-point-in-time",
            "source": "fixture.csv",
            "source_sha256": "abc",
            "start_date": "2020-01-01",
            "end_date": "2025-01-01",
            "horizon_days": 10,
            "attempted": 2,
            "succeeded": 1,
            "failed": 1,
            "inserted": 25,
            "failure_details": ["OLD: unavailable"],
        }
    )

    assert build_id == 1
