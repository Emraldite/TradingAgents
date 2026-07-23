from datetime import date, timedelta

import numpy as np
import pandas as pd

from backtests.ml_lightgbm import (
    add_cross_sectional_features,
    evaluate_ranked_predictions,
    load_ml_frame,
    save_lightgbm_artifacts,
    split_frame_chronologically,
    train_lightgbm_alpha_model,
)
from backtests.ml_shadow import FEATURE_VERSION, MLShadowLedger


def _row(ticker, sample_date, label_date, momentum, volume, alpha):
    return {
        "ticker": ticker,
        "sample_date": pd.Timestamp(sample_date),
        "label_date": pd.Timestamp(label_date),
        "momentum": momentum,
        "volume": volume,
        "alpha_pct": alpha,
        "outperformed": int(alpha > 0),
    }


def test_sql_loader_extracts_json_features_without_object_rows(tmp_path):
    ledger = MLShadowLedger(tmp_path / "shadow.db")
    samples = []
    for index in range(3):
        sample_date = date(2024, 1, 2) + timedelta(days=index)
        samples.append(
            {
                "ticker": f"T{index}",
                "sample_date": sample_date.isoformat(),
                "label_date": (sample_date + timedelta(days=10)).isoformat(),
                "horizon_days": 10,
                "feature_version": FEATURE_VERSION,
                "features": {"momentum": index / 10, "volume": index + 1},
                "stock_return_pct": index,
                "benchmark_return_pct": 0,
                "alpha_pct": index,
                "outperformed": index > 0,
            }
        )
    ledger.record(samples)

    frame, features = load_ml_frame(tmp_path / "shadow.db")

    assert features == ["momentum", "volume"]
    assert len(frame) == 3
    assert frame["momentum"].tolist() == [0.0, 0.1, 0.2]


def test_cross_sectional_ranks_use_only_same_date_and_drop_small_dates():
    rows = []
    for ticker, value in (("A", 1.0), ("B", 2.0), ("C", 3.0)):
        rows.append(_row(ticker, "2024-01-02", "2024-01-12", value, value, value))
    rows.append(_row("A", "2024-01-03", "2024-01-13", 100.0, 100.0, 1.0))

    ranked, features, dropped = add_cross_sectional_features(
        pd.DataFrame(rows), ["momentum", "volume"], min_cross_section=3
    )

    assert dropped == 1
    assert "rank_momentum" in features
    assert ranked.set_index("ticker")["rank_momentum"].to_dict() == {
        "A": 1 / 3,
        "B": 2 / 3,
        "C": 1.0,
    }


def test_dataframe_split_purges_labels_crossing_boundaries():
    frame = pd.DataFrame(
        [
            _row("A", "2021-12-20", "2021-12-30", 0, 1, 0),
            _row("A", "2021-12-25", "2022-01-03", 0, 1, 0),
            _row("A", "2022-06-01", "2022-06-15", 0, 1, 0),
            _row("A", "2022-12-20", "2023-01-01", 0, 1, 0),
            _row("A", "2023-01-02", "2023-01-12", 0, 1, 0),
        ]
    )

    split = split_frame_chronologically(
        frame, validation_start="2022-01-01", test_start="2023-01-01"
    )

    assert len(split["train"]) == 1
    assert len(split["validation"]) == 1
    assert len(split["test"]) == 1
    assert len(split["purged"]) == 2


def test_per_date_quintiles_and_cost_adjustment_are_exact():
    rows = []
    predictions = []
    for day in ("2024-01-02", "2024-01-03"):
        for index in range(10):
            rows.append(
                _row(
                    f"T{index}",
                    day,
                    pd.Timestamp(day) + pd.Timedelta(days=10),
                    index,
                    1,
                    index,
                )
            )
            predictions.append(index)

    metrics = evaluate_ranked_predictions(
        pd.DataFrame(rows),
        np.asarray(predictions),
        round_trip_cost_bps=40,
        rebalance_every=1,
    )

    assert metrics["top_decile_samples"] == 2
    assert metrics["top_decile_mean_alpha_pct"] == 9.0
    assert metrics["top_decile_cost_adjusted_alpha_pct"] == 8.6
    assert metrics["bottom_decile_mean_alpha_pct"] == 0.0
    assert metrics["top_bottom_spread_pct"] == 9.0
    assert [item["samples"] for item in metrics["quintiles"]] == [4, 4, 4, 4, 4]


def test_lightgbm_beats_fair_linear_baseline_on_nonlinear_fixture(tmp_path):
    rng = np.random.default_rng(7)
    rows = []
    dates = pd.bdate_range("2020-01-01", periods=520)
    for day_index, sample_date in enumerate(dates):
        for ticker_index in range(30):
            position = ticker_index / 29
            momentum = position + 0.1 * np.sin(day_index / 17)
            volume = 1.0 + 0.2 * np.cos(day_index / 11 + ticker_index)
            alpha = ((position - 0.5) ** 2 - 0.08) * 16 + rng.normal(0, 0.15)
            rows.append(
                _row(
                    f"T{ticker_index:02d}",
                    sample_date,
                    sample_date + pd.Timedelta(days=14),
                    momentum,
                    volume,
                    alpha,
                )
            )
    raw = pd.DataFrame(rows)
    frame, features, _ = add_cross_sectional_features(
        raw, ["momentum", "volume"], min_cross_section=30
    )

    model, report = train_lightgbm_alpha_model(
        frame,
        features,
        validation_start="2021-01-01",
        test_start="2021-07-01",
        min_samples=1_000,
        min_tickers=20,
        n_jobs=1,
    )
    model_path, report_path = save_lightgbm_artifacts(
        model,
        report,
        model_path=tmp_path / "model.txt",
        report_path=tmp_path / "report.json",
    )

    lightgbm_test = report["test_metrics"]
    linear_test = report["fair_linear_baseline"]["test_metrics"]
    assert lightgbm_test["top_bottom_spread_pct"] > 1
    assert (
        lightgbm_test["top_decile_mean_alpha_pct"]
        > linear_test["top_decile_mean_alpha_pct"]
    )
    assert model_path.exists()
    assert report_path.exists()
