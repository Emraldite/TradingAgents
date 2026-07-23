import math
from datetime import date, timedelta

from backtests.ml_model import (
    chronological_split,
    load_model,
    predict_feature_rows,
    save_model,
    train_linear_alpha_model,
)
from backtests.ml_shadow import FEATURE_VERSION


def _sample(ticker, sample_date, label_date, value):
    return {
        "ticker": ticker,
        "sample_date": sample_date.isoformat(),
        "label_date": label_date.isoformat(),
        "horizon_days": 10,
        "feature_version": FEATURE_VERSION,
        "features": {"momentum": value, "volume": abs(value) + 0.1},
        "alpha_pct": value * 2,
        "outperformed": value > 0,
    }


def test_chronological_split_purges_actual_label_boundary_crossings():
    samples = [
        _sample("A", date(2021, 12, 20), date(2021, 12, 30), -1),
        _sample("A", date(2021, 12, 25), date(2022, 1, 3), 1),
        _sample("A", date(2022, 12, 20), date(2023, 1, 1), 1),
        _sample("A", date(2023, 1, 2), date(2023, 1, 12), -1),
    ]

    split = chronological_split(
        list(reversed(samples)),
        validation_start="2022-01-01",
        test_start="2023-01-01",
    )

    assert split.train == [samples[0]]
    assert split.validation == []
    assert split.test == [samples[3]]
    assert {item["sample_date"] for item in split.purged} == {
        "2021-12-25",
        "2022-12-20",
    }


def test_train_save_load_and_predict_transparent_json_model(tmp_path):
    samples = []
    start = date(2020, 1, 1)
    for index in range(1_500):
        sample_date = start + timedelta(days=index)
        value = math.sin(index / 13)
        for ticker in ("A", "B", "C"):
            samples.append(
                _sample(ticker, sample_date, sample_date + timedelta(days=10), value)
            )

    model = train_linear_alpha_model(
        samples,
        validation_start="2022-01-01",
        test_start="2023-01-01",
        min_samples=100,
        min_tickers=3,
    )
    path = save_model(model, tmp_path / "model.json")
    loaded = load_model(path)
    prediction = predict_feature_rows(
        loaded,
        [
            {
                "ticker": "A",
                "sample_date": "2024-02-01",
                "features": {"momentum": 0.8, "volume": 0.9},
            }
        ],
    )[0]

    assert model["sample_counts"]["purged"] > 0
    assert model["test_metrics"]["roc_auc"] > 0.9
    assert prediction["outperform_probability"] > 0.5
    assert prediction["expected_alpha_pct"] > 0
