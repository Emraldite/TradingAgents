from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backtests.ml_shadow import FEATURE_VERSION


MODEL_VERSION = "linear-alpha-v1"


@dataclass(frozen=True)
class ChronologicalSplit:
    train: list[dict[str, Any]]
    validation: list[dict[str, Any]]
    test: list[dict[str, Any]]
    purged: list[dict[str, Any]]


def chronological_split(
    samples: list[dict[str, Any]],
    *,
    validation_start: str | date,
    test_start: str | date,
) -> ChronologicalSplit:
    """Split globally by date and purge samples whose labels cross a boundary."""
    validation_date = _as_date(validation_start)
    test_date = _as_date(test_start)
    if validation_date >= test_date:
        raise ValueError("validation_start must be before test_start")

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    purged: list[dict[str, Any]] = []
    for sample in samples:
        sample_date = _as_date(sample["sample_date"])
        label_date = _as_date(sample["label_date"])
        if label_date <= sample_date:
            raise ValueError("label_date must be after sample_date")
        if sample_date < validation_date:
            (train if label_date < validation_date else purged).append(sample)
        elif sample_date < test_date:
            (validation if label_date < test_date else purged).append(sample)
        else:
            test.append(sample)
    return ChronologicalSplit(train, validation, test, purged)


def default_split_dates(samples: list[dict[str, Any]]) -> tuple[date, date]:
    if not samples:
        raise ValueError("the ML ledger contains no samples")
    latest = max(_as_date(sample["sample_date"]) for sample in samples)
    return date(latest.year - 2, 1, 1), date(latest.year - 1, 1, 1)


def train_linear_alpha_model(
    samples: list[dict[str, Any]],
    *,
    validation_start: str | date,
    test_start: str | date,
    min_samples: int = 1_000,
    min_tickers: int = 10,
) -> dict[str, Any]:
    """Train transparent standardized logistic/ridge models with a held-out test."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        mean_absolute_error,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler

    if len(samples) < min_samples:
        raise ValueError(f"need at least {min_samples} samples; found {len(samples)}")
    tickers = {str(sample["ticker"]) for sample in samples}
    if len(tickers) < min_tickers:
        raise ValueError(f"need at least {min_tickers} tickers; found {len(tickers)}")

    feature_versions = {sample["feature_version"] for sample in samples}
    if feature_versions != {FEATURE_VERSION}:
        raise ValueError(f"expected only feature version {FEATURE_VERSION}")
    feature_names = sorted(samples[0]["features"])
    if any(sorted(sample["features"]) != feature_names for sample in samples):
        raise ValueError("samples do not share one feature schema")

    split = chronological_split(
        samples,
        validation_start=validation_start,
        test_start=test_start,
    )
    for name, values in (
        ("training", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        if not values:
            raise ValueError(f"{name} split is empty after boundary purging")

    train_x, train_y, train_alpha = _arrays(split.train, feature_names)
    validation_x, validation_y, validation_alpha = _arrays(split.validation, feature_names)
    test_x, test_y, test_alpha = _arrays(split.test, feature_names)

    evaluation_scaler = StandardScaler().fit(train_x)
    classifier = LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=0)
    regressor = Ridge(alpha=10.0)
    classifier.fit(evaluation_scaler.transform(train_x), train_y)
    regressor.fit(evaluation_scaler.transform(train_x), train_alpha)
    validation_metrics = _metrics(
        validation_y,
        validation_alpha,
        classifier.predict_proba(evaluation_scaler.transform(validation_x))[:, 1],
        regressor.predict(evaluation_scaler.transform(validation_x)),
        accuracy_score,
        brier_score_loss,
        mean_absolute_error,
        roc_auc_score,
    )

    deploy_x = np.concatenate((train_x, validation_x))
    deploy_y = np.concatenate((train_y, validation_y))
    deploy_alpha = np.concatenate((train_alpha, validation_alpha))
    deploy_scaler = StandardScaler().fit(deploy_x)
    classifier.fit(deploy_scaler.transform(deploy_x), deploy_y)
    regressor.fit(deploy_scaler.transform(deploy_x), deploy_alpha)
    test_metrics = _metrics(
        test_y,
        test_alpha,
        classifier.predict_proba(deploy_scaler.transform(test_x))[:, 1],
        regressor.predict(deploy_scaler.transform(test_x)),
        accuracy_score,
        brier_score_loss,
        mean_absolute_error,
        roc_auc_score,
    )

    return {
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "validation_start": _as_date(validation_start).isoformat(),
        "test_start": _as_date(test_start).isoformat(),
        "feature_names": feature_names,
        "sample_counts": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
            "purged": len(split.purged),
            "tickers": len(tickers),
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "scaler": {
            "mean": deploy_scaler.mean_.tolist(),
            "scale": deploy_scaler.scale_.tolist(),
        },
        "classifier": {
            "coefficients": classifier.coef_[0].tolist(),
            "intercept": float(classifier.intercept_[0]),
        },
        "alpha_regressor": {
            "coefficients": regressor.coef_.tolist(),
            "intercept": float(regressor.intercept_),
        },
        "decision_threshold": 0.55,
        "safety": "shadow-only; never consumed by the scheduler or order router",
    }


def save_model(model: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(json.dumps(model, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    return destination


def load_model(path: str | Path) -> dict[str, Any]:
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    if model.get("model_version") != MODEL_VERSION:
        raise ValueError("unsupported ML model version")
    if model.get("feature_version") != FEATURE_VERSION:
        raise ValueError("model feature version does not match this code")
    return model


def predict_feature_rows(
    model: dict[str, Any], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    names = model["feature_names"]
    mean = np.asarray(model["scaler"]["mean"], dtype=float)
    scale = np.asarray(model["scaler"]["scale"], dtype=float)
    class_coef = np.asarray(model["classifier"]["coefficients"], dtype=float)
    alpha_coef = np.asarray(model["alpha_regressor"]["coefficients"], dtype=float)
    results = []
    for row in rows:
        if sorted(row["features"]) != sorted(names):
            raise ValueError("prediction feature schema does not match the model")
        values = np.asarray([row["features"][name] for name in names], dtype=float)
        standardized = (values - mean) / scale
        logit = float(standardized @ class_coef + model["classifier"]["intercept"])
        probability = _sigmoid(logit)
        expected_alpha = float(
            standardized @ alpha_coef + model["alpha_regressor"]["intercept"]
        )
        results.append(
            {
                "ticker": row["ticker"],
                "sample_date": row["sample_date"],
                "outperform_probability": probability,
                "expected_alpha_pct": expected_alpha,
                "shadow_signal": probability >= float(model["decision_threshold"]),
            }
        )
    return results


def _arrays(
    samples: list[dict[str, Any]], feature_names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(
        [[sample["features"][name] for name in feature_names] for sample in samples],
        dtype=float,
    )
    if not np.isfinite(x).all():
        raise ValueError("feature matrix contains non-finite values")
    return (
        x,
        np.asarray([int(sample["outperformed"]) for sample in samples], dtype=int),
        np.asarray([float(sample["alpha_pct"]) for sample in samples], dtype=float),
    )


def _metrics(
    truth: np.ndarray,
    alpha: np.ndarray,
    probability: np.ndarray,
    predicted_alpha: np.ndarray,
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
) -> dict[str, float | None]:
    cutoff = float(np.quantile(probability, 0.9))
    selected = alpha[probability >= cutoff]
    auc = float(roc_auc_score(truth, probability)) if len(set(truth.tolist())) > 1 else None
    correlation = (
        float(np.corrcoef(alpha, predicted_alpha)[0, 1])
        if len(alpha) > 1 and np.std(alpha) > 0 and np.std(predicted_alpha) > 0
        else None
    )
    return {
        "roc_auc": auc,
        "accuracy_at_0_5": float(accuracy_score(truth, probability >= 0.5)),
        "brier_score": float(brier_score_loss(truth, probability)),
        "alpha_mae_pct": float(mean_absolute_error(alpha, predicted_alpha)),
        "alpha_correlation": correlation,
        "baseline_mean_alpha_pct": float(np.mean(alpha)),
        "top_decile_mean_alpha_pct": float(np.mean(selected)),
        "top_decile_samples": int(len(selected)),
    }


def _as_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)
