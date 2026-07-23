from __future__ import annotations

import json
import math
import re
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtests.ml_shadow import FEATURE_VERSION


MODEL_VERSION = "cross-sectional-lightgbm-v1"
_SAFE_FEATURE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_ml_frame(
    database: str | Path,
    *,
    horizon_days: int = 10,
    feature_version: str = FEATURE_VERSION,
) -> tuple[pd.DataFrame, list[str]]:
    """Load numeric features directly through SQLite JSON1 without Python row dicts."""
    path = Path(database)
    if not path.exists():
        raise ValueError(f"ML database does not exist: {path}")
    with sqlite3.connect(path) as conn:
        schema_row = conn.execute(
            """
            SELECT features_json
            FROM ml_shadow_samples
            WHERE horizon_days=? AND feature_version=?
            LIMIT 1
            """,
            (horizon_days, feature_version),
        ).fetchone()
        if schema_row is None:
            raise ValueError("no matching ML samples were found")
        feature_names = sorted(json.loads(schema_row[0]))
        if not feature_names or any(not _SAFE_FEATURE.match(name) for name in feature_names):
            raise ValueError("feature schema contains an unsafe or invalid name")

        feature_sql = ", ".join(
            f"CAST(json_extract(features_json, '$.{name}') AS REAL) AS \"{name}\""
            for name in feature_names
        )
        query = f"""
            SELECT ticker, sample_date, label_date, alpha_pct, outperformed,
                   {feature_sql}
            FROM ml_shadow_samples
            WHERE horizon_days=? AND feature_version=?
            ORDER BY sample_date, ticker
        """
        frame = pd.read_sql_query(query, conn, params=(horizon_days, feature_version))

    frame["sample_date"] = pd.to_datetime(frame["sample_date"], errors="raise")
    frame["label_date"] = pd.to_datetime(frame["label_date"], errors="raise")
    numeric = [*feature_names, "alpha_pct", "outperformed"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if frame[numeric].isna().any().any():
        raise ValueError("ML dataset contains missing or invalid numeric values")
    if (frame["label_date"] <= frame["sample_date"]).any():
        raise ValueError("ML dataset contains labels that do not follow sample dates")
    return frame, feature_names


def add_cross_sectional_features(
    frame: pd.DataFrame,
    raw_feature_names: list[str],
    *,
    min_cross_section: int = 20,
) -> tuple[pd.DataFrame, list[str], int]:
    """Add percentile ranks computed only among stocks on the same decision date."""
    if min_cross_section < 2:
        raise ValueError("min_cross_section must be at least 2")
    counts = frame.groupby("sample_date")["ticker"].transform("size")
    eligible = frame.loc[counts >= min_cross_section].copy()
    dropped = int(len(frame) - len(eligible))
    if eligible.empty:
        raise ValueError("no dates contain the minimum cross-section size")

    rank_names: list[str] = []
    grouped = eligible.groupby("sample_date", sort=False)
    for name in raw_feature_names:
        rank_name = f"rank_{name}"
        eligible[rank_name] = grouped[name].rank(method="average", pct=True)
        rank_names.append(rank_name)

    if {"return_5d", "return_20d"} <= set(raw_feature_names):
        eligible["momentum_acceleration"] = (
            eligible["return_5d"] - eligible["return_20d"] / 4.0
        )
        eligible["rank_momentum_acceleration"] = eligible.groupby(
            "sample_date", sort=False
        )["momentum_acceleration"].rank(method="average", pct=True)
        raw_feature_names = [*raw_feature_names, "momentum_acceleration"]
        rank_names.append("rank_momentum_acceleration")

    model_features = [*raw_feature_names, *rank_names]
    if not np.isfinite(eligible[model_features].to_numpy(dtype=float)).all():
        raise ValueError("cross-sectional feature matrix contains non-finite values")
    return eligible, model_features, dropped


def split_frame_chronologically(
    frame: pd.DataFrame,
    *,
    validation_start: str | date,
    test_start: str | date,
) -> dict[str, pd.DataFrame]:
    """Split globally by date and purge any future label crossing a boundary."""
    validation_date = pd.Timestamp(validation_start)
    test_date = pd.Timestamp(test_start)
    if validation_date >= test_date:
        raise ValueError("validation_start must be before test_start")

    sample = frame["sample_date"]
    label = frame["label_date"]
    train_mask = (sample < validation_date) & (label < validation_date)
    validation_mask = (
        (sample >= validation_date) & (sample < test_date) & (label < test_date)
    )
    test_mask = sample >= test_date
    assigned = train_mask | validation_mask | test_mask
    result = {
        "train": frame.loc[train_mask].copy(),
        "validation": frame.loc[validation_mask].copy(),
        "test": frame.loc[test_mask].copy(),
        "purged": frame.loc[~assigned].copy(),
    }
    for name in ("train", "validation", "test"):
        if result[name].empty:
            raise ValueError(f"{name} split is empty after boundary purging")
    return result


def default_frame_split_dates(frame: pd.DataFrame) -> tuple[date, date]:
    if frame.empty:
        raise ValueError("the ML frame is empty")
    latest = frame["sample_date"].max().date()
    return date(latest.year - 2, 1, 1), date(latest.year - 1, 1, 1)


def train_lightgbm_alpha_model(
    frame: pd.DataFrame,
    feature_names: list[str],
    *,
    validation_start: str | date,
    test_start: str | date,
    min_samples: int = 10_000,
    min_tickers: int = 30,
    round_trip_cost_bps: int = 40,
    n_jobs: int = 2,
    rebalance_every: int = 10,
) -> tuple[Any, dict[str, Any]]:
    """Train a nonlinear alpha regressor and evaluate date-relative rankings."""
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error, roc_auc_score

    if len(frame) < min_samples:
        raise ValueError(f"need at least {min_samples} samples; found {len(frame)}")
    ticker_count = int(frame["ticker"].nunique())
    if ticker_count < min_tickers:
        raise ValueError(f"need at least {min_tickers} tickers; found {ticker_count}")
    splits = split_frame_chronologically(
        frame,
        validation_start=validation_start,
        test_start=test_start,
    )

    params = {
        "objective": "regression_l1",
        "n_estimators": 1_500,
        "learning_rate": 0.025,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "random_state": 0,
        "n_jobs": max(1, n_jobs),
        "verbosity": -1,
    }
    train = splits["train"]
    validation = splits["validation"]
    train_weight = _equal_date_weights(train)
    validation_weight = _equal_date_weights(validation)
    evaluation_model = lgb.LGBMRegressor(**params)
    evaluation_model.fit(
        train[feature_names],
        train["alpha_pct"],
        sample_weight=train_weight,
        eval_X=validation[feature_names],
        eval_y=validation["alpha_pct"],
        eval_sample_weight=[validation_weight],
        eval_metric="l1",
        callbacks=[
            lgb.early_stopping(stopping_rounds=75, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iteration = int(evaluation_model.best_iteration_ or params["n_estimators"])
    validation_prediction = evaluation_model.predict(
        validation[feature_names], num_iteration=best_iteration
    )
    validation_metrics = evaluate_ranked_predictions(
        validation,
        validation_prediction,
        round_trip_cost_bps=round_trip_cost_bps,
        rebalance_every=rebalance_every,
        roc_auc_score=roc_auc_score,
        mean_absolute_error=mean_absolute_error,
    )

    deploy = pd.concat((train, validation), ignore_index=True)
    deploy_weight = _equal_date_weights(deploy)
    deploy_params = {**params, "n_estimators": best_iteration}
    model = lgb.LGBMRegressor(**deploy_params)
    model.fit(
        deploy[feature_names],
        deploy["alpha_pct"],
        sample_weight=deploy_weight,
    )
    test = splits["test"]
    test_prediction = model.predict(test[feature_names])
    test_metrics = evaluate_ranked_predictions(
        test,
        test_prediction,
        round_trip_cost_bps=round_trip_cost_bps,
        rebalance_every=rebalance_every,
        roc_auc_score=roc_auc_score,
        mean_absolute_error=mean_absolute_error,
    )

    importance = sorted(
        (
            {
                "feature": name,
                "gain": float(gain),
                "splits": int(split_count),
            }
            for name, gain, split_count in zip(
                feature_names,
                model.booster_.feature_importance(importance_type="gain"),
                model.booster_.feature_importance(importance_type="split"),
                strict=True,
            )
        ),
        key=lambda item: item["gain"],
        reverse=True,
    )
    total_gain = sum(item["gain"] for item in importance) or 1.0
    for item in importance:
        item["gain_pct"] = item["gain"] / total_gain * 100.0

    report = {
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "validation_start": pd.Timestamp(validation_start).date().isoformat(),
        "test_start": pd.Timestamp(test_start).date().isoformat(),
        "feature_names": feature_names,
        "best_iteration": best_iteration,
        "round_trip_cost_bps": round_trip_cost_bps,
        "rebalance_every_trading_dates": rebalance_every,
        "holdout_status": (
            "development holdout; prior linear results on these dates were inspected"
        ),
        "timing_limit": (
            "price-volume-v1 labels assume sample-date open entry; the live 08:45 CT "
            "cycle starts after that price. Research-only until execution-aligned labels."
        ),
        "sample_counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "purged": len(splits["purged"]),
            "tickers": ticker_count,
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "fair_linear_baseline": train_fair_linear_baseline(
            splits,
            feature_names,
            round_trip_cost_bps=round_trip_cost_bps,
            rebalance_every=rebalance_every,
        ),
        "feature_importance": importance,
        "safety": "shadow-only; never consumed by the scheduler or order router",
    }
    return model, report


def evaluate_ranked_predictions(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    round_trip_cost_bps: int,
    rebalance_every: int = 10,
    roc_auc_score=None,
    mean_absolute_error=None,
) -> dict[str, Any]:
    """Evaluate cross-sectional scores by decision-date deciles and calendar year."""
    if len(frame) != len(prediction):
        raise ValueError("prediction length does not match evaluation frame")
    if rebalance_every < 1:
        raise ValueError("rebalance_every must be at least 1")
    evaluated = frame.loc[
        :, ["ticker", "sample_date", "alpha_pct", "outperformed"]
    ].copy()
    evaluated["prediction"] = np.asarray(prediction, dtype=float)
    if not np.isfinite(evaluated[["alpha_pct", "prediction"]].to_numpy()).all():
        raise ValueError("evaluation contains non-finite values")

    evaluated["prediction_pct"] = evaluated.groupby("sample_date")["prediction"].rank(
        method="first", pct=True
    )
    evaluated["quintile"] = np.ceil(evaluated["prediction_pct"] * 5).clip(1, 5).astype(int)
    evaluated["decile"] = np.ceil(evaluated["prediction_pct"] * 10).clip(1, 10).astype(int)
    cost_pct = round_trip_cost_bps / 100.0
    daily_ic = evaluated.groupby("sample_date").apply(
        lambda group: group["prediction"].corr(group["alpha_pct"], method="spearman"),
        include_groups=False,
    ).dropna()
    top = evaluated[evaluated["decile"] == 10]
    bottom = evaluated[evaluated["decile"] == 1]
    quintiles = [
        {
            "quintile": int(quintile),
            "samples": int(len(group)),
            "mean_alpha_pct": float(group["alpha_pct"].mean()),
            "cost_adjusted_alpha_pct": float(group["alpha_pct"].mean() - cost_pct),
        }
        for quintile, group in evaluated.groupby("quintile", sort=True)
    ]
    unique_dates = sorted(evaluated["sample_date"].unique())
    rebalance_dates = set(unique_dates[::rebalance_every])
    rebalanced = evaluated[evaluated["sample_date"].isin(rebalance_dates)]
    rebalanced_top = rebalanced[rebalanced["decile"] == 10]
    turnover_values: list[float] = []
    prior_names: set[str] | None = None
    for _, cohort in rebalanced_top.groupby("sample_date", sort=True):
        names = set(cohort["ticker"].astype(str))
        if prior_names is not None and prior_names:
            turnover_values.append(1.0 - len(names & prior_names) / len(prior_names))
        prior_names = names

    yearly = []
    evaluated["year"] = evaluated["sample_date"].dt.year
    for year, group in evaluated.groupby("year", sort=True):
        year_top = group[group["decile"] == 10]
        year_bottom = group[group["decile"] == 1]
        year_ic = group.groupby("sample_date").apply(
            lambda daily: daily["prediction"].corr(daily["alpha_pct"], method="spearman"),
            include_groups=False,
        ).dropna()
        yearly.append(
            {
                "year": int(year),
                "samples": int(len(group)),
                "dates": int(group["sample_date"].nunique()),
                "average_names_per_date": float(
                    group.groupby("sample_date")["ticker"].size().mean()
                ),
                "baseline_alpha_pct": float(group["alpha_pct"].mean()),
                "top_decile_alpha_pct": float(year_top["alpha_pct"].mean()),
                "bottom_decile_alpha_pct": float(year_bottom["alpha_pct"].mean()),
                "top_bottom_spread_pct": float(
                    year_top["alpha_pct"].mean() - year_bottom["alpha_pct"].mean()
                ),
                "rank_ic": float(year_ic.mean()) if len(year_ic) else None,
            }
        )

    truth = evaluated["outperformed"].astype(int).to_numpy()
    auc = None
    if roc_auc_score is not None and len(np.unique(truth)) > 1:
        auc = float(roc_auc_score(truth, evaluated["prediction"]))
    mae = (
        float(mean_absolute_error(evaluated["alpha_pct"], evaluated["prediction"]))
        if mean_absolute_error is not None
        else float(np.mean(np.abs(evaluated["alpha_pct"] - evaluated["prediction"])))
    )
    return {
        "roc_auc": auc,
        "alpha_mae_pct": mae,
        "alpha_correlation": _safe_correlation(
            evaluated["alpha_pct"].to_numpy(), evaluated["prediction"].to_numpy()
        ),
        "mean_daily_rank_ic": float(daily_ic.mean()) if len(daily_ic) else None,
        "positive_rank_ic_days_pct": float((daily_ic > 0).mean() * 100) if len(daily_ic) else None,
        "baseline_mean_alpha_pct": float(evaluated["alpha_pct"].mean()),
        "top_decile_mean_alpha_pct": float(top["alpha_pct"].mean()),
        "top_decile_cost_adjusted_alpha_pct": float(top["alpha_pct"].mean() - cost_pct),
        "bottom_decile_mean_alpha_pct": float(bottom["alpha_pct"].mean()),
        "top_bottom_spread_pct": float(
            top["alpha_pct"].mean() - bottom["alpha_pct"].mean()
        ),
        "top_decile_samples": int(len(top)),
        "non_overlapping_top_decile_alpha_pct": float(
            rebalanced_top["alpha_pct"].mean()
        ),
        "non_overlapping_top_decile_cost_adjusted_alpha_pct": float(
            rebalanced_top["alpha_pct"].mean() - cost_pct
        ),
        "non_overlapping_rebalance_dates": int(len(rebalance_dates)),
        "average_top_decile_turnover_pct": (
            float(np.mean(turnover_values) * 100) if turnover_values else None
        ),
        "quintiles": quintiles,
        "yearly": yearly,
    }


def train_fair_linear_baseline(
    splits: dict[str, pd.DataFrame],
    feature_names: list[str],
    *,
    round_trip_cost_bps: int,
    rebalance_every: int,
) -> dict[str, Any]:
    """Fit Ridge on identical rows/features and evaluate with identical date buckets."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    train = splits["train"]
    validation = splits["validation"]
    train_scaler = StandardScaler().fit(
        train[feature_names],
        sample_weight=_equal_date_weights(train),
    )
    validation_model = Ridge(alpha=10.0)
    validation_model.fit(
        train_scaler.transform(train[feature_names]),
        train["alpha_pct"],
        sample_weight=_equal_date_weights(train),
    )
    validation_prediction = validation_model.predict(
        train_scaler.transform(validation[feature_names])
    )
    validation_metrics = evaluate_ranked_predictions(
        validation,
        validation_prediction,
        round_trip_cost_bps=round_trip_cost_bps,
        rebalance_every=rebalance_every,
        roc_auc_score=roc_auc_score,
        mean_absolute_error=mean_absolute_error,
    )

    deploy = pd.concat((train, validation), ignore_index=True)
    deploy_weight = _equal_date_weights(deploy)
    deploy_scaler = StandardScaler().fit(
        deploy[feature_names],
        sample_weight=deploy_weight,
    )
    model = Ridge(alpha=10.0)
    model.fit(
        deploy_scaler.transform(deploy[feature_names]),
        deploy["alpha_pct"],
        sample_weight=deploy_weight,
    )
    test = splits["test"]
    test_prediction = model.predict(deploy_scaler.transform(test[feature_names]))
    test_metrics = evaluate_ranked_predictions(
        test,
        test_prediction,
        round_trip_cost_bps=round_trip_cost_bps,
        rebalance_every=rebalance_every,
        roc_auc_score=roc_auc_score,
        mean_absolute_error=mean_absolute_error,
    )
    return {
        "model_version": "fair-cross-sectional-ridge-v1",
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }


def compare_linear_baseline(
    report: dict[str, Any],
    baseline_model_path: str | Path,
) -> dict[str, Any] | None:
    path = Path(baseline_model_path)
    if not path.exists():
        return None
    baseline = json.loads(path.read_text(encoding="utf-8"))
    if (
        baseline.get("model_version") != "linear-alpha-v1"
        or baseline.get("validation_start") != report["validation_start"]
        or baseline.get("test_start") != report["test_start"]
    ):
        return None
    baseline_test = baseline.get("test_metrics", {})
    lightgbm_test = report["test_metrics"]
    return {
        "linear_roc_auc": baseline_test.get("roc_auc"),
        "lightgbm_roc_auc": lightgbm_test.get("roc_auc"),
        "linear_top_decile_alpha_pct": baseline_test.get("top_decile_mean_alpha_pct"),
        "lightgbm_top_decile_alpha_pct": lightgbm_test.get("top_decile_mean_alpha_pct"),
        "top_decile_improvement_pct": _difference(
            lightgbm_test.get("top_decile_mean_alpha_pct"),
            baseline_test.get("top_decile_mean_alpha_pct"),
        ),
    }


def save_lightgbm_artifacts(
    model: Any,
    report: dict[str, Any],
    *,
    model_path: str | Path,
    report_path: str | Path,
) -> tuple[Path, Path]:
    model_destination = Path(model_path)
    report_destination = Path(report_path)
    model_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    model_temp = model_destination.with_suffix(f"{model_destination.suffix}.tmp")
    report_temp = report_destination.with_suffix(f"{report_destination.suffix}.tmp")
    model.booster_.save_model(str(model_temp))
    report_temp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    model_temp.replace(model_destination)
    report_temp.replace(report_destination)
    return model_destination, report_destination


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def _equal_date_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("sample_date")["ticker"].transform("size").to_numpy(dtype=float)
    weights = 1.0 / counts
    return weights / np.mean(weights)


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)
