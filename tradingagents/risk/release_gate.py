from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tradingagents.risk.scorecard import Scorecard
from tradingagents.state_store import StrategyStateStore


RELEASE_CONFIG_KEYS = (
    "strategy_version",
    "stop_loss_pct",
    "take_profit_pct",
    "max_hold_trading_days",
    "manipulation_sell_threshold",
    "limit_slippage_bps",
    "shadow_slippage_bps",
    "market_snapshot_max_age_minutes",
    "max_quote_spread_pct",
    "llm_provider",
    "quick_think_llm",
    "deep_think_llm",
    "secondary_llm_provider",
    "secondary_quick_think_llm",
    "secondary_deep_think_llm",
    "groq_max_output_tokens",
    "groq_reasoning_effort",
    "cerebras_max_output_tokens",
    "news_article_limit",
    "global_news_article_limit",
    "insider_lookback_days",
    "insider_max_filings",
    "scheduler_buy_position_pct",
    "scheduler_overweight_position_pct",
    "scheduler_max_position_pct",
    "scheduler_max_open_positions",
    "scheduler_min_daily_volume",
    "discovery_max_candidates",
    "discovery_min_volume_ratio",
    "daily_drawdown_limit",
    "weekly_drawdown_limit",
    "total_drawdown_limit",
    "benchmark_ticker",
    "scorecard_horizon_days",
    "scorecard_stop_loss_pct",
    "scorecard_warmup_position_pct",
    "scorecard_tier1_position_pct",
    "scorecard_tier2_position_pct",
    "scorecard_min_resolved_decisions",
    "scorecard_tier2_min_decisions",
    "max_real_money_notional",
    "real_position_pct",
    "real_max_open_positions",
    "real_max_exposure_pct",
)

REQUIRED_RELEASE_CHECKS = {
    "out_of_sample_backtest",
    "resolved_decisions",
    "positive_alpha",
    "controlled_scorecard_drawdown",
    "paper_duration",
    "paper_cycles_exist",
    "acceptable_cycle_failure_rate",
    "no_unresolved_cycles",
    "no_unacknowledged_critical_events",
    "no_unprotected_positions",
    "no_active_orders",
    "account_id_configured",
}


def release_strategy_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config.get(key) for key in RELEASE_CONFIG_KEYS}


def strategy_fingerprint(strategy_key: str, strategy_config: dict[str, Any]) -> str:
    """Stable identity for the strategy version and risk settings being approved."""
    payload = json.dumps(
        {"strategy_key": strategy_key, "strategy_config": strategy_config},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_backtest_validation(
    path: str | Path | None,
    *,
    strategy_key: str,
) -> tuple[dict[str, Any] | None, bool]:
    if not path:
        return None, False
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, False
    result = report.get("result") or {}
    try:
        numeric_checks = {
            "diversified_tickers": int(result.get("ticker_count", 0)) >= 5,
            "enough_trades": int(result.get("num_trades", 0)) >= 30,
            "positive_alpha": float(result.get("alpha_pct", 0)) > 0,
            "controlled_drawdown": float(result.get("max_drawdown_pct", -100)) > -10,
        }
    except (TypeError, ValueError):
        numeric_checks = {
            "diversified_tickers": False,
            "enough_trades": False,
            "positive_alpha": False,
            "controlled_drawdown": False,
        }
    checks = {
        "strategy_matches": report.get("strategy_key") == strategy_key,
        **numeric_checks,
        "has_date_range": bool(result.get("start_date") and result.get("end_date")),
    }
    report["checks"] = checks
    return report, all(checks.values())


def build_release_report(
    *,
    store: StrategyStateStore,
    scorecard: Scorecard,
    strategy_key: str,
    account_id: str,
    output_path: str | Path,
    strategy_config: dict[str, Any] | None = None,
    backtest_validation_path: str | Path | None = None,
    min_resolved_decisions: int = 100,
    min_paper_days: int = 90,
) -> dict[str, Any]:
    strategy_config = strategy_config or {}
    score = scorecard.strategy_summary(strategy_key)
    operations = store.release_statistics()
    health = store.health_snapshot()
    backtest, backtest_passed = _load_backtest_validation(
        backtest_validation_path,
        strategy_key=strategy_key,
    )
    checks = {
        "out_of_sample_backtest": backtest_passed,
        "resolved_decisions": int(score["resolved_decisions"]) >= min_resolved_decisions,
        "positive_alpha": float(score["avg_alpha_pct"]) > 0,
        "controlled_scorecard_drawdown": float(score["max_drawdown_pct"]) > -3,
        "paper_duration": int(operations["paper_span_days"]) >= min_paper_days,
        "paper_cycles_exist": int(operations["paper_cycles"]) >= min_resolved_decisions,
        "acceptable_cycle_failure_rate": float(operations["failure_rate_pct"]) <= 1.0,
        "no_unresolved_cycles": int(operations["unresolved_cycles"]) == 0,
        "no_unacknowledged_critical_events": int(
            operations["unacknowledged_critical_events"]
        ) == 0,
        "no_unprotected_positions": not health["unprotected_tickers"],
        "no_active_orders": int(health["active_orders"]) == 0,
        "account_id_configured": bool(account_id.strip()),
    }
    report = {
        "schema_version": 2,
        "approved": all(checks.values()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "account_id": account_id.strip(),
        "strategy_key": strategy_key,
        "strategy_fingerprint": strategy_fingerprint(strategy_key, strategy_config),
        "strategy_config": strategy_config,
        "checks": checks,
        "scorecard": score,
        "operations": operations,
        "performance": health["performance"],
        "backtest_validation": backtest,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
