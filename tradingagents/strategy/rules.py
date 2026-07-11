from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StrategyRules:
    stop_loss_pct: float = -0.07
    take_profit_pct: float = 0.12
    max_hold_trading_days: int = 10
    manipulation_sell_threshold: float = 0.85
    buy_position_pct: float = 0.01
    overweight_position_pct: float = 0.02
    max_position_pct: float = 0.03
    max_open_positions: int = 5
    min_daily_volume: float = 500_000


def strategy_rules_from_config(config: dict[str, Any]) -> StrategyRules:
    return StrategyRules(
        stop_loss_pct=float(config.get("stop_loss_pct", -0.07)),
        take_profit_pct=float(config.get("take_profit_pct", 0.12)),
        max_hold_trading_days=int(config.get("max_hold_trading_days", 10)),
        manipulation_sell_threshold=float(
            config.get("manipulation_sell_threshold", 0.85)
        ),
        buy_position_pct=float(config.get("scheduler_buy_position_pct", 0.01)),
        overweight_position_pct=float(
            config.get("scheduler_overweight_position_pct", 0.02)
        ),
        max_position_pct=float(config.get("scheduler_max_position_pct", 0.03)),
        max_open_positions=int(config.get("scheduler_max_open_positions", 5)),
        min_daily_volume=float(config.get("scheduler_min_daily_volume", 500_000)),
    )


def position_pct_for_rating(rating: str, rules: StrategyRules) -> float:
    if rating == "Buy":
        return rules.buy_position_pct
    if rating == "Overweight":
        return rules.overweight_position_pct
    return 0.0


def exit_reason(
    *,
    entry_price: float,
    current_price: float,
    holding_days: int,
    manipulation_score: float,
    rules: StrategyRules,
) -> str | None:
    if entry_price <= 0 or current_price <= 0:
        return None
    return_pct = (current_price - entry_price) / entry_price
    if return_pct <= rules.stop_loss_pct:
        return "stop_loss"
    if return_pct >= rules.take_profit_pct:
        return "take_profit"
    if holding_days >= rules.max_hold_trading_days:
        return "time_exit"
    if manipulation_score >= rules.manipulation_sell_threshold:
        return "manipulation_spike"
    return None


def stop_price(entry_price: float, rules: StrategyRules) -> float:
    if entry_price <= 0:
        return 0.0
    return round(entry_price * (1 + rules.stop_loss_pct), 2)


def take_profit_price(entry_price: float, rules: StrategyRules) -> float:
    if entry_price <= 0:
        return 0.0
    return round(entry_price * (1 + rules.take_profit_pct), 2)
