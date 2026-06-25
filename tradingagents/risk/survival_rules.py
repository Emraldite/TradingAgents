from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

MAX_POSITION_PCT = 0.08
DAILY_DRAWDOWN_LIMIT = 0.03
WEEKLY_DRAWDOWN_LIMIT = 0.07
MAX_OPEN_POSITIONS = 5
MIN_DAILY_VOLUME = 500_000


@dataclass
class SurvivalRules:
    halted_until: float = 0.0  # timestamp
    daily_start_value: float | None = None
    weekly_start_value: float | None = None
    daily_low: float | None = None
    weekly_low: float | None = None

    def check_kill_switch(self, portfolio_value: float) -> str | None:
        now = time.time()
        if now < self.halted_until:
            remaining = timedelta(seconds=int(self.halted_until - now))
            msg = f"Trading halted for another {remaining}"
            logger.warning(msg)
            return msg

        if self.daily_start_value is None:
            self.daily_start_value = portfolio_value
            self.daily_low = portfolio_value
        if self.weekly_start_value is None:
            self.weekly_start_value = portfolio_value
            self.weekly_low = portfolio_value

        self.daily_low = min(self.daily_low, portfolio_value)
        self.weekly_low = min(self.weekly_low, portfolio_value)

        daily_loss = (self.daily_start_value - portfolio_value) / self.daily_start_value
        if daily_loss > DAILY_DRAWDOWN_LIMIT:
            self.halted_until = now + 86400
            msg = f"KILL SWITCH — daily loss {daily_loss:.1%} exceeds {DAILY_DRAWDOWN_LIMIT:.1%}. Halted 24h."
            logger.error(msg)
            return msg

        weekly_loss = (self.weekly_start_value - portfolio_value) / self.weekly_start_value
        if weekly_loss > WEEKLY_DRAWDOWN_LIMIT:
            self.halted_until = now + 259200
            msg = f"KILL SWITCH — weekly loss {weekly_loss:.1%} exceeds {WEEKLY_DRAWDOWN_LIMIT:.1%}. Halted 72h."
            logger.error(msg)
            return msg

        return None

    def reset_daily(self) -> None:
        self.daily_start_value = None
        self.daily_low = None

    def reset_weekly(self) -> None:
        self.weekly_start_value = None
        self.weekly_low = None


def check_kill_switch(
    portfolio_value: float, rules: SurvivalRules
) -> str | None:
    return rules.check_kill_switch(portfolio_value)


def validate_trade(
    ticker: str,
    position_size: float,
    portfolio_value: float,
    open_positions: int,
    daily_volume: float | None,
    rules: SurvivalRules | None = None,
) -> tuple[bool, str]:
    if rules and rules.halted_until > time.time():
        return False, "Trading is halted by kill switch"

    if position_size > portfolio_value * MAX_POSITION_PCT:
        return (
            False,
            f"Position ${position_size:.0f} exceeds {MAX_POSITION_PCT:.0%} of portfolio (${portfolio_value:.0f})",
        )

    if open_positions >= MAX_OPEN_POSITIONS:
        return False, f"Already at max open positions ({MAX_OPEN_POSITIONS})"

    if daily_volume is not None and daily_volume < MIN_DAILY_VOLUME:
        return (
            False,
            f"Daily volume {daily_volume:.0f} below minimum {MIN_DAILY_VOLUME:,}",
        )

    return True, "Trade validated"
