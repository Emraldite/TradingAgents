from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.market_data_validator import build_verified_market_snapshot


@tool
def get_verified_market_snapshot(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "the current trading date, YYYY-mm-dd"],
    look_back_days: Annotated[
        int | str,
        "positive integer or numeric string for recent rows to sanity-check",
    ] = 30,
) -> str:
    """Return a deterministic market-data snapshot for exact price claims."""
    try:
        days = int(look_back_days)
    except (TypeError, ValueError):
        return f"Invalid look_back_days: {look_back_days!r}; expected a positive integer"
    if days <= 0:
        return f"Invalid look_back_days: {look_back_days!r}; expected a positive integer"
    return build_verified_market_snapshot(symbol, curr_date, days)
