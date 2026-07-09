from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.market_data_validator import build_verified_market_snapshot


@tool
def get_verified_market_snapshot(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "the current trading date, YYYY-mm-dd"],
    look_back_days: Annotated[
        int, "number of recent trading rows to include for sanity-checking"
    ] = 30,
) -> str:
    """Return a deterministic market-data snapshot for exact price claims."""
    return build_verified_market_snapshot(symbol, curr_date, look_back_days)
