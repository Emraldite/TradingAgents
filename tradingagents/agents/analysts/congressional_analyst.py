from datetime import datetime, timedelta

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.dataflows.congressional_data import (
    get_conviction_watchlist,
    POLITICIAN_COMMITTEE_MAP,
    COMMITTEE_SECTOR_MAP,
)


def create_congressional_analyst(llm):
    def congressional_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        lookback_days = 45

        watchlist = get_conviction_watchlist(
            lookback_days=lookback_days, min_score=6
        )
        ticker_trades = watchlist[watchlist["ticker"] == ticker] if not watchlist.empty else watchlist
        top_tickers = (
            watchlist.groupby("ticker")
            .agg({"conviction_score": "max", "politician": lambda x: list(x)})
            .sort_values("conviction_score", ascending=False)
            .head(10)
            if not watchlist.empty
            else []
        )

        instrument_context = build_instrument_context(ticker)
        system_message = _build_system_message(
            ticker=ticker,
            ticker_trades=ticker_trades,
            top_tickers=top_tickers,
            end_date=end_date,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    "\n{system_message}\n"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=end_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm
        result = chain.invoke(state["messages"])

        return {
            "messages": [result],
            "congressional_report": result.content,
        }

    return congressional_analyst_node


def _format_ticker_trades(ticker_trades) -> str:
    if ticker_trades.empty:
        return "No congressional trades found for this ticker in the lookback period."
    lines = []
    for _, row in ticker_trades.iterrows():
        lines.append(
            f"  - {row.get('politician', '?')} | "
            f"Type: {row.get('trade_type', '?')} | "
            f"Amount: ${row.get('amount', 0):,.0f} | "
            f"Score: {row.get('conviction_score', 0)}/10"
        )
    return "\n".join(lines)


def _format_top_tickers(top_tickers) -> str:
    if top_tickers is None or (hasattr(top_tickers, "empty") and top_tickers.empty) or (isinstance(top_tickers, list) and not top_tickers):
        return "No tickers on the conviction watchlist."
    lines = ["Ticker | Conviction Score | Politicians"]
    for ticker, row in top_tickers.iterrows():
        politicians = ", ".join(row["politician"][:3])
        lines.append(f"  {ticker} | {row['conviction_score']}/10 | {politicians}")
    return "\n".join(lines)


def _build_system_message(
    *,
    ticker: str,
    ticker_trades,
    top_tickers,
    end_date: str,
) -> str:
    return f"""You are a congressional trading analyst. Your task is to analyze US politician stock trades for {ticker} using disclosed financial transactions from the STOCK Act.

## Congressional trading data (pre-fetched)

### Trades for {ticker} in the lookback period
{_format_ticker_trades(ticker_trades)}

### Top conviction tickers across all politicians
{_format_top_tickers(top_tickers)}

### Committee-Sector mapping (reference)
Known committee-sector alignments that give politicians informational advantage.

{_format_committee_map()}

### Politician-Committee mapping (reference)
{_format_politician_committee_map()}

## How to analyze this data

1. **Committee alignment matters most.** A politician on the Armed Services committee buying defense stocks is a stronger signal than a Finance committee member buying the same stock.

2. **Trade size relative to historical average.** Larger-than-usual trades signal higher conviction.

3. **Recency.** More recent disclosures are more relevant — older trades may already be priced in.

4. **Cross-politician corroboration.** If multiple politicians are buying the same sector, that's stronger than isolated trades.

5. **Consider both buy and sell activity.** Heavy selling by well-committee-aligned politicians can be a bearish signal.

## Output

Produce a congressional trading report covering:

1. **Congressional signal for {ticker}** — is there any congressional activity, and what does it suggest?
2. **Conviction score breakdown** — committee relevance, trade size, recency, and corroboration for this ticker.
3. **Context from broader watchlist** — what are the top conviction tickers overall and are there sector trends?
4. **Signal conclusion** — Bullish / Bearish / Neutral based on congressional data alone, with a confidence note.

{get_language_instruction()}"""


def _format_committee_map() -> str:
    lines = []
    for committee, tickers in COMMITTEE_SECTOR_MAP.items():
        lines.append(f"  {committee}: {', '.join(tickers)}")
    return "\n".join(lines)


def _format_politician_committee_map() -> str:
    lines = []
    for politician, committees in POLITICIAN_COMMITTEE_MAP.items():
        lines.append(f"  {politician}: {', '.join(committees)}")
    return "\n".join(lines)
