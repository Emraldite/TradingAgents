from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.sec_insider_data import (
    get_sec_insider_activity,
    summarize_sec_insider_activity,
)


def create_insider_analyst(llm):
    def insider_analyst_node(state):
        ticker = state["company_of_interest"]
        analysis_date = state["trade_date"]
        config = get_config()
        activity = get_sec_insider_activity(
            ticker,
            as_of_date=analysis_date,
            lookback_days=int(config.get("insider_lookback_days", 30)),
        )
        system_message = _build_system_message(
            ticker=ticker,
            analysis_date=analysis_date,
            activity=activity,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a corporate-insider activity analyst collaborating with other analysts. "
                    "Use only the pre-fetched official SEC Form 4 evidence below. Do not invent missing "
                    "transactions or treat the word insider as proof of illegal activity.\n"
                    "{system_message}\nFor your reference, the analysis date is {analysis_date}. "
                    "{instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        ).partial(
            system_message=system_message,
            analysis_date=analysis_date,
            instrument_context=build_instrument_context(ticker),
        )
        result = (prompt | llm).invoke(state["messages"])
        return {"messages": [result], "insider_report": result.content}

    return insider_analyst_node


def _format_activity(activity) -> str:
    status = str(activity.attrs.get("data_status", "unavailable"))
    reason = str(activity.attrs.get("data_reason", ""))
    if status == "unavailable":
        return f"SEC data unavailable: {reason or 'unknown error'}. Treat the signal as neutral."
    if activity.empty:
        return "No qualifying open-market Form 4 purchases or sales were found. Treat this as neutral."

    lines = []
    for _, row in activity.head(15).iterrows():
        plan_label = " | 10b5-1 plan" if row.get("planned_10b5_1") else ""
        value = float(row.get("value", 0) or 0)
        lines.append(
            f"- Filed {row.get('filing_date')} | traded {row.get('transaction_date')} | "
            f"{row.get('owner')} ({row.get('role')}) | code {row.get('transaction_code')} "
            f"{row.get('transaction_type')} | value ${value:,.0f} | "
            f"deterministic score {int(row.get('signal_score', 0)):+d}{plan_label}"
        )
    return "\n".join(lines)


def _build_system_message(*, ticker: str, analysis_date: str, activity) -> str:
    summary = summarize_sec_insider_activity(activity)
    return f"""Analyze official SEC Form 4 corporate-insider activity for {ticker} using only filings public on or before {analysis_date}.

## Data status and qualifying transactions
{_format_activity(activity)}

## Deterministic summary
- Open-market purchases: {summary['purchase_count']} totaling ${summary['purchase_value']:,.0f}
- Open-market sales: {summary['sale_count']} totaling ${summary['sale_value']:,.0f}
- Unique open-market buyers: {summary['unique_buyers']}
- Conservative aggregate score: {summary['signal_score']:+d} on a -10 to +10 scale

## Interpretation rules
1. Code P open-market purchases are the strongest positive evidence, especially multiple independent buyers.
2. Code S sales are weak negative evidence because diversification, taxes, and personal liquidity can explain them.
3. Transactions identified as Rule 10b5-1 planned trades receive less weight.
4. Awards, gifts, option exercises, tax withholding, and derivative transactions were removed before analysis.
5. No activity or unavailable data is neutral, never bullish or bearish.
6. This is one supporting signal. Do not override company fundamentals, market evidence, or deterministic risk controls.

Produce a concise report covering the evidence, its limitations, and a Bullish/Bearish/Neutral insider signal with confidence.

{get_language_instruction()}"""
