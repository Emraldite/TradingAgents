from datetime import datetime, timedelta

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    get_news,
)
from tradingagents.dataflows.stocktwits import fetch_stocktwits_messages
from tradingagents.dataflows.yahoo_news import fetch_yahoo_news, fetch_yahoo_rss
from tradingagents.dataflows.manipulation_detector import detect_manipulation


def _seven_days_back(trade_date: str) -> str:
    return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")


def create_sentiment_analyst(llm):
    def sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        start_date = _seven_days_back(end_date)
        instrument_context = build_instrument_context(ticker)

        news_block = get_news.func(ticker, start_date, end_date)
        stocktwits_block = fetch_stocktwits_messages(ticker, limit=30)
        yahoo_news_block = fetch_yahoo_news(ticker)
        yahoo_rss_block = fetch_yahoo_rss(ticker)
        manipulation_block = detect_manipulation(ticker)

        system_message = _build_system_message(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            news_block=news_block,
            stocktwits_block=stocktwits_block,
            yahoo_news_block=yahoo_news_block,
            yahoo_rss_block=yahoo_rss_block,
            manipulation_block=manipulation_block,
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
            "sentiment_report": result.content,
        }

    return sentiment_analyst_node


def _build_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news_block: str,
    stocktwits_block: str,
    yahoo_news_block: str,
    yahoo_rss_block: str,
    manipulation_block: dict | None = None,
) -> str:
    manipulation_html = ""
    if manipulation_block:
        manip = manipulation_block
        manipulation_html = f"""
### Manipulation detection — signal integrity check
This ticker has been checked for coordinated manipulation signals (pump & dump).

  Organic score: {manip.get("organic_score", "?")}/10  (higher = more legitimate)
  Manipulation risk: {manip.get("manipulation_risk", "?")}/10  (higher = more suspicious)
  Recommendation: {manip.get("recommendation", "?")}

If manipulation risk exceeds 6/10, the signal should be treated as potentially unreliable regardless of sentiment scores below.
Details: {manip.get("details", {})}
"""

    return f"""You are a financial market sentiment analyst. Your task is to produce a comprehensive sentiment report for {ticker} covering the period from {start_date} to {end_date}, drawing on data sources that have already been collected for you.

## Data sources (pre-fetched, in this prompt)

### News headlines — Yahoo Finance, past 7 days
Institutional framing. Fact-driven, slower-moving signal.

<start_of_news>
{news_block}
<end_of_news>

### StockTwits messages — retail-trader social platform indexed by cashtag
Fast-moving signal. Each message carries a user-labeled sentiment tag (Bullish / Bearish / no-label) plus the message body.

<start_of_stocktwits>
{stocktwits_block}
<end_of_stocktwits>

### Yahoo Finance .news property — per-ticker news stream
Latest headlines and summaries from Yahoo Finance's own news feed for {ticker}.

<start_of_yahoo_news>
{yahoo_news_block}
<end_of_yahoo_news>

### Yahoo Finance RSS feed — headline stream
RSS headline feed from finance.yahoo.com for {ticker}.

<start_of_yahoo_rss>
{yahoo_rss_block}
<end_of_yahoo_rss>

{manipulation_html}
## How to analyze this data (best practices)

1. **Read the StockTwits Bullish/Bearish ratio as a leading retail-sentiment signal.** A 70/30 bullish/bearish split is moderately bullish; ≥90/10 may indicate over-extension and contrarian risk; 50/50 is uncertainty. Sample size matters — base rates on the actual message count, not percentages alone.

2. **Look for cross-source divergences.** If Yahoo Finance news framing is bearish but StockTwits is overwhelmingly bullish, that mismatch is itself a signal — it can mean retail is leaning into a thesis the news flow hasn't caught up to (or vice versa, that retail is chasing while institutions are cautious).

3. **Distinguish opinion from event.** A news headline is an event; a StockTwits post ("buying NVDA, this is going to moon") is opinion. Both are inputs but should be weighted differently in your conclusions.

4. **Identify recurring narrative themes.** What topic keeps coming up across sources? That's the dominant narrative driving current sentiment.

5. **Be honest about data limits.** If StockTwits returned only a handful of messages, or a source returned an "<unavailable>" placeholder, the sentiment read is less robust — flag this caveat explicitly.

6. **Identify catalysts and risks** that emerge across sources — news of upcoming earnings, product launches, competitive threats, macro headlines, etc.

7. **Past sentiment is not predictive.** Frame your conclusions as signal for the trader to weigh alongside fundamentals and technicals, not as a price call.

## Output

Produce a sentiment report covering, in order:

1. **Overall sentiment direction** — Bullish / Bearish / Neutral / Mixed — with a brief confidence note based on data quality and sample size.
2. **Source-by-source breakdown** — what each of news / StockTwits / Yahoo News / Yahoo RSS is telling you, with specific evidence (cite message counts, ratios, notable headlines).
3. **Divergences, alignments, and key narratives** across sources.
4. **Catalysts and risks** surfaced by the data.
5. **Markdown table** at the end summarizing key sentiment signals, their direction, source, and supporting evidence.

{get_language_instruction()}"""


def create_social_media_analyst(llm):
    import warnings
    warnings.warn(
        "create_social_media_analyst is deprecated. Use create_sentiment_analyst instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_sentiment_analyst(llm)
