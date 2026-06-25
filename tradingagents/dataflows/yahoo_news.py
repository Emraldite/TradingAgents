from __future__ import annotations

import logging
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_yahoo_news(ticker: str, max_articles: int = 10) -> str:
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
    except Exception as exc:
        logger.warning("yfinance news failed for %s: %s", ticker, exc)
        return "<yahoo finance news unavailable>"

    if not news:
        return f"<no Yahoo Finance news found for {ticker}>"

    lines = []
    for article in news[:max_articles]:
        content = article.get("content", {})
        title = content.get("title", "").replace("\n", " ").strip()
        provider = content.get("provider", {})
        publisher = provider.get("displayName", "") if isinstance(provider, dict) else ""
        summary = content.get("summary", "").replace("\n", " ").strip()[:300]
        pub_date = content.get("pubDate", "")
        lines.append(f"  [{pub_date}] {title} ({publisher})")
        if summary:
            lines.append(f"    {summary}")
    if not lines:
        return f"<no parseable Yahoo Finance news for {ticker}>"
    return "\n".join(lines)


def fetch_yahoo_rss(ticker: str, max_items: int = 10) -> str:
    url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    req = Request(url, headers={"User-Agent": "tradingagents-extended/0.1"})
    try:
        with urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, TimeoutError) as exc:
        logger.warning("Yahoo RSS failed for %s: %s", ticker, exc)
        return "<yahoo rss unavailable>"

    items = re.findall(r"<title><!\[CDATA\[(.+?)\]\]></title>", html)
    items = [t.strip() for t in items if t.strip() and t.strip() != ticker]

    if not items:
        return f"<no Yahoo RSS headlines found for {ticker}>"
    return "\n".join(f"  {t}" for t in items[:max_items])
