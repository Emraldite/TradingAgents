import inspect

from tradingagents.agents.analysts.news_analyst import create_news_analyst


def test_news_prompt_uses_ticker_argument_name():
    source = inspect.getsource(create_news_analyst)

    assert "get_news(ticker, start_date, end_date)" in source
    assert "get_news(query" not in source
