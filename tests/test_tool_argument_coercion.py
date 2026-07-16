from tradingagents.agents.utils import market_data_validation_tools
from tradingagents.agents.utils import news_data_tools
from tradingagents.agents.utils import technical_indicators_tools


def _schema_types(tool, field: str) -> set[str]:
    schema = tool.args_schema.model_json_schema()["properties"][field]
    choices = schema.get("anyOf", [schema])
    return {choice.get("type", "") for choice in choices}


def test_indicator_tool_schema_accepts_numeric_strings_and_coerces_them(monkeypatch):
    captured = {}

    def route(name, symbol, indicator, curr_date, look_back_days):
        captured["look_back_days"] = look_back_days
        return "ok"

    monkeypatch.setattr(technical_indicators_tools, "route_to_vendor", route)

    result = technical_indicators_tools.get_indicators.invoke(
        {
            "symbol": "NVDA",
            "indicator": "close_50_sma",
            "curr_date": "2026-07-16",
            "look_back_days": "50",
        }
    )

    assert {"integer", "string"} <= _schema_types(
        technical_indicators_tools.get_indicators, "look_back_days"
    )
    assert result == "ok"
    assert captured["look_back_days"] == 50
    assert isinstance(captured["look_back_days"], int)


def test_market_snapshot_tool_coerces_numeric_string(monkeypatch):
    captured = {}

    def snapshot(symbol, curr_date, look_back_days):
        captured["look_back_days"] = look_back_days
        return "verified"

    monkeypatch.setattr(
        market_data_validation_tools,
        "build_verified_market_snapshot",
        snapshot,
    )

    result = market_data_validation_tools.get_verified_market_snapshot.invoke(
        {
            "symbol": "AAPL",
            "curr_date": "2026-07-16",
            "look_back_days": "30",
        }
    )

    assert result == "verified"
    assert captured["look_back_days"] == 30


def test_global_news_tool_coerces_numeric_strings(monkeypatch):
    captured = {}

    def route(name, curr_date, look_back_days, limit):
        captured.update(look_back_days=look_back_days, limit=limit)
        return "news"

    monkeypatch.setattr(news_data_tools, "route_to_vendor", route)

    result = news_data_tools.get_global_news.invoke(
        {
            "curr_date": "2026-07-16",
            "look_back_days": "7",
            "limit": "20",
        }
    )

    assert result == "news"
    assert captured == {"look_back_days": 7, "limit": 20}


def test_numeric_tool_arguments_reject_non_positive_or_non_numeric_values(monkeypatch):
    monkeypatch.setattr(
        technical_indicators_tools,
        "route_to_vendor",
        lambda *args: "should not run",
    )

    non_numeric = technical_indicators_tools.get_indicators.invoke(
        {
            "symbol": "NVDA",
            "indicator": "rsi",
            "curr_date": "2026-07-16",
            "look_back_days": "many",
        }
    )
    non_positive = technical_indicators_tools.get_indicators.invoke(
        {
            "symbol": "NVDA",
            "indicator": "rsi",
            "curr_date": "2026-07-16",
            "look_back_days": "0",
        }
    )

    assert "expected a positive integer" in non_numeric
    assert "expected a positive integer" in non_positive
