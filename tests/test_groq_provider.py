from unittest.mock import patch

import pytest

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.openai_client import OpenAIClient


MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


@patch("tradingagents.llm_clients.openai_client.NormalizedChatOpenAI")
def test_groq_uses_openai_compatible_endpoint_and_key(mock_chat, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

    OpenAIClient(MODEL, provider="groq").get_llm()

    kwargs = mock_chat.call_args.kwargs
    assert kwargs["base_url"] == "https://api.groq.com/openai/v1"
    assert kwargs["api_key"] == "gsk-test"
    assert "use_responses_api" not in kwargs


def test_groq_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        OpenAIClient(MODEL, provider="groq").get_llm()


@patch("tradingagents.llm_clients.openai_client.NormalizedChatOpenAI")
def test_groq_forwards_shared_rate_limiter_and_retry_limit(mock_chat, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    limiter = object()

    OpenAIClient(
        MODEL,
        provider="groq",
        rate_limiter=limiter,
        max_retries=1,
    ).get_llm()

    kwargs = mock_chat.call_args.kwargs
    assert kwargs["rate_limiter"] is limiter
    assert kwargs["max_retries"] == 1


def test_groq_provider_kwargs_are_conservative_and_validated():
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {
        "llm_provider": "groq",
        "groq_requests_per_minute": 3,
        "groq_max_retries": 1,
    }

    kwargs = graph._get_provider_kwargs()

    assert kwargs["max_retries"] == 1
    assert kwargs["rate_limiter"] is not None


@pytest.mark.parametrize(
    "rpm,retries,message",
    [(0, 1, "must be positive"), (3, -1, "cannot be negative")],
)
def test_groq_provider_kwargs_reject_invalid_limits(rpm, retries, message):
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {
        "llm_provider": "groq",
        "groq_requests_per_minute": rpm,
        "groq_max_retries": retries,
    }

    with pytest.raises(ValueError, match=message):
        graph._get_provider_kwargs()
