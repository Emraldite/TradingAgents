from unittest.mock import patch

import pytest

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.capabilities import get_capabilities
from tradingagents.llm_clients.openai_client import OpenAIClient


MODEL = "openai/gpt-oss-20b"


@pytest.mark.parametrize(
    "model", ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "gpt-oss-120b"]
)
def test_gpt_oss_uses_json_schema_instead_of_schema_tool_calls(model):
    assert get_capabilities(model).preferred_structured_method == "json_schema"


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
        max_tokens=512,
        reasoning_effort="low",
        max_retries=1,
    ).get_llm()

    kwargs = mock_chat.call_args.kwargs
    assert kwargs["rate_limiter"] is limiter
    assert kwargs["max_tokens"] == 512
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["max_retries"] == 1


def test_groq_provider_kwargs_are_conservative_and_validated():
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {
        "llm_provider": "groq",
        "groq_requests_per_minute": 1,
        "groq_max_output_tokens": 512,
        "groq_max_retries": 1,
        "groq_reasoning_effort": "low",
    }

    kwargs = graph._get_provider_kwargs()

    assert kwargs["max_retries"] == 1
    assert kwargs["max_tokens"] == 512
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["rate_limiter"] is not None


@pytest.mark.parametrize(
    "rpm,max_tokens,retries,message",
    [
        (0, 1024, 1, "requests_per_minute must be positive"),
        (1, 0, 1, "max_output_tokens must be positive"),
        (1, 1024, -1, "max_retries cannot be negative"),
    ],
)
def test_groq_provider_kwargs_reject_invalid_limits(rpm, max_tokens, retries, message):
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {
        "llm_provider": "groq",
        "groq_requests_per_minute": rpm,
        "groq_max_output_tokens": max_tokens,
        "groq_max_retries": retries,
    }

    with pytest.raises(ValueError, match=message):
        graph._get_provider_kwargs()


def test_groq_provider_kwargs_reject_invalid_reasoning_effort():
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {
        "llm_provider": "groq",
        "groq_requests_per_minute": 1,
        "groq_max_output_tokens": 512,
        "groq_max_retries": 1,
        "groq_reasoning_effort": "extreme",
    }

    with pytest.raises(ValueError, match="must be low, medium, or high"):
        graph._get_provider_kwargs()
