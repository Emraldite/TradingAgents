from unittest.mock import patch

import httpx
import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from tradingagents.graph.trading_graph import (
    TradingAgentsGraph,
    _FallbackAuditCallback,
    _with_retryable_fallback,
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.openai_client import OpenAIClient
from tradingagents.llm_clients.openai_client import NormalizedChatOpenAI


class _StructuredResult(BaseModel):
    rating: str


def _response(status: int) -> httpx.Response:
    request = httpx.Request("POST", "https://provider.invalid/v1/chat/completions")
    return httpx.Response(status, request=request)


def _raising(exc: Exception):
    def invoke(_):
        raise exc

    return RunnableLambda(invoke)


@pytest.mark.parametrize(
    "error",
    [
        RateLimitError("limited", response=_response(429), body={}),
        APIConnectionError(request=httpx.Request("POST", "https://groq.invalid")),
        InternalServerError("down", response=_response(503), body={}),
    ],
)
def test_retryable_provider_failures_use_secondary(error):
    wrapped = _with_retryable_fallback(
        _raising(error), RunnableLambda(lambda _: "cerebras")
    )

    assert wrapped.invoke("request") == "cerebras"


@pytest.mark.parametrize(
    "error",
    [
        BadRequestError("bad tool call", response=_response(400), body={}),
        AuthenticationError("bad key", response=_response(401), body={}),
        APIStatusError("too large", response=_response(413), body={}),
    ],
)
def test_request_auth_and_oversize_errors_do_not_switch_provider(error):
    fallback_calls = []
    wrapped = _with_retryable_fallback(
        _raising(error),
        RunnableLambda(lambda value: fallback_calls.append(value) or "cerebras"),
    )

    with pytest.raises(type(error)):
        wrapped.invoke("request")

    assert fallback_calls == []


@patch("tradingagents.llm_clients.openai_client.NormalizedChatOpenAI")
def test_cerebras_uses_openai_compatible_endpoint_and_key(mock_chat, monkeypatch):
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-test")

    OpenAIClient("gpt-oss-120b", provider="cerebras").get_llm()

    kwargs = mock_chat.call_args.kwargs
    assert kwargs["base_url"] == "https://api.cerebras.ai/v1"
    assert kwargs["api_key"] == "csk-test"
    assert "use_responses_api" not in kwargs


def test_cerebras_requires_api_key(monkeypatch):
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)

    with pytest.raises(ValueError, match="CEREBRAS_API_KEY"):
        OpenAIClient("gpt-oss-120b", provider="cerebras").get_llm()


def test_cerebras_provider_kwargs_match_account_headroom():
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {
        "cerebras_requests_per_minute": 3,
        "cerebras_max_output_tokens": 1024,
        "cerebras_max_retries": 1,
    }

    kwargs = graph._get_provider_kwargs("cerebras")

    assert kwargs["rate_limiter"] is not None
    assert kwargs["max_tokens"] == 1024
    assert kwargs["max_retries"] == 1


def test_tools_and_structured_output_keep_the_fallback_chain():
    primary = NormalizedChatOpenAI(model="primary", api_key="test")
    fallback = NormalizedChatOpenAI(model="fallback", api_key="test")
    wrapped = _with_retryable_fallback(primary, fallback)

    assert len(wrapped.bind_tools([]).fallbacks) == 1
    assert len(wrapped.with_structured_output(_StructuredResult).fallbacks) == 1


@pytest.mark.parametrize(
    "primary,quick,deep,secondary,secondary_quick,secondary_deep",
    [
        (
            "cerebras", "gpt-oss-120b", "gpt-oss-120b",
            "groq", "openai/gpt-oss-20b", "openai/gpt-oss-120b",
        ),
        (
            "groq", "openai/gpt-oss-20b", "openai/gpt-oss-120b",
            "cerebras", "gpt-oss-120b", "gpt-oss-120b",
        ),
    ],
)
def test_graph_builds_configurable_primary_secondary_pair(
    monkeypatch,
    tmp_path,
    primary,
    quick,
    deep,
    secondary,
    secondary_quick,
    secondary_deep,
):
    import tradingagents.graph.trading_graph as graph_module

    calls = []

    class FakeLLM:
        def __init__(self, name):
            self.name = name
            self.fallbacks = []

        def with_fallbacks(self, fallbacks, *, exceptions_to_handle):
            self.fallbacks = list(fallbacks)
            self.exceptions_to_handle = exceptions_to_handle
            return self

    class FakeClient:
        def __init__(self, llm):
            self.llm = llm

        def get_llm(self):
            return self.llm

    def fake_create_llm_client(**kwargs):
        calls.append(kwargs)
        return FakeClient(FakeLLM(f"{kwargs['provider']}:{kwargs['model']}"))

    class FakeWorkflow:
        def compile(self):
            return object()

    class FakeGraphSetup:
        def __init__(self, *args, **kwargs):
            pass

        def setup_graph(self, selected_analysts):
            return FakeWorkflow()

    monkeypatch.setattr(graph_module, "create_llm_client", fake_create_llm_client)
    monkeypatch.setattr(graph_module, "TradingMemoryLog", lambda config: object())
    monkeypatch.setattr(graph_module, "GraphSetup", FakeGraphSetup)
    monkeypatch.setattr(TradingAgentsGraph, "_create_tool_nodes", lambda self: {})

    config = dict(DEFAULT_CONFIG)
    config.update(
        {
            "data_cache_dir": str(tmp_path / "cache"),
            "results_dir": str(tmp_path / "results"),
            "llm_provider": primary,
            "quick_think_llm": quick,
            "deep_think_llm": deep,
            "secondary_llm_provider": secondary,
            "secondary_quick_think_llm": secondary_quick,
            "secondary_deep_think_llm": secondary_deep,
        }
    )
    graph = TradingAgentsGraph(selected_analysts=[], config=config)

    assert [call["provider"] for call in calls] == [
        primary, primary, secondary, secondary
    ]
    assert calls[-1]["max_tokens"] == 1024
    assert calls[-1]["max_retries"] == 1
    assert calls[0]["rate_limiter"] is calls[1]["rate_limiter"]
    assert calls[2]["rate_limiter"] is calls[3]["rate_limiter"]
    assert graph.quick_thinking_llm.fallbacks[0].name == (
        f"{secondary}:{secondary_quick}"
    )
    assert graph.deep_thinking_llm.fallbacks[0].name == (
        f"{secondary}:{secondary_deep}"
    )


def test_fallback_audit_records_only_completed_provider_calls(caplog):
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {"llm_provider": "cerebras"}
    graph._fallback_providers_used = set()
    callback = _FallbackAuditCallback("groq", graph._fallback_providers_used)

    callback.on_chat_model_start({}, [])
    assert graph.llm_provider_audit_label() == "cerebras"

    callback.on_llm_end(None)
    assert graph.llm_provider_audit_label() == "cerebras+groq"
    assert "fallback completed through groq" in caplog.text

    graph.reset_llm_provider_audit()
    assert graph.llm_provider_audit_label() == "cerebras"
