"""Tests for TRADINGAGENTS_* env-var overlay onto DEFAULT_CONFIG."""

from __future__ import annotations

import importlib

import pytest

import tradingagents.default_config as default_config_module


def _reload_with_env(monkeypatch, **overrides):
    """Set/clear env vars then reload default_config to re-evaluate DEFAULT_CONFIG."""
    for key in list(default_config_module._ENV_OVERRIDES):
        monkeypatch.delenv(key, raising=False)
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    return importlib.reload(default_config_module)


def test_no_env_uses_built_in_defaults(monkeypatch):
    dc = _reload_with_env(monkeypatch)
    assert dc.DEFAULT_CONFIG["llm_provider"] == "groq"
    assert dc.DEFAULT_CONFIG["deep_think_llm"] == "meta-llama/llama-4-scout-17b-16e-instruct"
    assert dc.DEFAULT_CONFIG["quick_think_llm"] == "meta-llama/llama-4-scout-17b-16e-instruct"
    assert dc.DEFAULT_CONFIG["groq_requests_per_minute"] == 3
    assert dc.DEFAULT_CONFIG["groq_max_retries"] == 1
    assert dc.DEFAULT_CONFIG["backend_url"] is None
    assert dc.DEFAULT_CONFIG["max_debate_rounds"] == 1
    assert dc.DEFAULT_CONFIG["checkpoint_enabled"] is False


def test_string_overrides(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_LLM_PROVIDER="google",
        TRADINGAGENTS_DEEP_THINK_LLM="gemini-3-pro-preview",
        TRADINGAGENTS_QUICK_THINK_LLM="gemini-3-flash-preview",
        TRADINGAGENTS_LLM_BACKEND_URL="https://example.invalid/v1",
        TRADINGAGENTS_OUTPUT_LANGUAGE="Chinese",
    )
    assert dc.DEFAULT_CONFIG["llm_provider"] == "google"
    assert dc.DEFAULT_CONFIG["deep_think_llm"] == "gemini-3-pro-preview"
    assert dc.DEFAULT_CONFIG["quick_think_llm"] == "gemini-3-flash-preview"
    assert dc.DEFAULT_CONFIG["backend_url"] == "https://example.invalid/v1"
    assert dc.DEFAULT_CONFIG["output_language"] == "Chinese"


def test_int_coercion(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_MAX_DEBATE_ROUNDS="3",
        TRADINGAGENTS_MAX_RISK_ROUNDS="2",
        TRADINGAGENTS_GROQ_REQUESTS_PER_MINUTE="9",
        TRADINGAGENTS_GROQ_MAX_RETRIES="2",
    )
    assert dc.DEFAULT_CONFIG["max_debate_rounds"] == 3
    assert isinstance(dc.DEFAULT_CONFIG["max_debate_rounds"], int)
    assert dc.DEFAULT_CONFIG["max_risk_discuss_rounds"] == 2
    assert isinstance(dc.DEFAULT_CONFIG["max_risk_discuss_rounds"], int)
    assert dc.DEFAULT_CONFIG["groq_requests_per_minute"] == 9
    assert dc.DEFAULT_CONFIG["groq_max_retries"] == 2


def test_scheduler_float_coercion(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_BUY_POSITION_PCT="0.015",
        TRADINGAGENTS_SCORECARD_WARMUP_POSITION_PCT="0.004",
        TRADINGAGENTS_STOP_LOSS_PCT="-0.04",
        TRADINGAGENTS_MAX_HOLD_TRADING_DAYS="7",
    )
    assert dc.DEFAULT_CONFIG["scheduler_buy_position_pct"] == 0.015
    assert isinstance(dc.DEFAULT_CONFIG["scheduler_buy_position_pct"], float)
    assert dc.DEFAULT_CONFIG["scorecard_warmup_position_pct"] == 0.004
    assert isinstance(dc.DEFAULT_CONFIG["scorecard_warmup_position_pct"], float)
    assert dc.DEFAULT_CONFIG["stop_loss_pct"] == -0.04
    assert dc.DEFAULT_CONFIG["max_hold_trading_days"] == 7


def test_data_collection_overrides(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_NEWS_ARTICLE_LIMIT="12",
        TRADINGAGENTS_CONGRESSIONAL_LOOKBACK_DAYS="30",
        TRADINGAGENTS_CONGRESSIONAL_CACHE_HOURS="2",
    )
    assert dc.DEFAULT_CONFIG["news_article_limit"] == 12
    assert dc.DEFAULT_CONFIG["congressional_lookback_days"] == 30
    assert dc.DEFAULT_CONFIG["congressional_cache_hours"] == 2


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True), ("True", True), ("1", True), ("yes", True), ("on", True),
        ("false", False), ("False", False), ("0", False), ("no", False), ("off", False),
    ],
)
def test_bool_coercion(monkeypatch, raw, expected):
    dc = _reload_with_env(monkeypatch, TRADINGAGENTS_CHECKPOINT_ENABLED=raw)
    assert dc.DEFAULT_CONFIG["checkpoint_enabled"] is expected


def test_empty_env_value_is_passthrough(monkeypatch):
    """Empty TRADINGAGENTS_* values must not clobber the built-in default."""
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_LLM_PROVIDER="",
        TRADINGAGENTS_MAX_DEBATE_ROUNDS="",
    )
    assert dc.DEFAULT_CONFIG["llm_provider"] == "groq"
    assert dc.DEFAULT_CONFIG["max_debate_rounds"] == 1


def test_invalid_int_raises(monkeypatch):
    """Garbage int values should surface a ValueError at import, not silently misconfigure."""
    monkeypatch.setenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "not-a-number")
    with pytest.raises(ValueError):
        importlib.reload(default_config_module)
    # Restore module state for subsequent tests in this process
    monkeypatch.delenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", raising=False)
    importlib.reload(default_config_module)


def test_unknown_env_var_is_ignored(monkeypatch):
    """Env vars outside _ENV_OVERRIDES must not bleed into DEFAULT_CONFIG."""
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_NONEXISTENT_KEY="oops",
    )
    assert "nonexistent_key" not in dc.DEFAULT_CONFIG
