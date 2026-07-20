import os

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

# Single source of truth for env-var → config-key overrides. To expose
# a new config key for environment-based override, add a row here — no
# entry-point script changes required. Coercion is driven by the type
# of the existing default, so users can keep writing plain strings in
# their .env file.
_ENV_OVERRIDES = {
    "TRADINGAGENTS_LLM_PROVIDER":         "llm_provider",
    "TRADINGAGENTS_DEEP_THINK_LLM":       "deep_think_llm",
    "TRADINGAGENTS_QUICK_THINK_LLM":      "quick_think_llm",
    "TRADINGAGENTS_SECONDARY_LLM_PROVIDER": "secondary_llm_provider",
    "TRADINGAGENTS_SECONDARY_DEEP_THINK_LLM": "secondary_deep_think_llm",
    "TRADINGAGENTS_SECONDARY_QUICK_THINK_LLM": "secondary_quick_think_llm",
    "TRADINGAGENTS_LLM_BACKEND_URL":      "backend_url",
    "TRADINGAGENTS_GOOGLE_THINKING_LEVEL": "google_thinking_level",
    "TRADINGAGENTS_GROQ_REQUESTS_PER_MINUTE": "groq_requests_per_minute",
    "TRADINGAGENTS_GROQ_MAX_OUTPUT_TOKENS": "groq_max_output_tokens",
    "TRADINGAGENTS_GROQ_MAX_RETRIES":     "groq_max_retries",
    "TRADINGAGENTS_GROQ_REASONING_EFFORT": "groq_reasoning_effort",
    "TRADINGAGENTS_CEREBRAS_REQUESTS_PER_MINUTE": "cerebras_requests_per_minute",
    "TRADINGAGENTS_CEREBRAS_MAX_OUTPUT_TOKENS": "cerebras_max_output_tokens",
    "TRADINGAGENTS_CEREBRAS_MAX_RETRIES": "cerebras_max_retries",
    "TRADINGAGENTS_OUTPUT_LANGUAGE":      "output_language",
    "TRADINGAGENTS_MAX_DEBATE_ROUNDS":    "max_debate_rounds",
    "TRADINGAGENTS_MAX_RISK_ROUNDS":      "max_risk_discuss_rounds",
    "TRADINGAGENTS_CHECKPOINT_ENABLED":   "checkpoint_enabled",
    "TRADINGAGENTS_NEWS_ARTICLE_LIMIT":   "news_article_limit",
    "TRADINGAGENTS_GLOBAL_NEWS_ARTICLE_LIMIT": "global_news_article_limit",
    "TRADINGAGENTS_GLOBAL_NEWS_LOOKBACK_DAYS": "global_news_lookback_days",
    "TRADINGAGENTS_SEC_USER_AGENT": "sec_user_agent",
    "TRADINGAGENTS_INSIDER_LOOKBACK_DAYS": "insider_lookback_days",
    "TRADINGAGENTS_INSIDER_CACHE_HOURS": "insider_cache_hours",
    "TRADINGAGENTS_INSIDER_MAX_FILINGS": "insider_max_filings",
    "TRADINGAGENTS_BENCHMARK_TICKER":     "benchmark_ticker",
    "TRADINGAGENTS_BUY_POSITION_PCT":     "scheduler_buy_position_pct",
    "TRADINGAGENTS_OVERWEIGHT_POSITION_PCT": "scheduler_overweight_position_pct",
    "TRADINGAGENTS_MAX_POSITION_PCT":     "scheduler_max_position_pct",
    "TRADINGAGENTS_MAX_OPEN_POSITIONS":   "scheduler_max_open_positions",
    "TRADINGAGENTS_MIN_DAILY_VOLUME":     "scheduler_min_daily_volume",
    "TRADINGAGENTS_STRATEGY_VERSION": "strategy_version",
    "TRADINGAGENTS_STOP_LOSS_PCT": "stop_loss_pct",
    "TRADINGAGENTS_TAKE_PROFIT_PCT": "take_profit_pct",
    "TRADINGAGENTS_MAX_HOLD_TRADING_DAYS": "max_hold_trading_days",
    "TRADINGAGENTS_MANIPULATION_SELL_THRESHOLD": "manipulation_sell_threshold",
    "TRADINGAGENTS_LIMIT_SLIPPAGE_BPS": "limit_slippage_bps",
    "TRADINGAGENTS_SHADOW_SLIPPAGE_BPS": "shadow_slippage_bps",
    "TRADINGAGENTS_MARKET_SNAPSHOT_MAX_AGE_MINUTES": "market_snapshot_max_age_minutes",
    "TRADINGAGENTS_MAX_QUOTE_SPREAD_PCT": "max_quote_spread_pct",
    "TRADINGAGENTS_DAILY_DRAWDOWN_LIMIT": "daily_drawdown_limit",
    "TRADINGAGENTS_WEEKLY_DRAWDOWN_LIMIT": "weekly_drawdown_limit",
    "TRADINGAGENTS_TOTAL_DRAWDOWN_LIMIT": "total_drawdown_limit",
    "TRADINGAGENTS_ALLOW_REAL_MONEY": "allow_real_money",
    "TRADINGAGENTS_EXPECTED_REAL_ACCOUNT_ID": "expected_real_account_id",
    "TRADINGAGENTS_MAX_REAL_MONEY_NOTIONAL": "max_real_money_notional",
    "TRADINGAGENTS_REAL_POSITION_PCT": "real_position_pct",
    "TRADINGAGENTS_REAL_MAX_OPEN_POSITIONS": "real_max_open_positions",
    "TRADINGAGENTS_REAL_MAX_EXPOSURE_PCT": "real_max_exposure_pct",
    "TRADINGAGENTS_RELEASE_REPORT_MAX_AGE_HOURS": "release_report_max_age_hours",
    "TRADINGAGENTS_SCORECARD_HORIZON_DAYS": "scorecard_horizon_days",
    "TRADINGAGENTS_SCORECARD_STOP_LOSS_PCT": "scorecard_stop_loss_pct",
    "TRADINGAGENTS_SCORECARD_WARMUP_POSITION_PCT": "scorecard_warmup_position_pct",
    "TRADINGAGENTS_SCORECARD_TIER1_POSITION_PCT": "scorecard_tier1_position_pct",
    "TRADINGAGENTS_SCORECARD_TIER2_POSITION_PCT": "scorecard_tier2_position_pct",
    "TRADINGAGENTS_SCORECARD_MIN_RESOLVED_DECISIONS": "scorecard_min_resolved_decisions",
    "TRADINGAGENTS_SCORECARD_TIER2_MIN_DECISIONS": "scorecard_tier2_min_decisions",
}


def _coerce(value: str, reference):
    """Coerce env-var string to the type of the existing default value."""
    if isinstance(reference, bool):
        return value.strip().lower() in ("true", "1", "yes", "on")
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


def _apply_env_overrides(config: dict) -> dict:
    """Apply TRADINGAGENTS_* env vars to the config dict in-place."""
    for env_var, key in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        config[key] = _coerce(raw, config.get(key))
    return config


DEFAULT_CONFIG = _apply_env_overrides({
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", os.path.join(_TRADINGAGENTS_HOME, "logs")),
    "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR", os.path.join(_TRADINGAGENTS_HOME, "cache")),
    "memory_log_path": os.getenv("TRADINGAGENTS_MEMORY_LOG_PATH", os.path.join(_TRADINGAGENTS_HOME, "memory", "trading_memory.md")),
    "real_money_validation_report": os.getenv(
        "TRADINGAGENTS_REAL_MONEY_VALIDATION_REPORT",
        os.path.join(_TRADINGAGENTS_HOME, "real_money_validation.json"),
    ),
    # Optional cap on the number of resolved memory log entries. When set,
    # the oldest resolved entries are pruned once this limit is exceeded.
    # Pending entries are never pruned. None disables rotation entirely.
    "memory_log_max_entries": None,
    # LLM settings
    "llm_provider": "groq",
    "deep_think_llm": "openai/gpt-oss-120b",
    "quick_think_llm": "openai/gpt-oss-20b",
    "secondary_llm_provider": "none",
    "secondary_deep_think_llm": "gpt-oss-120b",
    "secondary_quick_think_llm": "gpt-oss-120b",
    # When None, each provider's client falls back to its own default endpoint
    # (api.openai.com for OpenAI, generativelanguage.googleapis.com for Gemini, ...).
    # The CLI overrides this per provider when the user picks one. Keeping a
    # provider-specific URL here would leak (e.g. OpenAI's /v1 was previously
    # being forwarded to Gemini, producing malformed request URLs).
    "backend_url": None,
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "groq_requests_per_minute": 1,
    "groq_max_output_tokens": 512,
    "groq_max_retries": 1,
    "groq_reasoning_effort": "low",
    "cerebras_requests_per_minute": 3,
    "cerebras_max_output_tokens": 1024,
    "cerebras_max_retries": 1,
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    # Checkpoint/resume: when True, LangGraph saves state after each node
    # so a crashed run can resume from the last successful step.
    "checkpoint_enabled": False,
    # Output language for analyst reports and final decision
    # Internal agent debate stays in English for reasoning quality
    "output_language": "English",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    "analyst_concurrency_limit": 1,
    # News / data fetching parameters
    # Increase for longer lookback strategies or to broaden macro coverage;
    # decrease to reduce token usage in agent prompts.
    "news_article_limit": 8,              # max articles per ticker (ticker-news)
    "global_news_article_limit": 5,       # max articles for global/macro news
    "global_news_lookback_days": 7,       # macro news lookback window
    # Official SEC Form 4 corporate-insider data settings.
    "sec_user_agent": "",
    "insider_lookback_days": 30,
    "insider_cache_hours": 12,
    "insider_max_filings": 20,
    # Low-risk scheduler execution defaults. The graph decides direction; these
    # caps decide how much capital a single AI decision is allowed to touch.
    "scheduler_buy_position_pct": 0.01,
    "scheduler_overweight_position_pct": 0.02,
    "scheduler_max_position_pct": 0.03,
    "scheduler_max_open_positions": 5,
    "scheduler_min_daily_volume": 500_000,
    "strategy_version": "trader-v2-sec-insider",
    "stop_loss_pct": -0.07,
    "take_profit_pct": 0.12,
    "max_hold_trading_days": 10,
    "manipulation_sell_threshold": 0.85,
    "limit_slippage_bps": 20,
    "shadow_slippage_bps": 10,
    "market_snapshot_max_age_minutes": 5,
    "max_quote_spread_pct": 0.02,
    "daily_drawdown_limit": 0.005,
    "weekly_drawdown_limit": 0.01,
    "total_drawdown_limit": 0.03,
    "allow_real_money": False,
    "expected_real_account_id": "",
    "max_real_money_notional": 0.0,
    "real_position_pct": 0.005,
    "real_max_open_positions": 1,
    "real_max_exposure_pct": 0.015,
    "release_report_max_age_hours": 24,
    "backtest_validation_report": os.getenv(
        "TRADINGAGENTS_BACKTEST_VALIDATION_REPORT",
        os.path.join(_TRADINGAGENTS_HOME, "backtest_validation.json"),
    ),
    # AI decision scorecard gate. These caps are intentionally lower than the
    # base scheduler sizes until the model proves positive alpha.
    "scorecard_horizon_days": 10,
    "scorecard_stop_loss_pct": -0.05,
    "scorecard_warmup_position_pct": 0.005,
    "scorecard_tier1_position_pct": 0.01,
    "scorecard_tier2_position_pct": 0.02,
    "scorecard_min_resolved_decisions": 30,
    "scorecard_tier2_min_decisions": 60,
    # Search queries used by get_global_news for macro headlines. Extend or
    # replace to broaden geographic / sector coverage.
    "global_news_queries": [
        "Federal Reserve interest rates inflation",
        "S&P 500 earnings GDP economic outlook",
        "geopolitical risk trade war sanctions",
        "ECB Bank of England BOJ central bank policy",
        "oil commodities supply chain energy",
    ],
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance
        "news_data": "yfinance",             # Options: alpha_vantage, yfinance
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
    # Benchmark for alpha calculation in the reflection layer.
    # ``benchmark_ticker`` (when set) overrides the suffix map for all
    # tickers; leave it None to use ``benchmark_map`` for auto-detection
    # based on the ticker's exchange suffix. SPY remains the US default
    # so the reflection label keeps reading "Alpha vs SPY" for US tickers
    # while non-US tickers get their regional index automatically.
    "benchmark_ticker": None,
    "benchmark_map": {
        ".NS":  "^NSEI",    # NSE India (Nifty 50)
        ".BO":  "^BSESN",   # BSE India (Sensex)
        ".T":   "^N225",    # Tokyo (Nikkei 225)
        ".HK":  "^HSI",     # Hong Kong (Hang Seng)
        ".L":   "^FTSE",    # London (FTSE 100)
        ".TO":  "^GSPTSE",  # Toronto (TSX Composite)
        ".AX":  "^AXJO",    # Australia (ASX 200)
        "":     "SPY",      # default for US-listed tickers (no suffix)
    },
})
