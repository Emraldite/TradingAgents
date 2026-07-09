# Trader — PRD

## Overview
- Multi-agent trading research and paper-trading project built from a TradingAgents fork.
- Target users: the project owner learning trading systems and AI-assisted analysis.

## Tech Stack
- Frontend: CLI / terminal interface
- Backend: Python
- Other tools: yfinance, Alpaca paper trading, SQLite, LangGraph

## Features (alive — update as you build)

### Implemented
- Congressional trade watchlist ingestion and scoring
- Technical-screen-based automated cycle runner
- Social manipulation checks
- Portfolio risk controls and state tracking
- Interactive multi-agent analysis workflow
- Manual ticker selection that is no longer blocked by congressional watchlist data
- Congressional parser fallback for current non-table site layouts
- Scheduler entries now run through the full multi-analyst graph before trading
- Verified market-data snapshot tool for grounding exact market analyst claims
- Upstream prompt/router stability fixes for news analysis and graph flow
- Low-risk scheduler sizing defaults with env-configurable position/risk caps
- No-foresight walk-forward backtest CLI for compressed historical simulation

### TODO
- Verify live congressional ingestion against current upstream HTML in runtime
- Evaluate scorecard/copy-trading ideas from AI-Trader-style repos before adding social or leaderboard features
- Add a model/strategy scorecard gate before increasing live position size

## Architecture
- Scheduler and interactive analysis are separate paths, but scheduler entries now call the same full analyst graph before opening positions.
- Congressional data now acts as one signal source instead of being the default manual-ticker gate.
- Exact market-data claims should be grounded by deterministic tools, not left to the LLM.
- Historical tests should use walk-forward execution where signals are computed from prior data and executed on the next bar.

## API / Integrations
- Alpaca paper trading API
- yfinance
- Capitol Trades
- Quiver Quant
