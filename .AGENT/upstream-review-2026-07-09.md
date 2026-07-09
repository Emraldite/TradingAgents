# Upstream Review - 2026-07-09

## Current State
- Local branch: `main`
- Local HEAD: `8bb94576fe5511beda00d34042e90060d23d7dbc`
- Upstream ref fetched as: `upstream/main`
- Upstream repo: `https://github.com/TauricResearch/TradingAgents.git`
- Ahead/behind from local HEAD to upstream: `1` ahead, `61` behind
- Latest upstream tag fetched: `v0.3.1`

## Recommendation
- Do not run a blind merge while the worktree has uncommitted custom trading changes.
- Do not replace the fork with upstream wholesale.
- Adopt upstream selectively, keeping this fork's custom trading layer: scheduler, Alpaca execution, risk controls, state store, alerts, congressional data, manipulation detection, and backtests.

## Why
- Upstream is now more of an analysis framework and has useful stability/data fixes.
- This fork has a trading execution layer that upstream does not have.
- A virtual merge reports conflicts in shared framework files, but local-only trading files should be preserved by a normal merge.

## Virtual Merge Conflicts
- `.env.example`
- `.gitignore`
- `README.md`
- `pyproject.toml`
- `tradingagents/agents/__init__.py`
- `tradingagents/agents/analysts/sentiment_analyst.py`
- `tradingagents/graph/trading_graph.py`
- `uv.lock` deleted upstream but modified locally

## Preserve From This Fork
- `tradingagents/scheduler/runner.py`
- `tradingagents/execution/alpaca_executor.py`
- `tradingagents/risk/survival_rules.py`
- `tradingagents/risk/performance_tracker.py`
- `tradingagents/state_store.py`
- `tradingagents/alerts.py`
- `tradingagents/dataflows/congressional_data.py`
- `tradingagents/dataflows/manipulation_detector.py`
- `backtests/backtest_runner.py`
- `tests/test_scheduler_gating.py`
- `tests/test_congressional_watchlist_empty.py`
- `tests/test_alpaca_executor.py`
- `tests/test_state_store.py`

## Highest-Value Upstream Changes To Adopt
- `d7b40a2` - fixes wrong-company hallucination by resolving instrument identity.
- `47cbb32` and `4e7821d` - adds verified market-data snapshot grounding.
- `9fd54f8` - rejects stale yfinance OHLCV instead of reporting wrong prices.
- `eeb84aa` and `308757c` - hardens Reddit and StockTwits data fetching.
- `622f99d` - fixes the news analyst prompt to match the actual tool signature.
- `3570f2e` - applies Alpha Vantage fundamentals look-ahead filtering.
- `b47a828` - fixes graph router path maps.
- `daf1da9` - makes checkpoint identity include graph shape and adds LLM retry budget.
- `517eeaf` - hardens structured output for local/openai-compatible servers.
- `0f70af2` - refreshes Claude model catalog support.

## Medium-Priority Upstream Changes
- Provider registry and extra providers: Bedrock, NVIDIA NIM, Kimi, Groq, Mistral.
- FRED macro indicators.
- Polymarket prediction-market data.
- Shared report writer.
- CI and ruff configuration.

## Suggested Merge Path
1. Finish and manually commit the current custom trading changes in VS Code.
2. Create a working branch such as `codex/upstream-v0.3.1-review`.
3. Merge `upstream/main` with conflicts resolved in favor of preserving this fork's trading layer.
4. Resolve the known conflict files one by one.
5. Run focused tests for scheduler, Alpaca executor, state store, congressional parser, analyst graph, and dataflows.
6. If the full merge is too noisy, cherry-pick the highest-value upstream commits instead of merging all 61.

## Commands Already Run
```powershell
git fetch https://github.com/TauricResearch/TradingAgents.git main:refs/remotes/upstream/main
git rev-list --left-right --count HEAD...upstream/main
git log --oneline --decorate HEAD..upstream/main
git merge-tree --write-tree --name-only HEAD upstream/main
```
