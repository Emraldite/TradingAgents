# TradingAgents Extended

Swing trading research bot built from a fork of TradingAgents. This project adds congressional trading signals, social manipulation checks, portfolio risk controls, multi-analyst graph-driven entries, backtesting, and Alpaca paper trading.

This is a personal learning and portfolio project. It is not financial advice, and it should stay in paper trading until the strategy has been validated over time.

## What It Does

- Builds a congressional trade watchlist from Capitol Trades and Quiver pages.
- Scores tickers by conviction using recency, trade size, source overlap, and committee relevance.
- Runs the full analyst graph across congressional, market, sentiment, news, and fundamentals before opening new positions.
- Checks social manipulation risk with StockTwits activity signals.
- Applies basic survival rules before any order: position sizing, max open positions, volume filters, and kill switch.
- Sends paper orders to Alpaca through direct REST API calls.
- Tracks cycles and real paper-trade entries in SQLite.
- Keeps dry-run simulations separate from real trade records.

## Setup

Use `uv` for dependency management.

```powershell
uv sync
```

Create a local `.env` file with the keys you use:

```env
GOOGLE_API_KEY=
NEWS_API_KEY=
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

Do not commit `.env`.

## Verify The Project

Run the test suite:

```powershell
uv run pytest
```

Check Alpaca paper account, open positions, and recent orders:

```powershell
uv run tradingagents broker-status
```

Show more recent orders:

```powershell
uv run tradingagents broker-status --limit 20
```

## Trading Cycle

Dry-run mode is the default and does not write fake trades into the performance tracker:

```powershell
uv run python -c "import logging; logging.basicConfig(level=logging.INFO); from tradingagents.scheduler.runner import run_cycle; run_cycle(dry_run=True)"
```

Manual tickers bypass the congressional watchlist by default. This command will trade `AAPL` even if congressional data is empty:

```powershell
uv run python -c "import logging; logging.basicConfig(level=logging.INFO); from tradingagents.scheduler.runner import run_cycle; run_cycle(tickers=['AAPL'], dry_run=True)"
```

If you want the old congressional-only gate, explicitly turn the override off:

```powershell
uv run python -c "import logging; logging.basicConfig(level=logging.INFO); from tradingagents.scheduler.runner import run_cycle; run_cycle(tickers=['AAPL'], dry_run=True, allow_manual_tickers=False)"
```

Place real paper orders only after dry-run behavior looks sane:

```powershell
uv run python -c "import logging; logging.basicConfig(level=logging.INFO); from tradingagents.scheduler.runner import run_cycle; run_cycle(dry_run=False)"
```

Run a no-foresight historical simulation before trusting a signal live:

```powershell
uv run tradingagents walk-forward --ticker AAPL --start 2020-01-01 --end 2025-01-01 --position-pct 0.02
```

## CLI Analysis

Run the interactive multi-agent analyst workflow:

```powershell
uv run tradingagents analyze
```

Resume support is available with checkpoints:

```powershell
uv run tradingagents analyze --checkpoint
```

## Current Safety Defaults

- Scheduler defaults to `dry_run=True`.
- Manual tickers are allowed by default; set `allow_manual_tickers=False` if you want congressional-only gating.
- New entries require a graph rating of `Buy` or `Overweight`; `Hold`, `Underweight`, and `Sell` are skipped.
- Position sizing is intentionally low-risk by default: `Buy` = 2%, `Overweight` = 1%, max single position = 3%.
- Alpaca uses the paper endpoint from `ALPACA_BASE_URL`.
- Dry-runs log simulated counts but do not create fake trade entries.
- Real paper orders are tracked after Alpaca accepts the order.

## Project Status

Tests currently pass with one expected skipped live DeepSeek test when `DEEPSEEK_API_KEY` is not configured.

Next useful improvements:

- Add a first-class scheduler CLI command instead of one-line Python calls.
- Improve congressional data reliability if Quiver continues serving table rows through JavaScript.
- Add exit logic and sell-side tracking before unattended paper trading.
