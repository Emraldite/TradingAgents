# TradingAgents Extended

An AI-assisted swing-trading research and execution system. It collects market,
news, social, fundamental, and official SEC Form 4 insider signals; asks a multi-agent
analysis graph for a rating; applies deterministic risk rules; and can submit
Alpaca orders.

This is an educational project, not financial advice or a proven profitable
strategy. Keep it in paper mode until every machine-checked release gate passes.

## Resume from another account

The `.agents/` memory and PRD files are local working notes and must never be
staged or committed. Read them before changing code when they exist. If you need
them on another machine, transfer them privately outside Git.

On the new account or machine:

```powershell
git clone <your-repository-url> trader
Set-Location trader
uv venv
uv pin
uv sync
Copy-Item .env.example .env
uv run pytest
git status --short
```

Copy the old machine's `.env` securely or fill the new one manually. Never send
credentials through chat or commit them. Local `.agents/` notes, runtime SQLite
databases, logs, and backups are intentionally not portable through Git; copy
them separately only if needed, and keep a backup before replacing local state.

### Current verified state (2026-07-20)

- Last recorded full verification: 363 tests passed and 1 optional live-API test
  was skipped. Run the suite again after cloning because this is a recorded
  checkpoint, not a guarantee about a new environment.
- Cerebras GPT-OSS 120B is the free hosted primary, with Groq GPT-OSS
  20B/120B as the secondary. Live calls require your own free-tier keys and
  cannot be verified by the offline test suite.
- The Oracle service omits user-systemd directives that caused exit
  `218/CAPABILITIES`, while retaining non-root execution, `NoNewPrivileges`, and
  `UMask=0077`.
- Review `git status` and `git diff --cached` before editing; do not discard
  deployment or provider changes that have not been committed yet.
- The project is currently paper-first. Real-money mode remains intentionally
  locked until all release gates pass.

## Free-first stack

- Cerebras GPT-OSS 120B as primary, with Groq GPT-OSS 20B/120B as secondary
- Gemini's free API tier or local Ollama as manual alternatives
- yfinance and public web sources for delayed/research data
- Alpaca's free paper-trading account for execution testing
- SQLite for the local order, fill, risk, health, and performance ledger
- Oracle Always Free as an optional paper-bot host

No paid software is required. Real trading is not literally cost-free: spreads,
slippage, regulatory fees, taxes, internet/electricity, and provider quota changes
can still create costs.

## How a cycle works

1. Select a small ticker universe from explicit tickers or the technical screen.
2. Reject missing/stale prices and unavailable manipulation data in broker modes.
3. Run the analyst graph and store its versioned Buy, Overweight, Hold,
   Underweight, or Sell rating.
4. Apply non-AI rules for position count, volume, exposure, scorecard confidence,
   and persistent daily/weekly/total loss limits.
5. Submit an idempotent bracket order: entry plus broker-native stop-loss and
   take-profit protection.
6. Reconcile WebSocket and REST broker updates into SQLite. Only confirmed fill
   increases alter positions or performance.
7. Halt safely when account state, protection, data, or configured limits are bad.

The AI proposes a direction. Deterministic code controls whether it may trade and
how much money it may touch.

## Setup

Use `uv`; do not install dependencies with bare `pip`.

```powershell
uv venv
uv pin
uv sync
Copy-Item .env.example .env
```

Fill in `GROQ_API_KEY`, `CEREBRAS_API_KEY`, `ALPACA_API_KEY`, and
`ALPACA_SECRET_KEY`. Replace
`TRADINGAGENTS_SEC_USER_AGENT` with an app identifier and your real contact email
so SEC requests comply with fair-access policy. Leave
`ALPACA_BASE_URL=https://paper-api.alpaca.markets` and all real-money locks at
their defaults. Never commit `.env`.

Keep both provider accounts on their free tiers. The scheduler accepts only the
checked-in GPT-OSS models, paces Cerebras at three requests per minute and Groq at
one, caps completions at 1,024 tokens, and allows one client retry. The primary and
secondary can be swapped through `.env`; set
`TRADINGAGENTS_SECONDARY_LLM_PROVIDER=none` to disable automatic failover.
Only a primary rate limit, timeout/connection failure, or 5xx error retries that
individual request on the secondary; request/authentication/oversize errors do not.
If both providers fail, the ticker fails and produces zero orders. These safeguards
prevent fallback to a paid model, but each provider controls its plans and limits,
so verify both accounts still have no billing enabled before each deployment.
Gemini and Ollama remain manual alternatives. NVIDIA NIM is not used because its
hosted access is an evaluation service with variable limits.
The insider analyst reads official SEC Form 4 XML, keeps only open-market purchase
and sale codes, discounts planned sales, excludes amendments, awards/options/gifts/tax events,
and treats missing SEC data as neutral. Its report is supporting evidence in the
bull/bear debate; deterministic risk code remains in control.
Free Alpaca market
data is [real-time IEX-only](https://docs.alpaca.markets/docs/about-market-data-api),
so the bot rejects stale or unusually wide IEX quotes;
it does not pretend that this free feed has full-market SIP coverage.

```powershell
uv run pytest
uv run tradingagents broker-status
uv run tradingagents health
```

## Safe progression

### 1. Dry-run

Analyzes data but neither simulates holdings nor contacts the broker for orders.

```powershell
uv run tradingagents run-cycle --mode dry-run --tickers AAPL
```

### 2. Shadow

Uses verified broker account value but keeps simulated fills separate from broker
fills. Use this to observe behavior without submitting orders.

```powershell
uv run tradingagents run-cycle --mode shadow --tickers AAPL
```

### 3. Paper

Submits actual orders to Alpaca's paper endpoint. The deprecated mode name `live`
is accepted only as a safe alias for `paper`; it never means real money.

```powershell
uv run tradingagents run-cycle --mode paper --tickers AAPL
uv run tradingagents run-bot --mode paper --tickers AAPL,NVDA,MSFT
```

The bot runs once immediately and then at 8:45 AM America/Chicago each weekday.
Keep the terminal open and use `Ctrl+C` for a clean stop. Use `--wait-first`,
`--daily-at HH:MM`, or `--interval MINUTES` when needed.

### 4. Evidence and release audit

Resolve forward outcomes, replay the stored decisions with fees/slippage against
buy-and-hold, and run the release audit:

```powershell
uv run tradingagents scorecard --resolve
uv run tradingagents replay-backtest --tickers AAPL,MSFT,NVDA,AMZN,GOOGL
uv run tradingagents release-audit
```

Real mode stays locked unless the current strategy has at least 100 resolved
directional decisions, positive forward alpha, controlled drawdown, at least 100
paper cycles spanning 90 days, an acceptable failure rate, a qualifying replay,
no unresolved cycles or critical health events, and no active/unprotected orders.
Changing a model or risk setting invalidates the report.

### 5. Real money (locked by default)

Only the legal account owner should ever configure this. Alpaca currently requires
an [individual applicant to be at least 18](https://alpaca.markets/support/requirements-alpaca-brokerage-account);
its [custodial documentation](https://docs.alpaca.markets/docs/custodial-accounts)
says a minor is only the beneficiary and cannot trade the account. Set a very small fixed
notional cap, exact account ID, real Alpaca endpoint, free model provider, and the
two validation-report paths in `.env`. A fresh approved audit and the exact phrase
`ENABLE REAL MONEY` are still required at runtime.

```powershell
uv run tradingagents run-cycle --mode real --confirm-real-money "ENABLE REAL MONEY"
```

Do not run real mode as an unattended service initially. The included deployment
template intentionally permits paper mode only.

## Operations

```powershell
uv run tradingagents health
uv run tradingagents halt-trading --reason "operator review"
uv run tradingagents acknowledge-health --note "reviewed and corrected"
uv run tradingagents resume-trading --confirmation "RESUME TRADING"
uv run tradingagents backup-state
```

The process lock prevents two bots from trading the same local state at once.
Critical errors remain in an acknowledgement audit trail. Backups are made with
SQLite's backup API and pass an integrity check before success is reported.

## Backtests

`walk-forward` tests the older deterministic technical proxy. `replay-backtest`
is the relevant validation command because it executes stored graph decisions on
the next bar using the same strategy exits and configured sizing rules as the bot.
One ticker is useful for inspection, but release validation requires at least five.

```powershell
uv run tradingagents walk-forward --ticker AAPL --start 2020-01-01 --end 2025-01-01
uv run tradingagents replay-backtest --ticker AAPL
```

Neither result guarantees future returns. Avoid selecting only the best ticker or
date range after seeing results; that is overfitting.

## Deployment

See [deploy/README.md](deploy/README.md) for the free Oracle VM paper-only service
template. Keep `.env`, databases, backups, caches, and logs outside Git and test a
backup restore before relying on unattended operation.
