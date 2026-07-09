# Legacy `.AGENTS` Archive - 2026-07-09

This file preserves the useful history from the accidental `.AGENTS/` folder before removing it. `.AGENT/` is the active memory folder.

## Profile / Setup
- User is a high school sophomore in Katy, TX pursuing CS at UT Austin.
- User is early in formal development, learning fast, and interested in AI plus quantitative finance.
- Setup: Windows, VS Code, uv + venv.
- Local `.env` has required API keys; actual key values must never be stored in memory.

## Preferences
- Be direct, no filler.
- Explain non-obvious decisions briefly.
- Keep solutions simple and avoid over-engineering.
- Handle errors explicitly.
- Confirm before adding dependencies.
- Always use uv for dependency management; never use bare pip.
- Never commit or push for the user.
- Do not start dev servers unless explicitly asked.
- Prefer low-cost LLM usage during analysis.
- Prefer Oracle VM deployment via git clone / git updates, while excluding logs, caches, DBs, venvs, and secrets.

## Older Architecture Decisions
- Forked TradingAgents directly rather than wrapping it as a dependency to control graph wiring and agent internals.
- Reddit and noisy social signals should be reject-only filters, not buy-signal generators.
- Congressional data can be cached for hours because disclosures do not change intraday.
- Alpaca execution should use direct HTTP instead of the SDK to avoid dependency conflicts.
- Scheduled trading should default to dry-run.
- Dry-run entries should not pollute real trade/performance records.
- Broker state is the source of truth for quantities and order status.
- Non-dry-run cycles should reconcile broker state, evaluate sells, then evaluate buys.
- Live entries should use marketable limits; stop-loss exits should prioritize immediacy.
- Native Alpaca stop orders are required before unattended live paper trading.

## Older Build History
- Built congressional trading, social manipulation detection, risk controls, Alpaca paper execution, backtesting, scheduler, and CLI status/replay commands.
- Added persistent SQLite strategy state, broker reconciliation, cooldowns, sell logic, corporate-action split handling, and native stop order tracking.
- Prior full-suite checkpoints reported passing test counts in the 231-255 range before later upstream/custom changes.
