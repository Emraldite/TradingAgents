# External Trading Repo Notes - 2026-07-09

## HKUDS/AI-Trader
- Useful ideas: live mark-to-market scoring, agent leaderboard, signal publishing, copy-trading API, broker/signal sync, multi-market support.
- Not a direct replacement for this repo: it is more of an agent-native trading platform/community service than a local Alpaca paper-trading strategy engine.
- Best idea to borrow later: a local scorecard that tracks every model/strategy decision out of sample before allowing larger position sizes.

## FinRL / FinRL-X
- Useful ideas: separate research/backtest/execution layers and keep portfolio weights as the shared interface.
- Best idea to borrow later: make scheduler output target weights, then let execution/risk code translate weights into orders.

## FinGPT / Finance-Specialized Models
- Useful ideas: finance-tuned models can help with sentiment extraction, filing/news classification, and event tagging.
- Do not rely on them as standalone traders. Use them as feature generators that feed deterministic risk and backtesting gates.

## Model Direction
- General frontier LLMs are still useful for synthesis and tool orchestration.
- Specialized finance models should be added first as optional sentiment/classification tools, not as the final trade decision-maker.
- The bot should improve through better data grounding, evals, paper-trade feedback, and risk constraints before chasing model swaps.
