# Decisions

- 2026-07-07: Project memory is stored in `.AGENT/` with a maintained `PRD.md`, per repo instructions.
- 2026-07-07: Manual ticker selection now defaults to broad mode instead of congressional-only gating, because congressional data should inform trades rather than block the entire universe.
- 2026-07-07: Scheduler entries now depend on the full analyst graph, with only `Buy` and `Overweight` treated as entry signals to keep execution logic simple.
- 2026-07-09: Upstream should be adopted selectively, not blindly merged, because upstream has valuable analysis/data fixes but does not include this fork's custom trading execution layer.
- 2026-07-09: Backported upstream stability ideas by implementing deterministic market snapshots, fixing the news prompt tool signature, and using complete graph router path maps.
- 2026-07-09: Low-risk defaults are now favored for scheduler execution: `Buy` uses 2%, `Overweight` uses 1%, and max single position defaults to 3% unless configured otherwise.
- 2026-07-09: Historical testing should start with no-foresight walk-forward simulation before trusting AI-generated signals live.
- Legacy import from `.AGENTS`: This project intentionally forked TradingAgents directly for full graph/source control, uses Reddit/social manipulation as a reject-only filter, keeps Alpaca execution on direct HTTP to avoid SDK dependency conflicts, reconciles broker state before buys, evaluates sells before buys, and requires native broker-side stops before unattended live paper trading.
