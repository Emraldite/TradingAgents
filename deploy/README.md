# Oracle VM deployment (recommended)

Use `uv` plus a user-level `systemd` service. This is lighter than Docker and
keeps the bot, SQLite state, logs, and backups easy to inspect. The checked-in
service is intentionally locked to Alpaca **paper mode** and three liquid tickers.

These commands assume Ubuntu and that your repository will live at `~/trader`.
Replace the GitHub URL before running them.

## 1. Install system tools and uv

```bash
sudo apt update
sudo apt install -y git curl build-essential libgomp1
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

The installer command comes from the official uv documentation. `uv` will use the
checked-in `.python-version` and can download that Python version when necessary.

## 2. Clone and reproduce the locked environment

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git ~/trader
cd ~/trader
uv sync --frozen
uv run pytest -q
```

`--frozen` makes deployment use `uv.lock` without silently changing it. Do not use
`pip install` or regenerate the lockfile on the VM.

## 3. Create the private VM configuration

```bash
cd ~/trader
cp .env.example .env
nano .env
chmod 600 .env
mkdir -p ~/.tradingagents ~/trader/logs ~/trader/backups
```

Set the Gemini and Alpaca **paper** keys. Keep all of these unchanged:

```dotenv
ALPACA_BASE_URL=https://paper-api.alpaca.markets
TRADINGAGENTS_ALLOW_REAL_MONEY=false
TRADINGAGENTS_MAX_REAL_MONEY_NOTIONAL=0
```

The checked-in free model pair is:

```dotenv
TRADINGAGENTS_DEEP_THINK_LLM=gemini-3.5-flash
TRADINGAGENTS_QUICK_THINK_LLM=gemini-3.1-flash-lite
```

The scheduler rejects paid or unknown hosted models and never switches models
after a quota error. If 3.5 Flash has no free quota, stop the service and manually
set both variables to `gemini-3.1-flash-lite`. This starts a distinct strategy
history; do not mix its evidence with the 3.5 Flash configuration.

Never copy your local `.env` into GitHub. Transfer its values privately or create
the VM file manually.

Do not paste or share the output of `docker compose config`: Compose expands the
values from `.env` into that diagnostic output.

## 4. Verify before enabling automation

```bash
cd ~/trader
uv run tradingagents broker-status
uv run tradingagents run-cycle --mode dry-run --tickers AAPL,MSFT,NVDA
uv run tradingagents health
```

Do not continue if the broker status points at a real-money endpoint, either
configured Gemini model has zero active free quota, or the dry run reports
configuration/data errors. Leave billing disabled; quota exhaustion should stop
the bot rather than create a charge.

## 5. Install the paper-only service

```bash
mkdir -p ~/.config/systemd/user
cp ~/trader/deploy/tradingagents.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now tradingagents
sudo loginctl enable-linger "$USER"
```

Check it without printing `.env`:

```bash
systemctl --user status tradingagents
journalctl --user -u tradingagents --since today
cd ~/trader && .venv/bin/tradingagents health
```

## Updating from GitHub

Stop the bot before changing its code or environment:

```bash
systemctl --user stop tradingagents
cd ~/trader
git pull --ff-only
uv sync --frozen
uv run pytest -q
systemctl --user start tradingagents
systemctl --user status tradingagents
```

If tests fail, do not restart the service. `git pull --ff-only` also refuses an
unexpected merge, making VM updates easier to reason about.

## Operations and recovery

```bash
cd ~/trader
.venv/bin/tradingagents health
.venv/bin/tradingagents halt-trading --reason "operator review"
.venv/bin/tradingagents backup-state --output-dir ~/trader/backups
journalctl --user -u tradingagents -f
```

Test restoring a backup before relying on it. The SQLite state under
`~/.tradingagents` and `.env` are VM-local and ignored by Git.

## Optional Docker smoke test

Docker is supported but is not the recommended Oracle VM setup. The default
Compose command performs one safe dry-run and exits:

```bash
docker compose build
docker compose run --rm tradingagents
```

To run paper mode interactively with the persisted named volume:

```bash
docker compose run --rm tradingagents run-bot --mode paper --tickers AAPL,MSFT,NVDA
```

Keep the systemd and Docker bots mutually exclusive so two processes cannot
compete for broker orders or maintain separate state ledgers.
