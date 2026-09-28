# Discord Trade Monitor

Discord signal intake → LLM parsing → SQLite → evaluator → Telegram,
evolving toward a **paper-trading-only** agentic options pipeline.

## Run the app

```bash
cd /home/ec2-user/discord-trade-monitor
source venv/bin/activate
python app.py
```

The database (`trade_monitor.db`, next to `app.py`) is migrated
automatically on startup (additive only).

## Tests (offline)

```bash
venv/bin/python -m pytest
```

Tests block all network access, use a temporary database and dummy
credentials; they never start Discord or call Telegram/OpenAI.

## Live smoke scripts (hit real services)

```bash
venv/bin/python scripts/smoke_llm.py
venv/bin/python scripts/smoke_telegram.py
venv/bin/python scripts/smoke_discord.py
venv/bin/python scripts/live_price_check.py
venv/bin/python scripts/agent1_demo.py "BTC break above 84,000 targets 85,200, invalid below 82,600"
```

## Configuration (.env)

| Variable | Default | Purpose |
|---|---|---|
| `AGENT1_ENABLED` | `false` | Run Agent 1 (signal analysis) after each new signal |
| `AGENT1_MAX_TARGET_PROGRESS_PCT` | `50` | Already-confirmed setups past this share of entry→target are STALE |
| `AGENT1_REQUIRE_INVALIDATION` | `true` | Signals without invalidation are NOT_TRADEABLE |
| `AGENT_MODEL` | `OPENAI_MODEL` | Model for agents |
| `SIGNAL_EXPIRY_HOURS_{SCALP,INTRADAY,SWING,POSITION,UNKNOWN}` | `6/24/120/720/48` | Signal expiry by timeframe |
| `MARKET_CONTEXT_PROVIDERS` | `tradingview_mcp,binance_us` | Agent market-context provider order |
| `MARKET_PROVIDER_COOLDOWN_SECONDS` | `300` | Skip a failed provider for this long |
| `MONITOR_CANDLE_PAGE_SIZE` / `MONITOR_MAX_CANDLE_PAGES` | `1000` / `10` | Candle catch-up paging |
| `TRADE_MONITOR_DB` | project `trade_monitor.db` | DB path override |
