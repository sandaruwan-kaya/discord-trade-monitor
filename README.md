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
venv/bin/python scripts/agent2_demo.py BTCUSDT bullish 85200 82600
TRADE_MONITOR_DB=/tmp/paper_demo.db venv/bin/python scripts/paper_broker_demo.py   # scratch DB only
TRADE_MONITOR_DB=/tmp/policy_demo.db venv/bin/python scripts/policy_demo.py        # scratch DB only
TRADE_MONITOR_DB=/tmp/execution_demo.db venv/bin/python scripts/execution_demo.py  # scratch DB only
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
| `TRADING_KILL_SWITCH` | `0` | `1` blocks all NEW entries at the Policy Engine (exits are exempt by design) |
| `KILL_SWITCH_FILE` | project `KILL_SWITCH` | If this file exists, new entries are blocked (`touch KILL_SWITCH` / `rm KILL_SWITCH`) |
| `EXECUTION_CLAIM_TIMEOUT_SECONDS` | `120` | A CLAIMED/SUBMITTING execution older than this is reconciled by client_order_id (never re-submitted) |
| `TRADING_MODE` | `paper` | Only `paper` exists; any other value makes the broker refuse to start |
| `TRADE_MONITOR_DB` | project `trade_monitor.db` | DB path override |
| `AGENT2_ENABLED` | `false` | Create paper option proposals on first WORKING transition |
| `AGENT2_MAX_CONFIRMATION_AGE_MINUTES` | `15` | WORKING detected later than this after confirmation: no proposal |
| `AGENT2_MAX_UNDERLYING_MOVE_PCT` | `0.5` | Chase protection: price already beyond entry by more than this % in the trade direction → SKIPPED (`stale_entry_move_extended`); exactly equal is allowed |
| `AGENT2_MAX_LLM_CANDIDATES` | `8` | Pre-filtered candidates shown to Agent 2 |
| `AGENT2_PENDING_TIMEOUT_MINUTES` | `10` | Interrupted PENDING claims become FAILED (never retried) |
| `ALLOWED_UNDERLYINGS` | `BTCUSDT,ETHUSDT` | Underlyings eligible for option proposals |
| `MAX_PREMIUM_PER_TRADE`, `MAX_CONTRACTS`, `MIN_DTE`, `MAX_DTE`, `MIN_ABS_DELTA`, `MAX_ABS_DELTA`, `MAX_BID_ASK_SPREAD_PCT`, `MIN_OPEN_INTEREST`, `MIN_OPTION_VOLUME`, `MAX_QUOTE_AGE_SECONDS`, `OPTION_FEE_PER_CONTRACT` | `500, 2, 7, 45, 0.35, 0.65, 10, 100, 10, 60, 0` | Paper-trading demo playbook limits |
| `MOCK_OPTION_{MULTIPLIER,STRIKE_STEP,TICK,BASE_IV,PEAK_OI}_{BTC,ETH}` | BTC `0.01/1000/5/0.50/3000`, ETH `0.1/25/0.5/0.65/5000` | Mock Deribit-style chain |
