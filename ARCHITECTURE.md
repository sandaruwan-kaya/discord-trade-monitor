# Architecture

Paper trading only. There is no live broker in this repository.

## Providers at a glance

| Concern | Provider | Notes |
|---|---|---|
| Discord message parsing | **OpenAI** (`llm_parser.py`) | unchanged |
| Agent 1 signal analysis | **OpenAI** (`llm_client.OpenAILLMClient`) | deterministic facts + validated LLM output |
| Agent 2 option selection | **OpenAI** | may only pick a pre-filtered candidate |
| Agent 3 risk analysis | **OpenAI** | hard rules first, LLM throttled |
| Agent 4 lifecycle | deterministic (no LLM) | |
| Underlying market data (primary) | **TradingView MCP** when `TRADINGVIEW_MCP_ENABLED=true` and authenticated | `BINANCE:BTCUSDT` / `BINANCE:ETHUSDT` via `mcp-tv-get-ohlcv` (these are **Binance.com** prices) |
| Underlying market data (fallback) | **Binance.US** (`market_data.py`) | always available; the only source when TradingView is disabled |
| Technical context for agents | TradingView technicals (`mcp-tv-get-technicals-rating`) + locally computed RSI/EMA | context only; never overrides deterministic confirmation / invalidation |
| Option chain (bid/ask/Greeks) | **MockDeribitChainProvider** | simulated, priced off the underlying feed; TradingView has no option data |
| Order execution | **PaperBroker** only, via `core/execution_service.py` | factory refuses any non-paper mode |

## Market-data chain

```
TradingViewMCPProvider --(failure / not authenticated / malformed data)--> BinanceUSProvider
          ^ retried after MARKET_PROVIDER_FAILURE_COOLDOWN_SECONDS (default 300)
```

* `tools/market_tools.build_market_provider(tradingview_client)` builds the chain (Binance.US only when the client is `None`).
* **Source consistency:** a unit of work that needs several calls (all candle pages of one signal evaluation; the Agent 3
  price + candles snapshot) runs through `run_consistent()`: one provider for the whole unit; on a mid-way failure the
  partial data is discarded and the whole unit re-runs on the fallback. Sources are never mixed within one evaluation.
  The provider used is recorded (`eval_state.market_source`, `STATUS_CHANGE.details.market_source`,
  `CREATED.details.price_source`).
* Historical signal state is never rewritten when the provider changes; evaluation simply continues from the cursor.
* TradingView payloads are validated: exact symbol match (never an aggregate `BTCUSD` for `BINANCE:BTCUSDT`),
  numeric/positive prices, high/low consistent with open/close, strictly increasing timestamps.
* Both providers down: market-dependent steps fail closed (no evaluation, `price_at_signal = NULL`, policy rejects with
  `underlying_price_unavailable` / `quote_unavailable`, Agent 3 `DATA_UNAVAILABLE`, exits `WAITING_FOR_QUOTE`).
  No price is ever invented and no order is placed.

## TradingView MCP runtime client

* `tradingview_mcp_client.py` - official MCP Python SDK (`mcp` 2.x, streamable HTTP), one short-lived connection per
  call, sync + thread-safe API, `401/403` -> one forced token refresh + retry, failures normalized to
  `ProviderUnavailable` (auth) / `ProviderError` (transport, tool error, malformed data).
* `tradingview_auth.py` - the app's **own** OAuth 2.1 public client: dynamic client registration, PKCE (S256)
  authorization-code flow, rotating refresh tokens. Claude Code's TradingView login is never read or reused.
* Credentials: `~/.config/discord-trade-monitor/tradingview/` (`client.json`, `tokens.json`), directory `0700`,
  files `0600`, atomic writes; override with `TRADINGVIEW_MCP_CONFIG_DIR`.
* Secrets (access/refresh tokens, authorization headers) are never logged, printed, put in exception messages or reprs.
* Setup: `venv/bin/python scripts/setup_tradingview_mcp.py` (or `--manual` to paste the redirect URL on a headless host).

## Trading pipeline

See README ("Paper-trading pipeline"): Discord -> signal intake -> Agent 1 -> evaluator -> Agent 2 proposal ->
fresh entry policy -> ExecutionService -> PaperBroker -> Trade -> Agent 3 -> Agent 4 -> ExitRequest -> exit policy ->
ExecutionService -> CLOSED. Every autonomous stage is behind its own flag (all default `false`).
