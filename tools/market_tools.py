"""
Market-data provider abstraction.

    MarketDataProvider (protocol)
      ├── BinanceUSProvider        wraps existing market_data.py (fallback)
      ├── TradingViewMCPProvider   TradingView MCP via an injected tool caller
      └── FallbackMarketDataProvider  tries providers in order, with cooldown

Agents depend on this module, never on a vendor directly.

Candle format everywhere (same as market_data.py):
    {"timestamp": ms, "open", "high", "low", "close", "volume"}
"""
import time
from typing import Callable, Protocol

import config
import market_data
from schemas import MarketContext


class ProviderError(RuntimeError):
    pass


class ProviderUnavailable(ProviderError):
    """Provider is not configured / not authenticated."""


class MarketDataProvider(Protocol):
    name: str

    def get_current_price(self, symbol: str) -> float: ...

    def get_candles(
        self,
        symbol: str,
        interval: str = "15m",
        limit: int = 200,
        start_ms: int | None = None,
        end_ms: int | None = None
    ) -> list[dict]: ...

    def get_market_context(self, symbol: str) -> MarketContext: ...


INTERVAL_MINUTES = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "4h": 240, "1d": 1440, "1D": 1440,
}


#
# Local indicators (so every provider can supply basic context)
#

def ema(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None

    k = 2 / (period + 1)
    result = sum(values[:period]) / period

    for value in values[period:]:
        result = value * k + result * (1 - k)

    return result


def rsi(values: list[float], period: int = 14) -> float | None:
    if len(values) <= period:
        return None

    gains = []
    losses = []

    for prev, cur in zip(values, values[1:]):
        change = cur - prev
        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for gain, loss in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0

    return 100 - 100 / (1 + avg_gain / avg_loss)


def context_from_candles(provider, symbol, candles, indicators=None) -> MarketContext:
    """Build context from 15m candles (last 96 = 24h)."""
    if not candles:
        raise ProviderError(f"{provider}: no candles for {symbol}")

    closes = [c["close"] for c in candles]
    day = candles[-96:]

    computed = {
        "rsi14_15m": rsi(closes),
        "ema20_15m": ema(closes, 20),
        "ema50_15m": ema(closes, 50),
    }
    computed.update(indicators or {})

    first_open = day[0]["open"]

    return MarketContext(
        provider=provider,
        symbol=symbol,
        as_of_ms=candles[-1]["timestamp"],
        current_price=closes[-1],
        change_pct_24h=(closes[-1] - first_open) / first_open * 100 if first_open else None,
        high_24h=max(c["high"] for c in day),
        low_24h=min(c["low"] for c in day),
        indicators=computed
    )


#
# Binance.US (existing implementation, now the fallback)
#

class BinanceUSProvider:
    name = "binance_us"

    def get_current_price(self, symbol):
        return market_data.get_current_price(symbol)

    def get_candles(self, symbol, interval="15m", limit=200, start_ms=None, end_ms=None):
        return market_data.get_candles(symbol, interval, limit, start_ms, end_ms)

    def get_market_context(self, symbol):
        candles = self.get_candles(symbol, "15m", 100)
        return context_from_candles(self.name, symbol, candles)


#
# TradingView MCP
#

ToolCaller = Callable[[str, dict], dict]


class TradingViewMCPProvider:
    """
    TradingView MCP adapter.

    call_tool(tool_name, arguments) -> parsed JSON dict is injected.
    The runtime transport (MCP streamable-HTTP client with the app's
    own OAuth client + refresh token) is a later step; until it is
    configured this provider raises ProviderUnavailable and the
    fallback chain moves on to Binance.US.

    Exact-pair prices come from get_ohlcv, NOT the quote tool: the
    quote tool resolves BINANCE:BTCUSDT to the CRYPTO:BTCUSD index.
    """
    name = "tradingview_mcp"

    MAX_BARS = 5000

    def __init__(self, call_tool: ToolCaller | None = None, now_ms: Callable[[], int] | None = None):
        self._call_tool = call_tool
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))

    @staticmethod
    def tv_symbol(symbol: str) -> str:
        if ":" in symbol:
            return symbol
        return f"BINANCE:{symbol}"

    def _call(self, tool, args):
        if self._call_tool is None:
            raise ProviderUnavailable("TradingView MCP runtime client is not configured")

        payload = self._call_tool(tool, args)

        if not isinstance(payload, dict) or not payload.get("success", False):
            error = payload.get("error") if isinstance(payload, dict) else payload
            raise ProviderError(f"{tool} failed: {str(error)[:200]}")

        return payload

    def _ohlcv(self, symbol, interval, count):
        payload = self._call(
            "mcp-tv-get-ohlcv",
            {
                "symbol": self.tv_symbol(symbol),
                "interval": interval,
                "count": max(1, min(int(count), self.MAX_BARS)),
            }
        )

        return [
            {
                "timestamp": int(bar["t"]) * 1000,
                "open": float(bar["o"]),
                "high": float(bar["h"]),
                "low": float(bar["l"]),
                "close": float(bar["c"]),
                "volume": float(bar.get("v") or 0),
            }
            for bar in payload.get("bars", [])
        ]

    def get_current_price(self, symbol):
        candles = self._ohlcv(symbol, "1m", 1)

        if not candles:
            raise ProviderError(f"no bars for {symbol}")

        return candles[-1]["close"]

    def get_candles(self, symbol, interval="15m", limit=200, start_ms=None, end_ms=None):
        # TradingView has no start/end parameters: fetch enough bars
        # back from now, then trim to Binance-style semantics.
        count = limit

        if start_ms is not None:
            minutes = INTERVAL_MINUTES.get(interval, 1)
            span = (self._now_ms() - start_ms) // (minutes * 60_000) + 2
            count = max(limit, span)

        candles = self._ohlcv(symbol, interval, count)

        if start_ms is not None:
            candles = [c for c in candles if c["timestamp"] >= start_ms]

        if end_ms is not None:
            candles = [c for c in candles if c["timestamp"] <= end_ms]

        if start_ms is not None:
            return candles[:limit]

        return candles[-limit:]

    def get_market_context(self, symbol):
        candles = self._ohlcv(symbol, "15m", 100)

        indicators = {}

        try:
            tech = self._call(
                "mcp-tv-get-technicals-rating",
                {"symbol": self.tv_symbol(symbol), "interval": "15m"}
            ).get("data", {})

            osc = tech.get("oscillators", {})
            mas = tech.get("moving_averages", {})

            indicators = {
                "tv_rsi_15m": osc.get("rsi"),
                "tv_macd_15m": osc.get("macd"),
                "tv_macd_signal_15m": osc.get("macd_signal"),
                "tv_adx_15m": osc.get("adx"),
                "tv_ema20_15m": mas.get("ema20"),
                "tv_ema50_15m": mas.get("ema50"),
                "tv_ema200_15m": mas.get("ema200"),
                "tv_recommendation_15m": tech.get("summary", {}).get("recommendation"),
            }
        except ProviderError:
            # Indicators are optional context; bars are what matter.
            pass

        return context_from_candles(self.name, symbol, candles, indicators)


#
# Fallback chain
#

class FallbackMarketDataProvider:
    """
    Tries providers in order. A provider that fails is skipped for
    a cooldown period so a dead/unauthenticated provider does not
    add latency to every call.
    """

    def __init__(self, providers, cooldown_seconds=None, clock=time.monotonic, log=print):
        if not providers:
            raise ValueError("at least one provider required")

        self.providers = list(providers)
        self.cooldown_seconds = (
            config.provider_failure_cooldown_seconds()
            if cooldown_seconds is None else cooldown_seconds
        )
        self._clock = clock
        self._log = log
        self._failed_at: dict[str, float] = {}
        self.name = "fallback(" + ",".join(p.name for p in self.providers) + ")"

    def _available(self):
        now = self._clock()
        available = [
            p for p in self.providers
            if now - self._failed_at.get(p.name, float("-inf")) >= self.cooldown_seconds
        ]
        # If everything is cooling down, still try the last resort.
        return available or self.providers[-1:]

    def _run(self, method, *args, **kwargs):
        last_error = None

        for provider in self._available():
            try:
                result = getattr(provider, method)(*args, **kwargs)
                self._failed_at.pop(provider.name, None)
                return result
            except Exception as error:
                self._failed_at[provider.name] = self._clock()
                last_error = error
                # Type + truncated message only; never payloads/headers.
                self._log(
                    f"[MARKET] {provider.name}.{method} failed: "
                    f"{type(error).__name__}: {str(error)[:200]}"
                )

        raise ProviderError(f"all market providers failed for {method}") from last_error

    def get_current_price(self, symbol):
        return self._run("get_current_price", symbol)

    def get_candles(self, symbol, interval="15m", limit=200, start_ms=None, end_ms=None):
        return self._run("get_candles", symbol, interval, limit, start_ms, end_ms)

    def get_market_context(self, symbol):
        return self._run("get_market_context", symbol)


def build_provider(name: str, tradingview_call_tool: ToolCaller | None = None):
    if name == "binance_us":
        return BinanceUSProvider()

    if name == "tradingview_mcp":
        return TradingViewMCPProvider(call_tool=tradingview_call_tool)

    raise ValueError(f"unknown market provider: {name}")


def build_market_context_provider(tradingview_call_tool: ToolCaller | None = None):
    """
    Provider chain for agent context. TradingView MCP is included only when
    its runtime client is actually configured (the app's own OAuth client,
    never Claude Code's token); otherwise the chain is Binance.US only.
    """
    names = [
        n for n in config.market_context_providers()
        if n != "tradingview_mcp" or tradingview_call_tool is not None
    ] or ["binance_us"]
    return FallbackMarketDataProvider([build_provider(n, tradingview_call_tool) for n in names])
