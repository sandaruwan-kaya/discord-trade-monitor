"""Market provider abstraction, TradingView adapter and fallback (offline)."""
import pytest

from tools import market_tools
from tools.market_tools import (
    BinanceUSProvider,
    FallbackMarketDataProvider,
    ProviderError,
    ProviderUnavailable,
    TradingViewMCPProvider,
)

from helpers import MINUTE_MS, T0


def tv_bars(n, start_ms=T0, step_ms=MINUTE_MS, base=100.0):
    return [
        {"t": (start_ms + i * step_ms) // 1000, "o": base + i, "h": base + i + 0.5,
         "l": base + i - 0.5, "c": base + i + 0.2, "v": 1.5}
        for i in range(n)
    ]


class FakeTV:
    def __init__(self, bars=None, success=True):
        self.bars = bars or tv_bars(10)
        self.success = success
        self.calls = []

    def __call__(self, tool, args):
        self.calls.append((tool, args))
        if not self.success:
            return {"success": False, "error": "auth expired"}
        if tool == "mcp-tv-get-ohlcv":
            return {"success": True, "bars": self.bars[-args["count"]:]}
        if tool == "mcp-tv-get-technicals-rating":
            return {"success": True, "data": {
                "oscillators": {"rsi": 29.0, "macd": -3.1, "macd_signal": -2.4, "adx": 26.2},
                "moving_averages": {"ema20": 101.0, "ema50": 102.0, "ema200": 103.0},
                "summary": {"recommendation": "SELL"},
            }}
        raise AssertionError(f"unexpected tool {tool}")


class FakeProvider:
    def __init__(self, name, price=None, error=None):
        self.name = name
        self.price = price
        self.error = error
        self.calls = 0

    def get_current_price(self, symbol):
        self.calls += 1
        if self.error:
            raise self.error
        return self.price


def test_tradingview_unconfigured_is_unavailable():
    with pytest.raises(ProviderUnavailable):
        TradingViewMCPProvider().get_current_price("BTCUSDT")


def test_tradingview_symbol_mapping():
    assert TradingViewMCPProvider.tv_symbol("BTCUSDT") == "BINANCE:BTCUSDT"
    assert TradingViewMCPProvider.tv_symbol("ETHUSDT") == "BINANCE:ETHUSDT"
    assert TradingViewMCPProvider.tv_symbol("COINBASE:BTCUSD") == "COINBASE:BTCUSD"


def test_tradingview_price_uses_ohlcv_not_quote_tool():
    tv = FakeTV()
    price = TradingViewMCPProvider(call_tool=tv).get_current_price("BTCUSDT")

    assert price == 109.2
    assert tv.calls == [("mcp-tv-get-ohlcv", {"symbol": "BINANCE:BTCUSDT", "interval": "1m", "count": 1})]


def test_tradingview_candles_converted_and_trimmed_from_start():
    tv = FakeTV(tv_bars(10))
    provider = TradingViewMCPProvider(call_tool=tv, now_ms=lambda: T0 + 9 * MINUTE_MS)

    candles = provider.get_candles("BTCUSDT", "1m", limit=3, start_ms=T0 + 4 * MINUTE_MS)

    assert [c["timestamp"] for c in candles] == [T0 + i * MINUTE_MS for i in (4, 5, 6)]
    assert candles[0] == {"timestamp": T0 + 4 * MINUTE_MS, "open": 104.0, "high": 104.5,
                          "low": 103.5, "close": 104.2, "volume": 1.5}
    assert tv.calls[0][1]["count"] == 7             # enough bars to reach start_ms


def test_tradingview_bar_count_capped_at_5000():
    tv = FakeTV()
    provider = TradingViewMCPProvider(call_tool=tv, now_ms=lambda: T0 + 10_000 * MINUTE_MS)
    provider.get_candles("BTCUSDT", "1m", 200, start_ms=T0)
    assert tv.calls[0][1]["count"] == 5000


def test_tradingview_error_payload_raises():
    with pytest.raises(ProviderError):
        TradingViewMCPProvider(call_tool=FakeTV(success=False)).get_current_price("BTCUSDT")


def test_tradingview_market_context_includes_indicators():
    tv = FakeTV(tv_bars(100, step_ms=15 * MINUTE_MS))
    ctx = TradingViewMCPProvider(call_tool=tv).get_market_context("ETHUSDT")

    assert ctx.provider == "tradingview_mcp"
    assert ctx.current_price == 199.2
    assert ctx.indicators["tv_rsi_15m"] == 29.0
    assert ctx.indicators["tv_recommendation_15m"] == "SELL"
    assert ctx.indicators["rsi14_15m"] == 100.0     # locally computed on rising closes


def test_fallback_uses_binance_when_tradingview_fails():
    tv = FakeProvider("tradingview_mcp", error=ProviderUnavailable("not authenticated"))
    binance = FakeProvider("binance_us", price=83305.0)
    logs = []
    chain = FallbackMarketDataProvider([tv, binance], cooldown_seconds=300,
                                       clock=lambda: 0.0, log=logs.append)

    assert chain.get_current_price("BTCUSDT") == 83305.0
    assert "tradingview_mcp.get_current_price failed" in logs[0]


def test_fallback_cooldown_skips_failed_provider_then_retries():
    now = [0.0]
    tv = FakeProvider("tradingview_mcp", error=ProviderError("rate limited"))
    binance = FakeProvider("binance_us", price=1.0)
    chain = FallbackMarketDataProvider([tv, binance], cooldown_seconds=300,
                                       clock=lambda: now[0], log=lambda *_: None)

    chain.get_current_price("BTCUSDT")
    chain.get_current_price("BTCUSDT")
    assert tv.calls == 1                             # skipped during cooldown

    now[0] = 301.0
    tv.error = None
    tv.price = 2.0
    assert chain.get_current_price("BTCUSDT") == 2.0  # recovered, preferred again


def test_fallback_all_fail_raises():
    chain = FallbackMarketDataProvider(
        [FakeProvider("a", error=ConnectionError()), FakeProvider("b", error=ConnectionError())],
        clock=lambda: 0.0, log=lambda *_: None
    )
    with pytest.raises(ProviderError):
        chain.get_current_price("BTCUSDT")


def test_binance_provider_wraps_existing_market_data(monkeypatch):
    seen = {}

    def fake_get_candles(symbol, interval, limit, start_ms, end_ms):
        seen.update(symbol=symbol, interval=interval, limit=limit, start_ms=start_ms)
        return [{"timestamp": T0, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 3}]

    monkeypatch.setattr(market_tools.market_data, "get_candles", fake_get_candles)
    monkeypatch.setattr(market_tools.market_data, "get_current_price", lambda s: 42.0)

    provider = BinanceUSProvider()
    assert provider.get_current_price("BTCUSDT") == 42.0
    assert provider.get_candles("BTCUSDT", "1m", 1000, T0)[0]["close"] == 1.5
    assert seen == {"symbol": "BTCUSDT", "interval": "1m", "limit": 1000, "start_ms": T0}


def test_default_chain_prefers_tradingview_then_binance():
    chain = market_tools.build_market_context_provider()
    assert [p.name for p in chain.providers] == ["tradingview_mcp", "binance_us"]


def test_rsi_and_ema():
    assert market_tools.rsi([float(i) for i in range(30)]) == 100.0
    assert market_tools.ema([1.0] * 25, 20) == pytest.approx(1.0)
    assert market_tools.ema([1.0] * 5, 20) is None
