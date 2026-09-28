"""Offline fakes shared by tests."""
from models import TradeSignal


MINUTE_MS = 60_000

# Minute-aligned epoch ms used as "signal time" in tests.
T0 = (1_790_000_000_000 // MINUTE_MS) * MINUTE_MS


def candle(ts, low, high, close=None, open_=None):
    close = close if close is not None else (low + high) / 2
    return {
        "timestamp": ts,
        "open": open_ if open_ is not None else close,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1.0,
    }


class FakeMarket:
    """
    Binance-like 1m feed: get_candles(start_ms) returns the FIRST
    `limit` candles at/after start_ms that have opened by now.
    """
    name = "fake"

    def __init__(self, path, now_ms, price=100.0, fail_price=False):
        self.path = path            # minute index -> (low, high, close)
        self.now_ms = now_ms
        self.price = price
        self.fail_price = fail_price
        self.calls = []

    def get_current_price(self, symbol):
        if self.fail_price:
            raise ConnectionError("price feed down")
        return self.price

    def get_candles(self, symbol, interval="1m", limit=200, start_ms=None, end_ms=None):
        self.calls.append({"start_ms": start_ms, "limit": limit})
        out = []
        minute = 0 if start_ms is None else max(0, (start_ms - T0) // MINUTE_MS)
        while len(out) < limit:
            ts = T0 + minute * MINUTE_MS
            if ts > self.now_ms:
                break
            low, high, close = self.path(minute)
            out.append(candle(ts, low, high, close))
            minute += 1
        return out


def bullish_signal(**overrides):
    data = dict(
        is_trade_signal=True,
        symbol="BTCUSDT",
        bias="conditional_bullish",
        timeframe="1h",
        summary="Break above 100.5 targets 102; below 98 invalidates.",
        support_levels=[99.0],
        resistance_levels=[100.5],
        targets=[102.0],
        confirmation_above=100.5,
        invalidation_below=98.0,
    )
    data.update(overrides)
    return TradeSignal(**data)


def bearish_signal(**overrides):
    data = dict(
        is_trade_signal=True,
        symbol="ETHUSDT",
        bias="bearish",
        timeframe="4h",
        summary="Lose 99.5 and it heads to 97; above 101 invalidates.",
        support_levels=[99.5],
        resistance_levels=[101.0],
        targets=[97.0],
        confirmation_below=99.5,
        invalidation_above=101.0,
    )
    data.update(overrides)
    return TradeSignal(**data)
