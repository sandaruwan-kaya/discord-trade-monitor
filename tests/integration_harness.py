"""Offline harness wiring every real component; only market data, parser, LLMs and Telegram are fakes."""
import asyncio

from agents.options_agent import OptionsSelectionAgent
from agents.risk_agent import RiskAnalysisAgent
from agents.signal_agent import SignalAnalysisAgent
from broker.paper_broker import PaperBroker
from models import TradeSignal
from notifications import Notifier
from schemas import RiskLLMOutput
from signal_service import Services, process_discord_message, run_monitor_cycle, run_proposal_cycle
from tools.options_tools import MockDeribitChainProvider
from trading_pipeline import TradingPipeline

from helpers import MINUTE_MS, T0


BULL = TradeSignal(is_trade_signal=True, symbol="BTCUSDT", bias="conditional_bullish", timeframe="1h",
                   summary="BTC break above 84,000 targets 86,000; below 82,600 invalidates.",
                   support_levels=[83000.0], resistance_levels=[84000.0], targets=[86000.0],
                   confirmation_above=84000.0, invalidation_below=82600.0)

BEAR = TradeSignal(is_trade_signal=True, symbol="ETHUSDT", bias="conditional_bearish", timeframe="4h",
                   summary="ETH lose 2,620 targets 2,540; above 2,720 invalidates.",
                   support_levels=[2620.0], resistance_levels=[2700.0], targets=[2540.0],
                   confirmation_below=2620.0, invalidation_above=2720.0)


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


class PriceFeed:
    """Binance-like feed driven by a per-minute price schedule (price holds until changed)."""

    def __init__(self, clock, prices, width):
        self.clock = clock
        self.schedule = {}          # symbol -> {minute: price}
        for symbol, p in prices.items():
            self.schedule[symbol] = {0: p}
        self.width = width
        self.fail = False

    def set(self, symbol, minute, price):
        self.schedule[symbol][minute] = price

    def price_at(self, symbol, minute):
        points = self.schedule[symbol]
        return points[max(m for m in points if m <= minute)]

    def get_current_price(self, symbol):
        if self.fail:
            raise ConnectionError("SIMULATED market data outage")
        return self.price_at(symbol, (self.clock.now - T0) // MINUTE_MS)

    def get_candles(self, symbol, interval="1m", limit=200, start_ms=None, end_ms=None):
        if self.fail:
            raise ConnectionError("SIMULATED market data outage")
        now_minute = (self.clock.now - T0) // MINUTE_MS
        if start_ms is None:
            minutes = range(max(0, now_minute - limit + 1), now_minute + 1)
        else:
            first = max(0, (start_ms - T0) // MINUTE_MS)
            minutes = range(first, min(now_minute, first + limit - 1) + 1)
        w = self.width[symbol]
        return [{"timestamp": T0 + m * MINUTE_MS, "open": self.price_at(symbol, m),
                 "high": self.price_at(symbol, m) + w, "low": self.price_at(symbol, m) - w,
                 "close": self.price_at(symbol, m), "volume": 1.0} for m in minutes]


class ScriptedLLM:
    def __init__(self, recommendation="HOLD"):
        self.recommendation = recommendation
        self.calls = 0

    def structured(self, system, user, schema):
        self.calls += 1
        codes = {"HOLD": ["THESIS_INTACT"], "REDUCE": ["THETA_DECAY"], "EXIT": ["THESIS_DETERIORATING"]}
        return RiskLLMOutput(recommendation=self.recommendation, confidence=0.7,
                             reason_codes=codes[self.recommendation], rationale="scripted")


class Harness:

    def __init__(self, risk_llm=None, clock=None, feed=None, send=None, telegram=None):
        self.clock = clock or Clock(T0 + 15_000)
        self.feed = feed or PriceFeed(self.clock, {"BTCUSDT": 83400.0, "ETHUSDT": 2650.0},
                                      width={"BTCUSDT": 20.0, "ETHUSDT": 1.0})
        self.parsed = {}
        self.legacy_telegram = telegram if telegram is not None else []
        self.trade_messages = []
        self.chain = MockDeribitChainProvider(self.feed, now_ms=self.clock)
        self.broker = PaperBroker(self.chain, now_ms=self.clock, log=lambda *a: None)
        self.risk_llm = risk_llm
        self.services = Services(
            parse_message=lambda text: self.parsed[text],
            market=self.feed,
            notify=self.legacy_telegram.append,
            context_market=self.feed_context(),
            signal_agent=SignalAnalysisAgent(None, now_ms=self.clock, log=lambda *a: None),
            option_chain=self.chain,
            options_agent=OptionsSelectionAgent(None, log=lambda *a: None),
            now_ms=self.clock,
            log=lambda *a, **k: None,
        )
        self.logs = []
        self.pipeline = TradingPipeline(
            signal_services=self.services, broker=self.broker, chain_provider=self.chain, market=self.feed,
            risk_agent=RiskAnalysisAgent(risk_llm, log=lambda *a: None),
            notifier=Notifier(send or self.trade_messages.append, log=lambda *a: None, now_ms=self.clock),
            now_ms=self.clock, log=self.logs.append,
        )

    def feed_context(self):
        from tools.market_tools import BinanceUSProvider, FallbackMarketDataProvider

        feed = self.feed

        class Context(BinanceUSProvider):
            name = "fake_binance_us"

            def get_current_price(self, symbol):
                return feed.get_current_price(symbol)

            def get_candles(self, *a, **k):
                return feed.get_candles(*a, **k)

        return FallbackMarketDataProvider([Context()], log=lambda *a: None)

    def post(self, message_id, signal, text=None):
        text = text or f"message {message_id}"
        self.parsed[text] = signal
        return asyncio.run(process_discord_message(message_id, text, self.clock.now, self.services))

    def at(self, minutes):
        self.clock.now = T0 + int(minutes * MINUTE_MS)

    def cycle(self):
        asyncio.run(run_monitor_cycle(self.services))
        finished = asyncio.run(run_proposal_cycle(self.services))
        self.pipeline.after_proposals(finished)
        self.pipeline.entry_cycle()
        self.pipeline.risk_cycle()
        self.pipeline.lifecycle_cycle()
        self.pipeline.exit_cycle()
