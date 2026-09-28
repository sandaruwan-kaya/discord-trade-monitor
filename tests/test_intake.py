"""Existing Discord -> SQLite -> Telegram intake flow (offline)."""
import asyncio
import json

import database
from agents.signal_agent import SignalAnalysisAgent
from models import TradeSignal
from signal_service import Services, process_discord_message

from helpers import T0, FakeMarket, bullish_signal


def services_for(signal, sent, market=None, **kw):
    return Services(
        parse_message=lambda text: signal,
        market=market or FakeMarket(lambda m: (99.9, 100.1, 100.0), now_ms=T0),
        notify=sent.append,
        log=lambda *a, **k: None,
        **kw
    )


def run(services, message_id="1001", text="BTC break above 100.5 -> 102, below 98 invalid"):
    return asyncio.run(process_discord_message(message_id, text, T0 + 15_000, services))


def test_bullish_signal_saved_and_telegram_sent():
    sent = []
    signal_id = run(services_for(bullish_signal(), sent))

    row = database.get_signal(signal_id)
    assert row["symbol"] == "BTCUSDT"
    assert row["status"] == "DEVELOPING"
    assert row["active"] == 1
    assert row["price_at_signal"] == 100.0
    assert row["timeframe"] == "1h"
    assert row["raw_message"].startswith("BTC break above")
    assert row["created_at"] and row["updated_at"]
    assert json.loads(row["targets"]) == [102.0]

    assert len(sent) == 1
    assert sent[0].startswith("🆕 NEW DISCORD TRADE SIGNAL")
    assert "Price when posted:\n100.0" in sent[0]
    assert "Ignored inconsistent targets" not in sent[0]

    events = database.get_signal_events(signal_id)
    assert events[0]["event_type"] == "CREATED"


def test_non_trade_message_ignored():
    sent = []
    result = run(services_for(TradeSignal(is_trade_signal=False), sent), text="gm everyone")

    assert result is None
    assert sent == []
    assert database.get_active_signals() == []


def test_trade_signal_without_symbol_ignored():
    sent = []
    result = run(services_for(TradeSignal(is_trade_signal=True, bias="bullish"), sent))

    assert result is None
    assert sent == []


def test_price_lookup_failure_still_saves_signal():
    sent = []
    market = FakeMarket(lambda m: (0, 0, 0), now_ms=T0, fail_price=True)
    signal_id = run(services_for(bullish_signal(), sent, market=market))

    row = database.get_signal(signal_id)
    assert row is not None
    assert row["price_at_signal"] is None
    assert "Price when posted:\nUnavailable" in sent[0]


def test_duplicate_discord_message_not_saved_twice():
    sent = []
    services = services_for(bullish_signal(), sent)

    assert run(services) is not None
    assert run(services) is None
    assert len(sent) == 1
    assert len(database.get_active_signals()) == 1


def test_inconsistent_target_flagged():
    sent = []
    signal_id = run(services_for(bullish_signal(targets=[99.0, 102.0]), sent))

    flags = json.loads(database.get_signal(signal_id)["target_flags"])
    assert flags[0]["target"] == 99.0
    assert "Ignored inconsistent targets: 99.0" in sent[0]


def test_telegram_failure_does_not_lose_signal():
    def broken(_):
        raise ConnectionError("telegram down")

    services = services_for(bullish_signal(), [])
    services.notify = broken

    signal_id = run(services)
    assert database.get_signal(signal_id) is not None


def test_agent1_disabled_by_default_runs_nothing():
    class Exploding:
        def analyze(self, _):
            raise AssertionError("Agent 1 must not run when disabled")

    signal_id = run(services_for(bullish_signal(), [], signal_agent=Exploding()))
    assert database.get_signal_analyses(signal_id) == []


def test_agent1_enabled_persists_analysis(monkeypatch):
    monkeypatch.setenv("AGENT1_ENABLED", "true")
    agent = SignalAnalysisAgent(llm=None, now_ms=lambda: T0 + 60_000)

    signal_id = run(services_for(bullish_signal(), [], signal_agent=agent))

    analyses = database.get_signal_analyses(signal_id)
    assert len(analyses) == 1
    assert analyses[0]["signal_state"] == "DEVELOPING"
    assert analyses[0]["next_step"] == "WAIT_FOR_CONFIRMATION"
    stored_input = json.loads(analyses[0]["input_json"])
    assert stored_input["raw_message"].startswith("BTC break above")
    assert stored_input["current_price"] == 100.0


def test_agent1_failure_does_not_break_intake(monkeypatch):
    monkeypatch.setenv("AGENT1_ENABLED", "true")

    class Broken:
        def analyze(self, _):
            raise RuntimeError("boom")

    sent = []
    signal_id = run(services_for(bullish_signal(), sent, signal_agent=Broken()))

    assert database.get_signal(signal_id) is not None
    assert len(sent) == 1
