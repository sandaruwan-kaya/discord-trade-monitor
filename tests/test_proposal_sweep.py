"""
Agent 2 trigger: first WORKING transition -> exactly one proposal.
All offline: fake market, fake chain, fake LLMs.
"""
import asyncio
import json
import sqlite3

import pytest

import config
import database
from agents.options_agent import OptionsSelectionAgent
from agents.signal_agent import SignalAnalysisAgent
from schemas import OptionsSelectionLLMOutput
from signal_service import Services, run_monitor_cycle, run_proposal_sweep
from tools.options_tools import MockDeribitChainProvider

from helpers import MINUTE_MS, T0, FakeMarket, bullish_signal


#
# Scenario: BTC signal at 83,400; break above 84,000 -> 86,000; below 82,600 invalid.
# Price confirms at minute 10.
#

SIGNAL = bullish_signal(
    timeframe="1h", support_levels=[83000.0], resistance_levels=[84000.0],
    targets=[86000.0], confirmation_above=84000.0, invalidation_below=82600.0,
)


def path(confirm_minute=10, after=84200.0):
    def f(minute):
        if minute < confirm_minute:
            return (83300.0, 83500.0, 83400.0)
        return (after - 100, after + 100, after)
    return f


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


class SelectFirst:
    def __init__(self):
        self.calls = 0

    def structured(self, system, user, schema):
        self.calls += 1
        return OptionsSelectionLLMOutput(decision="SELECT", selected_candidate_id="C1",
                                         ranking=["C1"], confidence=0.8, reason="rank 1")


@pytest.fixture(autouse=True)
def agent2_on(monkeypatch):
    monkeypatch.setenv("AGENT2_ENABLED", "true")


def build(now_minutes, confirm_minute=10, after=84200.0, llm=None):
    clock = Clock(T0 + now_minutes * MINUTE_MS)
    market = FakeMarket(path(confirm_minute, after), now_ms=clock.now, price=after)
    llm = llm or SelectFirst()
    services = Services(
        parse_message=None,
        market=market,
        notify=lambda _: None,
        signal_agent=SignalAnalysisAgent(None, now_ms=clock),
        option_chain=MockDeribitChainProvider(market, now_ms=clock),
        options_agent=OptionsSelectionAgent(llm, log=lambda *a: None),
        now_ms=clock,
        log=lambda *a, **k: None,
    )
    return services, clock, market, llm


def create_signal(agent1_state="DEVELOPING", message_id="m1"):
    database.save_signal(message_id, SIGNAL, T0 + 15_000, 83400.0, raw_message="raw")
    signal_id = database.get_signal_by_discord_message_id(message_id)["id"]
    if agent1_state:
        store_analysis(signal_id, agent1_state)
    return signal_id


def store_analysis(signal_id, state):
    from agents.signal_agent import build_input

    agent = SignalAnalysisAgent(None, now_ms=lambda: T0 + 20_000)
    inp = build_input(database.get_signal(signal_id), current_price=83400.0)
    analysis = agent.analyze(inp).model_copy(update={"signal_state": state})
    database.save_signal_analysis(signal_id, inp, analysis)


def tick(services):
    asyncio.run(run_monitor_cycle(services))


def advance(services, clock, market, minutes):
    clock.now = market.now_ms = T0 + minutes * MINUTE_MS


#
# Tests
#

def test_first_working_creates_exactly_one_proposal():
    signal_id = create_signal()
    services, clock, market, llm = build(now_minutes=10.5)   # minute 10 confirming candle still open

    tick(services)                                            # -> WORKING (in-progress candle)
    assert database.get_signal(signal_id)["status"] == "WORKING"
    assert database.get_proposals_for_signal(signal_id) == []  # waits for the closed confirming candle

    advance(services, clock, market, 11.5)
    tick(services)                                            # confirming candle closed -> claim + propose
    proposals = database.get_proposals_for_signal(signal_id)
    assert len(proposals) == 1
    p = proposals[0]
    assert p["status"] == "PROPOSED"
    assert p["proposal_key"] == f"signal:{signal_id}:entry"
    assert p["trigger"] == "FIRST_WORKING"
    assert p["strategy"] == "LONG_CALL"
    assert p["estimated_max_loss"] <= 500 and p["quantity"] <= 2
    output = json.loads(p["output_json"])
    assert output["side"] == "BUY_TO_OPEN"

    for m in (12.5, 13.5, 30.5):                              # later ticks: no new proposals, no new LLM calls
        advance(services, clock, market, m)
        tick(services)
    assert len(database.get_proposals_for_signal(signal_id)) == 1
    assert llm.calls == 1

    events = [e["event_type"] for e in database.get_signal_events(signal_id)]
    assert events.count("PROPOSAL_PROPOSED") == 1


def test_never_triggers_while_developing():
    signal_id = create_signal()
    services, *_ = build(now_minutes=9, confirm_minute=10)

    tick(services)

    assert database.get_signal(signal_id)["status"] == "DEVELOPING"
    assert database.get_proposals_for_signal(signal_id) == []


def test_working_back_to_working_does_not_retrigger():
    signal_id = create_signal()
    services, clock, market, llm = build(now_minutes=11.5)
    tick(services)
    first = database.get_signal(signal_id)["first_working_at"]

    database.update_status(signal_id, "DEVELOPING")
    database.update_status(signal_id, "WORKING")
    asyncio.run(run_proposal_sweep(services))

    assert database.get_signal(signal_id)["first_working_at"] == first
    assert len(database.get_proposals_for_signal(signal_id)) == 1
    assert llm.calls == 1


def test_claim_is_atomic():
    signal_id = create_signal()
    key = f"signal:{signal_id}:entry"
    assert database.claim_proposal(key, signal_id) is True
    assert database.claim_proposal(key, signal_id) is False

    conn = sqlite3.connect(config.db_path())
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO trade_proposals (proposal_key, signal_id, status, created_at, updated_at) VALUES (?, ?, 'PENDING', 'x', 'x')",
            (key, signal_id),
        )


def test_late_working_detection_is_skipped():
    # Confirmed at minute 10 but first seen at minute 40 (e.g. downtime catch-up).
    signal_id = create_signal()
    services, clock, market, llm = build(now_minutes=40.5)

    tick(services)

    p = database.get_proposals_for_signal(signal_id)[0]
    assert p["status"] == "SKIPPED"
    assert "too late" in p["reason"]
    assert llm.calls == 0


def test_freshness_window_configurable(monkeypatch):
    monkeypatch.setenv("AGENT2_MAX_CONFIRMATION_AGE_MINUTES", "60")
    signal_id = create_signal()
    services, *_ = build(now_minutes=40.5)

    tick(services)

    assert database.get_proposals_for_signal(signal_id)[0]["status"] == "PROPOSED"


def test_already_confirmed_fresh_is_allowed():
    signal_id = create_signal(agent1_state="ALREADY_CONFIRMED")
    services, *_ = build(now_minutes=11.5)

    tick(services)

    p = database.get_proposals_for_signal(signal_id)[0]
    assert p["status"] == "PROPOSED"
    assert p["trigger"] == "ALREADY_CONFIRMED_FRESH"


def test_already_confirmed_but_stale_now_is_skipped():
    # Price ran to 85,500 = 75% of 84,000 -> 86,000: fails the re-check.
    signal_id = create_signal(agent1_state="ALREADY_CONFIRMED")
    services, clock, market, llm = build(now_minutes=11.5, after=85500.0)

    tick(services)

    p = database.get_proposals_for_signal(signal_id)[0]
    assert p["status"] == "SKIPPED"
    assert "staleness re-check: STALE" in p["reason"]
    assert llm.calls == 0


@pytest.mark.parametrize("state", ["STALE", "NOT_TRADEABLE"])
def test_agent1_non_tradeable_states_skip(state):
    signal_id = create_signal(agent1_state=state)
    services, clock, market, llm = build(now_minutes=11.5)

    tick(services)

    p = database.get_proposals_for_signal(signal_id)[0]
    assert p["status"] == "SKIPPED"
    assert state in p["reason"]
    assert llm.calls == 0


def test_missing_agent1_analysis_runs_agent1_at_trigger():
    signal_id = create_signal(agent1_state=None)
    services, *_ = build(now_minutes=11.5)

    tick(services)

    assert len(database.get_signal_analyses(signal_id)) == 1
    p = database.get_proposals_for_signal(signal_id)[0]
    assert p["status"] == "PROPOSED"
    assert p["trigger"] == "ALREADY_CONFIRMED_FRESH"


def test_chain_failure_marks_failed_and_never_retries():
    signal_id = create_signal()
    services, clock, market, llm = build(now_minutes=11.5)

    class Broken:
        def get_option_chain(self, *a, **k):
            raise ConnectionError("chain down")

    services.option_chain = Broken()
    tick(services)
    p = database.get_proposals_for_signal(signal_id)[0]
    assert p["status"] == "FAILED"
    assert "ConnectionError" in p["error"]

    services.option_chain = MockDeribitChainProvider(market, now_ms=clock)
    advance(services, clock, market, 12.5)
    tick(services)
    assert [p["status"] for p in database.get_proposals_for_signal(signal_id)] == ["FAILED"]


def test_interrupted_pending_marked_failed_not_retried():
    signal_id = create_signal()
    key = f"signal:{signal_id}:entry"
    database.claim_proposal(key, signal_id)
    conn = sqlite3.connect(config.db_path())
    conn.execute("UPDATE trade_proposals SET created_at = '2000-01-01T00:00:00+00:00' WHERE proposal_key = ?", (key,))
    conn.commit()

    services, clock, market, llm = build(now_minutes=11.5)
    services.now_ms = lambda: 1_790_600_000_000          # real-ish "now" so the old claim is past the timeout
    asyncio.run(run_proposal_sweep(services))

    p = database.get_proposal(key)
    assert p["status"] == "FAILED"
    assert "interrupted" in p["error"]
    assert llm.calls == 0


def test_finish_never_overwrites_final_status():
    signal_id = create_signal()
    key = f"signal:{signal_id}:entry"
    database.claim_proposal(key, signal_id)
    assert database.finish_proposal(key, "SKIPPED", reason="x") is True
    assert database.finish_proposal(key, "PROPOSED", reason="y") is False
    assert database.get_proposal(key)["status"] == "SKIPPED"


def test_disabled_flag_runs_no_sweep(monkeypatch):
    monkeypatch.setenv("AGENT2_ENABLED", "false")
    signal_id = create_signal()
    services, *_ = build(now_minutes=11.5)

    tick(services)

    assert database.get_signal(signal_id)["status"] == "WORKING"
    assert database.get_proposals_for_signal(signal_id) == []


def test_agent2_never_receives_a_broker():
    import inspect
    import agents.options_agent as mod

    source = inspect.getsource(mod)
    assert "broker" not in source.lower().replace("no broker", "").replace("has no broker", "")
    params = inspect.signature(OptionsSelectionAgent.propose).parameters
    assert list(params) == ["self", "inp"]
