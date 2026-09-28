"""
Stale-entry / move-extension guard (chase protection) for Agent 2.

Rule: skip when the underlying has already moved MORE THAN
AGENT2_MAX_UNDERLYING_MOVE_PCT beyond the entry level in the trade
direction. Boundary: exactly equal to the threshold is ALLOWED.
Checked before the chain is fetched/ranked and before any LLM call.
"""
import asyncio

import pytest

import database
from agents.options_agent import OptionsSelectionAgent
from agents.signal_agent import SignalAnalysisAgent, build_input
from schemas import OptionsSelectionLLMOutput
from signal_service import Services, run_monitor_cycle, run_proposal_cycle
from tools.options_tools import (
    STALE_ENTRY_REASON,
    MockDeribitChainProvider,
    check_entry_extension,
    entry_move_pct,
)

from helpers import MINUTE_MS, T0, FakeMarket, bearish_signal, bullish_signal


#
# Unit: formula + boundary
#

def test_bullish_formula_matches_spec_example():
    # confirmation_above 84,000, price 84,600 -> 0.714%
    assert entry_move_pct("bullish", 84000.0, 84600.0) == pytest.approx(0.7142857, abs=1e-6)


def test_bearish_formula_is_mirrored():
    assert entry_move_pct("bearish", 2620.0, 2600.0) == pytest.approx(0.7633588, abs=1e-6)
    assert entry_move_pct("bearish", 2620.0, 2640.0) < 0      # moved against: not extended


@pytest.mark.parametrize("direction,entry,price,allowed", [
    ("bullish", 84000.0, 84400.0, True),     # +0.476%
    ("bullish", 84000.0, 84420.0, True),     # +0.500% exactly -> allowed (boundary)
    ("bullish", 84000.0, 84420.01, False),   # just over
    ("bullish", 84000.0, 84600.0, False),    # +0.714%
    ("bullish", 84000.0, 83900.0, True),     # pulled back below entry
    ("bearish", 2620.0, 2610.0, True),       # +0.382%
    ("bearish", 2620.0, 2606.9, True),       # +0.500% exactly -> allowed (boundary)
    ("bearish", 2620.0, 2606.8, False),      # just over
    ("bearish", 2620.0, 2600.0, False),      # +0.763%
])
def test_threshold_and_boundary(direction, entry, price, allowed):
    ok, move, reason = check_entry_extension(direction, entry, price, 0.5)
    assert ok is allowed
    if not allowed:
        assert reason.startswith(STALE_ENTRY_REASON)


def test_missing_entry_reference_is_rejected():
    ok, move, reason = check_entry_extension("bullish", None, 84000.0, 0.5)
    assert ok is False and move is None
    assert reason.startswith(STALE_ENTRY_REASON)


#
# Sweep integration
#

class SpyLLM:
    def __init__(self):
        self.calls = 0

    def structured(self, system, user, schema):
        self.calls += 1
        return OptionsSelectionLLMOutput(decision="SELECT", selected_candidate_id="C1",
                                         ranking=["C1"], confidence=0.8, reason="rank 1")


class SpyChain(MockDeribitChainProvider):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.chain_calls = 0

    def get_option_chain(self, *a, **k):
        self.chain_calls += 1
        return super().get_option_chain(*a, **k)


BULL = bullish_signal(
    timeframe="1h", support_levels=[83000.0], resistance_levels=[84000.0],
    targets=[86000.0], confirmation_above=84000.0, invalidation_below=82600.0,
)
BEAR = bearish_signal(
    timeframe="4h", support_levels=[2620.0], resistance_levels=[2700.0],
    targets=[2540.0], confirmation_below=2620.0, invalidation_above=2720.0,
)


class World:
    """Signal posted at T0, confirms at minute 10, price then sits at `price`."""

    def __init__(self, signal, pre_price, price, width, agent1_state="DEVELOPING"):
        self.clock_ms = T0 + int(11.5 * MINUTE_MS)
        self.price = price

        def path(minute):
            p = pre_price if minute < 10 else self.price
            return (p - width, p + width, p)

        self.market = FakeMarket(path, now_ms=self.clock_ms, price=price)
        self.llm = SpyLLM()
        clock = lambda: self.clock_ms
        self.chain = SpyChain(self.market, now_ms=clock)
        self.services = Services(
            parse_message=None, market=self.market, notify=lambda _: None,
            signal_agent=SignalAnalysisAgent(None, now_ms=clock),
            option_chain=self.chain,
            options_agent=OptionsSelectionAgent(self.llm, log=lambda *a: None),
            now_ms=clock, log=lambda *a, **k: None,
        )

        database.save_signal("m1", signal, T0 + 15_000, pre_price, raw_message="raw")
        self.signal_id = database.get_signal_by_discord_message_id("m1")["id"]

        agent = SignalAnalysisAgent(None, now_ms=lambda: T0 + 20_000)
        inp = build_input(database.get_signal(self.signal_id), current_price=pre_price)
        database.save_signal_analysis(
            self.signal_id, inp, agent.analyze(inp).model_copy(update={"signal_state": agent1_state})
        )

    def tick(self):
        asyncio.run(run_monitor_cycle(self.services))
        asyncio.run(run_proposal_cycle(self.services))

    def advance(self, minutes, price=None):
        self.clock_ms = self.market.now_ms = T0 + int(minutes * MINUTE_MS)
        if price is not None:
            self.price = self.market.price = price

    def proposals(self):
        return database.get_proposals_for_signal(self.signal_id)


@pytest.fixture(autouse=True)
def agent2_on(monkeypatch):
    monkeypatch.setenv("AGENTIC_TRADING_ENABLED", "true")
    monkeypatch.setenv("AGENT2_ENABLED", "true")
    monkeypatch.setenv("AGENT2_MAX_UNDERLYING_MOVE_PCT", "0.5")


def bull_world(price, **kw):
    return World(BULL, 83400.0, price, 50.0, **kw)


def bear_world(price, **kw):
    return World(BEAR, 2650.0, price, 2.0, **kw)


# 1
def test_bullish_move_below_threshold_allowed():
    w = bull_world(84400.0)                                   # +0.476%
    w.tick()
    [p] = w.proposals()
    assert p["status"] == "PROPOSED"
    assert w.llm.calls == 1


# 2
def test_bullish_move_beyond_threshold_skipped():
    w = bull_world(84600.0)                                   # +0.714% (spec example)
    w.tick()
    [p] = w.proposals()
    assert p["status"] == "SKIPPED"
    assert p["reason"].startswith("stale_entry_move_extended")
    assert "+0.714%" in p["reason"]
    assert p["contract_symbol"] is None and p["output_json"] is None


# 3
def test_bearish_move_below_threshold_allowed():
    w = bear_world(2610.0)                                    # +0.382% in the bearish direction
    w.tick()
    [p] = w.proposals()
    assert p["status"] == "PROPOSED"
    assert p["strategy"] == "LONG_PUT"


# 4
def test_bearish_move_beyond_threshold_skipped():
    w = bear_world(2600.0)                                    # +0.763% in the bearish direction
    w.tick()
    [p] = w.proposals()
    assert p["status"] == "SKIPPED"
    assert p["reason"].startswith("stale_entry_move_extended")


# 5
def test_threshold_exactly_reached_is_allowed():
    w = bull_world(84420.0)                                   # exactly +0.500%
    w.tick()
    [p] = w.proposals()
    assert p["status"] == "PROPOSED"


def test_threshold_configurable(monkeypatch):
    monkeypatch.setenv("AGENT2_MAX_UNDERLYING_MOVE_PCT", "1.0")
    w = bull_world(84600.0)
    w.tick()
    assert w.proposals()[0]["status"] == "PROPOSED"


# 6
def test_skip_happens_before_chain_and_llm():
    w = bull_world(84600.0)
    w.tick()
    assert w.proposals()[0]["status"] == "SKIPPED"
    assert w.chain.chain_calls == 0                           # chain never fetched / ranked
    assert w.llm.calls == 0                                   # Agent 2 LLM never called


# 7
def test_skipped_stale_entry_is_never_retried():
    w = bull_world(84600.0)
    w.tick()
    assert w.proposals()[0]["status"] == "SKIPPED"

    # Price comes back into range; later ticks must not create a proposal.
    for minutes, price in ((12.5, 84200.0), (13.5, 84100.0), (20.5, 84050.0)):
        w.advance(minutes, price)
        w.tick()

    assert [p["status"] for p in w.proposals()] == ["SKIPPED"]
    assert w.llm.calls == 0 and w.chain.chain_calls == 0
    events = [e["event_type"] for e in database.get_signal_events(w.signal_id)]
    assert events.count("PROPOSAL_SKIPPED") == 1


def test_all_staleness_controls_remain_independent():
    # Within the move limit but confirmed 40 min earlier -> freshness gate still skips.
    w = bull_world(84200.0)
    w.advance(40.5)
    w.tick()
    [p] = w.proposals()
    assert p["status"] == "SKIPPED"
    assert "too late" in p["reason"]
