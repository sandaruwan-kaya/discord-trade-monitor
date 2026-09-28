"""
Agent 4 (trade lifecycle) demo on a SCRATCH database (never the real one).

Live Binance.US spot -> mock chain -> ExecutionService entries (paper) ->
Agent 4 materializes trades -> Agent 3 assessments -> Agent 4 lifecycle.
Agent 3's LLM answers for A/B are SCRIPTED (labelled) so the lifecycle
behaviour is deterministic; hard exits use the real deterministic rules.
Agent 4 never closes anything; step F's close is simulated demo setup.

    TRADE_MONITOR_DB=/tmp/trade_manager_demo.db python scripts/trade_manager_demo.py
"""
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import database
from agents.risk_agent import RiskAnalysisAgent
from agents.trade_manager import TradeManager
from broker.factory import create_broker
from core.execution_service import ExecutionService
from risk_service import RiskService
from schemas import RiskLLMOutput
from tools.options_tools import MockDeribitChainProvider

sys.path.insert(0, str(Path(__file__).resolve().parent))
from execution_demo import make_proposal                     # noqa: E402  (same proposal setup)
from risk_demo import Clock, ShiftableMarket                  # noqa: E402


class ScriptedLLM:
    """Stands in for Agent 3's contextual LLM so the demo is deterministic."""

    def __init__(self):
        self.next = "HOLD"
        self.calls = 0

    def structured(self, system, user, schema):
        self.calls += 1
        codes = {"HOLD": ["THESIS_INTACT"], "REDUCE": ["THETA_DECAY", "RISK_REWARD_DETERIORATED"]}[self.next]
        return RiskLLMOutput(recommendation=self.next, confidence=0.7, reason_codes=codes,
                             rationale=f"(scripted) {self.next}")


class FlakyQuotes:
    def __init__(self, provider):
        self.provider = provider
        self.down = False

    def get_option_contract(self, symbol):
        if self.down:
            raise ConnectionError("SIMULATED option quote outage")
        return self.provider.get_option_contract(symbol)


def step(label, tm, risk, trade_id, position_id):
    a = risk.assess_position(position_id)
    d = tm.process(trade_id)
    print(f"\n### {label}")
    print(f"Agent 3: {a.recommendation} ({a.risk_state}) hard={a.hard_rule_codes} data={a.data_issue_codes}")
    print(f"Agent 4: {d.previous_state} -> {d.next_state} action={d.action} codes={d.reason_codes}")
    for r in tm.get_exit_requests(trade_id):
        print(f"  exit_request {r.exit_request_id} {r.request_key} {r.requested_action} qty={r.requested_quantity} "
              f"hard={r.hard_exit} status={r.status} reasons={r.reason_codes} data={r.data_issue_codes}")
    return d


def main():
    if config.db_path() == str(config.PROJECT_DIR / "trade_monitor.db"):
        sys.exit("refusing to run against the real database; set TRADE_MONITOR_DB")
    if os.path.exists(config.db_path()):
        os.remove(config.db_path())
    database.initialize_database()

    clock = Clock()
    market = ShiftableMarket()
    chain = MockDeribitChainProvider(market, now_ms=clock)
    broker = create_broker(chain, now_ms=clock, log=lambda *a: None)
    execution = ExecutionService(broker, chain, market, now_ms=clock, log=lambda *a: None)
    tm = TradeManager(now_ms=clock, log=lambda *a: None)

    trades = []
    for signal_id in (1, 2):
        p = make_proposal(signal_id, market, chain, marketable=True)
        r = execution.execute_entry(p.proposal_key)
        t = tm.materialize_trade(r.execution_key)
        assert tm.materialize_trade(r.execution_key) == t            # idempotent
        trades.append(t)
        print(f"  execution {r.status} -> trade {t['trade_id']} state={t['state']} position={t['position_id']}")

    (t1, t2) = trades
    quotes = FlakyQuotes(chain)
    llm = ScriptedLLM()
    risk = RiskService(RiskAnalysisAgent(llm, log=lambda *a: None), quotes, market, now_ms=clock)

    invalid_below = database.get_signal(1)["invalidation_below"]

    clock.offset_ms = 30 * 60_000
    step("A. OPEN + healthy HOLD (scripted LLM)", tm, risk, t1["trade_id"], t1["position_id"])

    llm.next = "REDUCE"
    clock.offset_ms = 50 * 60_000
    step("B. WORKING + contextual REDUCE (scripted LLM)", tm, risk, t1["trade_id"], t1["position_id"])

    market.shift_pct = (invalid_below - 50) / market.base * 100 - 100
    clock.offset_ms = 55 * 60_000
    step("C. SIMULATED drop through invalidation -> hard EXIT", tm, risk, t1["trade_id"], t1["position_id"])

    quotes.down = True
    step("D. Trade 2: hard EXIT while the option quote is unavailable", tm, risk, t2["trade_id"], t2["position_id"])
    quotes.down = False

    clock.offset_ms = 56 * 60_000
    d = step("E. Trade 1: repeated EXIT assessment (new snapshot)", tm, risk, t1["trade_id"], t1["position_id"])
    print(f"  exit requests for trade 1: {len(tm.get_exit_requests(t1['trade_id']))} (no duplicate)")

    market.shift_pct = 0.0
    symbol = sqlite3.connect(config.db_path()).execute(
        "SELECT contract_symbol FROM paper_positions WHERE position_id = ?", (t1["position_id"],)).fetchone()[0]
    q = chain.get_option_contract(symbol)
    broker.close_position(t1["position_id"], q.bid, "demo-simulated-close")   # SIMULATED broker-side close
    d = tm.process(t1["trade_id"])
    print("\n### F. SIMULATED broker position CLOSED -> Agent 4 reconciles")
    print(f"Agent 4: {d.previous_state} -> {d.next_state} action={d.action} codes={d.reason_codes}")

    print("\n### Saved trade row (trade 1)")
    print(json.dumps(tm.get_trade(t1["trade_id"]), indent=2))
    print("\n### Saved ExitRequest (trade 1, full exit)")
    full = [r for r in tm.get_exit_requests(t1["trade_id"]) if r.requested_action == "EXIT"][0]
    print(full.model_dump_json(indent=2))
    print("\n### ExitRequest (trade 2, hard exit with data issue)")
    print(tm.get_exit_requests(t2["trade_id"])[0].model_dump_json(indent=2, include={
        "request_key", "requested_quantity", "hard_exit", "reason_codes", "data_issue_codes", "status"}))
    print("\n### Lifecycle event history (trade 1)")
    for e in tm.get_events(t1["trade_id"]):
        print(f"  {e['event_id']:>2} {str(e['from_state']):<17} -> {e['to_state']:<17} {e['action']:<15} "
              f"{json.loads(e['reason_codes_json'])} assessment={e['risk_assessment_id']} exit_request={e['exit_request_id']}")

    conn = sqlite3.connect(config.db_path())
    print(f"\nscripted Agent 3 LLM calls: {llm.calls} | paper_orders: "
          f"{conn.execute('SELECT side, COUNT(*) FROM paper_orders GROUP BY side').fetchall()} "
          "(2 entries via ExecutionService + 1 SIMULATED close in step F; Agent 4 submitted nothing)")


if __name__ == "__main__":
    main()
