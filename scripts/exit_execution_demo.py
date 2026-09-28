"""
Exit execution demo on a SCRATCH database (never the real one). Paper only, no LLM.

For each scenario: entry via ExecutionService -> Agent 4 trade -> Agent 3
hard EXIT (SIMULATED drop through invalidation) -> Agent 4 ExitRequest ->
ExecutionService.execute_exit().

    TRADE_MONITOR_DB=/tmp/exit_demo.db python scripts/exit_execution_demo.py
"""
import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import database
from agents.risk_agent import RiskAnalysisAgent
from agents.trade_manager import TradeManager
from broker.factory import create_broker
from core.execution_service import ExecutionService
from risk_service import RiskService
from tools.options_tools import MockDeribitChainProvider

from execution_demo import make_proposal                         # noqa: E402
from risk_demo import Clock, ShiftableMarket                     # noqa: E402
from trade_manager_demo import FlakyQuotes                       # noqa: E402


def q(sql, params=()):
    conn = sqlite3.connect(config.db_path())
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params)]
    finally:
        conn.close()


class Env:
    def __init__(self):
        self.clock = Clock()
        self.market = ShiftableMarket()
        self.chain = MockDeribitChainProvider(self.market, now_ms=self.clock)
        self.quotes = FlakyQuotes(self.chain)
        self.broker = create_broker(self.quotes, now_ms=self.clock, log=lambda *a: None)
        self.svc = ExecutionService(self.broker, self.quotes, self.market, now_ms=self.clock, log=lambda *a: None)
        self.tm = TradeManager(now_ms=self.clock, log=lambda *a: None)
        self.risk = RiskService(RiskAnalysisAgent(None, log=lambda *a: None), self.quotes, self.market, now_ms=self.clock)

    def open_trade_with_exit_request(self, signal_id):
        self.market.shift_pct = 0.0
        p = make_proposal(signal_id, self.market, self.chain, marketable=True)
        entry = self.svc.execute_entry(p.proposal_key)
        trade = self.tm.materialize_trade(entry.execution_key)
        invalid = database.get_signal(signal_id)["invalidation_below"]
        self.market.shift_pct = (invalid - 50) / self.market.base * 100 - 100      # SIMULATED drop
        self.clock.offset_ms += 60_000
        a = self.risk.assess_position(trade["position_id"])
        d = self.tm.process(trade["trade_id"])
        [xr] = [r for r in self.tm.get_exit_requests(trade["trade_id"]) if r.requested_action == "EXIT"]
        print(f"  trade {trade['trade_id']}: Agent 3 {a.recommendation} {a.hard_rule_codes} data={a.data_issue_codes} "
              f"-> Agent 4 {d.next_state}, exit request {xr.exit_request_id} qty={xr.requested_quantity}")
        return trade, xr


def show(r):
    print(f"  -> execute_exit: status={r.status} replay={r.replay} execution={r.execution_id} "
          f"policy={r.policy_decision_id} reasons={r.policy_reasons} close_order={r.broker_order_id} error={r.error_code}")


def lifecycle(env, trade_id):
    return " -> ".join([e["to_state"] for e in env.tm.get_events(trade_id)])


def main():
    if config.db_path() == str(config.PROJECT_DIR / "trade_monitor.db"):
        sys.exit("refusing to run against the real database; set TRADE_MONITOR_DB")
    if os.path.exists(config.db_path()):
        os.remove(config.db_path())
    database.initialize_database()
    env = Env()

    print("\n### A. hard EXIT -> valid bid -> SELL_TO_CLOSE -> CLOSED")
    ta, xa = env.open_trade_with_exit_request(1)
    quote = env.quotes.get_option_contract(q("SELECT contract_symbol FROM paper_positions WHERE position_id = ?",
                                             (ta["position_id"],))[0]["contract_symbol"])
    print(f"  fresh quote: bid {quote.bid} / ask {quote.ask} / mark {quote.mark:.1f}")
    ra = env.svc.execute_exit(xa.exit_request_id)
    show(ra)
    print(f"  trade state: {env.tm.get_trade(ta['trade_id'])['state']} | lifecycle: {lifecycle(env, ta['trade_id'])}")

    print("\n### B. hard EXIT + quote unavailable -> WAITING_FOR_QUOTE -> quote restored -> same execution closes")
    tb, xb = env.open_trade_with_exit_request(2)
    env.quotes.down = True
    rb = env.svc.execute_exit(xb.exit_request_id)
    show(rb)
    print(f"  trade state: {env.tm.get_trade(tb['trade_id'])['state']} | exit request: "
          f"{q('SELECT status, execution_status FROM exit_requests WHERE exit_request_id = ?', (xb.exit_request_id,))[0]} "
          f"| close orders: {len(q('SELECT 1 FROM paper_orders WHERE position_id = ? AND side = ?', (tb['position_id'], 'SELL_TO_CLOSE')))}")
    env.quotes.down = False
    env.clock.offset_ms += 60_000
    [rb2] = [r for r in env.svc.reconcile_pending_executions() if r.exit_request_id == xb.exit_request_id]
    show(rb2)
    print(f"  same execution: {rb2.execution_id == rb.execution_id} | trade state: {env.tm.get_trade(tb['trade_id'])['state']}")

    print("\n### C. kill switch enabled -> EXIT still allowed")
    tc, xc = env.open_trade_with_exit_request(3)
    os.environ["TRADING_KILL_SWITCH"] = "1"
    rc = env.svc.execute_exit(xc.exit_request_id)
    del os.environ["TRADING_KILL_SWITCH"]
    show(rc)
    snap = json.loads(q("SELECT policy_snapshot_json FROM policy_decisions WHERE decision_id = ?", (rc.policy_decision_id,))[0]["policy_snapshot_json"])
    print(f"  kill switch recorded in the EXIT policy snapshot: {snap['kill_switch']} | trade: {env.tm.get_trade(tc['trade_id'])['state']}")

    print("\n### D. same exit called twice -> same execution / same close order")
    rd = env.svc.execute_exit(xa.exit_request_id)
    show(rd)
    print(f"  same execution as A: {rd.execution_id == ra.execution_id}, same close order: {rd.broker_order_id == ra.broker_order_id}, "
          f"close orders on A's position: {len(q('SELECT 1 FROM paper_orders WHERE position_id = ? AND side = ?', (ta['position_id'], 'SELL_TO_CLOSE')))}")

    print("\n### E. SIMULATED crash after the broker accepted the close -> restart -> reconcile, no duplicate")
    te, xe = env.open_trade_with_exit_request(4)
    real = env.broker.close_position

    def close_then_crash(*a, **k):
        real(*a, **k)
        raise SystemExit("simulated crash")

    env.broker.close_position = close_then_crash
    try:
        env.svc.execute_exit(xe.exit_request_id)
    except SystemExit:
        pass
    env.broker.close_position = real
    key = f"exit:{te['position_id']}:full"
    print(f"  after crash: execution {q('SELECT status FROM execution_attempts WHERE execution_key = ?', (key,))[0]['status']}, "
          f"trade {env.tm.get_trade(te['trade_id'])['state']}, close orders "
          f"{len(q('SELECT 1 FROM paper_orders WHERE position_id = ? AND side = ?', (te['position_id'], 'SELL_TO_CLOSE')))}")
    env.clock.offset_ms += 5 * 60_000
    restarted = ExecutionService(create_broker(env.quotes, now_ms=env.clock, log=lambda *a: None), env.quotes,
                                 env.market, now_ms=env.clock, log=lambda *a: None)
    re_ = restarted.execute_exit(xe.exit_request_id)
    show(re_)
    print(f"  close orders after restart: "
          f"{len(q('SELECT 1 FROM paper_orders WHERE position_id = ? AND side = ?', (te['position_id'], 'SELL_TO_CLOSE')))} | "
          f"lifecycle: {lifecycle(env, te['trade_id'])}")

    print("\n### Persisted EXIT policy decision (A)")
    d = json.loads(q("SELECT decision_json FROM policy_decisions WHERE decision_id = ?", (ra.policy_decision_id,))[0]["decision_json"])
    print(json.dumps({k: d[k] for k in ("decision_id", "purpose", "approved", "decision", "reasons", "retryable",
                                         "exit_limit_price", "exit_quantity", "quote_age_seconds", "checked_at",
                                         "policy_snapshot")}, indent=2))

    print("\n### Persisted EXIT execution row (A)")
    print(json.dumps(q("SELECT execution_id, execution_key, purpose, status, exit_request_id, policy_decision_id, "
                       "client_order_id, broker_order_id, error_code, created_at, completed_at FROM execution_attempts "
                       "WHERE execution_id = ?", (ra.execution_id,))[0], indent=2))

    print("\n### Close broker order (A)")
    print(json.dumps(q("SELECT order_id, client_order_id, position_id, contract_symbol, side, quantity, limit_price, "
                       "status, fill_price, last_quote_bid, last_quote_ask, filled_at FROM paper_orders WHERE order_id = ?",
                       (ra.broker_order_id,))[0], indent=2))

    print("\n### ExitRequest final status (A)")
    print(json.dumps(q("SELECT exit_request_id, request_key, requested_action, requested_quantity, hard_exit, "
                       "status, execution_id, execution_status, completed_at FROM exit_requests WHERE exit_request_id = ?",
                       (xa.exit_request_id,))[0], indent=2))

    print("\n### Trade lifecycle history (A)")
    for e in env.tm.get_events(ta["trade_id"]):
        print(f"  {str(e['from_state']):<17} -> {e['to_state']:<19} {e['action']:<22} {json.loads(e['reason_codes_json'])}")

    print(f"\nTotals: BUY_TO_OPEN={len(q('SELECT 1 FROM paper_orders WHERE side = ?', ('BUY_TO_OPEN',)))} "
          f"SELL_TO_CLOSE={len(q('SELECT 1 FROM paper_orders WHERE side = ?', ('SELL_TO_CLOSE',)))} "
          f"exit executions={len(q('SELECT 1 FROM execution_attempts WHERE purpose = ?', ('EXIT',)))} "
          f"open positions={len(q('SELECT 1 FROM paper_positions WHERE status = ?', ('OPEN',)))}")


if __name__ == "__main__":
    main()
