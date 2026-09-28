"""
Execution Service demo on a SCRATCH database (never the real one).
Live Binance.US spot -> mock Deribit-style chain -> deterministic Agent 2
proposal (LLM disabled) -> ExecutionService -> fresh Policy Engine ->
PaperBroker. No Telegram, no live broker.

    TRADE_MONITOR_DB=/tmp/execution_demo.db python scripts/execution_demo.py
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
from agents.options_agent import OptionsSelectionAgent
from broker.factory import create_broker
from core.execution_service import ExecutionService
from models import TradeSignal
from schemas import OptionsSelectionInput, ThesisSnapshot
from tools.market_tools import BinanceUSProvider
from tools.options_tools import MockDeribitChainProvider, build_playbook, prefilter


class ShiftableSpot:
    def __init__(self):
        self.base = BinanceUSProvider().get_current_price("BTCUSDT")
        self.shift_pct = 0.0

    def get_current_price(self, symbol):
        return round(self.base * (1 + self.shift_pct / 100), 2)


def q(sql, params=()):
    conn = sqlite3.connect(config.db_path())
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params)]
    finally:
        conn.close()


def make_proposal(signal_id, spot, chain_provider, marketable):
    """Signal WORKING just above its entry, then Agent 2 (deterministic) -> PROPOSED row."""
    px = spot.get_current_price("BTCUSDT")
    entry = round(px - 100, -1)
    now = int(time.time() * 1000)
    signal = TradeSignal(is_trade_signal=True, symbol="BTCUSDT", bias="conditional_bullish", timeframe="1h",
                         support_levels=[entry - 500], resistance_levels=[entry], targets=[entry + 2000],
                         confirmation_above=entry, invalidation_below=entry - 1500, summary="demo")
    database.save_signal(f"demo-{signal_id}", signal, now - 5 * 60_000, px - 300, raw_message="demo")
    database.update_status(signal_id, "WORKING")

    thesis = ThesisSnapshot(underlying="BTCUSDT", direction="bullish", entry_level=entry, target=entry + 2000,
                            invalidation=entry - 1500, time_horizon="intraday", agent1_state="DEVELOPING")
    chain = chain_provider.get_option_chain("BTCUSDT", option_type="call")
    candidates, summary = prefilter(chain, thesis, build_playbook(), now)
    proposal = OptionsSelectionAgent(None, log=lambda *a: None).propose(OptionsSelectionInput(
        signal_id=signal_id, trigger="FIRST_WORKING", thesis=thesis, underlying_price=px, as_of_ms=now,
        playbook=build_playbook(), chain_source=chain.source, candidates=candidates, filter_summary=summary))
    if marketable:
        # Demo setup only: a marketable limit (= ask) so the paper fill happens immediately.
        proposal = proposal.model_copy(update={"limit_price": proposal.ask,
                                               "estimated_max_loss": proposal.ask * proposal.contract_multiplier * proposal.quantity})
    database.claim_proposal(proposal.proposal_key, signal_id)
    database.finish_proposal(proposal.proposal_key, "PROPOSED", proposal=proposal)
    print(f"  proposal {proposal.proposal_key}: {proposal.strategy} {proposal.contract_symbol} x{proposal.quantity} "
          f"limit {proposal.limit_price} (bid {proposal.bid} / ask {proposal.ask}) @ spot {px}")
    return proposal


def show(r):
    print(f"  -> status={r.status} replay={r.replay} policy={r.policy_decision_id} reasons={r.policy_reasons} "
          f"broker_order={r.broker_order_id} ({r.broker_order_status}) error={r.error_code}")


def main():
    if config.db_path() == str(config.PROJECT_DIR / "trade_monitor.db"):
        sys.exit("refusing to run against the real database; set TRADE_MONITOR_DB")
    if os.path.exists(config.db_path()):
        os.remove(config.db_path())
    database.initialize_database()

    spot = ShiftableSpot()
    chain_provider = MockDeribitChainProvider(spot)
    broker = create_broker(chain_provider, log=lambda *a: None)
    svc = ExecutionService(broker, chain_provider, spot, log=lambda *a: None)

    print("\n### A. PROPOSED -> fresh policy APPROVED -> BUY_TO_OPEN -> FILLED")
    a = make_proposal(1, spot, chain_provider, marketable=True)
    ra = svc.execute_entry(a.proposal_key)
    show(ra)

    print("\n### B. PROPOSED -> fresh policy APPROVED -> order PENDING (limit at mid, below ask)")
    b = make_proposal(2, spot, chain_provider, marketable=False)
    show(svc.execute_entry(b.proposal_key))

    print("\n### C. PROPOSED -> market moves +0.8% (SIMULATED) -> fresh policy REJECTED -> no broker order")
    c = make_proposal(3, spot, chain_provider, marketable=True)
    spot.shift_pct = 0.8
    rc = svc.execute_entry(c.proposal_key)
    show(rc)
    print(f"  broker orders for {c.proposal_key}: {len(q('SELECT 1 FROM paper_orders WHERE proposal_key = ?', (c.proposal_key,)))}")
    spot.shift_pct = 0.0

    print("\n### D. Execute proposal A again -> same execution, one broker order")
    show(svc.execute_entry(a.proposal_key))
    print(f"  broker orders for {a.proposal_key}: {len(q('SELECT 1 FROM paper_orders WHERE proposal_key = ?', (a.proposal_key,)))}")

    print("\n### E. Crash after broker accepted the order, before the result was saved -> restart -> reconcile")
    e = make_proposal(4, spot, chain_provider, marketable=True)
    real_submit = broker.submit_order

    def submit_then_crash(request):
        real_submit(request)
        raise SystemExit("simulated crash")

    broker.submit_order = submit_then_crash
    try:
        svc.execute_entry(e.proposal_key)
    except SystemExit:
        pass
    broker.submit_order = real_submit
    stuck = q("SELECT status FROM execution_attempts WHERE proposal_key = ?", (e.proposal_key,))[0]["status"]
    print(f"  after crash: execution status={stuck}, broker orders={len(q('SELECT 1 FROM paper_orders WHERE proposal_key = ?', (e.proposal_key,)))}")
    later = lambda: int(time.time() * 1000) + 5 * 60_000
    restarted = ExecutionService(create_broker(chain_provider, log=lambda *a: None), chain_provider, spot,
                                 now_ms=later, log=lambda *a: None)
    for r in restarted.reconcile_pending_executions():
        if r.proposal_key == e.proposal_key:
            show(r)
    print(f"  broker orders for {e.proposal_key} after reconcile: "
          f"{len(q('SELECT 1 FROM paper_orders WHERE proposal_key = ?', (e.proposal_key,)))}")
    ev = q("SELECT from_status, to_status FROM execution_events e JOIN execution_attempts a USING (execution_id) "
           "WHERE a.proposal_key = ? ORDER BY e.id", (e.proposal_key,))
    print("  transitions:", " -> ".join(x["to_status"] for x in ev))

    print("\n### Persisted execution row (A)")
    [row] = q("SELECT execution_id, execution_key, proposal_key, signal_id, purpose, status, policy_decision_id, "
              "client_order_id, broker_order_id, error_code, created_at, updated_at, completed_at "
              "FROM execution_attempts WHERE proposal_key = ?", (a.proposal_key,))
    print(json.dumps(row, indent=2))

    print("\n### Linked policy decision")
    [d] = q("SELECT decision_id, proposal_key, approved, decision, reasons_json, calculated_max_loss, "
            "quote_age_seconds, checked_at FROM policy_decisions WHERE decision_id = ?", (row["policy_decision_id"],))
    print(json.dumps(d, indent=2))

    print("\n### Linked broker order")
    [o] = q("SELECT order_id, client_order_id, proposal_key, contract_symbol, side, quantity, limit_price, "
            "status, fill_price, filled_at, position_id FROM paper_orders WHERE order_id = ?", (row["broker_order_id"],))
    print(json.dumps(o, indent=2))

    print(f"\nTotals: executions={q('SELECT COUNT(*) n FROM execution_attempts')[0]['n']} "
          f"policy_decisions={q('SELECT COUNT(*) n FROM policy_decisions')[0]['n']} "
          f"broker_orders={q('SELECT COUNT(*) n FROM paper_orders')[0]['n']}")


if __name__ == "__main__":
    main()
