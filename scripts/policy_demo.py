"""
Policy Engine demo on a SCRATCH database (never the real one).
Live Binance.US spot -> mock Deribit-style chain -> deterministic Agent 2
candidate (LLM disabled) -> Policy Engine. Nothing is ever submitted.

    TRADE_MONITOR_DB=/tmp/policy_demo.db python scripts/policy_demo.py
"""
import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import database
from agents.options_agent import OptionsSelectionAgent
from core import policy_engine as pe
from models import TradeSignal
from schemas import OptionsSelectionInput, ThesisSnapshot
from tools.market_tools import BinanceUSProvider
from tools.options_tools import MockDeribitChainProvider, build_playbook, prefilter


class ShiftableSpot:
    def __init__(self):
        self.real = BinanceUSProvider()
        self.base = None
        self.shift_pct = 0.0

    def get_current_price(self, symbol):
        if self.base is None:
            self.base = self.real.get_current_price(symbol)
        return round(self.base * (1 + self.shift_pct / 100), 2)


def show(label, d):
    print(f"\n### {label}")
    print(f"decision={d.decision} reasons={d.reasons}")
    print(f"max_loss={d.calculated_max_loss} total_open_risk={d.calculated_total_open_risk} "
          f"quote_age_s={d.quote_age_seconds} decision_id={d.decision_id}")


def main():
    if config.db_path() == str(config.PROJECT_DIR / "trade_monitor.db"):
        sys.exit("refusing to run against the real database; set TRADE_MONITOR_DB")
    if os.path.exists(config.db_path()):
        os.remove(config.db_path())
    database.initialize_database()
    os.environ["KILL_SWITCH_FILE"] = str(Path(tempfile.gettempdir()) / "policy_demo_KILL_SWITCH")
    Path(os.environ["KILL_SWITCH_FILE"]).unlink(missing_ok=True)

    spot = ShiftableSpot()
    chain_provider = MockDeribitChainProvider(spot)
    px = spot.get_current_price("BTCUSDT")
    now = int(time.time() * 1000)

    entry = round(px - 100, -1)
    signal = TradeSignal(is_trade_signal=True, symbol="BTCUSDT", bias="conditional_bullish", timeframe="1h",
                         support_levels=[entry - 500], resistance_levels=[entry], targets=[entry + 2000],
                         confirmation_above=entry, invalidation_below=entry - 1500, summary="demo")
    database.save_signal("demo-1", signal, now - 5 * 60_000, px - 300, raw_message="demo")
    database.update_status(1, "WORKING")
    print(f"BTC spot {px} | thesis: break above {entry} -> {entry + 2000}, invalid below {entry - 1500}")

    thesis = ThesisSnapshot(underlying="BTCUSDT", direction="bullish", entry_level=entry, target=entry + 2000,
                            invalidation=entry - 1500, time_horizon="intraday", agent1_state="DEVELOPING")
    chain = chain_provider.get_option_chain("BTCUSDT", option_type="call")
    candidates, summary = prefilter(chain, thesis, build_playbook(), now)
    proposal = OptionsSelectionAgent(None, log=lambda *a: None).propose(OptionsSelectionInput(
        signal_id=1, trigger="FIRST_WORKING", thesis=thesis, underlying_price=px, as_of_ms=now,
        playbook=build_playbook(), chain_source=chain.source, candidates=candidates, filter_summary=summary))
    print(f"proposal: {proposal.strategy} {proposal.contract_symbol} x{proposal.quantity} limit {proposal.limit_price} "
          f"(bid {proposal.bid} / ask {proposal.ask}, delta {proposal.delta:.2f})")

    approved = pe.evaluate_entry(proposal, chain_provider, spot, now)
    show("1. APPROVED", approved)

    os.environ["MAX_CONTRACTS"] = "1"
    show("2. Single-rule rejection (MAX_CONTRACTS=1)", pe.evaluate_entry(proposal, chain_provider, spot, now + 1000))
    del os.environ["MAX_CONTRACTS"]

    Path(os.environ["KILL_SWITCH_FILE"]).touch()
    show("3. Kill switch (KILL_SWITCH file present)", pe.evaluate_entry(proposal, chain_provider, spot, now + 2000))
    Path(os.environ["KILL_SWITCH_FILE"]).unlink()

    os.environ["TRADING_KILL_SWITCH"] = "1"
    os.environ["MAX_PREMIUM_PER_TRADE"] = "20"
    spot.shift_pct = 0.8
    show("4. Multi-rule rejection (env kill switch + $20 max premium + simulated +0.8% move)",
         pe.evaluate_entry(proposal, chain_provider, spot, now + 3000))
    del os.environ["TRADING_KILL_SWITCH"], os.environ["MAX_PREMIUM_PER_TRADE"]

    spot.shift_pct = 3.0
    fresh = chain_provider.get_option_contract(proposal.contract_symbol)
    print(f"\n### 5. Fresh-quote revalidation after a SIMULATED +3% move")
    print(f"proposal-time quote: bid {proposal.bid} / ask {proposal.ask} / delta {proposal.delta:.3f}")
    print(f"fresh quote:         bid {fresh.bid} / ask {fresh.ask} / delta {fresh.delta:.3f} (spot {fresh.underlying_price})")
    d = pe.evaluate_entry(proposal, chain_provider, spot, now + 4000)
    print(f"decision={d.decision} reasons={d.reasons} max_loss={d.calculated_max_loss:.2f}")
    print(f"proposal unchanged: limit {proposal.limit_price}, qty {proposal.quantity}")

    print("\n### 6. Idempotency")
    spot.shift_pct = 0.0
    conn = sqlite3.connect(config.db_path())
    count = lambda: conn.execute("SELECT COUNT(*) FROM policy_decisions").fetchone()[0]
    inp = pe.gather_entry_input(proposal, chain_provider, spot, now + 5000)
    first = pe.persist_decision(pe.evaluate(inp), inp)
    rows_after_first = count()
    repeat = pe.persist_decision(pe.evaluate(inp), inp)
    print(f"same snapshot evaluated twice -> {first.decision_id} / {repeat.decision_id}; "
          f"rows {rows_after_first} -> {count()} (no duplicate)")
    regathered = pe.gather_entry_input(proposal, chain_provider, spot, now + 6000)
    newer = pe.persist_decision(pe.evaluate(regathered), regathered)
    print(f"re-gathered with a fresh quote/time -> new decision {newer.decision_id}; rows {count()} "
          f"(legitimate re-evaluation)")

    print("\n### 7. Persisted PolicyDecision (APPROVED)")
    row = sqlite3.connect(config.db_path()).execute(
        "SELECT decision_json FROM policy_decisions WHERE decision_id = ?", (approved.decision_id,)).fetchone()
    print(json.dumps(json.loads(row[0]), indent=2))

    orders = sqlite3.connect(config.db_path()).execute("SELECT COUNT(*) FROM paper_orders").fetchone()[0]
    print(f"\npaper_orders rows: {orders}  (the Policy Engine authorizes only; nothing was submitted)")


if __name__ == "__main__":
    main()
