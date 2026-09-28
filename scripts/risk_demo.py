"""
Agent 3 (risk analysis) demo on a SCRATCH database (never the real one).
Live Binance.US spot -> mock Deribit-style chain -> a paper position ->
RiskService + real OpenAI for contextual cases. No Telegram, no orders
beyond the demo's own entry fill, no exits are executed.

    TRADE_MONITOR_DB=/tmp/risk_demo.db python scripts/risk_demo.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import database
from agents.risk_agent import RiskAnalysisAgent
from broker.factory import create_broker
from llm_client import OpenAILLMClient
from models import TradeSignal
from risk_service import RiskService
from schemas import OrderRequest
from tools.market_tools import BinanceUSProvider
from tools.options_tools import MockDeribitChainProvider


class Clock:
    def __init__(self):
        self.offset_ms = 0

    def __call__(self):
        return int(time.time() * 1000) + self.offset_ms


class ShiftableMarket:
    """Real Binance.US spot and candles, with an optional SIMULATED % shift."""

    def __init__(self):
        self.real = BinanceUSProvider()
        self.base = self.real.get_current_price("BTCUSDT")
        self.shift_pct = 0.0

    def get_current_price(self, symbol):
        return round(self.base * (1 + self.shift_pct / 100), 2)

    def get_candles(self, symbol, interval="1m", limit=200, start_ms=None, end_ms=None):
        p = self.get_current_price(symbol)
        return [{"timestamp": (start_ms or int(time.time() * 1000)) // 60_000 * 60_000,
                 "open": p, "high": p * 1.0003, "low": p * 0.9997, "close": p, "volume": 1.0}]


class QuoteShock:
    """Wraps the chain provider; SIMULATED multiplicative shock to bid/ask/mark (e.g. an IV collapse)."""

    def __init__(self, provider):
        self.provider = provider
        self.factor = 1.0

    def get_option_contract(self, symbol):
        q = self.provider.get_option_contract(symbol)
        if self.factor == 1.0:
            return q
        f = self.factor
        return q.model_copy(update={"bid": round(q.bid * f, 1), "ask": round(q.ask * f, 1), "mark": q.mark * f})


class CountingLLM:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def structured(self, *a, **k):
        self.calls += 1
        return self.inner.structured(*a, **k)


def show(label, a, llm):
    print(f"\n### {label}")
    print(f"recommendation={a.recommendation} state={a.risk_state} hard={a.hard_rule_triggered} "
          f"codes={a.hard_rule_codes} data={a.data_issue_codes} llm={a.llm_status} ({a.llm_trigger}) "
          f"llm_calls_so_far={llm.calls}")
    m = a.metrics
    print(f"underlying={a.underlying_price} bid={a.option_bid} mark={m.option_mark:.1f} | "
          f"PnL(bid)=${a.unrealized_pnl:.2f} ({a.unrealized_pnl_pct:+.1f}%)  PnL(mark, display)=${m.mark_pnl:.2f} | "
          f"progress={m.target_progress_pct:.0f}% dist_inv={m.distance_to_invalidation_pct:.2f}% "
          f"in_trade={m.time_in_trade_hours:.1f}h")
    for r in a.reasons:
        print("  -", r)
    for n in a.validation_notes:
        print("  note:", n)


def main():
    if config.db_path() == str(config.PROJECT_DIR / "trade_monitor.db"):
        sys.exit("refusing to run against the real database; set TRADE_MONITOR_DB")
    if os.path.exists(config.db_path()):
        os.remove(config.db_path())
    database.initialize_database()

    clock = Clock()
    market = ShiftableMarket()
    chain = MockDeribitChainProvider(market, now_ms=clock)
    quotes = QuoteShock(chain)
    px = market.get_current_price("BTCUSDT")

    entry = round(px - 100, -1)
    signal = TradeSignal(is_trade_signal=True, symbol="BTCUSDT", bias="conditional_bullish", timeframe="swing",
                         support_levels=[entry - 500], resistance_levels=[entry], targets=[entry + 2000],
                         confirmation_above=entry, invalidation_below=entry - 1500, summary="demo")
    database.save_signal("demo-1", signal, clock() - 10 * 60_000, px - 300, raw_message="demo")
    database.update_status(1, "WORKING")

    contract = min((c for c in chain.get_option_chain("BTCUSDT", option_type="call").contracts if 10 < c.dte < 20),
                   key=lambda c: abs(c.strike - (entry + 1000)))
    broker = create_broker(chain, now_ms=clock, log=lambda *a: None)     # demo setup: open the position
    pos = broker.submit_order(OrderRequest(client_order_id="entry:signal:1:entry", signal_id=1,
                                           proposal_key="signal:1:entry", contract_symbol=contract.contract_symbol,
                                           side="BUY_TO_OPEN", quantity=2, limit_price=contract.ask)).position
    print(f"BTC spot {px} | thesis: above {entry} -> {entry + 2000}, invalid below {entry - 1500}")
    print(f"position {pos.position_id}: {pos.contract_symbol} x{pos.quantity} @ {pos.entry_price} (x{pos.contract_multiplier})")

    llm = CountingLLM(OpenAILLMClient())
    service = RiskService(RiskAnalysisAgent(llm, log=lambda *a: None), quotes, market, now_ms=clock)

    clock.offset_ms = 30 * 60_000
    show("A. Healthy open call (+30 min)", service.assess_position(pos.position_id), llm)

    clock.offset_ms = 31 * 60_000
    show("Throttle: one minute later, nothing material changed", service.assess_position(pos.position_id), llm)

    before = llm.calls
    market.shift_pct = (entry - 1500 - 50) / market.base * 100 - 100
    clock.offset_ms = 40 * 60_000
    show("B. SIMULATED drop through invalidation -> hard EXIT", service.assess_position(pos.position_id), llm)
    print(f"  LLM calls for B: {llm.calls - before}")

    market.shift_pct = 0.0
    quotes.factor = 0.45
    clock.offset_ms = 50 * 60_000
    before = llm.calls
    show("C. SIMULATED option price shock (x0.45) -> premium stop on the BID", service.assess_position(pos.position_id), llm)
    print(f"  LLM calls for C: {llm.calls - before}")

    quotes.factor = 1.0
    market.shift_pct = -1.2                                           # toward invalidation, not through
    clock.offset_ms = 3 * 24 * 3_600_000                              # 3 days of theta
    show("D. No hard stop; SIMULATED -1.2% drift + 3 days elapsed -> contextual LLM", service.assess_position(pos.position_id), llm)

    market.shift_pct = (entry + 2000 + 50) / market.base * 100 - 100
    clock.offset_ms = 3 * 24 * 3_600_000 + 60 * 60_000
    before = llm.calls
    show("E. SIMULATED move through target -> hard EXIT (thesis complete)", service.assess_position(pos.position_id), llm)
    print(f"  LLM calls for E: {llm.calls - before}")

    import sqlite3
    conn = sqlite3.connect(config.db_path())
    print("\nrisk_assessments rows:", conn.execute("SELECT COUNT(*) FROM risk_assessments").fetchone()[0],
          "| paper_orders rows:", conn.execute("SELECT COUNT(*) FROM paper_orders").fetchone()[0],
          "(only the demo's own entry; Agent 3 submitted nothing)")
    print("position still:", conn.execute("SELECT status, quantity FROM paper_positions").fetchone())
    row = conn.execute("SELECT output_json FROM risk_assessments WHERE risk_state = 'HARD_EXIT' "
                       "ORDER BY assessed_at_ms LIMIT 1").fetchone()
    Path(os.environ.get("RISK_DEMO_JSON", "/dev/null")).write_text(row[0])


if __name__ == "__main__":
    main()
