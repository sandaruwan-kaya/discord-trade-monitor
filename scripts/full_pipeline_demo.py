"""
Full paper-trading pipeline demo on a SCRATCH database.

Real: Discord-message parser (OpenAI), Agent 1/2/3 LLM calls (OpenAI),
      Binance.US spot (anchors the price path), every production component.
Simulated (labelled): the minute-by-minute price path around the live spot,
      the clock (minutes pass instantly), the invalidation move.
Never: production DB, Telegram sends (captured), Discord writes, live broker.

    TRADE_MONITOR_DB=/tmp/full_pipeline_demo.db python scripts/full_pipeline_demo.py
"""
import asyncio
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

if config.db_path() == str(config.PROJECT_DIR / "trade_monitor.db"):
    sys.exit("refusing to run against the real database; set TRADE_MONITOR_DB")

# Flags for THIS process only (the .env file is not touched).
for flag in ("AGENTIC_TRADING_ENABLED", "AGENT1_ENABLED", "AGENT2_ENABLED", "PAPER_EXECUTION_ENABLED",
             "AGENT3_ENABLED", "AGENT4_ENABLED", "AUTO_EXIT_ENABLED"):
    os.environ[flag] = "true"
os.environ["TRADING_MODE"] = "paper"

import database
from agents.options_agent import OptionsSelectionAgent
from agents.risk_agent import RiskAnalysisAgent
from agents.signal_agent import SignalAnalysisAgent
from broker.factory import create_broker
from llm_client import OpenAILLMClient
from llm_parser import parse_discord_message
from notifications import Notifier
from signal_service import Services, process_discord_message, run_monitor_cycle, run_proposal_cycle
from tools.market_tools import BinanceUSProvider, FallbackMarketDataProvider
from tools.options_tools import MockDeribitChainProvider
from trading_pipeline import TradingPipeline

MINUTE = 60_000


class Clock:
    def __init__(self, start):
        self.now = start

    def __call__(self):
        return self.now


class ScriptedPath:
    """Binance-like 1m feed: SIMULATED minute prices around the LIVE Binance.US spot."""
    name = "binance_us_anchored_path"

    def __init__(self, clock, t0, base):
        self.clock, self.t0, self.points, self.width = clock, t0, {0: base}, 20.0

    def set(self, minute, price):
        self.points[minute] = round(price, 2)

    def _price(self, minute):
        return self.points[max(m for m in self.points if m <= minute)]

    def get_current_price(self, symbol):
        return self._price((self.clock.now - self.t0) // MINUTE)

    def get_candles(self, symbol, interval="1m", limit=200, start_ms=None, end_ms=None):
        now_m = (self.clock.now - self.t0) // MINUTE
        first = max(0, (start_ms - self.t0) // MINUTE) if start_ms is not None else max(0, now_m - limit + 1)
        return [{"timestamp": self.t0 + m * MINUTE, "open": self._price(m), "high": self._price(m) + self.width,
                 "low": self._price(m) - self.width, "close": self._price(m), "volume": 1.0}
                for m in range(first, min(now_m, first + limit - 1) + 1)]

    def get_market_context(self, symbol):
        from tools.market_tools import context_from_candles
        return context_from_candles(self.name, symbol, self.get_candles(symbol, "1m", 100))


def q(sql, params=()):
    conn = sqlite3.connect(config.db_path())
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params)]
    finally:
        conn.close()


def main():
    if os.path.exists(config.db_path()):
        os.remove(config.db_path())
    database.initialize_database()

    live = BinanceUSProvider().get_current_price("BTCUSDT")
    t0 = (int(time.time() * 1000) // MINUTE) * MINUTE
    clock = Clock(t0 + 15_000)
    feed = ScriptedPath(clock, t0, live)

    entry = round(live + 300, -1)
    target, invalid = entry + 2000, entry - 1500
    text = (f"BTC holding {live - 500:,.0f} support. Clean break above {entry:,.0f} opens {target:,.0f}. "
            f"Lose {invalid:,.0f} and the idea is dead. 1h chart.")

    legacy, trade_msgs = [], []
    chain = MockDeribitChainProvider(feed, now_ms=clock)
    services = Services(
        parse_message=parse_discord_message, market=feed, notify=legacy.append,
        context_market=FallbackMarketDataProvider([feed], log=lambda *a: None),
        signal_agent=SignalAnalysisAgent(OpenAILLMClient(), now_ms=clock),
        option_chain=chain, options_agent=OptionsSelectionAgent(OpenAILLMClient()),
        now_ms=clock, log=lambda *a, **k: None,
    )
    logs = []
    pipeline = TradingPipeline(
        signal_services=services, broker=create_broker(chain, now_ms=clock, log=lambda *a: None),
        chain_provider=chain, market=feed, risk_agent=RiskAnalysisAgent(OpenAILLMClient()),
        notifier=Notifier(trade_msgs.append, log=lambda *a: None, now_ms=clock),
        context_market=FallbackMarketDataProvider([feed], log=lambda *a: None),
        now_ms=clock, log=logs.append,
    )

    def cycle(label):
        asyncio.run(run_monitor_cycle(services))
        pipeline.after_proposals(asyncio.run(run_proposal_cycle(services)))
        pipeline.entry_cycle()
        pipeline.risk_cycle()
        pipeline.lifecycle_cycle()
        pipeline.exit_cycle()
        print(f"\n--- {label} (t+{(clock.now - t0) / MINUTE:.1f} min, price {feed.get_current_price('BTCUSDT')})")
        for line in logs:
            print("  ", line)
        logs.clear()

    print(f"Live Binance.US BTC: {live}")
    print(f"Discord message (simulated post): {text!r}")
    pipeline.startup_reconcile()

    asyncio.run(process_discord_message("demo-msg-1", text, clock.now, services))
    s = q("SELECT * FROM signals")[0]
    a = q("SELECT signal_state, output_json FROM signal_analyses")[0]
    print(f"\nSignal #{s['id']}: bias={s['bias']} confirm>{s['confirmation_above']} target={json.loads(s['targets'])} "
          f"invalid<{s['invalidation_below']} | Agent 1: {a['signal_state']} "
          f"(llm_used={json.loads(a['output_json'])['llm_used']})")

    clock.now = t0 + int(1 * MINUTE)
    cycle("DEVELOPING")
    feed.set(10, entry + 200)                                  # SIMULATED breakout
    clock.now = t0 + int(10.5 * MINUTE)
    cycle("SIMULATED breakout above entry -> WORKING")
    clock.now = t0 + int(11.5 * MINUTE)
    cycle("confirming candle closed -> Agent 2 proposal -> fresh entry policy -> paper entry")
    feed.set(12, entry + 100)                                  # SIMULATED small dip: option ask reaches the limit
    clock.now = t0 + int(12.5 * MINUTE)
    cycle("entry fill -> Trade -> Agent 3 -> Agent 4")
    clock.now = t0 + int(30.5 * MINUTE)
    cycle("healthy monitoring")
    feed.set(40, invalid - 100)                                # SIMULATED drop through invalidation
    clock.now = t0 + int(40.5 * MINUTE)
    cycle("SIMULATED invalidation -> hard EXIT -> exit policy -> paper close")

    [sig] = q("SELECT * FROM signals")
    [an] = q("SELECT id FROM signal_analyses")
    [prop] = q("SELECT * FROM trade_proposals")
    entry_ex = q("SELECT * FROM execution_attempts WHERE purpose = 'ENTRY'")
    if not entry_ex or entry_ex[0]["status"] != "ORDER_FILLED" or not q("SELECT 1 FROM trades"):
        print("\nNo filled paper trade in this run (proposal/policy outcome shown above); lineage stops here.")
        print("proposal:", prop["status"], prop["reason"], "| entry:", entry_ex and entry_ex[0]["status"])
        return
    [tr] = q("SELECT * FROM trades")
    [pos] = q("SELECT * FROM paper_positions WHERE position_id = ?", (tr["position_id"],))
    xr = q("SELECT * FROM exit_requests WHERE trade_id = ? AND requested_action = 'EXIT'", (tr["trade_id"],))
    exit_ex = q("SELECT * FROM execution_attempts WHERE purpose = 'EXIT'")
    entry_ex = entry_ex[0]

    print("\n=== LINEAGE ===")
    lineage = [
        ("discord_message_id", sig["discord_message_id"]), ("signal_id", sig["id"]), ("agent1 analysis_id", an["id"]),
        ("proposal_key", prop["proposal_key"]), ("entry policy decision", entry_ex["policy_decision_id"]),
        ("entry execution_id", entry_ex["execution_id"]), ("entry order_id", entry_ex["broker_order_id"]),
        ("position_id", pos["position_id"]), ("trade_id", tr["trade_id"]),
    ]
    if xr:
        lineage += [("exit assessment_id", xr[0]["risk_assessment_id"]), ("exit_request_id", xr[0]["exit_request_id"])]
    if exit_ex:
        lineage += [("exit policy decision", exit_ex[0]["policy_decision_id"]),
                    ("exit execution_id", exit_ex[0]["execution_id"]), ("close order_id", exit_ex[0]["broker_order_id"])]
    lineage += [("trade state", tr["state"])]
    for k, v in lineage:
        print(f"  {k:<24} {v}")

    close = q("SELECT fill_price FROM paper_orders WHERE position_id = ? AND side = 'SELL_TO_CLOSE' AND status = 'FILLED'",
              (pos["position_id"],))
    print(f"\n{pos['contract_symbol']} x{pos['original_quantity']}: entry {pos['entry_price']} -> exit "
          f"{close[0]['fill_price'] if close else None} | realized P&L ${pos['realized_pnl']:.2f}")
    print("\nlifecycle:", " -> ".join(e["to_state"] for e in q(
        "SELECT to_state FROM trade_lifecycle_events WHERE trade_id = ? ORDER BY event_id", (tr["trade_id"],))))
    print("\nTelegram (captured, NOT sent):")
    for m in legacy + trade_msgs:
        print("  |", m.split("\n")[0])


if __name__ == "__main__":
    main()
