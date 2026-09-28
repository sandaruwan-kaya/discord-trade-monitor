"""
End-to-end paper-trading integration (offline). Every real component is wired;
only the market feed, Discord parser, LLMs and Telegram are fakes.
"""
import asyncio
import json
import os
import pathlib
import re
import sqlite3

import pytest

import config
import database
from integration_harness import BEAR, BULL, Harness, ScriptedLLM
from helpers import MINUTE_MS, T0


ALL_FLAGS = ("AGENTIC_TRADING_ENABLED", "AGENT1_ENABLED", "AGENT2_ENABLED", "PAPER_EXECUTION_ENABLED",
             "AGENT3_ENABLED", "AGENT4_ENABLED", "AUTO_EXIT_ENABLED")


@pytest.fixture
def flags(monkeypatch):
    def set_flags(**overrides):
        for f in ALL_FLAGS:
            monkeypatch.setenv(f, "true")
        for k, v in overrides.items():
            monkeypatch.setenv(k, v)
    return set_flags


def q(sql, params=()):
    conn = sqlite3.connect(config.db_path())
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params)]
    finally:
        conn.close()


def n(table, where="1=1", params=()):
    return q(f"SELECT COUNT(*) AS n FROM {table} WHERE {where}", params)[0]["n"]


def bull_to_working(h):
    """Post the BTC signal, confirm at minute 10, persist the confirming candle at minute 11."""
    h.feed.set("BTCUSDT", 10, 84200.0)
    h.post("m1", BULL)
    for m in (1, 10.5, 11.5):
        h.at(m)
        h.cycle()


def bull_open(h):
    bull_to_working(h)                      # proposal + resting entry at the mid
    h.feed.set("BTCUSDT", 12, 84100.0)      # option ask dips to the limit (still above entry)
    h.at(12.5)
    h.cycle()


def trade(h):
    [t] = q("SELECT * FROM trades")
    return t


# 1 / 14 (legacy)
def test_master_switch_off_is_legacy_signal_monitor_only(monkeypatch):
    for f in ALL_FLAGS[1:]:
        monkeypatch.setenv(f, "true")                  # every component flag on, master OFF
    h = Harness()
    bull_open(h)
    h.feed.set("BTCUSDT", 20, 86100.0)
    h.at(20.5)
    h.cycle()

    assert h.legacy_telegram[0].startswith("🆕 NEW DISCORD TRADE SIGNAL")
    assert any("STATUS:\nWORKING" in m for m in h.legacy_telegram)
    assert any("STATUS:\nTARGET_HIT" in m for m in h.legacy_telegram)
    for table in ("signal_analyses", "trade_proposals", "execution_attempts", "paper_orders", "trades",
                  "risk_assessments", "exit_requests", "notifications", "policy_decisions"):
        assert n(table) == 0, table
    assert h.pipeline.startup_reconcile() == {"skipped": "AGENTIC_TRADING_ENABLED=false"}
    assert h.trade_messages == []


# 2
def test_proposal_created_but_execution_flag_off_places_no_order(flags):
    flags(PAPER_EXECUTION_ENABLED="false")
    h = Harness()
    bull_to_working(h)
    assert q("SELECT status FROM trade_proposals")[0]["status"] == "PROPOSED"
    assert n("paper_orders") == 0 and n("execution_attempts") == 0
    assert any(m.startswith("🧾 OPTION PROPOSAL CREATED") for m in h.trade_messages)


# 3 / 7 / 9 / 10
def test_full_bullish_path_to_open_trade(flags):
    flags()
    h = Harness()
    bull_to_working(h)

    assert q("SELECT signal_state FROM signal_analyses")[0]["signal_state"] == "DEVELOPING"
    assert q("SELECT status FROM execution_attempts")[0]["status"] == "ORDER_PENDING"      # 7: PENDING first
    assert n("trades") == 0

    h.feed.set("BTCUSDT", 12, 84100.0)
    h.at(12.5)
    h.cycle()

    ex = q("SELECT * FROM execution_attempts WHERE purpose = 'ENTRY'")[0]
    assert ex["status"] == "ORDER_FILLED"                                                  # 7: later FILLED
    t = trade(h)
    assert t["state"] == "WORKING"                                                         # 10: HOLD -> WORKING
    assert q("SELECT strategy FROM trade_proposals")[0]["strategy"] == "LONG_CALL"
    for _ in range(3):
        h.cycle()
    assert n("trades") == 1                                                                # 9: once
    kinds = [m.split("\n")[0] for m in h.trade_messages]
    assert kinds == ["🧾 OPTION PROPOSAL CREATED (paper)", "⏳ PAPER ENTRY ORDER PENDING",
                     "✅ PAPER ENTRY ORDER FILLED", "📈 PAPER POSITION OPENED"]


# 4
def test_bearish_path_long_put(flags):
    flags()
    h = Harness()
    h.feed.set("ETHUSDT", 10, 2610.0)
    h.post("e1", BEAR)
    for m in (1, 10.5, 11.5):
        h.at(m)
        h.cycle()
    h.feed.set("ETHUSDT", 12, 2618.0)                  # put ask dips to the limit (still below entry 2620)
    h.at(12.5)
    h.cycle()

    p = json.loads(q("SELECT output_json FROM trade_proposals")[0]["output_json"])
    assert p["strategy"] == "LONG_PUT" and p["option_type"] == "put"
    assert trade(h)["state"] in ("WORKING", "AT_RISK")
    assert q("SELECT side, status FROM paper_orders")[0] == {"side": "BUY_TO_OPEN", "status": "FILLED"}


def test_backlog_proposal_from_before_execution_was_enabled_is_never_executed(flags, monkeypatch):
    flags(PAPER_EXECUTION_ENABLED="false")
    h = Harness()
    bull_to_working(h)                                     # PROPOSED while execution is off
    conn = sqlite3.connect(config.db_path())
    conn.execute("UPDATE trade_proposals SET updated_at = '2000-01-01T00:00:00+00:00'")   # an old backlog item
    conn.commit()
    monkeypatch.setenv("PAPER_EXECUTION_ENABLED", "true")
    h.cycle()
    assert n("execution_attempts") == 0 and n("paper_orders") == 0
    assert any("older than the execution window" in l for l in h.logs)


def test_proposal_age_window_uses_wall_clock_not_simulated_clock(flags):
    flags()
    h = Harness()
    h.clock.now = int(__import__("time").time() * 1000) + 3 * 3_600_000   # simulated clock far ahead
    # (a proposal just finished in wall-clock terms must still be eligible)
    assert h.pipeline is not None
    from trading_pipeline import _iso_to_ms
    import time as _t
    assert _t.time() * 1000 - _iso_to_ms(database.utc_now_iso()) < 5_000


# 5 / 6
def test_stale_entry_rejected_by_fresh_policy_no_order(flags, monkeypatch):
    flags()
    h = Harness()
    h.feed.set("BTCUSDT", 10, 84200.0)
    h.post("m1", BULL)
    for m in (1, 10.5):
        h.at(m)
        h.cycle()
    # Proposal sweep and the entry run in the same cycle; move the price between them.
    real_entry = h.pipeline.entry_cycle

    def jump_then_enter():
        h.feed.set("BTCUSDT", 11, 84700.0)            # +0.83% past entry before execution
        return real_entry()

    h.pipeline.entry_cycle = jump_then_enter
    h.at(11.5)
    h.cycle()

    ex = q("SELECT * FROM execution_attempts")[0]
    assert ex["status"] == "POLICY_REJECTED" and "stale_entry_move_extended" in ex["error_message"]
    assert n("paper_orders") == 0
    assert any(m.startswith("⛔ ENTRY POLICY REJECTED") for m in h.trade_messages)
    for _ in range(3):
        h.cycle()
    assert n("execution_attempts") == 1 and n("paper_orders") == 0                        # no retry


# 8
def test_restart_while_entry_pending(flags):
    flags()
    h = Harness()
    bull_to_working(h)
    assert q("SELECT status FROM paper_orders")[0]["status"] == "PENDING"

    h2 = Harness(clock=h.clock, feed=h.feed)                                               # "restart"
    h2.parsed = h.parsed
    h2.pipeline.startup_reconcile()
    assert n("paper_orders") == 1
    h2.feed.set("BTCUSDT", 12, 84100.0)
    h2.at(12.5)
    h2.cycle()
    assert n("paper_orders") == 1 and n("trades") == 1
    assert q("SELECT status FROM paper_orders")[0]["status"] == "FILLED"


# 11 / 12 / 37
def test_near_risk_hold_is_at_risk(flags):
    flags()
    h = Harness(risk_llm=ScriptedLLM("HOLD"))
    bull_open(h)
    h.feed.set("BTCUSDT", 20, 82950.0)                 # 0.42% above invalidation
    h.at(30.5)
    h.cycle()
    assert trade(h)["state"] == "AT_RISK"
    assert any(m.startswith("⚠️ PAPER POSITION AT RISK") for m in h.trade_messages)


def test_contextual_reduce_is_requested_never_executed(flags):
    flags()
    h = Harness(risk_llm=ScriptedLLM("REDUCE"))
    bull_open(h)
    for m in (13.5, 30.5, 50.5):
        h.at(m)
        h.cycle()
    assert trade(h)["state"] == "REDUCE_REQUESTED"
    [req] = q("SELECT * FROM exit_requests")                    # throttled HOLDs no longer flap it
    assert req["requested_action"] == "REDUCE" and req["requested_quantity"] is None and req["status"] == "ACTIVE"
    assert n("paper_orders", "side = 'SELL_TO_CLOSE'") == 0
    assert n("execution_attempts", "purpose = 'EXIT'") == 0
    assert any(m.startswith("🔻 REDUCE REQUESTED") for m in h.trade_messages)


# 13 / 14 / 15
def test_hard_invalidation_exit_requested_but_not_executed_when_auto_exit_off(flags):
    flags(AUTO_EXIT_ENABLED="false")
    h = Harness()
    bull_open(h)
    h.feed.set("BTCUSDT", 20, 82500.0)
    h.at(20.5)
    h.cycle()
    h.cycle()
    assert trade(h)["state"] == "EXIT_REQUESTED"
    assert q("SELECT status, requested_action FROM exit_requests")[0] == {"status": "ACTIVE", "requested_action": "EXIT"}
    assert n("paper_orders", "side = 'SELL_TO_CLOSE'") == 0


def test_hard_invalidation_auto_exit_closes(flags):
    flags()
    h = Harness()
    bull_open(h)
    h.feed.set("BTCUSDT", 20, 82500.0)
    h.at(20.5)
    h.cycle()
    t = trade(h)
    assert t["state"] == "CLOSED"
    states = [e["to_state"] for e in q("SELECT to_state FROM trade_lifecycle_events ORDER BY event_id")]
    assert states[-3:] == ["EXIT_REQUESTED", "CLOSE_ORDER_PENDING", "CLOSED"]
    close = q("SELECT * FROM paper_orders WHERE side = 'SELL_TO_CLOSE'")[0]
    assert close["status"] == "FILLED" and close["limit_price"] == close["last_quote_bid"]
    closed_msg = [m for m in h.trade_messages if m.startswith("🏁 PAPER POSITION CLOSED")][0]
    # The signal monitor runs first and already resolved INVALIDATED, so Agent 3
    # reports the evaluator-resolved code (precedence as designed).
    assert "Realized P&L" in closed_msg and "SIGNAL_INVALIDATED" in closed_msg


# 16 / 17
def test_quote_outage_waits_then_same_exit_closes(flags):
    flags()
    h = Harness()
    bull_open(h)
    real = h.chain.get_option_contract
    h.chain.get_option_contract = lambda s: (_ for _ in ()).throw(ConnectionError("quote outage"))
    h.feed.set("BTCUSDT", 20, 82500.0)
    h.at(20.5)
    h.cycle()
    ex = q("SELECT * FROM execution_attempts WHERE purpose = 'EXIT'")[0]
    assert ex["status"] == "WAITING_FOR_QUOTE" and trade(h)["state"] == "EXIT_REQUESTED"
    assert any(m.startswith("⏸️ EXIT WAITING FOR QUOTE") for m in h.trade_messages)

    h.chain.get_option_contract = real
    h.at(21.5)
    h.cycle()
    ex2 = q("SELECT * FROM execution_attempts WHERE purpose = 'EXIT'")
    assert len(ex2) == 1 and ex2[0]["execution_id"] == ex["execution_id"] and ex2[0]["status"] == "ORDER_FILLED"
    assert trade(h)["state"] == "CLOSED"


# 18 / 19 / 20
def test_kill_switch_blocks_entry(flags):
    flags(TRADING_KILL_SWITCH="1")
    h = Harness()
    bull_to_working(h)
    assert q("SELECT status FROM execution_attempts")[0]["status"] == "POLICY_REJECTED"
    assert n("paper_orders") == 0


@pytest.mark.parametrize("env", [{"TRADING_KILL_SWITCH": "1"}, {"MAX_DAILY_LOSS": "0"}])
def test_kill_switch_and_daily_loss_do_not_block_exit(flags, monkeypatch, env):
    flags()
    h = Harness()
    bull_open(h)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    h.feed.set("BTCUSDT", 20, 82500.0)
    h.at(20.5)
    h.cycle()
    assert trade(h)["state"] == "CLOSED"


def test_max_daily_loss_blocks_entry(flags):
    flags(MAX_DAILY_LOSS="0")
    h = Harness()
    bull_to_working(h)
    ex = q("SELECT * FROM execution_attempts")[0]
    assert ex["status"] == "POLICY_REJECTED" and "max_daily_loss_reached" in ex["error_message"]


# 21-26
def test_repeated_loops_create_no_duplicates(flags):
    flags()
    h = Harness()
    bull_open(h)
    for m in range(13, 20):
        h.at(m + 0.5)
        h.cycle()
    h.feed.set("BTCUSDT", 20, 82500.0)
    for m in range(20, 26):
        h.at(m + 0.5)
        h.cycle()

    assert n("trade_proposals") == 1
    assert n("execution_attempts", "purpose = 'ENTRY'") == 1 and n("paper_orders", "side = 'BUY_TO_OPEN'") == 1
    assert n("trades") == 1
    assert n("exit_requests") == 1 and n("execution_attempts", "purpose = 'EXIT'") == 1
    assert n("paper_orders", "side = 'SELL_TO_CLOSE'") == 1
    dup = q("SELECT risk_assessment_id, COUNT(*) c FROM trade_lifecycle_events WHERE risk_assessment_id IS NOT NULL "
            "GROUP BY trade_id, risk_assessment_id HAVING c > 1")
    assert dup == []                                                                       # 24


# 27
def test_restart_after_entry_broker_submission(flags):
    flags()
    h = Harness()
    h.feed.set("BTCUSDT", 10, 84200.0)
    h.post("m1", BULL)
    for m in (1, 10.5):
        h.at(m)
        h.cycle()
    real = h.broker.submit_order

    def submit_then_die(request):
        real(request)
        raise SystemExit("killed after broker accepted the entry")

    h.broker.submit_order = submit_then_die
    h.at(11.5)
    with pytest.raises(SystemExit):
        h.cycle()
    assert q("SELECT status FROM execution_attempts")[0]["status"] == "SUBMITTING" and n("paper_orders") == 1

    h.at(20)                                                                               # restart after timeout
    h2 = Harness(clock=h.clock, feed=h.feed)
    h2.parsed = h.parsed
    h2.pipeline.startup_reconcile()
    assert q("SELECT status FROM execution_attempts")[0]["status"] == "ORDER_PENDING"
    assert n("paper_orders") == 1


# 28
def test_restart_after_exit_broker_submission(flags):
    flags()
    h = Harness()
    bull_open(h)
    real = h.broker.close_position

    def close_then_die(*a, **k):
        real(*a, **k)
        raise SystemExit("killed after broker accepted the close")

    h.broker.close_position = close_then_die
    h.feed.set("BTCUSDT", 20, 82500.0)
    h.at(20.5)
    with pytest.raises(SystemExit):
        h.cycle()
    assert n("paper_orders", "side = 'SELL_TO_CLOSE'") == 1

    h.at(30)
    h2 = Harness(clock=h.clock, feed=h.feed)
    h2.parsed = h.parsed
    h2.pipeline.startup_reconcile()
    h2.cycle()
    assert n("paper_orders", "side = 'SELL_TO_CLOSE'") == 1
    assert trade(h2)["state"] == "CLOSED"


# 29 / 30
def test_telegram_failure_never_rolls_back_and_restart_does_not_resend(flags):
    flags()
    failing = {"on": True}
    sent = []

    def send(text):
        if failing["on"]:
            raise ConnectionError("telegram down")
        sent.append(text)

    h = Harness(send=send)
    bull_open(h)
    assert trade(h)["state"] == "WORKING"                                                  # state persisted anyway
    assert n("notifications", "status = 'FAILED'") >= 3 and sent == []

    assert max(r["attempts"] for r in q("SELECT attempts FROM notifications")) == 1       # one try per interval

    failing["on"] = False
    h.at(14)                                                                               # retry interval elapsed
    h.cycle()                                                                              # retries unsent keys
    first = list(sent)
    assert any(m.startswith("📈 PAPER POSITION OPENED") for m in first)

    h2 = Harness(clock=h.clock, feed=h.feed, send=sent.append)                             # restart
    h2.parsed = h.parsed
    h2.pipeline.startup_reconcile()
    h2.cycle()
    assert sent == first                                                                   # nothing resent


# 31
def test_one_loop_failing_does_not_stop_others():
    import app

    counts = {"good": 0, "bad": 0}

    async def good():
        counts["good"] += 1

    async def bad():
        counts["bad"] += 1
        raise RuntimeError("boom")

    async def scenario():
        app.stop_event.clear()
        tasks = [asyncio.create_task(app.run_loop("GOOD", 0.01, good)),
                 asyncio.create_task(app.run_loop("BAD", 0.01, bad))]
        await asyncio.sleep(0.15)
        app.stop_event.set()
        await asyncio.wait(tasks, timeout=2)

    asyncio.run(scenario())
    assert counts["good"] >= 3 and counts["bad"] >= 3


# 32
def test_context_provider_failure_falls_back(flags):
    from tools.market_tools import FallbackMarketDataProvider

    flags()
    h = Harness()

    class Down:
        name = "tradingview_mcp"

        def get_market_context(self, symbol):
            raise ConnectionError("tradingview unavailable")

    h.services.context_market = FallbackMarketDataProvider([Down(), h.services.context_market.providers[0]],
                                                           log=lambda *a: None)
    bull_to_working(h)
    [a] = q("SELECT input_json FROM signal_analyses")
    assert json.loads(a["input_json"])["market_context"]["provider"] == "fake_binance_us"


# 33
def test_inconsistent_lifecycle_fails_closed(flags):
    flags()
    h = Harness()
    bull_open(h)
    conn = sqlite3.connect(config.db_path())
    conn.execute("INSERT INTO trades (trade_id, position_id, state, created_at, updated_at)"
                 " VALUES ('TR-orphan', 'PP-missing', 'WORKING', 't', 't')")
    conn.commit()
    h.at(20.5)
    h.cycle()
    assert q("SELECT state FROM trades WHERE trade_id = 'TR-orphan'")[0]["state"] == "WORKING"
    assert n("exit_requests", "trade_id = 'TR-orphan'") == 0
    assert n("risk_assessments", "position_id = 'PP-missing'") == 0


# 34 / 35
def test_app_import_does_not_start_discord_or_pipeline():
    import app
    assert app.client.user is None and app.pipeline is None


def test_tests_cannot_touch_production_db():
    assert config.db_path() != str(config.PROJECT_DIR / "trade_monitor.db")


# 38 / 39
def test_closed_position_never_assessed_again_and_cannot_reopen(flags):
    flags()
    h = Harness(risk_llm=ScriptedLLM("HOLD"))
    bull_open(h)
    h.feed.set("BTCUSDT", 20, 82500.0)
    h.at(20.5)
    h.cycle()
    assert trade(h)["state"] == "CLOSED"
    before = n("risk_assessments")
    h.feed.set("BTCUSDT", 25, 84500.0)
    for m in (25.5, 40.5, 60.5):
        h.at(m)
        h.cycle()
    assert n("risk_assessments") == before and trade(h)["state"] == "CLOSED"


# 40 — complete audit lineage
def test_end_to_end_audit_lineage(flags):
    flags()
    h = Harness()
    bull_open(h)
    h.feed.set("BTCUSDT", 20, 82500.0)
    h.at(20.5)
    h.cycle()

    [sig] = q("SELECT * FROM signals")
    [analysis] = q("SELECT * FROM signal_analyses WHERE signal_id = ?", (sig["id"],))
    [prop] = q("SELECT * FROM trade_proposals WHERE signal_id = ?", (sig["id"],))
    [entry] = q("SELECT * FROM execution_attempts WHERE purpose = 'ENTRY' AND proposal_key = ?", (prop["proposal_key"],))
    [entry_policy] = q("SELECT * FROM policy_decisions WHERE decision_id = ?", (entry["policy_decision_id"],))
    [entry_order] = q("SELECT * FROM paper_orders WHERE order_id = ?", (entry["broker_order_id"],))
    [position] = q("SELECT * FROM paper_positions WHERE position_id = ?", (entry_order["position_id"],))
    [tr] = q("SELECT * FROM trades WHERE execution_id = ? AND position_id = ?",
             (entry["execution_id"], position["position_id"]))
    [xr] = q("SELECT * FROM exit_requests WHERE trade_id = ?", (tr["trade_id"],))
    [assessment] = q("SELECT * FROM risk_assessments WHERE assessment_id = ?", (xr["risk_assessment_id"],))
    [exit_ex] = q("SELECT * FROM execution_attempts WHERE purpose = 'EXIT' AND exit_request_id = ?", (xr["exit_request_id"],))
    [exit_policy] = q("SELECT * FROM policy_decisions WHERE decision_id = ?", (exit_ex["policy_decision_id"],))
    [close_order] = q("SELECT * FROM paper_orders WHERE order_id = ?", (exit_ex["broker_order_id"],))

    assert sig["discord_message_id"] == "m1"
    assert analysis["signal_id"] == sig["id"]
    assert json.loads(prop["output_json"])["analysis_id"] == analysis["id"]
    assert entry_policy["purpose"] == "ENTRY" and entry_policy["approved"] == 1
    assert entry_order["client_order_id"] == entry["execution_key"] == f"entry:{prop['proposal_key']}"
    assert assessment["position_id"] == position["position_id"] and assessment["hard_rule_triggered"] == 1
    assert exit_policy["purpose"] == "EXIT" and exit_policy["approved"] == 1
    assert close_order["position_id"] == position["position_id"] and close_order["side"] == "SELL_TO_CLOSE"
    assert close_order["client_order_id"] == f"close:{position['position_id']}:full"
    assert tr["state"] == "CLOSED" and xr["status"] == "COMPLETED" and position["status"] == "CLOSED"
    assert tr["signal_id"] == sig["id"] and tr["proposal_key"] == prop["proposal_key"]


# 36 + paper-only safety scan
def _production_files():
    root = pathlib.Path(config.PROJECT_DIR)
    return [p for p in root.rglob("*.py")
            if not any(part in ("venv", "tests", "scripts", "__pycache__") for part in p.relative_to(root).parts)]


def _code(path):
    return "\n".join(l for l in path.read_text().splitlines() if not l.strip().startswith("#"))


def test_paper_only_safety_scan():
    from broker.base import BrokerBase, BrokerConfigError
    from broker.factory import create_broker
    import broker.paper_broker  # noqa: F401

    root = pathlib.Path(config.PROJECT_DIR)
    assert [c.__name__ for c in BrokerBase.__subclasses__()] == ["PaperBroker"]
    for mode in ("live", "LIVE", "testnet", "real", "production", "robinhood", "deribit"):
        os.environ["TRADING_MODE"] = mode
        try:
            with pytest.raises(BrokerConfigError):
                create_broker(object())
        finally:
            del os.environ["TRADING_MODE"]

    callers = {}
    for path in _production_files():
        code = _code(path)
        for call in (".submit_order(", ".close_position(", ".cancel_order("):
            if call in code:
                callers.setdefault(str(path.relative_to(root)), set()).add(call)
    # Only the execution service (and the broker's own internal close->submit) invoke the broker.
    assert callers == {"core/execution_service.py": {".submit_order(", ".close_position(", ".cancel_order("},
                       "broker/paper_broker.py": {".submit_order("}}

    for path in (root / "agents").glob("*.py"):
        code = _code(path)
        assert "broker" not in [w for l in code.splitlines() if l.startswith(("import", "from")) for w in re.split(r"[ .,]", l)]
        assert "execution_service" not in code

    # REDUCE has no execution path; auto exit is full-exit only.
    exec_code = _code(root / "core/execution_service.py")
    assert 'if xr["requested_action"] != "EXIT":' in exec_code and "NOT_EXECUTABLE_V1" in exec_code
    assert "requested_action = 'EXIT'" in _code(root / "trading_pipeline.py")

    names = [p.name.lower() for p in _production_files()]
    assert not [x for x in names if "robinhood" in x or "deribit" in x or "live" in x]
    classes = re.findall(r"^class (\w+)", "\n".join(_code(p) for p in _production_files()), re.M)
    assert [c for c in classes if "deribit" in c.lower() or "robinhood" in c.lower()] == ["MockDeribitChainProvider"]


def test_throttled_cycles_do_not_flap_reduce_or_renotify(flags):
    flags()
    llm = ScriptedLLM("REDUCE")
    h = Harness(risk_llm=llm)
    bull_open(h)                                                 # first (fresh) LLM run -> REDUCE
    assert trade(h)["state"] == "REDUCE_REQUESTED"
    events = n("trade_lifecycle_events")
    reduce_msgs = [m for m in h.trade_messages if m.startswith("🔻")]

    for minute in range(13, 25):                                 # 12 throttled minutes (interval 15)
        h.at(minute + 0.5)
        h.cycle()
    assert trade(h)["state"] == "REDUCE_REQUESTED"
    assert n("trade_lifecycle_events") == events
    assert [m for m in h.trade_messages if m.startswith("🔻")] == reduce_msgs == reduce_msgs[:1]
    assert n("exit_requests") == 1 and llm.calls == 1

    llm.recommendation = "HOLD"                                  # next FRESH run says HOLD
    h.at(30.5)
    h.cycle()
    assert trade(h)["state"] == "WORKING" and llm.calls == 2
    assert q("SELECT status FROM exit_requests")[0]["status"] == "SUPERSEDED"
