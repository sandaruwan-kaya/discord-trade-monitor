"""Exit execution (offline): Agent 4 ExitRequest -> exit policy -> one SELL_TO_CLOSE."""
import json
import os
import sqlite3
import subprocess
import sys
import threading

import pytest

import config
import database
from agents.trade_manager import TradeManager
from core.execution_service import ExecutionService, close_client_order_id, exit_execution_key

from test_execution_service import SpyBroker, service as exec_service, store_proposal
from test_policy_engine import NOW, Quotes, Spot
from test_trade_manager import add_assessment


class CloseSpyBroker(SpyBroker):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.closes = 0

    def close_position(self, *a, **k):
        self.closes += 1
        return super().close_position(*a, **k)


def rows(sql, params=()):
    conn = sqlite3.connect(config.db_path())
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params)]
    finally:
        conn.close()


def closes():
    return rows("SELECT * FROM paper_orders WHERE side = 'SELL_TO_CLOSE'")


class ClockQuotes(Quotes):
    """Fake chain provider whose quotes are stamped with the world's clock (fresh unless overridden)."""

    def __init__(self, clock):
        super().__init__()
        self.clock = clock

    def get_option_contract(self, symbol):
        from options_helpers import contract
        if self.error:
            raise self.error
        return contract(**{"contract_symbol": symbol, "as_of_ms": self.clock["now"], **self.kw})


class World:
    """Entry FILLED via ExecutionService -> Agent 4 trade -> Agent 3 EXIT -> Agent 4 ExitRequest."""

    def __init__(self, hard=True, codes=("UNDERLYING_INVALIDATION_HIT",), data=(), reasons=("THESIS_INTACT",)):
        store_proposal(limit_price=2260.0)
        self.clock = {"now": NOW}
        self.quotes = ClockQuotes(self.clock)
        clock = lambda: self.clock["now"]
        self.broker = CloseSpyBroker(self.quotes, now_ms=clock, log=lambda *a: None)
        self.svc = ExecutionService(self.broker, self.quotes, Spot(84200.0), now_ms=clock, log=lambda *a: None)
        assert self.svc.execute_entry("signal:1:entry").status == "ORDER_FILLED"
        self.tm = TradeManager(now_ms=clock, log=lambda *a: None)
        self.trade = self.tm.materialize_trade("entry:signal:1:entry")
        self.position_id = self.trade["position_id"]
        add_assessment(self.position_id, "EXIT", hard=hard, codes=list(codes), data=list(data), reasons=list(reasons))
        assert self.tm.process(self.trade["trade_id"]).next_state == "EXIT_REQUESTED"
        [self.request] = self.tm.get_exit_requests(self.trade["trade_id"])
        self.key = exit_execution_key(self.position_id)

    def restart(self, advance_ms=5 * 60_000):
        self.clock["now"] += advance_ms
        clock = lambda: self.clock["now"]
        self.broker = CloseSpyBroker(self.quotes, now_ms=clock, log=lambda *a: None)
        self.svc = ExecutionService(self.broker, self.quotes, Spot(84200.0), now_ms=clock, log=lambda *a: None)

    def exit(self):
        return self.svc.execute_exit(self.request.exit_request_id)

    def trade_state(self):
        return self.tm.get_trade(self.trade["trade_id"])["state"]

    def request_row(self):
        return rows("SELECT * FROM exit_requests WHERE exit_request_id = ?", (self.request.exit_request_id,))[0]

    def lifecycle(self):
        return [(e["from_state"], e["to_state"]) for e in self.tm.get_events(self.trade["trade_id"])]


# 1 / 2 / 22 / 21
@pytest.mark.parametrize("hard,codes,reasons", [
    (True, ("UNDERLYING_INVALIDATION_HIT",), ("THESIS_INTACT",)),
    (False, (), ("NEAR_INVALIDATION",)),
])
def test_full_exit_with_valid_quote_closes(hard, codes, reasons):
    w = World(hard=hard, codes=codes, reasons=reasons)
    r = w.exit()

    assert r.ok and r.status == "ORDER_FILLED" and r.purpose == "EXIT"
    [close] = closes()
    assert close["client_order_id"] == close_client_order_id(w.position_id)
    assert close["quantity"] == 2 and close["status"] == "FILLED"
    assert rows("SELECT status, quantity FROM paper_positions")[0] == {"status": "CLOSED", "quantity": 0}
    assert w.trade_state() == "CLOSED"
    assert w.lifecycle()[-2:] == [("EXIT_REQUESTED", "CLOSE_ORDER_PENDING"), ("CLOSE_ORDER_PENDING", "CLOSED")]
    req = w.request_row()
    assert req["status"] == "COMPLETED" and req["completed_at"] and req["execution_status"] == "ORDER_FILLED"


# 3 / 4
def test_fresh_bid_is_the_close_limit_not_mark_mid_or_ask():
    w = World()
    w.quotes.kw = {"bid": 2400.0, "ask": 2480.0, "mark": 2440.0}
    w.exit()
    [close] = closes()
    assert close["limit_price"] == 2400.0 and close["fill_price"] == 2400.0
    assert close["limit_price"] not in (2440.0, 2480.0)


# 5 / 6 / 7
@pytest.mark.parametrize("env", [
    {"TRADING_KILL_SWITCH": "1"},
    {"MAX_DAILY_LOSS": "0.01"},
    {"MAX_OPEN_POSITIONS": "1"},
    {"MAX_PREMIUM_PER_TRADE": "1"},
    {"AGENT2_MAX_UNDERLYING_MOVE_PCT": "0"},
])
def test_entry_limits_and_kill_switch_do_not_block_exit(monkeypatch, env):
    w = World()
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    r = w.exit()
    assert r.status == "ORDER_FILLED" and w.trade_state() == "CLOSED"
    decision = rows("SELECT * FROM policy_decisions WHERE decision_id = ?", (r.policy_decision_id,))[0]
    assert decision["purpose"] == "EXIT" and decision["approved"] == 1
    snap = json.loads(decision["policy_snapshot_json"])
    if "TRADING_KILL_SWITCH" in env:
        assert snap["kill_switch"]["env_enabled"] is True                 # recorded, not blocking


# 8 / 9 / 10
@pytest.mark.parametrize("break_quote", ["missing", "zero_bid", "no_bid", "crossed", "stale"])
def test_bad_quote_waits_then_same_execution_closes(break_quote):
    w = World(data=("QUOTE_UNAVAILABLE",))
    if break_quote == "missing":
        w.quotes.error = ConnectionError("chain down")
    else:
        w.quotes.kw = {"zero_bid": {"bid": 0.0}, "no_bid": {"bid": None},
                       "crossed": {"bid": 2300.0, "ask": 2250.0}, "stale": {"as_of_ms": NOW - 120_000}}[break_quote]

    r = w.exit()
    assert r.status == "WAITING_FOR_QUOTE" and r.error_code == "EXIT_BLOCKED_DATA"
    assert closes() == [] and w.broker.closes == 0
    assert w.trade_state() == "EXIT_REQUESTED"
    assert w.request_row()["status"] == "ACTIVE" and w.request_row()["execution_status"] == "WAITING_FOR_QUOTE"
    first_id = r.execution_id

    w.exit()                                                              # still broken: still waiting, no order
    assert closes() == []

    w.quotes.error, w.quotes.kw = None, {}
    [r2] = [x for x in w.svc.reconcile_pending_executions() if x.purpose == "EXIT"]
    assert r2.status == "ORDER_FILLED" and r2.execution_id == first_id    # same execution continues
    assert len(closes()) == 1 and w.trade_state() == "CLOSED"
    assert rows("SELECT COUNT(*) n FROM execution_attempts WHERE purpose = 'EXIT'")[0]["n"] == 1


# 11
def test_reduce_request_is_not_executable_v1():
    w = World()
    conn = sqlite3.connect(config.db_path())
    conn.execute("INSERT INTO exit_requests (exit_request_id, request_key, trade_id, position_id, requested_action,"
                 " requested_quantity, hard_exit, reason_codes_json, data_issue_codes_json, status, created_at,"
                 " updated_at) VALUES ('XR-R', 'reduce:x', ?, ?, 'REDUCE', NULL, 0, '[]', '[]', 'ACTIVE', 't', 't')",
                 (w.trade["trade_id"], w.position_id))
    conn.commit()
    r = w.svc.execute_exit("XR-R")
    assert r.status == "NOT_EXECUTABLE_V1" and not r.persisted
    assert closes() == [] and w.broker.closes == 0


# 12
def test_already_closed_position_reconciles_with_zero_new_orders():
    w = World()
    w.broker.close_position(w.position_id, 2200.0, "external-close")      # closed elsewhere (test setup)
    before = len(closes())

    r = w.exit()

    assert r.status == "ALREADY_CLOSED" and len(closes()) == before
    assert w.trade_state() == "CLOSED" and w.request_row()["status"] == "COMPLETED"


# 13
def test_quantity_changed_fails_closed():
    w = World()
    w.broker.close_position(w.position_id, 2200.0, "partial", quantity=1)  # something else closed 1 of 2
    r = w.exit()
    assert r.status == "POLICY_REJECTED" and "position_quantity_changed" in r.policy_reasons
    assert [c["client_order_id"] for c in closes()] == ["partial"]         # no new close
    assert w.request_row()["requested_quantity"] == 2                      # request untouched
    assert w.request_row()["status"] == "ACTIVE" and w.trade_state() == "EXIT_REQUESTED"


# 14
def test_wrong_contract_rejected():
    w = World()
    w.quotes.get_option_contract = lambda s: ClockQuotes(w.clock).get_option_contract("BTC-9OCT26-84000-C")
    r = w.exit()
    assert r.status == "POLICY_REJECTED" and "contract_mismatch" in r.policy_reasons
    assert closes() == []


# 15 / 26 / 27
def test_duplicate_execute_exit_one_close_order():
    w = World()
    a, b = w.exit(), w.exit()
    assert b.replay and a.execution_id == b.execution_id and a.broker_order_id == b.broker_order_id
    assert len(closes()) == 1 and w.broker.closes == 1
    [ex] = rows("SELECT * FROM execution_attempts WHERE purpose = 'EXIT'")
    assert ex["exit_request_id"] == w.request.exit_request_id                   # 26
    assert ex["execution_key"] == f"exit:{w.position_id}:full"
    [close] = closes()
    assert close["order_id"] == ex["broker_order_id"] and close["position_id"] == w.position_id   # 27


# 16
def test_concurrent_exit_calls_one_close_order():
    w = World()
    services = [ExecutionService(CloseSpyBroker(w.quotes, now_ms=lambda: NOW, log=lambda *a: None), w.quotes,
                                 Spot(84200.0), now_ms=lambda: NOW, log=lambda *a: None) for _ in range(8)]
    barrier, results = threading.Barrier(8), []

    def run(s):
        barrier.wait()
        results.append(s.execute_exit(w.request.exit_request_id))

    ts = [threading.Thread(target=run, args=(s,)) for s in services]
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    assert len(results) == 8 and len(closes()) == 1
    assert rows("SELECT COUNT(*) n FROM execution_attempts WHERE purpose = 'EXIT'")[0]["n"] == 1


# 17 / 18
def test_crash_after_broker_fill_before_persistence_reconciles_one_close():
    w = World()
    real = w.broker.close_position

    def close_then_die(*a, **k):
        real(*a, **k)
        raise SystemExit("killed after the broker filled the close")

    w.broker.close_position = close_then_die
    with pytest.raises(SystemExit):
        w.exit()
    assert w.svc.get_attempt(w.key)["status"] == "SUBMITTING" and len(closes()) == 1
    assert w.trade_state() == "EXIT_REQUESTED"

    w.restart()
    r = w.exit()
    assert r.status == "ORDER_FILLED" and w.broker.closes == 0 and len(closes()) == 1
    assert w.trade_state() == "CLOSED"
    assert w.lifecycle()[-2:] == [("EXIT_REQUESTED", "CLOSE_ORDER_PENDING"), ("CLOSE_ORDER_PENDING", "CLOSED")]


def test_crash_before_broker_call_resumes_with_same_client_order_id():
    w = World()

    def die(*a, **k):
        raise SystemExit("killed before the broker saw the close")

    w.broker.close_position = die
    with pytest.raises(SystemExit):
        w.exit()
    assert closes() == []

    w.restart()
    [r] = [x for x in w.svc.reconcile_pending_executions() if x.purpose == "EXIT"]
    assert r.status == "INTERRUPTED"                                      # confirmed no close exists
    r = w.exit()                                                          # exit still required: resumes
    assert r.status == "ORDER_FILLED" and len(closes()) == 1
    assert closes()[0]["client_order_id"] == close_client_order_id(w.position_id)


def test_submitting_within_timeout_is_not_marked_interrupted():
    w = World()
    w.broker.close_position = lambda *a, **k: (_ for _ in ()).throw(SystemExit("killed"))
    with pytest.raises(SystemExit):
        w.exit()
    assert w.exit().status == "SUBMITTING"                                # another caller may be mid-submit


# 19 / 20
def test_pending_close_stays_one_order_then_fills():
    w = World()
    w.quotes.kw = {"bid": 2200.0, "ask": 2260.0}
    real = w.broker.close_position
    w.broker.close_position = lambda pid, limit_price, client_order_id, quantity=None: real(
        pid, limit_price + 50, client_order_id, quantity)                  # force a resting close (test only)
    r = w.exit()
    assert r.status == "ORDER_PENDING"
    assert w.trade_state() == "CLOSE_ORDER_PENDING" and w.request_row()["status"] == "ACTIVE"

    for _ in range(4):
        w.svc.reconcile_pending_executions()
        w.exit()
    assert len(closes()) == 1 and closes()[0]["status"] == "PENDING"

    w.quotes.kw = {"bid": 2260.0, "ask": 2300.0}
    w.svc.reconcile_pending_executions()
    assert w.svc.get_attempt(w.key)["status"] == "ORDER_FILLED"
    assert w.trade_state() == "CLOSED" and w.request_row()["status"] == "COMPLETED"
    assert len(closes()) == 1


# 23 / 24
def test_no_exit_requirement_cannot_close():
    w = World()
    assert w.svc.execute_exit("XR-does-not-exist").status == "REFUSED"
    conn = sqlite3.connect(config.db_path())
    conn.execute("UPDATE exit_requests SET status = 'SUPERSEDED'")
    conn.commit()
    r = w.exit()
    assert r.status == "REFUSED" and r.error_code == "EXIT_REQUEST_NOT_ACTIVE" and closes() == []


def test_completed_request_cannot_create_another_order():
    w = World()
    w.exit()
    assert w.request_row()["status"] == "COMPLETED"
    r = w.exit()
    assert r.replay and len(closes()) == 1


def test_trade_not_exit_requested_is_rejected():
    w = World()
    conn = sqlite3.connect(config.db_path())
    conn.execute("UPDATE trades SET state = 'WORKING'")
    conn.commit()
    r = w.exit()
    assert r.status == "POLICY_REJECTED" and "trade_not_exit_requested" in r.policy_reasons and closes() == []


def test_existing_foreign_close_order_blocks():
    w = World()
    w.broker.close_position = w.broker.close_position
    real = CloseSpyBroker(w.quotes, now_ms=lambda: NOW, log=lambda *a: None)
    real.close_position(w.position_id, 9999.0, "someone-else", quantity=2)  # resting foreign close
    r = w.exit()
    assert r.status == "POLICY_REJECTED" and "close_order_already_exists" in r.policy_reasons
    assert [c["client_order_id"] for c in closes()] == ["someone-else"]


def test_non_paper_mode_cannot_close(monkeypatch):
    w = World()
    monkeypatch.setenv("TRADING_MODE", "live")
    r = w.exit()
    assert r.status == "POLICY_REJECTED" and "trading_mode_not_paper" in r.policy_reasons and closes() == []


# 25
def test_exit_policy_decision_persisted():
    w = World()
    r = w.exit()
    [d] = rows("SELECT * FROM policy_decisions WHERE decision_id = ?", (r.policy_decision_id,))
    decision = json.loads(d["decision_json"])
    assert d["purpose"] == "EXIT" and d["approved"] == 1
    assert decision["exit_limit_price"] == 2200.0 and decision["exit_quantity"] == 2
    inp = json.loads(d["input_json"])
    assert inp["exit_request"]["exit_request_id"] == w.request.exit_request_id
    assert inp["position"]["quantity"] == 2 and inp["quote"]["bid"] == 2200.0


def test_waiting_decision_is_retryable_and_persisted():
    w = World()
    w.quotes.error = ConnectionError("down")
    r = w.exit()
    [d] = rows("SELECT * FROM policy_decisions WHERE decision_id = ?", (r.policy_decision_id,))
    assert json.loads(d["decision_json"])["retryable"] is True
    assert json.loads(d["reasons_json"]) == ["quote_unavailable"]


def test_migration_v8_preserves_entry_executions():
    w = World()
    [entry] = rows("SELECT * FROM execution_attempts WHERE purpose = 'ENTRY'")
    assert entry["status"] == "ORDER_FILLED" and entry["exit_request_id"] is None
    conn = sqlite3.connect(config.db_path())
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO execution_attempts (execution_id, execution_key, proposal_key, purpose, status,"
                     " client_order_id, created_at, updated_at) VALUES ('x', 'x', 'p', 'ENTRY', 'WAITING_FOR_QUOTE',"
                     " 'x', 't', 't')")


# 28 / 29 / 30
def _code(rel):
    text = (config.PROJECT_DIR / rel).read_text()
    return "\n".join(l for l in text.splitlines() if not l.strip().startswith("#"))


def test_agent4_still_cannot_execute():
    for rel in ("agents/trade_manager.py", "core/trade_lifecycle.py"):
        code = _code(rel)
        for banned in ("execution_service", "ExecutionService", "execute_exit", "from broker", "import broker",
                       ".close_position(", ".submit_order("):
            assert banned not in code, (rel, banned)


def test_exit_execution_has_no_llm_dependency():
    probe = ("import sys; import core.execution_service, core.lifecycle_sync; "
             "bad = sorted(m for m in sys.modules if m in ('openai', 'llm_client', 'llm_parser') "
             "or m.startswith('openai.') or m.startswith('agents')); print(','.join(bad))")
    result = subprocess.run([sys.executable, "-c", probe], cwd=config.PROJECT_DIR, env=dict(os.environ),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0 and result.stdout.strip() == "", result.stderr


def test_no_live_broker_path_exists():
    import pathlib
    import broker as pkg
    from broker.base import BrokerBase
    assert sorted(p.name for p in pathlib.Path(pkg.__file__).parent.glob("*.py")) == \
        ["__init__.py", "base.py", "factory.py", "paper_broker.py"]
    assert [c.__name__ for c in BrokerBase.__subclasses__()] == ["PaperBroker"]
    imports = [l for l in _code("core/execution_service.py").splitlines() if l.startswith(("import ", "from "))]
    assert [l for l in imports if "broker" in l] == ["from broker.base import BrokerBase"]
    from broker.base import BrokerConfigError
    from broker.factory import create_broker
    for mode in ("live", "testnet", "real"):
        os.environ["TRADING_MODE"] = mode
        try:
            with pytest.raises(BrokerConfigError):
                create_broker(Quotes())
        finally:
            del os.environ["TRADING_MODE"]
