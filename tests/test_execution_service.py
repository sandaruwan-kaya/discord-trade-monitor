"""Execution Service (offline): fresh policy -> paper broker, exactly once."""
import json
import os
import sqlite3
import subprocess
import sys
import threading

import pytest

import config
import database
from broker.paper_broker import PaperBroker
from core.execution_service import ExecutionService, entry_execution_key
from schemas import OrderRequest

from test_policy_engine import NOW, SYMBOL, Quotes, Spot, proposal, store_signal
from options_helpers import contract


KEY = "signal:1:entry"
EXEC_KEY = "entry:signal:1:entry"


class SpyBroker(PaperBroker):
    """Real PaperBroker that counts submissions."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.submits = 0

    def submit_order(self, request):
        self.submits += 1
        return super().submit_order(request)


class Clock:
    def __init__(self, now=NOW):
        self.now = now

    def __call__(self):
        return self.now


def store_proposal(**kw):
    store_signal()
    p = proposal(**kw)
    database.claim_proposal(p.proposal_key, p.signal_id)
    database.finish_proposal(p.proposal_key, "PROPOSED", proposal=p)
    return p


def service(spot=84200.0, quotes=None, clock=None):
    quotes = quotes or Quotes()
    clock = clock or Clock()
    broker = SpyBroker(quotes, now_ms=clock, log=lambda *a: None)
    return ExecutionService(broker, quotes, Spot(spot), now_ms=clock, log=lambda *a: None), broker, quotes, clock


def rows(sql, params=()):
    conn = sqlite3.connect(config.db_path())
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params)]
    finally:
        conn.close()


def order_count():
    return rows("SELECT COUNT(*) AS n FROM paper_orders")[0]["n"]


# 1
def test_clean_proposal_approved_and_filled():
    store_proposal(limit_price=2260.0)                       # at the ask -> fills
    svc, broker, *_ = service()

    r = svc.execute_entry(KEY)

    assert r.ok and r.status == "ORDER_FILLED"
    assert r.broker_order_status == "FILLED"
    assert r.client_order_id == EXEC_KEY
    [order] = rows("SELECT * FROM paper_orders")
    assert order["client_order_id"] == EXEC_KEY
    assert (order["side"], order["quantity"], order["limit_price"], order["contract_symbol"]) == \
        ("BUY_TO_OPEN", 2, 2260.0, SYMBOL)
    assert broker.submits == 1


# 2
def test_clean_proposal_broker_pending():
    store_proposal()                                          # limit 2230 < ask 2260
    svc, *_ = service()

    r = svc.execute_entry(KEY)

    assert r.ok and r.status == "ORDER_PENDING"
    assert rows("SELECT status FROM paper_orders")[0]["status"] == "PENDING"


# 3-5: time-of-check/time-of-use: the market moved after Agent 2
@pytest.mark.parametrize("spot,reason", [
    (84600.0, "stale_entry_move_extended"),
    (86100.0, "target_reached"),
    (82500.0, "invalidation_hit"),
])
def test_fresh_policy_rejects_after_market_moves(spot, reason):
    store_proposal()
    svc, broker, *_ = service(spot=spot)

    r = svc.execute_entry(KEY)

    assert r.status == "POLICY_REJECTED" and not r.ok
    assert reason in r.policy_reasons
    assert broker.submits == 0 and order_count() == 0


# 6
def test_kill_switch_blocks_submission(monkeypatch):
    monkeypatch.setenv("TRADING_KILL_SWITCH", "1")
    store_proposal()
    svc, broker, *_ = service()

    r = svc.execute_entry(KEY)

    assert r.status == "POLICY_REJECTED"
    assert r.policy_reasons == ["kill_switch_enabled"]
    assert broker.submits == 0


# 7
def test_daily_loss_blocks_submission(monkeypatch):
    monkeypatch.setenv("MAX_DAILY_LOSS", "20")
    store_proposal()
    quotes = Quotes()
    other = PaperBroker(quotes, now_ms=lambda: NOW, log=lambda *a: None)
    pos = other.submit_order(OrderRequest(client_order_id="old", signal_id=9, proposal_key="signal:9:entry",
                                          contract_symbol=SYMBOL, side="BUY_TO_OPEN", quantity=2,
                                          limit_price=2260.0)).position
    quotes.kw = {"bid": 1100.0, "ask": 1150.0}
    other.close_position(pos.position_id, 1100.0, "old-close")          # realized -$23.20
    quotes.kw = {}

    svc, broker, *_ = service(quotes=quotes)
    r = svc.execute_entry(KEY)

    assert r.policy_reasons == ["max_daily_loss_reached"]
    assert broker.submits == 0


# 8
def test_open_position_limit_blocks_submission(monkeypatch):
    monkeypatch.setenv("MAX_OPEN_POSITIONS", "1")
    store_proposal()
    quotes = Quotes()
    PaperBroker(quotes, now_ms=lambda: NOW, log=lambda *a: None).submit_order(OrderRequest(
        client_order_id="other", signal_id=9, proposal_key="signal:9:entry", contract_symbol=SYMBOL,
        side="BUY_TO_OPEN", quantity=1, limit_price=2260.0))

    svc, broker, *_ = service(quotes=quotes)
    r = svc.execute_entry(KEY)

    assert r.policy_reasons == ["max_open_positions_reached"]
    assert broker.submits == 0


# 9
def test_fresh_option_quote_fails_policy():
    store_proposal()
    svc, broker, *_ = service(quotes=Quotes(bid=1800.0))                 # spread 22%

    r = svc.execute_entry(KEY)

    assert r.policy_reasons == ["spread_too_wide"]
    assert broker.submits == 0


# 10
def test_proposal_not_proposed_is_rejected():
    store_signal()
    database.claim_proposal(KEY, 1)
    database.finish_proposal(KEY, "SKIPPED", reason="stale_entry_move_extended: ...")
    svc, broker, *_ = service()

    r = svc.execute_entry(KEY)

    assert r.status == "FAILED" and r.error_code == "PROPOSAL_NOT_PROPOSED"
    assert broker.submits == 0


def test_pending_proposal_is_refused_without_consuming_the_key():
    store_signal()
    database.claim_proposal(KEY, 1)                                      # Agent 2 still running
    svc, *_ = service()

    r = svc.execute_entry(KEY)

    assert r.status == "REFUSED" and r.error_code == "PROPOSAL_NOT_READY" and not r.persisted
    assert svc.get_attempt(EXEC_KEY) is None


# 11
def test_missing_proposal_rejected():
    svc, broker, *_ = service()

    r = svc.execute_entry("signal:404:entry")

    assert r.status == "REFUSED" and r.error_code == "PROPOSAL_NOT_FOUND" and not r.persisted
    assert broker.submits == 0
    assert rows("SELECT COUNT(*) AS n FROM execution_attempts")[0]["n"] == 0


# 12
def test_malformed_proposal_rejected():
    store_proposal()
    conn = sqlite3.connect(config.db_path())
    conn.execute("UPDATE trade_proposals SET output_json = '{\"status\": \"PROPOSED\"}' WHERE proposal_key = ?", (KEY,))
    conn.commit()
    svc, broker, *_ = service()

    r = svc.execute_entry(KEY)

    assert r.status == "FAILED" and r.error_code == "PROPOSAL_MALFORMED"
    assert broker.submits == 0


# 13
def test_execution_key_is_unique():
    store_proposal()
    svc, *_ = service()
    svc.execute_entry(KEY)

    conn = sqlite3.connect(config.db_path())
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO execution_attempts (execution_id, execution_key, proposal_key, purpose, status,"
            " client_order_id, created_at, updated_at) VALUES ('EX-dup', ?, ?, 'ENTRY', 'CLAIMED', 'other', 't', 't')",
            (EXEC_KEY, KEY),
        )
    assert entry_execution_key(KEY) == EXEC_KEY


# 14
def test_two_calls_create_one_broker_order():
    store_proposal(limit_price=2260.0)
    svc, broker, *_ = service()

    first = svc.execute_entry(KEY)
    second = svc.execute_entry(KEY)

    assert second.replay and not first.replay
    assert second.execution_id == first.execution_id
    assert second.broker_order_id == first.broker_order_id
    assert second.status == first.status == "ORDER_FILLED"
    assert broker.submits == 1 and order_count() == 1


# 15
def test_concurrent_execution_claims_create_one_broker_order():
    store_proposal(limit_price=2260.0)
    quotes = Quotes()
    services = [service(quotes=quotes)[0] for _ in range(8)]
    results, barrier = [], threading.Barrier(len(services))

    def run(svc):
        barrier.wait()
        results.append(svc.execute_entry(KEY))

    threads = [threading.Thread(target=run, args=(s,)) for s in services]
    [t.start() for t in threads]
    [t.join(timeout=30) for t in threads]

    assert len(results) == 8
    assert sum(1 for r in results if not r.replay) == 1
    assert order_count() == 1
    assert rows("SELECT COUNT(*) AS n FROM execution_attempts")[0]["n"] == 1
    assert rows("SELECT COUNT(*) AS n FROM policy_decisions")[0]["n"] == 1


# 16
def test_restart_returns_existing_execution():
    store_proposal(limit_price=2260.0)
    before, *_ = service()
    first = before.execute_entry(KEY)
    del before

    after, broker, *_ = service()                                      # "restart"
    again = after.execute_entry(KEY)

    assert again.replay and again.execution_id == first.execution_id
    assert broker.submits == 0 and order_count() == 1


# 17
def test_crash_after_submit_before_persist_reconciles_by_client_order_id():
    """Crash point D: broker has the order, execution row still says SUBMITTING."""
    store_proposal(limit_price=2260.0)
    svc, broker, quotes, clock = service()

    real_submit = broker.submit_order

    def submit_then_die(request):
        real_submit(request)
        raise SystemExit("process killed after broker accepted the order")

    broker.submit_order = submit_then_die
    with pytest.raises(SystemExit):
        svc.execute_entry(KEY)

    assert svc.get_attempt(EXEC_KEY)["status"] == "SUBMITTING"
    assert order_count() == 1

    clock.now += 5 * 60_000                                           # restart after the claim timeout
    restarted, broker2, *_ = service(quotes=quotes, clock=clock)
    [r] = restarted.reconcile_pending_executions()

    assert r.status == "ORDER_FILLED"
    assert r.broker_order_id == rows("SELECT order_id FROM paper_orders")[0]["order_id"]
    assert broker2.submits == 0 and order_count() == 1                # never re-submitted
    assert restarted.execute_entry(KEY).replay
    events = [e["to_status"] for e in restarted.get_events(r.execution_id)]
    assert events == ["CLAIMED", "SUBMITTING", "ORDER_FILLED"]


def test_crash_after_submitting_before_broker_call_becomes_interrupted():
    """Crash point C: SUBMITTING persisted, broker never reached."""
    store_proposal(limit_price=2260.0)
    svc, broker, quotes, clock = service()

    def die(request):
        raise SystemExit("killed before the broker saw anything")

    broker.submit_order = die
    with pytest.raises(SystemExit):
        svc.execute_entry(KEY)

    clock.now += 5 * 60_000
    restarted, broker2, *_ = service(quotes=quotes, clock=clock)
    [r] = restarted.reconcile_pending_executions()

    assert r.status == "INTERRUPTED" and r.error_code == "SUBMIT_UNCONFIRMED"
    assert order_count() == 0 and broker2.submits == 0
    assert restarted.execute_entry(KEY).status == "INTERRUPTED"      # not retried


def test_crash_after_claim_becomes_interrupted_not_retried():
    """Crash points A/B: CLAIMED, policy maybe evaluated, nothing submitted."""
    store_proposal(limit_price=2260.0)
    svc, broker, quotes, clock = service()
    svc._run_claimed = lambda *a: (_ for _ in ()).throw(SystemExit("killed during policy"))
    with pytest.raises(SystemExit):
        svc.execute_entry(KEY)
    assert svc.get_attempt(EXEC_KEY)["status"] == "CLAIMED"

    fresh, *_ = service(quotes=quotes, clock=clock)
    assert fresh.reconcile_execution(EXEC_KEY).status == "CLAIMED"     # within timeout: still in progress

    clock.now += 5 * 60_000
    r = fresh.reconcile_execution(EXEC_KEY)
    assert r.status == "INTERRUPTED" and r.error_code == "CLAIM_INTERRUPTED"
    assert order_count() == 0


# 18
def test_broker_replay_is_idempotent():
    store_proposal(limit_price=2260.0)
    svc, broker, *_ = service()
    r = svc.execute_entry(KEY)

    replay = broker.submit_order(OrderRequest(
        client_order_id=EXEC_KEY, signal_id=1, proposal_key=KEY, contract_symbol=SYMBOL,
        side="BUY_TO_OPEN", quantity=2, limit_price=2260.0))

    assert replay.idempotent_replay and replay.order.order_id == r.broker_order_id
    assert order_count() == 1


# 19
def test_policy_rejected_never_calls_broker(monkeypatch):
    store_proposal()
    svc, broker, *_ = service(spot=84600.0)
    monkeypatch.setattr(broker, "submit_order", lambda *a, **k: pytest.fail("broker called"))

    assert svc.execute_entry(KEY).status == "POLICY_REJECTED"
    assert svc.execute_entry(KEY).status == "POLICY_REJECTED"            # replay, still no broker


# 20
def test_pending_execution_stays_one_order_over_repeated_reconciliation():
    store_proposal()
    svc, broker, *_ = service()
    svc.execute_entry(KEY)

    for _ in range(5):
        svc.reconcile_pending_executions()
        svc.reconcile_execution(EXEC_KEY, process_pending=True)

    assert svc.get_attempt(EXEC_KEY)["status"] == "ORDER_PENDING"
    assert order_count() == 1 and broker.submits == 1
    assert rows("SELECT limit_price FROM paper_orders")[0]["limit_price"] == 2230.0   # never repriced


# 21
def test_pending_later_becomes_filled():
    store_proposal()
    svc, broker, quotes, _ = service()
    r = svc.execute_entry(KEY)
    assert r.status == "ORDER_PENDING"

    quotes.kw = {"bid": 2180.0, "ask": 2225.0}                          # market comes to the limit
    [r2] = svc.reconcile_pending_executions()

    assert r2.status == "ORDER_FILLED"
    assert r2.broker_order_id == r.broker_order_id
    assert rows("SELECT fill_price FROM paper_orders")[0]["fill_price"] == 2225.0
    assert broker.submits == 1


# 22
def test_proposal_is_never_modified():
    store_proposal()
    before = database.get_proposal(KEY)
    svc, *_ = service()
    svc.execute_entry(KEY)
    svc.reconcile_pending_executions()
    assert database.get_proposal(KEY) == before


# 23
def test_execution_service_has_no_llm_or_agent_dependency():
    probe = (
        "import sys; import core.execution_service; "
        "bad = sorted(m for m in sys.modules if m == 'openai' or m.startswith('openai.') "
        "or m in ('llm_client', 'llm_parser') or m.startswith('agents')); print(','.join(bad))"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=config.PROJECT_DIR, env=dict(os.environ),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def _code(path):
    text = (config.PROJECT_DIR / path).read_text()
    return "\n".join(l for l in text.splitlines() if not l.strip().startswith("#"))


# 24
def test_agent2_and_app_layers_still_cannot_reach_broker_or_execution():
    for rel in ["agents/options_agent.py", "agents/signal_agent.py", "signal_service.py", "app.py",
                "tools/options_tools.py", "tools/market_tools.py", "signal_facts.py"]:
        code = _code(rel)
        for banned in ("import broker", "from broker", "execution_service", "ExecutionService",
                       ".submit_order(", ".close_position(", ".cancel_order("):
            assert banned not in code, (rel, banned)


# 25
def test_policy_engine_still_cannot_submit():
    code = _code("core/policy_engine.py")
    for banned in ("import broker", "from broker", "execution_service", ".submit_order(",
                   ".close_position(", ".cancel_order("):
        assert banned not in code


def test_execution_service_is_the_only_core_broker_caller():
    code = _code("core/execution_service.py")
    assert "self.broker.submit_order(" in code
    imports = [l for l in code.splitlines() if l.startswith(("import ", "from "))]
    assert not [l for l in imports if "llm" in l or "openai" in l or "agents" in l], imports


# 26 / 27
def test_policy_decision_and_broker_order_linked():
    store_proposal(limit_price=2260.0)
    svc, *_ = service()
    r = svc.execute_entry(KEY)

    [attempt] = rows("SELECT * FROM execution_attempts")
    [decision] = rows("SELECT * FROM policy_decisions WHERE decision_id = ?", (attempt["policy_decision_id"],))
    [order] = rows("SELECT * FROM paper_orders WHERE order_id = ?", (attempt["broker_order_id"],))

    assert decision["approved"] == 1 and decision["proposal_key"] == KEY
    assert order["client_order_id"] == attempt["client_order_id"] == EXEC_KEY
    assert order["proposal_key"] == KEY
    assert r.policy_decision_id == attempt["policy_decision_id"]


def test_rejected_execution_links_its_policy_decision():
    store_proposal()
    svc, *_ = service(spot=84600.0)
    r = svc.execute_entry(KEY)
    [decision] = rows("SELECT * FROM policy_decisions WHERE decision_id = ?", (r.policy_decision_id,))
    assert decision["approved"] == 0
    assert json.loads(decision["reasons_json"]) == ["stale_entry_move_extended"]


# 28
def test_broker_error_on_submit_does_not_corrupt_db():
    store_proposal(limit_price=2260.0)
    svc, broker, *_ = service()

    def down(request):
        raise ConnectionError("paper broker unavailable")

    broker.submit_order = down
    r = svc.execute_entry(KEY)

    assert r.status == "INTERRUPTED" and r.error_code == "SUBMIT_UNCONFIRMED"
    assert order_count() == 0
    assert rows("SELECT COUNT(*) AS n FROM paper_fills")[0]["n"] == 0


def test_provider_error_during_fill_leaves_order_pending():
    store_proposal(limit_price=2260.0)
    calls = {"n": 0}

    class FlakyQuotes(Quotes):
        def get_option_contract(self, symbol):
            calls["n"] += 1
            if calls["n"] >= 2:                                        # policy quote OK, broker fill quote fails
                raise TimeoutError("chain provider timeout")
            return super().get_option_contract(symbol)

    svc, *_ = service(quotes=FlakyQuotes())
    r = svc.execute_entry(KEY)

    assert r.status == "ORDER_PENDING"
    assert order_count() == 1 and rows("SELECT COUNT(*) AS n FROM paper_fills")[0]["n"] == 0


# 29
def test_transaction_rollback_leaves_execution_state_consistent():
    store_proposal(limit_price=2260.0)
    svc, broker, *_ = service()
    real_event = svc._event

    def fail_on_submitting(conn, execution_id, from_status, to_status, details, now):
        if to_status == "SUBMITTING":
            raise sqlite3.OperationalError("disk I/O error")
        return real_event(conn, execution_id, from_status, to_status, details, now)

    svc._event = fail_on_submitting
    r = svc.execute_entry(KEY)

    # The SUBMITTING update rolled back with its event; the broker was never called.
    assert r.status == "FAILED" and r.error_code == "INTERNAL_ERROR"
    assert broker.submits == 0 and order_count() == 0
    events = [e["to_status"] for e in svc.get_events(r.execution_id)]
    assert events == ["CLAIMED", "FAILED"]


# 30
def test_non_paper_mode_cannot_execute(monkeypatch):
    store_proposal(limit_price=2260.0)
    svc, broker, *_ = service()                                        # built while paper
    monkeypatch.setenv("TRADING_MODE", "live")

    r = svc.execute_entry(KEY)

    assert r.status == "POLICY_REJECTED"
    assert "trading_mode_not_paper" in r.policy_reasons
    assert broker.submits == 0 and order_count() == 0
    with pytest.raises(Exception):
        PaperBroker(Quotes())
