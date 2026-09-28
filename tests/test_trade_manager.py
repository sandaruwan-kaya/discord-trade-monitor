"""Agent 4 — trade lifecycle (offline, deterministic, no LLM)."""
import json
import os
import sqlite3
import subprocess
import sys

import pytest

import config
import database
from agents.trade_manager import TradeManager
from core import trade_lifecycle as tl
from schemas import (
    BrokerOrderState,
    BrokerPositionState,
    ExitRequestSummary,
    RiskAssessmentSummary,
    TradeManagementInput,
)

from test_policy_engine import NOW


POS = "PP-1"


def ra(rec="HOLD", aid="RA-1", hard=False, codes=None, data=None, reasons=None, pnl=5.0, dist=2.0, pos=POS,
       risk_state=None, llm_used=None, llm_status=None):
    default_state = "HARD_EXIT" if hard else ("DATA_UNAVAILABLE" if rec == "DATA_UNAVAILABLE" else "CONTEXTUAL")
    state = risk_state or default_state
    fresh = state == "CONTEXTUAL"
    return RiskAssessmentSummary(
        assessment_id=aid, position_id=pos, recommendation=rec,
        llm_used=fresh if llm_used is None else llm_used,
        llm_status=(llm_status or ("CALLED" if fresh else {"THROTTLED": "THROTTLED", "FALLBACK": "FAILED"}.get(state))),
        risk_state=state,
        hard_rule_triggered=hard, hard_rule_codes=codes or [], data_issue_codes=data or [],
        reasons=reasons if reasons is not None else ["THESIS_INTACT"], unrealized_pnl_pct=pnl,
        distance_to_invalidation_pct=dist, assessed_at_ms=NOW,
    )


def inp(state="WORKING", assessment=None, position="open", processed=False, exit_req=None, reduce_req=None,
        close=None):
    presets = {"open": BrokerPositionState(position_id=POS, status="OPEN", quantity=2, contract_symbol="C"),
               "closed": BrokerPositionState(position_id=POS, status="CLOSED", quantity=0, contract_symbol="C")}
    pos = presets[position] if isinstance(position, str) else position
    return TradeManagementInput(
        trade_id="TR-1", signal_id=1, proposal_key="signal:1:entry", execution_id="EX-1", position_id=POS,
        current_state=state, position=pos, close_order=close, latest_risk_assessment=assessment,
        assessment_already_processed=processed, active_exit_request=exit_req, active_reduce_request=reduce_req,
        now_ms=NOW, limits=tl.current_limits(),
    )


def active_exit(hard=True):
    return ExitRequestSummary(exit_request_id="XR-1", request_key=f"exit:{POS}:full", requested_action="EXIT",
                              status="ACTIVE", hard_exit=hard)


def active_reduce():
    return ExitRequestSummary(exit_request_id="XR-R", request_key=f"reduce:{POS}:RA-0", requested_action="REDUCE",
                              status="ACTIVE", hard_exit=False)


#
# Pure state machine
#

# 1 / 2
@pytest.mark.parametrize("state", ["OPEN", "WORKING"])
def test_hold_goes_or_stays_working(state):
    d = tl.decide(inp(state, ra("HOLD")))
    assert (d.next_state, d.action) == ("WORKING", "NONE") and d.exit_request is None


def test_hold_close_to_invalidation_is_at_risk_and_recovers():
    d = tl.decide(inp("WORKING", ra("HOLD", dist=0.3)))
    assert d.next_state == "AT_RISK"
    assert tl.decide(inp("AT_RISK", ra("HOLD", aid="RA-2", dist=2.0))).next_state == "WORKING"


# 3
def test_reduce_requested():
    d = tl.decide(inp("WORKING", ra("REDUCE", reasons=["THETA_DECAY", "rationale text"])))
    assert (d.next_state, d.action) == ("REDUCE_REQUESTED", "REQUEST_REDUCE")
    assert d.exit_request.requested_action == "REDUCE"
    assert d.exit_request.request_key == f"reduce:{POS}:RA-1"
    assert d.exit_request.reason_codes == ["THETA_DECAY"]


# 18
def test_reduce_request_has_no_llm_derived_quantity():
    d = tl.decide(inp("WORKING", ra("REDUCE")))
    assert d.exit_request.requested_quantity is None


# 4
def test_exit_requested_with_full_quantity():
    d = tl.decide(inp("WORKING", ra("EXIT", hard=True, codes=["HARD_PREMIUM_STOP"])))
    assert (d.next_state, d.action) == ("EXIT_REQUESTED", "REQUEST_EXIT")
    e = d.exit_request
    assert (e.request_key, e.requested_action, e.requested_quantity, e.hard_exit) == \
        (f"exit:{POS}:full", "EXIT", 2, True)
    assert e.reason_codes == ["HARD_PREMIUM_STOP"]                                  # 22


# 5 / 23
def test_hard_exit_with_bad_quote_still_requests_exit_and_carries_data_issue():
    d = tl.decide(inp("WORKING", ra("EXIT", hard=True, codes=["UNDERLYING_INVALIDATION_HIT"],
                                     data=["QUOTE_UNAVAILABLE"])))
    assert d.next_state == "EXIT_REQUESTED"
    assert d.exit_request.data_issue_codes == ["QUOTE_UNAVAILABLE"]
    assert any("wait for usable data" in r for r in d.reasons)


def test_contextual_exit_carries_agent3_codes():
    d = tl.decide(inp("WORKING", ra("EXIT", reasons=["NEAR_INVALIDATION", "ADVERSE_MOMENTUM", "text"])))
    assert d.exit_request.hard_exit is False
    assert d.exit_request.reason_codes == ["NEAR_INVALIDATION", "ADVERSE_MOMENTUM", "AGENT3_CONTEXTUAL_EXIT"]


# 6
def test_data_unavailable_without_exit_keeps_state_and_no_request():
    d = tl.decide(inp("WORKING", ra("DATA_UNAVAILABLE", data=["QUOTE_NO_BID"])))
    assert (d.next_state, d.action) == ("WORKING", "WAIT_FOR_DATA") and d.exit_request is None


# 7
def test_data_unavailable_cannot_cancel_active_exit():
    d = tl.decide(inp("EXIT_REQUESTED", ra("DATA_UNAVAILABLE"), exit_req=active_exit()))
    assert d.next_state == "EXIT_REQUESTED" and d.action == "WAIT_FOR_DATA"
    assert d.reason_codes == ["EXIT_REQUEST_RETAINED"] and not d.resolve_requests


# 9 / 10
@pytest.mark.parametrize("hard", [True, False])
def test_exit_followed_by_hold_stays_exit_requested(hard):
    d = tl.decide(inp("EXIT_REQUESTED", ra("HOLD", aid="RA-2"), exit_req=active_exit(hard)))
    assert d.next_state == "EXIT_REQUESTED" and d.exit_request is None and not d.resolve_requests


# 27
def test_reduce_then_exit_escalates_and_supersedes_reduce():
    d = tl.decide(inp("REDUCE_REQUESTED", ra("EXIT", aid="RA-2", hard=True, codes=["MAX_HOLD_EXCEEDED"]),
                      reduce_req=active_reduce()))
    assert d.next_state == "EXIT_REQUESTED"
    assert d.resolve_requests == {"XR-R": "SUPERSEDED"}


# 28
def test_exit_cannot_downgrade_to_reduce():
    d = tl.decide(inp("EXIT_REQUESTED", ra("REDUCE", aid="RA-2"), exit_req=active_exit()))
    assert d.next_state == "EXIT_REQUESTED" and d.exit_request is None


def test_repeat_exit_adds_evidence_only():
    d = tl.decide(inp("EXIT_REQUESTED", ra("EXIT", aid="RA-2", hard=True, codes=["HARD_PREMIUM_STOP"]),
                      exit_req=active_exit()))
    assert d.exit_request is None and d.evidence_for_exit_request_id == "XR-1"
    assert d.reason_codes == ["EXIT_ALREADY_REQUESTED"]


# 11 / 12 / 13
def test_closed_position_reconciles_and_completes_request():
    d = tl.decide(inp("EXIT_REQUESTED", ra("EXIT", aid="RA-9", hard=True), position="closed", exit_req=active_exit()))
    assert (d.next_state, d.action) == ("CLOSED", "MARK_CLOSED")
    assert d.exit_request is None
    assert d.resolve_requests == {"XR-1": "COMPLETED"}
    assert d.risk_assessment_id is None                             # reconciliation, not assessment-driven


def test_close_order_pending_and_back():
    pending = BrokerOrderState(order_id="PO-c", side="SELL_TO_CLOSE", status="PENDING", quantity=2)
    assert tl.decide(inp("EXIT_REQUESTED", close=pending, exit_req=active_exit())).next_state == "CLOSE_ORDER_PENDING"
    cancelled = pending.model_copy(update={"status": "CANCELLED"})
    assert tl.decide(inp("CLOSE_ORDER_PENDING", close=cancelled, exit_req=active_exit())).next_state == "EXIT_REQUESTED"


# 14 / 15
@pytest.mark.parametrize("a,b", [
    ("CLOSED", "WORKING"), ("REJECTED", "OPEN"), ("PROPOSED", "CLOSED"), ("OPEN", "POLICY_APPROVED"),
    ("EXIT_REQUESTED", "WORKING"), ("EXIT_REQUESTED", "REDUCE_REQUESTED"), ("FAILED", "OPEN"),
    ("CANCELLED", "PROPOSED"), ("WORKING", "OPEN"), ("OPEN", "BOGUS"),
])
def test_illegal_transitions_rejected(a, b):
    assert not tl.is_transition_allowed(a, b)
    with pytest.raises(tl.IllegalTransition):
        tl.assert_transition(a, b)


@pytest.mark.parametrize("a,b", [
    ("PROPOSED", "POLICY_APPROVED"), ("POLICY_APPROVED", "ORDER_SUBMITTED"), ("ORDER_SUBMITTED", "ORDER_PENDING"),
    ("ORDER_SUBMITTED", "OPEN"), ("ORDER_SUBMITTED", "REJECTED"), ("OPEN", "WORKING"), ("WORKING", "AT_RISK"),
    ("WORKING", "REDUCE_REQUESTED"), ("WORKING", "EXIT_REQUESTED"), ("AT_RISK", "WORKING"),
    ("REDUCE_REQUESTED", "WORKING"), ("REDUCE_REQUESTED", "EXIT_REQUESTED"),
    ("EXIT_REQUESTED", "CLOSE_ORDER_PENDING"), ("CLOSE_ORDER_PENDING", "CLOSED"),
])
def test_legal_transitions(a, b):
    assert tl.is_transition_allowed(a, b)


@pytest.mark.parametrize("state", sorted(tl.TERMINAL_STATES))
def test_terminal_states_cannot_reopen(state):
    assert tl.TRANSITIONS[state] == frozenset()
    d = tl.decide(inp(state, ra("EXIT", hard=True)))
    assert d.next_state == state and d.exit_request is None and d.reason_codes == ["TERMINAL_STATE"]


# 29
def test_missing_broker_position_is_safe():
    d = tl.decide(inp("WORKING", ra("EXIT", hard=True), position=None))
    assert d.next_state == "WORKING" and d.action == "WAIT_FOR_DATA" and d.exit_request is None
    assert d.reason_codes == ["BROKER_POSITION_MISSING"]


# 30
@pytest.mark.parametrize("kwargs,code", [
    ({"state": "BOGUS"}, "INVALID_TRADE_STATE"),
    ({"state": "PROPOSED"}, "STATE_NOT_MANAGED"),
    ({"position": BrokerPositionState(position_id=POS, status="OPEN", quantity=0, contract_symbol="C")}, "POSITION_INCONSISTENT"),
    ({"position": BrokerPositionState(position_id="PP-other", status="OPEN", quantity=2, contract_symbol="C")}, "POSITION_MISMATCH"),
    ({"assessment": ra("EXIT", hard=True, pos="PP-other")}, "ASSESSMENT_POSITION_MISMATCH"),
])
def test_inconsistent_data_fails_closed(kwargs, code):
    args = {"state": "WORKING", "assessment": ra("EXIT", hard=True)}
    args.update(kwargs)
    d = tl.decide(inp(**args))
    assert d.transition_allowed is False
    assert d.reason_codes == [code] and d.exit_request is None and d.next_state == d.previous_state


def test_already_processed_assessment_is_no_op():
    d = tl.decide(inp("WORKING", ra("EXIT", hard=True), processed=True))
    assert d.next_state == "WORKING" and d.exit_request is None


#
# DB-backed TradeManager
#

from test_execution_service import EXEC_KEY, service as exec_service, store_proposal


def filled_trade():
    store_proposal(limit_price=2260.0)
    svc, broker, quotes, clock = exec_service()
    assert svc.execute_entry("signal:1:entry").status == "ORDER_FILLED"
    tm = TradeManager(now_ms=lambda: NOW, log=lambda *a: None)
    trade = tm.materialize_trade(EXEC_KEY)
    return tm, trade, broker, quotes


_n = {"i": 0}


def add_assessment(position_id, rec, hard=False, codes=(), data=(), reasons=("THESIS_INTACT",), pnl=5.0, dist=2.0):
    _n["i"] += 1
    aid = f"RA-{_n['i']:04d}"
    out = {"assessment_id": aid, "position_id": position_id, "recommendation": rec,
           "risk_state": "HARD_EXIT" if hard else ("DATA_UNAVAILABLE" if rec == "DATA_UNAVAILABLE" else "CONTEXTUAL"),
           "hard_rule_triggered": hard, "hard_rule_codes": list(codes), "data_issue_codes": list(data),
           "reasons": list(reasons), "unrealized_pnl_pct": pnl, "distance_to_invalidation_pct": dist}
    conn = sqlite3.connect(config.db_path())
    conn.execute(
        "INSERT INTO risk_assessments (assessment_id, assessment_key, position_id, recommendation, risk_state,"
        " hard_rule_triggered, hard_rule_codes_json, data_issue_codes_json, llm_used, llm_status, reasons_json,"
        " assessed_at, assessed_at_ms, input_json, output_json, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, '[]', '[]', 0, 'x', '[]', 't', ?, '{}', ?, 't')",
        (aid, aid, position_id, rec, out["risk_state"], 1 if hard else 0, NOW + _n["i"], json.dumps(out)))
    conn.commit()
    return aid


def rows(sql, params=()):
    conn = sqlite3.connect(config.db_path())
    conn.row_factory = sqlite3.Row
    return [dict(r) for r in conn.execute(sql, params)]


# 16
def test_trade_creation_from_filled_execution_is_idempotent():
    tm, trade, *_ = filled_trade()
    again = tm.materialize_trade(EXEC_KEY)
    assert trade == again and trade["state"] == "OPEN"
    assert len(rows("SELECT * FROM trades")) == 1
    events = tm.get_events(trade["trade_id"])
    assert [(e["from_state"], e["to_state"], e["action"]) for e in events] == [(None, "OPEN", "MATERIALIZED")]
    details = json.loads(events[0]["details_json"])
    assert details["policy_decision_id"] and details["broker_order_id"]


def test_trade_not_created_for_unfilled_execution():
    store_proposal()                                              # limit below ask -> PENDING
    svc, *_ = exec_service()
    assert svc.execute_entry("signal:1:entry").status == "ORDER_PENDING"
    assert TradeManager(now_ms=lambda: NOW).materialize_trade(EXEC_KEY) is None


# 8 / 26 / 25 / 17 / 24
def test_repeated_exits_one_request_and_full_audit():
    tm, trade, *_ = filled_trade()
    tid, pid = trade["trade_id"], trade["position_id"]

    add_assessment(pid, "HOLD")
    assert tm.process(tid).next_state == "WORKING"
    assert tm.process(tid).reason_codes == ["NO_NEW_ASSESSMENT"]            # same assessment twice: idempotent

    a1 = add_assessment(pid, "EXIT", hard=True, codes=["HARD_PREMIUM_STOP"])
    assert tm.process(tid).next_state == "EXIT_REQUESTED"
    a2 = add_assessment(pid, "EXIT", hard=True, codes=["UNDERLYING_INVALIDATION_HIT"])
    tm.process(tid)
    tm.process(tid)

    reqs = tm.get_exit_requests(tid)
    assert len(reqs) == 1 and reqs[0].status == "ACTIVE" and reqs[0].risk_assessment_id == a1
    assert reqs[0].requested_quantity == 2

    events = tm.get_events(tid)
    assert [(e["from_state"], e["to_state"]) for e in events] == [
        (None, "OPEN"), ("OPEN", "WORKING"), ("WORKING", "EXIT_REQUESTED"), ("EXIT_REQUESTED", "EXIT_REQUESTED")]
    assert events[-1]["risk_assessment_id"] == a2 and events[-1]["exit_request_id"] == reqs[0].exit_request_id
    assert tm.get_trade(tid)["state"] == "EXIT_REQUESTED"


def test_db_enforces_one_active_full_exit_and_one_event_per_assessment():
    tm, trade, *_ = filled_trade()
    tid, pid = trade["trade_id"], trade["position_id"]
    add_assessment(pid, "EXIT", hard=True, codes=["HARD_PREMIUM_STOP"])
    tm.process(tid)

    conn = sqlite3.connect(config.db_path())
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO exit_requests (exit_request_id, request_key, trade_id, position_id, requested_action,"
                     " requested_quantity, hard_exit, reason_codes_json, data_issue_codes_json, status, created_at,"
                     " updated_at) VALUES ('XR-dup', 'other-key', ?, ?, 'EXIT', 2, 1, '[]', '[]', 'ACTIVE', 't', 't')",
                     (tid, pid))
    aid = rows("SELECT risk_assessment_id FROM trade_lifecycle_events WHERE risk_assessment_id IS NOT NULL")[0]["risk_assessment_id"]
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO trade_lifecycle_events (trade_id, from_state, to_state, action, reason_codes_json,"
                     " reasons_json, risk_assessment_id, created_at) VALUES (?, 'WORKING', 'WORKING', 'NONE', '[]',"
                     " '[]', ?, 't')", (tid, aid))


def test_hold_after_hard_exit_persisted_as_retained():
    tm, trade, *_ = filled_trade()
    tid, pid = trade["trade_id"], trade["position_id"]
    add_assessment(pid, "EXIT", hard=True, codes=["HARD_PREMIUM_STOP"], data=["QUOTE_STALE"])
    tm.process(tid)
    add_assessment(pid, "HOLD")
    d = tm.process(tid)
    assert d.next_state == "EXIT_REQUESTED" and d.reason_codes == ["EXIT_REQUEST_RETAINED"]
    [req] = tm.get_exit_requests(tid)
    assert req.status == "ACTIVE" and req.data_issue_codes == ["QUOTE_STALE"]


# 11-13 (DB)
def test_closed_position_reconciles_trade_and_completes_request():
    tm, trade, broker, quotes = filled_trade()
    tid, pid = trade["trade_id"], trade["position_id"]
    add_assessment(pid, "EXIT", hard=True, codes=["UNDERLYING_TARGET_HIT"])
    tm.process(tid)

    broker.close_position(pid, 2200.0, "test-close")             # simulated external close (test setup only)
    d = tm.process(tid)

    assert d.next_state == "CLOSED" and d.action == "MARK_CLOSED"
    t = tm.get_trade(tid)
    assert t["state"] == "CLOSED" and t["closed_at"]
    [req] = tm.get_exit_requests(tid)
    assert req.status == "COMPLETED" and req.completed_at
    assert tm.get_events(tid)[-1]["exit_request_id"] == req.exit_request_id   # CLOSED event links the request
    assert len(rows("SELECT * FROM exit_requests")) == 1          # no new request

    add_assessment(pid, "EXIT", hard=True, codes=["HARD_PREMIUM_STOP"])
    assert tm.process(tid).reason_codes == ["TERMINAL_STATE"]    # terminal: never reopens
    assert tm.get_trade(tid)["state"] == "CLOSED"


def test_reduce_advisory_persisted_without_quantity_then_escalated():
    tm, trade, *_ = filled_trade()
    tid, pid = trade["trade_id"], trade["position_id"]
    add_assessment(pid, "REDUCE", reasons=["THETA_DECAY"])
    assert tm.process(tid).next_state == "REDUCE_REQUESTED"
    [r] = tm.get_exit_requests(tid)
    assert r.requested_action == "REDUCE" and r.requested_quantity is None

    add_assessment(pid, "EXIT", codes=[], reasons=["NEAR_INVALIDATION"])
    assert tm.process(tid).next_state == "EXIT_REQUESTED"
    statuses = {x.requested_action: x.status for x in tm.get_exit_requests(tid)}
    assert statuses == {"REDUCE": "SUPERSEDED", "EXIT": "ACTIVE"}


def test_concurrent_state_change_is_not_overwritten():
    tm, trade, *_ = filled_trade()
    tid, pid = trade["trade_id"], trade["position_id"]
    add_assessment(pid, "EXIT", hard=True, codes=["HARD_PREMIUM_STOP"])
    stale_input = tm.build_input(tid)
    conn = sqlite3.connect(config.db_path())
    conn.execute("UPDATE trades SET state = 'WORKING' WHERE trade_id = ?", (tid,))  # moved by someone else
    conn.commit()
    tm._persist(tl.decide(stale_input), stale_input)                # decision computed for OPEN
    assert tm.get_trade(tid)["state"] == "WORKING"
    assert tm.get_exit_requests(tid) == []


# 19-21
def _code(rel):
    text = (config.PROJECT_DIR / rel).read_text()
    return "\n".join(l for l in text.splitlines() if not l.strip().startswith("#"))


def test_agent4_source_has_no_broker_execution_or_llm_references():
    for rel in ("agents/trade_manager.py", "core/trade_lifecycle.py"):
        code = _code(rel)
        imports = [l for l in code.splitlines() if l.startswith(("import ", "from "))]
        for banned in ("broker", "execution_service", "llm", "openai", "risk_agent", "signal_agent", "options_agent"):
            assert not [l for l in imports if banned in l], (rel, banned, imports)
        for call in (".submit_order(", ".close_position(", ".cancel_order(", ".structured("):
            assert call not in code, (rel, call)


def test_agent4_modules_load_no_broker_execution_or_llm():
    probe = (
        "import sys; import agents.trade_manager, core.trade_lifecycle; "
        "bad = sorted(m for m in sys.modules if m.startswith('broker') or m == 'core.execution_service' "
        "or m in ('openai', 'llm_client', 'llm_parser') or m.startswith('openai.') "
        "or m in ('agents.risk_agent', 'agents.signal_agent', 'agents.options_agent')); print(','.join(bad))"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=config.PROJECT_DIR, env=dict(os.environ),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_exit_persisted_before_a_later_hold_is_never_skipped():
    """Agent 3 wrote EXIT then HOLD before Agent 4 ran: both are applied in order; EXIT wins."""
    tm, trade, *_ = filled_trade()
    tid, pid = trade["trade_id"], trade["position_id"]
    add_assessment(pid, "EXIT", hard=True, codes=["HARD_PREMIUM_STOP"])
    add_assessment(pid, "HOLD")

    decisions = [d for d in tm.process_all() if d.risk_assessment_id]

    assert [(d.next_state, d.reason_codes) for d in decisions] == [
        ("EXIT_REQUESTED", ["HARD_PREMIUM_STOP"]), ("EXIT_REQUESTED", ["EXIT_REQUEST_RETAINED"])]
    assert tm.get_trade(tid)["state"] == "EXIT_REQUESTED"
    assert len(tm.get_exit_requests(tid)) == 1
    assert tm.process_all() == [] or all(d.risk_assessment_id is None for d in tm.process_all())


#
# REDUCE flapping fix: only a FRESH contextual LLM HOLD clears REDUCE_REQUESTED
#

def test_fresh_llm_reduce_enters_reduce_requested():
    d = tl.decide(inp("WORKING", ra("REDUCE", reasons=["THETA_DECAY"])))
    assert d.next_state == "REDUCE_REQUESTED" and d.exit_request.requested_action == "REDUCE"


@pytest.mark.parametrize("kind,kwargs", [
    ("throttled", {"risk_state": "THROTTLED", "llm_used": False, "llm_status": "THROTTLED"}),
    ("fallback-failed", {"risk_state": "FALLBACK", "llm_used": False, "llm_status": "FAILED"}),
    ("fallback-no-llm", {"risk_state": "FALLBACK", "llm_used": False, "llm_status": "NOT_CONFIGURED"}),
    ("deterministic", {"risk_state": "CONTEXTUAL", "llm_used": False, "llm_status": "CALLED"}),
])
def test_non_fresh_hold_keeps_reduce_requested(kind, kwargs):
    d = tl.decide(inp("REDUCE_REQUESTED", ra("HOLD", aid="RA-2", **kwargs), reduce_req=active_reduce()))
    assert d.next_state == "REDUCE_REQUESTED", kind
    assert d.exit_request is None and not d.resolve_requests
    assert d.reason_codes == ["REDUCE_RETAINED_HOLD_NOT_FRESH"]


def test_fresh_llm_hold_clears_reduce_and_supersedes_request():
    d = tl.decide(inp("REDUCE_REQUESTED", ra("HOLD", aid="RA-2"), reduce_req=active_reduce()))
    assert d.next_state == "WORKING" and d.resolve_requests == {"XR-R": "SUPERSEDED"}


def test_exit_escalates_from_reduce_requested_even_when_not_fresh():
    d = tl.decide(inp("REDUCE_REQUESTED", ra("EXIT", aid="RA-3", hard=True, codes=["HARD_PREMIUM_STOP"]),
                      reduce_req=active_reduce()))
    assert d.next_state == "EXIT_REQUESTED" and d.resolve_requests == {"XR-R": "SUPERSEDED"}


def _add_throttled_hold(pid):
    aid = add_assessment(pid, "HOLD")
    conn = sqlite3.connect(config.db_path())
    out = json.loads(conn.execute("SELECT output_json FROM risk_assessments WHERE assessment_id = ?", (aid,)).fetchone()[0])
    out.update(risk_state="THROTTLED", llm_used=False, llm_status="THROTTLED")
    conn.execute("UPDATE risk_assessments SET output_json = ? WHERE assessment_id = ?", (json.dumps(out), aid))
    conn.commit()
    return aid


def _add_fresh(pid, rec, reasons):
    aid = add_assessment(pid, rec, reasons=reasons)
    conn = sqlite3.connect(config.db_path())
    out = json.loads(conn.execute("SELECT output_json FROM risk_assessments WHERE assessment_id = ?", (aid,)).fetchone()[0])
    out.update(llm_used=True, llm_status="CALLED")
    conn.execute("UPDATE risk_assessments SET output_json = ? WHERE assessment_id = ?", (json.dumps(out), aid))
    conn.commit()
    return aid


def test_db_reduce_sequence_no_flapping_no_duplicate_events():
    tm, trade, *_ = filled_trade()
    tid, pid = trade["trade_id"], trade["position_id"]

    _add_fresh(pid, "REDUCE", ["THETA_DECAY"])
    tm.process_all()
    assert tm.get_trade(tid)["state"] == "REDUCE_REQUESTED"
    events_before = len(tm.get_events(tid))

    for _ in range(5):                                          # repeated throttled cycles
        _add_throttled_hold(pid)
        tm.process_all()
    assert tm.get_trade(tid)["state"] == "REDUCE_REQUESTED"
    assert len(tm.get_events(tid)) == events_before             # no new lifecycle events
    assert len(tm.get_exit_requests(tid)) == 1                  # no new REDUCE request
    assert len(rows("SELECT * FROM trade_assessment_applications WHERE trade_id = ?", (tid,))) == 6  # each applied once

    _add_fresh(pid, "HOLD", ["THESIS_INTACT"])                  # fresh real LLM HOLD clears it
    tm.process_all()
    assert tm.get_trade(tid)["state"] == "WORKING"
    assert [r.status for r in tm.get_exit_requests(tid)] == ["SUPERSEDED"]

    _add_fresh(pid, "REDUCE", ["RISK_REWARD_DETERIORATED"])     # a new fresh REDUCE is allowed
    tm.process_all()
    assert tm.get_trade(tid)["state"] == "REDUCE_REQUESTED"
    assert sorted(r.status for r in tm.get_exit_requests(tid)) == ["ACTIVE", "SUPERSEDED"]

    add_assessment(pid, "EXIT", hard=True, codes=["MAX_HOLD_EXCEEDED"])
    tm.process_all()
    assert tm.get_trade(tid)["state"] == "EXIT_REQUESTED"
