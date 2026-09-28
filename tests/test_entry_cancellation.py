"""Resting paper ENTRY orders: deterministic expiry / invalidity cancellation (offline)."""
import json
import sqlite3

import pytest

import config
from core import policy_engine as pe
from schemas import KillSwitchState

from test_integration import Harness, bull_to_working, flags, n, q  # noqa: F401  (flags is a fixture)
from test_policy_engine import NOW, proposal, signal


# -------- pure validity check -------------------------------------------------

def reasons(price=84200.0, status="WORKING", agent1="DEVELOPING", age_min=5, kill=False, **kw):
    return pe.resting_entry_cancel_reasons(
        proposal(), signal(**kw), status, agent1, price, NOW - int(age_min * 60_000), NOW,
        pe.current_risk_limits(), KillSwitchState(env_enabled=kill, file_enabled=False, file_path="x"), 15)


def test_valid_resting_entry_has_no_reasons():
    assert reasons() == []


def test_temporary_pullback_below_entry_is_not_a_cancel():
    # Confirmed at 84,000; price slips to 83,950 and the evaluator may say DEVELOPING.
    assert reasons(price=83950.0, status="DEVELOPING") == []


@pytest.mark.parametrize("kw,code", [
    ({"age_min": 15.01}, "ENTRY_ORDER_EXPIRED"),
    ({"status": "INVALIDATED"}, "SIGNAL_INVALIDATED"),
    ({"price": 82500.0}, "SIGNAL_INVALIDATED"),
    ({"status": "TARGET_HIT"}, "TARGET_ALREADY_REACHED"),
    ({"price": 86100.0}, "TARGET_ALREADY_REACHED"),
    ({"status": "EXPIRED"}, "SIGNAL_EXPIRED"),
    ({"agent1": "STALE"}, "THESIS_STALE"),
    ({"agent1": "NOT_TRADEABLE"}, "THESIS_NOT_TRADEABLE"),
    ({"price": 84600.0}, "ENTRY_STALE"),
    ({"kill": True}, "KILL_SWITCH_ENABLED"),
])
def test_each_cancel_reason(kw, code):
    assert code in reasons(**kw)


def test_age_exactly_at_limit_stays():
    assert reasons(age_min=15) == []


def test_target_progress_reason(monkeypatch):
    monkeypatch.setenv("AGENT2_MAX_UNDERLYING_MOVE_PCT", "5")
    assert reasons(price=85100.0) == ["TARGET_PROGRESS_EXCEEDED"]


# -------- integration ------------------------------------------------------------

def pending_entry(h):
    bull_to_working(h)                                       # entry submitted at t+11.5 and resting
    [ex] = q("SELECT * FROM execution_attempts WHERE purpose = 'ENTRY'")
    assert ex["status"] == "ORDER_PENDING"
    return ex


def cancelled(h):
    return [m for m in h.trade_messages if m.startswith("🚫 PAPER ENTRY CANCELLED")]


def entry_status():
    return q("SELECT status, error_code, error_message FROM execution_attempts WHERE purpose = 'ENTRY'")[0]


# 5
def test_pending_entry_below_max_age_remains_pending(flags):
    flags()
    h = Harness()
    pending_entry(h)
    h.at(26.0)                                               # 14.5 min old
    h.cycle()
    assert entry_status()["status"] == "ORDER_PENDING" and cancelled(h) == []


# 6 / 15 / 16 / 18 / 19 / 20
def test_age_limit_cancels_once_with_full_audit(flags):
    flags()
    h = Harness()
    ex = pending_entry(h)
    h.at(27.0)                                               # 15.5 min old
    h.cycle()

    st = entry_status()
    assert st["status"] == "ORDER_CANCELLED" and st["error_code"] == "ENTRY_ORDER_EXPIRED"
    assert q("SELECT status FROM paper_orders")[0]["status"] == "CANCELLED"
    assert n("trades") == 0                                  # 15
    [msg] = cancelled(h)                                     # 18
    assert "Contract: BTC-" in msg and "Limit: " in msg and "Order age: 16 min" in msg
    assert "Reason: ENTRY_ORDER_EXPIRED" in msg

    events_before = n("execution_events")
    for m in (28, 29, 40):                                   # 16: idempotent
        h.feed.set("BTCUSDT", m, 84000.0)                    # even if the market later reaches the old limit
        h.at(m + 0.5)
        h.cycle()
    assert n("paper_orders") == 1 and n("execution_attempts") == 1 and n("trades") == 0
    assert n("execution_events") == events_before and len(cancelled(h)) == 1
    assert n("paper_orders", "side = 'SELL_TO_CLOSE'") == 0  # 19

    history = [e["to_status"] for e in q("SELECT to_status FROM execution_events WHERE execution_id = ? ORDER BY id",
                                         (ex["execution_id"],))]
    assert history == ["CLAIMED", "SUBMITTING", "ORDER_PENDING", "ORDER_CANCELLED"]   # 20
    details = json.loads(q("SELECT details FROM execution_events WHERE to_status = 'ORDER_CANCELLED'")[0]["details"])
    assert details["error_code"] == "ENTRY_ORDER_EXPIRED"
    result = json.loads(q("SELECT result_json FROM execution_attempts")[0]["result_json"])
    assert result["cancel_reasons"] == ["ENTRY_ORDER_EXPIRED"]


# 7
def test_invalidation_cancels_before_the_cheap_fill(flags):
    flags()
    h = Harness()
    pending_entry(h)
    h.feed.set("BTCUSDT", 13, 82500.0)                       # through invalidation: the call ask collapses
    h.at(13.5)
    h.cycle()
    st = entry_status()
    assert st["status"] == "ORDER_CANCELLED" and "SIGNAL_INVALIDATED" in st["error_message"]
    assert n("trades") == 0 and q("SELECT status FROM paper_orders")[0]["status"] == "CANCELLED"


# 8
def test_signal_expired_cancels(flags, monkeypatch):
    flags()
    monkeypatch.setenv("SIGNAL_EXPIRY_HOURS_INTRADAY", "0.2")   # 12 minutes
    h = Harness()
    pending_entry(h)
    h.at(13.5)
    h.cycle()
    st = entry_status()
    assert st["status"] == "ORDER_CANCELLED" and "SIGNAL_EXPIRED" in st["error_message"]


# 9
def test_target_reached_cancels(flags):
    flags()
    h = Harness()
    pending_entry(h)
    h.feed.set("BTCUSDT", 13, 86100.0)
    h.at(13.5)
    h.cycle()
    assert "TARGET_ALREADY_REACHED" in entry_status()["error_message"]


# 10
def test_stale_entry_cancels(flags):
    flags()
    h = Harness()
    pending_entry(h)
    h.feed.set("BTCUSDT", 13, 84600.0)                       # +0.71% past entry
    h.at(13.5)
    h.cycle()
    st = entry_status()
    assert st["status"] == "ORDER_CANCELLED" and st["error_message"] == "ENTRY_STALE"


# 11
def test_target_progress_cancels(flags, monkeypatch):
    flags()
    h = Harness()
    pending_entry(h)
    monkeypatch.setenv("AGENT2_MAX_UNDERLYING_MOVE_PCT", "5")
    h.feed.set("BTCUSDT", 13, 85100.0)                       # 55% of entry -> target
    h.at(13.5)
    h.cycle()
    assert entry_status()["error_message"] == "TARGET_PROGRESS_EXCEEDED"


# 12
def test_kill_switch_cancels_pending_entry(flags, monkeypatch):
    flags()
    h = Harness()
    pending_entry(h)
    monkeypatch.setenv("TRADING_KILL_SWITCH", "1")
    h.at(13.5)
    h.cycle()
    st = entry_status()
    assert st["status"] == "ORDER_CANCELLED" and st["error_code"] == "KILL_SWITCH_ENABLED"
    assert "Reason: KILL_SWITCH_ENABLED" in cancelled(h)[0]


# 13
def test_kill_switch_never_touches_pending_exit(flags, monkeypatch):
    flags()
    h = Harness()
    bull_to_working(h)
    h.feed.set("BTCUSDT", 12, 84100.0)
    h.at(12.5)
    h.cycle()                                                # entry filled, trade open
    real = h.broker.close_position
    h.broker.close_position = lambda pid, limit_price, client_order_id, quantity=None: real(
        pid, limit_price + 500, client_order_id, quantity)    # force a resting close (test setup)
    h.feed.set("BTCUSDT", 20, 82500.0)
    h.at(20.5)
    h.cycle()
    assert q("SELECT status FROM execution_attempts WHERE purpose = 'EXIT'")[0]["status"] == "ORDER_PENDING"

    monkeypatch.setenv("TRADING_KILL_SWITCH", "1")
    for m in (21.5, 22.5):
        h.at(m)
        h.cycle()
    assert q("SELECT status FROM paper_orders WHERE side = 'SELL_TO_CLOSE'")[0]["status"] == "PENDING"
    assert q("SELECT status FROM execution_attempts WHERE purpose = 'EXIT'")[0]["status"] == "ORDER_PENDING"
    assert cancelled(h) == []


# 14
def test_fill_before_cancellation_check_filled_wins(flags, monkeypatch):
    flags()
    h = Harness()
    pending_entry(h)
    h.feed.set("BTCUSDT", 12, 84100.0)
    h.at(12.5)
    h.broker.process_pending_orders()                        # broker fills first (execution row still PENDING)
    assert q("SELECT status FROM paper_orders")[0]["status"] == "FILLED"

    monkeypatch.setenv("TRADING_KILL_SWITCH", "1")           # now a cancel condition appears
    h.cycle()
    assert entry_status()["status"] == "ORDER_FILLED"
    assert q("SELECT status FROM paper_orders")[0]["status"] == "FILLED"
    assert n("trades") == 1 and cancelled(h) == []


def test_cancel_race_lost_records_fill(flags):
    flags()
    h = Harness()
    pending_entry(h)
    h.feed.set("BTCUSDT", 12, 84100.0)
    h.at(12.5)
    real_cancel = h.broker.cancel_order

    def fill_then_cancel(order_id):
        h.broker.process_pending_orders()                    # fills in between the check and the cancel
        return real_cancel(order_id)

    h.broker.cancel_order = fill_then_cancel
    r = h.pipeline.execution.cancel_entry("entry:signal:1:entry", ["ENTRY_ORDER_EXPIRED"])
    assert r.status == "ORDER_FILLED"
    assert q("SELECT status FROM paper_orders")[0]["status"] == "FILLED"


# 17
def test_restart_preserves_cancelled_state_and_notification(flags):
    flags()
    h = Harness()
    pending_entry(h)
    h.at(27.0)
    h.cycle()
    assert entry_status()["status"] == "ORDER_CANCELLED"
    sent = list(h.trade_messages)

    h2 = Harness(clock=h.clock, feed=h.feed, send=h.trade_messages.append)
    h2.parsed = h.parsed
    h2.pipeline.startup_reconcile()
    h2.cycle()
    assert entry_status()["status"] == "ORDER_CANCELLED"
    assert n("execution_attempts") == 1 and n("paper_orders") == 1 and n("trades") == 0
    assert h.trade_messages == sent                            # nothing resent


def test_cancellation_runs_even_when_new_entries_are_disabled(flags, monkeypatch):
    flags()
    h = Harness()
    pending_entry(h)
    monkeypatch.setenv("PAPER_EXECUTION_ENABLED", "false")
    h.at(27.0)
    h.cycle()
    assert entry_status()["status"] == "ORDER_CANCELLED"
