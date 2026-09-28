"""Deterministic Policy Engine (offline). No LLM, no broker submission."""
import json
import os
import sqlite3
import subprocess
import sys
from datetime import date

import pytest

import config
import database
from core import policy_engine as pe
from schemas import (
    AccountSnapshot,
    KillSwitchState,
    OpenPositionSummary,
    OptionProposal,
    PolicyEvaluationInput,
    SignalAnalysisInput,
    ThesisContext,
)

from helpers import MINUTE_MS, T0
from options_helpers import contract


NOW = T0 + 12 * MINUTE_MS
SYMBOL = "BTC-9OCT26-85000-C"


#
# Clean snapshot: bullish BTC, entry 84,000 -> target 86,000, invalid 82,600,
# confirmed, price 84,200 (+0.238% past entry, 10% of the way to target).
#

def signal(**kw):
    data = dict(
        signal_id=1, discord_message_id="42", raw_message="raw", symbol="BTCUSDT",
        bias="conditional_bullish", timeframe="1h", support_levels=[83000.0],
        resistance_levels=[84000.0], targets=[86000.0], confirmation_above=84000.0,
        invalidation_below=82600.0, signal_time_ms=T0, price_at_signal=83400.0,
    )
    data.update(kw)
    return SignalAnalysisInput(**data)


def proposal(**kw):
    data = dict(
        proposal_key="signal:1:entry", signal_id=1, analysis_id=1, trigger="FIRST_WORKING",
        status="PROPOSED", strategy="LONG_CALL", side="BUY_TO_OPEN", contract_symbol=SYMBOL,
        underlying="BTCUSDT", option_type="call", expiry=date(2026, 10, 9), dte=14.0,
        strike=85000.0, delta=0.45, iv=0.5, bid=2200.0, ask=2260.0, mid=2230.0,
        limit_price=2230.0, quantity=2, contract_multiplier=0.01, estimated_max_loss=44.6,
        breakeven=87230.0, thesis_target=86000.0, thesis_invalidation=82600.0,
        selected_by="llm", llm_used=True, reason="test",
    )
    data.update(kw)
    return OptionProposal(**data)


def account(**kw):
    data = dict(open_position_count=0, pending_entry_count=0, open_premium_at_risk=0.0,
                realized_pnl_today=0.0, unrealized_pnl=0.0)
    data.update(kw)
    return AccountSnapshot(**data)


def make(proposal_kw=None, quote_kw=None, account_kw=None, limits_kw=None, thesis_kw=None,
         signal_kw=None, **top):
    quote = top.pop("quote", "default")
    data = dict(
        purpose="ENTRY",
        proposal=proposal(**(proposal_kw or {})),
        thesis=ThesisContext(**{"signal": signal(**(signal_kw or {})), "signal_status": "WORKING",
                                "signal_active": True, "agent1_state": "DEVELOPING", **(thesis_kw or {})}),
        underlying_price=84200.0,
        quote=contract(**{"as_of_ms": NOW, **(quote_kw or {})}) if quote == "default" else quote,
        account=account(**(account_kw or {})),
        open_positions=[],
        pending_entries=[],
        filled_entry_proposal_keys=[],
        now_ms=NOW,
        limits=pe.current_risk_limits().model_copy(update=limits_kw or {}),
        trading_mode="paper",
        kill_switch=KillSwitchState(env_enabled=False, file_enabled=False, file_path="/nonexistent/KILL_SWITCH"),
    )
    data.update(top)
    return PolicyEvaluationInput(**data)


def reasons(inp):
    return pe.evaluate(inp).reasons


# 1
def test_clean_proposal_approved():
    d = pe.evaluate(make())
    assert d.approved and d.decision == "APPROVED"
    assert d.reasons == []
    assert d.calculated_max_loss == pytest.approx(2230 * 0.01 * 2)      # the order's own limit
    assert d.calculated_total_open_risk == pytest.approx(44.6)
    assert d.quote_age_seconds == 0
    assert d.proposal_key == "signal:1:entry" and d.signal_id == 1


# 2
@pytest.mark.parametrize("mode", ["live", "LIVE", "testnet", ""])
def test_non_paper_mode_rejected(mode):
    assert "trading_mode_not_paper" in reasons(make(trading_mode=mode))


# 3
def test_env_kill_switch_rejects_entry(monkeypatch):
    monkeypatch.setenv("TRADING_KILL_SWITCH", "1")
    monkeypatch.setenv("KILL_SWITCH_FILE", "/nonexistent/KILL_SWITCH")
    ks = pe.read_kill_switch()
    assert ks.env_enabled and not ks.file_enabled
    assert reasons(make(kill_switch=ks)) == ["kill_switch_enabled"]


# 4
def test_kill_switch_file_rejects_entry(monkeypatch, tmp_path):
    flag = tmp_path / "KILL_SWITCH"
    monkeypatch.setenv("KILL_SWITCH_FILE", str(flag))
    assert pe.read_kill_switch().active is False
    flag.touch()
    ks = pe.read_kill_switch()
    assert ks.file_enabled and not ks.env_enabled
    assert reasons(make(kill_switch=ks)) == ["kill_switch_enabled"]


def test_kill_switch_default_file_is_project_level(monkeypatch):
    monkeypatch.delenv("KILL_SWITCH_FILE", raising=False)
    assert config.kill_switch_file() == config.PROJECT_DIR / "KILL_SWITCH"


def test_kill_switch_does_not_block_exits_by_design():
    assert pe.KILL_SWITCH_BLOCKS == {"ENTRY": True, "EXIT": False}
    with pytest.raises(NotImplementedError):
        pe.evaluate(make(purpose="EXIT"))


# 5
def test_quantity_over_limit():
    assert "max_contracts_exceeded" in reasons(make(limits_kw={"max_contracts": 1}))


# 6
def test_max_premium_exceeded():
    assert reasons(make(limits_kw={"max_premium_per_trade": 40.0})) == ["max_loss_per_trade_exceeded"]


def test_fees_count_toward_max_loss():
    d = pe.evaluate(make(limits_kw={"fee_per_contract": 2.5, "max_premium_per_trade": 49.0}))
    assert d.calculated_max_loss == pytest.approx(44.6 + 5.0)
    assert d.reasons == ["max_loss_per_trade_exceeded"]


# 7
def test_max_daily_loss_exceeded():
    assert reasons(make(account_kw={"realized_pnl_today": -1200.0})) == ["max_daily_loss_reached"]


# 8
def test_max_open_positions_reached():
    assert reasons(make(account_kw={"open_position_count": 3})) == ["max_open_positions_reached"]
    assert reasons(make(account_kw={"open_position_count": 2, "pending_entry_count": 1})) == ["max_open_positions_reached"]


# 9
def test_duplicate_signal_position_rejected():
    existing = OpenPositionSummary(position_id="PP-1", signal_id=1, proposal_key="signal:1:entry",
                                   contract_symbol=SYMBOL, quantity=2, entry_price=2230.0,
                                   contract_multiplier=0.01, premium_at_risk=44.6)
    assert "duplicate_signal_position" in reasons(make(open_positions=[existing]))


# 10
def test_duplicate_proposal_already_filled_rejected():
    assert reasons(make(filled_entry_proposal_keys=["signal:1:entry"])) == ["duplicate_proposal_entry"]


# 11 / 12
def test_dte_too_low():
    assert reasons(make(quote_kw={"dte": 6.9})) == ["dte_below_min"]


def test_dte_too_high():
    assert reasons(make(quote_kw={"dte": 45.1})) == ["dte_above_max"]


# 13
@pytest.mark.parametrize("delta,code", [(0.30, "delta_below_min"), (0.70, "delta_above_max"),
                                        (None, "delta_unavailable")])
def test_delta_out_of_range(delta, code):
    assert reasons(make(quote_kw={"delta": delta})) == [code]


# 14
def test_spread_too_wide():
    assert reasons(make(quote_kw={"bid": 1800.0, "ask": 2260.0})) == ["spread_too_wide"]


def test_no_valid_quote():
    assert "no_valid_quote" in reasons(make(quote_kw={"bid": 0.0}))


# 15 / 16
def test_insufficient_open_interest():
    assert reasons(make(quote_kw={"open_interest": 99})) == ["open_interest_below_min"]


def test_insufficient_volume():
    assert reasons(make(quote_kw={"volume_24h": 9})) == ["volume_below_min"]


# 17
def test_stale_quote():
    d = pe.evaluate(make(quote_kw={"as_of_ms": NOW - 61_000}))
    assert d.reasons == ["quote_stale"]
    assert d.quote_age_seconds == 61


def test_quote_unavailable():
    assert reasons(make(quote=None)) == ["quote_unavailable"]


# 18
def test_wrong_contract_returned_by_provider():
    assert reasons(make(quote_kw={"contract_symbol": "BTC-9OCT26-84000-C", "strike": 84000.0})) == ["contract_mismatch"]


def test_contract_multiplier_changed():
    assert reasons(make(quote_kw={"contract_multiplier": 0.1})) == ["contract_terms_changed"]


# 19
def test_bearish_strategy_on_bullish_thesis_rejected():
    r = reasons(make(
        proposal_kw={"strategy": "LONG_PUT", "option_type": "put", "contract_symbol": "BTC-9OCT26-85000-P", "delta": -0.45},
        quote_kw={"contract_symbol": "BTC-9OCT26-85000-P", "option_type": "put", "delta": -0.45},
    ))
    assert "direction_strategy_mismatch" in r


def test_quote_option_type_mismatch_rejected():
    r = reasons(make(quote_kw={"option_type": "put", "delta": -0.45}))
    assert "option_type_strategy_mismatch" in r and "contract_mismatch" in r


# 20
def test_target_already_hit_by_status():
    assert "signal_target_hit" in reasons(make(thesis_kw={"signal_status": "TARGET_HIT", "signal_active": False}))


def test_target_already_reached_at_current_price():
    assert "target_reached" in reasons(make(underlying_price=86100.0))


# 21
def test_invalidation_already_hit():
    r = reasons(make(underlying_price=82500.0))
    assert "invalidation_hit" in r


def test_invalidated_status():
    assert "signal_invalidated" in reasons(make(thesis_kw={"signal_status": "INVALIDATED", "signal_active": False}))


# 22
def test_signal_expired_by_age():
    later = T0 + 25 * 3_600_000
    r = reasons(make(now_ms=later, quote_kw={"as_of_ms": later}))
    assert r == ["signal_expired"]


def test_signal_expired_status():
    assert "signal_expired" in reasons(make(thesis_kw={"signal_status": "EXPIRED", "signal_active": False}))


@pytest.mark.parametrize("state,code", [("STALE", "agent1_stale"), ("NOT_TRADEABLE", "agent1_not_tradeable")])
def test_agent1_state_rejected(state, code):
    assert reasons(make(thesis_kw={"agent1_state": state})) == [code]


# 23
def test_stale_entry_extension_exceeded():
    assert reasons(make(underlying_price=84600.0)) == ["stale_entry_move_extended"]   # +0.714%


# 24
def test_target_progress_threshold_exceeded():
    r = reasons(make(underlying_price=85100.0, limits_kw={"max_underlying_move_pct": 5.0}))
    assert r == ["target_progress_exceeded"]                                         # 55% > 50%


# 25 (corrected): max loss comes from the ORDER LIMIT, never from the fresh ask.

def _cheap(limit=100.0, fresh_bid=115.0, fresh_ask=120.0, **quote_kw):
    """Proposal: limit 100 x 0.01 x 2. Fresh quote as given."""
    return make(
        proposal_kw={"bid": 98.0, "ask": 100.0, "mid": 99.0, "limit_price": limit, "estimated_max_loss": limit * 0.02},
        quote_kw={"bid": fresh_bid, "ask": fresh_ask, "mark": (fresh_bid + fresh_ask) / 2, **quote_kw},
    )


def test_fresh_ask_above_limit_does_not_raise_max_loss_and_can_be_approved():
    d = pe.evaluate(_cheap(fresh_bid=115.0, fresh_ask=120.0))
    assert d.calculated_max_loss == pytest.approx(100 * 0.01 * 2)       # 2.00, not 120-based 2.40
    assert d.approved, d.reasons


def test_approved_resting_limit_would_simply_stay_pending_at_broker():
    from broker.paper_broker import PaperBroker
    from schemas import OrderRequest

    class Quote:
        def get_option_contract(self, symbol):
            return contract(contract_symbol=symbol, bid=115.0, ask=120.0, mark=117.5, as_of_ms=NOW)

    paper = PaperBroker(Quote(), now_ms=lambda: NOW, log=lambda *a: None)
    r = paper.submit_order(OrderRequest(client_order_id="x", signal_id=1, proposal_key="signal:1:entry",
                                        contract_symbol=SYMBOL, side="BUY_TO_OPEN", quantity=2, limit_price=100.0))
    assert r.order.status == "PENDING" and r.fill is None


def test_fresh_ask_below_limit_still_uses_order_limit():
    d = pe.evaluate(_cheap(fresh_bid=92.0, fresh_ask=95.0))
    assert d.calculated_max_loss == pytest.approx(100 * 0.01 * 2)       # 2.00, not 95-based 1.90
    assert d.approved, d.reasons


def test_fresh_spread_too_wide_still_rejected_independently():
    d = pe.evaluate(_cheap(fresh_bid=80.0, fresh_ask=120.0))
    assert d.reasons == ["spread_too_wide"]
    assert d.calculated_max_loss == pytest.approx(2.0)


def test_changed_multiplier_rejected_as_contract_terms_changed():
    d = pe.evaluate(_cheap(contract_multiplier=0.1))
    assert d.reasons == ["contract_terms_changed"]


def test_quantity_is_never_resized():
    inp = make(limits_kw={"max_premium_per_trade": 30.0})               # 1 contract would fit; 2 do not
    before = inp.proposal.model_dump()
    d = pe.evaluate(inp)
    assert d.reasons == ["max_loss_per_trade_exceeded"]                 # rejected, not shrunk to 1
    assert d.calculated_max_loss == pytest.approx(44.6)
    assert inp.proposal.model_dump() == before
    assert inp.proposal.quantity == 2 and inp.proposal.limit_price == 2230.0


# 26
@pytest.mark.parametrize("label,kw,approved", [
    ("contracts == max", {"limits_kw": {"max_contracts": 2}}, True),
    ("max loss == limit", {"limits_kw": {"max_premium_per_trade": 44.6}}, True),
    ("max loss just over", {"limits_kw": {"max_premium_per_trade": 44.59}}, False),
    ("dte == min", {"quote_kw": {"dte": 7.0}}, True),
    ("dte == max", {"quote_kw": {"dte": 45.0}}, True),
    ("|delta| == min", {"quote_kw": {"delta": 0.35}}, True),
    ("|delta| == max", {"quote_kw": {"delta": 0.65}}, True),
    ("spread == max (10%)", {"quote_kw": {"bid": 1900.0, "ask": 2100.0}}, True),
    ("OI == min", {"quote_kw": {"open_interest": 100}}, True),
    ("volume == min", {"quote_kw": {"volume_24h": 10}}, True),
    ("quote age == max", {"quote_kw": {"as_of_ms": NOW - 60_000}}, True),
    ("move == max (0.5%)", {"underlying_price": 84420.0}, True),
    ("progress == max (50%)", {"underlying_price": 85000.0, "limits_kw": {"max_underlying_move_pct": 5.0}}, True),
    ("open positions == max-1", {"account_kw": {"open_position_count": 2}}, True),
    ("open positions == max", {"account_kw": {"open_position_count": 3}}, False),
    ("daily loss just under", {"account_kw": {"realized_pnl_today": -999.99}}, True),
    ("daily loss == limit", {"account_kw": {"realized_pnl_today": -1000.0}}, False),
])
def test_boundaries(label, kw, approved):
    """Ranges and maxima are inclusive; open-position count and daily loss block AT the limit."""
    d = pe.evaluate(make(**kw))
    assert d.approved is approved, (label, d.reasons)


# 27
def test_multiple_rejection_reasons_returned_together():
    r = reasons(make(
        kill_switch=KillSwitchState(env_enabled=True, file_enabled=False, file_path="x"),
        account_kw={"realized_pnl_today": -1500.0, "open_position_count": 3},
        quote_kw={"dte": 3.0, "open_interest": 10},
    ))
    assert r == ["kill_switch_enabled", "max_daily_loss_reached", "max_open_positions_reached",
                 "dte_below_min", "open_interest_below_min"]


#
# Determinism / idempotency / audit
#

def test_same_snapshot_same_decision():
    a, b = pe.evaluate(make()), pe.evaluate(make())
    assert a == b
    assert a.evaluation_key == b.evaluation_key and a.decision_id == b.decision_id


def test_newer_snapshot_is_a_new_evaluation():
    a = pe.evaluate(make())
    b = pe.evaluate(make(now_ms=NOW + 1000, quote_kw={"as_of_ms": NOW + 1000}))
    assert a.evaluation_key != b.evaluation_key


def test_persist_is_idempotent_per_snapshot():
    inp = make()
    first = pe.persist_decision(pe.evaluate(inp), inp)
    second = pe.persist_decision(pe.evaluate(inp), inp)
    assert first == second

    newer = make(now_ms=NOW + 5000, quote_kw={"as_of_ms": NOW + 5000})
    pe.persist_decision(pe.evaluate(newer), newer)

    conn = sqlite3.connect(config.db_path())
    assert conn.execute("SELECT COUNT(*) FROM policy_decisions").fetchone()[0] == 2


# 30
def test_historical_decision_retains_policy_snapshot(monkeypatch):
    inp = make()
    old = pe.persist_decision(pe.evaluate(inp), inp)
    assert old.approved

    monkeypatch.setenv("MAX_PREMIUM_PER_TRADE", "10")          # config changes later
    newer = make(now_ms=NOW + 1000, quote_kw={"as_of_ms": NOW + 1000})
    new = pe.persist_decision(pe.evaluate(newer), newer)
    assert new.reasons == ["max_loss_per_trade_exceeded"]
    assert new.policy_snapshot.limits.max_premium_per_trade == 10

    stored = pe.get_decision(old.decision_id)
    assert stored.approved
    assert stored.policy_snapshot.limits.max_premium_per_trade == 500
    conn = sqlite3.connect(config.db_path())
    snap, inp_json = conn.execute(
        "SELECT policy_snapshot_json, input_json FROM policy_decisions WHERE decision_id = ?", (old.decision_id,)
    ).fetchone()
    assert json.loads(snap)["limits"]["max_premium_per_trade"] == 500
    assert json.loads(inp_json)["limits"]["max_premium_per_trade"] == 500


def test_db_enforces_decision_consistency():
    conn = sqlite3.connect(config.db_path())
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO policy_decisions (decision_id, evaluation_key, proposal_key, purpose, approved, decision,"
            " reasons_json, checked_at, input_json, policy_snapshot_json, decision_json, created_at)"
            " VALUES ('d', 'k', 'p', 'ENTRY', 1, 'REJECTED', '[]', 't', '{}', '{}', '{}', 't')"
        )


#
# Integration: DB snapshot gathering (read-only) + no submission
#

class Spot:
    def __init__(self, price):
        self.price = price

    def get_current_price(self, symbol):
        return self.price


class Quotes:
    def __init__(self, **kw):
        self.kw = kw
        self.error = None

    def get_option_contract(self, symbol):
        if self.error:
            raise self.error
        return contract(contract_symbol=symbol, as_of_ms=NOW, **self.kw)


def store_signal():
    from helpers import bullish_signal
    database.save_signal("42", bullish_signal(
        timeframe="1h", support_levels=[83000.0], resistance_levels=[84000.0], targets=[86000.0],
        confirmation_above=84000.0, invalidation_below=82600.0), T0, 83400.0, raw_message="raw")
    database.update_status(1, "WORKING")


def test_gather_and_evaluate_from_db_approves_clean_state():
    store_signal()
    d = pe.evaluate_entry(proposal(), Quotes(), Spot(84200.0), NOW)
    assert d.approved, d.reasons


def test_gather_marks_fetch_failures_as_rejections():
    store_signal()
    quotes = Quotes()
    quotes.error = ConnectionError("down")

    class DownSpot:
        def get_current_price(self, s):
            raise TimeoutError()

    d = pe.evaluate_entry(proposal(), quotes, DownSpot(), NOW)
    assert "quote_unavailable" in d.reasons and "underlying_price_unavailable" in d.reasons


def test_missing_signal_rejected():
    d = pe.evaluate_entry(proposal(signal_id=999, proposal_key="signal:999:entry"), Quotes(), Spot(84200.0), NOW)
    assert "signal_not_found" in d.reasons


def test_duplicates_detected_from_paper_broker_state():
    from broker.paper_broker import PaperBroker
    from schemas import OrderRequest

    store_signal()
    paper = PaperBroker(Quotes(), now_ms=lambda: NOW, log=lambda *a: None)
    paper.submit_order(OrderRequest(client_order_id="signal:1:entry:open", signal_id=1, proposal_key="signal:1:entry",
                                    contract_symbol=SYMBOL, side="BUY_TO_OPEN", quantity=2, limit_price=2260.0))

    d = pe.evaluate_entry(proposal(), Quotes(), Spot(84200.0), NOW)
    assert "duplicate_signal_position" in d.reasons
    assert "duplicate_proposal_entry" in d.reasons
    inp = json.loads(sqlite3.connect(config.db_path()).execute(
        "SELECT input_json FROM policy_decisions WHERE decision_id = ?", (d.decision_id,)).fetchone()[0])
    assert inp["account"]["open_position_count"] == 1
    assert inp["account"]["open_premium_at_risk"] == pytest.approx(45.2)


def test_realized_daily_loss_read_from_fills():
    from broker.paper_broker import PaperBroker
    from schemas import OrderRequest

    quotes = Quotes()
    paper = PaperBroker(quotes, now_ms=lambda: NOW, log=lambda *a: None)
    r = paper.submit_order(OrderRequest(client_order_id="o", signal_id=5, proposal_key="signal:5:entry",
                                        contract_symbol=SYMBOL, side="BUY_TO_OPEN", quantity=2, limit_price=2260.0))
    quotes.kw = {"bid": 1100.0, "ask": 1150.0}
    paper.close_position(r.position.position_id, 1100.0, "c")

    account, *_ = pe.portfolio_state(NOW)
    assert account.realized_pnl_today == pytest.approx((1100 - 2260) * 0.01 * 2)
    assert account.open_position_count == 0


# 28
def test_policy_evaluation_never_submits_an_order(monkeypatch):
    from broker.paper_broker import PaperBroker

    def forbidden(*a, **k):
        raise AssertionError("policy engine must never submit")

    for name in ("submit_order", "close_position", "cancel_order"):
        monkeypatch.setattr(PaperBroker, name, forbidden)

    store_signal()
    d = pe.evaluate_entry(proposal(), Quotes(), Spot(84200.0), NOW)
    assert d.approved

    conn = sqlite3.connect(config.db_path())
    assert conn.execute("SELECT COUNT(*) FROM paper_orders").fetchone()[0] == 0

    source = open(pe.__file__).read()
    code = "\n".join(l for l in source.splitlines() if not l.strip().startswith("#"))
    for banned in ("import broker", "from broker", ".submit_order(", ".close_position(", ".cancel_order("):
        assert banned not in code


# 29
def test_policy_engine_has_no_llm_or_agent_or_broker_dependency():
    probe = (
        "import sys; import core.policy_engine; "
        "bad = sorted(m for m in sys.modules if m == 'openai' or m.startswith('openai.') "
        "or m in ('llm_client', 'llm_parser') or m.startswith('agents') or m.startswith('broker')); "
        "print(','.join(bad))"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=config.PROJECT_DIR, env=dict(os.environ),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
