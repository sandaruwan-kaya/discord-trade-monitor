"""Agent 3 — risk analysis (offline). Hard rules first; LLM is a fake."""
import os
import sqlite3
import subprocess
import sys

import pytest

import config
import database
from agents.risk_agent import RiskAnalysisAgent
from core import risk_rules
from risk_service import RiskService, gather_risk_input
from schemas import (
    Candle,
    LastLLMAssessment,
    RiskAnalysisInput,
    RiskLLMOutput,
    RiskPosition,
    RiskThesis,
)

from helpers import MINUTE_MS, T0
from options_helpers import contract


NOW = T0 + 2 * 3_600_000
HOUR = 3_600_000
SYMBOL = "BTC-9OCT26-85000-C"


class FakeLLM:
    def __init__(self, recommendation="HOLD", error=None, **kw):
        self.out = RiskLLMOutput(recommendation=recommendation, confidence=kw.get("confidence", 0.7),
                                 reason_codes=kw.get("reason_codes", ["THESIS_INTACT"]),
                                 rationale=kw.get("rationale", "Thesis intact; price between entry and target."))
        self.error = error
        self.calls = 0

    def structured(self, system, user, schema):
        self.calls += 1
        if self.error:
            raise self.error
        return self.out


def thesis(**kw):
    data = dict(direction="bullish", entry_level=84000.0, target=86000.0, invalidation=82600.0,
                timeframe="1h", signal_time_ms=T0, signal_status="WORKING",
                recent_high=84600.0, recent_low=83300.0)
    data.update(kw)
    return RiskThesis(**data)


def make(quote_kw=None, thesis_kw=None, position_kw=None, limits_kw=None, **top):
    quote = top.pop("quote", "default")
    data = dict(
        position_id="PP-1", signal_id=1, proposal_key="signal:1:entry", contract_symbol=SYMBOL,
        thesis=thesis(**(thesis_kw or {})),
        position=RiskPosition(**{"status": "OPEN", "quantity": 2, "contract_multiplier": 0.01,
                                 "entry_option_price": 2260.0, "opened_at_ms": NOW - 2 * HOUR,
                                 **(position_kw or {})}),
        quote=contract(**{"bid": 2300.0, "ask": 2360.0, "mark": 2330.0, "as_of_ms": NOW,
                          "expiry_ts_ms": NOW + 14 * 24 * HOUR, **(quote_kw or {})}) if quote == "default" else quote,
        underlying_price=84500.0,
        underlying_price_at_entry=84200.0,
        last_candle=Candle(timestamp=NOW - MINUTE_MS, high=84550.0, low=84450.0, close=84500.0),
        now_ms=NOW,
        limits=risk_rules.current_limits().model_copy(update=limits_kw or {}),
    )
    data.update(top)
    return RiskAnalysisInput(**data)


def assess(inp, llm=None, last=None, previous=None):
    return RiskAnalysisAgent(llm, log=lambda *a: None).assess(inp, last, previous)


# 1
def test_healthy_position_hold():
    llm = FakeLLM("HOLD")
    a = assess(make(), llm)
    assert a.recommendation == "HOLD" and a.risk_state == "CONTEXTUAL"
    assert not a.hard_rule_triggered and a.hard_rule_codes == []
    assert a.llm_used and llm.calls == 1 and a.llm_trigger == "first_contextual_assessment"


# 2
def test_underlying_invalidation_hard_exit_without_llm():
    llm = FakeLLM("HOLD")
    a = assess(make(underlying_price=82500.0,
                    last_candle=Candle(timestamp=NOW - MINUTE_MS, high=82700.0, low=82450.0, close=82500.0)), llm)
    assert a.recommendation == "EXIT" and a.hard_rule_triggered
    assert a.hard_rule_codes == ["UNDERLYING_INVALIDATION_HIT"]
    assert llm.calls == 0 and not a.llm_used and a.llm_status == "SKIPPED_HARD_EXIT"


# 3
def test_target_hit_hard_exit_without_llm():
    llm = FakeLLM("HOLD")
    a = assess(make(underlying_price=86100.0,
                    last_candle=Candle(timestamp=NOW - MINUTE_MS, high=86150.0, low=85900.0, close=86100.0)), llm)
    assert a.hard_rule_codes == ["UNDERLYING_TARGET_HIT"]
    assert a.recommendation == "EXIT" and llm.calls == 0


# 4 / 5
@pytest.mark.parametrize("status,code", [("INVALIDATED", "SIGNAL_INVALIDATED"), ("TARGET_HIT", "SIGNAL_TARGET_HIT")])
def test_signal_status_hard_exit(status, code):
    llm = FakeLLM("HOLD")
    a = assess(make(thesis_kw={"signal_status": status}), llm)
    assert a.hard_rule_codes == [code] and a.recommendation == "EXIT" and llm.calls == 0


# 6
def test_premium_stop_hit_using_bid():
    a = assess(make(quote_kw={"bid": 1130.0, "ask": 1180.0, "mark": 1155.0}), FakeLLM("HOLD"))
    assert a.hard_rule_codes == ["HARD_PREMIUM_STOP"]
    assert a.unrealized_pnl_pct == pytest.approx(-50.0)                     # (1130-2260)/2260


# 7
def test_mark_looks_safe_but_bid_breaches_stop():
    # mark 1200 -> -46.9% (safe), bid 1100 -> -51.3% (stop)
    a = assess(make(quote_kw={"bid": 1100.0, "ask": 1300.0, "mark": 1200.0}), FakeLLM("HOLD"))
    assert a.hard_rule_codes == ["HARD_PREMIUM_STOP"]
    assert a.metrics.mark_pnl / a.metrics.entry_cost * 100 == pytest.approx(-46.9, abs=0.1)
    assert a.unrealized_pnl_pct == pytest.approx(-51.33, abs=0.01)


# 8
def test_max_holding_period_exceeded():
    a = assess(make(position_kw={"opened_at_ms": NOW - 5 * 24 * HOUR}), FakeLLM("HOLD"))
    assert a.hard_rule_codes == ["MAX_HOLD_EXCEEDED"]


def test_just_under_max_hold_is_not_an_exit():
    a = assess(make(position_kw={"opened_at_ms": NOW - 5 * 24 * HOUR + 1000}), FakeLLM("HOLD"))
    assert not a.hard_rule_triggered


# 9
def test_absolute_max_loss_exceeded():
    # bid 1800 -> loss (2260-1800)*0.02 = $9.20 (-20%, above the 50% stop)
    a = assess(make(quote_kw={"bid": 1800.0, "ask": 1850.0, "mark": 1825.0}, limits_kw={"max_position_loss": 9.0}),
               FakeLLM("HOLD"))
    assert a.hard_rule_codes == ["MAX_POSITION_LOSS"]


def test_max_loss_disabled_by_default():
    assert risk_rules.current_limits().max_position_loss == 0
    a = assess(make(quote_kw={"bid": 1800.0, "ask": 1850.0, "mark": 1825.0}), FakeLLM("HOLD"))
    assert not a.hard_rule_triggered


def test_option_expired_is_hard_exit():
    a = assess(make(quote_kw={"expiry_ts_ms": NOW - 1}), FakeLLM("HOLD"))
    assert "OPTION_EXPIRED" in a.hard_rule_codes


# 10
def test_closed_position_is_not_analyzed():
    with pytest.raises(ValueError):
        assess(make(position_kw={"status": "CLOSED"}))


# 11 / 12
@pytest.mark.parametrize("kw,code", [
    ({"quote": None}, "QUOTE_UNAVAILABLE"),
    ({"quote_kw": {"bid": 0.0}}, "QUOTE_NO_BID"),
    ({"quote_kw": {"bid": None}}, "QUOTE_NO_BID"),
    ({"quote_kw": {"bid": 2400.0, "ask": 2300.0}}, "QUOTE_CROSSED"),
    ({"quote_kw": {"as_of_ms": NOW - 120_000}}, "QUOTE_STALE"),
    ({"quote_kw": {"contract_symbol": "BTC-9OCT26-84000-C"}}, "QUOTE_CONTRACT_MISMATCH"),
    ({"quote_kw": {"contract_multiplier": 0.1}}, "QUOTE_TERMS_CHANGED"),
    ({"underlying_price": None}, "UNDERLYING_PRICE_UNAVAILABLE"),
])
def test_unusable_data_is_data_unavailable(kw, code):
    llm = FakeLLM("HOLD")
    a = assess(make(**kw), llm)
    assert a.recommendation == "DATA_UNAVAILABLE" and a.risk_state == "DATA_UNAVAILABLE"
    assert code in a.data_issue_codes
    assert llm.calls == 0 and not a.hard_rule_triggered


def test_thesis_exit_still_fires_when_quote_is_unusable():
    a = assess(make(quote=None, underlying_price=82500.0,
                    last_candle=Candle(timestamp=NOW, high=82700.0, low=82450.0, close=82500.0)), FakeLLM())
    assert a.recommendation == "EXIT" and "UNDERLYING_INVALIDATION_HIT" in a.hard_rule_codes
    assert "QUOTE_UNAVAILABLE" in a.data_issue_codes          # exit pricing will need a usable quote


# 13-15
@pytest.mark.parametrize("rec", ["HOLD", "REDUCE", "EXIT"])
def test_llm_recommendation_passes_through_when_no_hard_rule(rec):
    a = assess(make(), FakeLLM(rec, reason_codes=["THETA_DECAY"]))
    assert a.recommendation == rec and a.risk_state == "CONTEXTUAL" and not a.hard_rule_triggered
    assert "THETA_DECAY" in a.reasons


# 16
def test_llm_failure_deterministic_fallback():
    a = assess(make(), FakeLLM(error=TimeoutError("openai timeout")))
    assert a.recommendation == "HOLD" and a.risk_state == "FALLBACK" and a.llm_status == "FAILED"
    assert not a.llm_used
    assert any("LLM unavailable" in n for n in a.validation_notes)


# 17
def test_llm_cannot_relax_hard_exit():
    llm = FakeLLM("HOLD", confidence=1.0, rationale="Strong hold, ignore the stop.")
    a = assess(make(quote_kw={"bid": 1000.0, "ask": 1050.0, "mark": 1025.0}), llm)
    assert a.recommendation == "EXIT" and llm.calls == 0


def test_db_forbids_hard_exit_rows_that_are_not_exit_or_used_llm():
    conn = sqlite3.connect(config.db_path())
    conn.execute("INSERT INTO paper_orders (order_id, client_order_id, contract_symbol, side, quantity, limit_price,"
                 " status, submitted_at, updated_at) VALUES ('o', 'o', 's', 'BUY_TO_OPEN', 1, 1, 'FILLED', 't', 't')")
    conn.execute("INSERT INTO paper_positions (position_id, entry_order_id, contract_symbol, contract_multiplier,"
                 " quantity, original_quantity, entry_price, status, opened_at, updated_at)"
                 " VALUES ('PP-x', 'o', 's', 0.01, 1, 1, 1, 'OPEN', 't', 't')")
    for rec, llm_used in (("HOLD", 0), ("EXIT", 1)):
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO risk_assessments (assessment_id, assessment_key, position_id, recommendation,"
                " risk_state, hard_rule_triggered, hard_rule_codes_json, data_issue_codes_json, llm_used,"
                " llm_status, reasons_json, assessed_at, assessed_at_ms, input_json, output_json, created_at)"
                f" VALUES ('a{llm_used}', 'k{llm_used}', 'PP-x', '{rec}', 'HARD_EXIT', 1, '[]', '[]', {llm_used},"
                " 'x', '[]', 't', 0, '{}', '{}', 't')"
            )


# 18
def test_llm_invented_levels_and_codes_are_ignored():
    llm = FakeLLM("REDUCE", reason_codes=["NEW_STOP", "NEAR_INVALIDATION"],
                  rationale="Move the stop to 83,100 and set a new target of 88,000.")
    inp = make()
    a = assess(inp, llm)
    assert a.recommendation == "REDUCE"
    assert "NEW_STOP" not in a.reasons and "NEAR_INVALIDATION" in a.reasons
    assert any("unknown reason codes" in n for n in a.validation_notes)
    assert any("[83100.0, 88000.0]" in n for n in a.validation_notes)
    snap = a.input_snapshot.thesis
    assert (snap.entry_level, snap.target, snap.invalidation) == (84000.0, 86000.0, 82600.0)
    assert "stop" not in RiskLLMOutput.model_fields and "target" not in RiskLLMOutput.model_fields


# 19-21: throttle
def last(minutes_ago=5, price=84500.0, pnl_pct=1.77):
    return LastLLMAssessment(assessed_at_ms=NOW - minutes_ago * MINUTE_MS, underlying_price=price,
                             unrealized_pnl_pct=pnl_pct, recommendation="HOLD", risk_state="CONTEXTUAL",
                             confidence=0.7)


def test_throttled_llm_does_not_run():
    llm = FakeLLM("EXIT")
    a = assess(make(), llm, last=last(5), previous="CONTEXTUAL")
    assert llm.calls == 0
    assert a.risk_state == "THROTTLED" and a.recommendation == "HOLD"
    assert a.last_contextual_recommendation == "HOLD"


def test_interval_elapsed_runs_llm():
    llm = FakeLLM("HOLD")
    a = assess(make(), llm, last=last(15), previous="CONTEXTUAL")
    assert llm.calls == 1 and a.llm_trigger == "interval_elapsed"


def test_material_move_bypasses_throttle():
    llm = FakeLLM("REDUCE")
    a = assess(make(underlying_price=84100.0), llm, last=last(2, price=84550.0), previous="CONTEXTUAL")
    assert llm.calls == 1 and a.llm_trigger.startswith("material_underlying_move")


def test_material_pnl_change_bypasses_throttle():
    llm = FakeLLM("HOLD")
    a = assess(make(), llm, last=last(2, pnl_pct=15.0), previous="CONTEXTUAL")
    assert llm.calls == 1 and a.llm_trigger.startswith("material_pnl_change")


def test_recovery_from_data_unavailable_bypasses_throttle():
    llm = FakeLLM("HOLD")
    a = assess(make(), llm, last=last(2), previous="DATA_UNAVAILABLE")
    assert llm.calls == 1 and a.llm_trigger.startswith("risk_state_changed")


def test_hard_stop_bypasses_throttle_immediately():
    llm = FakeLLM("HOLD")
    a = assess(make(quote_kw={"bid": 1000.0, "ask": 1050.0, "mark": 1025.0}), llm, last=last(1), previous="CONTEXTUAL")
    assert a.recommendation == "EXIT" and a.hard_rule_codes == ["HARD_PREMIUM_STOP"]
    assert llm.calls == 0
    assert a.last_contextual_recommendation == "HOLD"                       # old HOLD kept separately, not used


# 22-24: metrics
def test_pnl_uses_bid_not_mark():
    m = risk_rules.compute_metrics(make(quote_kw={"bid": 78.0, "ask": 86.0, "mark": 82.0},
                                        position_kw={"entry_option_price": 100.0}))
    assert m.unrealized_pnl == pytest.approx((78 - 100) * 0.02)
    assert m.unrealized_pnl_pct == pytest.approx(-22.0)
    assert m.liquidation_value == pytest.approx(78 * 0.02)
    assert m.mark_pnl == pytest.approx((82 - 100) * 0.02)                   # display only


def test_target_progress():
    assert risk_rules.compute_metrics(make(underlying_price=85000.0)).target_progress_pct == pytest.approx(50.0)
    bear = make(thesis_kw={"direction": "bearish", "entry_level": 2620.0, "target": 2540.0, "invalidation": 2720.0},
                underlying_price=2600.0, underlying_price_at_entry=2615.0,
                last_candle=Candle(timestamp=NOW, high=2605.0, low=2595.0, close=2600.0))
    assert risk_rules.compute_metrics(bear).target_progress_pct == pytest.approx(25.0)


def test_distance_to_invalidation():
    m = risk_rules.compute_metrics(make(underlying_price=84500.0))
    assert m.distance_to_invalidation_pct == pytest.approx((84500 - 82600) / 84500 * 100)
    assert m.distance_to_target_pct == pytest.approx((86000 - 84500) / 84500 * 100)
    assert m.underlying_move_since_entry_pct == pytest.approx((84500 - 84200) / 84200 * 100)


# 30
def test_ambiguous_intrabar_order_is_deterministic():
    llm = FakeLLM("HOLD")
    a = assess(make(underlying_price=84300.0,
                    last_candle=Candle(timestamp=NOW, high=86100.0, low=82500.0, close=84300.0)), llm)
    assert a.hard_rule_codes == ["AMBIGUOUS_INTRABAR_ORDER"]
    assert a.recommendation == "EXIT" and llm.calls == 0


def test_evaluator_resolution_wins_over_ambiguity():
    a = assess(make(thesis_kw={"signal_status": "TARGET_HIT", "evaluator_target_time_ms": NOW - MINUTE_MS},
                    last_candle=Candle(timestamp=NOW, high=86100.0, low=82500.0, close=84300.0)))
    assert a.hard_rule_codes == ["SIGNAL_TARGET_HIT"]


#
# Service (DB) tests
#

class Spot:
    def __init__(self, price=84500.0, candle=None):
        self.price = price
        self.candle = candle or {"timestamp": NOW - MINUTE_MS, "open": 84500.0, "high": 84550.0,
                                 "low": 84450.0, "close": 84500.0, "volume": 1.0}

    def get_current_price(self, symbol):
        return self.price

    def get_candles(self, symbol, interval="1m", limit=200, start_ms=None, end_ms=None):
        if start_ms is not None:
            return [{**self.candle, "timestamp": start_ms, "close": 84200.0}]
        return [self.candle]


class Quotes:
    def __init__(self, **kw):
        self.kw = kw

    def get_option_contract(self, symbol):
        return contract(**{"contract_symbol": symbol, "bid": 2300.0, "ask": 2360.0, "mark": 2330.0,
                           "as_of_ms": self.now, **self.kw})

    now = NOW


def open_position(opened_ms=NOW - 2 * HOUR):
    from broker.paper_broker import PaperBroker
    from schemas import OrderRequest
    from helpers import bullish_signal

    database.save_signal("42", bullish_signal(timeframe="1h", support_levels=[83000.0], resistance_levels=[84000.0],
                                              targets=[86000.0], confirmation_above=84000.0,
                                              invalidation_below=82600.0), T0, 83400.0, raw_message="raw")
    database.update_status(1, "WORKING")
    q = Quotes(bid=2200.0, ask=2260.0, mark=2230.0)
    broker = PaperBroker(q, now_ms=lambda: opened_ms, log=lambda *a: None)
    return broker.submit_order(OrderRequest(client_order_id="entry:signal:1:entry", signal_id=1,
                                            proposal_key="signal:1:entry", contract_symbol=SYMBOL,
                                            side="BUY_TO_OPEN", quantity=2, limit_price=2260.0)).position


def svc(llm, clock, spot=None, quotes=None):
    quotes = quotes or Quotes()
    return RiskService(RiskAnalysisAgent(llm, log=lambda *a: None), quotes, spot or Spot(), now_ms=clock), quotes


def count():
    return sqlite3.connect(config.db_path()).execute("SELECT COUNT(*) FROM risk_assessments").fetchone()[0]


def test_service_assesses_open_position_from_db():
    pos = open_position()
    llm = FakeLLM("HOLD")
    service, _ = svc(llm, lambda: NOW)
    a = service.assess_position(pos.position_id)
    assert a.recommendation == "HOLD"
    assert a.input_snapshot.underlying_price_at_entry == 84200.0
    assert a.input_snapshot.thesis.target == 86000.0
    assert a.unrealized_pnl == pytest.approx((2300 - 2260) * 0.02)


# 10 (service)
def test_service_skips_closed_position():
    from broker.paper_broker import PaperBroker
    pos = open_position()
    q = Quotes(bid=2300.0, ask=2360.0)
    PaperBroker(q, now_ms=lambda: NOW, log=lambda *a: None).close_position(pos.position_id, 2300.0, "close")
    llm = FakeLLM("HOLD")
    service, _ = svc(llm, lambda: NOW)
    assert service.assess_position(pos.position_id) is None
    assert count() == 0 and llm.calls == 0


# 25 / 26
def test_same_snapshot_is_idempotent_and_new_quote_creates_new_assessment():
    pos = open_position()
    llm = FakeLLM("HOLD")
    clock = {"now": NOW}
    service, quotes = svc(llm, lambda: clock["now"])

    first = service.assess_position(pos.position_id)
    second = service.assess_position(pos.position_id)
    assert first == second and count() == 1 and llm.calls == 1

    clock["now"] = NOW + MINUTE_MS
    quotes.now = NOW + MINUTE_MS
    third = service.assess_position(pos.position_id)
    assert third.assessment_key != first.assessment_key and count() == 2
    assert third.risk_state == "THROTTLED" and llm.calls == 1            # 1 minute later: throttled


def test_service_throttle_uses_persisted_history_and_hard_rules_every_cycle():
    pos = open_position()
    llm = FakeLLM("HOLD")
    clock = {"now": NOW}
    quotes = Quotes()
    service, _ = svc(llm, lambda: clock["now"], quotes=quotes)
    service.assess_position(pos.position_id)                             # LLM #1

    for minute in range(1, 6):                                           # throttled cycles
        clock["now"] = quotes.now = NOW + minute * MINUTE_MS
        assert service.assess_position(pos.position_id).risk_state == "THROTTLED"
    assert llm.calls == 1

    clock["now"] = quotes.now = NOW + 6 * MINUTE_MS
    quotes.kw = {"bid": 1000.0, "ask": 1050.0, "mark": 1025.0}           # premium collapses
    a = service.assess_position(pos.position_id)
    assert a.recommendation == "EXIT" and a.hard_rule_codes == ["HARD_PREMIUM_STOP"] and llm.calls == 1


# 29
def test_no_position_mutation():
    pos = open_position()
    before = sqlite3.connect(config.db_path()).execute("SELECT * FROM paper_positions").fetchall()
    orders_before = sqlite3.connect(config.db_path()).execute("SELECT * FROM paper_orders").fetchall()
    service, _ = svc(FakeLLM("EXIT"), lambda: NOW)
    assert service.assess_position(pos.position_id).recommendation == "EXIT"
    assert sqlite3.connect(config.db_path()).execute("SELECT * FROM paper_positions").fetchall() == before
    assert sqlite3.connect(config.db_path()).execute("SELECT * FROM paper_orders").fetchall() == orders_before


# 27 / 28
def _code(rel):
    text = (config.PROJECT_DIR / rel).read_text()
    return "\n".join(l for l in text.splitlines() if not l.strip().startswith("#"))


def test_no_broker_or_execution_dependency_in_source():
    for rel in ("agents/risk_agent.py", "core/risk_rules.py", "risk_service.py"):
        code = _code(rel)
        for banned in ("import broker", "from broker", "BrokerBase", "PaperBroker", "execution_service",
                       "ExecutionService", ".submit_order(", ".close_position(", ".cancel_order("):
            assert banned not in code, (rel, banned)


def test_no_broker_or_execution_modules_loaded():
    probe = (
        "import sys; import agents.risk_agent, risk_service, core.risk_rules; "
        "bad = sorted(m for m in sys.modules if m.startswith('broker') or m == 'core.execution_service'); "
        "print(','.join(bad))"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=config.PROJECT_DIR, env=dict(os.environ),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_hard_rules_module_has_no_llm_dependency():
    probe = ("import sys; import core.risk_rules; "
             "print(','.join(m for m in sys.modules if m in ('openai', 'llm_client') or m.startswith('agents')))")
    result = subprocess.run([sys.executable, "-c", probe], cwd=config.PROJECT_DIR, env=dict(os.environ),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0 and result.stdout.strip() == ""
