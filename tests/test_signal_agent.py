"""Agent 1 — Signal Analysis. All offline: the LLM is a fake."""
import json

import pytest

import database
from agents.signal_agent import SignalAnalysisAgent, build_input
from schemas import SignalAnalysisInput, SignalAnalysisLLMOutput

from helpers import T0, bearish_signal, bullish_signal


NOW = T0 + 5 * 60_000


class FakeLLM:
    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error
        self.calls = []

    def structured(self, system, user, schema):
        self.calls.append({"system": system, "user": user, "schema": schema})
        if self.error:
            raise self.error
        return self.output


def llm_out(**overrides):
    data = dict(
        direction="bullish",
        entry_condition_type="break_above",
        entry_level=100.5,
        target=102.0,
        invalidation=98.0,
        time_horizon="intraday",
        signal_state="DEVELOPING",
        confidence=0.8,
        reasons=["clean breakout setup"],
    )
    data.update(overrides)
    return SignalAnalysisLLMOutput(**data)


def make_input(signal=None, current_price=100.0, price_at_signal=100.0, **overrides):
    signal = signal or bullish_signal()
    data = dict(
        signal_id=1,
        discord_message_id="42",
        raw_message="raw text",
        symbol=signal.symbol,
        bias=signal.bias,
        timeframe=signal.timeframe,
        summary=signal.summary,
        support_levels=signal.support_levels,
        resistance_levels=signal.resistance_levels,
        targets=signal.targets,
        confirmation_above=signal.confirmation_above,
        confirmation_below=signal.confirmation_below,
        invalidation_above=signal.invalidation_above,
        invalidation_below=signal.invalidation_below,
        signal_time_ms=T0,
        price_at_signal=price_at_signal,
        current_price=current_price,
    )
    data.update(overrides)
    return SignalAnalysisInput(**data)


def analyze(inp, llm=None, now=NOW):
    return SignalAnalysisAgent(llm, now_ms=lambda: now, log=lambda *a: None).analyze(inp)


def input_levels(inp):
    return {x for x in [inp.confirmation_above, inp.confirmation_below,
                        inp.invalidation_above, inp.invalidation_below,
                        *inp.support_levels, *inp.resistance_levels, *inp.targets]
            if x is not None}


def assert_no_invented_levels(inp, result):
    for level in (result.entry_level, result.target, result.invalidation):
        assert level is None or level in input_levels(inp)


def test_bullish_developing():
    inp = make_input()
    result = analyze(inp, FakeLLM(llm_out()))

    assert result.underlying == "BTCUSDT"
    assert result.direction == "bullish"
    assert result.entry_condition_type == "break_above"
    assert (result.entry_level, result.target, result.invalidation) == (100.5, 102.0, 98.0)
    assert result.signal_state == "DEVELOPING"
    assert result.next_step == "WAIT_FOR_CONFIRMATION"
    assert result.eligible_for_options is False
    assert result.llm_used is True
    assert result.confidence == 0.8
    assert_no_invented_levels(inp, result)


def test_bearish_developing():
    inp = make_input(bearish_signal())
    out = llm_out(direction="bearish", entry_condition_type="break_below",
                  entry_level=99.5, target=97.0, invalidation=101.0, time_horizon="swing")
    result = analyze(inp, FakeLLM(out))

    assert result.direction == "bearish"
    assert (result.entry_level, result.target, result.invalidation) == (99.5, 97.0, 101.0)
    assert result.signal_state == "DEVELOPING"
    assert_no_invented_levels(inp, result)


def test_already_confirmed_and_fresh():
    inp = make_input(current_price=100.7)          # 13% of the way to target
    result = analyze(inp, FakeLLM(llm_out(signal_state="ALREADY_CONFIRMED")))

    assert result.signal_state == "ALREADY_CONFIRMED"
    assert result.next_step == "ELIGIBLE_FOR_OPTIONS_REVIEW"
    assert result.eligible_for_options is True
    assert result.facts.target_progress_pct == pytest.approx(13.33, abs=0.01)


def test_already_confirmed_but_stale_from_progress():
    inp = make_input(current_price=101.6)          # 73% of entry->target done
    result = analyze(inp, FakeLLM(llm_out(signal_state="ALREADY_CONFIRMED")))

    assert result.signal_state == "STALE"
    assert result.eligible_for_options is False
    assert result.next_step == "NO_ACTION"


def test_progress_threshold_configurable(monkeypatch):
    monkeypatch.setenv("AGENT1_MAX_TARGET_PROGRESS_PCT", "80")
    result = analyze(make_input(current_price=101.6))
    assert result.signal_state == "ALREADY_CONFIRMED"


def test_llm_cannot_call_crossed_signal_developing():
    result = analyze(make_input(current_price=100.7), FakeLLM(llm_out(signal_state="DEVELOPING")))

    assert result.signal_state == "ALREADY_CONFIRMED"
    assert any("ignored" in n for n in result.validation_notes)


def test_llm_cannot_relax_invalidated_signal():
    result = analyze(make_input(current_price=97.9), FakeLLM(llm_out(signal_state="DEVELOPING")))

    assert result.signal_state == "STALE"
    assert result.facts.invalidation_hit is True


def test_target_already_reached_is_stale():
    result = analyze(make_input(current_price=102.5), FakeLLM(llm_out(signal_state="ALREADY_CONFIRMED")))
    assert result.signal_state == "STALE"


def test_llm_may_escalate_to_not_tradeable():
    result = analyze(make_input(), FakeLLM(llm_out(signal_state="NOT_TRADEABLE")))

    assert result.signal_state == "NOT_TRADEABLE"
    assert result.next_step == "NO_ACTION"


def test_invented_levels_rejected():
    inp = make_input()
    out = llm_out(entry_level=100.55, target=105.0, invalidation=97.5)
    result = analyze(inp, FakeLLM(out))

    assert (result.entry_level, result.target, result.invalidation) == (100.5, 102.0, 98.0)
    assert sum("not in parsed signal levels" in n for n in result.validation_notes) == 3
    assert_no_invented_levels(inp, result)


def test_directionally_inconsistent_target_not_used():
    inp = make_input(bullish_signal(targets=[99.0, 102.0]))
    result = analyze(inp, FakeLLM(llm_out(target=99.0)))

    assert result.target == 102.0
    assert any("99.0" in n for n in result.validation_notes)


def test_neutral_bias_not_tradeable():
    inp = make_input(bullish_signal(bias="neutral"))
    result = analyze(inp, FakeLLM(llm_out(direction="neutral", entry_condition_type="none")))

    assert result.signal_state == "NOT_TRADEABLE"
    assert result.direction == "neutral"


def test_missing_bias_is_not_guessed():
    result = analyze(make_input(bullish_signal(bias=None)), FakeLLM(llm_out()))

    assert result.direction == "neutral"
    assert result.signal_state == "NOT_TRADEABLE"


def test_no_target_not_tradeable():
    result = analyze(make_input(bullish_signal(targets=[])))
    assert result.signal_state == "NOT_TRADEABLE"
    assert "no target in parsed signal" in result.reasons


def test_missing_invalidation_policy(monkeypatch):
    inp = make_input(bullish_signal(invalidation_below=None))
    assert analyze(inp).signal_state == "NOT_TRADEABLE"

    monkeypatch.setenv("AGENT1_REQUIRE_INVALIDATION", "false")
    assert analyze(inp).signal_state == "DEVELOPING"


def test_llm_direction_contradiction_blocks():
    result = analyze(make_input(), FakeLLM(llm_out(direction="bearish")))

    assert result.signal_state == "NOT_TRADEABLE"
    assert result.direction == "bullish"


def test_llm_failure_falls_back_to_deterministic():
    result = analyze(make_input(), FakeLLM(error=TimeoutError("openai timeout")))

    assert result.llm_used is False
    assert result.signal_state == "DEVELOPING"
    assert result.confidence == 0.0
    assert any("LLM unavailable" in n for n in result.validation_notes)


def test_expired_signal_is_stale(monkeypatch):
    monkeypatch.setenv("SIGNAL_EXPIRY_HOURS_INTRADAY", "24")
    result = analyze(make_input(), now=T0 + 25 * 3_600_000)
    assert result.signal_state == "STALE"


def test_immediate_entry_uses_price_at_signal_for_progress():
    inp = make_input(bullish_signal(confirmation_above=None), current_price=100.4, price_at_signal=100.0)
    result = analyze(inp)

    assert result.entry_condition_type == "immediate"
    assert result.entry_level is None
    assert result.signal_state == "ALREADY_CONFIRMED"
    assert result.facts.target_progress_pct == pytest.approx(20.0)


def test_confidence_clamped():
    assert analyze(make_input(), FakeLLM(llm_out(confidence=7))).confidence == 1.0


def test_inconsistent_entry_type_replaced():
    result = analyze(make_input(), FakeLLM(llm_out(entry_condition_type="break_below")))
    assert result.entry_condition_type == "break_above"


def test_immediate_entry_type_rejected_when_entry_level_exists():
    # Seen live: LLM said "immediate" while an entry level existed.
    result = analyze(make_input(current_price=100.7), FakeLLM(llm_out(entry_condition_type="immediate")))
    assert result.entry_condition_type == "break_above"
    assert result.entry_level == 100.5


def test_break_entry_type_rejected_when_no_entry_level():
    inp = make_input(bullish_signal(confirmation_above=None), current_price=100.4)
    result = analyze(inp, FakeLLM(llm_out(entry_condition_type="break_above", entry_level=None)))
    assert result.entry_condition_type == "immediate"


def test_prompt_contains_allowed_levels_and_no_secrets():
    llm = FakeLLM(llm_out())
    analyze(make_input(), llm)

    payload = json.loads(llm.calls[0]["user"])
    assert payload["ALLOWED_LEVELS"]["target"] == [102.0]
    assert payload["ALLOWED_LEVELS"]["invalidation"] == [98.0]
    assert "test-openai-key" not in llm.calls[0]["user"]
    assert llm.calls[0]["schema"] is SignalAnalysisLLMOutput


def test_build_input_from_db_row():
    database.save_signal("77", bullish_signal(), T0, 100.0, raw_message="the raw text")
    row = database.get_signal_by_discord_message_id("77")

    inp = build_input(row, current_price=100.2)

    assert inp.raw_message == "the raw text"
    assert inp.timeframe == "1h"
    assert inp.targets == [102.0]
    assert inp.current_price == 100.2
    assert inp.price_at_signal == 100.0
