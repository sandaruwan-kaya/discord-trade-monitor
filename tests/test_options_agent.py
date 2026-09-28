"""Agent 2 — Options Selection. LLM is a fake; no network."""
import json

import pytest
from pydantic import ValidationError

from agents.options_agent import OptionsSelectionAgent, proposal_key
from schemas import OptionProposal, OptionsSelectionInput, OptionsSelectionLLMOutput
from tools.options_tools import prefilter

from helpers import T0
from options_helpers import bull_thesis, chain, contract, playbook


class FakeLLM:
    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error
        self.calls = []

    def structured(self, system, user, schema):
        self.calls.append({"user": user, "schema": schema})
        if self.error:
            raise self.error
        return self.output


def selection_input(contracts=None, thesis=None):
    contracts = contracts if contracts is not None else [
        contract(contract_symbol=f"BTC-9OCT26-{k}-C", strike=float(k), delta=d)
        for k, d in [(83000, 0.55), (84000, 0.50), (85000, 0.45)]
    ]
    thesis = thesis or bull_thesis()
    candidates, summary = prefilter(chain(contracts), thesis, playbook(), T0)
    return OptionsSelectionInput(
        signal_id=7, analysis_id=3, trigger="FIRST_WORKING", thesis=thesis,
        underlying_price=83400.0, as_of_ms=T0, playbook=playbook(),
        chain_source="test", candidates=candidates, filter_summary=summary,
    )


def llm(decision="SELECT", selected="C2", **kw):
    return OptionsSelectionLLMOutput(
        decision=decision, selected_candidate_id=selected,
        ranking=kw.get("ranking", ["C2", "C1"]), confidence=kw.get("confidence", 0.7),
        reason=kw.get("reason", "balanced delta and value at target"),
    )


def agent(fake):
    return OptionsSelectionAgent(fake, log=lambda *a: None)


def test_llm_selects_candidate_numbers_come_from_candidate():
    inp = selection_input()
    chosen = next(c for c in inp.candidates if c.candidate_id == "C2")

    p = agent(FakeLLM(llm(selected="C2"))).propose(inp)

    assert p.status == "PROPOSED"
    assert p.proposal_key == "signal:7:entry"
    assert p.selected_by == "llm"
    assert p.strategy == "LONG_CALL" and p.side == "BUY_TO_OPEN" and p.option_type == "call"
    assert p.contract_symbol == chosen.contract.contract_symbol
    assert p.limit_price == chosen.limit_price
    assert p.quantity == chosen.quantity
    assert p.estimated_max_loss == chosen.estimated_max_loss
    assert p.contract_multiplier == 0.01
    assert p.thesis_target == 86000.0 and p.thesis_invalidation == 82600.0


def test_unknown_candidate_falls_back_to_deterministic_top():
    inp = selection_input()
    p = agent(FakeLLM(llm(selected="C99"))).propose(inp)

    assert p.status == "PROPOSED"
    assert p.selected_by == "deterministic_fallback"
    assert p.contract_symbol == inp.candidates[0].contract.contract_symbol
    assert any("C99" in n for n in p.validation_notes)


def test_llm_cannot_inject_contract_symbol_as_id():
    inp = selection_input()
    p = agent(FakeLLM(llm(selected="BTC-9OCT26-99000-C"))).propose(inp)
    assert p.contract_symbol != "BTC-9OCT26-99000-C"
    assert p.selected_by == "deterministic_fallback"


def test_llm_may_decline():
    p = agent(FakeLLM(llm(decision="NO_TRADE", selected=None, reason="all too short-dated"))).propose(selection_input())

    assert p.status == "DECLINED"
    assert p.contract_symbol is None and p.quantity is None
    assert p.reason == "all too short-dated"


def test_llm_failure_uses_deterministic_top():
    inp = selection_input()
    p = agent(FakeLLM(error=TimeoutError("openai"))).propose(inp)

    assert p.status == "PROPOSED"
    assert p.llm_used is False
    assert p.selected_by == "deterministic_fallback"
    assert p.contract_symbol == inp.candidates[0].contract.contract_symbol


def test_no_candidates_skips_llm():
    fake = FakeLLM(llm())
    inp = selection_input(contracts=[contract(bid=1900.0)])      # spread too wide

    p = agent(fake).propose(inp)

    assert p.status == "NO_CANDIDATES"
    assert fake.calls == []
    assert p.filter_summary == {"spread_too_wide": 1}


def test_confidence_clamped():
    assert agent(FakeLLM(llm(confidence=5))).propose(selection_input()).confidence == 1.0


def test_prompt_only_contains_candidates_and_thesis():
    fake = FakeLLM(llm())
    inp = selection_input()
    agent(fake).propose(inp)

    payload = json.loads(fake.calls[0]["user"])
    assert [c["candidate_id"] for c in payload["candidates"]] == [c.candidate_id for c in inp.candidates]
    assert payload["thesis"]["target"] == 86000.0
    assert fake.calls[0]["schema"] is OptionsSelectionLLMOutput
    assert "test-openai-key" not in fake.calls[0]["user"]


def test_proposal_cannot_express_selling():
    with pytest.raises(ValidationError):
        OptionProposal(proposal_key="k", signal_id=1, underlying="BTCUSDT", status="DECLINED", side="SELL_TO_OPEN")


def test_proposal_rejects_strategy_type_mismatch():
    good = agent(FakeLLM(llm())).propose(selection_input())
    with pytest.raises(ValidationError):
        OptionProposal(**{**good.model_dump(), "option_type": "put"})


def test_proposal_rejects_limit_above_ask():
    good = agent(FakeLLM(llm())).propose(selection_input())
    with pytest.raises(ValidationError):
        OptionProposal(**{**good.model_dump(), "limit_price": good.ask + 5})


def test_proposal_key_format():
    assert proposal_key(42) == "signal:42:entry"
