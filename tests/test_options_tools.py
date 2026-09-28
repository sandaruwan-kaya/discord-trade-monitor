"""Mock chain + deterministic pre-filter (offline)."""
from datetime import datetime, timezone

import pytest

from tools.options_tools import (
    MockDeribitChainProvider,
    OptionChainError,
    black_scholes,
    deribit_symbol,
    prefilter,
    size_order,
)

from helpers import T0
from options_helpers import bear_thesis, bull_thesis, chain, contract, playbook


class Spot:
    def __init__(self, price):
        self.price = price

    def get_current_price(self, symbol):
        return self.price


def mock(price=83400.0, now=T0):
    return MockDeribitChainProvider(Spot(price), now_ms=lambda: now)


#
# Pricing
#

def test_black_scholes_put_call_parity():
    call = black_scholes("call", 100, 105, 30 / 365, 0.5)
    put = black_scholes("put", 100, 105, 30 / 365, 0.5)
    assert call["price"] - put["price"] == pytest.approx(100 - 105, abs=1e-9)
    assert 0 < call["delta"] < 1
    assert -1 < put["delta"] < 0
    assert call["delta"] - put["delta"] == pytest.approx(1.0)


def test_black_scholes_expired_is_intrinsic():
    assert black_scholes("call", 110, 100, 0, 0.5)["price"] == 10
    assert black_scholes("put", 110, 100, 0, 0.5)["price"] == 0


def test_deribit_symbol_format():
    from datetime import date
    assert deribit_symbol("BTC", date(2026, 10, 9), 85000.0, "call") == "BTC-9OCT26-85000-C"
    assert deribit_symbol("ETH", date(2026, 12, 25), 2537.5, "put") == "ETH-25DEC26-2537.5-P"


#
# Mock chain
#

def test_mock_chain_structure_and_multipliers():
    btc = mock().get_option_chain("BTCUSDT")
    eth = MockDeribitChainProvider(Spot(2650.0), now_ms=lambda: T0).get_option_chain("ETHUSDT")

    assert btc.source == "mock_deribit"
    assert btc.underlying_price == 83400.0
    assert {c.contract_multiplier for c in btc.contracts} == {0.01}
    assert {c.contract_multiplier for c in eth.contracts} == {0.1}
    assert all(c.style == "european" for c in btc.contracts)
    assert all(c.as_of_ms == T0 for c in btc.contracts)
    strikes = sorted({c.strike for c in btc.contracts})
    assert strikes[0] >= 83400 * 0.75 and strikes[-1] <= 83400 * 1.25
    assert all(s % 1000 == 0 for s in strikes)


def test_mock_multiplier_configurable(monkeypatch):
    monkeypatch.setenv("MOCK_OPTION_MULTIPLIER_BTC", "0.1")
    assert mock().get_option_chain("BTCUSDT").contracts[0].contract_multiplier == 0.1


def test_mock_expiries_are_0800_utc_and_include_fridays():
    provider = mock()
    chain_ = provider.get_option_chain("BTCUSDT")
    expiry_times = {datetime.fromtimestamp(c.expiry_ts_ms / 1000, timezone.utc).hour for c in chain_.contracts}
    assert expiry_times == {8}
    assert len([e for e in chain_.expiries if e.weekday() == 4]) >= 12
    assert all(c.dte > 0 for c in chain_.contracts)


def test_mock_quotes_are_sane():
    for c in mock().get_option_chain("BTCUSDT").contracts:
        assert c.ask > 0
        assert c.bid == 0 or c.bid <= c.mark <= c.ask
        assert c.ask % c.tick_size == pytest.approx(0, abs=1e-6) or c.ask % c.tick_size == pytest.approx(c.tick_size)


def test_mock_is_deterministic():
    a = mock().get_option_chain("BTCUSDT")
    b = mock().get_option_chain("BTCUSDT")
    assert a == b


def test_mock_atm_cheaper_spread_and_more_liquid_than_wings():
    contracts = [c for c in mock().get_option_chain("BTCUSDT").contracts if c.option_type == "call" and 25 < c.dte < 35]
    atm = min(contracts, key=lambda c: abs(c.strike - 83400))
    wing = max(contracts, key=lambda c: c.strike)
    spread = lambda c: (c.ask - c.bid) / ((c.ask + c.bid) / 2) if c.bid else float("inf")
    assert spread(atm) < spread(wing)
    assert atm.open_interest > wing.open_interest


def test_mock_get_option_contract_reprices_at_current_spot():
    provider = mock()
    symbol = provider.get_option_chain("BTCUSDT").contracts[500].contract_symbol
    before = provider.get_option_contract(symbol)
    provider.market.price = 90000.0
    after = provider.get_option_contract(symbol)
    assert after.contract_symbol == symbol
    assert after.underlying_price == 90000.0
    assert after.mark != before.mark


def test_mock_rejects_unknown_and_expired():
    with pytest.raises(OptionChainError):
        mock().get_option_chain("SOLUSDT")
    with pytest.raises(OptionChainError):
        mock().get_option_contract("garbage")
    with pytest.raises(OptionChainError):
        mock().get_option_contract("BTC-1JAN20-50000-C")


#
# Pre-filter: one test per rejection reason
#

NOW = T0


def only_reason(c, thesis=None, pb=None, underlying="BTCUSDT"):
    candidates, summary = prefilter(chain([c], underlying), thesis or bull_thesis(), pb or playbook(), NOW)
    return candidates, summary


def test_valid_contract_accepted_with_deterministic_numbers():
    candidates, summary = only_reason(contract())
    assert summary == {}
    c = candidates[0]
    assert c.candidate_id == "C1"
    assert c.strategy == "LONG_CALL"
    assert c.limit_price == 2230.0                       # mid, on tick, <= ask
    assert c.quantity == 2                               # MAX_CONTRACTS cap
    assert c.estimated_max_loss == pytest.approx(2230.0 * 0.01 * 2)
    assert c.breakeven == 85000 + 2230
    assert c.intrinsic_at_target == 1000


@pytest.mark.parametrize("overrides,reason", [
    ({"option_type": "put", "delta": -0.45}, "wrong_option_type"),
    ({"dte": 3.0}, "dte_out_of_range"),
    ({"dte": 60.0}, "dte_out_of_range"),
    ({"delta": 0.20}, "delta_out_of_range"),
    ({"delta": 0.80}, "delta_out_of_range"),
    ({"delta": None}, "delta_out_of_range"),
    ({"bid": 0.0}, "no_valid_quote"),
    ({"bid": None}, "no_valid_quote"),
    ({"bid": 2300.0, "ask": 2260.0}, "no_valid_quote"),
    ({"as_of_ms": NOW - 120_000}, "stale_quote"),
    ({"bid": 1900.0, "ask": 2260.0}, "spread_too_wide"),
    ({"open_interest": 50}, "insufficient_liquidity"),
    ({"volume_24h": 5}, "insufficient_liquidity"),
    ({"strike": 86000.0}, "strike_beyond_target"),
    ({"strike": 87000.0}, "strike_beyond_target"),
])
def test_each_rejection_reason(overrides, reason):
    candidates, summary = only_reason(contract(**overrides))
    assert candidates == []
    assert summary == {reason: 1}


def test_rejected_due_to_max_loss_unaffordable():
    # One contract costs 2230 * 1.0 = $2,230 > $500 max premium.
    candidates, summary = only_reason(contract(contract_multiplier=1.0))
    assert candidates == []
    assert summary == {"unaffordable": 1}


def test_quantity_limited_by_premium_budget(monkeypatch):
    monkeypatch.setenv("MAX_CONTRACTS", "10")
    monkeypatch.setenv("MAX_PREMIUM_PER_TRADE", "100")
    candidates, _ = only_reason(contract())
    assert candidates[0].quantity == 4                  # floor(100 / 22.30)
    assert candidates[0].estimated_max_loss <= 100


def test_fees_included_in_max_loss(monkeypatch):
    monkeypatch.setenv("OPTION_FEE_PER_CONTRACT", "0.5")
    candidates, _ = only_reason(contract())
    assert candidates[0].estimated_max_loss == pytest.approx((22.30 + 0.5) * 2)


def test_limit_never_above_ask():
    c = contract(bid=2200.0, ask=2205.0, tick_size=5.0)   # mid 2202.5 rounds up to 2205
    limit, qty, _ = size_order(c, playbook())
    assert limit == 2205.0
    assert limit <= c.ask


def test_bearish_thesis_selects_puts_only():
    put = contract(contract_symbol="ETH-9OCT26-2700-P", underlying="ETHUSDT", underlying_index="ETH",
                   option_type="put", strike=2700.0, delta=-0.5, bid=150.0, ask=155.0, mark=152.5,
                   contract_multiplier=0.1, tick_size=0.5)
    call = put.model_copy(update={"option_type": "call", "delta": 0.5, "contract_symbol": "ETH-9OCT26-2700-C"})
    candidates, summary = prefilter(chain([put, call], "ETHUSDT"), bear_thesis(), playbook(), NOW)
    assert [c.strategy for c in candidates] == ["LONG_PUT"]
    assert summary == {"wrong_option_type": 1}


def test_neutral_thesis_produces_nothing():
    candidates, summary = prefilter(chain([contract()]), bull_thesis(direction="neutral"), playbook(), NOW)
    assert candidates == []
    assert summary == {"no_allowed_strategy_for_direction": 1}


def test_disallowed_underlying():
    candidates, summary = prefilter(chain([contract()]), bull_thesis(), playbook(allowed_underlyings=["ETHUSDT"]), NOW)
    assert candidates == []
    assert summary == {"underlying_not_allowed": 1}


def test_ranking_and_cutoff():
    contracts = [
        contract(contract_symbol=f"BTC-9OCT26-{k}-C", strike=float(k), delta=0.4 + i * 0.02)
        for i, k in enumerate(range(80000, 86000, 1000))
    ]
    candidates, summary = prefilter(chain(contracts), bull_thesis(), playbook(max_llm_candidates=3), NOW)
    assert [c.candidate_id for c in candidates] == ["C1", "C2", "C3"]
    assert [c.deterministic_rank for c in candidates] == [1, 2, 3]
    rr = [c.reward_to_risk_at_target for c in candidates]
    assert rr == sorted(rr, reverse=True)
    assert summary["ranked_below_cutoff"] == 3


def test_real_mock_chain_passes_through_filter():
    btc = mock().get_option_chain("BTCUSDT")
    candidates, summary = prefilter(btc, bull_thesis(target=85200.0), playbook(), T0)
    assert 1 <= len(candidates) <= 8
    for c in candidates:
        k = c.contract
        assert k.option_type == "call"
        assert 7 <= k.dte <= 45
        assert 0.35 <= abs(k.delta) <= 0.65
        assert k.strike < 85200
        assert c.estimated_max_loss <= 500
        assert c.quantity <= 2
