"""Builders for option-related tests."""
from datetime import date

from schemas import OptionChain, OptionContract, ThesisSnapshot
from tools.options_tools import build_playbook

from helpers import T0


def contract(**overrides):
    data = dict(
        contract_symbol="BTC-9OCT26-85000-C",
        underlying="BTCUSDT",
        underlying_index="BTC",
        option_type="call",
        style="european",
        strike=85000.0,
        expiry=date(2026, 10, 9),
        expiry_ts_ms=T0 + 14 * 86_400_000,
        dte=14.0,
        bid=2200.0,
        ask=2260.0,
        mark=2230.0,
        iv=0.5,
        delta=0.45,
        open_interest=1500,
        volume_24h=150,
        contract_multiplier=0.01,
        tick_size=5.0,
        underlying_price=83400.0,
        as_of_ms=T0,
        source="test",
    )
    data.update(overrides)
    return OptionContract(**data)


def chain(contracts, underlying="BTCUSDT"):
    return OptionChain(
        underlying=underlying,
        underlying_price=83400.0,
        as_of_ms=T0,
        source="test",
        expiries=sorted({c.expiry for c in contracts}),
        contracts=contracts,
    )


def bull_thesis(**overrides):
    data = dict(
        underlying="BTCUSDT", direction="bullish", entry_level=84000.0,
        target=86000.0, invalidation=82600.0, time_horizon="intraday",
        agent1_state="DEVELOPING",
    )
    data.update(overrides)
    return ThesisSnapshot(**data)


def bear_thesis(**overrides):
    data = dict(
        underlying="ETHUSDT", direction="bearish", entry_level=2620.0,
        target=2540.0, invalidation=2720.0, time_horizon="swing",
        agent1_state="DEVELOPING",
    )
    data.update(overrides)
    return ThesisSnapshot(**data)


def playbook(**overrides):
    return build_playbook().model_copy(update=overrides)
