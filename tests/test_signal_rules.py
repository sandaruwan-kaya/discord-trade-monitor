import pytest

import signal_rules


@pytest.mark.parametrize("timeframe,bucket", [
    (None, "unknown"),
    ("", "unknown"),
    ("5m", "scalp"),
    ("15 min", "scalp"),
    ("scalp", "scalp"),
    ("30m", "intraday"),
    ("1h", "intraday"),
    ("intraday", "intraday"),
    ("4h", "swing"),
    ("daily", "swing"),
    ("swing", "swing"),
    ("weekly", "position"),
    ("1W", "position"),
])
def test_classify_timeframe(timeframe, bucket):
    assert signal_rules.classify_timeframe(timeframe) == bucket


def test_expiry_is_configurable(monkeypatch):
    monkeypatch.setenv("SIGNAL_EXPIRY_HOURS_INTRADAY", "2")
    hour = 3_600_000

    assert not signal_rules.is_expired(0, "1h", 2 * hour)
    assert signal_rules.is_expired(0, "1h", 2 * hour + 1)


def test_bullish_targets_below_levels_rejected():
    valid, rejected = signal_rules.check_targets(
        "bullish", [99.0, 100.2, 102.0, 97.0],
        price_at_signal=100.0, confirmation_above=100.5, invalidation_below=98.0
    )
    assert valid == [102.0]
    assert [r["target"] for r in rejected] == [99.0, 100.2, 97.0]


def test_bearish_targets_above_levels_rejected():
    valid, rejected = signal_rules.check_targets(
        "conditional_bearish", [97.0, 101.5],
        price_at_signal=100.0, confirmation_below=99.5, invalidation_above=101.0
    )
    assert valid == [97.0]
    assert rejected[0]["target"] == 101.5


def test_targets_checked_structurally_when_price_unknown():
    valid, rejected = signal_rules.check_targets(
        "bullish", [99.0, 102.0], price_at_signal=None, confirmation_above=100.5
    )
    assert valid == [102.0]


def test_neutral_bias_rejects_nothing():
    valid, rejected = signal_rules.check_targets("neutral", [90.0, 110.0], price_at_signal=100.0)
    assert valid == [90.0, 110.0]
    assert rejected == []


def test_direction_from_bias_never_guesses():
    assert signal_rules.direction_from_bias("conditional_bullish") == "bullish"
    assert signal_rules.direction_from_bias("BEARISH") == "bearish"
    assert signal_rules.direction_from_bias(None) == "neutral"
    assert signal_rules.direction_from_bias("unclear") == "neutral"
