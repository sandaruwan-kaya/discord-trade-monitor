"""Evaluator keeps its original semantics and can resume from state."""
import pytest

from evaluator import evaluate_signal

from helpers import MINUTE_MS, T0, bearish_signal, bullish_signal, candle


def series(points):
    return [candle(T0 + i * MINUTE_MS, lo, hi) for i, (lo, hi) in enumerate(points)]


def test_developing_working_target():
    c = series([(99.9, 100.1), (100.3, 100.6), (101.0, 102.2)])
    assert evaluate_signal(bullish_signal(), c[:1])["status"] == "DEVELOPING"
    assert evaluate_signal(bullish_signal(), c[:2])["status"] == "WORKING"
    assert evaluate_signal(bullish_signal(), c)["status"] == "TARGET_HIT"


def test_target_then_invalidation_stays_target_hit():
    c = series([(100.3, 102.1), (97.0, 99.0)])
    assert evaluate_signal(bullish_signal(), c)["status"] == "TARGET_HIT"


def test_invalidation_before_target():
    c = series([(97.9, 100.0), (101.0, 102.5)])
    result = evaluate_signal(bullish_signal(), c)
    assert result["status"] == "INVALIDATED"
    assert result["invalidation_time"] == T0


def test_bearish_target():
    c = series([(99.4, 100.0), (96.9, 99.0)])
    assert evaluate_signal(bearish_signal(), c)["status"] == "TARGET_HIT"


def test_no_candles_without_state_raises():
    with pytest.raises(ValueError):
        evaluate_signal(bullish_signal(), [])


def test_targets_override():
    c = series([(99.9, 100.1)])
    # With the raw targets a bogus 99.0 target would fire immediately.
    raw = bullish_signal(targets=[99.0, 102.0])
    assert evaluate_signal(raw, c)["status"] == "TARGET_HIT"
    assert evaluate_signal(raw, c, targets=[102.0])["status"] == "DEVELOPING"


@pytest.mark.parametrize("split", range(1, 8))
def test_incremental_equals_full_walk(split):
    points = [(99.9, 100.1), (99.5, 100.2), (100.3, 100.6), (100.2, 100.9),
              (100.4, 101.1), (100.1, 100.8), (100.5, 101.5), (101.0, 102.3)]
    c = series(points)
    full = evaluate_signal(bullish_signal(), c)
    resumed = evaluate_signal(bullish_signal(), c[split:], evaluate_signal(bullish_signal(), c[:split]))
    assert resumed == full


def test_terminal_state_is_sticky():
    state = evaluate_signal(bullish_signal(), series([(97.0, 99.0)]))
    later = [candle(T0 + 5 * MINUTE_MS, 101.0, 103.0)]
    assert evaluate_signal(bullish_signal(), later, state)["status"] == "INVALIDATED"
