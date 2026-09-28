"""Signal monitor: incremental candles, >200-minute signals, expiry."""
import asyncio

import database
from evaluator import evaluate_signal
from signal_service import Services, run_monitor_cycle

from helpers import MINUTE_MS, T0, FakeMarket, bullish_signal


def target_at_400(minute):
    """Flat 100 -> confirms at minute 10 -> target 102 hit at minute 400."""
    if minute < 10:
        return (99.9, 100.1, 100.0)
    if minute < 400:
        return (100.4, 100.7, 100.6)
    return (101.0, 102.1, 102.0)


def flat(minute):
    return (99.9, 100.1, 100.0)


class Clock:
    def __init__(self, now_ms):
        self.now_ms = now_ms

    def __call__(self):
        return self.now_ms


def setup_signal(signal=None, message_id="m1"):
    database.save_signal(
        discord_message_id=message_id,
        signal=signal or bullish_signal(timeframe="swing"),
        signal_time_ms=T0 + 15_000,
        price_at_signal=100.0,
        raw_message="raw"
    )
    return database.get_signal_by_discord_message_id(message_id)["id"]


def make_services(market, clock, sent):
    return Services(
        parse_message=None,
        market=market,
        notify=sent.append,
        now_ms=clock,
        log=lambda *a, **k: None
    )


def tick(services):
    asyncio.run(run_monitor_cycle(services))


def test_old_200_candle_window_would_miss_the_target():
    """Documents the original bug this step fixes."""
    market = FakeMarket(target_at_400, now_ms=T0 + 450 * MINUTE_MS)
    first_200 = market.get_candles("BTCUSDT", "1m", 200, T0)

    result = evaluate_signal(bullish_signal(), first_200)

    assert result["status"] == "WORKING"          # frozen, never sees minute 400
    assert result["current_price"] == 100.6


def test_signal_older_than_200_minutes_processes_current_candles():
    signal_id = setup_signal()
    clock = Clock(T0 + 450 * MINUTE_MS)
    market = FakeMarket(target_at_400, now_ms=clock.now_ms)
    sent = []

    tick(make_services(market, clock, sent))

    row = database.get_signal(signal_id)
    assert row["status"] == "TARGET_HIT"
    assert row["active"] == 0
    assert "TARGET_HIT" in sent[0]
    assert "102.0" in sent[0]                     # current price is recent


def test_incremental_fetch_only_requests_new_candles():
    signal_id = setup_signal()
    clock = Clock(T0 + 300 * MINUTE_MS)
    market = FakeMarket(target_at_400, now_ms=clock.now_ms)
    sent = []
    services = make_services(market, clock, sent)

    # Tick 1 at +300 min: full catch-up from the signal minute.
    tick(services)
    row = database.get_signal(signal_id)
    assert row["status"] == "WORKING"
    assert market.calls[0]["start_ms"] == T0
    # Last CLOSED candle is minute 299 (minute 300 just opened).
    assert row["last_processed_candle_ms"] == T0 + 299 * MINUTE_MS
    assert len(sent) == 1

    # Tick 2 at +302 min: only candles after the cursor.
    market.calls.clear()
    clock.now_ms = market.now_ms = T0 + 302 * MINUTE_MS
    tick(services)
    assert [c["start_ms"] for c in market.calls] == [T0 + 300 * MINUTE_MS]
    assert database.get_signal(signal_id)["last_processed_candle_ms"] == T0 + 301 * MINUTE_MS
    assert len(sent) == 1                         # no status change, no Telegram

    # Tick 3 at +401 min: target candle (minute 400) arrives.
    market.calls.clear()
    clock.now_ms = market.now_ms = T0 + 401 * MINUTE_MS
    tick(services)
    assert market.calls[0]["start_ms"] == T0 + 302 * MINUTE_MS
    row = database.get_signal(signal_id)
    assert row["status"] == "TARGET_HIT"
    assert row["active"] == 0
    assert len(sent) == 2

    events = database.get_signal_events(signal_id)
    assert [(e["old_status"], e["new_status"]) for e in events if e["event_type"] == "STATUS_CHANGE"] == [
        ("DEVELOPING", "WORKING"),
        ("WORKING", "TARGET_HIT"),
    ]
    assert row["first_working_at"] is not None


def test_catch_up_pages_through_long_gaps(monkeypatch):
    monkeypatch.setenv("MONITOR_CANDLE_PAGE_SIZE", "100")
    signal_id = setup_signal()
    clock = Clock(T0 + 450 * MINUTE_MS)
    market = FakeMarket(target_at_400, now_ms=clock.now_ms)

    tick(make_services(market, clock, []))

    starts = [c["start_ms"] for c in market.calls]
    assert starts[:5] == [T0 + i * 100 * MINUTE_MS for i in range(5)]
    assert database.get_signal(signal_id)["status"] == "TARGET_HIT"


def test_in_progress_candle_counts_for_status_but_not_cursor():
    signal_id = setup_signal()
    # Minute 10 (the confirming candle) is still open.
    clock = Clock(T0 + 10 * MINUTE_MS + 30_000)
    market = FakeMarket(target_at_400, now_ms=clock.now_ms)

    tick(make_services(market, clock, []))

    row = database.get_signal(signal_id)
    assert row["status"] == "WORKING"
    assert row["last_processed_candle_ms"] == T0 + 9 * MINUTE_MS


def test_signal_expires_by_timeframe(monkeypatch):
    monkeypatch.setenv("SIGNAL_EXPIRY_HOURS_SCALP", "6")
    signal_id = setup_signal(bullish_signal(timeframe="15m"))
    clock = Clock(T0 + 7 * 60 * MINUTE_MS)
    market = FakeMarket(flat, now_ms=clock.now_ms)
    sent = []

    tick(make_services(market, clock, sent))

    row = database.get_signal(signal_id)
    assert row["status"] == "EXPIRED"
    assert row["active"] == 0
    assert "STATUS:\nEXPIRED" in sent[0]


def test_not_expired_within_window(monkeypatch):
    monkeypatch.setenv("SIGNAL_EXPIRY_HOURS_SCALP", "6")
    signal_id = setup_signal(bullish_signal(timeframe="15m"))
    clock = Clock(T0 + 5 * 60 * MINUTE_MS)

    tick(make_services(FakeMarket(flat, now_ms=clock.now_ms), clock, []))

    assert database.get_signal(signal_id)["status"] == "DEVELOPING"


def test_telegram_failure_retries_status_change_next_tick():
    signal_id = setup_signal()
    clock = Clock(T0 + 450 * MINUTE_MS)
    market = FakeMarket(target_at_400, now_ms=clock.now_ms)

    def broken(_):
        raise ConnectionError("telegram down")

    services = make_services(market, clock, [])
    services.notify = broken
    tick(services)
    assert database.get_signal(signal_id)["status"] == "DEVELOPING"

    sent = []
    services.notify = sent.append
    tick(services)
    assert database.get_signal(signal_id)["status"] == "TARGET_HIT"
    assert len(sent) == 1


def test_inconsistent_target_is_ignored_by_monitor():
    # Bullish with a "target" below price: must not fire TARGET_HIT.
    signal_id = setup_signal(bullish_signal(targets=[99.0, 102.0]))
    clock = Clock(T0 + 20 * MINUTE_MS)
    market = FakeMarket(flat, now_ms=clock.now_ms)   # low 99.9 never hits 99, never hits 102

    def dips(minute):
        return (98.9 if minute == 5 else 99.9, 100.1, 100.0)

    market.path = dips   # touches 99.0 (the bogus target) at minute 5, stays above 98 invalidation
    tick(make_services(market, clock, []))

    assert database.get_signal(signal_id)["status"] == "DEVELOPING"


def test_market_error_is_logged_not_raised():
    signal_id = setup_signal()

    class Down:
        def get_candles(self, *a, **k):
            raise ConnectionError("binance down")

    logs = []
    services = Services(parse_message=None, market=Down(), notify=lambda _: None,
                        now_ms=lambda: T0 + 5 * MINUTE_MS, log=lambda *a: logs.append(a))
    asyncio.run(run_monitor_cycle(services))

    assert database.get_signal(signal_id)["status"] == "DEVELOPING"
    assert any("MONITOR ERROR" in str(entry) for entry in logs)
