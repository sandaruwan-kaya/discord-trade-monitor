"""PaperBroker (offline). Quotes come from a controllable fake provider."""
import sqlite3
from itertools import count

import pytest

import config
import database
from broker.base import BrokerBase, BrokerConfigError
from broker.factory import create_broker
from broker.paper_broker import PaperBroker
from schemas import OrderRequest
from tools.options_tools import MockDeribitChainProvider, OptionChainError

from helpers import T0
from options_helpers import contract


SYMBOL = "BTC-9OCT26-85000-C"


class QuoteBook:
    """Fake OptionChainProvider: set bid/ask/mark per symbol, or make it fail."""

    def __init__(self):
        self.quotes = {}
        self.error = None
        self.calls = 0

    def set(self, symbol=SYMBOL, bid=2200.0, ask=2260.0, mark=None):
        self.quotes[symbol] = contract(contract_symbol=symbol, bid=bid, ask=ask,
                                       mark=mark if mark is not None else (bid + ask) / 2)

    def get_option_contract(self, symbol):
        self.calls += 1
        if self.error:
            raise self.error
        if symbol not in self.quotes:
            raise OptionChainError(f"unknown contract {symbol}")
        return self.quotes[symbol]

    def get_option_chain(self, underlying, **kw):
        raise NotImplementedError


def ids():
    n = count(1)
    return lambda prefix: f"{prefix}-{next(n):04d}"


@pytest.fixture
def book():
    b = QuoteBook()
    b.set()
    return b


@pytest.fixture
def broker(book):
    return PaperBroker(book, now_ms=lambda: T0, new_id=ids(), log=lambda *a: None)


def buy(client_id="signal:1:entry:open", limit=2260.0, qty=2, symbol=SYMBOL):
    return OrderRequest(client_order_id=client_id, signal_id=1, proposal_key="signal:1:entry",
                        contract_symbol=symbol, side="BUY_TO_OPEN", quantity=qty, limit_price=limit)


def open_position(broker, qty=2):
    result = broker.submit_order(buy(qty=qty))
    assert result.order.status == "FILLED"
    return result.position


def count_rows(table):
    conn = sqlite3.connect(config.db_path())
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


# 1
def test_valid_buy_to_open_fills(broker):
    r = broker.submit_order(buy(limit=2265.0))

    assert r.ok
    assert r.order.status == "FILLED"
    assert r.order.fill_price == 2260.0                     # min(ask, limit) = ask
    assert r.fill.price == 2260.0 and r.fill.quantity == 2
    p = r.position
    assert p.status == "OPEN"
    assert p.quantity == 2 and p.original_quantity == 2
    assert p.entry_price == 2260.0
    assert p.contract_multiplier == 0.01
    assert p.signal_id == 1 and p.proposal_key == "signal:1:entry"
    assert r.order.position_id == p.position_id


# 2
def test_buy_limit_below_ask_stays_pending(broker):
    r = broker.submit_order(buy(limit=2250.0))

    assert r.ok
    assert r.order.status == "PENDING"
    assert r.order.last_quote_ask == 2260.0
    assert r.fill is None
    assert broker.list_positions() == []


def test_pending_buy_fills_when_market_comes_to_it(broker, book):
    order = broker.submit_order(buy(limit=2250.0)).order
    book.set(bid=2190.0, ask=2245.0)

    [r] = broker.process_pending_orders()

    assert r.order.order_id == order.order_id
    assert r.order.status == "FILLED"
    assert r.order.fill_price == 2245.0


# 3
def test_cancel_pending_order(broker):
    order = broker.submit_order(buy(limit=2250.0)).order

    r = broker.cancel_order(order.order_id)

    assert r.ok
    assert r.order.status == "CANCELLED"
    assert r.order.cancelled_at is not None
    assert broker.process_pending_orders() == []            # never fills later


# 4
def test_filled_order_cannot_be_cancelled(broker):
    order = broker.submit_order(buy()).order

    r = broker.cancel_order(order.order_id)

    assert not r.ok
    assert "cannot cancel a FILLED order" in r.message
    assert broker.get_order(order.order_id).status == "FILLED"


def test_cancel_unknown_order(broker):
    assert not broker.cancel_order("PO-nope").ok


# 5
def test_duplicate_submit_is_idempotent(broker):
    first = broker.submit_order(buy())
    second = broker.submit_order(buy())
    assert first.order.status == "FILLED"                   # replay after fill (position_id now set)

    assert second.idempotent_replay
    assert second.ok
    assert second.order.order_id == first.order.order_id
    assert count_rows("paper_orders") == 1
    assert count_rows("paper_fills") == 1
    assert count_rows("paper_positions") == 1


def test_duplicate_submit_of_pending_order_is_idempotent(broker, book):
    first = broker.submit_order(buy(limit=2250.0))
    second = broker.submit_order(buy(limit=2250.0))
    assert second.idempotent_replay and second.order.status == "PENDING"
    assert second.order.order_id == first.order.order_id
    assert count_rows("paper_orders") == 1


def test_duplicate_close_is_idempotent(broker, book):
    position = open_position(broker)
    book.set(bid=2400.0, ask=2460.0)
    broker.close_position(position.position_id, 2400.0, "close")
    replay = broker.close_position(position.position_id, 2400.0, "close")
    assert replay.idempotent_replay
    assert count_rows("paper_fills") == 2                   # one entry fill + one close fill


def test_reused_client_id_with_different_params_is_refused(broker):
    broker.submit_order(buy(qty=2))
    r = broker.submit_order(buy(qty=1))

    assert not r.ok
    assert "different parameters" in r.message
    assert count_rows("paper_orders") == 1


# 6
def test_restart_preserves_pending_orders(book):
    before = PaperBroker(book, now_ms=lambda: T0, new_id=ids(), log=lambda *a: None)
    order = before.submit_order(buy(limit=2250.0)).order
    del before

    after = PaperBroker(book, now_ms=lambda: T0 + 60_000, log=lambda *a: None)   # "restart"
    assert [o.order_id for o in after.list_orders("PENDING")] == [order.order_id]

    book.set(bid=2190.0, ask=2250.0)
    [r] = after.process_pending_orders()
    assert r.order.status == "FILLED"


# 7
def test_restart_preserves_open_positions(book):
    before = PaperBroker(book, now_ms=lambda: T0, new_id=ids(), log=lambda *a: None)
    position = open_position(before)
    del before

    after = PaperBroker(book, now_ms=lambda: T0 + 60_000, log=lambda *a: None)
    [p] = after.list_positions("OPEN")
    assert p.position_id == position.position_id
    assert p.quantity == 2 and p.entry_price == 2260.0


# 8
def test_mark_to_market_updates_unrealized_pnl(broker, book):
    position = open_position(broker)
    book.set(bid=2480.0, ask=2520.0, mark=2500.0)

    p = broker.get_position(position.position_id)

    assert p.current_mark == 2500.0
    assert p.unrealized_pnl == pytest.approx((2500.0 - 2260.0) * 0.01 * 2)   # +$4.80
    [q] = broker.mark_to_market()
    assert q.unrealized_pnl == pytest.approx(4.80)


# 9
def test_sell_to_close_closes_position(broker, book):
    position = open_position(broker)
    book.set(bid=2400.0, ask=2460.0)

    r = broker.close_position(position.position_id, limit_price=2390.0, client_order_id="signal:1:entry:close")

    assert r.ok
    assert r.order.side == "SELL_TO_CLOSE"
    assert r.order.fill_price == 2400.0                     # max(bid, limit) = bid
    p = r.position
    assert p.status == "CLOSED"
    assert p.quantity == 0
    assert p.closed_at is not None
    assert p.unrealized_pnl == 0
    assert p.realized_pnl == pytest.approx((2400.0 - 2260.0) * 0.01 * 2)     # +$2.80


def test_partial_close_then_full_close(broker, book):
    position = open_position(broker)
    book.set(bid=2400.0, ask=2460.0)

    r1 = broker.close_position(position.position_id, 2400.0, "close-1", quantity=1)
    assert r1.position.status == "OPEN" and r1.position.quantity == 1

    r2 = broker.close_position(position.position_id, 2400.0, "close-2")
    assert r2.position.status == "CLOSED"
    assert r2.position.realized_pnl == pytest.approx(2.80)


def test_sell_limit_above_bid_stays_pending(broker, book):
    position = open_position(broker)

    r = broker.close_position(position.position_id, limit_price=2300.0, client_order_id="close")

    assert r.order.status == "PENDING"
    assert broker.get_position(position.position_id, refresh_mark=False).status == "OPEN"


# 10
def test_close_more_than_open_is_rejected(broker):
    position = open_position(broker, qty=2)

    r = broker.close_position(position.position_id, 2200.0, "close", quantity=3)

    assert not r.ok
    assert r.order.status == "REJECTED"
    assert "exceeds available 2" in r.order.rejection_reason
    assert broker.get_position(position.position_id, refresh_mark=False).quantity == 2


# 11
def test_naked_sell_to_close_is_rejected(broker):
    r = broker.submit_order(OrderRequest(
        client_order_id="naked", contract_symbol=SYMBOL, side="SELL_TO_CLOSE",
        quantity=1, limit_price=2200.0, position_id="PP-does-not-exist",
    ))

    assert not r.ok
    assert r.order.status == "REJECTED"
    assert "naked sell rejected" in r.order.rejection_reason
    assert broker.list_positions() == []
    assert count_rows("paper_fills") == 0


def test_sell_without_position_cannot_even_be_expressed():
    with pytest.raises(ValueError):
        OrderRequest(client_order_id="x", contract_symbol=SYMBOL, side="SELL_TO_CLOSE",
                     quantity=1, limit_price=1.0)
    with pytest.raises(ValueError):
        OrderRequest(client_order_id="x", contract_symbol=SYMBOL, side="SELL_TO_OPEN",
                     quantity=1, limit_price=1.0)


def test_close_on_closed_position_rejected(broker, book):
    position = open_position(broker)
    book.set(bid=2400.0, ask=2460.0)
    broker.close_position(position.position_id, 2400.0, "close-1")

    r = broker.close_position(position.position_id, 2400.0, "close-2")

    assert not r.ok
    assert "CLOSED" in r.order.rejection_reason


def test_close_wrong_contract_rejected(broker):
    position = open_position(broker)
    r = broker.submit_order(OrderRequest(
        client_order_id="c", contract_symbol="BTC-9OCT26-84000-C", side="SELL_TO_CLOSE",
        quantity=1, limit_price=1.0, position_id=position.position_id,
    ))
    assert not r.ok and "does not match" in r.order.rejection_reason


# 12
def test_no_short_positions_from_stacked_pending_closes(broker):
    position = open_position(broker, qty=2)

    # Two resting closes of 1 each reserve the full position ...
    assert broker.close_position(position.position_id, 9999.0, "c1", quantity=1).order.status == "PENDING"
    assert broker.close_position(position.position_id, 9999.0, "c2", quantity=1).order.status == "PENDING"
    # ... so a third close is rejected even though it would fit the raw open quantity.
    r = broker.close_position(position.position_id, 9999.0, "c3", quantity=1)
    assert r.order.status == "REJECTED"
    assert "pending closes 2" in r.order.rejection_reason


def test_db_forbids_negative_position_quantity(broker):
    position = open_position(broker)
    conn = sqlite3.connect(config.db_path())
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE paper_positions SET quantity = -1 WHERE position_id = ?", (position.position_id,))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO paper_orders (order_id, client_order_id, contract_symbol, side, quantity, limit_price,"
            " status, submitted_at, updated_at) VALUES ('x', 'x', 's', 'SELL_TO_OPEN', 1, 1, 'PENDING', 't', 't')"
        )


# 13
def test_duplicate_fills_cannot_create_duplicate_positions(book):
    # Two broker instances (e.g. two loops) both see the same PENDING order.
    a = PaperBroker(book, now_ms=lambda: T0, new_id=ids(), log=lambda *a: None)
    b = PaperBroker(book, now_ms=lambda: T0, log=lambda *a: None)
    order = a.submit_order(buy(limit=2250.0)).order
    book.set(bid=2190.0, ask=2245.0)

    ra = a._try_fill(order.order_id)
    rb = b._try_fill(order.order_id)
    b._apply_fill(order.order_id, book.quotes[SYMBOL], 2245.0)   # forced second fill attempt

    assert ra.order.status == "FILLED"
    assert "not pending" in rb.message
    assert count_rows("paper_fills") == 1
    assert count_rows("paper_positions") == 1


def test_db_forbids_second_position_for_same_entry_order(broker):
    position = open_position(broker)
    conn = sqlite3.connect(config.db_path())
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO paper_positions (position_id, entry_order_id, contract_symbol, contract_multiplier,"
            " quantity, original_quantity, entry_price, status, opened_at, updated_at)"
            " VALUES ('PP-dup', ?, 's', 0.01, 1, 1, 1, 'OPEN', 't', 't')",
            (position.entry_order_id,),
        )


# 14
@pytest.mark.parametrize("mode", ["live", "LIVE", "real", "testnet", "production"])
def test_factory_and_broker_refuse_non_paper_mode(monkeypatch, book, mode):
    monkeypatch.setenv("TRADING_MODE", mode)
    with pytest.raises(BrokerConfigError):
        create_broker(book)
    with pytest.raises(BrokerConfigError):
        PaperBroker(book)


def test_factory_returns_paper_broker_only(book):
    broker = create_broker(book)
    assert type(broker) is PaperBroker
    assert isinstance(broker, BrokerBase)


def test_repository_contains_no_live_broker():
    import pathlib
    import broker as pkg

    files = sorted(p.name for p in pathlib.Path(pkg.__file__).parent.glob("*.py"))
    assert files == ["__init__.py", "base.py", "factory.py", "paper_broker.py"]
    assert [c.__name__ for c in BrokerBase.__subclasses__()] == ["PaperBroker"]


# 15
def test_provider_failure_on_submit_keeps_order_pending_and_state_clean(broker, book):
    book.error = ConnectionError("chain provider down")

    r = broker.submit_order(buy())

    assert r.ok
    assert r.order.status == "PENDING"
    assert "quote unavailable" in r.message
    assert count_rows("paper_fills") == 0 and count_rows("paper_positions") == 0

    book.error = None
    [r2] = broker.process_pending_orders()
    assert r2.order.status == "FILLED"


def test_provider_failure_on_mark_keeps_last_values(broker, book):
    position = open_position(broker)
    before = broker.get_position(position.position_id, refresh_mark=False)
    book.error = TimeoutError("slow")

    p = broker.get_position(position.position_id)

    assert p == before


def test_unknown_contract_buy_is_rejected(broker):
    r = broker.submit_order(buy(symbol="BTC-9OCT26-99000-C"))
    assert not r.ok
    assert r.order.status == "REJECTED"
    assert "contract unavailable" in r.order.rejection_reason


# 16
def test_transaction_rollback_leaves_db_consistent(broker, book, monkeypatch):
    order = broker.submit_order(buy(limit=2250.0)).order
    book.set(bid=2190.0, ask=2245.0)

    def explode(*a, **k):
        raise RuntimeError("disk full while inserting position")

    # Scoped patch: undoing it must not undo the conftest temp-DB override.
    with monkeypatch.context() as m:
        m.setattr(broker, "_insert_position", explode)
        with pytest.raises(RuntimeError):
            broker._try_fill(order.order_id)

    # Fill insert + order update were rolled back with the failed position insert.
    assert broker.get_order(order.order_id).status == "PENDING"
    assert count_rows("paper_fills") == 0
    assert count_rows("paper_positions") == 0

    [r] = broker.process_pending_orders()
    assert r.order.status == "FILLED"
    assert count_rows("paper_fills") == 1 and count_rows("paper_positions") == 1


def test_works_with_mock_deribit_chain(broker):
    class Spot:
        def get_current_price(self, symbol):
            return 83400.0

    provider = MockDeribitChainProvider(Spot(), now_ms=lambda: T0)
    paper = PaperBroker(provider, now_ms=lambda: T0, log=lambda *a: None)
    quote = next(c for c in paper.get_option_chain("BTCUSDT", option_type="call").contracts
                 if 20 < c.dte < 30 and c.strike == 84000.0)

    r = paper.submit_order(buy(symbol=quote.contract_symbol, limit=quote.ask, qty=1))

    assert r.order.status == "FILLED"
    assert r.position.contract_multiplier == 0.01
    assert r.position.entry_price == quote.ask


# Agent 2 isolation
def test_no_path_from_agents_or_app_to_broker():
    import pathlib
    root = pathlib.Path(config.PROJECT_DIR)
    for rel in ["agents/options_agent.py", "agents/signal_agent.py", "agents/risk_agent.py",
                "agents/trade_manager.py", "signal_service.py", "trading_pipeline.py",
                "tools/options_tools.py", "tools/market_tools.py"]:
        text = (root / rel).read_text()
        assert "import broker" not in text and "from broker" not in text, rel
        assert "submit_order" not in text and "close_position" not in text, rel

    # app.py is the composition root: it may only CONSTRUCT the paper broker via the
    # paper-only factory (handed to ExecutionService); it never calls it.
    app = (root / "app.py").read_text()
    broker_imports = [l.strip() for l in app.splitlines() if "broker" in l and "import" in l]
    assert broker_imports == ["from broker.factory import create_broker"]
    for call in (".submit_order(", ".close_position(", ".cancel_order("):
        assert call not in app
