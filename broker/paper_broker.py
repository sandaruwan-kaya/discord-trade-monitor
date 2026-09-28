"""
PaperBroker — simulated broker, persisted in SQLite, restart-safe.

Fill model (V1, deterministic, no slippage, no partial fills):

    BUY_TO_OPEN:   fresh quote; if ask <= limit -> fill at min(ask, limit)
    SELL_TO_CLOSE: fresh quote; if bid >= limit -> fill at max(bid, limit)
    otherwise the order stays PENDING (process_pending_orders re-checks)

Safety:
    - refuses to initialize unless TRADING_MODE=paper
    - only BUY_TO_OPEN / SELL_TO_CLOSE exist (schema + DB CHECK)
    - SELL_TO_CLOSE must reference an OPEN position and may not exceed
      its open quantity minus quantity already reserved by other
      PENDING closes -> no naked sells, no short positions
    - fill + order + position updates happen in ONE transaction that
      re-checks the order is still PENDING -> no duplicate fills/positions
    - network (quote) calls happen outside transactions; a provider
      failure never writes state
"""
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import config
import database
from broker.base import BrokerBase, BrokerConfigError
from schemas import (
    BrokerOrderResult,
    OrderRequest,
    PaperFill,
    PaperOrder,
    PaperPosition,
)
from tools.options_tools import OptionChainError


_REQUEST_FIELDS = (
    "signal_id", "proposal_key", "position_id",
    "contract_symbol", "side", "quantity", "limit_price",
)


class PaperBroker(BrokerBase):
    name = "paper"

    def __init__(self, chain_provider, now_ms=None, new_id=None, log=print):
        if config.trading_mode() != "paper":
            raise BrokerConfigError(
                f"PaperBroker requires TRADING_MODE=paper (got {config.trading_mode()!r})"
            )

        self.chain_provider = chain_provider
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))
        self._new_id = new_id or (lambda prefix: f"{prefix}-{uuid.uuid4().hex[:16]}")
        self._log = log

    #
    # Infrastructure
    #

    def _now_iso(self):
        return datetime.fromtimestamp(self._now_ms() / 1000, timezone.utc).isoformat(timespec="seconds")

    def _connect(self):
        conn = database.get_connection()
        conn.isolation_level = None           # explicit BEGIN/COMMIT below
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _txn(self):
        """BEGIN IMMEDIATE ... COMMIT; any exception rolls everything back."""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def _read(self, sql, params=()):
        conn = self._connect()
        try:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()

    @staticmethod
    def _order(row):
        return PaperOrder(**row) if row else None

    @staticmethod
    def _position(row):
        return PaperPosition(**row) if row else None

    #
    # Market data
    #

    def get_option_chain(self, underlying, *, option_type=None, min_dte=None, max_dte=None):
        return self.chain_provider.get_option_chain(
            underlying, option_type=option_type, min_dte=min_dte, max_dte=max_dte
        )

    def get_option_contract(self, contract_symbol):
        return self.chain_provider.get_option_contract(contract_symbol)

    #
    # Orders
    #

    def submit_order(self, request: OrderRequest) -> BrokerOrderResult:
        now = self._now_iso()

        with self._txn() as conn:
            existing = conn.execute(
                "SELECT * FROM paper_orders WHERE client_order_id = ?",
                (request.client_order_id,)
            ).fetchone()

            if existing:
                existing = dict(existing)
                # A BUY's position_id is assigned by the broker at fill time,
                # so it is only part of the request identity for closes.
                fields = [f for f in _REQUEST_FIELDS
                          if f != "position_id" or request.side == "SELL_TO_CLOSE"]
                same = all(existing[f] == getattr(request, f) for f in fields)
                if not same:
                    return BrokerOrderResult(
                        ok=False,
                        order=self._order(existing),
                        message="client_order_id already used with different parameters; nothing submitted",
                    )
                return BrokerOrderResult(
                    ok=existing["status"] != "REJECTED",
                    order=self._order(existing),
                    idempotent_replay=True,
                    message=f"idempotent replay: order already {existing['status']}",
                )

            rejection = None
            if request.side == "SELL_TO_CLOSE":
                rejection = self._close_rejection(conn, request)

            order_id = self._new_id("PO")
            conn.execute(
                """
                INSERT INTO paper_orders (
                    order_id, client_order_id, signal_id, proposal_key, position_id,
                    contract_symbol, side, quantity, limit_price, status,
                    rejection_reason, submitted_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order_id, request.client_order_id, request.signal_id,
                    request.proposal_key, request.position_id, request.contract_symbol,
                    request.side, request.quantity, request.limit_price,
                    "REJECTED" if rejection else "PENDING", rejection, now, now,
                )
            )

        if rejection:
            return BrokerOrderResult(ok=False, order=self.get_order(order_id), message=rejection)

        return self._try_fill(order_id)

    def _close_rejection(self, conn, request):
        """Reason a SELL_TO_CLOSE must be rejected, or None."""
        position = conn.execute(
            "SELECT * FROM paper_positions WHERE position_id = ?",
            (request.position_id,)
        ).fetchone()

        if position is None:
            return "naked sell rejected: no such position"

        if position["status"] != "OPEN":
            return f"position {request.position_id} is {position['status']}"

        if position["contract_symbol"] != request.contract_symbol:
            return "contract does not match the position being closed"

        reserved = conn.execute(
            """
            SELECT COALESCE(SUM(quantity), 0) FROM paper_orders
            WHERE position_id = ? AND side = 'SELL_TO_CLOSE' AND status = 'PENDING'
            """,
            (request.position_id,)
        ).fetchone()[0]

        available = position["quantity"] - reserved

        if request.quantity > available:
            return (
                f"close quantity {request.quantity} exceeds available {available} "
                f"(open {position['quantity']}, pending closes {reserved}); would create a short position"
            )

        return None

    def get_order(self, order_id):
        rows = self._read("SELECT * FROM paper_orders WHERE order_id = ?", (order_id,))
        return self._order(rows[0] if rows else None)

    def get_order_by_client_id(self, client_order_id):
        rows = self._read("SELECT * FROM paper_orders WHERE client_order_id = ?", (client_order_id,))
        return self._order(rows[0] if rows else None)

    def list_orders(self, status=None):
        if status:
            rows = self._read("SELECT * FROM paper_orders WHERE status = ? ORDER BY submitted_at, order_id", (status,))
        else:
            rows = self._read("SELECT * FROM paper_orders ORDER BY submitted_at, order_id")
        return [self._order(r) for r in rows]

    def get_fill_for_order(self, order_id):
        rows = self._read("SELECT * FROM paper_fills WHERE order_id = ?", (order_id,))
        return PaperFill(**rows[0]) if rows else None

    def cancel_order(self, order_id):
        now = self._now_iso()

        with self._txn() as conn:
            row = conn.execute("SELECT * FROM paper_orders WHERE order_id = ?", (order_id,)).fetchone()

            if row is None:
                return BrokerOrderResult(ok=False, message=f"order {order_id} not found")

            if row["status"] != "PENDING":
                return BrokerOrderResult(
                    ok=False, order=self._order(dict(row)),
                    message=f"cannot cancel a {row['status']} order"
                )

            conn.execute(
                """
                UPDATE paper_orders
                SET status = 'CANCELLED', cancelled_at = ?, updated_at = ?
                WHERE order_id = ? AND status = 'PENDING'
                """,
                (now, now, order_id)
            )

        return BrokerOrderResult(ok=True, order=self.get_order(order_id), message="cancelled")

    def process_pending_orders(self):
        """Re-check every PENDING order against a fresh quote."""
        return [self._try_fill(o.order_id) for o in self.list_orders("PENDING")]

    #
    # Fill simulation
    #

    def _try_fill(self, order_id) -> BrokerOrderResult:
        order = self.get_order(order_id)

        if order is None or order.status != "PENDING":
            return BrokerOrderResult(ok=order is not None, order=order,
                                     message="order is not pending" if order else "order not found")

        # Network call OUTSIDE any transaction. Failures write nothing.
        try:
            quote = self.chain_provider.get_option_contract(order.contract_symbol)
        except OptionChainError as error:
            if order.side == "BUY_TO_OPEN":
                return self._reject_pending(order_id, f"contract unavailable: {error}")
            return BrokerOrderResult(ok=True, order=order,
                                     message=f"quote unavailable ({error}); order remains PENDING")
        except Exception as error:
            self._log(f"[PAPER] quote failed for {order.contract_symbol}: {type(error).__name__}")
            return BrokerOrderResult(ok=True, order=order,
                                     message=f"quote unavailable ({type(error).__name__}); order remains PENDING")

        price = None
        if order.side == "BUY_TO_OPEN":
            if quote.ask is not None and quote.ask > 0 and quote.ask <= order.limit_price:
                price = min(quote.ask, order.limit_price)
        else:
            if quote.bid is not None and quote.bid > 0 and quote.bid >= order.limit_price:
                price = max(quote.bid, order.limit_price)

        if price is None:
            self._record_check(order_id, quote)
            return BrokerOrderResult(
                ok=True, order=self.get_order(order_id),
                message=f"not marketable (bid {quote.bid} / ask {quote.ask} vs limit {order.limit_price}); PENDING"
            )

        return self._apply_fill(order_id, quote, price)

    def _record_check(self, order_id, quote):
        now = self._now_iso()
        with self._txn() as conn:
            conn.execute(
                """
                UPDATE paper_orders
                SET last_quote_bid = ?, last_quote_ask = ?, last_checked_at = ?, updated_at = ?
                WHERE order_id = ? AND status = 'PENDING'
                """,
                (quote.bid, quote.ask, now, now, order_id)
            )

    def _reject_pending(self, order_id, reason):
        now = self._now_iso()
        with self._txn() as conn:
            conn.execute(
                """
                UPDATE paper_orders
                SET status = 'REJECTED', rejection_reason = ?, updated_at = ?
                WHERE order_id = ? AND status = 'PENDING'
                """,
                (reason, now, order_id)
            )
        return BrokerOrderResult(ok=False, order=self.get_order(order_id), message=reason)

    def _apply_fill(self, order_id, quote, price) -> BrokerOrderResult:
        """Fill + order + position in one transaction. Re-checks PENDING inside it."""
        now = self._now_iso()
        position_id = None

        with self._txn() as conn:
            row = conn.execute("SELECT * FROM paper_orders WHERE order_id = ?", (order_id,)).fetchone()

            if row is None or row["status"] != "PENDING":
                # Someone else filled/cancelled it first: do nothing.
                return BrokerOrderResult(ok=row is not None, order=self._order(dict(row)) if row else None,
                                         message="order no longer pending; no fill applied")

            order = dict(row)

            if order["side"] == "SELL_TO_CLOSE":
                position = conn.execute(
                    "SELECT * FROM paper_positions WHERE position_id = ?", (order["position_id"],)
                ).fetchone()
                if position is None or position["status"] != "OPEN" or position["quantity"] < order["quantity"]:
                    conn.execute(
                        """
                        UPDATE paper_orders SET status = 'REJECTED', rejection_reason = ?, updated_at = ?
                        WHERE order_id = ?
                        """,
                        ("position no longer has enough open quantity", now, order_id)
                    )
                    return BrokerOrderResult(ok=False, message="position no longer has enough open quantity")

            fill_id = self._new_id("PF")
            self._insert_fill(conn, fill_id, order, quote, price, now)

            conn.execute(
                """
                UPDATE paper_orders
                SET status = 'FILLED', fill_price = ?, filled_at = ?, updated_at = ?,
                    last_quote_bid = ?, last_quote_ask = ?, last_checked_at = ?
                WHERE order_id = ? AND status = 'PENDING'
                """,
                (price, now, now, quote.bid, quote.ask, now, order_id)
            )

            if order["side"] == "BUY_TO_OPEN":
                position_id = self._new_id("PP")
                self._insert_position(conn, position_id, order, quote, price, now)
                conn.execute("UPDATE paper_orders SET position_id = ? WHERE order_id = ?", (position_id, order_id))
            else:
                position_id = order["position_id"]
                self._reduce_position(conn, dict(position), order["quantity"], price, quote, now)

        return BrokerOrderResult(
            ok=True,
            order=self.get_order(order_id),
            fill=self.get_fill_for_order(order_id),
            position=self.get_position(position_id, refresh_mark=False),
            message=f"filled {order['side']} {order['quantity']} @ {price}",
        )

    def _insert_fill(self, conn, fill_id, order, quote, price, now):
        conn.execute(
            """
            INSERT INTO paper_fills (
                fill_id, order_id, contract_symbol, side, quantity, price,
                quote_bid, quote_ask, filled_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (fill_id, order["order_id"], order["contract_symbol"], order["side"],
             order["quantity"], price, quote.bid, quote.ask, now)
        )

    def _insert_position(self, conn, position_id, order, quote, price, now):
        mark = quote.mark if quote.mark is not None else price
        conn.execute(
            """
            INSERT INTO paper_positions (
                position_id, entry_order_id, signal_id, proposal_key, contract_symbol,
                underlying, option_type, contract_multiplier, quantity, original_quantity,
                entry_price, current_mark, mark_as_of, unrealized_pnl, realized_pnl,
                status, opened_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'OPEN', ?, ?)
            """,
            (
                position_id, order["order_id"], order["signal_id"], order["proposal_key"],
                order["contract_symbol"], quote.underlying, quote.option_type,
                quote.contract_multiplier, order["quantity"], order["quantity"], price,
                mark, now, (mark - price) * quote.contract_multiplier * order["quantity"],
                now, now,
            )
        )

    def _reduce_position(self, conn, position, quantity, price, quote, now):
        remaining = position["quantity"] - quantity
        realized = position["realized_pnl"] + (price - position["entry_price"]) * position["contract_multiplier"] * quantity
        mark = quote.mark if quote.mark is not None else price

        if remaining == 0:
            conn.execute(
                """
                UPDATE paper_positions
                SET quantity = 0, status = 'CLOSED', realized_pnl = ?, unrealized_pnl = 0,
                    current_mark = ?, mark_as_of = ?, closed_at = ?, updated_at = ?
                WHERE position_id = ?
                """,
                (realized, mark, now, now, now, position["position_id"])
            )
        else:
            conn.execute(
                """
                UPDATE paper_positions
                SET quantity = ?, realized_pnl = ?, current_mark = ?, mark_as_of = ?,
                    unrealized_pnl = ?, updated_at = ?
                WHERE position_id = ?
                """,
                (remaining, realized, mark, now,
                 (mark - position["entry_price"]) * position["contract_multiplier"] * remaining,
                 now, position["position_id"])
            )

    #
    # Positions
    #

    def get_position(self, position_id, refresh_mark=True):
        rows = self._read("SELECT * FROM paper_positions WHERE position_id = ?", (position_id,))
        position = self._position(rows[0] if rows else None)

        if position is None or not refresh_mark or position.status != "OPEN":
            return position

        return self._mark(position)

    def list_positions(self, status=None):
        if status:
            rows = self._read("SELECT * FROM paper_positions WHERE status = ? ORDER BY opened_at, position_id", (status,))
        else:
            rows = self._read("SELECT * FROM paper_positions ORDER BY opened_at, position_id")
        return [self._position(r) for r in rows]

    def mark_to_market(self):
        return [self._mark(p) for p in self.list_positions("OPEN")]

    def _mark(self, position: PaperPosition) -> PaperPosition:
        """Refresh mark from the provider. On failure keep the stored values."""
        try:
            quote = self.chain_provider.get_option_contract(position.contract_symbol)
        except Exception as error:
            self._log(f"[PAPER] mark unavailable for {position.contract_symbol}: {type(error).__name__}")
            return position

        mark = quote.mark
        if mark is None and quote.bid is not None and quote.ask is not None:
            mark = (quote.bid + quote.ask) / 2
        if mark is None:
            return position

        now = self._now_iso()
        with self._txn() as conn:
            conn.execute(
                """
                UPDATE paper_positions
                SET current_mark = ?, mark_as_of = ?,
                    unrealized_pnl = (? - entry_price) * contract_multiplier * quantity,
                    updated_at = ?
                WHERE position_id = ? AND status = 'OPEN'
                """,
                (mark, now, mark, now, position.position_id)
            )

        return self.get_position(position.position_id, refresh_mark=False)

    def close_position(self, position_id, limit_price, client_order_id, quantity=None):
        position = self.get_position(position_id, refresh_mark=False)

        if position is None:
            return BrokerOrderResult(ok=False, message=f"position {position_id} not found; nothing to close")

        # Replaying a close must reproduce the original request, not
        # re-derive the quantity from the (now smaller/closed) position.
        previous = self.get_order_by_client_id(client_order_id)
        if quantity is None and previous is not None:
            quantity = previous.quantity

        return self.submit_order(OrderRequest(
            client_order_id=client_order_id,
            signal_id=position.signal_id,
            proposal_key=position.proposal_key,
            position_id=position_id,
            contract_symbol=position.contract_symbol,
            side="SELL_TO_CLOSE",
            quantity=quantity if quantity is not None else max(position.quantity, 1),
            limit_price=limit_price,
        ))
