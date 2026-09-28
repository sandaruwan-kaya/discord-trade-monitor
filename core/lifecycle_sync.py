"""
Narrow, deterministic lifecycle reconciliation used by the Execution
Service after it acts on an ExitRequest.

Direction of control stays one-way:
    Agent 4 creates ExitRequests  ->  ExecutionService consumes them
    ExecutionService -> (this module) records what the broker did
Agent 4 never calls the ExecutionService.

Every change goes through core.trade_lifecycle's transition table and a
guarded UPDATE (WHERE state = expected), so concurrent callers (including
Agent 4's own broker reconciliation) can never double-apply a transition.
"""
import json
import sqlite3
from contextlib import contextmanager

import database
from core.trade_lifecycle import is_transition_allowed


@contextmanager
def _txn():
    conn = database.get_connection()
    conn.isolation_level = None
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def _transition(conn, trade_id, to_state, action, codes, reasons, exit_request_id, details, now):
    row = conn.execute("SELECT state FROM trades WHERE trade_id = ?", (trade_id,)).fetchone()
    if row is None:
        return None
    current = row["state"]
    if current == to_state or not is_transition_allowed(current, to_state):
        return current

    conn.execute(
        """
        UPDATE trades SET state = ?, updated_at = ?,
            closed_at = CASE WHEN ? = 'CLOSED' THEN ? ELSE closed_at END
        WHERE trade_id = ? AND state = ?
        """,
        (to_state, now, to_state, now, trade_id, current)
    )
    conn.execute(
        """
        INSERT INTO trade_lifecycle_events (
            trade_id, from_state, to_state, action, reason_codes_json, reasons_json,
            risk_assessment_id, exit_request_id, details_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
        """,
        (trade_id, current, to_state, action, json.dumps(codes), json.dumps(reasons),
         exit_request_id, json.dumps(details, default=str), now)
    )
    return to_state


def set_exit_request_execution(exit_request_id, execution_id, execution_status, now):
    with _txn() as conn:
        conn.execute(
            "UPDATE exit_requests SET execution_id = ?, execution_status = ?, updated_at = ? WHERE exit_request_id = ?",
            (execution_id, execution_status, now, exit_request_id)
        )


def sync_after_close(trade_id, exit_request_id, execution_id, order_id, order_status, position_status, now):
    """
    After a close order exists: EXIT_REQUESTED -> CLOSE_ORDER_PENDING; and if
    the broker position is actually CLOSED: -> CLOSED and the exit request
    becomes COMPLETED. Both steps are separate, visible events.
    """
    with _txn() as conn:
        state = conn.execute("SELECT state FROM trades WHERE trade_id = ?", (trade_id,)).fetchone()
        if state is None:
            return None
        state = state["state"]

        if order_status in ("PENDING", "FILLED") and state == "EXIT_REQUESTED":
            state = _transition(conn, trade_id, "CLOSE_ORDER_PENDING", "CLOSE_ORDER_SUBMITTED",
                                ["CLOSE_ORDER_SUBMITTED"], [f"SELL_TO_CLOSE {order_id} submitted"],
                                exit_request_id, {"execution_id": execution_id, "broker_order_id": order_id}, now)

        if position_status == "CLOSED" and state in ("CLOSE_ORDER_PENDING", "EXIT_REQUESTED"):
            state = _transition(conn, trade_id, "CLOSED", "MARK_CLOSED", ["BROKER_POSITION_CLOSED"],
                                ["close order filled; broker position closed"], exit_request_id,
                                {"execution_id": execution_id, "broker_order_id": order_id}, now)
            conn.execute(
                """
                UPDATE exit_requests SET status = 'COMPLETED', completed_at = ?, updated_at = ?,
                    execution_id = ?, execution_status = 'ORDER_FILLED'
                WHERE exit_request_id = ? AND status = 'ACTIVE'
                """,
                (now, now, execution_id, exit_request_id)
            )
        else:
            conn.execute(
                "UPDATE exit_requests SET execution_id = ?, execution_status = ?, updated_at = ? "
                "WHERE exit_request_id = ?",
                (execution_id, "ORDER_PENDING" if order_status == "PENDING" else order_status, now, exit_request_id)
            )

        return state


def reconcile_already_closed(trade_id, exit_request_id, now):
    """Position closed without this exit execution: reconcile, never order."""
    with _txn() as conn:
        state = _transition(conn, trade_id, "CLOSED", "MARK_CLOSED", ["BROKER_POSITION_CLOSED"],
                            ["broker position already closed; no close order needed"], exit_request_id,
                            {"reconciled_by": "execution_service"}, now)
        conn.execute(
            """
            UPDATE exit_requests SET status = 'COMPLETED', completed_at = ?, updated_at = ?,
                execution_status = COALESCE(execution_status, 'ALREADY_CLOSED')
            WHERE exit_request_id = ? AND status = 'ACTIVE'
            """,
            (now, now, exit_request_id)
        )
        return state
