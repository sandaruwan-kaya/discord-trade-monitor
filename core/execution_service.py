"""
Execution Service — the ONLY application component that calls the broker.

    Agent 2 -> OptionProposal (data)
    ExecutionService.execute_entry(proposal_key):
        load proposal -> CLAIM execution key (atomic, unique)
        -> fresh Policy Engine evaluation (persisted) -> if APPROVED:
        SUBMITTING (persisted) -> broker.submit_order(client_order_id = execution key)
        -> persist broker result

Guarantees:
    - One execution per proposal: execution_key = "entry:{proposal_key}" is
      UNIQUE and claimed with INSERT OR IGNORE inside BEGIN IMMEDIATE.
    - Time-of-check/time-of-use: policy is ALWAYS evaluated fresh, right
      before submission. Old approvals are never reused.
    - The order is built only from the validated proposal (contract, side,
      quantity, limit). Nothing is repriced, resized or re-selected.
    - SUBMITTING is persisted BEFORE the broker call, and the broker's
      client_order_id is the execution key. After a crash anywhere, the
      broker order (if any) is found by that stable id and reconciled;
      a second order is never created.

Crash points and recovery (never re-submits):
    A. after CLAIM, before policy       -> CLAIMED  -> stale: INTERRUPTED
    B. after policy, before SUBMITTING  -> CLAIMED  -> stale: INTERRUPTED
    C. after SUBMITTING, before submit  -> SUBMITTING, no broker order -> INTERRUPTED
    D. after submit, before persisting  -> SUBMITTING, broker order exists
                                           -> reconciled to ORDER_* by client_order_id

Must never: call an LLM, import agents, modify proposals, bypass the Policy
Engine or the broker, submit live orders, or retry with a new order id.
"""
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import config
import database
from broker.base import BrokerBase
from core import lifecycle_sync, policy_engine
from schemas import ExecutionResult, OptionProposal, OrderRequest


TERMINAL = {"POLICY_REJECTED", "ORDER_FILLED", "ORDER_REJECTED", "ORDER_CANCELLED", "FAILED", "INTERRUPTED"}
IN_FLIGHT = {"CLAIMED", "SUBMITTING"}

BROKER_TO_EXECUTION = {
    "PENDING": "ORDER_PENDING",
    "FILLED": "ORDER_FILLED",
    "REJECTED": "ORDER_REJECTED",
    "CANCELLED": "ORDER_CANCELLED",
}


def entry_execution_key(proposal_key: str) -> str:
    return f"entry:{proposal_key}"


# EXIT canonical identity: one full close per POSITION (Agent 4 guarantees a
# single full-exit request per position, exit:{position_id}:full).
def exit_execution_key(position_id: str) -> str:
    return f"exit:{position_id}:full"


def close_client_order_id(position_id: str) -> str:
    return f"close:{position_id}:full"


# Exit states from which a later call may continue. Safe because every
# resumption first looks the close order up by its stable client_order_id,
# and the broker de-duplicates on it: a second close order cannot exist.
EXIT_RESUMABLE = {"WAITING_FOR_QUOTE", "INTERRUPTED", "FAILED"}
EXIT_FINAL = {"ORDER_FILLED", "POLICY_REJECTED", "ORDER_REJECTED", "ORDER_CANCELLED"}


class ExecutionService:

    def __init__(self, broker: BrokerBase, chain_provider, market, now_ms=None, log=print):
        self.broker = broker
        self.chain_provider = chain_provider
        self.market = market
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))
        self._log = log

    #
    # Persistence helpers
    #

    def _now_iso(self):
        return datetime.fromtimestamp(self._now_ms() / 1000, timezone.utc).isoformat(timespec="seconds")

    def _connect(self):
        conn = database.get_connection()
        conn.isolation_level = None
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _txn(self):
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

    def get_attempt(self, execution_key):
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM execution_attempts WHERE execution_key = ?", (execution_key,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_events(self, execution_id):
        conn = self._connect()
        try:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM execution_events WHERE execution_id = ? ORDER BY id", (execution_id,)
            )]
        finally:
            conn.close()

    def _transition(self, execution_key, expected, new_status, **fields):
        """
        Guarded status change: only applies if the current status is in
        `expected`. Returns True if applied. Appends an execution_event.
        """
        now = self._now_iso()
        with self._txn() as conn:
            row = conn.execute(
                "SELECT execution_id, status FROM execution_attempts WHERE execution_key = ?",
                (execution_key,)
            ).fetchone()
            if row is None or row["status"] not in expected:
                return False

            sets = {"status": new_status, "updated_at": now, **fields}
            if new_status in TERMINAL:
                sets["completed_at"] = now

            conn.execute(
                f"UPDATE execution_attempts SET {', '.join(f'{k} = ?' for k in sets)} "
                "WHERE execution_key = ? AND status = ?",
                (*sets.values(), execution_key, row["status"])
            )
            self._event(conn, row["execution_id"], row["status"], new_status,
                        {k: v for k, v in fields.items() if k not in ("result_json", "input_json")}, now)
        return True

    @staticmethod
    def _event(conn, execution_id, from_status, to_status, details, now):
        conn.execute(
            """
            INSERT INTO execution_events (execution_id, from_status, to_status, details, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (execution_id, from_status, to_status, json.dumps(details, default=str), now)
        )

    def _result(self, execution_key, replay=False, message=""):
        a = self.get_attempt(execution_key)
        reasons = []
        order_status = None

        if a.get("result_json"):
            result = json.loads(a["result_json"])
            reasons = result.get("policy_reasons", [])
            order_status = result.get("broker_order_status")

        return ExecutionResult(
            ok=a["status"] in ("ORDER_PENDING", "ORDER_FILLED"),
            status=a["status"],
            proposal_key=a["proposal_key"],
            purpose=a["purpose"],
            exit_request_id=a.get("exit_request_id"),
            execution_id=a["execution_id"],
            execution_key=a["execution_key"],
            replay=replay,
            policy_decision_id=a["policy_decision_id"],
            policy_reasons=reasons,
            client_order_id=a["client_order_id"],
            broker_order_id=a["broker_order_id"],
            broker_order_status=order_status,
            error_code=a["error_code"],
            message=message or a["error_message"] or "",
        )

    #
    # Entry execution
    #

    def execute_entry(self, proposal_key: str) -> ExecutionResult:
        proposal_row = database.get_proposal(proposal_key)

        # Not persisted: nothing to claim yet, and the proposal may still appear.
        if proposal_row is None:
            return ExecutionResult(ok=False, status="REFUSED", proposal_key=proposal_key, persisted=False,
                                   error_code="PROPOSAL_NOT_FOUND", message="no such proposal")
        if proposal_row["status"] == "PENDING":
            return ExecutionResult(ok=False, status="REFUSED", proposal_key=proposal_key, persisted=False,
                                   error_code="PROPOSAL_NOT_READY", message="Agent 2 has not finished this proposal")

        key = entry_execution_key(proposal_key)

        # CLAIM (atomic). Loser of a race, or any later call, gets the existing record.
        now = self._now_iso()
        with self._txn() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO execution_attempts (
                    execution_id, execution_key, proposal_key, signal_id, purpose, status,
                    client_order_id, created_at, updated_at, input_json
                )
                VALUES (?, ?, ?, ?, 'ENTRY', 'CLAIMED', ?, ?, ?, ?)
                """,
                (
                    f"EX-{uuid.uuid4().hex[:16]}", key, proposal_key, proposal_row["signal_id"],
                    key, now, now, json.dumps({"proposal_row": proposal_row}, default=str),
                )
            )
            claimed = cursor.rowcount == 1
            if claimed:
                execution_id = conn.execute(
                    "SELECT execution_id FROM execution_attempts WHERE execution_key = ?", (key,)
                ).fetchone()[0]
                self._event(conn, execution_id, None, "CLAIMED", {}, now)

        if not claimed:
            return self._result(key, replay=True, message="execution already exists for this proposal")

        try:
            return self._run_claimed(key, proposal_row)
        except Exception as error:
            # Unexpected failure after CLAIM. If we got as far as SUBMITTING the
            # outcome is uncertain: reconcile by stable id instead of guessing.
            self._log(f"[EXEC] {key}: {type(error).__name__}: {str(error)[:200]}")
            attempt = self.get_attempt(key)
            if attempt and attempt["status"] == "SUBMITTING":
                return self._reconcile_submitting(key)
            self._transition(key, {"CLAIMED"}, "FAILED", error_code="INTERNAL_ERROR",
                             error_message=f"{type(error).__name__}: {str(error)[:300]}")
            return self._result(key)

    def _run_claimed(self, key, proposal_row) -> ExecutionResult:
        # 1. Validate the proposal (final states are recorded, never retried).
        if proposal_row["status"] != "PROPOSED":
            self._transition(key, {"CLAIMED"}, "FAILED", error_code="PROPOSAL_NOT_PROPOSED",
                             error_message=f"proposal status is {proposal_row['status']}")
            return self._result(key)

        try:
            proposal = OptionProposal.model_validate_json(proposal_row["output_json"] or "")
            if (
                proposal.status != "PROPOSED" or proposal.proposal_key != proposal_row["proposal_key"]
                or proposal.side != "BUY_TO_OPEN"
            ):
                raise ValueError("proposal record is inconsistent")
        except Exception as error:
            self._transition(key, {"CLAIMED"}, "FAILED", error_code="PROPOSAL_MALFORMED",
                             error_message=f"{type(error).__name__}: {str(error)[:300]}")
            return self._result(key)

        # 2. FRESH policy evaluation (persisted). Never reuse an old approval.
        try:
            decision = policy_engine.evaluate_entry(proposal, self.chain_provider, self.market, self._now_ms())
        except Exception as error:
            self._transition(key, {"CLAIMED"}, "FAILED", error_code="POLICY_ERROR",
                             error_message=f"{type(error).__name__}: {str(error)[:300]}")
            return self._result(key)

        if not decision.approved:
            self._transition(
                key, {"CLAIMED"}, "POLICY_REJECTED",
                policy_decision_id=decision.decision_id, error_code="POLICY_REJECTED",
                error_message=",".join(decision.reasons),
                result_json=json.dumps({"policy_reasons": decision.reasons}),
            )
            return self._result(key)

        # 3. Order built ONLY from the validated proposal.
        request = OrderRequest(
            client_order_id=key,
            signal_id=proposal.signal_id,
            proposal_key=proposal.proposal_key,
            contract_symbol=proposal.contract_symbol,
            side="BUY_TO_OPEN",
            quantity=proposal.quantity,
            limit_price=proposal.limit_price,
        )

        # 4. Persist SUBMITTING before touching the broker.
        if not self._transition(key, {"CLAIMED"}, "SUBMITTING", policy_decision_id=decision.decision_id,
                                result_json=json.dumps({"policy_reasons": [], "order_request": request.model_dump()})):
            return self._result(key)

        # 5. Submit (idempotent on client_order_id).
        try:
            broker_result = self.broker.submit_order(request)
        except Exception as error:
            self._log(f"[EXEC] {key}: broker error {type(error).__name__}; reconciling by client_order_id")
            return self._reconcile_submitting(key, broker_error=error)

        # 6. Persist broker outcome.
        return self._apply_broker_order(key, {"SUBMITTING"}, broker_result.order, broker_result.message)

    def _apply_broker_order(self, key, expected, order, message=""):
        if order is None:
            self._transition(key, expected, "FAILED", error_code="BROKER_NO_ORDER",
                             error_message=message[:300] or "broker returned no order")
            return self._result(key)

        status = BROKER_TO_EXECUTION[order.status]
        self._transition(
            key, expected, status,
            broker_order_id=order.order_id,
            error_code="ORDER_REJECTED" if order.status == "REJECTED" else None,
            error_message=order.rejection_reason,
            result_json=json.dumps({"policy_reasons": [], "broker_order_status": order.status,
                                    "broker_order": order.model_dump(), "message": message}),
        )
        return self._result(key)

    #
    # Reconciliation (never submits)
    #

    def _reconcile_submitting(self, key, broker_error=None):
        """SUBMITTING with unknown outcome: look the order up by its stable id."""
        attempt = self.get_attempt(key)
        try:
            order = self.broker.get_order_by_client_id(attempt["client_order_id"])
        except Exception as error:
            self._log(f"[EXEC] {key}: broker lookup failed ({type(error).__name__}); left SUBMITTING")
            return self._result(key, message="broker unreachable; execution left SUBMITTING for reconciliation")

        if order is not None:
            return self._apply_broker_order(key, {"SUBMITTING"}, order, "reconciled by client_order_id")

        detail = f"{type(broker_error).__name__}: {str(broker_error)[:200]}" if broker_error else \
            "no broker order exists for client_order_id; not retried"
        self._transition(key, {"SUBMITTING"}, "INTERRUPTED", error_code="SUBMIT_UNCONFIRMED",
                         error_message=detail)
        return self._result(key)

    def reconcile_execution(self, execution_key, process_pending=False) -> ExecutionResult | None:
        attempt = self.get_attempt(execution_key)
        if attempt is None:
            return None

        if attempt["purpose"] == "EXIT":
            return self._reconcile_exit_attempt(attempt, process_pending)

        status = attempt["status"]

        if status in IN_FLIGHT:
            age = (self._now_ms() - _iso_to_ms(attempt["updated_at"])) / 1000
            if age < config.execution_claim_timeout_seconds():
                return self._result(execution_key, message="execution in progress")
            if status == "SUBMITTING":
                return self._reconcile_submitting(execution_key)
            # CLAIMED: submission is only ever attempted after SUBMITTING is
            # persisted, so no broker order can exist. Verify anyway.
            order = self.broker.get_order_by_client_id(attempt["client_order_id"])
            if order is not None:
                return self._apply_broker_order(execution_key, {"CLAIMED"}, order, "reconciled by client_order_id")
            self._transition(execution_key, {"CLAIMED"}, "INTERRUPTED", error_code="CLAIM_INTERRUPTED",
                             error_message="process stopped before submission; not retried")
            return self._result(execution_key)

        if status == "ORDER_PENDING":
            if process_pending:
                self.broker.process_pending_orders()
            order = self.broker.get_order(attempt["broker_order_id"])
            if order is not None and order.status != "PENDING":
                return self._apply_broker_order(execution_key, {"ORDER_PENDING"}, order, "reconciled")

        return self._result(execution_key)

    def reconcile_pending_executions(self, process_pending=True):
        """Advance every PENDING / in-flight execution. Never submits a new order."""
        if process_pending:
            self.broker.process_pending_orders()

        conn = self._connect()
        try:
            keys = [r[0] for r in conn.execute(
                "SELECT execution_key FROM execution_attempts "
                "WHERE status IN ('ORDER_PENDING', 'CLAIMED', 'SUBMITTING') "
                "   OR (purpose = 'EXIT' AND status IN ('WAITING_FOR_QUOTE', 'INTERRUPTED', 'FAILED')) "
                "ORDER BY created_at"
            )]
        finally:
            conn.close()

        return [self.reconcile_execution(k, process_pending=False) for k in keys]


    #
    # EXIT execution (full exits only; paper only; no LLM)
    #

    def _exit_request(self, exit_request_id):
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM exit_requests WHERE exit_request_id = ?", (exit_request_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def _position(self, position_id):
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM paper_positions WHERE position_id = ?", (position_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def _stale(self, attempt):
        age = (self._now_ms() - _iso_to_ms(attempt["updated_at"])) / 1000
        return age >= config.execution_claim_timeout_seconds()

    def execute_exit(self, exit_request_id: str) -> ExecutionResult:
        """
        Execute a persisted full EXIT request: fresh exit policy -> one
        SELL_TO_CLOSE at the fresh BID for the full open quantity.
        """
        xr = self._exit_request(exit_request_id)
        if xr is None:
            return ExecutionResult(ok=False, status="REFUSED", purpose="EXIT", exit_request_id=exit_request_id,
                                   persisted=False, error_code="EXIT_REQUEST_NOT_FOUND")
        if xr["requested_action"] != "EXIT":
            return ExecutionResult(ok=False, status="NOT_EXECUTABLE_V1", purpose="EXIT",
                                   exit_request_id=exit_request_id, persisted=False,
                                   error_code="REDUCE_NOT_EXECUTABLE_V1",
                                   message="REDUCE requests are advisory in V1; no order created")

        position_id = xr["position_id"]
        key = exit_execution_key(position_id)
        coid = close_client_order_id(position_id)
        attempt = self.get_attempt(key)

        if xr["status"] != "ACTIVE":
            if attempt:
                return self._result(key, replay=True, message=f"exit request is {xr['status']}")
            return ExecutionResult(ok=False, status="REFUSED", purpose="EXIT", exit_request_id=exit_request_id,
                                   persisted=False, error_code="EXIT_REQUEST_NOT_ACTIVE",
                                   message=f"exit request is {xr['status']}; nothing to execute")

        if attempt is not None:
            status = attempt["status"]
            if status in ("SUBMITTING", "ORDER_PENDING") or (status == "CLAIMED" and not self._stale(attempt)):
                return self._reconcile_exit_attempt(attempt, process_pending=False)
            if status in EXIT_FINAL:
                self._sync_exit_lifecycle(key)
                return self._result(key, replay=True)
            # Resumable (WAITING_FOR_QUOTE / INTERRUPTED / FAILED / stale CLAIMED):
            # never submit before checking the broker by stable id.
            order = self.broker.get_order_by_client_id(coid)
            if order is not None:
                return self._apply_exit_order(key, {status}, order, "reconciled by client_order_id")
            position = self._position(position_id)
            if position is not None and position["status"] == "CLOSED":
                lifecycle_sync.reconcile_already_closed(xr["trade_id"], exit_request_id, self._now_iso())
                return self._result(key, message="position closed elsewhere; lifecycle reconciled")
            if not self._transition(key, {status}, "CLAIMED"):
                return self._result(key, replay=True, message="another caller resumed this exit")
            return self._run_exit(key, xr, coid)

        position = self._position(position_id)
        if position is not None and position["status"] == "CLOSED":
            lifecycle_sync.reconcile_already_closed(xr["trade_id"], exit_request_id, self._now_iso())
            return ExecutionResult(ok=True, status="ALREADY_CLOSED", purpose="EXIT", exit_request_id=exit_request_id,
                                   persisted=False, message="position already closed; lifecycle reconciled, no order")

        trade = None
        conn = self._connect()
        try:
            row = conn.execute("SELECT proposal_key, signal_id FROM trades WHERE trade_id = ?", (xr["trade_id"],)).fetchone()
            trade = dict(row) if row else None
        finally:
            conn.close()

        now = self._now_iso()
        with self._txn() as conn:
            claimed = conn.execute(
                """
                INSERT OR IGNORE INTO execution_attempts (
                    execution_id, execution_key, proposal_key, signal_id, purpose, status,
                    client_order_id, created_at, updated_at, input_json, exit_request_id
                )
                VALUES (?, ?, ?, ?, 'EXIT', 'CLAIMED', ?, ?, ?, ?, ?)
                """,
                (f"EX-{uuid.uuid4().hex[:16]}", key,
                 (trade or {}).get("proposal_key") or xr["request_key"], (trade or {}).get("signal_id"),
                 coid, now, now, json.dumps({"exit_request": xr}, default=str), exit_request_id)
            ).rowcount == 1
            if claimed:
                execution_id = conn.execute(
                    "SELECT execution_id FROM execution_attempts WHERE execution_key = ?", (key,)
                ).fetchone()[0]
                self._event(conn, execution_id, None, "CLAIMED", {"exit_request_id": exit_request_id}, now)

        if not claimed:
            return self._result(key, replay=True, message="exit execution already exists for this position")

        return self._run_exit(key, xr, coid)

    def _run_exit(self, key, xr, coid) -> ExecutionResult:
        attempt = self.get_attempt(key)
        try:
            inp = policy_engine.gather_exit_input(xr["exit_request_id"], self.chain_provider, self._now_ms(), coid)
            decision = policy_engine.persist_decision(policy_engine.evaluate_exit(inp), inp)

            if not decision.approved:
                status = "WAITING_FOR_QUOTE" if decision.retryable else "POLICY_REJECTED"
                self._transition(
                    key, {"CLAIMED"}, status,
                    policy_decision_id=decision.decision_id,
                    error_code="EXIT_BLOCKED_DATA" if decision.retryable else "EXIT_POLICY_REJECTED",
                    error_message=",".join(decision.reasons),
                    result_json=json.dumps({"policy_reasons": decision.reasons}),
                )
                lifecycle_sync.set_exit_request_execution(xr["exit_request_id"], attempt["execution_id"],
                                                          status, self._now_iso())
                return self._result(key)

            if not self._transition(
                key, {"CLAIMED"}, "SUBMITTING", policy_decision_id=decision.decision_id,
                result_json=json.dumps({"policy_reasons": [], "close": {
                    "position_id": xr["position_id"], "limit_price": decision.exit_limit_price,
                    "quantity": decision.exit_quantity, "client_order_id": coid}}),
            ):
                return self._result(key)
            lifecycle_sync.set_exit_request_execution(xr["exit_request_id"], attempt["execution_id"],
                                                      "SUBMITTING", self._now_iso())

            try:
                broker_result = self.broker.close_position(
                    xr["position_id"],
                    limit_price=decision.exit_limit_price,     # the fresh BID, never mark/mid/ask
                    client_order_id=coid,
                    quantity=decision.exit_quantity,           # full open quantity, revalidated
                )
            except Exception as error:
                self._log(f"[EXEC] {key}: broker error {type(error).__name__}; reconciling by client_order_id")
                return self._reconcile_exit_attempt(self.get_attempt(key), process_pending=False, force=True,
                                                    broker_error=error)

            return self._apply_exit_order(key, {"SUBMITTING"}, broker_result.order, broker_result.message)

        except Exception as error:
            self._log(f"[EXEC] {key}: {type(error).__name__}: {str(error)[:200]}")
            current = self.get_attempt(key)
            if current and current["status"] == "SUBMITTING":
                return self._reconcile_exit_attempt(current, process_pending=False, force=True)
            self._transition(key, {"CLAIMED"}, "FAILED", error_code="INTERNAL_ERROR",
                             error_message=f"{type(error).__name__}: {str(error)[:300]} (exit resumable)")
            return self._result(key)

    def _apply_exit_order(self, key, expected, order, message=""):
        result = self._apply_broker_order(key, expected, order, message)
        self._sync_exit_lifecycle(key)
        return self._result(key, message=result.message)

    def _sync_exit_lifecycle(self, key):
        a = self.get_attempt(key)
        if a is None or not a["broker_order_id"]:
            return
        order = self.broker.get_order(a["broker_order_id"])
        xr = self._exit_request(a["exit_request_id"])
        position = self._position(xr["position_id"]) if xr else None
        if order is None or xr is None:
            return
        if order.status in ("PENDING", "FILLED"):
            lifecycle_sync.sync_after_close(xr["trade_id"], xr["exit_request_id"], a["execution_id"], order.order_id,
                                            order.status, position["status"] if position else None, self._now_iso())
        else:
            lifecycle_sync.set_exit_request_execution(xr["exit_request_id"], a["execution_id"],
                                                      a["status"], self._now_iso())

    def _reconcile_exit_attempt(self, attempt, process_pending=False, force=False, broker_error=None):
        """Advance an exit execution from broker state. Submits only when resuming a resumable exit."""
        key, status = attempt["execution_key"], attempt["status"]

        if status in EXIT_RESUMABLE or (status == "CLAIMED" and self._stale(attempt)):
            return self.execute_exit(attempt["exit_request_id"])

        if status == "SUBMITTING":
            try:
                order = self.broker.get_order_by_client_id(attempt["client_order_id"])
            except Exception as error:
                self._log(f"[EXEC] {key}: broker lookup failed ({type(error).__name__}); left SUBMITTING")
                return self._result(key, message="broker unreachable; exit left SUBMITTING")
            if order is not None:
                return self._apply_exit_order(key, {"SUBMITTING"}, order, "reconciled by client_order_id")
            if force or self._stale(attempt):
                detail = (f"{type(broker_error).__name__}: {str(broker_error)[:200]}" if broker_error
                          else "no close order exists for client_order_id") + "; exit remains required (resumable)"
                self._transition(key, {"SUBMITTING"}, "INTERRUPTED", error_code="SUBMIT_UNCONFIRMED",
                                 error_message=detail)
                lifecycle_sync.set_exit_request_execution(attempt["exit_request_id"], attempt["execution_id"],
                                                          "INTERRUPTED", self._now_iso())
            return self._result(key)

        if status == "ORDER_PENDING":
            if process_pending:
                self.broker.process_pending_orders()
            order = self.broker.get_order(attempt["broker_order_id"])
            if order is not None and order.status != "PENDING":
                return self._apply_exit_order(key, {"ORDER_PENDING"}, order, "reconciled")
            self._sync_exit_lifecycle(key)
            return self._result(key)

        if status in EXIT_FINAL:
            self._sync_exit_lifecycle(key)
        return self._result(key)


def _iso_to_ms(value):
    return int(datetime.fromisoformat(value).timestamp() * 1000)
