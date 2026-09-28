"""
Agent 4 — Trade Manager (lifecycle). Deterministic: NO LLM call.

Loads the persisted trade, broker position/order state (read-only SQL),
the latest Agent 3 RiskAssessment and active exit requests, asks the
pure state machine (core/trade_lifecycle.decide) what should happen, and
persists the transition + any ExitRequest atomically.

It never submits, closes or cancels an order and has no broker or
execution-service access. An ExitRequest means "exit required", not
"exit executed".
"""
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import database
from core.trade_lifecycle import current_limits, decide
from schemas import (
    BrokerOrderState,
    BrokerPositionState,
    ExitRequest,
    ExitRequestSummary,
    RiskAssessmentSummary,
    TradeManagementDecision,
    TradeManagementInput,
)


class TradeManager:

    def __init__(self, now_ms=None, log=print):
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))
        self._log = log

    #
    # DB helpers
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

    def _one(self, sql, params=()):
        conn = self._connect()
        try:
            row = conn.execute(sql, params).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def _all(self, sql, params=()):
        conn = self._connect()
        try:
            return [dict(r) for r in conn.execute(sql, params)]
        finally:
            conn.close()

    def get_trade(self, trade_id):
        return self._one("SELECT * FROM trades WHERE trade_id = ?", (trade_id,))

    def get_trade_for_position(self, position_id):
        return self._one("SELECT * FROM trades WHERE position_id = ?", (position_id,))

    def get_events(self, trade_id):
        return self._all("SELECT * FROM trade_lifecycle_events WHERE trade_id = ? ORDER BY event_id", (trade_id,))

    def get_exit_requests(self, trade_id):
        rows = self._all("SELECT * FROM exit_requests WHERE trade_id = ? ORDER BY created_at, exit_request_id", (trade_id,))
        return [self._exit_request(r) for r in rows]

    @staticmethod
    def _exit_request(r):
        return ExitRequest(
            exit_request_id=r["exit_request_id"], request_key=r["request_key"], trade_id=r["trade_id"],
            position_id=r["position_id"], signal_id=r["signal_id"], proposal_key=r["proposal_key"],
            risk_assessment_id=r["risk_assessment_id"], requested_action=r["requested_action"],
            requested_quantity=r["requested_quantity"], hard_exit=bool(r["hard_exit"]),
            reason_codes=json.loads(r["reason_codes_json"]), data_issue_codes=json.loads(r["data_issue_codes_json"]),
            status=r["status"], created_at=r["created_at"], updated_at=r["updated_at"], completed_at=r["completed_at"],
        )

    #
    # Trade creation from a FILLED execution (idempotent)
    #

    def materialize_trade(self, execution_key):
        """Create the Trade for a FILLED entry execution + OPEN paper position. Safe to call repeatedly."""
        ex = self._one("SELECT * FROM execution_attempts WHERE execution_key = ?", (execution_key,))
        if ex is None or ex["status"] != "ORDER_FILLED" or not ex["broker_order_id"]:
            return None

        order = self._one("SELECT * FROM paper_orders WHERE order_id = ?", (ex["broker_order_id"],))
        if order is None or order["status"] != "FILLED" or not order["position_id"]:
            return None

        pos = self._one("SELECT * FROM paper_positions WHERE position_id = ?", (order["position_id"],))
        if pos is None:
            return None

        now = self._now_iso()
        trade_id = f"TR-{ex['execution_id'].removeprefix('EX-')}"

        with self._txn() as conn:
            inserted = conn.execute(
                """
                INSERT OR IGNORE INTO trades (
                    trade_id, signal_id, proposal_key, execution_id, position_id,
                    state, opened_at, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, 'OPEN', ?, ?, ?)
                """,
                (trade_id, ex["signal_id"], ex["proposal_key"], ex["execution_id"], pos["position_id"],
                 pos["opened_at"], now, now)
            ).rowcount == 1

            if inserted:
                self._event(conn, trade_id, None, "OPEN", "MATERIALIZED", ["ENTRY_FILLED"],
                            ["trade created from FILLED entry execution"], None, None,
                            {"execution_id": ex["execution_id"], "policy_decision_id": ex["policy_decision_id"],
                             "broker_order_id": ex["broker_order_id"]}, now)

        return self.get_trade_for_position(pos["position_id"])

    #
    # Lifecycle
    #

    def build_input(self, trade_id) -> TradeManagementInput | None:
        trade = self.get_trade(trade_id)
        if trade is None:
            return None

        pos = self._one("SELECT * FROM paper_positions WHERE position_id = ?", (trade["position_id"],))
        close = self._one(
            """
            SELECT * FROM paper_orders WHERE position_id = ? AND side = 'SELL_TO_CLOSE'
            ORDER BY submitted_at DESC, order_id DESC LIMIT 1
            """,
            (trade["position_id"],)
        )
        # The OLDEST assessment not yet applied to this trade. Applying them
        # in order means an EXIT persisted by Agent 3 can never be skipped
        # because a later HOLD arrived before this loop ran.
        ra = self._one(
            """
            SELECT ra.assessment_id, ra.output_json, ra.assessed_at_ms FROM risk_assessments ra
            WHERE ra.position_id = ?
              AND NOT EXISTS (
                  SELECT 1 FROM trade_assessment_applications x
                  WHERE x.trade_id = ? AND x.assessment_id = ra.assessment_id
              )
            ORDER BY ra.assessed_at_ms ASC, ra.rowid ASC LIMIT 1
            """,
            (trade["position_id"], trade_id)
        )

        summary = None
        if ra:
            o = json.loads(ra["output_json"])
            summary = RiskAssessmentSummary(
                assessment_id=o["assessment_id"], position_id=o["position_id"],
                recommendation=o["recommendation"], risk_state=o["risk_state"],
                hard_rule_triggered=o["hard_rule_triggered"], hard_rule_codes=o["hard_rule_codes"],
                data_issue_codes=o["data_issue_codes"], reasons=o["reasons"],
                unrealized_pnl_pct=o.get("unrealized_pnl_pct"),
                distance_to_invalidation_pct=o.get("distance_to_invalidation_pct"),
                assessed_at_ms=ra["assessed_at_ms"],
                llm_used=bool(o.get("llm_used")),
                llm_status=o.get("llm_status"),
            )

        processed = bool(summary) and self._one(
            "SELECT 1 FROM trade_assessment_applications WHERE trade_id = ? AND assessment_id = ?",
            (trade_id, summary.assessment_id)
        ) is not None

        def active(action):
            r = self._one(
                """
                SELECT * FROM exit_requests WHERE position_id = ? AND requested_action = ? AND status = 'ACTIVE'
                ORDER BY created_at DESC LIMIT 1
                """,
                (trade["position_id"], action)
            )
            return ExitRequestSummary(exit_request_id=r["exit_request_id"], request_key=r["request_key"],
                                      requested_action=r["requested_action"], status=r["status"],
                                      hard_exit=bool(r["hard_exit"])) if r else None

        return TradeManagementInput(
            trade_id=trade_id,
            signal_id=trade["signal_id"],
            proposal_key=trade["proposal_key"],
            execution_id=trade["execution_id"],
            position_id=trade["position_id"],
            current_state=trade["state"],
            position=BrokerPositionState(position_id=pos["position_id"], status=pos["status"],
                                         quantity=pos["quantity"], contract_symbol=pos["contract_symbol"]) if pos else None,
            close_order=BrokerOrderState(order_id=close["order_id"], side=close["side"], status=close["status"],
                                         quantity=close["quantity"]) if close else None,
            latest_risk_assessment=summary,
            assessment_already_processed=processed,
            active_exit_request=active("EXIT"),
            active_reduce_request=active("REDUCE"),
            opened_at=trade["opened_at"],
            now_ms=self._now_ms(),
            limits=current_limits(),
        )

    def process(self, trade_id) -> TradeManagementDecision | None:
        inp = self.build_input(trade_id)
        if inp is None:
            return None

        decision = decide(inp)
        self._persist(decision, inp)
        return decision

    def process_until_current(self, trade_id, max_steps=500):
        """Apply every unapplied assessment in order; stop when nothing changes."""
        decisions = []
        for _ in range(max_steps):
            d = self.process(trade_id)
            if d is None:
                break
            decisions.append(d)
            if not d.transition_allowed or d.risk_assessment_id is None:
                break
        return decisions

    def process_all(self):
        trades = self._all(
            "SELECT trade_id FROM trades WHERE state NOT IN ('CLOSED', 'REJECTED', 'CANCELLED', 'FAILED') ORDER BY created_at"
        )
        return [d for t in trades for d in self.process_until_current(t["trade_id"])]

    def _persist(self, d: TradeManagementDecision, inp: TradeManagementInput):
        if not d.transition_allowed:
            self._log(f"[AGENT4] {d.trade_id}: refused ({d.reason_codes}); nothing written")
            return

        nothing_to_record = (
            d.next_state == d.previous_state and d.exit_request is None and not d.resolve_requests
            and d.evidence_for_exit_request_id is None and d.risk_assessment_id is None
        )
        if nothing_to_record:
            return

        now = self._now_iso()
        try:
            with self._txn() as conn:
                current = conn.execute("SELECT state FROM trades WHERE trade_id = ?", (d.trade_id,)).fetchone()
                if current is None or current["state"] != d.previous_state:
                    raise _Stale()                                  # someone else moved it: write nothing

                exit_request_id = d.evidence_for_exit_request_id
                if d.exit_request is not None:
                    exit_request_id = self._upsert_request(conn, d, inp, now)

                if exit_request_id is None:
                    completed = [rid for rid, st in d.resolve_requests.items() if st == "COMPLETED"]
                    exit_request_id = completed[0] if completed else None

                for request_id, status in d.resolve_requests.items():
                    conn.execute(
                        """
                        UPDATE exit_requests SET status = ?, updated_at = ?, completed_at = ?
                        WHERE exit_request_id = ? AND status = 'ACTIVE'
                        """,
                        (status, now, now, request_id)
                    )

                if d.next_state != d.previous_state:
                    conn.execute(
                        """
                        UPDATE trades SET state = ?, updated_at = ?,
                            closed_at = CASE WHEN ? = 'CLOSED' THEN ? ELSE closed_at END
                        WHERE trade_id = ? AND state = ?
                        """,
                        (d.next_state, now, d.next_state, now, d.trade_id, d.previous_state)
                    )

                if d.risk_assessment_id is not None:
                    # Applied exactly once (PRIMARY KEY); a duplicate raises and rolls back.
                    conn.execute(
                        "INSERT INTO trade_assessment_applications (trade_id, assessment_id, outcome, applied_at) "
                        "VALUES (?, ?, ?, ?)",
                        (d.trade_id, d.risk_assessment_id, ",".join(d.reason_codes) or d.action, now)
                    )

                meaningful = (
                    d.next_state != d.previous_state or d.exit_request is not None
                    or d.evidence_for_exit_request_id is not None or bool(d.resolve_requests)
                )
                if meaningful:
                    self._event(conn, d.trade_id, d.previous_state, d.next_state, d.action, d.reason_codes,
                                d.reasons, d.risk_assessment_id, exit_request_id,
                                {"resolved_requests": d.resolve_requests}, now)
        except (_Stale, sqlite3.IntegrityError) as error:
            # Duplicate processing of the same assessment, or a concurrent change: no-op.
            self._log(f"[AGENT4] {d.trade_id}: not applied ({type(error).__name__})")

    def _upsert_request(self, conn, d, inp, now):
        intent = d.exit_request
        existing = conn.execute("SELECT exit_request_id FROM exit_requests WHERE request_key = ?",
                                (intent.request_key,)).fetchone()
        if existing:
            return existing["exit_request_id"]                     # same active request; no duplicate

        exit_request_id = f"XR-{uuid.uuid4().hex[:16]}"
        conn.execute(
            """
            INSERT INTO exit_requests (
                exit_request_id, request_key, trade_id, position_id, signal_id, proposal_key,
                risk_assessment_id, requested_action, requested_quantity, hard_exit,
                reason_codes_json, data_issue_codes_json, status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?)
            """,
            (exit_request_id, intent.request_key, d.trade_id, inp.position_id, inp.signal_id, inp.proposal_key,
             d.risk_assessment_id, intent.requested_action, intent.requested_quantity, 1 if intent.hard_exit else 0,
             json.dumps(intent.reason_codes), json.dumps(intent.data_issue_codes), now, now)
        )
        return exit_request_id

    @staticmethod
    def _event(conn, trade_id, from_state, to_state, action, codes, reasons, assessment_id, exit_request_id,
               details, now):
        conn.execute(
            """
            INSERT INTO trade_lifecycle_events (
                trade_id, from_state, to_state, action, reason_codes_json, reasons_json,
                risk_assessment_id, exit_request_id, details_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (trade_id, from_state, to_state, action, json.dumps(list(codes)), json.dumps(list(reasons)),
             assessment_id, exit_request_id, json.dumps(details, default=str), now)
        )


class _Stale(Exception):
    pass
