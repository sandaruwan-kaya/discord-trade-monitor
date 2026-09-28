"""
Paper-trading pipeline wiring. Each cycle is small, flag-gated, and only
DELEGATES to the component that owns the work:

    loop B  proposal_entry_cycle   Agent 2 sweep (signal_service) -> ExecutionService.execute_entry
                                   + pending-entry reconciliation + Trade materialization (Agent 4 helper)
    loop C  risk_cycle             Agent 3 (RiskService) for OPEN trades only
    loop D  lifecycle_cycle        Agent 4 (TradeManager) applies each assessment once, creates ExitRequests
    loop E  exit_cycle             ExecutionService.execute_exit for ACTIVE full EXIT requests

Ownership is never blurred: this module holds no trading logic, calls no
LLM itself, and never touches the broker except through ExecutionService.

Notifications are sent AFTER the state they describe is committed, through
the persisted, idempotent Notifier (restarts never resend).
"""
import json
import sqlite3
import time
import traceback
from contextlib import closing
from datetime import datetime, timezone

import config
import database
from agents.trade_manager import TradeManager
from core.execution_service import ExecutionService
from risk_service import RiskService


def _iso_to_ms(value):
    return int(datetime.fromisoformat(value).timestamp() * 1000)


def _rows(sql, params=()):
    with closing(database.get_connection()) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, params)]


class TradingPipeline:

    def __init__(self, signal_services, broker, chain_provider, market, risk_agent, notifier,
                 context_market=None, now_ms=None, log=print):
        self.services = signal_services
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))
        self._log = log
        self.notifier = notifier
        self.execution = ExecutionService(broker, chain_provider, market, now_ms=self._now_ms, log=log)
        self.trades = TradeManager(now_ms=self._now_ms, log=log)
        self.risk = RiskService(risk_agent, chain_provider, market, context_market=context_market, now_ms=self._now_ms)
        self.stopping = False

    #
    # Logging / notifications
    #

    def log(self, tag, message, **ids):
        suffix = " ".join(f"{k}={v}" for k, v in ids.items() if v is not None)
        self._log(f"[{tag}] {message}" + (f" {suffix}" if suffix else ""))

    def _notify(self, key, kind, text):
        try:
            self.notifier.notify(key, kind, text)
        except Exception as error:                      # notifications never break trading state
            self.log("NOTIFY", f"notification bookkeeping failed: {type(error).__name__}", key=key)

    #
    # Startup reconciliation (before any new entry can be created)
    #

    def startup_reconcile(self):
        if not config.agentic_trading_enabled():
            return {"skipped": "AGENTIC_TRADING_ENABLED=false"}

        summary = {}
        summary["interrupted_proposals"] = database.fail_interrupted_proposals(database.utc_now_iso())
        summary["cancelled_entries"] = [(r.execution_key, r.status) for r in self.cancel_invalid_entries()]
        summary["entry_executions"] = [
            (r.execution_key, r.status)
            for r in self.execution.reconcile_pending_executions(process_pending=True, purpose="ENTRY")
        ]
        summary["exit_executions"] = [
            (r.execution_key, r.status)
            for r in self.execution.reconcile_pending_executions(
                process_pending=False, purpose="EXIT", resume_exits=config.auto_exit_enabled())
        ]
        summary["materialized_trades"] = self.materialize_filled_entries()
        if config.agent4_enabled():
            summary["lifecycle_decisions"] = len(self.trades.process_all())
        self.notify_trade_scans()
        self.log("RECONCILE", f"startup reconciliation done: {json.dumps(summary, default=str)[:500]}")
        return summary

    #
    # Loop B: proposals + entries
    #

    def after_proposals(self, finished):
        """Notify newly created proposals (called with run_proposal_cycle's result)."""
        for proposal_key, status in finished or []:
            row = database.get_proposal(proposal_key)
            self.log("AGENT2", f"proposal {status}", proposal_key=proposal_key)
            if status != "PROPOSED" or not row or not row["output_json"]:
                continue
            p = json.loads(row["output_json"])
            self._notify(
                f"proposal:{proposal_key}", "proposal",
                "🧾 OPTION PROPOSAL CREATED (paper)\n\n"
                f"Underlying: {p['underlying']}\nStrategy: {p['strategy']}\nContract: {p['contract_symbol']}\n"
                f"Quantity: {p['quantity']}\nLimit: {p['limit_price']}\nMax loss: ${p['estimated_max_loss']:.2f}\n"
                f"Target / invalidation: {p['thesis_target']} / {p['thesis_invalidation']}"
            )

    def cancel_invalid_entries(self):
        """Before any fill is attempted: cancel resting entries that are no longer valid."""
        out = []
        for r in self.execution.cancel_invalid_pending_entries():
            out.append(r)
            if r.status == "ORDER_CANCELLED":
                self._notify_entry_cancelled(r)
            else:
                self._notify_entry(r)
        return out

    def _notify_entry_cancelled(self, r):
        attempt = self.execution.get_attempt(r.execution_key)
        result = json.loads(attempt["result_json"] or "{}")
        order = result.get("broker_order", {})
        reasons = result.get("cancel_reasons", [])
        age = ""
        if order.get("submitted_at"):
            age = f"{(self._now_ms() - _iso_to_ms(order['submitted_at'])) / 60_000:.0f} min"
        prop = database.get_proposal(r.proposal_key) or {}
        underlying = json.loads(prop["output_json"])["underlying"] if prop.get("output_json") else "?"
        self.log("ENTRY", f"resting entry cancelled {reasons}", proposal_key=r.proposal_key,
                 execution_id=r.execution_id, order=r.broker_order_id)
        self._notify(f"entry-cancelled:{r.execution_id}", "entry",
                     "🚫 PAPER ENTRY CANCELLED\n\n"
                     f"Underlying: {underlying}\nContract: {order.get('contract_symbol')}\n"
                     f"Limit: {order.get('limit_price')}\nOrder age: {age}\nReason: {', '.join(reasons)}")

    def entry_cycle(self):
        # Risk-reducing, so it runs whenever the pipeline runs (even with entries disabled).
        self.cancel_invalid_entries()

        if not config.paper_execution_enabled():
            return []

        results = []
        for r in self.execution.reconcile_pending_executions(process_pending=True, purpose="ENTRY"):
            self._notify_entry(r)
            results.append(r)

        window_ms = config.paper_entry_max_proposal_age_seconds() * 1000
        candidates = _rows(
            """
            SELECT p.proposal_key, p.updated_at FROM trade_proposals p
            WHERE p.status = 'PROPOSED'
              AND NOT EXISTS (SELECT 1 FROM execution_attempts e WHERE e.execution_key = 'entry:' || p.proposal_key)
            ORDER BY p.updated_at
            """
        )
        for c in candidates:
            if self.stopping:
                break
            # updated_at is wall-clock (database layer), so compare with the wall clock.
            if time.time() * 1000 - _iso_to_ms(c["updated_at"]) > window_ms:
                self.log("ENTRY", "proposal older than the execution window; not executed",
                         proposal_key=c["proposal_key"])
                continue                                    # stale backlog is never executed
            r = self.execution.execute_entry(c["proposal_key"])
            self.log("ENTRY", f"execute_entry -> {r.status}", proposal_key=c["proposal_key"],
                     execution_id=r.execution_id, order=r.broker_order_id)
            self._notify_entry(r)
            results.append(r)

        self.materialize_filled_entries()
        self.notify_trade_scans()
        return results

    def _notify_entry(self, r):
        if r is None or not r.execution_id:
            return
        if r.status == "POLICY_REJECTED":
            self.log("POLICY", f"entry rejected: {r.policy_reasons}", proposal_key=r.proposal_key,
                     execution_id=r.execution_id)
            self._notify(f"entry-rejected:{r.execution_id}", "entry",
                         f"⛔ ENTRY POLICY REJECTED (paper)\n\n{r.proposal_key}\nReasons: {', '.join(r.policy_reasons)}")
        elif r.status == "ORDER_PENDING":
            self._notify(f"entry-pending:{r.execution_id}", "entry",
                         f"⏳ PAPER ENTRY ORDER PENDING\n\n{r.proposal_key}\nOrder: {r.broker_order_id}")
        elif r.status == "ORDER_FILLED":
            self._notify(f"entry-filled:{r.execution_id}", "entry",
                         f"✅ PAPER ENTRY ORDER FILLED\n\n{r.proposal_key}\nOrder: {r.broker_order_id}")

    def materialize_filled_entries(self):
        created = []
        for row in _rows(
            """
            SELECT e.execution_key FROM execution_attempts e
            WHERE e.purpose = 'ENTRY' AND e.status = 'ORDER_FILLED'
              AND NOT EXISTS (SELECT 1 FROM trades t WHERE t.execution_id = e.execution_id)
            """
        ):
            trade = self.trades.materialize_trade(row["execution_key"])
            if trade:
                created.append(trade["trade_id"])
                self.log("TRADE", "materialized", trade_id=trade["trade_id"], position_id=trade["position_id"])
        return created

    #
    # Loop C: risk
    #

    def risk_cycle(self):
        if not config.agent3_enabled():
            return []
        out = []
        for t in _rows(
            """
            SELECT t.trade_id, t.position_id FROM trades t
            JOIN paper_positions p ON p.position_id = t.position_id
            WHERE p.status = 'OPEN' AND t.state NOT IN ('CLOSED', 'REJECTED', 'CANCELLED', 'FAILED')
            ORDER BY t.created_at
            """
        ):
            try:
                a = self.risk.assess_position(t["position_id"])
            except Exception as error:
                self.log("RISK", f"assessment failed: {type(error).__name__}: {str(error)[:200]}",
                         trade_id=t["trade_id"])
                continue
            if a is not None:
                self.log("RISK", f"{a.recommendation} ({a.risk_state}) hard={a.hard_rule_codes} data={a.data_issue_codes}",
                         trade_id=t["trade_id"], assessment_id=a.assessment_id)
                out.append(a)
        return out

    #
    # Loop D: lifecycle
    #

    def lifecycle_cycle(self):
        if not config.agent4_enabled():
            return []
        self.materialize_filled_entries()
        decisions = self.trades.process_all()
        for d in decisions:
            if d.next_state == d.previous_state and d.action == "NONE":
                continue
            self.log("TRADE", f"{d.previous_state} -> {d.next_state} {d.action} {d.reason_codes}",
                     trade_id=d.trade_id, assessment_id=d.risk_assessment_id)
            if d.next_state == "AT_RISK" and d.previous_state != "AT_RISK":
                self._notify(f"at-risk:{d.trade_id}:{d.risk_assessment_id}", "risk",
                             f"⚠️ PAPER POSITION AT RISK\n\nTrade {d.trade_id}\n{'; '.join(d.reasons)}")
            if d.action in ("REQUEST_REDUCE", "REQUEST_EXIT"):
                for req in self.trades.get_exit_requests(d.trade_id):
                    if req.risk_assessment_id != d.risk_assessment_id:
                        continue
                    kind = "reduce" if req.requested_action == "REDUCE" else "exit-request"
                    self._notify(
                        f"{kind}:{req.exit_request_id}", kind,
                        ("🔻 REDUCE REQUESTED (advisory, not executed in V1)" if kind == "reduce"
                         else f"🚪 EXIT REQUESTED ({'hard rule' if req.hard_exit else 'contextual'})")
                        + f"\n\nTrade {d.trade_id}\nReasons: {', '.join(req.reason_codes)}"
                        + (f"\nData issues: {', '.join(req.data_issue_codes)}" if req.data_issue_codes else "")
                    )
        self.notify_trade_scans()
        return decisions

    #
    # Loop E: exits (full EXIT only; REDUCE is never executed)
    #

    def exit_cycle(self):
        if not config.auto_exit_enabled():
            # Exits may be requested and persisted, never executed. Only reconcile
            # already-submitted closes against the broker (no submission).
            self.execution.reconcile_pending_executions(process_pending=False, purpose="EXIT", resume_exits=False)
            return []

        results = []
        for req in _rows(
            """
            SELECT exit_request_id FROM exit_requests
            WHERE status = 'ACTIVE' AND requested_action = 'EXIT' ORDER BY created_at
            """
        ):
            if self.stopping:
                break
            r = self.execution.execute_exit(req["exit_request_id"])
            results.append(r)
            self.log("EXIT", f"execute_exit -> {r.status} {r.policy_reasons or ''}",
                     exit_request_id=req["exit_request_id"], execution_id=r.execution_id, order=r.broker_order_id)
            if r.status == "WAITING_FOR_QUOTE":
                self._notify(f"exit-wait:{r.execution_id}", "exit",
                             f"⏸️ EXIT WAITING FOR QUOTE\n\nExit request {req['exit_request_id']}\n"
                             f"Reasons: {', '.join(r.policy_reasons)}\nThe exit remains required and will be retried.")
            elif r.status == "ORDER_PENDING":
                self._notify(f"close-pending:{r.execution_id}", "exit",
                             f"⏳ CLOSE ORDER PENDING\n\nExit request {req['exit_request_id']}\nOrder: {r.broker_order_id}")
            elif r.status == "POLICY_REJECTED":
                self._notify(f"exit-rejected:{r.execution_id}", "exit",
                             f"🛑 EXIT BLOCKED (needs operator)\n\nExit request {req['exit_request_id']}\n"
                             f"Reasons: {', '.join(r.policy_reasons)}")
        self.notify_trade_scans()
        return results

    #
    # Scans for state-derived notifications (idempotent by key)
    #

    def notify_trade_scans(self):
        for t in _rows(
            """
            SELECT t.trade_id, t.proposal_key, p.contract_symbol, p.quantity, p.original_quantity, p.entry_price
            FROM trades t JOIN paper_positions p ON p.position_id = t.position_id
            """
        ):
            self._notify(f"opened:{t['trade_id']}", "position",
                         f"📈 PAPER POSITION OPENED\n\n{t['contract_symbol']} x{t['original_quantity']} @ {t['entry_price']}\n"
                         f"Trade {t['trade_id']} ({t['proposal_key']})")

        for t in _rows(
            """
            SELECT t.trade_id, p.position_id, p.contract_symbol, p.entry_price, p.original_quantity, p.realized_pnl,
                   p.contract_multiplier,
                   (SELECT fill_price FROM paper_orders o WHERE o.position_id = p.position_id
                      AND o.side = 'SELL_TO_CLOSE' AND o.status = 'FILLED' ORDER BY o.filled_at DESC LIMIT 1) AS exit_price,
                   (SELECT reason_codes_json FROM exit_requests x WHERE x.position_id = p.position_id
                      AND x.requested_action = 'EXIT' ORDER BY x.created_at DESC LIMIT 1) AS reasons
            FROM trades t JOIN paper_positions p ON p.position_id = t.position_id
            WHERE t.state = 'CLOSED'
            """
        ):
            reasons = ", ".join(json.loads(t["reasons"])) if t["reasons"] else "BROKER_POSITION_CLOSED"
            cost = t["entry_price"] * t["contract_multiplier"] * t["original_quantity"]
            pct = t["realized_pnl"] / cost * 100 if cost else 0
            self._notify(f"closed:{t['trade_id']}", "position",
                         f"🏁 PAPER POSITION CLOSED\n\n{t['contract_symbol']} x{t['original_quantity']}\n"
                         f"Entry: {t['entry_price']}  Exit: {t['exit_price']}\n"
                         f"Realized P&L: ${t['realized_pnl']:.2f} ({pct:+.1f}%)\nClose reason: {reasons}")


def run_cycle_safely(name, fn, log=print):
    """One loop iteration; a failure is logged and never propagates."""
    try:
        return fn()
    except Exception as error:
        log(f"[{name}] cycle failed: {type(error).__name__}: {str(error)[:300]}")
        traceback.print_exc()
        return None
