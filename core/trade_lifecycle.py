"""
Trade lifecycle state machine (Agent 4 core). Pure and deterministic:
no LLM, no broker, no execution service, no I/O, no clock.

    Agent 3: "what is the current risk recommendation?"
    Agent 4: "what should happen to the trade lifecycle now?"
    Execution (later): "how do we safely execute the requested action?"

decide() maps (current state, broker position/order state, latest Agent 3
assessment, active exit requests) -> TradeManagementDecision, optionally
with an ExitRequestIntent. It never submits or closes anything.

Exit priority (V1):
    - any EXIT (hard or contextual) -> EXIT_REQUESTED and one active
      full-exit request (exit:{position_id}:full)
    - once EXIT_REQUESTED, HOLD / REDUCE / DATA_UNAVAILABLE never move the
      trade back; later EXITs only add evidence
    - REDUCE is advisory: requested_quantity is None (never LLM-derived)
Fails closed: inconsistent or malformed inputs produce no transition.
"""
from datetime import datetime, timezone

import config
from schemas import (
    RISK_REASON_CODES,
    ExitRequestIntent,
    LifecycleLimits,
    TradeManagementDecision,
    TradeManagementInput,
)


TERMINAL_STATES = frozenset({"CLOSED", "REJECTED", "CANCELLED", "FAILED"})

MANAGED_STATES = frozenset({"OPEN", "WORKING", "AT_RISK", "REDUCE_REQUESTED",
                            "EXIT_REQUESTED", "CLOSE_ORDER_PENDING"})

TRANSITIONS: dict[str, frozenset[str]] = {
    "PROPOSED": frozenset({"POLICY_APPROVED", "REJECTED", "CANCELLED", "FAILED"}),
    "POLICY_APPROVED": frozenset({"ORDER_SUBMITTED", "CANCELLED", "FAILED"}),
    "ORDER_SUBMITTED": frozenset({"ORDER_PENDING", "OPEN", "REJECTED", "FAILED"}),
    "ORDER_PENDING": frozenset({"OPEN", "CANCELLED", "REJECTED", "FAILED"}),
    "OPEN": frozenset({"WORKING", "AT_RISK", "REDUCE_REQUESTED", "EXIT_REQUESTED", "CLOSED"}),
    "WORKING": frozenset({"AT_RISK", "REDUCE_REQUESTED", "EXIT_REQUESTED", "CLOSED"}),
    "AT_RISK": frozenset({"WORKING", "REDUCE_REQUESTED", "EXIT_REQUESTED", "CLOSED"}),
    "REDUCE_REQUESTED": frozenset({"WORKING", "EXIT_REQUESTED", "CLOSED"}),
    "EXIT_REQUESTED": frozenset({"CLOSE_ORDER_PENDING", "CLOSED"}),
    "CLOSE_ORDER_PENDING": frozenset({"CLOSED", "EXIT_REQUESTED"}),   # close order cancelled/rejected: exit still required
    "CLOSED": frozenset(),
    "REJECTED": frozenset(),
    "CANCELLED": frozenset(),
    "FAILED": frozenset(),
}

ALL_STATES = frozenset(TRANSITIONS)


class IllegalTransition(ValueError):
    pass


def is_transition_allowed(current: str, nxt: str) -> bool:
    """Staying in the same (valid) state is a no-op and always allowed."""
    if current not in ALL_STATES or nxt not in ALL_STATES:
        return False
    return current == nxt or nxt in TRANSITIONS[current]


def assert_transition(current: str, nxt: str) -> None:
    if not is_transition_allowed(current, nxt):
        raise IllegalTransition(f"illegal lifecycle transition {current} -> {nxt}")


def current_limits() -> LifecycleLimits:
    return LifecycleLimits(
        at_risk_invalidation_distance_pct=config.agent4_at_risk_invalidation_distance_pct(),
        at_risk_pnl_pct=config.agent4_at_risk_pnl_pct(),
    )


def full_exit_key(position_id):
    return f"exit:{position_id}:full"


def reduce_key(position_id, assessment_id):
    return f"reduce:{position_id}:{assessment_id}"


def _iso(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat(timespec="seconds")


def decide(inp: TradeManagementInput) -> TradeManagementDecision:
    state = inp.current_state
    a = inp.latest_risk_assessment

    def out(next_state=None, action="NONE", codes=(), reasons=(), allowed=True, applies_assessment=True, **extra):
        nxt = state if next_state is None else next_state
        if allowed and not is_transition_allowed(state, nxt):
            return out(None, "NONE", ["ILLEGAL_TRANSITION"],
                       [f"refused illegal transition {state} -> {nxt}"], allowed=False)
        return TradeManagementDecision(
            trade_id=inp.trade_id,
            previous_state=state,
            next_state=nxt if allowed else state,
            action=action if allowed else "NONE",
            reason_codes=list(codes),
            reasons=list(reasons),
            # Only decisions DRIVEN by the assessment cite it (it is applied once per trade).
            risk_assessment_id=a.assessment_id if (a and applies_assessment and allowed) else None,
            transition_allowed=allowed,
            decided_at=_iso(inp.now_ms),
            **(extra if allowed else {}),
        )

    # ---- integrity: fail closed -------------------------------------------------
    if state not in ALL_STATES:
        return out(codes=["INVALID_TRADE_STATE"], reasons=[f"unknown trade state {state!r}"], allowed=False)

    if state in TERMINAL_STATES:
        return out(codes=["TERMINAL_STATE"], reasons=[f"{state} is terminal; nothing to do"], applies_assessment=False)

    if state not in MANAGED_STATES:
        return out(codes=["STATE_NOT_MANAGED"], reasons=[f"{state} is before the position opened"], allowed=False)

    if inp.position is None:
        return out(action="WAIT_FOR_DATA", codes=["BROKER_POSITION_MISSING"],
                   reasons=["broker position not found; no lifecycle change"], applies_assessment=False)

    pos = inp.position
    if pos.position_id != inp.position_id:
        return out(codes=["POSITION_MISMATCH"], reasons=["broker position does not belong to this trade"], allowed=False)
    if pos.status not in ("OPEN", "CLOSED") or (pos.status == "OPEN" and pos.quantity <= 0) or pos.quantity < 0:
        return out(codes=["POSITION_INCONSISTENT"],
                   reasons=[f"broker position {pos.status} with quantity {pos.quantity}"], allowed=False)
    if a is not None and a.position_id != inp.position_id:
        return out(codes=["ASSESSMENT_POSITION_MISMATCH"],
                   reasons=["risk assessment refers to a different position"], allowed=False)

    active_exit = inp.active_exit_request
    active_reduce = inp.active_reduce_request

    # ---- reconciliation with the broker (before any assessment) -----------------
    if pos.status == "CLOSED":
        resolve = {}
        if active_exit:
            resolve[active_exit.exit_request_id] = "COMPLETED"
        if active_reduce:
            resolve[active_reduce.exit_request_id] = "SUPERSEDED"
        return out("CLOSED", "MARK_CLOSED", ["BROKER_POSITION_CLOSED"],
                   ["broker position is closed; lifecycle reconciled"], applies_assessment=False,
                   resolve_requests=resolve)

    close = inp.close_order
    if state == "EXIT_REQUESTED" and close and close.side == "SELL_TO_CLOSE" and close.status == "PENDING":
        return out("CLOSE_ORDER_PENDING", "NONE", ["CLOSE_ORDER_PENDING"], [f"close order {close.order_id} is resting"],
                   applies_assessment=False)
    if state == "CLOSE_ORDER_PENDING" and (close is None or close.status in ("CANCELLED", "REJECTED")):
        return out("EXIT_REQUESTED", "NONE", ["CLOSE_ORDER_ENDED_POSITION_OPEN"],
                   ["close order ended without closing the position; exit still required"], applies_assessment=False)

    # ---- risk assessment -----------------------------------------------------------
    if a is None or inp.assessment_already_processed:
        return out(codes=["NO_NEW_ASSESSMENT"], reasons=["no new risk assessment to apply"], applies_assessment=False)

    rec = a.recommendation
    contextual_codes = [r for r in a.reasons if r in RISK_REASON_CODES]

    if state in ("EXIT_REQUESTED", "CLOSE_ORDER_PENDING"):
        if rec == "EXIT":
            return out(codes=["EXIT_ALREADY_REQUESTED"],
                       reasons=["exit already requested; assessment recorded as evidence"],
                       evidence_for_exit_request_id=active_exit.exit_request_id if active_exit else None)
        action = "WAIT_FOR_DATA" if rec == "DATA_UNAVAILABLE" else "NONE"
        return out(action=action, codes=["EXIT_REQUEST_RETAINED"],
                   reasons=[f"Agent 3 {rec} cannot cancel or weaken an active exit request"])

    if rec == "EXIT":
        codes = a.hard_rule_codes if a.hard_rule_triggered else contextual_codes + ["AGENT3_CONTEXTUAL_EXIT"]
        intent = ExitRequestIntent(
            request_key=full_exit_key(inp.position_id),
            requested_action="EXIT",
            requested_quantity=pos.quantity,                      # full open quantity (V1)
            hard_exit=a.hard_rule_triggered,
            reason_codes=codes,
            data_issue_codes=a.data_issue_codes,
        )
        resolve = {active_reduce.exit_request_id: "SUPERSEDED"} if active_reduce else {}
        reasons = ["hard exit rule triggered" if a.hard_rule_triggered else "contextual exit recommended by Agent 3"]
        if a.data_issue_codes:
            reasons.append(f"exit required; execution may need to wait for usable data ({', '.join(a.data_issue_codes)})")
        return out("EXIT_REQUESTED", "REQUEST_EXIT", codes, reasons, exit_request=intent, resolve_requests=resolve)

    if rec == "REDUCE":
        if state == "REDUCE_REQUESTED" and active_reduce:
            return out(codes=["REDUCE_ALREADY_REQUESTED"], reasons=["advisory reduce already active"])
        intent = ExitRequestIntent(
            request_key=reduce_key(inp.position_id, a.assessment_id),
            requested_action="REDUCE",
            requested_quantity=None,                              # advisory; no LLM-derived quantity
            hard_exit=False,
            reason_codes=contextual_codes or ["AGENT3_REDUCE"],
            data_issue_codes=a.data_issue_codes,
        )
        return out("REDUCE_REQUESTED", "REQUEST_REDUCE", intent.reason_codes,
                   ["advisory reduce requested (no quantity in V1)"], exit_request=intent)

    if rec == "DATA_UNAVAILABLE":
        return out(action="WAIT_FOR_DATA", codes=["DATA_UNAVAILABLE"] + a.data_issue_codes,
                   reasons=["market data unusable; lifecycle state retained"])

    # HOLD
    lim = inp.limits
    at_risk = (
        (a.distance_to_invalidation_pct is not None
         and a.distance_to_invalidation_pct <= lim.at_risk_invalidation_distance_pct)
        or (a.unrealized_pnl_pct is not None and a.unrealized_pnl_pct <= -lim.at_risk_pnl_pct)
        or "NEAR_INVALIDATION" in contextual_codes
    )

    if state == "REDUCE_REQUESTED":
        # Only a FRESH contextual LLM HOLD may clear an advisory REDUCE.
        # Throttled / fallback / deterministic HOLDs carry no new judgement.
        if not a.fresh_contextual:
            return out(codes=["REDUCE_RETAINED_HOLD_NOT_FRESH"],
                       reasons=[f"HOLD is {a.risk_state}/{a.llm_status}, not a fresh contextual analysis; "
                                "advisory reduce kept"])
        resolve = {active_reduce.exit_request_id: "SUPERSEDED"} if active_reduce else {}
        return out("WORKING", "NONE", ["HOLD_AFTER_REDUCE"], ["fresh Agent 3 HOLD; advisory reduce superseded"],
                   resolve_requests=resolve)

    if at_risk:
        return out("AT_RISK", "NONE", ["HOLD_AT_RISK"], ["HOLD, but close to invalidation or deep in loss"])
    return out("WORKING", "NONE", ["HOLD"], ["Agent 3 HOLD"])
