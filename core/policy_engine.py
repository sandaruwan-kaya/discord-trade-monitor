"""
Deterministic Policy Engine — authorizes (or refuses) a proposed paper trade.

    Agents recommend. The Policy Engine authorizes. The PaperBroker simulates.

Layers:
    evaluate(inp)            PURE: PolicyEvaluationInput -> PolicyDecision.
                             No I/O, no clock, no config reads: everything it
                             judges by is inside the input snapshot, so the
                             same snapshot always yields the same decision.
    gather_entry_input(...)  Builds the snapshot: DB state (read-only SQL),
                             a FRESH quote from the OptionChainProvider and
                             the current underlying price. Never a broker.
    persist_decision(...)    Idempotent audit write keyed by evaluation_key.

Idempotency design:
    evaluation_key = sha256(canonical JSON of the full input snapshot).
    - Re-running the SAME snapshot -> same key -> same decision; the
      audit row is written once (INSERT OR IGNORE on UNIQUE key).
    - A NEWER snapshot (fresh quote, later time, changed account) -> new
      key -> new decision row. Legitimate re-evaluation is never blocked.
    - decision_id is derived from evaluation_key, so an execution attempt
      can cite exactly the decision (and snapshot) that authorized it.
      Double-submission protection belongs to execution (client_order_id
      per proposal_key), not to the policy layer.

Kill switch:
    KILL_SWITCH_BLOCKS says which purposes the kill switch blocks. ENTRY is
    blocked; EXIT is explicitly NOT, so a future exit policy can always
    close risk. Exit evaluation itself is not implemented yet.

This module must never import an LLM client, an agent, or a broker.
"""
import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

import config
import database
import signal_rules
from schemas import (
    AccountSnapshot,
    KillSwitchState,
    OpenPositionSummary,
    OptionProposal,
    PendingEntrySummary,
    PolicyDecision,
    PolicyEvaluationInput,
    PolicySnapshot,
    RiskLimits,
    ThesisContext,
)
from signal_facts import build_input, compute_facts
from tools.options_tools import check_entry_extension


KILL_SWITCH_BLOCKS = {
    "ENTRY": True,
    "EXIT": False,     # risk-reducing actions must stay possible when halted
}

STRATEGY_FOR_DIRECTION = {"bullish": "LONG_CALL", "bearish": "LONG_PUT"}
OPTION_TYPE_FOR_STRATEGY = {"LONG_CALL": "call", "LONG_PUT": "put"}

TERMINAL_SIGNAL_REASONS = {
    "TARGET_HIT": "signal_target_hit",
    "INVALIDATED": "signal_invalidated",
    "EXPIRED": "signal_expired",
}

EXPIRY_BUCKETS = ("scalp", "intraday", "swing", "position", "unknown")


def _gt(value, limit):
    """value > limit, immune to float noise at the boundary."""
    return round(value, 9) > round(limit, 9)


def _lt(value, limit):
    return round(value, 9) < round(limit, 9)


def _iso(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat(timespec="seconds")


#
# Snapshot helpers (config / environment -> data)
#

def current_risk_limits() -> RiskLimits:
    return RiskLimits(
        allowed_underlyings=config.allowed_underlyings(),
        allowed_strategies=["LONG_CALL", "LONG_PUT"],
        max_contracts=config.risk("MAX_CONTRACTS"),
        max_premium_per_trade=config.risk("MAX_PREMIUM_PER_TRADE"),
        max_daily_loss=config.risk("MAX_DAILY_LOSS"),
        max_open_positions=config.risk("MAX_OPEN_POSITIONS"),
        min_dte=config.risk("MIN_DTE"),
        max_dte=config.risk("MAX_DTE"),
        min_abs_delta=config.risk("MIN_ABS_DELTA"),
        max_abs_delta=config.risk("MAX_ABS_DELTA"),
        max_spread_pct=config.risk("MAX_BID_ASK_SPREAD_PCT"),
        min_open_interest=config.risk("MIN_OPEN_INTEREST"),
        min_volume=config.risk("MIN_OPTION_VOLUME"),
        max_quote_age_seconds=config.risk("MAX_QUOTE_AGE_SECONDS"),
        fee_per_contract=config.risk("OPTION_FEE_PER_CONTRACT"),
        max_underlying_move_pct=config.agent2_max_underlying_move_pct(),
        max_target_progress_pct=config.agent1_max_target_progress_pct(),
        signal_expiry_hours={b: config.signal_expiry_hours(b) for b in EXPIRY_BUCKETS},
    )


def read_kill_switch() -> KillSwitchState:
    path = config.kill_switch_file()
    return KillSwitchState(
        env_enabled=config.kill_switch_env(),
        file_enabled=path.exists(),
        file_path=str(path),
    )


def evaluation_key(inp: PolicyEvaluationInput) -> str:
    canonical = json.dumps(inp.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


#
# Pure evaluation
#

def evaluate(inp: PolicyEvaluationInput) -> PolicyDecision:
    if inp.purpose != "ENTRY":
        raise NotImplementedError(
            "EXIT policy is a later step; KILL_SWITCH_BLOCKS already exempts exits"
        )

    reasons: list[str] = []

    def reject(code):
        if code not in reasons:
            reasons.append(code)

    lim = inp.limits
    p: OptionProposal = inp.proposal
    q = inp.quote

    # 1. Trading mode
    if inp.trading_mode != "paper":
        reject("trading_mode_not_paper")

    # 2. Kill switch (blocks entries only)
    if inp.kill_switch.active and KILL_SWITCH_BLOCKS[inp.purpose]:
        reject("kill_switch_enabled")

    # 3. Proposal validity
    if p.status != "PROPOSED":
        reject("proposal_not_proposed")
    if p.strategy not in ("LONG_CALL", "LONG_PUT"):
        reject("invalid_strategy")
    if p.side != "BUY_TO_OPEN":
        reject("invalid_side")
    if p.quantity is None or p.quantity < 1:
        reject("invalid_quantity")
    if p.limit_price is None or p.limit_price <= 0:
        reject("invalid_limit_price")
    if not p.contract_symbol:
        reject("missing_contract")

    # 4. Allowed underlying / 5. allowed strategy
    if p.underlying not in lim.allowed_underlyings:
        reject("underlying_not_allowed")
    if p.strategy is not None and p.strategy not in lim.allowed_strategies:
        reject("strategy_not_allowed")

    # 6. Max contracts
    if p.quantity is not None and p.quantity > lim.max_contracts:
        reject("max_contracts_exceeded")

    # 7. Max loss per trade = the order's own exposure:
    #    proposal.limit_price x fresh contract multiplier x qty + fees.
    #    A BUY limit can never pay more than its limit (the broker leaves it
    #    PENDING while ask > limit), so a fresh ask above the limit does NOT
    #    raise the max loss, and a lower ask does not lower it. The proposal
    #    is never repriced, re-limited or resized here.
    max_loss = None
    if p.limit_price and p.limit_price > 0 and p.quantity and p.quantity > 0:
        multiplier = q.contract_multiplier if q is not None else p.contract_multiplier
        if multiplier:
            max_loss = p.limit_price * multiplier * p.quantity + lim.fee_per_contract * p.quantity
            if _gt(max_loss, lim.max_premium_per_trade):
                reject("max_loss_per_trade_exceeded")

    # 8. Max daily loss (realized). At or beyond the limit blocks entries.
    if not _lt(-inp.account.realized_pnl_today, lim.max_daily_loss):
        reject("max_daily_loss_reached")

    # 9. Max open positions (open + pending entries count)
    if inp.account.open_position_count + inp.account.pending_entry_count >= lim.max_open_positions:
        reject("max_open_positions_reached")

    # 10. Duplicate exposure
    if p.signal_id is not None and (
        any(o.signal_id == p.signal_id for o in inp.open_positions)
        or any(e.signal_id == p.signal_id for e in inp.pending_entries)
    ):
        reject("duplicate_signal_position")
    if p.proposal_key in inp.filled_entry_proposal_keys:
        reject("duplicate_proposal_entry")

    # 11-17. Fresh quote revalidation
    quote_age = None
    if q is None:
        reject("quote_unavailable")
    else:
        # 17. Contract identity
        if (
            q.contract_symbol != p.contract_symbol
            or (p.option_type is not None and q.option_type != p.option_type)
            or (p.strike is not None and q.strike != p.strike)
            or (p.expiry is not None and q.expiry != p.expiry)
        ):
            reject("contract_mismatch")
        if q.underlying != p.underlying:
            reject("underlying_mismatch")
        # Same contract, different economics: never recalculate silently.
        if p.contract_multiplier is None or q.contract_multiplier != p.contract_multiplier:
            reject("contract_terms_changed")

        # 11. DTE
        if _lt(q.dte, lim.min_dte):
            reject("dte_below_min")
        if _gt(q.dte, lim.max_dte):
            reject("dte_above_max")

        # 12. Delta
        if q.delta is None:
            reject("delta_unavailable")
        else:
            if _lt(abs(q.delta), lim.min_abs_delta):
                reject("delta_below_min")
            if _gt(abs(q.delta), lim.max_abs_delta):
                reject("delta_above_max")

        # 13. Spread
        if not q.bid or not q.ask or q.bid <= 0 or q.ask < q.bid:
            reject("no_valid_quote")
        else:
            spread_pct = (q.ask - q.bid) / ((q.ask + q.bid) / 2) * 100
            if _gt(spread_pct, lim.max_spread_pct):
                reject("spread_too_wide")

        # 14-15. Liquidity
        if _lt(q.open_interest or 0, lim.min_open_interest):
            reject("open_interest_below_min")
        if _lt(q.volume_24h or 0, lim.min_volume):
            reject("volume_below_min")

        # 16. Quote freshness
        quote_age = max(0.0, (inp.now_ms - q.as_of_ms) / 1000)
        if _gt(quote_age, lim.max_quote_age_seconds):
            reject("quote_stale")

    # 18-23. Thesis re-checks at the CURRENT underlying price
    thesis: ThesisContext = inp.thesis

    if thesis.signal is None:
        reject("signal_not_found")
    else:
        status = thesis.signal_status
        if status in TERMINAL_SIGNAL_REASONS:
            reject(TERMINAL_SIGNAL_REASONS[status])
        elif status != "WORKING":
            reject("signal_not_working")
        if not thesis.signal_active and status not in TERMINAL_SIGNAL_REASONS:
            reject("signal_inactive")

        if thesis.agent1_state == "STALE":
            reject("agent1_stale")
        if thesis.agent1_state == "NOT_TRADEABLE":
            reject("agent1_not_tradeable")

        # Expiry from the snapshot's limits (not live config).
        bucket = signal_rules.classify_timeframe(thesis.signal.timeframe)
        expiry_ms = lim.signal_expiry_hours.get(bucket, lim.signal_expiry_hours["unknown"]) * 3_600_000
        if inp.now_ms - thesis.signal.signal_time_ms > expiry_ms:
            reject("signal_expired")

        if inp.underlying_price is None:
            reject("underlying_price_unavailable")
        else:
            signal = thesis.signal.model_copy(update={"current_price": inp.underlying_price})
            facts = compute_facts(signal, inp.now_ms)

            # 18. Direction / strategy / option-type consistency
            expected_strategy = STRATEGY_FOR_DIRECTION.get(facts.direction)
            if expected_strategy is None or facts.default_target is None:
                reject("thesis_not_tradeable")
            elif p.strategy is not None and p.strategy != expected_strategy:
                reject("direction_strategy_mismatch")
            if p.strategy in OPTION_TYPE_FOR_STRATEGY and p.option_type != OPTION_TYPE_FOR_STRATEGY[p.strategy]:
                reject("option_type_strategy_mismatch")
            if q is not None and p.strategy in OPTION_TYPE_FOR_STRATEGY and q.option_type != OPTION_TYPE_FOR_STRATEGY[p.strategy]:
                reject("option_type_strategy_mismatch")

            # 22. Invalidation / 23. target
            if facts.invalidation_hit:
                reject("invalidation_hit")
            if facts.target_reached:
                reject("target_reached")

            # 21. Target progress
            if (
                facts.entry_crossed
                and facts.target_progress_pct is not None
                and _gt(facts.target_progress_pct, lim.max_target_progress_pct)
            ):
                reject("target_progress_exceeded")

            # 20. Entry extension / chase protection (same guard as pre-Agent 2)
            if facts.direction in STRATEGY_FOR_DIRECTION:
                entry_ref = facts.default_entry_level
                if entry_ref is None:
                    entry_ref = thesis.signal.price_at_signal
                allowed, _, _ = check_entry_extension(
                    facts.direction, entry_ref, inp.underlying_price, lim.max_underlying_move_pct
                )
                if not allowed:
                    reject("stale_entry_move_extended")

    key = evaluation_key(inp)
    approved = not reasons

    return PolicyDecision(
        decision_id=f"PD-{key[:20]}",
        evaluation_key=key,
        purpose=inp.purpose,
        approved=approved,
        decision="APPROVED" if approved else "REJECTED",
        reasons=reasons,
        checked_at=_iso(inp.now_ms),
        proposal_key=p.proposal_key,
        signal_id=p.signal_id,
        policy_snapshot=PolicySnapshot(
            limits=lim, trading_mode=inp.trading_mode, kill_switch=inp.kill_switch
        ),
        calculated_max_loss=max_loss,
        calculated_total_open_risk=(
            inp.account.open_premium_at_risk + max_loss if max_loss is not None else None
        ),
        quote_age_seconds=quote_age,
    )


#
# Snapshot gathering (read-only; no broker object)
#

def _rows(conn, sql, params=()):
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def portfolio_state(now_ms):
    """Read paper-trading state straight from SQLite (read-only)."""
    day_start = datetime.fromtimestamp(now_ms / 1000, timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).isoformat(timespec="seconds")

    with closing(database.get_connection()) as conn:
        conn.row_factory = sqlite3.Row

        positions = [
            OpenPositionSummary(
                position_id=r["position_id"], signal_id=r["signal_id"], proposal_key=r["proposal_key"],
                contract_symbol=r["contract_symbol"], quantity=r["quantity"],
                entry_price=r["entry_price"], contract_multiplier=r["contract_multiplier"],
                premium_at_risk=r["entry_price"] * r["contract_multiplier"] * r["quantity"],
            )
            for r in _rows(conn, "SELECT * FROM paper_positions WHERE status = 'OPEN' ORDER BY position_id")
        ]
        unrealized = sum(
            r["unrealized_pnl"] or 0
            for r in _rows(conn, "SELECT unrealized_pnl FROM paper_positions WHERE status = 'OPEN'")
        )

        pending = []
        for r in _rows(conn, """
            SELECT o.*, (SELECT contract_multiplier FROM paper_positions p
                         WHERE p.contract_symbol = o.contract_symbol LIMIT 1) AS known_multiplier
            FROM paper_orders o
            WHERE o.side = 'BUY_TO_OPEN' AND o.status = 'PENDING'
            ORDER BY o.order_id
        """):
            multiplier = r["known_multiplier"] or 1.0      # unknown -> conservative (full premium)
            pending.append(PendingEntrySummary(
                order_id=r["order_id"], signal_id=r["signal_id"], proposal_key=r["proposal_key"],
                contract_symbol=r["contract_symbol"], quantity=r["quantity"], limit_price=r["limit_price"],
                premium_at_risk=r["limit_price"] * multiplier * r["quantity"],
            ))

        filled_keys = [
            r["proposal_key"] for r in _rows(conn, """
                SELECT DISTINCT proposal_key FROM paper_orders
                WHERE side = 'BUY_TO_OPEN' AND status = 'FILLED' AND proposal_key IS NOT NULL
                ORDER BY proposal_key
            """)
        ]

        realized_today = sum(
            (r["price"] - r["entry_price"]) * r["contract_multiplier"] * r["quantity"]
            for r in _rows(conn, """
                SELECT f.price, f.quantity, p.entry_price, p.contract_multiplier
                FROM paper_fills f
                JOIN paper_orders o ON o.order_id = f.order_id
                JOIN paper_positions p ON p.position_id = o.position_id
                WHERE f.side = 'SELL_TO_CLOSE' AND f.filled_at >= ?
            """, (day_start,))
        )

    account = AccountSnapshot(
        open_position_count=len(positions),
        pending_entry_count=len(pending),
        open_premium_at_risk=sum(x.premium_at_risk for x in positions) + sum(x.premium_at_risk for x in pending),
        realized_pnl_today=realized_today,
        unrealized_pnl=unrealized,
    )

    return account, positions, pending, filled_keys


def gather_entry_input(proposal: OptionProposal, chain_provider, market, now_ms) -> PolicyEvaluationInput:
    """Snapshot everything the ENTRY policy needs. Fetch failures become None (and reject)."""
    row = database.get_signal(proposal.signal_id) if proposal.signal_id is not None else None
    analysis = database.get_latest_signal_analysis(proposal.signal_id) if row else None
    agent1_state = json.loads(analysis["output_json"])["signal_state"] if analysis else None

    try:
        quote = chain_provider.get_option_contract(proposal.contract_symbol) if proposal.contract_symbol else None
    except Exception:
        quote = None

    try:
        underlying_price = market.get_current_price(proposal.underlying)
    except Exception:
        underlying_price = None

    account, positions, pending, filled_keys = portfolio_state(now_ms)

    return PolicyEvaluationInput(
        purpose="ENTRY",
        proposal=proposal,
        thesis=ThesisContext(
            signal=build_input(row) if row else None,
            signal_status=row["status"] if row else "MISSING",
            signal_active=bool(row["active"]) if row else False,
            agent1_state=agent1_state,
        ),
        underlying_price=underlying_price,
        quote=quote,
        account=account,
        open_positions=positions,
        pending_entries=pending,
        filled_entry_proposal_keys=filled_keys,
        now_ms=now_ms,
        limits=current_risk_limits(),
        trading_mode=config.trading_mode(),
        kill_switch=read_kill_switch(),
    )


#
# Audit persistence
#

def persist_decision(decision: PolicyDecision, inp: PolicyEvaluationInput) -> PolicyDecision:
    """Write once per evaluation_key; returns the stored decision."""
    with closing(database.get_connection()) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO policy_decisions (
                decision_id, evaluation_key, proposal_key, signal_id, purpose,
                approved, decision, reasons_json, calculated_max_loss,
                calculated_total_open_risk, quote_age_seconds, checked_at,
                input_json, policy_snapshot_json, decision_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.decision_id, decision.evaluation_key, decision.proposal_key,
                decision.signal_id, decision.purpose, 1 if decision.approved else 0,
                decision.decision, json.dumps(decision.reasons), decision.calculated_max_loss,
                decision.calculated_total_open_risk, decision.quote_age_seconds, decision.checked_at,
                inp.model_dump_json(), decision.policy_snapshot.model_dump_json(),
                decision.model_dump_json(), database.utc_now_iso(),
            )
        )
        conn.commit()

        stored = conn.execute(
            "SELECT decision_json FROM policy_decisions WHERE evaluation_key = ?",
            (decision.evaluation_key,)
        ).fetchone()

    return PolicyDecision.model_validate_json(stored[0])


def get_decision(decision_id):
    with closing(database.get_connection()) as conn:
        row = conn.execute(
            "SELECT decision_json FROM policy_decisions WHERE decision_id = ?", (decision_id,)
        ).fetchone()
    return PolicyDecision.model_validate_json(row[0]) if row else None


def evaluate_entry(proposal: OptionProposal, chain_provider, market, now_ms) -> PolicyDecision:
    """gather -> evaluate -> persist. Authorizes only; never submits anything."""
    inp = gather_entry_input(proposal, chain_provider, market, now_ms)
    return persist_decision(evaluate(inp), inp)


#
# EXIT policy — "can we safely execute the already-required risk reduction?"
#
# Deliberately NOT applied to exits: kill switch, max daily loss, max open
# positions, max premium, chase protection, target progress, entry strategy
# limits. A halted system must still be able to close risk.
#
# Reasons are split into:
#   structural (final for this execution): the exit as requested is not valid
#   transient  (retryable):                 the exit is valid, the data is not
#

EXIT_TRANSIENT_REASONS = frozenset({
    "quote_unavailable", "no_valid_bid", "quote_stale", "quote_terms_changed",
})


def current_exit_limits():
    from schemas import ExitPolicyLimits
    return ExitPolicyLimits(max_quote_age_seconds=config.risk("MAX_QUOTE_AGE_SECONDS"))


def evaluate_exit(inp) -> PolicyDecision:
    """Pure. ExitPolicyInput -> PolicyDecision (purpose EXIT)."""
    from schemas import ExitPolicySnapshot

    reasons: list[str] = []

    def reject(code):
        if code not in reasons:
            reasons.append(code)

    xr, trade, pos, q = inp.exit_request, inp.trade, inp.position, inp.quote

    if inp.trading_mode != "paper":
        reject("trading_mode_not_paper")

    # The exit requirement itself
    if xr.status != "ACTIVE":
        reject("exit_request_not_active")
    if xr.requested_action != "EXIT":
        reject("not_a_full_exit")
    if trade is None:
        reject("trade_missing")
    else:
        if trade.position_id != xr.position_id:
            reject("trade_position_mismatch")
        if trade.state not in ("EXIT_REQUESTED", "CLOSE_ORDER_PENDING"):
            reject("trade_not_exit_requested")

    # The position to close
    quantity = None
    if pos is None:
        reject("position_missing")
    else:
        if pos.position_id != xr.position_id:
            reject("position_mismatch")
        if pos.status != "OPEN":
            reject("position_not_open")
        if pos.quantity <= 0:
            reject("position_quantity_zero")
        elif xr.requested_quantity is not None and pos.quantity != xr.requested_quantity:
            reject("position_quantity_changed")
        else:
            quantity = pos.quantity

    # No other close already representing this exit
    if any(o.client_order_id != inp.close_client_order_id for o in inp.pending_close_orders):
        reject("close_order_already_exists")

    # Fresh quote (transient problems only)
    limit_price, quote_age = None, None
    if q is None:
        reject("quote_unavailable")
    else:
        if pos is not None and q.contract_symbol != pos.contract_symbol:
            reject("contract_mismatch")
        if pos is not None and q.contract_multiplier != pos.contract_multiplier:
            reject("quote_terms_changed")
        if q.bid is None or q.bid <= 0 or (q.ask is not None and q.ask < q.bid):
            reject("no_valid_bid")
        quote_age = max(0.0, (inp.now_ms - q.as_of_ms) / 1000)
        if _gt(quote_age, inp.limits.max_quote_age_seconds):
            reject("quote_stale")
        if q.bid is not None and q.bid > 0:
            limit_price = q.bid                        # deterministic: SELL_TO_CLOSE at the fresh BID

    key = evaluation_key(inp)
    approved = not reasons

    return PolicyDecision(
        decision_id=f"PD-{key[:20]}",
        evaluation_key=key,
        purpose="EXIT",
        approved=approved,
        decision="APPROVED" if approved else "REJECTED",
        reasons=reasons,
        checked_at=_iso(inp.now_ms),
        proposal_key=(trade.proposal_key if trade and trade.proposal_key else xr.request_key),
        signal_id=trade.signal_id if trade else None,
        policy_snapshot=ExitPolicySnapshot(limits=inp.limits, trading_mode=inp.trading_mode,
                                           kill_switch=inp.kill_switch),
        quote_age_seconds=quote_age,
        retryable=bool(reasons) and set(reasons) <= EXIT_TRANSIENT_REASONS,
        exit_limit_price=limit_price if approved else None,
        exit_quantity=quantity if approved else None,
    )


def gather_exit_input(exit_request_id, chain_provider, now_ms, close_client_order_id):
    """Read-only snapshot for the EXIT policy (no broker object)."""
    from schemas import (
        CloseOrderState,
        ExitPolicyInput,
        ExitPositionState,
        ExitRequestState,
        TradeStateSummary,
    )

    with closing(database.get_connection()) as conn:
        conn.row_factory = sqlite3.Row
        xr = conn.execute("SELECT * FROM exit_requests WHERE exit_request_id = ?", (exit_request_id,)).fetchone()
        if xr is None:
            return None
        xr = dict(xr)
        trade = conn.execute("SELECT * FROM trades WHERE trade_id = ?", (xr["trade_id"],)).fetchone()
        pos = conn.execute("SELECT * FROM paper_positions WHERE position_id = ?", (xr["position_id"],)).fetchone()
        pending = _rows(conn, """
            SELECT order_id, client_order_id, status, quantity, limit_price FROM paper_orders
            WHERE position_id = ? AND side = 'SELL_TO_CLOSE' AND status = 'PENDING' ORDER BY order_id
        """, (xr["position_id"],))

    quote, quote_error = None, None
    if pos is not None:
        try:
            quote = chain_provider.get_option_contract(pos["contract_symbol"])
        except Exception as error:
            quote_error = f"{type(error).__name__}: {str(error)[:200]}"

    return ExitPolicyInput(
        exit_request=ExitRequestState(
            exit_request_id=xr["exit_request_id"], request_key=xr["request_key"], trade_id=xr["trade_id"],
            position_id=xr["position_id"], requested_action=xr["requested_action"],
            requested_quantity=xr["requested_quantity"], status=xr["status"], hard_exit=bool(xr["hard_exit"]),
            data_issue_codes=json.loads(xr["data_issue_codes_json"]),
        ),
        trade=TradeStateSummary(trade_id=trade["trade_id"], state=trade["state"], position_id=trade["position_id"],
                                proposal_key=trade["proposal_key"], signal_id=trade["signal_id"]) if trade else None,
        position=ExitPositionState(position_id=pos["position_id"], status=pos["status"], quantity=pos["quantity"],
                                   contract_symbol=pos["contract_symbol"],
                                   contract_multiplier=pos["contract_multiplier"],
                                   entry_price=pos["entry_price"]) if pos else None,
        quote=quote,
        quote_error=quote_error,
        close_client_order_id=close_client_order_id,
        pending_close_orders=[CloseOrderState(**r) for r in pending],
        now_ms=now_ms,
        limits=current_exit_limits(),
        trading_mode=config.trading_mode(),
        kill_switch=read_kill_switch(),
    )


#
# RESTING ENTRY validity — should a still-PENDING entry order stay alive?
#
# Pure and deterministic. A raw status flip (e.g. WORKING -> DEVELOPING on a
# small pull-back below the entry level) is NOT a reason by itself; only a
# genuinely invalid entry opportunity is.
#

def resting_entry_cancel_reasons(
    proposal: OptionProposal,
    signal,                     # SignalAnalysisInput or None
    signal_status,
    agent1_state,
    underlying_price,
    order_submitted_ms,
    now_ms,
    limits: RiskLimits,
    kill_switch: KillSwitchState,
    max_age_minutes,
) -> list[str]:
    reasons: list[str] = []

    def add(code):
        if code not in reasons:
            reasons.append(code)

    if _gt((now_ms - order_submitted_ms) / 60_000, max_age_minutes):
        add("ENTRY_ORDER_EXPIRED")
    if kill_switch.active:
        add("KILL_SWITCH_ENABLED")

    if signal is None:
        add("SIGNAL_MISSING")
        return reasons

    if signal_status == "INVALIDATED":
        add("SIGNAL_INVALIDATED")
    elif signal_status == "TARGET_HIT":
        add("TARGET_ALREADY_REACHED")
    elif signal_status == "EXPIRED":
        add("SIGNAL_EXPIRED")

    if agent1_state == "STALE":
        add("THESIS_STALE")
    elif agent1_state == "NOT_TRADEABLE":
        add("THESIS_NOT_TRADEABLE")

    bucket = signal_rules.classify_timeframe(signal.timeframe)
    expiry_ms = limits.signal_expiry_hours.get(bucket, limits.signal_expiry_hours["unknown"]) * 3_600_000
    if now_ms - signal.signal_time_ms > expiry_ms:
        add("SIGNAL_EXPIRED")

    if underlying_price is None:
        return reasons                          # cannot judge the price rules; age/kill switch still apply

    facts = compute_facts(signal.model_copy(update={"current_price": underlying_price}), now_ms)

    if facts.direction not in STRATEGY_FOR_DIRECTION or facts.default_target is None:
        add("THESIS_NOT_TRADEABLE")
        return reasons
    if facts.invalidation_hit:
        add("SIGNAL_INVALIDATED")
    if facts.target_reached:
        add("TARGET_ALREADY_REACHED")
    if (facts.entry_crossed and facts.target_progress_pct is not None
            and _gt(facts.target_progress_pct, limits.max_target_progress_pct)):
        add("TARGET_PROGRESS_EXCEEDED")

    entry_ref = facts.default_entry_level if facts.default_entry_level is not None else signal.price_at_signal
    allowed, _, _ = check_entry_extension(facts.direction, entry_ref, underlying_price, limits.max_underlying_move_pct)
    if not allowed:
        add("ENTRY_STALE")

    return reasons
