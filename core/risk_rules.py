"""
Deterministic hard-risk rules for an OPEN long-option paper position.

No LLM, no broker, no execution, no network, no clock: everything comes
from the RiskAnalysisInput snapshot.

PnL for risk decisions uses the option BID (a long option is sold to close
at the bid). Mark/mid is computed for display only.

Hard EXIT codes (any one -> EXIT, LLM never consulted):
    SIGNAL_INVALIDATED, SIGNAL_TARGET_HIT        evaluator status (persisted,
                                                 chronologically resolved)
    UNDERLYING_INVALIDATION_HIT, UNDERLYING_TARGET_HIT
    AMBIGUOUS_INTRABAR_ORDER                     target AND invalidation inside
                                                 the same coarse candle, order
                                                 unknown and not resolved by the
                                                 evaluator: exit, cause undetermined
    OPTION_EXPIRED, MAX_HOLD_EXCEEDED
    HARD_PREMIUM_STOP, MAX_POSITION_LOSS         need a valid bid

Data issues (no EXIT rule active -> DATA_UNAVAILABLE, LLM never consulted):
    QUOTE_UNAVAILABLE, QUOTE_CONTRACT_MISMATCH, QUOTE_NO_BID, QUOTE_CROSSED,
    QUOTE_STALE, QUOTE_TERMS_CHANGED, UNDERLYING_PRICE_UNAVAILABLE

Thesis-based exits do not depend on the option quote, so they still fire
when the quote is unusable; quote-based stops only run on a valid quote.
"""
import hashlib
import json

import config
from schemas import RiskAnalysisInput, RiskMetrics, RiskRuleLimits, LastLLMAssessment


HOUR_MS = 3_600_000


def current_limits() -> RiskRuleLimits:
    return RiskRuleLimits(
        hard_premium_stop_pct=config.risk("HARD_PREMIUM_STOP_PCT"),
        max_position_loss=config.risk("MAX_POSITION_LOSS"),
        max_hold_days=config.risk("MAX_HOLD_DAYS"),
        max_quote_age_seconds=config.risk("MAX_QUOTE_AGE_SECONDS"),
        llm_interval_minutes=config.agent3_llm_interval_minutes(),
        material_move_pct=config.agent3_material_move_pct(),
        material_pnl_change_pct=config.agent3_material_pnl_change_pct(),
    )


def _sign(direction):
    return 1 if direction == "bullish" else -1


def _r(x):
    return None if x is None else round(x, 9)


#
# Quote validity
#

def quote_issues(inp: RiskAnalysisInput) -> list[str]:
    q = inp.quote
    if q is None:
        return ["QUOTE_UNAVAILABLE"]

    issues = []
    if q.contract_symbol != inp.contract_symbol:
        issues.append("QUOTE_CONTRACT_MISMATCH")
    if q.bid is None or q.bid <= 0:
        issues.append("QUOTE_NO_BID")
    if q.bid is not None and q.ask is not None and q.ask < q.bid:
        issues.append("QUOTE_CROSSED")
    if (inp.now_ms - q.as_of_ms) / 1000 > inp.limits.max_quote_age_seconds:
        issues.append("QUOTE_STALE")
    if q.contract_multiplier != inp.position.contract_multiplier:
        issues.append("QUOTE_TERMS_CHANGED")
    return issues


#
# Metrics
#

def compute_metrics(inp: RiskAnalysisInput) -> RiskMetrics:
    pos = inp.position
    th = inp.thesis
    q = inp.quote
    units = pos.contract_multiplier * pos.quantity
    entry_cost = pos.entry_option_price * units

    bid = q.bid if q is not None and q.bid is not None and q.bid > 0 else None
    mark = None
    if q is not None:
        mark = q.mark
        if mark is None and q.bid is not None and q.ask is not None:
            mark = (q.bid + q.ask) / 2

    liquidation = bid * units if bid is not None else None
    pnl = liquidation - entry_cost if liquidation is not None else None
    pnl_pct = pnl / entry_cost * 100 if pnl is not None and entry_cost > 0 else None
    mark_value = mark * units if mark is not None else None

    price = inp.underlying_price
    s = _sign(th.direction)
    progress = distance_to_target = distance_to_invalidation = move_since_entry = None

    if price is not None and th.direction != "neutral":
        base = th.entry_level if th.entry_level is not None else inp.underlying_price_at_entry
        if th.target is not None and base is not None and s * (th.target - base) > 0:
            progress = s * (price - base) / (s * (th.target - base)) * 100
        if th.target is not None:
            distance_to_target = s * (th.target - price) / price * 100
        if th.invalidation is not None:
            distance_to_invalidation = s * (price - th.invalidation) / price * 100
        if inp.underlying_price_at_entry:
            move_since_entry = s * (price - inp.underlying_price_at_entry) / inp.underlying_price_at_entry * 100

    time_in_trade_h = (inp.now_ms - pos.opened_at_ms) / HOUR_MS
    expiry_ms = q.expiry_ts_ms if q is not None else inp.option_expiry_ts_ms

    return RiskMetrics(
        entry_cost=entry_cost,
        liquidation_value=liquidation,
        unrealized_pnl=pnl,
        unrealized_pnl_pct=pnl_pct,
        mark_value=mark_value,
        mark_pnl=mark_value - entry_cost if mark_value is not None else None,
        option_bid=bid,
        option_mark=mark,
        quote_age_seconds=(inp.now_ms - q.as_of_ms) / 1000 if q is not None else None,
        target_progress_pct=progress,
        distance_to_target_pct=distance_to_target,
        distance_to_invalidation_pct=distance_to_invalidation,
        underlying_move_since_entry_pct=move_since_entry,
        time_in_trade_hours=time_in_trade_h,
        hold_remaining_hours=inp.limits.max_hold_days * 24 - time_in_trade_h,
        dte=(expiry_ms - inp.now_ms) / (24 * HOUR_MS) if expiry_ms is not None else None,
    )


#
# Hard rules
#

def evaluate_hard_rules(inp: RiskAnalysisInput, metrics: RiskMetrics):
    """Returns (exit_codes, data_issue_codes, reasons)."""
    th = inp.thesis
    lim = inp.limits
    exits, reasons = [], []

    def exit_(code, reason):
        if code not in exits:
            exits.append(code)
            reasons.append(reason)

    # Evaluator status: persisted and chronologically resolved.
    if th.signal_status == "INVALIDATED":
        exit_("SIGNAL_INVALIDATED", "signal status INVALIDATED (evaluator)")
    if th.signal_status == "TARGET_HIT":
        exit_("SIGNAL_TARGET_HIT", "signal status TARGET_HIT: thesis objective completed")

    resolved = th.signal_status in ("INVALIDATED", "TARGET_HIT")
    s = _sign(th.direction)
    price = inp.underlying_price
    candle = inp.last_candle

    if th.direction != "neutral" and not resolved:
        target_touched = invalidation_touched = False
        both_in_candle = False

        if candle is not None and th.target is not None and th.invalidation is not None:
            t_in = (candle.high >= th.target) if s > 0 else (candle.low <= th.target)
            i_in = (candle.low <= th.invalidation) if s > 0 else (candle.high >= th.invalidation)
            both_in_candle = t_in and i_in
            target_touched |= t_in
            invalidation_touched |= i_in

        if price is not None:
            if th.target is not None and s * (price - th.target) >= 0:
                target_touched = True
            if th.invalidation is not None and s * (price - th.invalidation) <= 0:
                invalidation_touched = True

        if both_in_candle:
            exit_("AMBIGUOUS_INTRABAR_ORDER",
                  "target and invalidation both inside the same candle; order unknown, "
                  "not resolved by the evaluator; exit with cause undetermined")
        else:
            if invalidation_touched:
                exit_("UNDERLYING_INVALIDATION_HIT", f"underlying crossed invalidation {th.invalidation}")
            if target_touched:
                exit_("UNDERLYING_TARGET_HIT", f"underlying reached target {th.target}: thesis objective completed")

    # Time
    expiry_ms = inp.quote.expiry_ts_ms if inp.quote is not None else inp.option_expiry_ts_ms
    if expiry_ms is not None and inp.now_ms >= expiry_ms:
        exit_("OPTION_EXPIRED", "option expiry reached")

    if round(metrics.time_in_trade_hours, 9) >= round(lim.max_hold_days * 24, 9):
        exit_("MAX_HOLD_EXCEEDED", f"held {metrics.time_in_trade_hours:.1f}h >= max {lim.max_hold_days:g} days")

    # Quote-dependent stops: only on a valid quote.
    data = quote_issues(inp)
    if price is None:
        data.append("UNDERLYING_PRICE_UNAVAILABLE")

    if not [d for d in data if d.startswith("QUOTE_")] and metrics.unrealized_pnl_pct is not None:
        if _r(metrics.unrealized_pnl_pct) <= -lim.hard_premium_stop_pct:
            exit_("HARD_PREMIUM_STOP",
                  f"bid-based PnL {metrics.unrealized_pnl_pct:.1f}% <= -{lim.hard_premium_stop_pct:g}%")
        if lim.max_position_loss > 0 and _r(-metrics.unrealized_pnl) >= lim.max_position_loss:
            exit_("MAX_POSITION_LOSS",
                  f"bid-based loss ${-metrics.unrealized_pnl:.2f} >= max ${lim.max_position_loss:g}")

    return exits, data, reasons


#
# LLM throttle (hard rules are never throttled; this only gates the LLM)
#

def llm_due(inp: RiskAnalysisInput, metrics: RiskMetrics, last: LastLLMAssessment | None,
            previous_risk_state: str | None = None):
    """
    Returns (due, trigger). `last` is the most recent LLM-backed assessment;
    `previous_risk_state` is the most recent assessment of any kind.
    """
    lim = inp.limits

    if last is None:
        return True, "first_contextual_assessment"

    if (inp.now_ms - last.assessed_at_ms) / 60_000 >= lim.llm_interval_minutes:
        return True, "interval_elapsed"

    if last.underlying_price and inp.underlying_price is not None:
        move = abs(inp.underlying_price - last.underlying_price) / last.underlying_price * 100
        if round(move, 9) >= lim.material_move_pct:
            return True, f"material_underlying_move_{move:.2f}pct"

    if last.unrealized_pnl_pct is not None and metrics.unrealized_pnl_pct is not None:
        change = abs(metrics.unrealized_pnl_pct - last.unrealized_pnl_pct)
        if round(change, 9) >= lim.material_pnl_change_pct:
            return True, f"material_pnl_change_{change:.1f}pts"

    if previous_risk_state in ("DATA_UNAVAILABLE", "FALLBACK"):
        return True, f"risk_state_changed_from_{previous_risk_state}"

    return False, None


def assessment_key(inp: RiskAnalysisInput) -> str:
    """Identity of the market/position snapshot (throttle metadata excluded)."""
    canonical = json.dumps(inp.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
