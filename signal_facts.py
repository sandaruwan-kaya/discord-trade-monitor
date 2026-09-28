"""
Deterministic signal facts (no LLM, no network, no broker).

Shared by Agent 1 (agents/signal_agent.py), the Agent 2 proposal
sweep and the Policy Engine, so every layer judges the thesis with
exactly the same rules.
"""
import json

import config
import signal_rules
from schemas import MarketContext, SignalAnalysisInput, SignalFacts


#
# Input builders
#

def build_input(
    row: dict,
    current_price: float | None = None,
    market_context: MarketContext | None = None
) -> SignalAnalysisInput:
    """Build Agent 1 input from a signals-table row."""

    def levels(key):
        return json.loads(row.get(key) or "[]")

    return SignalAnalysisInput(
        signal_id=row.get("id"),
        discord_message_id=row.get("discord_message_id"),
        raw_message=row.get("raw_message"),
        symbol=row["symbol"],
        bias=row.get("bias"),
        timeframe=row.get("timeframe"),
        summary=row.get("summary"),
        support_levels=levels("support_levels"),
        resistance_levels=levels("resistance_levels"),
        targets=levels("targets"),
        confirmation_above=row.get("confirmation_above"),
        confirmation_below=row.get("confirmation_below"),
        invalidation_above=row.get("invalidation_above"),
        invalidation_below=row.get("invalidation_below"),
        signal_time_ms=row["signal_time_ms"],
        price_at_signal=row.get("price_at_signal"),
        current_price=(
            current_price if current_price is not None
            else (market_context.current_price if market_context else None)
        ),
        market_context=market_context
    )


#
# Step 1 — deterministic facts
#

def compute_facts(inp: SignalAnalysisInput, now_ms: int) -> SignalFacts:
    direction = signal_rules.direction_from_bias(inp.bias)
    is_conditional = "conditional" in (inp.bias or "").lower()

    ref = inp.current_price if inp.current_price is not None else inp.price_at_signal

    valid_targets, rejected_targets = signal_rules.check_targets(
        inp.bias,
        inp.targets,
        price_at_signal=inp.price_at_signal,
        confirmation_above=inp.confirmation_above,
        confirmation_below=inp.confirmation_below,
        invalidation_above=inp.invalidation_above,
        invalidation_below=inp.invalidation_below
    )

    entry = target = invalidation = None

    if direction == "bullish":
        entry = inp.confirmation_above if inp.confirmation_above is not None else inp.confirmation_below
        target = min(valid_targets) if valid_targets else None
        invalidation = inp.invalidation_below

    elif direction == "bearish":
        entry = inp.confirmation_below if inp.confirmation_below is not None else inp.confirmation_above
        target = max(valid_targets) if valid_targets else None
        invalidation = inp.invalidation_above

    sign = 1 if direction == "bullish" else -1

    entry_crossed = target_reached = invalidation_hit = False
    progress = None

    if direction != "neutral" and ref is not None:

        entry_crossed = entry is None or sign * (ref - entry) >= 0

        if target is not None:
            target_reached = sign * (ref - target) >= 0

        if invalidation is not None:
            invalidation_hit = sign * (ref - invalidation) <= 0

        if entry_crossed and target is not None:
            base = entry
            if base is None:
                base = inp.price_at_signal if inp.price_at_signal is not None else ref

            distance = sign * (target - base)

            if distance > 0:
                progress = sign * (ref - base) / distance * 100

    bucket = signal_rules.classify_timeframe(inp.timeframe)

    return SignalFacts(
        direction=direction,
        is_conditional=is_conditional,
        reference_price=ref,
        valid_targets=valid_targets,
        rejected_targets=rejected_targets,
        default_entry_level=entry,
        default_target=target,
        default_invalidation=invalidation,
        entry_crossed=entry_crossed,
        target_reached=target_reached,
        invalidation_hit=invalidation_hit,
        target_progress_pct=progress,
        age_minutes=(now_ms - inp.signal_time_ms) / 60_000,
        expired=signal_rules.is_expired(inp.signal_time_ms, inp.timeframe, now_ms),
        timeframe_bucket=bucket
    )


def deterministic_state(inp: SignalAnalysisInput, facts: SignalFacts):
    """Returns (state, reasons). Pure function of the facts."""
    not_tradeable = []

    if facts.direction == "neutral":
        not_tradeable.append("no directional bias (bullish/bearish) in parsed signal")

    if facts.default_target is None:
        if facts.rejected_targets:
            not_tradeable.append("all targets are directionally inconsistent")
        else:
            not_tradeable.append("no target in parsed signal")

    if config.agent1_require_invalidation() and facts.default_invalidation is None:
        not_tradeable.append("no invalidation level for this direction")

    if not_tradeable:
        return "NOT_TRADEABLE", not_tradeable

    if facts.reference_price is None:
        return "NOT_TRADEABLE", ["no current price or price at signal available"]

    stale = []

    if facts.invalidation_hit:
        stale.append(
            f"price {facts.reference_price} already beyond invalidation {facts.default_invalidation}"
        )

    if facts.target_reached:
        stale.append(
            f"price {facts.reference_price} already reached target {facts.default_target}"
        )

    if facts.expired:
        stale.append(
            f"signal age {facts.age_minutes:.0f} min exceeds expiry for "
            f"'{facts.timeframe_bucket}' timeframe"
        )

    max_progress = config.agent1_max_target_progress_pct()

    if (
        facts.entry_crossed
        and facts.target_progress_pct is not None
        and facts.target_progress_pct > max_progress
    ):
        stale.append(
            f"{facts.target_progress_pct:.0f}% of entry->target move already done "
            f"(max {max_progress:.0f}%)"
        )

    if stale:
        return "STALE", stale

    if facts.entry_crossed:
        if facts.default_entry_level is None:
            return "ALREADY_CONFIRMED", ["no confirmation level: immediate entry setup"]
        return "ALREADY_CONFIRMED", [
            f"price {facts.reference_price} already beyond entry {facts.default_entry_level}"
        ]

    return "DEVELOPING", [
        f"waiting for entry {facts.default_entry_level} (price {facts.reference_price})"
    ]
