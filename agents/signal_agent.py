"""
Agent 1 — Signal Analysis Agent.

Enriches an already-parsed Discord TradeSignal into a thesis that
later agents can act on. It never trades and never sees a broker.

Pipeline:
    1. compute_facts()   deterministic facts (direction, levels, crossed,
                         target reached, invalidated, progress, age)
    2. LLM               structured SignalAnalysisLLMOutput (optional;
                         failure falls back to facts only)
    3. post-validation   - every price level must already exist in the
                           parsed input (no invented levels)
                         - hard facts win: the LLM may escalate a state to
                           STALE / NOT_TRADEABLE but never relax one
"""
import json
import math
import time

import config
import signal_rules
from llm_client import LLMClient
from schemas import (
    MarketContext,
    SignalAnalysis,
    SignalAnalysisInput,
    SignalAnalysisLLMOutput,
    SignalFacts,
)


SYSTEM_PROMPT = """
You are a trading-signal analyst. You receive a cryptocurrency trading
thesis that was already extracted from a Discord message, plus
deterministic facts and current market context.

Classify the setup. You do NOT decide whether to trade.

Return:
- direction: bullish, bearish or neutral (must agree with the parsed bias)
- entry_condition_type: break_above, break_below, hold_above, hold_below,
  immediate, or none
- entry_level, target, invalidation: each MUST be copied exactly from the
  corresponding ALLOWED list, or null. Never invent, round or adjust a level.
- time_horizon: scalp, intraday, short_term, swing, position or unknown
- signal_state:
    DEVELOPING        entry condition has not happened yet
    ALREADY_CONFIRMED entry condition already met at the current price
    STALE             move already played out, invalidated or too old
    NOT_TRADEABLE     no clear direction, no usable target/invalidation,
                      or not really a trade setup
- confidence: 0.0 to 1.0 in your classification
- reasons: at most 5 short bullet reasons

The deterministic facts are computed from real prices; do not contradict them.
Do not provide financial advice.
""".strip()


STATE_SEVERITY = {
    "DEVELOPING": 0,
    "ALREADY_CONFIRMED": 0,
    "STALE": 1,
    "NOT_TRADEABLE": 2,
}

HORIZON_FROM_BUCKET = {
    "scalp": "scalp",
    "intraday": "intraday",
    "swing": "swing",
    "position": "position",
    "unknown": "unknown",
}


def _now_ms():
    return int(time.time() * 1000)


def _in(value, allowed):
    return any(
        math.isclose(value, a, rel_tol=1e-9, abs_tol=1e-9)
        for a in allowed
    )


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


def default_entry_type(inp: SignalAnalysisInput, facts: SignalFacts) -> str:
    if facts.direction == "bullish":
        if inp.confirmation_above is not None:
            return "break_above"
        if inp.confirmation_below is not None:
            return "hold_above"
        return "immediate"

    if facts.direction == "bearish":
        if inp.confirmation_below is not None:
            return "break_below"
        if inp.confirmation_above is not None:
            return "hold_below"
        return "immediate"

    return "none"


def allowed_entry_types(facts: SignalFacts) -> set[str]:
    """Entry type must agree with direction AND with whether a level exists."""
    if facts.direction == "neutral":
        return {"none"}

    if facts.default_entry_level is None:
        return {"immediate"}

    if facts.direction == "bullish":
        return {"break_above", "hold_above"}

    return {"break_below", "hold_below"}


#
# Agent
#

class SignalAnalysisAgent:

    def __init__(self, llm: LLMClient | None, now_ms=_now_ms, log=print):
        self.llm = llm
        self._now_ms = now_ms
        self._log = log

    def build_prompt(self, inp: SignalAnalysisInput, facts: SignalFacts) -> str:
        allowed = {
            "entry_level": sorted({
                x for x in [inp.confirmation_above, inp.confirmation_below,
                            *inp.support_levels, *inp.resistance_levels]
                if x is not None
            }),
            "target": facts.valid_targets,
            "invalidation": [facts.default_invalidation] if facts.default_invalidation is not None else [],
        }

        payload = {
            "signal": inp.model_dump(exclude_none=True, exclude={"market_context"}),
            "market_context": inp.market_context.model_dump() if inp.market_context else None,
            "deterministic_facts": facts.model_dump(),
            "ALLOWED_LEVELS": allowed,
        }

        return json.dumps(payload, indent=2, default=str)

    def analyze(self, inp: SignalAnalysisInput) -> SignalAnalysis:
        now_ms = self._now_ms()
        facts = compute_facts(inp, now_ms)
        state, det_reasons = deterministic_state(inp, facts)

        notes = []
        llm_out = None

        if self.llm is not None:
            try:
                llm_out = self.llm.structured(
                    SYSTEM_PROMPT,
                    self.build_prompt(inp, facts),
                    SignalAnalysisLLMOutput
                )
            except Exception as error:
                self._log(f"[AGENT1] LLM failed: {type(error).__name__}: {str(error)[:200]}")
                notes.append(f"LLM unavailable ({type(error).__name__}); deterministic analysis only")
        else:
            notes.append("no LLM configured; deterministic analysis only")

        entry = facts.default_entry_level
        target = facts.default_target
        invalidation = facts.default_invalidation
        entry_type = default_entry_type(inp, facts)
        horizon = HORIZON_FROM_BUCKET[facts.timeframe_bucket]
        confidence = 0.0
        reasons = list(det_reasons)

        if llm_out is not None:

            # Direction must agree with the parsed bias.
            if llm_out.direction != facts.direction and facts.direction != "neutral":
                notes.append(
                    f"LLM direction '{llm_out.direction}' contradicts parsed bias "
                    f"'{facts.direction}'"
                )
                state = "NOT_TRADEABLE"
                reasons.append("LLM and parser disagree on direction")

            # Price levels: must already exist in the parsed input.
            entry_allowed = {
                x for x in [inp.confirmation_above, inp.confirmation_below,
                            *inp.support_levels, *inp.resistance_levels]
                if x is not None
            }

            entry = self._pick_level("entry_level", llm_out.entry_level, entry_allowed, entry, notes)
            target = self._pick_level("target", llm_out.target, set(facts.valid_targets), target, notes)
            invalidation = self._pick_level(
                "invalidation", llm_out.invalidation,
                {facts.default_invalidation} - {None}, invalidation, notes
            )

            if llm_out.entry_condition_type in allowed_entry_types(facts):
                entry_type = llm_out.entry_condition_type
            else:
                notes.append(
                    f"LLM entry type '{llm_out.entry_condition_type}' inconsistent with "
                    f"{facts.direction}; using '{entry_type}'"
                )

            horizon = llm_out.time_horizon
            confidence = min(max(float(llm_out.confidence), 0.0), 1.0)
            reasons.extend(r[:300] for r in llm_out.reasons[:5])

            # State: the LLM may only escalate severity.
            if STATE_SEVERITY[llm_out.signal_state] > STATE_SEVERITY[state]:
                notes.append(f"LLM escalated state {state} -> {llm_out.signal_state}")
                state = llm_out.signal_state
            elif llm_out.signal_state != state:
                notes.append(
                    f"LLM state {llm_out.signal_state} ignored; deterministic state {state} stands"
                )

        if facts.rejected_targets:
            notes.extend(r["reason"] for r in facts.rejected_targets)

        next_step = {
            "DEVELOPING": "WAIT_FOR_CONFIRMATION",
            "ALREADY_CONFIRMED": "ELIGIBLE_FOR_OPTIONS_REVIEW",
        }.get(state, "NO_ACTION")

        return SignalAnalysis(
            underlying=inp.symbol,
            direction=facts.direction,
            is_conditional=facts.is_conditional,
            entry_condition_type=entry_type,
            entry_level=entry,
            target=target,
            invalidation=invalidation,
            time_horizon=horizon,
            signal_state=state,
            next_step=next_step,
            eligible_for_options=(state == "ALREADY_CONFIRMED"),
            confidence=confidence,
            reasons=reasons,
            validation_notes=notes,
            llm_used=llm_out is not None,
            facts=facts
        )

    @staticmethod
    def _pick_level(name, proposed, allowed, default, notes):
        """
        The final level is always the deterministic one, so the output
        stays consistent with the facts the state was derived from.
        The LLM's pick is only checked and any disagreement recorded.
        """
        if proposed is None or (default is not None and _in(proposed, {default})):
            return default

        if not _in(proposed, allowed):
            notes.append(
                f"LLM {name} {proposed} not in parsed signal levels {sorted(allowed)}; "
                f"rejected, using {default}"
            )
        else:
            notes.append(
                f"LLM suggested {name} {proposed}; kept deterministic {default}"
            )

        return default
