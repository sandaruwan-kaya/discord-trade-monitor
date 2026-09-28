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
from signal_facts import build_input, compute_facts, deterministic_state  # noqa: F401 (re-exported)
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
