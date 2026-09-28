"""
Agent 3 — Risk Analysis Agent.

Assesses an EXISTING OPEN paper position and recommends HOLD / REDUCE /
EXIT. It never submits, closes or cancels anything and has no broker or
execution-service access.

Order of operations (every cycle):
    1. deterministic metrics + hard rules (core/risk_rules.py) — never throttled
    2. hard EXIT active     -> EXIT, LLM NOT called (cannot be relaxed)
    3. unusable data        -> DATA_UNAVAILABLE, LLM NOT called
    4. otherwise, if the throttle says the LLM is due -> contextual LLM
       recommendation (HOLD/REDUCE/EXIT, validated); if not due -> HOLD
       (THROTTLED) with the last contextual view carried separately

REDUCE is advisory only in V1; no quantity is computed or submitted.
"""
import json
import re
from datetime import datetime, timezone

from core.risk_rules import assessment_key, compute_metrics, evaluate_hard_rules, llm_due
from llm_client import LLMClient
from schemas import (
    RISK_REASON_CODES,
    LastLLMAssessment,
    RiskAnalysisInput,
    RiskAssessment,
    RiskLLMOutput,
)


SYSTEM_PROMPT = f"""
You are a risk analyst for an open LONG option paper position.

Deterministic hard stops have already been checked and none triggered.
Assess whether the position should be held, reduced, or exited given
the supplied facts only.

Rules:
- Use ONLY the supplied facts. Do NOT invent price levels, targets,
  invalidations or stops, and do not propose new ones.
- recommendation: HOLD, REDUCE or EXIT.
- reason_codes: choose from {list(RISK_REASON_CODES)}.
- rationale: 1-3 sentences referring to the supplied facts.
- confidence: 0.0 to 1.0.
PnL figures are bid-based (realistic exit value). Do not provide financial advice.
""".strip()


def _iso(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat(timespec="seconds")


def _supplied_levels(inp: RiskAnalysisInput):
    th = inp.thesis
    levels = [th.entry_level, th.target, th.invalidation, th.recent_high, th.recent_low,
              inp.underlying_price, inp.underlying_price_at_entry]
    if inp.quote is not None:
        levels.append(inp.quote.strike)
    if inp.last_candle is not None:
        levels += [inp.last_candle.high, inp.last_candle.low, inp.last_candle.close]
    if inp.market_context is not None:
        levels += [inp.market_context.high_24h, inp.market_context.low_24h, inp.market_context.current_price]
        levels += [v for v in inp.market_context.indicators.values() if isinstance(v, (int, float))]
    return [x for x in levels if x]


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def unsupported_levels(text: str, inp: RiskAnalysisInput):
    """Price-like numbers in the rationale that are not any supplied level."""
    if not text or not inp.underlying_price:
        return []
    price = inp.underlying_price
    supplied = _supplied_levels(inp)
    found = []
    for raw in _NUMBER.findall(text):
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if 0.5 * price <= value <= 1.5 * price and not any(abs(value - s) <= s * 0.001 for s in supplied):
            found.append(value)
    return found


class RiskAnalysisAgent:

    def __init__(self, llm: LLMClient | None, log=print):
        self.llm = llm
        self._log = log

    def build_prompt(self, inp: RiskAnalysisInput, metrics) -> str:
        q = inp.quote
        return json.dumps({
            "thesis": inp.thesis.model_dump(),
            "position": inp.position.model_dump(),
            "option": {
                "contract_symbol": inp.contract_symbol,
                "bid": q.bid, "ask": q.ask, "mark": metrics.option_mark,
                "delta": q.delta, "gamma": q.gamma, "theta": q.theta, "vega": q.vega, "iv": q.iv,
                "strike": q.strike, "dte": metrics.dte,
            } if q else None,
            "underlying_price": inp.underlying_price,
            "metrics": metrics.model_dump(),
            "market_context": inp.market_context.model_dump() if inp.market_context else None,
        }, indent=2, default=str)

    def assess(
        self,
        inp: RiskAnalysisInput,
        last_llm: LastLLMAssessment | None = None,
        previous_risk_state: str | None = None,
    ) -> RiskAssessment:
        if inp.position.status != "OPEN":
            raise ValueError("risk analysis only applies to OPEN positions")

        metrics = compute_metrics(inp)
        exits, data_issues, hard_reasons = evaluate_hard_rules(inp, metrics)
        notes: list[str] = []
        reasons = list(hard_reasons)
        llm_used = False
        trigger = None
        confidence = 0.0

        if exits:
            recommendation, state, llm_status, confidence = "EXIT", "HARD_EXIT", "SKIPPED_HARD_EXIT", 1.0

        elif data_issues:
            recommendation, state, llm_status = "DATA_UNAVAILABLE", "DATA_UNAVAILABLE", "SKIPPED_DATA"
            reasons.append(f"unusable market data: {', '.join(data_issues)}")
            if inp.quote_error:
                notes.append(f"quote error: {inp.quote_error}")

        elif self.llm is None:
            recommendation, state, llm_status = "HOLD", "FALLBACK", "NOT_CONFIGURED"
            reasons.append("no hard exit; no LLM configured")

        else:
            due, trigger = llm_due(inp, metrics, last_llm, previous_risk_state)

            if not due:
                recommendation, state, llm_status = "HOLD", "THROTTLED", "THROTTLED"
                reasons.append("no hard exit; contextual LLM throttled")
            else:
                try:
                    out = self.llm.structured(SYSTEM_PROMPT, self.build_prompt(inp, metrics), RiskLLMOutput)
                    llm_used = True
                except Exception as error:
                    self._log(f"[AGENT3] LLM failed: {type(error).__name__}: {str(error)[:200]}")
                    out = None
                    notes.append(f"LLM unavailable ({type(error).__name__}); deterministic fallback")

                if out is None:
                    recommendation, state, llm_status = "HOLD", "FALLBACK", "FAILED"
                    reasons.append("no hard exit; LLM unavailable")
                else:
                    recommendation, state, llm_status = out.recommendation, "CONTEXTUAL", "CALLED"
                    confidence = min(max(float(out.confidence), 0.0), 1.0)

                    valid = [c for c in out.reason_codes if c in RISK_REASON_CODES]
                    invalid = [c for c in out.reason_codes if c not in RISK_REASON_CODES]
                    if invalid:
                        notes.append(f"unknown reason codes ignored: {invalid}")
                    reasons.extend(valid)
                    if out.rationale:
                        reasons.append(out.rationale[:500])

                    invented = unsupported_levels(out.rationale, inp)
                    if invented:
                        notes.append(
                            f"rationale mentions level(s) not in supplied facts {invented}; ignored "
                            f"(thesis levels, stops and targets are never changed by the LLM)"
                        )

        key = assessment_key(inp)

        return RiskAssessment(
            assessment_id=f"RA-{key[:20]}",
            assessment_key=key,
            position_id=inp.position_id,
            signal_id=inp.signal_id,
            proposal_key=inp.proposal_key,
            recommendation=recommendation,
            risk_state=state,
            hard_rule_triggered=bool(exits),
            hard_rule_codes=exits,
            data_issue_codes=data_issues,          # kept even on EXIT: exit pricing needs a usable quote
            llm_used=llm_used,
            llm_status=llm_status,
            llm_trigger=trigger,
            last_contextual_recommendation=last_llm.recommendation if last_llm else None,
            last_contextual_at_ms=last_llm.assessed_at_ms if last_llm else None,
            confidence=confidence,
            reasons=reasons,
            metrics=metrics,
            underlying_price=inp.underlying_price,
            option_bid=metrics.option_bid,
            option_mark=metrics.option_mark,
            unrealized_pnl=metrics.unrealized_pnl,
            unrealized_pnl_pct=metrics.unrealized_pnl_pct,
            target_progress_pct=metrics.target_progress_pct,
            distance_to_invalidation_pct=metrics.distance_to_invalidation_pct,
            time_in_trade_hours=metrics.time_in_trade_hours,
            assessed_at=_iso(inp.now_ms),
            input_snapshot=inp,
            validation_notes=notes,
        )
