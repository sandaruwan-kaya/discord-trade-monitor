"""
Agent 2 — Options Selection Agent (strategy-constrained).

Input is an OptionsSelectionInput whose candidates were already
pre-filtered, sized and priced by deterministic code
(tools/options_tools.prefilter). The LLM may only:

    - pick one candidate_id, or
    - decline (NO_TRADE)

Price, quantity, max loss, strategy and side are copied from the
deterministic candidate, never from the LLM. The output is a
PROPOSAL only; this agent has no broker and submits nothing.
"""
import json

from llm_client import LLMClient
from schemas import (
    OptionCandidate,
    OptionProposal,
    OptionsSelectionInput,
    OptionsSelectionLLMOutput,
)


SYSTEM_PROMPT = """
You select one option contract for a paper-trading proposal.

The candidates were already filtered and sized by deterministic risk
rules. Every candidate is acceptable under policy. Your job is only to
choose the best fit for the thesis, or decline.

Rules:
- You may ONLY return a candidate_id from the list, or decision NO_TRADE.
- Do NOT propose prices, quantities, strikes or strategies of your own.
- Prefer contracts whose estimated value at target clearly exceeds the
  limit price, with enough time to expiry for the thesis horizon, and
  tighter spreads / better liquidity.
- Decline (NO_TRADE) if no candidate is a sensible expression of the thesis.
- ranking: candidate_ids best first. reason: 1-3 sentences.
- confidence: 0.0 to 1.0.
Do not provide financial advice.
""".strip()


def proposal_key(signal_id: int) -> str:
    return f"signal:{signal_id}:entry"


class OptionsSelectionAgent:

    def __init__(self, llm: LLMClient | None, log=print):
        self.llm = llm
        self._log = log

    def build_prompt(self, inp: OptionsSelectionInput) -> str:
        candidates = [
            {
                "candidate_id": c.candidate_id,
                "strategy": c.strategy,
                "contract_symbol": c.contract.contract_symbol,
                "expiry": str(c.contract.expiry),
                "dte": round(c.contract.dte, 2),
                "strike": c.contract.strike,
                "delta": round(c.contract.delta, 3) if c.contract.delta is not None else None,
                "iv": round(c.contract.iv, 3) if c.contract.iv is not None else None,
                "bid": c.contract.bid,
                "ask": c.contract.ask,
                "spread_pct": round(c.spread_pct, 2),
                "open_interest": c.contract.open_interest,
                "volume_24h": c.contract.volume_24h,
                "limit_price": c.limit_price,
                "quantity": c.quantity,
                "estimated_max_loss_usd": c.estimated_max_loss,
                "breakeven": c.breakeven,
                "estimated_value_at_target": round(c.estimated_value_at_target, 2),
                "estimated_value_at_invalidation": (
                    round(c.estimated_value_at_invalidation, 2)
                    if c.estimated_value_at_invalidation is not None else None
                ),
                "reward_to_risk_at_target": round(c.reward_to_risk_at_target, 3),
                "deterministic_rank": c.deterministic_rank,
            }
            for c in inp.candidates
        ]

        return json.dumps(
            {
                "thesis": inp.thesis.model_dump(),
                "underlying_price": inp.underlying_price,
                "trigger": inp.trigger,
                "candidates": candidates,
            },
            indent=2,
            default=str
        )

    def propose(self, inp: OptionsSelectionInput) -> OptionProposal:
        base = dict(
            proposal_key=proposal_key(inp.signal_id),
            signal_id=inp.signal_id,
            analysis_id=inp.analysis_id,
            trigger=inp.trigger,
            underlying=inp.thesis.underlying,
            thesis_target=inp.thesis.target,
            thesis_invalidation=inp.thesis.invalidation,
            candidates_considered=len(inp.candidates),
            filter_summary=inp.filter_summary,
        )

        if not inp.candidates:
            return OptionProposal(
                status="NO_CANDIDATES",
                reason="no contract passed the deterministic playbook filters",
                **base
            )

        by_id = {c.candidate_id: c for c in inp.candidates}
        top = min(inp.candidates, key=lambda c: c.deterministic_rank)
        notes = []
        llm_out = None

        if self.llm is not None:
            try:
                llm_out = self.llm.structured(
                    SYSTEM_PROMPT,
                    self.build_prompt(inp),
                    OptionsSelectionLLMOutput
                )
            except Exception as error:
                self._log(f"[AGENT2] LLM failed: {type(error).__name__}: {str(error)[:200]}")
                notes.append(f"LLM unavailable ({type(error).__name__}); deterministic top candidate used")
        else:
            notes.append("no LLM configured; deterministic top candidate used")

        if llm_out is None:
            return self._proposal(top, "deterministic_fallback", False, 0.0,
                                  f"deterministic rank 1 of {len(inp.candidates)}", notes, base)

        confidence = min(max(float(llm_out.confidence), 0.0), 1.0)
        reason = (llm_out.reason or "")[:1000]

        if llm_out.decision == "NO_TRADE":
            return OptionProposal(
                status="DECLINED",
                llm_used=True,
                confidence=confidence,
                reason=reason or "LLM declined all candidates",
                validation_notes=notes,
                **base
            )

        chosen = by_id.get(llm_out.selected_candidate_id or "")

        if chosen is None:
            notes.append(
                f"LLM selected unknown candidate {llm_out.selected_candidate_id!r}; "
                f"deterministic top candidate {top.candidate_id} used"
            )
            return self._proposal(top, "deterministic_fallback", True, confidence, reason, notes, base)

        return self._proposal(chosen, "llm", True, confidence, reason, notes, base)

    @staticmethod
    def _proposal(candidate: OptionCandidate, selected_by, llm_used, confidence, reason, notes, base):
        contract = candidate.contract

        return OptionProposal(
            status="PROPOSED",
            strategy=candidate.strategy,
            side="BUY_TO_OPEN",
            contract_symbol=contract.contract_symbol,
            option_type=contract.option_type,
            expiry=contract.expiry,
            dte=contract.dte,
            strike=contract.strike,
            delta=contract.delta,
            iv=contract.iv,
            bid=contract.bid,
            ask=contract.ask,
            mid=candidate.mid,
            limit_price=candidate.limit_price,
            quantity=candidate.quantity,
            contract_multiplier=contract.contract_multiplier,
            estimated_max_loss=candidate.estimated_max_loss,
            breakeven=candidate.breakeven,
            selected_by=selected_by,
            llm_used=llm_used,
            confidence=confidence,
            reason=reason,
            validation_notes=notes,
            **base
        )
