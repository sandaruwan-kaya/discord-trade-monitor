"""
Agent 3 orchestration: gather a risk snapshot for an OPEN paper position,
run the Risk Analysis Agent, persist the assessment.

Reads position/signal state straight from SQLite (read-only) and a FRESH
option quote from the OptionChainProvider. It never imports a broker or
the execution service, never mutates a position, proposal or decision.
"""
import json
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone

import database
from core.risk_rules import current_limits
from schemas import (
    Candle,
    LastLLMAssessment,
    RiskAnalysisInput,
    RiskAssessment,
    RiskPosition,
    RiskThesis,
)
from signal_facts import build_input, compute_facts


def _iso_to_ms(value):
    return int(datetime.fromisoformat(value).timestamp() * 1000)


def _one(sql, params=()):
    with closing(database.get_connection()) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(sql, params).fetchone()
    return dict(row) if row else None


def _proposal_expiry_ms(proposal_key):
    """Fallback expiry from the stored proposal (Deribit convention: 08:00 UTC)."""
    row = _one("SELECT output_json FROM trade_proposals WHERE proposal_key = ?", (proposal_key,)) if proposal_key else None
    if not row or not row["output_json"]:
        return None
    expiry = json.loads(row["output_json"]).get("expiry")
    if not expiry:
        return None
    return int(datetime.fromisoformat(expiry).replace(hour=8, tzinfo=timezone.utc).timestamp() * 1000)


def gather_risk_input(position_id, chain_provider, market, now_ms, context_market=None) -> RiskAnalysisInput | None:
    """None if the position does not exist or is not OPEN (no analysis for closed positions)."""
    pos = _one("SELECT * FROM paper_positions WHERE position_id = ?", (position_id,))
    if pos is None or pos["status"] != "OPEN":
        return None

    signal_row = _one("SELECT * FROM signals WHERE id = ?", (pos["signal_id"],)) if pos["signal_id"] else None
    underlying = pos["underlying"] or (signal_row["symbol"] if signal_row else None)

    quote, quote_error = None, None
    try:
        quote = chain_provider.get_option_contract(pos["contract_symbol"])
    except Exception as error:
        quote_error = f"{type(error).__name__}: {str(error)[:200]}"

    price = last_candle = price_at_entry = None
    try:
        price = market.get_current_price(underlying)
    except Exception:
        pass
    try:
        candles = market.get_candles(underlying, "1m", 2)
        if candles:
            c = candles[-1]
            last_candle = Candle(timestamp=c["timestamp"], high=c["high"], low=c["low"], close=c["close"])
    except Exception:
        pass
    try:
        opened_ms = _iso_to_ms(pos["opened_at"])
        entry_candles = market.get_candles(underlying, "1m", 1, (opened_ms // 60_000) * 60_000)
        if entry_candles:
            price_at_entry = entry_candles[0]["close"]
    except Exception:
        pass

    context = None
    if context_market is not None:
        try:
            context = context_market.get_market_context(underlying)
        except Exception:
            pass

    if signal_row:
        signal_input = build_input(signal_row, current_price=price)
        facts = compute_facts(signal_input, now_ms)
        state = json.loads(signal_row["eval_state"]) if signal_row.get("eval_state") else {}
        thesis = RiskThesis(
            direction=facts.direction,
            entry_level=facts.default_entry_level,
            target=facts.default_target,
            invalidation=facts.default_invalidation,
            timeframe=signal_row.get("timeframe"),
            signal_time_ms=signal_row["signal_time_ms"],
            signal_status=signal_row["status"],
            evaluator_target_time_ms=state.get("target_time"),
            evaluator_invalidation_time_ms=state.get("invalidation_time"),
            recent_high=state.get("highest_price"),
            recent_low=state.get("lowest_price"),
        )
    else:
        thesis = RiskThesis(direction="neutral", entry_level=None, target=None, invalidation=None,
                            timeframe=None, signal_time_ms=0, signal_status="MISSING")

    return RiskAnalysisInput(
        position_id=position_id,
        signal_id=pos["signal_id"],
        proposal_key=pos["proposal_key"],
        contract_symbol=pos["contract_symbol"],
        thesis=thesis,
        position=RiskPosition(
            status=pos["status"],
            quantity=pos["quantity"],
            contract_multiplier=pos["contract_multiplier"],
            entry_option_price=pos["entry_price"],
            opened_at_ms=_iso_to_ms(pos["opened_at"]),
        ),
        quote=quote,
        quote_error=quote_error,
        option_expiry_ts_ms=quote.expiry_ts_ms if quote else _proposal_expiry_ms(pos["proposal_key"]),
        underlying_price=price,
        underlying_price_at_entry=price_at_entry,
        last_candle=last_candle,
        market_context=context,
        now_ms=now_ms,
        limits=current_limits(),
    )


def last_llm_assessment(position_id) -> LastLLMAssessment | None:
    row = _one(
        """
        SELECT assessed_at_ms, underlying_price, unrealized_pnl_pct, recommendation, risk_state, output_json
        FROM risk_assessments WHERE position_id = ? AND llm_used = 1
        ORDER BY assessed_at_ms DESC, rowid DESC LIMIT 1
        """,
        (position_id,)
    )
    if row is None:
        return None
    return LastLLMAssessment(
        assessed_at_ms=row["assessed_at_ms"],
        underlying_price=row["underlying_price"],
        unrealized_pnl_pct=row["unrealized_pnl_pct"],
        recommendation=row["recommendation"],
        risk_state=row["risk_state"],
        confidence=json.loads(row["output_json"])["confidence"],
    )


def previous_risk_state(position_id):
    row = _one(
        "SELECT risk_state FROM risk_assessments WHERE position_id = ? ORDER BY assessed_at_ms DESC, rowid DESC LIMIT 1",
        (position_id,)
    )
    return row["risk_state"] if row else None


def get_assessment_by_key(key) -> RiskAssessment | None:
    row = _one("SELECT output_json FROM risk_assessments WHERE assessment_key = ?", (key,))
    return RiskAssessment.model_validate_json(row["output_json"]) if row else None


def persist_assessment(a: RiskAssessment) -> RiskAssessment:
    with closing(database.get_connection()) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO risk_assessments (
                assessment_id, assessment_key, position_id, signal_id, recommendation, risk_state,
                hard_rule_triggered, hard_rule_codes_json, data_issue_codes_json, llm_used, llm_status,
                reasons_json, underlying_price, option_bid, option_mark, unrealized_pnl, unrealized_pnl_pct,
                assessed_at, assessed_at_ms, input_json, output_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                a.assessment_id, a.assessment_key, a.position_id, a.signal_id, a.recommendation, a.risk_state,
                1 if a.hard_rule_triggered else 0, json.dumps(a.hard_rule_codes), json.dumps(a.data_issue_codes),
                1 if a.llm_used else 0, a.llm_status, json.dumps(a.reasons), a.underlying_price,
                a.option_bid, a.option_mark, a.unrealized_pnl, a.unrealized_pnl_pct,
                a.assessed_at, a.input_snapshot.now_ms, a.input_snapshot.model_dump_json(),
                a.model_dump_json(), database.utc_now_iso(),
            )
        )
        conn.commit()
    return get_assessment_by_key(a.assessment_key)


class RiskService:

    def __init__(self, agent, chain_provider, market, context_market=None, now_ms=None):
        self.agent = agent
        self.chain_provider = chain_provider
        self.market = market
        self.context_market = context_market
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))

    def assess_position(self, position_id) -> RiskAssessment | None:
        inp = gather_risk_input(position_id, self.chain_provider, self.market, self._now_ms(), self.context_market)
        if inp is None:
            return None                                   # closed / missing: no analysis, no action

        from core.risk_rules import assessment_key
        existing = get_assessment_by_key(assessment_key(inp))
        if existing is not None:
            return existing                               # identical snapshot: no duplicate row, no LLM call

        assessment = self.agent.assess(inp, last_llm_assessment(position_id), previous_risk_state(position_id))
        return persist_assessment(assessment)

    def assess_open_positions(self):
        with closing(database.get_connection()) as conn:
            ids = [r[0] for r in conn.execute(
                "SELECT position_id FROM paper_positions WHERE status = 'OPEN' ORDER BY opened_at"
            )]
        return [a for a in (self.assess_position(i) for i in ids) if a is not None]
