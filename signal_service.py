"""
Signal intake + monitoring, extracted from app.py so it can be
tested offline with injected dependencies.

Behaviour matches the original app.py flow:

    Discord text -> LLM parse -> price -> SQLite -> Telegram
    every 60s: new candles -> evaluator -> Telegram on status change

Step 0 fixes:
    - incremental candle processing (no 200-candle freeze)
    - signal expiry (EXPIRED)
    - raw message / timeframe / timestamps persisted
    - signal saved even if the price lookup fails
    - directionally inconsistent targets flagged and ignored
"""
import asyncio
import json
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import config
import database
import signal_rules
from evaluator import evaluate_signal, new_state
from models import TradeSignal


MINUTE_MS = 60 * 1000


def _now_ms():
    return int(time.time() * 1000)


@dataclass
class Services:
    parse_message: Callable[[str], TradeSignal]
    market: Any                       # evaluator candle feed (Binance.US)
    notify: Callable[[str], Any]
    context_market: Any = None        # agent context chain (TradingView -> Binance.US)
    signal_agent: Any = None          # agents.signal_agent.SignalAnalysisAgent
    option_chain: Any = None          # tools.options_tools.OptionChainProvider
    options_agent: Any = None         # agents.options_agent.OptionsSelectionAgent
    now_ms: Callable[[], int] = field(default=_now_ms)
    log: Callable[..., None] = field(default=print)


#
# Convert database row back into
# our Pydantic TradeSignal model.
#

def database_row_to_signal(row):

    return TradeSignal(
        is_trade_signal=True,

        symbol=row["symbol"],
        bias=row["bias"],
        timeframe=row.get("timeframe"),
        summary=row["summary"],

        support_levels=json.loads(
            row["support_levels"] or "[]"
        ),

        resistance_levels=json.loads(
            row["resistance_levels"] or "[]"
        ),

        targets=json.loads(
            row["targets"] or "[]"
        ),

        confirmation_above=row[
            "confirmation_above"
        ],

        confirmation_below=row[
            "confirmation_below"
        ],

        invalidation_above=row[
            "invalidation_above"
        ],

        invalidation_below=row[
            "invalidation_below"
        ]
    )


#
# Generate Telegram status message
#

def build_status_message(
    row,
    signal,
    assessment
):

    return f"""
📊 DISCORD TRADE UPDATE

Asset:
{signal.symbol}

Bias:
{signal.bias or "Unknown"}

Original thesis:
{signal.summary or "No summary"}

Price when posted:
{row["price_at_signal"]}

Current price:
{assessment["current_price"]}

High since post:
{assessment["highest_price"]}

Low since post:
{assessment["lowest_price"]}

Confirmation hit:
{assessment["confirmation_hit"]}

Target hit:
{assessment["target_hit"]}

Invalidated:
{assessment["invalidated"]}

STATUS:
{assessment["status"]}
""".strip()


def build_new_signal_message(signal, text, price_at_signal, rejected_targets):

    message = f"""
🆕 NEW DISCORD TRADE SIGNAL

Asset:
{signal.symbol}

Bias:
{signal.bias or "Unknown"}

Thesis:
{signal.summary or text[:300]}

Price when posted:
{price_at_signal if price_at_signal is not None else "Unavailable"}

Status:
TRACKING

The system will now monitor this signal automatically.
""".strip()

    if rejected_targets:
        ignored = ", ".join(str(r["target"]) for r in rejected_targets)
        message += f"\n\n⚠️ Ignored inconsistent targets: {ignored}"

    return message


#
# Intake
#

async def process_discord_message(
    discord_message_id,
    text,
    signal_time_ms,
    services: Services
):
    """
    Returns the new signal id, or None if nothing was saved.
    """
    log = services.log

    signal = await asyncio.to_thread(
        services.parse_message,
        text
    )

    log()
    log("PARSED SIGNAL")
    log(signal.model_dump_json(indent=2))

    #
    # Ignore chatter.
    #

    if not signal.is_trade_signal:
        log("Ignored: not a trade signal.")
        return None

    if not signal.symbol:
        log("Ignored: no market symbol.")
        return None

    #
    # Price when the Discord message was received.
    # A lookup failure must not lose a valid signal.
    #

    try:
        price_at_signal = await asyncio.to_thread(
            services.market.get_current_price,
            signal.symbol
        )
    except Exception as error:
        price_at_signal = None
        log(
            f"[INTAKE] price lookup failed for {signal.symbol}: "
            f"{type(error).__name__}; saving with price_at_signal=NULL"
        )

    _, rejected_targets = signal_rules.check_signal_targets(
        signal,
        price_at_signal
    )

    for rejected in rejected_targets:
        log(f"[INTAKE] target flagged: {rejected['reason']}")

    inserted = database.save_signal(
        discord_message_id=discord_message_id,
        signal=signal,
        signal_time_ms=signal_time_ms,
        price_at_signal=price_at_signal,
        raw_message=text,
        target_flags=rejected_targets
    )

    if not inserted:
        log("Signal already exists in database.")
        return None

    row = database.get_signal_by_discord_message_id(discord_message_id)
    signal_id = row["id"]

    database.record_signal_event(
        signal_id,
        "CREATED",
        new_status="DEVELOPING",
        details={
            "price_at_signal": price_at_signal,
            "rejected_targets": rejected_targets
        }
    )

    log()
    log("Signal saved as ACTIVE.")
    log(f"Price at signal: {price_at_signal}")

    #
    # Initial Telegram notification.
    #

    try:
        await asyncio.to_thread(
            services.notify,
            build_new_signal_message(signal, text, price_at_signal, rejected_targets)
        )
        log("Initial Telegram tracking notification sent.")
    except Exception as error:
        log(f"[TELEGRAM ERROR] {type(error).__name__}: {str(error)[:200]}")

    #
    # Agent 1 (optional, never breaks intake).
    #

    if config.agent1_enabled() and services.signal_agent is not None:
        await run_signal_analysis(signal_id, services)

    return signal_id


async def run_signal_analysis(signal_id, services: Services):
    """Run Agent 1 for one stored signal and persist the result."""
    from agents.signal_agent import build_input

    log = services.log

    try:
        row = database.get_signal(signal_id)

        context = None
        current_price = None

        if services.context_market is not None:
            try:
                context = await asyncio.to_thread(
                    services.context_market.get_market_context,
                    row["symbol"]
                )
            except Exception as error:
                log(f"[AGENT1] market context unavailable: {type(error).__name__}")

        if context is None:
            try:
                current_price = await asyncio.to_thread(
                    services.market.get_current_price,
                    row["symbol"]
                )
            except Exception as error:
                log(f"[AGENT1] current price unavailable: {type(error).__name__}")

        analysis_input = build_input(row, current_price, context)

        analysis = await asyncio.to_thread(
            services.signal_agent.analyze,
            analysis_input
        )

        database.save_signal_analysis(signal_id, analysis_input, analysis)

        database.record_signal_event(
            signal_id,
            "AGENT1_ANALYSIS",
            details={
                "signal_state": analysis.signal_state,
                "next_step": analysis.next_step,
                "llm_used": analysis.llm_used
            }
        )

        log()
        log("AGENT 1 ANALYSIS")
        log(analysis.model_dump_json(indent=2, exclude={"facts"}))

        return analysis

    except Exception as error:
        log(f"[AGENT1 ERROR] signal #{signal_id}: {repr(error)}")
        traceback.print_exc()
        return None


#
# Monitoring
#

def fetch_candles_since(market, symbol, start_ms):
    """
    Fetch 1m candles from start_ms forward, paging until caught up
    (bounded per tick). Only new candles are downloaded.
    """
    page_size = config.monitor_candle_page_size()
    candles = []

    for _ in range(config.monitor_max_candle_pages()):
        batch = market.get_candles(symbol, "1m", page_size, start_ms)

        if not batch:
            break

        candles.extend(batch)

        if len(batch) < page_size:
            break

        start_ms = batch[-1]["timestamp"] + MINUTE_MS

    return candles


async def monitor_signal(row, services: Services):
    log = services.log

    try:

        signal = database_row_to_signal(
            row
        )

        now_ms = services.now_ms()

        state = (
            json.loads(row["eval_state"])
            if row.get("eval_state") else None
        )

        cursor = row.get("last_processed_candle_ms")

        #
        # First run: start at the beginning of the Discord
        # message minute. Later runs: only candles newer
        # than the last fully processed (closed) candle.
        #

        if cursor is None:
            start_ms = (row["signal_time_ms"] // MINUTE_MS) * MINUTE_MS
        else:
            start_ms = cursor + MINUTE_MS

        candles = await asyncio.to_thread(
            fetch_candles_since,
            services.market,
            signal.symbol,
            start_ms
        )

        closed = [c for c in candles if c["timestamp"] + MINUTE_MS <= now_ms]
        in_progress = [c for c in candles if c["timestamp"] + MINUTE_MS > now_ms]

        valid_targets, _ = signal_rules.check_signal_targets(
            signal,
            row["price_at_signal"]
        )

        #
        # Persist running state for closed candles only; an
        # in-progress candle can still extend its high/low.
        #

        if closed:
            state = evaluate_signal(signal, closed, state, targets=valid_targets)
            cursor = closed[-1]["timestamp"]

            database.update_eval_state(row["id"], state, cursor)

        if in_progress:
            assessment = evaluate_signal(signal, in_progress, state, targets=valid_targets)
        else:
            assessment = state or new_state()

        expired = (
            assessment["status"] not in signal_rules.TERMINAL_STATUSES
            and signal_rules.is_expired(row["signal_time_ms"], signal.timeframe, now_ms)
        )

        if assessment["current_price"] is None and not expired:
            log(f"No candles for {signal.symbol}")
            return

        if expired:
            assessment = {**assessment, "status": "EXPIRED"}

        old_status = row["status"]
        new_status = assessment["status"]

        log(
            f"[MONITOR] "
            f"Signal #{row['id']} "
            f"{signal.symbol}: "
            f"{old_status} -> {new_status}"
        )

        #
        # Nothing changed.
        #

        if new_status == old_status:
            return

        #
        # Status changed.
        # Send Telegram notification. If this fails the status is
        # not updated, so the change is retried on the next tick.
        #

        message = build_status_message(
            row,
            signal,
            assessment
        )

        await asyncio.to_thread(
            services.notify,
            message
        )

        #
        # Target reached, invalidated or expired:
        # stop monitoring.
        #

        active = (
            0
            if new_status in signal_rules.TERMINAL_STATUSES
            else 1
        )

        database.update_status(
            row["id"],
            new_status,
            active
        )

        database.record_signal_event(
            row["id"],
            "STATUS_CHANGE",
            old_status=old_status,
            new_status=new_status,
            details={
                "current_price": assessment["current_price"],
                "last_processed_candle_ms": cursor
            }
        )

        log(
            f"[TELEGRAM] "
            f"Signal #{row['id']} "
            f"changed to {new_status}"
        )

        if active == 0:
            log(
                f"[CLOSED] "
                f"Signal #{row['id']} "
                f"will no longer be monitored."
            )

    except Exception as error:

        log(
            f"[MONITOR ERROR] "
            f"Signal #{row.get('id')}: "
            f"{repr(error)}"
        )

        traceback.print_exc()


async def run_monitor_cycle(services: Services):
    active_signals = database.get_active_signals()

    services.log(
        f"[MONITOR] "
        f"Active signals: "
        f"{len(active_signals)}"
    )

    for row in active_signals:
        await monitor_signal(row, services)


async def run_proposal_cycle(services: Services):
    """Loop B entry point: the Agent 2 sweep, only when enabled. Returns [(proposal_key, status)]."""
    if (
        not config.agent2_enabled()
        or services.option_chain is None
        or services.options_agent is None
    ):
        return []
    return await run_proposal_sweep(services)


#
# Agent 2 trigger: first transition to WORKING, exactly once.
#

def _iso_minus_minutes(now_ms, minutes):
    from datetime import datetime, timezone

    moment = datetime.fromtimestamp(now_ms / 1000 - minutes * 60, timezone.utc)
    return moment.isoformat(timespec="seconds")


async def run_proposal_sweep(services: Services):
    """
    For every active WORKING signal with first_working_at set and no
    trade_proposals row: claim the unique proposal key, run the gates,
    then Agent 2. Every outcome is persisted; nothing is retried.
    """
    from agents.options_agent import proposal_key

    now_ms = services.now_ms()

    for key in database.fail_interrupted_proposals(
        _iso_minus_minutes(now_ms, config.agent2_pending_timeout_minutes())
    ):
        services.log(f"[AGENT2] {key}: interrupted PENDING claim marked FAILED (not retried)")

    finished = []

    for row in database.get_signals_awaiting_proposal():
        key = proposal_key(row["id"])
        state = json.loads(row["eval_state"]) if row.get("eval_state") else {}

        #
        # The confirming candle is persisted once it closes (normally
        # the next tick). Wait for it rather than guess the time.
        #

        if state.get("status") != "WORKING" and state.get("confirmation_time") is None:
            continue

        if not database.claim_proposal(key, row["id"]):
            continue

        await _propose_for_signal(row, state, key, services, now_ms)
        finished.append((key, (database.get_proposal(key) or {}).get("status")))

    return finished


async def _propose_for_signal(row, state, key, services: Services, now_ms):
    from agents.signal_agent import build_input, compute_facts, deterministic_state
    from schemas import OptionsSelectionInput, SignalAnalysis, ThesisSnapshot
    from tools.options_tools import (
        OPTION_TYPE_BY_STRATEGY,
        STRATEGY_BY_DIRECTION,
        build_playbook,
        check_entry_extension,
        prefilter,
    )

    log = services.log
    signal_id = row["id"]

    def finish(status, reason=None, error=None, proposal=None, selection_input=None, trigger=None):
        database.finish_proposal(
            key, status, proposal=proposal, selection_input=selection_input,
            reason=reason, error=error, trigger=trigger
        )
        database.record_signal_event(
            signal_id,
            f"PROPOSAL_{status}",
            details={"proposal_key": key, "reason": proposal.reason if proposal else reason, "error": error}
        )
        log(f"[AGENT2] Signal #{signal_id}: {status} - {(proposal.reason if proposal else reason) or error}")

    try:
        #
        # Gate b: freshness of the WORKING transition.
        #

        confirmation_ms = state.get("confirmation_time")
        max_age = config.agent2_max_confirmation_age_minutes()

        if confirmation_ms is None:
            return finish("SKIPPED", "WORKING without a recorded confirmation time")

        age_minutes = (now_ms - confirmation_ms) / 60_000

        if age_minutes > max_age:
            return finish(
                "SKIPPED",
                f"WORKING detected {age_minutes:.1f} min after confirmation (max {max_age:g}); too late"
            )

        #
        # Gate a: Agent 1 analysis must exist and be tradeable.
        #

        analysis_row = database.get_latest_signal_analysis(signal_id)

        if analysis_row is None and services.signal_agent is not None:
            await run_signal_analysis(signal_id, services)
            analysis_row = database.get_latest_signal_analysis(signal_id)

        if analysis_row is None:
            return finish("SKIPPED", "no Agent 1 analysis available")

        analysis = SignalAnalysis.model_validate_json(analysis_row["output_json"])

        if analysis.signal_state not in ("DEVELOPING", "ALREADY_CONFIRMED"):
            return finish("SKIPPED", f"Agent 1 state is {analysis.signal_state}")

        #
        # Gate c: deterministic staleness re-check at the current price.
        #

        current_price = await asyncio.to_thread(services.market.get_current_price, row["symbol"])
        recheck_input = build_input(row, current_price=current_price)
        facts = compute_facts(recheck_input, now_ms)
        recheck_state, recheck_reasons = deterministic_state(recheck_input, facts)

        if recheck_state not in ("DEVELOPING", "ALREADY_CONFIRMED"):
            return finish("SKIPPED", f"staleness re-check: {recheck_state}: {'; '.join(recheck_reasons)}")

        #
        # Gate d: trigger type.
        #

        trigger = (
            "FIRST_WORKING" if analysis.signal_state == "DEVELOPING"
            else "ALREADY_CONFIRMED_FRESH"
        )

        strategy = STRATEGY_BY_DIRECTION.get(facts.direction)

        if strategy is None or facts.default_target is None:
            return finish("SKIPPED", "no directional strategy or target", trigger=trigger)

        #
        # Gate e: stale entry / move-extension (chase protection).
        # Runs before the chain is fetched, ranked or shown to the LLM.
        # Entry reference = confirmation level; for immediate-entry
        # setups (no confirmation level) the price at signal.
        #

        entry_reference = facts.default_entry_level
        if entry_reference is None:
            entry_reference = row.get("price_at_signal")

        allowed, move_pct, extension_reason = check_entry_extension(
            facts.direction,
            entry_reference,
            current_price,
            config.agent2_max_underlying_move_pct()
        )

        if not allowed:
            return finish("SKIPPED", extension_reason, trigger=trigger)

        thesis = ThesisSnapshot(
            underlying=row["symbol"],
            direction=facts.direction,
            entry_level=facts.default_entry_level,
            target=facts.default_target,
            invalidation=facts.default_invalidation,
            time_horizon=analysis.time_horizon,
            agent1_state=analysis.signal_state
        )

        #
        # Chain -> deterministic pre-filter -> Agent 2.
        #

        playbook = build_playbook()

        chain = await asyncio.to_thread(
            services.option_chain.get_option_chain,
            row["symbol"],
            option_type=OPTION_TYPE_BY_STRATEGY[strategy]
        )

        candidates, summary = prefilter(chain, thesis, playbook, now_ms)

        selection_input = OptionsSelectionInput(
            signal_id=signal_id,
            analysis_id=analysis_row["id"],
            trigger=trigger,
            thesis=thesis,
            underlying_price=chain.underlying_price,
            as_of_ms=chain.as_of_ms,
            playbook=playbook,
            chain_source=chain.source,
            candidates=candidates,
            filter_summary=summary
        )

        proposal = await asyncio.to_thread(services.options_agent.propose, selection_input)

        finish(proposal.status, proposal=proposal, selection_input=selection_input)

        if proposal.status == "PROPOSED":
            log(proposal.model_dump_json(indent=2, exclude={"filter_summary"}))

    except Exception as error:
        finish("FAILED", error=f"{type(error).__name__}: {str(error)[:300]}")
        traceback.print_exc()
