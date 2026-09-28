import os
import signal
import asyncio
import traceback

import discord
from dotenv import load_dotenv

from llm_parser import parse_discord_message
from telegram_client import send_telegram

import config
import database
from database import initialize_database
from llm_client import OpenAILLMClient
from agents.options_agent import OptionsSelectionAgent
from agents.signal_agent import SignalAnalysisAgent
from tools.market_tools import (
    build_market_context_provider,
    build_market_provider
)
from tools.options_tools import MockDeribitChainProvider
from tradingview_mcp_client import build_tradingview_client
from signal_service import (
    Services,
    database_row_to_signal,  # noqa: F401  (kept importable from app)
    build_status_message,    # noqa: F401
    process_discord_message,
    run_monitor_cycle,
    run_proposal_cycle
)


load_dotenv(config.PROJECT_DIR / ".env")


DISCORD_BOT_TOKEN = os.getenv(
    "DISCORD_BOT_TOKEN"
)

DISCORD_CHANNEL_ID = int(
    os.getenv("DISCORD_CHANNEL_ID")
)


#
# Discord configuration
#

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(
    intents=intents
)


#
# Runtime dependencies.
#
# LLM (all agents + parser): OpenAI, unchanged.
# Underlying market data:
#   TRADINGVIEW_MCP_ENABLED=false -> Binance.US only (exactly as before)
#   TRADINGVIEW_MCP_ENABLED=true  -> TradingView MCP (app's own OAuth) -> Binance.US fallback,
#                                    failed provider skipped for MARKET_PROVIDER_FAILURE_COOLDOWN_SECONDS
# Option chain: MockDeribitChainProvider (simulated bid/ask/Greeks), priced off that market feed.
# TradingView provides NO option data.
#

tradingview = build_tradingview_client()          # None unless TRADINGVIEW_MCP_ENABLED=true
market = build_market_provider(tradingview)

services = Services(
    parse_message=parse_discord_message,
    market=market,
    notify=send_telegram,
    context_market=build_market_context_provider(tradingview.call_tool if tradingview else None),
    signal_agent=SignalAnalysisAgent(OpenAILLMClient()),
    option_chain=MockDeribitChainProvider(market),
    options_agent=OptionsSelectionAgent(OpenAILLMClient())
)


def report_market_sources():
    """Startup check + clear logs of the active sources. Never raises, never logs secrets."""
    if tradingview is None:
        print("[MARKET] TradingView MCP: disabled (TRADINGVIEW_MCP_ENABLED=false)")
        print("[MARKET] Primary active provider: Binance.US")
    elif not tradingview.auth.has_credentials():
        print("[MARKET] TradingView MCP not authenticated; using Binance.US "
              "(run: venv/bin/python scripts/setup_tradingview_mcp.py)")
        for chain in (market, services.context_market):
            if hasattr(chain, "mark_unavailable"):
                chain.mark_unavailable("tradingview_mcp", "not authenticated")
        print("[MARKET] Primary active provider: Binance.US")
    elif tradingview.check():
        print("[MARKET] TradingView MCP: authenticated")
        print("[MARKET] Primary provider: TradingView MCP (BINANCE:BTCUSDT / BINANCE:ETHUSDT = Binance.com)")
        print("[MARKET] Fallback provider: Binance.US")
    else:
        print(f"[MARKET] TradingView MCP: unavailable ({tradingview.last_error}); using Binance.US")
        for chain in (market, services.context_market):
            if hasattr(chain, "mark_unavailable"):
                chain.mark_unavailable("tradingview_mcp", "startup check failed")
        print("[MARKET] Primary active provider: Binance.US")
    print("[MARKET] Option chain: MockDeribitChainProvider (simulated bid/ask/Greeks; TradingView has no options data)")


#
# Paper-trading pipeline (built only when the master switch is on; every
# component inside is additionally gated by its own flag).
#

stop_event = asyncio.Event()
pipeline = None


def build_pipeline():
    from agents.risk_agent import RiskAnalysisAgent
    from broker.factory import create_broker
    from notifications import Notifier
    from trading_pipeline import TradingPipeline

    return TradingPipeline(
        signal_services=services,
        broker=create_broker(services.option_chain),       # PaperBroker only; refuses non-paper modes
        chain_provider=services.option_chain,
        market=services.market,
        risk_agent=RiskAnalysisAgent(OpenAILLMClient()),
        notifier=Notifier(send_telegram),
        context_market=services.context_market,
    )


async def _sleep_or_stop(seconds):
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass


async def run_loop(name, seconds, cycle):
    """Isolated loop: its failures never stop Discord, the monitor, or other loops."""
    print(f"[{name}] loop started (every {seconds:g}s)")
    while not stop_event.is_set():
        try:
            await cycle()
        except Exception as error:
            print(f"[{name}] cycle failed: {type(error).__name__}: {str(error)[:300]}")
            traceback.print_exc()
        await _sleep_or_stop(seconds)
    print(f"[{name}] loop stopped")


async def proposal_entry_cycle():
    finished = await run_proposal_cycle(services)
    await asyncio.to_thread(pipeline.after_proposals, finished)
    await asyncio.to_thread(pipeline.entry_cycle)


async def risk_cycle():
    await asyncio.to_thread(pipeline.risk_cycle)


async def lifecycle_cycle():
    await asyncio.to_thread(pipeline.lifecycle_cycle)


async def exit_cycle():
    await asyncio.to_thread(pipeline.exit_cycle)


#
# Background monitor
#

async def signal_monitor():

    await client.wait_until_ready()

    print()
    print(
        "Background signal monitor started."
    )

    print(
        "Checking active signals every 60 seconds."
    )

    while not client.is_closed() and not stop_event.is_set():

        try:

            await run_monitor_cycle(
                services
            )

        except Exception as error:

            print(
                "[BACKGROUND ERROR]",
                repr(error)
            )

            traceback.print_exc()

        await _sleep_or_stop(
            60
        )


#
# Discord connected
#

@client.event
async def on_ready():

    print()
    print("=" * 60)

    print(
        f"Connected as: {client.user}"
    )

    print(
        f"Watching channel: "
        f"{DISCORD_CHANNEL_ID}"
    )

    print(
        "Trade monitor is running."
    )

    print(
        f"Agent 1: "
        f"{'ENABLED' if config.agent1_enabled() else 'disabled'}"
    )

    print(
        f"Agentic trading (master): "
        f"{'ENABLED' if config.agentic_trading_enabled() else 'disabled'}"
    )

    print(
        f"Agent 2 proposals: {'ENABLED' if config.agent2_enabled() else 'disabled'} | "
        f"paper entries: {'ENABLED' if config.paper_execution_enabled() else 'disabled'} | "
        f"Agent 3: {'ENABLED' if config.agent3_enabled() else 'disabled'} | "
        f"Agent 4: {'ENABLED' if config.agent4_enabled() else 'disabled'} | "
        f"auto exit: {'ENABLED' if config.auto_exit_enabled() else 'disabled'}"
    )

    print("=" * 60)


#
# New Discord message
#

@client.event
async def on_message(
    message
):

    if message.author == client.user:
        return

    if message.channel.id != DISCORD_CHANNEL_ID:
        return

    text = (
        message.content
        or ""
    ).strip()

    if not text:
        return

    print()
    print("=" * 60)
    print("NEW DISCORD MESSAGE")
    print("=" * 60)

    print(
        f"Author: {message.author}"
    )

    print(
        f"Time: {message.created_at}"
    )

    print(
        f"Message: {text}"
    )

    try:

        await process_discord_message(
            discord_message_id=message.id,
            text=text,
            signal_time_ms=int(
                message.created_at.timestamp()
                * 1000
            ),
            services=services
        )

    except Exception as error:

        print()
        print(
            "ERROR WHILE PROCESSING "
            "DISCORD MESSAGE"
        )

        print(
            repr(error)
        )

        traceback.print_exc()


#
# Start everything
#

async def shutdown():
    """Stop creating new work, let active cycles finish their DB writes, close Discord."""
    if stop_event.is_set():
        return
    print("\nShutdown requested: no new work will be started.")
    stop_event.set()
    if pipeline is not None:
        pipeline.stopping = True
    await client.close()


async def main():

    global pipeline

    #
    # 1. Create / migrate SQLite tables if required.
    #

    initialize_database()

    await asyncio.to_thread(report_market_sources)

    #
    # Single process: any PENDING proposal claim left over from a
    # previous run was interrupted. Mark FAILED, never retry.
    #

    for key in database.fail_interrupted_proposals(
        database.utc_now_iso()
    ):
        print(f"[AGENT2] {key}: interrupted at last shutdown, marked FAILED")

    #
    # 2-6. Startup reconciliation BEFORE any loop can create new work.
    #

    if config.agentic_trading_enabled():
        pipeline = build_pipeline()
        await asyncio.to_thread(pipeline.startup_reconcile)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.ensure_future(shutdown()))
        except NotImplementedError:
            pass

    #
    # 7. Start loops, then Discord.
    #

    tasks = [asyncio.create_task(signal_monitor())]

    if pipeline is not None:
        tasks += [
            asyncio.create_task(run_loop("AGENT2/ENTRY", config.loop_seconds("PROPOSAL", 30), proposal_entry_cycle)),
            asyncio.create_task(run_loop("RISK", config.loop_seconds("RISK", 60), risk_cycle)),
            asyncio.create_task(run_loop("LIFECYCLE", config.loop_seconds("LIFECYCLE", 60), lifecycle_cycle)),
            asyncio.create_task(run_loop("EXIT", config.loop_seconds("EXIT", 30), exit_cycle)),
        ]

    try:
        await client.start(
            DISCORD_BOT_TOKEN
        )
    finally:
        stop_event.set()
        if pipeline is not None:
            pipeline.stopping = True
        done, pending = await asyncio.wait(tasks, timeout=30)
        for task in pending:
            task.cancel()
        if tradingview is not None:
            tradingview.close()


if __name__ == "__main__":

    asyncio.run(
        main()
    )
