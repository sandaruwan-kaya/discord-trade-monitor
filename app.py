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
    BinanceUSProvider,
    build_market_context_provider
)
from tools.options_tools import MockDeribitChainProvider
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
# The evaluator keeps using Binance.US candles (one consistent
# feed per signal). Agent context prefers TradingView MCP and
# falls back to Binance.US.
#

binance_us = BinanceUSProvider()

services = Services(
    parse_message=parse_discord_message,
    market=binance_us,
    notify=send_telegram,
    context_market=build_market_context_provider(),
    signal_agent=SignalAnalysisAgent(OpenAILLMClient()),
    # V1: mock Deribit-style chain priced off the same Binance.US feed
    # the evaluator uses. Proposals only; no broker is wired.
    option_chain=MockDeribitChainProvider(binance_us),
    options_agent=OptionsSelectionAgent(OpenAILLMClient())
)


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


if __name__ == "__main__":

    asyncio.run(
        main()
    )
