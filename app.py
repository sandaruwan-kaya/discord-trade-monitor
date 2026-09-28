import os
import asyncio
import traceback

import discord
from dotenv import load_dotenv

from llm_parser import parse_discord_message
from telegram_client import send_telegram

import config
from database import initialize_database
from llm_client import OpenAILLMClient
from agents.signal_agent import SignalAnalysisAgent
from tools.market_tools import (
    BinanceUSProvider,
    build_market_context_provider
)
from signal_service import (
    Services,
    database_row_to_signal,  # noqa: F401  (kept importable from app)
    build_status_message,    # noqa: F401
    process_discord_message,
    run_monitor_cycle
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

services = Services(
    parse_message=parse_discord_message,
    market=BinanceUSProvider(),
    notify=send_telegram,
    context_market=build_market_context_provider(),
    signal_agent=SignalAnalysisAgent(OpenAILLMClient())
)


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

    while not client.is_closed():

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

        await asyncio.sleep(
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

async def main():

    #
    # Create / migrate SQLite tables if required.
    #

    initialize_database()

    #
    # Start background monitor.
    #

    asyncio.create_task(
        signal_monitor()
    )

    #
    # Start Discord.
    #

    await client.start(
        DISCORD_BOT_TOKEN
    )


if __name__ == "__main__":

    asyncio.run(
        main()
    )
