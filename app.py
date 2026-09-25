import os
import json
import asyncio
import traceback

import discord
from dotenv import load_dotenv

from models import TradeSignal
from llm_parser import parse_discord_message
from market_data import (
    get_current_price,
    get_candles
)
from evaluator import evaluate_signal
from telegram_client import send_telegram

from database import (
    initialize_database,
    save_signal,
    get_active_signals,
    update_status
)


load_dotenv()


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
# Convert database row back into
# our Pydantic TradeSignal model.
#

def database_row_to_signal(row):

    return TradeSignal(
        is_trade_signal=True,

        symbol=row["symbol"],
        bias=row["bias"],
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


#
# Check one active signal
#

async def monitor_signal(row):

    try:

        signal = database_row_to_signal(
            row
        )

        #
        # Round down to the beginning
        # of the Discord message minute.
        #

        minute_ms = 60 * 1000

        market_start_ms = (
            row["signal_time_ms"]
            // minute_ms
        ) * minute_ms

        #
        # Market API is synchronous,
        # so run it outside Discord's event loop.
        #

        candles = await asyncio.to_thread(
            get_candles,
            signal.symbol,
            "1m",
            200,
            market_start_ms
        )

        if not candles:

            print(
                f"No candles for {signal.symbol}"
            )

            return

        assessment = evaluate_signal(
            signal,
            candles
        )

        old_status = row["status"]
        new_status = assessment["status"]

        print(
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
        # Send Telegram notification.
        #

        message = build_status_message(
            row,
            signal,
            assessment
        )

        await asyncio.to_thread(
            send_telegram,
            message
        )

        #
        # Target reached or invalidated:
        # stop monitoring.
        #

        terminal_statuses = {
            "TARGET_HIT",
            "INVALIDATED"
        }

        active = (
            0
            if new_status in terminal_statuses
            else 1
        )

        update_status(
            row["id"],
            new_status,
            active
        )

        print(
            f"[TELEGRAM] "
            f"Signal #{row['id']} "
            f"changed to {new_status}"
        )

        if active == 0:

            print(
                f"[CLOSED] "
                f"Signal #{row['id']} "
                f"will no longer be monitored."
            )

    except Exception as error:

        print(
            f"[MONITOR ERROR] "
            f"Signal #{row.get('id')}: "
            f"{repr(error)}"
        )

        traceback.print_exc()


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

            active_signals = (
                get_active_signals()
            )

            print(
                f"[MONITOR] "
                f"Active signals: "
                f"{len(active_signals)}"
            )

            for row in active_signals:

                await monitor_signal(
                    row
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

        #
        # OpenAI parsing
        #

        signal = await asyncio.to_thread(
            parse_discord_message,
            text
        )

        print()
        print("PARSED SIGNAL")

        print(
            signal.model_dump_json(
                indent=2
            )
        )

        #
        # Ignore chatter.
        #

        if not signal.is_trade_signal:

            print(
                "Ignored: not a trade signal."
            )

            return

        if not signal.symbol:

            print(
                "Ignored: no market symbol."
            )

            return

        #
        # Get price when the Discord
        # message was received.
        #

        price_at_signal = (
            await asyncio.to_thread(
                get_current_price,
                signal.symbol
            )
        )

        signal_time_ms = int(
            message.created_at.timestamp()
            * 1000
        )

        #
        # Save signal.
        #

        inserted = save_signal(
            discord_message_id=message.id,
            signal=signal,
            signal_time_ms=signal_time_ms,
            price_at_signal=price_at_signal
        )

        if not inserted:

            print(
                "Signal already exists "
                "in database."
            )

            return

        print()
        print(
            "Signal saved as ACTIVE."
        )

        print(
            f"Price at signal: "
            f"{price_at_signal}"
        )

        #
        # Initial Telegram notification.
        #

        telegram_text = f"""
🆕 NEW DISCORD TRADE SIGNAL

Asset:
{signal.symbol}

Bias:
{signal.bias or "Unknown"}

Thesis:
{signal.summary or text[:300]}

Price when posted:
{price_at_signal}

Status:
TRACKING

The system will now monitor this signal automatically.
""".strip()

        await asyncio.to_thread(
            send_telegram,
            telegram_text
        )

        print(
            "Initial Telegram "
            "tracking notification sent."
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
    # Create SQLite table if required.
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