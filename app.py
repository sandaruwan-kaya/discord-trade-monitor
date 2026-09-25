import os
import traceback

import discord
from dotenv import load_dotenv

from evaluator import evaluate_signal
from llm_parser import parse_discord_message
from market_data import get_candles
from telegram_client import send_telegram


load_dotenv()


DISCORD_BOT_TOKEN = os.getenv(
    "DISCORD_BOT_TOKEN"
)

DISCORD_CHANNEL_ID_RAW = os.getenv(
    "DISCORD_CHANNEL_ID"
)


if not DISCORD_BOT_TOKEN:
    raise RuntimeError(
        "DISCORD_BOT_TOKEN is missing"
    )


if not DISCORD_CHANNEL_ID_RAW:
    raise RuntimeError(
        "DISCORD_CHANNEL_ID is missing"
    )


DISCORD_CHANNEL_ID = int(
    DISCORD_CHANNEL_ID_RAW
)


intents = discord.Intents.default()

intents.message_content = True


client = discord.Client(
    intents=intents
)


@client.event
async def on_ready():

    print(
        "----------------------------------"
    )

    print(
        f"Connected as: {client.user}"
    )

    print(
        f"Watching channel: {DISCORD_CHANNEL_ID}"
    )

    print(
        "Trade monitor is running."
    )

    print(
        "----------------------------------"
    )


@client.event
async def on_message(
    message
):

    #
    # Ignore ourselves.
    #

    if message.author == client.user:
        return

    #
    # Only monitor the configured channel.
    #

    if message.channel.id != DISCORD_CHANNEL_ID:
        return

    text = (
        message.content
        or ""
    ).strip()

    #
    # Ignore empty text messages for now.
    # Later we will support attached screenshots/images.
    #

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
        # STEP 1
        # Parse the Discord message with OpenAI.
        #

        signal = parse_discord_message(
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
        # Ignore non-trading chatter.
        #

        if not signal.is_trade_signal:

            print(
                "Ignored - not a trading signal."
            )

            return

        #
        # Need a market symbol to continue.
        #

        if not signal.symbol:

            print(
                "Ignored - no symbol extracted."
            )

            return

        #
        # STEP 2
        # Use the exact Discord message time.
        #

        message_time_ms = int(
            message.created_at.timestamp()
            * 1000
        )

        #
        # STEP 3
        # Get market candles beginning around
        # the Discord message time.
        #

        # Round down to the beginning of the minute.
        minute_ms = 60 * 1000

        market_start_ms = (
            message_time_ms // minute_ms
        ) * minute_ms

        candles = get_candles(
            symbol=signal.symbol,
            interval="1m",
            limit=200,
            start_ms=market_start_ms
        )

        if not candles:

            print(
                f"No market candles returned "
                f"for {signal.symbol}"
            )

            return

        #
        # STEP 4
        # Deterministically evaluate the thesis.
        #

        assessment = evaluate_signal(
            signal,
            candles
        )

        print()
        print("ASSESSMENT")

        print(
            assessment
        )

        #
        # STEP 5
        # Send Telegram message.
        #

        telegram_text = f"""
📊 DISCORD TRADE MONITOR

Asset: {signal.symbol}

Bias:
{signal.bias or "Unknown"}

Thesis:
{signal.summary or "No summary"}

Current price:
{assessment["current_price"]}

High since message:
{assessment["highest_price"]}

Low since message:
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

        send_telegram(
            telegram_text
        )

        print()
        print(
            "Telegram notification sent."
        )

    except Exception as error:

        print()
        print(
            "ERROR WHILE PROCESSING MESSAGE"
        )

        print(
            repr(error)
        )

        traceback.print_exc()


client.run(
    DISCORD_BOT_TOKEN
)
