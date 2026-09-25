import os

from dotenv import load_dotenv
from openai import OpenAI

from models import TradeSignal


load_dotenv()


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.4"
)


SYSTEM_PROMPT = """
You extract cryptocurrency trading theses from Discord messages.

Your job is NOT to determine whether the analyst is correct.

Your job is only to understand and structure the analyst's market thesis.

Identify:

- whether the message is actually trading/market analysis
- cryptocurrency symbol
- bias: bullish, bearish, neutral, conditional_bullish, or conditional_bearish
- timeframe if explicitly stated or strongly implied
- support levels
- resistance levels
- price targets
- confirmation-above level
- confirmation-below level
- invalidation-above level
- invalidation-below level
- concise summary

Normalize common cryptocurrency symbols:

BTC -> BTCUSDT
ETH -> ETHUSDT
SOL -> SOLUSDT
XRP -> XRPUSDT
DOGE -> DOGEUSDT

Important rules:

1. Do not invent price levels.
2. Do not invent targets.
3. Do not invent invalidation levels.
4. Do not provide financial advice.
5. If the message is only conversation, news, greetings, or unrelated content,
   set is_trade_signal=false.
"""


def parse_discord_message(message: str) -> TradeSignal:

    if not message or not message.strip():
        return TradeSignal(
            is_trade_signal=False
        )

    response = client.responses.parse(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": message
            }
        ],
        text_format=TradeSignal
    )

    parsed = response.output_parsed

    if parsed is None:
        raise RuntimeError(
            "OpenAI did not return a structured TradeSignal"
        )

    return parsed
