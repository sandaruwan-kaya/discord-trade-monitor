"""
Live Agent 1 demo: real OpenAI parse + real market context + real Agent 1.
No database writes, no Telegram. Hits the network; never collected by pytest.

    python scripts/agent1_demo.py "BTC break above 84,000 targets 85,200, invalid below 82,600"
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.signal_agent import SignalAnalysisAgent
from llm_client import OpenAILLMClient
from llm_parser import parse_discord_message
from schemas import SignalAnalysisInput
from tools.market_tools import BinanceUSProvider, build_market_context_provider


def run(text, context_provider, agent):
    print("=" * 70)
    print("DISCORD MESSAGE:", text)

    signal = parse_discord_message(text)

    if not signal.is_trade_signal or not signal.symbol:
        print("PARSER: is_trade_signal =", signal.is_trade_signal, "| symbol =", signal.symbol)
        print("-> ignored by existing intake; Agent 1 is never called")
        return

    now_ms = int(time.time() * 1000)
    context = context_provider.get_market_context(signal.symbol)

    inp = SignalAnalysisInput(
        discord_message_id="demo",
        raw_message=text,
        symbol=signal.symbol,
        bias=signal.bias,
        timeframe=signal.timeframe,
        summary=signal.summary,
        support_levels=signal.support_levels,
        resistance_levels=signal.resistance_levels,
        targets=signal.targets,
        confirmation_above=signal.confirmation_above,
        confirmation_below=signal.confirmation_below,
        invalidation_above=signal.invalidation_above,
        invalidation_below=signal.invalidation_below,
        signal_time_ms=now_ms,
        price_at_signal=context.current_price,
        current_price=context.current_price,
        market_context=context,
    )

    print("\nAGENT 1 INPUT:")
    print(inp.model_dump_json(indent=2, exclude_none=True))

    result = agent.analyze(inp)

    print("\nAGENT 1 OUTPUT:")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    messages = sys.argv[1:] or [
        "BTC holding 83,000 support. Clean break above 84,000 opens 85,200. "
        "Lose 82,600 and the idea is dead. 1h chart.",
    ]

    context_provider = build_market_context_provider()
    agent = SignalAnalysisAgent(OpenAILLMClient())

    for message in messages:
        run(message, context_provider, agent)
