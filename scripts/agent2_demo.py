"""
Live Agent 2 demo: real Binance.US price -> mock Deribit-style chain ->
deterministic pre-filter -> real OpenAI selection. Proposal only:
no database writes, no Telegram, no broker. Hits the network.

    python scripts/agent2_demo.py BTCUSDT bullish 85200 82600
    python scripts/agent2_demo.py ETHUSDT bearish 2540 2720
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.options_agent import OptionsSelectionAgent
from llm_client import OpenAILLMClient
from schemas import OptionsSelectionInput, ThesisSnapshot
from tools.market_tools import BinanceUSProvider
from tools.options_tools import (
    OPTION_TYPE_BY_STRATEGY,
    STRATEGY_BY_DIRECTION,
    MockDeribitChainProvider,
    build_playbook,
    prefilter,
)


def main(symbol, direction, target, invalidation):
    market = BinanceUSProvider()
    provider = MockDeribitChainProvider(market)
    now_ms = int(time.time() * 1000)

    thesis = ThesisSnapshot(
        underlying=symbol, direction=direction, entry_level=None,
        target=target, invalidation=invalidation,
        time_horizon="intraday", agent1_state="DEVELOPING"
    )

    strategy = STRATEGY_BY_DIRECTION[direction]
    chain = provider.get_option_chain(symbol, option_type=OPTION_TYPE_BY_STRATEGY[strategy])
    playbook = build_playbook()
    candidates, summary = prefilter(chain, thesis, playbook, now_ms)

    inp = OptionsSelectionInput(
        signal_id=0, analysis_id=None, trigger="FIRST_WORKING", thesis=thesis,
        underlying_price=chain.underlying_price, as_of_ms=chain.as_of_ms,
        playbook=playbook, chain_source=chain.source,
        candidates=candidates, filter_summary=summary
    )

    print("=" * 70)
    print(f"{symbol} spot {chain.underlying_price} | {len(chain.contracts)} {OPTION_TYPE_BY_STRATEGY[strategy]}s in chain")
    print("FILTER SUMMARY:", summary)
    print("CANDIDATES:")
    for c in candidates:
        k = c.contract
        print(f"  {c.candidate_id} {k.contract_symbol:22} dte={k.dte:5.1f} delta={k.delta:+.2f} "
              f"bid/ask={k.bid}/{k.ask} spread={c.spread_pct:.1f}% limit={c.limit_price} qty={c.quantity} "
              f"max_loss=${c.estimated_max_loss:.2f} value@target={c.estimated_value_at_target:.1f} rr={c.reward_to_risk_at_target:+.2f}")

    proposal = OptionsSelectionAgent(OpenAILLMClient()).propose(inp)

    print("\nAGENT 2 OUTPUT (proposal only):")
    print(proposal.model_dump_json(indent=2, exclude={"filter_summary"}))


if __name__ == "__main__":
    args = sys.argv[1:] or ["BTCUSDT", "bullish", "85200", "82600"]
    main(args[0], args[1], float(args[2]), float(args[3]))
