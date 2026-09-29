"""Print the latest 1m BTCUSDT candles from TradingView MCP. Hits the network.

Needs credentials from scripts/setup_tradingview_mcp.py. Works regardless of
TRADINGVIEW_MCP_ENABLED (only touches market data, no trading).
Note BINANCE:BTCUSDT on TradingView is Binance.com, not Binance.US.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.market_tools import TradingViewMCPProvider
from tradingview_mcp_client import TradingViewMCPClient


client = TradingViewMCPClient()
provider = TradingViewMCPProvider(call_tool=client.call_tool)

candles = provider.get_candles(
    "BTCUSDT",
    interval="1m",
    limit=5
)

for c in candles:
    print(c)

print("current price:", provider.get_current_price("BTCUSDT"))
