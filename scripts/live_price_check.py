"""Print the latest 1m BTCUSDT candles from Binance.US. Hits the network."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from market_data import get_candles


candles = get_candles(
    "BTCUSDT",
    interval="1m",
    limit=5
)

for c in candles:
    print(c)
