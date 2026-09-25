python - <<'PY'
from market_data import get_candles

candles = get_candles(
    "BTCUSDT",
    interval="1m",
    limit=5
)

for c in candles:
    print(c)
PY