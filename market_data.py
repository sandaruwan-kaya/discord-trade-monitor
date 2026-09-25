import requests


BASE_URL = "https://api.binance.us"


def get_current_price(symbol: str) -> float:
    url = f"{BASE_URL}/api/v3/ticker/price"

    response = requests.get(
        url,
        params={"symbol": symbol},
        timeout=15
    )

    response.raise_for_status()

    payload = response.json()

    return float(payload["price"])


def get_candles(
    symbol: str,
    interval: str = "15m",
    limit: int = 200,
    start_ms: int | None = None,
    end_ms: int | None = None
):
    url = f"{BASE_URL}/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    if start_ms is not None:
        params["startTime"] = start_ms

    if end_ms is not None:
        params["endTime"] = end_ms

    response = requests.get(
        url,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    rows = response.json()

    candles = []

    for row in rows:
        candles.append({
            "timestamp": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5])
        })

    return candles
