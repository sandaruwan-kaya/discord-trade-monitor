"""TradingView MCP runtime client, OAuth and provider fallback (offline: fake transport + fake OAuth HTTP)."""
import asyncio
import json
import os
import stat

import pytest

import config
from tools.market_tools import (
    BinanceUSProvider,
    FallbackMarketDataProvider,
    ProviderError,
    ProviderUnavailable,
    TradingViewMCPProvider,
    build_market_provider,
    run_consistent,
)
from tradingview_auth import AuthRequired, TokenStore, TradingViewAuth, pkce_pair
from tradingview_mcp_client import TradingViewMCPClient, TransportError, Unauthorized, build_tradingview_client

from helpers import T0

ACCESS = "SECRET-ACCESS-TOKEN-aaaaaaaaaaaa"
ACCESS2 = "SECRET-ACCESS-TOKEN-bbbbbbbbbbbb"
REFRESH = "SECRET-REFRESH-TOKEN-cccccccccccc"
REFRESH2 = "SECRET-REFRESH-TOKEN-dddddddddddd"
SECRETS = (ACCESS, ACCESS2, REFRESH, REFRESH2, "Bearer SECRET")


class Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


class FakeOAuthHTTP:
    """Stands in for the TradingView OAuth endpoints."""

    def __init__(self):
        self.calls = []
        self.refresh_response = Resp(200, {"access_token": ACCESS2, "refresh_token": REFRESH2, "expires_in": 3600})
        self.raise_on_refresh = None

    def __call__(self, url, data=None, json_body=None, timeout=20):
        self.calls.append((url, dict(data or {}), dict(json_body or {})))
        if url.endswith("/register"):
            return Resp(201, {"client_id": "client-123"})
        if data and data.get("grant_type") == "authorization_code":
            return Resp(200, {"access_token": ACCESS, "refresh_token": REFRESH, "expires_in": 3600})
        if data and data.get("grant_type") == "refresh_token":
            if self.raise_on_refresh:
                raise self.raise_on_refresh
            return self.refresh_response
        return Resp(404, {"error": "not_found"})


def bars(n=3, base=84000.0, t0=T0 // 1000):
    return [{"t": t0 + 60 * i, "o": base + i, "h": base + i + 5, "l": base + i - 5, "c": base + i + 1, "v": 2.0}
            for i in range(n)]


class FakeTransport:
    """Fake MCP server: records the bearer token seen, returns scripted results."""

    def __init__(self):
        self.calls = []
        self.results = []                   # queue of: dict | Exception
        self.default = lambda name, args: {"is_error": False, "structured": None,
                                           "texts": [json.dumps({"success": True, "symbol": args.get("symbol"),
                                                                 "bars": bars()})]}

    async def __call__(self, url, headers, name, arguments, timeout):
        self.calls.append({"token": headers["Authorization"].split(" ", 1)[1], "name": name, "args": arguments})
        if self.results:
            r = self.results.pop(0)
            if isinstance(r, BaseException):
                raise r
            return r
        return self.default(name, arguments)


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def authed(tmp, clock=None, http=None, expires_in=3600):
    """An auth object that already went through setup (client registered + tokens)."""
    clock = clock or Clock()
    http = http or FakeOAuthHTTP()
    logs = []
    auth = TradingViewAuth(store=TokenStore(tmp), http_post=http, now=clock, log=logs.append)
    auth.register_client("http://127.0.0.1:8765/callback")
    http.calls.clear()
    auth._store_token_response({"access_token": ACCESS, "refresh_token": REFRESH, "expires_in": expires_in}, None)
    return auth, http, clock, logs


def client_with(tmp, transport=None, **kw):
    auth, http, clock, logs = authed(tmp, **kw)
    transport = transport or FakeTransport()
    return TradingViewMCPClient(auth=auth, url="https://mcp.test/mcp", transport=transport, log=logs.append), \
        transport, auth, http, clock, logs


class FakeBinance:
    name = "binance_us"
    label = "Binance.US"

    def __init__(self, price=83300.0):
        self.price = price
        self.calls = 0
        self.fail = False

    def get_current_price(self, symbol):
        self.calls += 1
        if self.fail:
            raise ConnectionError("binance.us down")
        return self.price

    def get_candles(self, symbol, interval="1m", limit=200, start_ms=None, end_ms=None):
        self.calls += 1
        if self.fail:
            raise ConnectionError("binance.us down")
        return [{"timestamp": T0 + 60_000 * i, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 1.0,
                 "source": "binance"} for i in range(limit)]

    def get_market_context(self, symbol):
        raise ConnectionError("n/a")


def chain_for(client, binance=None, clock=None, logs=None):
    binance = binance or FakeBinance()
    return FallbackMarketDataProvider(
        [TradingViewMCPProvider(call_tool=client.call_tool), binance], cooldown_seconds=300,
        clock=clock or Clock(), log=(logs.append if logs is not None else (lambda *a: None))), binance


# ---------------------------------------------------------------- storage / OAuth

def test_default_credentials_dir_is_outside_repo(monkeypatch):
    monkeypatch.delenv("TRADINGVIEW_MCP_CONFIG_DIR")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    d = config.tradingview_config_dir()
    assert str(d).endswith(".config/discord-trade-monitor/tradingview")
    assert config.PROJECT_DIR not in d.parents


def test_token_files_are_private(tmp_path):
    auth, *_ = authed(tmp_path / "tv")
    for path in (auth.store.client_path, auth.store.tokens_path):
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(auth.store.directory).st_mode) == 0o700


def test_pkce_and_authorization_url():
    verifier, challenge = pkce_pair()
    import base64, hashlib
    assert challenge == base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    auth = TradingViewAuth(store=TokenStore("/nonexistent"), http_post=FakeOAuthHTTP())
    url = auth.authorization_url("client-123", "http://127.0.0.1:8765/callback", "st4te", challenge)
    for part in ("response_type=code", "client_id=client-123", "code_challenge_method=S256",
                 f"code_challenge={challenge}", "state=st4te", "scope=mcp%3Aread+mcp%3Atools",
                 "resource=https%3A%2F%2Fmcp.tradingview.com%2Fmcp",
                 "redirect_uri=http%3A%2F%2F127.0.0.1%3A8765%2Fcallback"):
        assert part in url
    assert url.startswith("https://www.tradingview.com/mcp/oauth/authorize?")


def test_registration_is_public_client_and_exchange_stores_tokens(tmp_path):
    http = FakeOAuthHTTP()
    auth = TradingViewAuth(store=TokenStore(tmp_path / "tv"), http_post=http, now=Clock())
    assert auth.register_client("http://127.0.0.1:8765/callback") == "client-123"
    assert http.calls[0][2]["token_endpoint_auth_method"] == "none"
    auth.register_client("http://127.0.0.1:8765/callback")                  # reused, not re-registered
    assert len(http.calls) == 1
    auth.exchange_code("code-1", "verifier-1", "client-123", "http://127.0.0.1:8765/callback")
    tokens = auth.store.load_tokens()
    assert tokens["access_token"] == ACCESS and tokens["refresh_token"] == REFRESH
    assert http.calls[-1][1]["code_verifier"] == "verifier-1" and http.calls[-1][1]["resource"]


# ---------------------------------------------------------------- scenario 1: authenticated -> TV used

def test_enabled_and_authenticated_uses_tradingview(tmp_path):
    client, transport, *_ , logs = client_with(tmp_path / "tv")
    chain, binance = chain_for(client, logs=logs)

    price = chain.get_current_price("BTCUSDT")

    assert price == 84001.0 + 2                                            # last TV bar close
    assert binance.calls == 0
    assert transport.calls[0]["name"] == "mcp-tv-get-ohlcv"
    assert transport.calls[0]["args"]["symbol"] == "BINANCE:BTCUSDT"       # explicit pair, not an aggregate
    assert chain.last_provider == "tradingview_mcp"
    assert any("active provider: TradingView MCP" in l for l in logs)


def test_tradingview_candles_normalized_like_binance(tmp_path):
    client, *_ = client_with(tmp_path / "tv")
    candles = TradingViewMCPProvider(call_tool=client.call_tool, now_ms=lambda: T0 + 3 * 60_000).get_candles(
        "BTCUSDT", "1m", 10, start_ms=T0)
    assert [c["timestamp"] for c in candles] == [T0, T0 + 60_000, T0 + 120_000]
    assert set(candles[0]) == {"timestamp", "open", "high", "low", "close", "volume"}


# ---------------------------------------------------------------- scenario 2: no token -> fallback

def test_enabled_without_token_falls_back_without_calling_tradingview(tmp_path):
    transport = FakeTransport()
    client = TradingViewMCPClient(auth=TradingViewAuth(store=TokenStore(tmp_path / "tv"), http_post=FakeOAuthHTTP()),
                                  transport=transport, log=lambda *a: None)
    assert client.status() == "not_authenticated"
    chain, binance = chain_for(client)
    assert chain.get_current_price("BTCUSDT") == 83300.0
    assert transport.calls == [] and binance.calls == 1
    with pytest.raises(ProviderUnavailable):
        client.call_tool("mcp-tv-get-ohlcv", {})


# ---------------------------------------------------------------- scenario 3 / 4: refresh

def test_expired_token_is_refreshed_before_the_call(tmp_path):
    clock = Clock()
    client, transport, auth, http, clock, logs = client_with(tmp_path / "tv", clock=clock, expires_in=30)
    client.call_tool("mcp-tv-get-ohlcv", {"symbol": "BINANCE:BTCUSDT", "interval": "1m", "count": 1})
    assert transport.calls[0]["token"] == ACCESS2                          # refreshed (30s < 60s skew)
    tokens = auth.store.load_tokens()
    assert tokens["refresh_token"] == REFRESH2                             # rotated refresh token persisted
    assert http.calls[-1][1]["grant_type"] == "refresh_token"


def test_server_401_forces_one_refresh_and_retry(tmp_path):
    transport = FakeTransport()
    transport.results = [Unauthorized()]
    client, transport, auth, http, *_ = client_with(tmp_path / "tv", transport=transport)
    payload = client.call_tool("mcp-tv-get-ohlcv", {"symbol": "BINANCE:BTCUSDT"})
    assert payload["success"] is True
    assert [c["token"] for c in transport.calls] == [ACCESS, ACCESS2]


def test_401_twice_is_unavailable(tmp_path):
    transport = FakeTransport()
    transport.results = [Unauthorized(), Unauthorized()]
    client, *_ = client_with(tmp_path / "tv", transport=transport)
    with pytest.raises(ProviderUnavailable):
        client.call_tool("mcp-tv-get-ohlcv", {})


@pytest.mark.parametrize("failure", ["rejected", "network"])
def test_refresh_failure_falls_back_to_binance(tmp_path, failure):
    http = FakeOAuthHTTP()
    if failure == "rejected":
        http.refresh_response = Resp(400, {"error": "invalid_grant", "error_description": REFRESH})
    else:
        http.raise_on_refresh = ConnectionError(f"cannot reach token endpoint {REFRESH}")
    client, transport, auth, *_ = client_with(tmp_path / "tv", http=http, expires_in=10)
    chain, binance = chain_for(client)

    assert chain.get_current_price("BTCUSDT") == 83300.0
    assert transport.calls == [] and client.status() == "refresh_failed"
    assert auth.store.load_tokens()["refresh_token"] == REFRESH             # kept for a later retry
    with pytest.raises(ProviderUnavailable) as err:
        client.call_tool("mcp-tv-get-ohlcv", {})
    assert REFRESH not in str(err.value)


# ---------------------------------------------------------------- scenario 5: disconnect

def test_transport_disconnect_falls_back(tmp_path):
    transport = FakeTransport()
    transport.results = [TransportError("RemoteProtocolError")]
    client, *_ = client_with(tmp_path / "tv", transport=transport)
    chain, binance = chain_for(client)
    assert chain.get_current_price("BTCUSDT") == 83300.0
    assert chain.status()["tradingview_mcp"] == "cooling_down"


def test_real_sdk_transport_failure_is_normalized_and_secret_free(tmp_path):
    """Default (official SDK) transport against an unreachable endpoint: network is blocked in tests."""
    auth, *_ = authed(tmp_path / "tv")
    client = TradingViewMCPClient(auth=auth, url="http://127.0.0.1:9/mcp", timeout=3, log=lambda *a: None)
    with pytest.raises(ProviderError) as err:
        client.call_tool("mcp-tv-get-ohlcv", {"symbol": "BINANCE:BTCUSDT"})
    assert "transport failed" in str(err.value)
    assert not any(s in str(err.value) or s in repr(client) for s in SECRETS)
    assert client.check() is False and client.status() == "unhealthy"


# ---------------------------------------------------------------- scenario 6 / 7: cooldown

def test_cooldown_prevents_constant_reconnects_then_retries(tmp_path):
    transport = FakeTransport()
    transport.results = [TransportError("ConnectError")]
    client, *_ = client_with(tmp_path / "tv", transport=transport)
    clock = Clock(0.0)
    chain, binance = chain_for(client, clock=clock)

    for _ in range(10):
        assert chain.get_current_price("BTCUSDT") == 83300.0
        clock.t += 20                                                      # 200s total < 300s cooldown
    assert len(transport.calls) == 1                                       # tried once, then skipped

    clock.t += 150                                                         # cooldown elapsed
    assert chain.get_current_price("BTCUSDT") == 84003.0                   # TradingView retried and used
    assert len(transport.calls) == 2 and chain.last_provider == "tradingview_mcp"


def test_cooldown_setting_names(monkeypatch):
    monkeypatch.setenv("MARKET_PROVIDER_FAILURE_COOLDOWN_SECONDS", "120")
    assert config.provider_failure_cooldown_seconds() == 120
    monkeypatch.delenv("MARKET_PROVIDER_FAILURE_COOLDOWN_SECONDS")
    monkeypatch.setenv("MARKET_PROVIDER_COOLDOWN_SECONDS", "90")
    assert config.provider_failure_cooldown_seconds() == 90                # legacy name still honoured


# ---------------------------------------------------------------- scenario 8: malformed data

def _text(payload):
    return {"is_error": False, "structured": None, "texts": [payload if isinstance(payload, str) else json.dumps(payload)]}


@pytest.mark.parametrize("result", [
    _text("not json"),
    {"is_error": True, "structured": None, "texts": ["boom"]},
    _text({"success": True, "symbol": "CRYPTO:BTCUSD", "bars": bars()}),            # aggregate substituted
    _text({"success": True, "symbol": "BINANCE:BTCUSDT"}),                          # no bars
    _text({"success": True, "bars": [{"t": 1, "o": 1}]}),                           # missing fields
    _text({"success": True, "bars": [{"t": 1, "o": 10, "h": 5, "l": 8, "c": 9}]}),  # high < low/open
    _text({"success": True, "bars": [{"t": 2, "o": 1, "h": 2, "l": 1, "c": 1},
                                     {"t": 1, "o": 1, "h": 2, "l": 1, "c": 1}]}),   # time goes backwards
    _text({"success": True, "bars": [{"t": 1, "o": -1, "h": 2, "l": -2, "c": 1}]}), # non-positive price
    _text({"success": False, "error": "rate limited"}),
    "not-a-dict",
])
def test_malformed_tradingview_data_rejected_and_falls_back(tmp_path, result):
    transport = FakeTransport()
    transport.results = [result]
    client, *_ = client_with(tmp_path / "tv", transport=transport)
    chain, binance = chain_for(client)
    assert chain.get_current_price("BTCUSDT") == 83300.0
    assert chain.status()["tradingview_mcp"] == "cooling_down"


# ---------------------------------------------------------------- source consistency

def test_evaluation_never_mixes_sources(tmp_path):
    """TV serves page 1 then fails on page 2: the whole fetch is redone on Binance.US."""
    from signal_service import fetch_candles_since

    pages = {"n": 0}

    class FlakyTV:
        name = "tradingview_mcp"

        def get_candles(self, symbol, interval="1m", limit=200, start_ms=None, end_ms=None):
            pages["n"] += 1
            if pages["n"] > 1:
                raise ProviderError("disconnected mid-evaluation")
            return [{"timestamp": start_ms + 60_000 * i, "open": 1, "high": 2, "low": 0.5, "close": 1.5,
                     "volume": 1, "source": "tv"} for i in range(limit)]

    binance = FakeBinance()
    chain = FallbackMarketDataProvider([FlakyTV(), binance], clock=Clock(), log=lambda *a: None)
    os.environ["MONITOR_CANDLE_PAGE_SIZE"] = "5"
    try:
        source, candles = run_consistent(chain, lambda p: fetch_candles_since(p, "BTCUSDT", T0), "candles")
    finally:
        del os.environ["MONITOR_CANDLE_PAGE_SIZE"]
    assert source == "binance_us"
    assert {c["source"] for c in candles} == {"binance"}


# ---------------------------------------------------------------- scenario 9 / 10: integration

def test_both_providers_down_fails_closed_no_orders(monkeypatch):
    from integration_harness import Harness
    from test_integration import ALL_FLAGS, bull_to_working, n, q

    for f in ALL_FLAGS:
        monkeypatch.setenv(f, "true")
    h = Harness()
    h.feed.fail = True                                                      # nothing can price anything
    from models import TradeSignal
    from integration_harness import BULL
    sid = h.post("m1", BULL)
    assert q("SELECT price_at_signal FROM signals")[0]["price_at_signal"] is None   # no invented price
    for m in (1, 10.5, 11.5, 12.5):
        h.at(m)
        h.cycle()
    assert q("SELECT status FROM signals")[0]["status"] == "DEVELOPING"     # no evaluation without data
    assert n("paper_orders") == 0 and n("trades") == 0 and n("execution_attempts") == 0


def test_provider_switching_does_not_duplicate_state(monkeypatch, tmp_path):
    """TradingView flaps on/off across cycles; signal/proposal/order/trade state stays single."""
    from integration_harness import Harness
    from test_integration import ALL_FLAGS, bull_open, n, q

    for f in ALL_FLAGS:
        monkeypatch.setenv(f, "true")
    h = Harness()
    feed = h.feed

    class FeedAsTV:
        """A 'TradingView' provider serving the same scripted prices, switchable on/off."""
        name = "tradingview_mcp"
        up = True

        def _check(self):
            if not self.up:
                raise ProviderError("tradingview down")

        def get_current_price(self, symbol):
            self._check()
            return feed.get_current_price(symbol)

        def get_candles(self, *a, **k):
            self._check()
            return feed.get_candles(*a, **k)

    tv = FeedAsTV()
    clock = Clock(0.0)
    chain = FallbackMarketDataProvider([tv, feed], cooldown_seconds=0, clock=clock, log=lambda *a: None)
    h.services.market = chain
    h.chain.market = chain
    h.pipeline.execution.market = chain
    h.pipeline.risk.market = chain

    bull_open(h)
    for minute in range(13, 20):
        tv.up = minute % 2 == 0                                             # flap every cycle
        h.at(minute + 0.5)
        h.cycle()

    assert n("signals") == 1 and n("trade_proposals") == 1
    assert n("execution_attempts", "purpose = 'ENTRY'") == 1 and n("paper_orders") == 1 and n("trades") == 1
    sources = {json.loads(e["details"]).get("market_source") for e in q(
        "SELECT details FROM signal_events WHERE event_type = 'STATUS_CHANGE'")}
    assert sources <= {"tradingview_mcp", "fake"}


# ---------------------------------------------------------------- secrets / wiring

def test_no_secret_ever_logged_or_raised(tmp_path, capsys):
    http = FakeOAuthHTTP()
    http.refresh_response = Resp(400, {"error": "invalid_grant", "leak": REFRESH2})
    transport = FakeTransport()
    transport.results = [Unauthorized(), TransportError("ConnectError")]
    client, transport, auth, http, clock, logs = client_with(tmp_path / "tv", transport=transport, http=http)
    chain, _ = chain_for(client, logs=logs)
    messages = []
    for _ in range(3):
        try:
            client.call_tool("mcp-tv-get-ohlcv", {"symbol": "BINANCE:BTCUSDT"})
        except Exception as e:
            messages.append(f"{type(e).__name__}: {e!r} {e}")
        chain.get_current_price("BTCUSDT")
        clock.t += 1000
    text = "\n".join(logs + messages + [repr(client), repr(auth), repr(auth.store), capsys.readouterr().out])
    for secret in SECRETS:
        assert secret not in text


def test_disabled_flag_initializes_nothing(tmp_path, monkeypatch, temp_tradingview_dir):
    monkeypatch.setenv("TRADINGVIEW_MCP_ENABLED", "false")
    assert build_tradingview_client() is None
    assert type(build_market_provider(None)) is BinanceUSProvider
    assert not temp_tradingview_dir.exists()


def test_enabled_flag_builds_client_without_network_or_files(monkeypatch, temp_tradingview_dir):
    monkeypatch.setenv("TRADINGVIEW_MCP_ENABLED", "true")
    client = build_tradingview_client(log=lambda *a: None)
    assert isinstance(client, TradingViewMCPClient) and client.status() == "not_authenticated"
    chain = build_market_provider(client, log=lambda *a: None)
    assert [p.name for p in chain.providers] == ["tradingview_mcp", "binance_us"]
    assert not temp_tradingview_dir.exists()


def test_app_wiring_openai_agents_mock_chain_and_binance_default():
    import app
    from llm_client import OpenAILLMClient
    from tools.options_tools import MockDeribitChainProvider

    assert app.tradingview is None and type(app.market) is BinanceUSProvider
    assert type(app.services.option_chain) is MockDeribitChainProvider            # options stay mock
    assert isinstance(app.services.signal_agent.llm, OpenAILLMClient)
    assert isinstance(app.services.options_agent.llm, OpenAILLMClient)
    import inspect
    assert "RiskAnalysisAgent(OpenAILLMClient())" in inspect.getsource(app.build_pipeline)
    assert "parse_message=parse_discord_message" in inspect.getsource(app)


def test_no_anthropic_or_second_llm_provider_added():
    reqs = (config.PROJECT_DIR / "requirements.txt").read_text().lower()
    assert "anthropic" not in reqs and "openai==" in reqs
    import pathlib
    for path in pathlib.Path(config.PROJECT_DIR).rglob("*.py"):
        if "venv" in path.parts or "tests" in path.parts:
            continue
        assert "import anthropic" not in path.read_text() and "from anthropic" not in path.read_text(), path


def test_tradingview_provides_no_option_chain():
    from tools import options_tools
    assert not hasattr(TradingViewMCPProvider, "get_option_chain")
    import inspect
    assert "tradingview" not in inspect.getsource(options_tools).lower()
