"""
Runtime TradingView MCP client (official MCP Python SDK, streamable HTTP).

Responsibilities: connection, OAuth token use/refresh (via TradingViewAuth,
the app's OWN credentials), tool calls, failure normalization, health.
OAuth never lives in the agents; agents only ever see market data.

Design (V1): one short-lived MCP connection per tool call. This keeps
reconnection trivial (every call is a fresh connection) and avoids holding
anyio task groups across threads. Calls are synchronous and thread-safe so
the existing sync MarketDataProvider interface (run via asyncio.to_thread)
can use them directly.

Failure mapping (all messages are secret-free):
    no/expired credentials, refresh failed, token rejected -> ProviderUnavailable
    transport error, timeout, tool error, malformed result  -> ProviderError
The fallback chain treats both as "use Binance.US and cool down".
"""
import asyncio
import concurrent.futures
import json
import threading

import config
from tools.market_tools import ProviderError, ProviderUnavailable
from tradingview_auth import AuthRequired, TradingViewAuth


class Unauthorized(Exception):
    """The server rejected the bearer token (HTTP 401/403)."""


class TransportError(Exception):
    pass


async def sdk_call_tool(url, headers, name, arguments, timeout):
    """Default transport: official MCP SDK over streamable HTTP. Returns a plain dict."""
    import httpx2
    from mcp import Client
    from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

    statuses = []

    async def record_status(response):
        statuses.append(response.status_code)

    http = create_mcp_http_client(headers=headers, timeout=httpx2.Timeout(timeout))
    http.event_hooks["response"].append(record_status)

    try:
        async with http:
            async with Client(streamable_http_client(url, http_client=http), read_timeout_seconds=timeout) as client:
                result = await client.call_tool(name, arguments)
    except Exception as error:
        if any(s in (401, 403) for s in statuses):
            raise Unauthorized() from None
        leaf = error
        while getattr(leaf, "exceptions", None):
            leaf = leaf.exceptions[0]
        # Exception TYPE only: messages from HTTP layers can echo request details.
        raise TransportError(type(leaf).__name__) from None

    return {
        "is_error": bool(result.is_error),
        "structured": result.structured_content,
        "texts": [c.text for c in (result.content or []) if getattr(c, "text", None)],
    }


def _run_sync(coro_factory):
    """Run a coroutine to completion from sync code, whether or not a loop is running here."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro_factory())).result()


class TradingViewMCPClient:

    def __init__(self, auth: TradingViewAuth | None = None, url=None, timeout=None, transport=None, log=print):
        self.auth = auth or TradingViewAuth(log=log)
        self.url = url or config.tradingview_mcp_url()
        self.timeout = timeout or config.tradingview_mcp_timeout_seconds()
        self._transport = transport or sdk_call_tool
        self._log = log
        self._lock = threading.Lock()
        self.last_error = None          # secret-free description of the last failure
        self.calls = 0

    def __repr__(self):
        return f"TradingViewMCPClient(url={self.url!r}, status={self.status()!r})"

    # -- health -----------------------------------------------------------

    def status(self) -> str:
        if not self.auth.has_credentials():
            return "not_authenticated"
        if self.auth.state == "refresh_failed":
            return "refresh_failed"
        return "unhealthy" if self.last_error else "authenticated"

    def check(self) -> bool:
        """Cheap live validation (one 1-minute bar). Never raises."""
        try:
            payload = self.call_tool("mcp-tv-get-ohlcv", {"symbol": "BINANCE:BTCUSDT", "interval": "1m", "count": 1})
            return bool(payload.get("bars"))
        except (ProviderError, ProviderUnavailable) as error:
            self.last_error = f"{type(error).__name__}: {error}"
            return False

    def close(self):
        """Nothing persistent to close in V1 (one connection per call)."""

    # -- tool calls ---------------------------------------------------------

    def call_tool(self, name: str, arguments: dict) -> dict:
        with self._lock:
            self.calls += 1
            try:
                token = self.auth.access_token()
            except AuthRequired as error:
                self.last_error = str(error)
                raise ProviderUnavailable(str(error)) from None

            try:
                raw = self._invoke(token, name, arguments)
            except Unauthorized:
                try:
                    token = self.auth.access_token(force_refresh=True)
                    raw = self._invoke(token, name, arguments)
                except (AuthRequired, Unauthorized):
                    self.last_error = "token rejected and refresh failed"
                    raise ProviderUnavailable("TradingView MCP token rejected; re-run setup") from None
                except TransportError as error:
                    self.last_error = f"transport: {error}"
                    raise ProviderError(f"TradingView MCP transport failed ({error})") from None
            except TransportError as error:
                self.last_error = f"transport: {error}"
                raise ProviderError(f"TradingView MCP transport failed ({error})") from None

            payload = self._normalize(name, raw)
            self.last_error = None
            return payload

    def _invoke(self, token, name, arguments):
        headers = {"Authorization": f"Bearer {token}"}

        async def go():
            try:
                return await asyncio.wait_for(
                    self._transport(self.url, headers, name, arguments, self.timeout), timeout=self.timeout + 5
                )
            except asyncio.TimeoutError:
                raise TransportError("Timeout") from None

        return _run_sync(go)

    @staticmethod
    def _normalize(name, raw) -> dict:
        if not isinstance(raw, dict):
            raise ProviderError(f"{name}: malformed MCP result")
        if raw.get("is_error"):
            raise ProviderError(f"{name}: tool returned an error")

        payload = raw.get("structured")
        if not isinstance(payload, dict):
            texts = raw.get("texts") or []
            try:
                payload = json.loads(texts[0]) if texts else None
            except (json.JSONDecodeError, TypeError):
                payload = None
        if not isinstance(payload, dict):
            raise ProviderError(f"{name}: malformed MCP payload")
        return payload


def build_tradingview_client(log=print):
    """None when TRADINGVIEW_MCP_ENABLED is false (nothing is initialized or touched)."""
    if not config.tradingview_mcp_enabled():
        return None
    return TradingViewMCPClient(log=log)
