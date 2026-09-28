"""
One-time TradingView MCP authorization for the Python runtime (its OWN OAuth
client; Claude Code's TradingView login is never read or reused).

    venv/bin/python scripts/setup_tradingview_mcp.py            # loopback callback
    venv/bin/python scripts/setup_tradingview_mcp.py --manual   # paste the redirect URL

On a headless EC2 host, either:
  * forward the callback port from your laptop first:
        ssh -L 8765:127.0.0.1:8765 ec2-user@<host>
    then open the printed URL in your laptop browser, or
  * use --manual: open the URL anywhere, approve, and paste the full URL the
    browser was redirected to (it may show "can't connect"; that's fine).

What it does: register client -> PKCE authorization -> code exchange ->
save credentials (chmod 600, outside the repo) -> validate with one
mcp-tv-get-ohlcv call for BINANCE:BTCUSDT. It does NOT start the app, the
trading loops, or change .env. Tokens are never printed.
"""
import argparse
import secrets
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from tradingview_auth import AuthRequired, TokenStore, TradingViewAuth, pkce_pair
from tradingview_mcp_client import TradingViewMCPClient


def wait_for_callback(port, timeout_seconds=600):
    result = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            query = parse_qs(urlparse(self.path).query)
            result.update({k: v[0] for k, v in query.items()})
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"TradingView authorization received. You can close this tab.")

        def log_message(self, *args):             # never echo the query (it holds the code)
            pass

    server = HTTPServer(("127.0.0.1", port), Handler)
    server.timeout = timeout_seconds
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    thread.join(timeout_seconds + 5)
    server.server_close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manual", action="store_true", help="paste the redirect URL instead of a local callback")
    parser.add_argument("--port", type=int, default=config.tradingview_oauth_redirect_port())
    args = parser.parse_args()

    store = TokenStore()
    auth = TradingViewAuth(store=store)
    redirect_uri = f"http://127.0.0.1:{args.port}/callback"

    print(f"Credential directory (outside the repo): {store.directory}")
    print(f"MCP server: {config.tradingview_mcp_url()}")

    try:
        client_id = auth.register_client(redirect_uri)
    except AuthRequired as error:
        sys.exit(f"Client registration failed: {error}")
    print("OAuth client registered for this app (client id stored, not printed).")

    verifier, challenge = pkce_pair()
    state = secrets.token_urlsafe(24)
    url = auth.authorization_url(client_id, redirect_uri, state, challenge)

    print("\nOpen this URL in a browser and approve access:\n")
    print(url)
    print()
    try:
        webbrowser.open(url)
    except Exception:
        pass

    if args.manual:
        pasted = input("Paste the FULL URL you were redirected to: ").strip()
        params = {k: v[0] for k, v in parse_qs(urlparse(pasted).query).items()}
    else:
        print(f"Waiting for the redirect on {redirect_uri} (Ctrl+C to abort; use --manual if the port "
              "is not forwarded) ...")
        params = wait_for_callback(args.port)

    if params.get("state") != state:
        sys.exit("Authorization failed: state mismatch or no callback received.")
    if "error" in params:
        sys.exit(f"Authorization failed: {params.get('error')}")
    if not params.get("code"):
        sys.exit("Authorization failed: no code in the redirect.")

    try:
        auth.exchange_code(params["code"], verifier, client_id, redirect_uri)
    except AuthRequired as error:
        sys.exit(f"Token exchange failed: {error}")
    print(f"Credentials saved to {store.tokens_path} (chmod 600).")

    client = TradingViewMCPClient(auth=auth)
    try:
        payload = client.call_tool("mcp-tv-get-ohlcv", {"symbol": "BINANCE:BTCUSDT", "interval": "1m", "count": 1})
        bars = payload.get("bars") or []
        close = bars[-1]["c"] if bars else None
    except Exception as error:
        sys.exit(f"Saved credentials, but validation failed: {type(error).__name__}: {error}")

    print(f"\nTradingView MCP connection OK (BINANCE:BTCUSDT last 1m close: {close})")
    print("Nothing else was changed. To use it at runtime set TRADINGVIEW_MCP_ENABLED=true (after approval).")


if __name__ == "__main__":
    main()
