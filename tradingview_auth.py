"""
TradingView MCP OAuth for the Python runtime — the app's OWN client.

Never reads, copies or reuses Claude Code's TradingView OAuth token or session.

Flow (OAuth 2.1 public client, as advertised by TradingView's metadata):
    1. dynamic client registration   POST /mcp/oauth/register  (token_endpoint_auth_method=none)
    2. authorization code + PKCE S256 (browser, loopback redirect)
    3. code exchange                  POST /mcp/oauth/token
    4. refresh (rotating)             POST /mcp/oauth/token  grant_type=refresh_token

Credentials live OUTSIDE the repository (config.tradingview_config_dir()):
    client.json   client registration (client_id, redirect_uri)
    tokens.json   access/refresh tokens + expiry
The directory is chmod 700 and both files chmod 600, written atomically.

Secrets are never logged, printed, put in exception messages or in repr().
"""
import base64
import hashlib
import json
import os
import secrets
import tempfile
import time
from pathlib import Path
from urllib.parse import urlencode

import config


AUTHORIZATION_ENDPOINT = "https://www.tradingview.com/mcp/oauth/authorize"
TOKEN_ENDPOINT = "https://www.tradingview.com/mcp/oauth/token"
REGISTRATION_ENDPOINT = "https://www.tradingview.com/mcp/oauth/register"
SCOPES = "mcp:read mcp:tools"
CLIENT_NAME = "discord-trade-monitor (paper trading runtime)"

REFRESH_SKEW_SECONDS = 60


class AuthRequired(RuntimeError):
    """No usable TradingView credentials (never contains secret material)."""


#
# Storage
#

class TokenStore:

    def __init__(self, directory: Path | None = None):
        self.directory = Path(directory) if directory else config.tradingview_config_dir()

    def __repr__(self):
        return f"TokenStore(directory={str(self.directory)!r})"

    @property
    def client_path(self):
        return self.directory / "client.json"

    @property
    def tokens_path(self):
        return self.directory / "tokens.json"

    def _ensure_dir(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        os.chmod(self.directory, 0o700)

    def _write(self, path: Path, data: dict):
        self._ensure_dir()
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".tmp-", suffix=".json")
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w") as fh:
                json.dump(data, fh)
            os.replace(tmp, path)
            os.chmod(path, 0o600)
        except BaseException:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise

    def _read(self, path: Path):
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def load_client(self):
        return self._read(self.client_path)

    def save_client(self, data):
        self._write(self.client_path, data)

    def load_tokens(self):
        return self._read(self.tokens_path)

    def save_tokens(self, data):
        self._write(self.tokens_path, data)


#
# PKCE
#

def pkce_pair():
    verifier = secrets.token_urlsafe(64)[:96]
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


#
# OAuth client
#

def _default_post(url, *, data=None, json_body=None, timeout=20):
    import requests
    return requests.post(url, data=data, json=json_body, timeout=timeout,
                         headers={"Accept": "application/json"})


def _safe_error(response):
    """OAuth error code only (never the raw body, which could echo credentials)."""
    try:
        body = response.json()
        return f"HTTP {response.status_code} {body.get('error', '')}".strip()
    except Exception:
        return f"HTTP {response.status_code}"


class TradingViewAuth:

    def __init__(self, store: TokenStore | None = None, http_post=None, now=time.time, log=print,
                 resource=None):
        self.store = store or TokenStore()
        self._post = http_post or _default_post
        self._now = now
        self._log = log
        self.resource = resource or config.tradingview_mcp_url()
        self.state = "unknown"          # unknown | not_authenticated | authenticated | refresh_failed

    def __repr__(self):
        return f"TradingViewAuth(store={self.store!r}, state={self.state!r})"

    # -- status -------------------------------------------------------------

    def has_credentials(self) -> bool:
        tokens = self.store.load_tokens() or {}
        client = self.store.load_client() or {}
        ok = bool(tokens.get("access_token")) or (bool(tokens.get("refresh_token")) and bool(client.get("client_id")))
        if not ok:
            self.state = "not_authenticated"
        return ok

    # -- registration + authorization (setup script) --------------------------

    def register_client(self, redirect_uri: str) -> str:
        existing = self.store.load_client()
        if existing and existing.get("client_id") and existing.get("redirect_uri") == redirect_uri:
            return existing["client_id"]

        response = self._post(REGISTRATION_ENDPOINT, json_body={
            "client_name": CLIENT_NAME,
            "redirect_uris": [redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
            "scope": SCOPES,
        })
        if response.status_code not in (200, 201):
            raise AuthRequired(f"client registration failed: {_safe_error(response)}")

        body = response.json()
        client_id = body.get("client_id")
        if not client_id:
            raise AuthRequired("client registration returned no client_id")

        self.store.save_client({
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "registered_at": int(self._now()),
            "registration_endpoint": REGISTRATION_ENDPOINT,
        })
        return client_id

    def authorization_url(self, client_id, redirect_uri, state, code_challenge) -> str:
        return AUTHORIZATION_ENDPOINT + "?" + urlencode({
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": SCOPES,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "resource": self.resource,
        })

    def exchange_code(self, code, code_verifier, client_id, redirect_uri):
        response = self._post(TOKEN_ENDPOINT, data={
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": code_verifier,
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "resource": self.resource,
        })
        if response.status_code != 200:
            raise AuthRequired(f"authorization code exchange failed: {_safe_error(response)}")
        self._store_token_response(response.json(), previous_refresh=None)
        self.state = "authenticated"

    # -- runtime ----------------------------------------------------------------

    def access_token(self, force_refresh=False) -> str:
        """A valid access token, refreshing when (nearly) expired. Raises AuthRequired."""
        tokens = self.store.load_tokens()
        if not tokens:
            self.state = "not_authenticated"
            raise AuthRequired("TradingView MCP not authenticated (run scripts/setup_tradingview_mcp.py)")

        expires_at = tokens.get("expires_at")
        fresh = tokens.get("access_token") and (
            expires_at is None or expires_at - REFRESH_SKEW_SECONDS > self._now()
        )
        if fresh and not force_refresh:
            self.state = "authenticated"
            return tokens["access_token"]

        return self.refresh(tokens)

    def refresh(self, tokens=None) -> str:
        tokens = tokens or self.store.load_tokens() or {}
        client = self.store.load_client() or {}
        refresh_token = tokens.get("refresh_token")

        if not refresh_token or not client.get("client_id"):
            self.state = "not_authenticated"
            raise AuthRequired("TradingView MCP credentials cannot be refreshed; re-run setup")

        try:
            response = self._post(TOKEN_ENDPOINT, data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client["client_id"],
                "resource": self.resource,
            })
        except Exception as error:
            # Network problem: keep the refresh token, try again later.
            self.state = "refresh_failed"
            raise AuthRequired(f"TradingView token refresh failed ({type(error).__name__})") from None

        if response.status_code != 200:
            self.state = "refresh_failed"
            raise AuthRequired(f"TradingView token refresh rejected: {_safe_error(response)}")

        access = self._store_token_response(response.json(), previous_refresh=refresh_token)
        self.state = "authenticated"
        self._log("[MARKET] TradingView MCP access token refreshed")
        return access

    def _store_token_response(self, body, previous_refresh):
        access = body.get("access_token")
        if not access:
            raise AuthRequired("token endpoint returned no access token")
        expires_in = body.get("expires_in")
        self.store.save_tokens({
            "access_token": access,
            # Rotating refresh tokens: keep the new one; fall back to the previous if none issued.
            "refresh_token": body.get("refresh_token") or previous_refresh,
            "token_type": body.get("token_type", "Bearer"),
            "scope": body.get("scope", SCOPES),
            "expires_at": (self._now() + float(expires_in)) if expires_in else None,
            "obtained_at": self._now(),
        })
        return access
