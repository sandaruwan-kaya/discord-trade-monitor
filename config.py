"""
Central, env-driven configuration.

Every value can be overridden in .env. Values are read at call
time (via functions) where tests need to override them.

Never log secrets from here.
"""
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent

load_dotenv(PROJECT_DIR / ".env")


def _float(name, default):
    value = os.getenv(name)
    return float(value) if value not in (None, "") else float(default)


def _int(name, default):
    value = os.getenv(name)
    return int(value) if value not in (None, "") else int(default)


def _bool(name, default):
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


#
# Database
#

def db_path() -> str:
    """Absolute DB path. TRADE_MONITOR_DB overrides (used by tests)."""
    override = os.getenv("TRADE_MONITOR_DB")
    if override:
        return str(Path(override).resolve())
    return str(PROJECT_DIR / "trade_monitor.db")


#
# Signal monitor
#

def monitor_candle_page_size() -> int:
    # Binance.US klines maximum is 1000.
    return _int("MONITOR_CANDLE_PAGE_SIZE", 1000)


def monitor_max_candle_pages() -> int:
    # Max pages fetched per signal per tick while catching up.
    return _int("MONITOR_MAX_CANDLE_PAGES", 10)


#
# Signal expiry, in hours, by timeframe bucket.
#

def signal_expiry_hours(bucket: str) -> float:
    defaults = {
        "scalp": 6,
        "intraday": 24,
        "swing": 120,
        "position": 720,
        "unknown": 48,
    }
    return _float(
        f"SIGNAL_EXPIRY_HOURS_{bucket.upper()}",
        defaults.get(bucket, defaults["unknown"])
    )


#
# Agent 1
#

def agent1_enabled() -> bool:
    # Off by default: the existing intake flow is unchanged
    # until this is switched on.
    return _bool("AGENT1_ENABLED", False)


def agent1_max_target_progress_pct() -> float:
    # An already-confirmed setup that has covered more than this
    # share of the entry->target distance is STALE.
    return _float("AGENT1_MAX_TARGET_PROGRESS_PCT", 50)


def agent1_require_invalidation() -> bool:
    return _bool("AGENT1_REQUIRE_INVALIDATION", True)


def agent_model() -> str:
    return os.getenv("AGENT_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-5.4"


#
# Market data providers
#

def market_context_providers() -> list[str]:
    # Order matters: first healthy provider wins.
    raw = os.getenv("MARKET_CONTEXT_PROVIDERS", "tradingview_mcp,binance_us")
    return [p.strip() for p in raw.split(",") if p.strip()]


def provider_failure_cooldown_seconds() -> float:
    return _float("MARKET_PROVIDER_COOLDOWN_SECONDS", 300)


#
# Paper-trading risk defaults (demo values only; consumed from Step 5 on).
#

TRADING_MODE = os.getenv("TRADING_MODE", "paper")

MAX_PREMIUM_PER_TRADE = _float("MAX_PREMIUM_PER_TRADE", 500)
MAX_CONTRACTS = _int("MAX_CONTRACTS", 2)
MAX_DAILY_LOSS = _float("MAX_DAILY_LOSS", 1000)
MAX_OPEN_POSITIONS = _int("MAX_OPEN_POSITIONS", 3)
MIN_DTE = _int("MIN_DTE", 7)
MAX_DTE = _int("MAX_DTE", 45)
MIN_ABS_DELTA = _float("MIN_ABS_DELTA", 0.35)
MAX_ABS_DELTA = _float("MAX_ABS_DELTA", 0.65)
MAX_BID_ASK_SPREAD_PCT = _float("MAX_BID_ASK_SPREAD_PCT", 10)
MIN_OPEN_INTEREST = _int("MIN_OPEN_INTEREST", 100)
MIN_OPTION_VOLUME = _int("MIN_OPTION_VOLUME", 10)
HARD_PREMIUM_STOP_PCT = _float("HARD_PREMIUM_STOP_PCT", 50)
MAX_HOLD_DAYS = _float("MAX_HOLD_DAYS", 5)
