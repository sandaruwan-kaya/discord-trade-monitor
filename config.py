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
# Paper-trading risk defaults (demo values only, all overridable in .env).
# Read at call time so .env / test overrides always apply.
#

def trading_mode() -> str:
    """Only 'paper' exists in this codebase. Read at call time."""
    return (os.getenv("TRADING_MODE") or "paper").strip().lower()

RISK_DEFAULTS = {
    "MAX_PREMIUM_PER_TRADE": 500.0,
    "MAX_CONTRACTS": 2,
    "MAX_DAILY_LOSS": 1000.0,
    "MAX_OPEN_POSITIONS": 3,
    "MIN_DTE": 7,
    "MAX_DTE": 45,
    "MIN_ABS_DELTA": 0.35,
    "MAX_ABS_DELTA": 0.65,
    "MAX_BID_ASK_SPREAD_PCT": 10.0,
    "MIN_OPEN_INTEREST": 100,
    "MIN_OPTION_VOLUME": 10,
    "HARD_PREMIUM_STOP_PCT": 50.0,
    "MAX_HOLD_DAYS": 5.0,
    "MAX_QUOTE_AGE_SECONDS": 60,
    "OPTION_FEE_PER_CONTRACT": 0.0,
    "MAX_POSITION_LOSS": 0.0,          # absolute $ loss per position; 0 = disabled
}


def risk(name):
    default = RISK_DEFAULTS[name]
    if isinstance(default, int):
        return _int(name, default)
    return _float(name, default)


def kill_switch_env() -> bool:
    return _bool("TRADING_KILL_SWITCH", False)


def kill_switch_file() -> Path:
    """Presence of this file blocks new entries (touch it to halt, rm to resume)."""
    override = os.getenv("KILL_SWITCH_FILE")
    return Path(override).resolve() if override else PROJECT_DIR / "KILL_SWITCH"


def allowed_underlyings() -> list[str]:
    raw = os.getenv("ALLOWED_UNDERLYINGS", "BTCUSDT,ETHUSDT")
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


#
# Agent 2
#

def agent2_enabled() -> bool:
    return _bool("AGENT2_ENABLED", False)


def agent2_max_confirmation_age_minutes() -> float:
    # A WORKING transition detected later than this after the
    # confirming candle never produces a proposal.
    return _float("AGENT2_MAX_CONFIRMATION_AGE_MINUTES", 15)


def agent2_max_underlying_move_pct() -> float:
    # Chase protection: if the underlying already moved more than this
    # beyond the entry level in the trade direction, no proposal.
    return _float("AGENT2_MAX_UNDERLYING_MOVE_PCT", 0.5)


def agent2_max_llm_candidates() -> int:
    return _int("AGENT2_MAX_LLM_CANDIDATES", 8)


def agent2_pending_timeout_minutes() -> float:
    # PENDING claims older than this are interrupted -> FAILED, never retried.
    return _float("AGENT2_PENDING_TIMEOUT_MINUTES", 10)


#
# Agent 3 (risk analysis)
#

def agent3_llm_interval_minutes() -> float:
    return _float("AGENT3_LLM_INTERVAL_MINUTES", 15)


def agent3_material_move_pct() -> float:
    # Underlying move since the last LLM assessment that bypasses the throttle.
    return _float("AGENT3_MATERIAL_MOVE_PCT", 0.5)


def agent3_material_pnl_change_pct() -> float:
    # Change in bid-based PnL % (points) since the last LLM assessment that bypasses the throttle.
    return _float("AGENT3_MATERIAL_PNL_CHANGE_PCT", 10)


#
# Agent 4 (trade lifecycle)
#

def agent4_at_risk_invalidation_distance_pct() -> float:
    # HOLD but the underlying is within this % of invalidation -> AT_RISK.
    return _float("AGENT4_AT_RISK_INVALIDATION_DISTANCE_PCT", 0.5)


def agent4_at_risk_pnl_pct() -> float:
    # HOLD but bid-based PnL is at or below -this % -> AT_RISK.
    return _float("AGENT4_AT_RISK_PNL_PCT", 35)


#
# Execution service
#

def execution_claim_timeout_seconds() -> float:
    # A CLAIMED/SUBMITTING execution older than this is considered
    # interrupted and is reconciled (never re-submitted).
    return _float("EXECUTION_CLAIM_TIMEOUT_SECONDS", 120)


#
# Mock Deribit-style option chain (V1 paper only).
# Per-index settings, e.g. MOCK_OPTION_MULTIPLIER_BTC=0.01
#

MOCK_OPTION_DEFAULTS = {
    "BTC": {"MULTIPLIER": 0.01, "STRIKE_STEP": 1000.0, "TICK": 5.0, "BASE_IV": 0.50, "PEAK_OI": 3000.0},
    "ETH": {"MULTIPLIER": 0.1, "STRIKE_STEP": 25.0, "TICK": 0.5, "BASE_IV": 0.65, "PEAK_OI": 5000.0},
}


def mock_option_setting(index: str, key: str) -> float:
    return _float(f"MOCK_OPTION_{key}_{index}", MOCK_OPTION_DEFAULTS[index][key])
