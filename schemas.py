"""
Pydantic schemas shared by agents and tools.

LLM-facing models avoid numeric Field constraints (not supported by
strict structured outputs); bounds are enforced in post-validation.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


Direction = Literal["bullish", "bearish", "neutral"]

EntryConditionType = Literal[
    "break_above",
    "break_below",
    "hold_above",
    "hold_below",
    "immediate",
    "none",
]

TimeHorizon = Literal[
    "scalp",
    "intraday",
    "short_term",
    "swing",
    "position",
    "unknown",
]

SignalState = Literal[
    "DEVELOPING",
    "ALREADY_CONFIRMED",
    "STALE",
    "NOT_TRADEABLE",
]

NextStep = Literal[
    "WAIT_FOR_CONFIRMATION",
    "ELIGIBLE_FOR_OPTIONS_REVIEW",
    "NO_ACTION",
]


#
# Market data
#

class MarketContext(BaseModel):
    provider: str
    symbol: str
    as_of_ms: int
    current_price: float
    change_pct_24h: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    indicators: dict[str, Optional[float | str]] = Field(default_factory=dict)


#
# Agent 1 — Signal Analysis
#

class SignalAnalysisInput(BaseModel):
    signal_id: Optional[int] = None
    discord_message_id: Optional[str] = None
    raw_message: Optional[str] = None

    symbol: str
    bias: Optional[str] = None
    timeframe: Optional[str] = None
    summary: Optional[str] = None

    support_levels: list[float] = Field(default_factory=list)
    resistance_levels: list[float] = Field(default_factory=list)
    targets: list[float] = Field(default_factory=list)

    confirmation_above: Optional[float] = None
    confirmation_below: Optional[float] = None
    invalidation_above: Optional[float] = None
    invalidation_below: Optional[float] = None

    signal_time_ms: int
    price_at_signal: Optional[float] = None
    current_price: Optional[float] = None

    market_context: Optional[MarketContext] = None


class SignalFacts(BaseModel):
    """Deterministic facts computed before and independent of the LLM."""
    direction: Direction
    is_conditional: bool
    reference_price: Optional[float]
    valid_targets: list[float]
    rejected_targets: list[dict]
    default_entry_level: Optional[float]
    default_target: Optional[float]
    default_invalidation: Optional[float]
    entry_crossed: bool
    target_reached: bool
    invalidation_hit: bool
    target_progress_pct: Optional[float]
    age_minutes: float
    expired: bool
    timeframe_bucket: str


class SignalAnalysisLLMOutput(BaseModel):
    """What the LLM is asked to return. Never trusted as-is."""
    direction: Direction
    entry_condition_type: EntryConditionType
    entry_level: Optional[float]
    target: Optional[float]
    invalidation: Optional[float]
    time_horizon: TimeHorizon
    signal_state: SignalState
    confidence: float
    reasons: list[str]


class SignalAnalysis(BaseModel):
    """Final, post-validated Agent 1 output."""
    underlying: str
    direction: Direction
    is_conditional: bool
    entry_condition_type: EntryConditionType
    entry_level: Optional[float]
    target: Optional[float]
    invalidation: Optional[float]
    time_horizon: TimeHorizon
    signal_state: SignalState
    next_step: NextStep
    eligible_for_options: bool
    confidence: float
    reasons: list[str] = Field(default_factory=list)
    validation_notes: list[str] = Field(default_factory=list)
    llm_used: bool
    facts: SignalFacts
