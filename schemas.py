"""
Pydantic schemas shared by agents and tools.

LLM-facing models avoid numeric Field constraints (not supported by
strict structured outputs); bounds are enforced in post-validation.
"""
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


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


#
# Options (venue-neutral)
#

OptionType = Literal["call", "put"]

Strategy = Literal["LONG_CALL", "LONG_PUT"]

ProposalTrigger = Literal["FIRST_WORKING", "ALREADY_CONFIRMED_FRESH"]

ProposalStatus = Literal[
    "PENDING",
    "PROPOSED",
    "NO_CANDIDATES",
    "DECLINED",
    "SKIPPED",
    "FAILED",
]


class OptionContract(BaseModel):
    """
    One option quote, normalized across venues.

    Prices are in quote_currency per 1 unit of the underlying
    (Deribit convention; per share for equity options). Cash cost:
        price * contract_multiplier * quantity
    contract_multiplier is required: every provider supplies its own.
    """
    contract_symbol: str
    underlying: str
    underlying_index: str
    option_type: OptionType
    style: Literal["european", "american"]
    strike: float
    expiry: date
    expiry_ts_ms: int
    dte: float

    bid: Optional[float] = None
    ask: Optional[float] = None
    mark: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None

    iv: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

    open_interest: Optional[float] = None
    volume_24h: Optional[float] = None

    contract_multiplier: float
    min_quantity: int = 1
    quantity_step: int = 1
    tick_size: float
    quote_currency: str = "USD"

    underlying_price: float
    as_of_ms: int
    source: str


class OptionChain(BaseModel):
    underlying: str
    underlying_price: float
    as_of_ms: int
    source: str
    expiries: list[date]
    contracts: list[OptionContract]


class OptionsPlaybook(BaseModel):
    """Snapshot of the approved-strategy limits at decision time."""
    allowed_strategies: list[Strategy]
    allowed_underlyings: list[str]
    min_dte: int
    max_dte: int
    min_abs_delta: float
    max_abs_delta: float
    max_spread_pct: float
    min_open_interest: int
    min_volume: int
    max_premium_per_trade: float
    max_contracts: int
    max_quote_age_seconds: int
    fee_per_contract: float
    max_llm_candidates: int


class ThesisSnapshot(BaseModel):
    underlying: str
    direction: Direction
    entry_level: Optional[float]
    target: float
    invalidation: Optional[float]
    time_horizon: TimeHorizon
    agent1_state: SignalState


class OptionCandidate(BaseModel):
    """A contract that passed every deterministic filter. All numbers here are deterministic."""
    candidate_id: str
    strategy: Strategy
    contract: OptionContract
    mid: float
    spread_pct: float
    limit_price: float
    quantity: int
    estimated_max_loss: float
    breakeven: float
    intrinsic_at_target: float
    estimated_value_at_target: float
    estimated_value_at_invalidation: Optional[float]
    reward_to_risk_at_target: float
    deterministic_rank: int


class OptionsSelectionInput(BaseModel):
    signal_id: int
    analysis_id: Optional[int] = None
    trigger: ProposalTrigger
    thesis: ThesisSnapshot
    underlying_price: float
    as_of_ms: int
    playbook: OptionsPlaybook
    chain_source: str
    candidates: list[OptionCandidate]
    filter_summary: dict[str, int]


class OptionsSelectionLLMOutput(BaseModel):
    """What the LLM may say. It can only reference candidate ids."""
    decision: Literal["SELECT", "NO_TRADE"]
    selected_candidate_id: Optional[str]
    ranking: list[str]
    confidence: float
    reason: str


class OptionProposal(BaseModel):
    """Final Agent 2 output. A proposal only; nothing is submitted."""
    proposal_key: str
    signal_id: int
    analysis_id: Optional[int] = None
    trigger: Optional[ProposalTrigger] = None
    status: ProposalStatus

    strategy: Optional[Strategy] = None
    # No SELL value exists: naked selling cannot be expressed.
    side: Optional[Literal["BUY_TO_OPEN"]] = None

    contract_symbol: Optional[str] = None
    underlying: str
    option_type: Optional[OptionType] = None
    expiry: Optional[date] = None
    dte: Optional[float] = None
    strike: Optional[float] = None
    delta: Optional[float] = None
    iv: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    mid: Optional[float] = None
    limit_price: Optional[float] = None
    quantity: Optional[int] = None
    contract_multiplier: Optional[float] = None
    estimated_max_loss: Optional[float] = None
    breakeven: Optional[float] = None

    thesis_target: Optional[float] = None
    thesis_invalidation: Optional[float] = None

    selected_by: Optional[Literal["llm", "deterministic_fallback"]] = None
    llm_used: bool = False
    confidence: float = 0.0
    reason: str = ""
    validation_notes: list[str] = Field(default_factory=list)
    candidates_considered: int = 0
    filter_summary: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self):
        if self.status != "PROPOSED":
            return self

        required = [
            "trigger", "strategy", "side", "contract_symbol", "option_type",
            "expiry", "strike", "bid", "ask", "limit_price", "quantity",
            "contract_multiplier", "estimated_max_loss",
        ]
        missing = [name for name in required if getattr(self, name) is None]
        if missing:
            raise ValueError(f"PROPOSED proposal missing {missing}")

        expected_type = {"LONG_CALL": "call", "LONG_PUT": "put"}[self.strategy]
        if self.option_type != expected_type:
            raise ValueError(f"{self.strategy} requires a {expected_type}")

        if self.quantity < 1:
            raise ValueError("quantity must be >= 1")

        if self.limit_price <= 0 or self.limit_price > self.ask:
            raise ValueError("limit price must be > 0 and <= ask for a buy")

        return self
