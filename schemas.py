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


#
# Broker (venue-neutral names; V1 implementation is paper only)
#

OrderSide = Literal["BUY_TO_OPEN", "SELL_TO_CLOSE"]   # no SELL_TO_OPEN: naked selling is unrepresentable

OrderStatus = Literal["PENDING", "FILLED", "CANCELLED", "REJECTED"]

PositionStatus = Literal["OPEN", "CLOSED"]


class OrderRequest(BaseModel):
    """
    What a caller asks the broker to do. client_order_id is the
    idempotency key: resubmitting the same key never creates a second order.
    SELL_TO_CLOSE must reference the open position it closes.
    """
    client_order_id: str
    signal_id: Optional[int] = None
    proposal_key: Optional[str] = None
    position_id: Optional[str] = None
    contract_symbol: str
    side: OrderSide
    quantity: int
    limit_price: float

    @model_validator(mode="after")
    def _valid(self):
        if not self.client_order_id.strip():
            raise ValueError("client_order_id is required")
        if self.quantity < 1:
            raise ValueError("quantity must be >= 1")
        if self.limit_price <= 0:
            raise ValueError("limit_price must be > 0")
        if self.side == "SELL_TO_CLOSE" and not self.position_id:
            raise ValueError("SELL_TO_CLOSE requires position_id")
        if self.side == "BUY_TO_OPEN" and self.position_id:
            raise ValueError("BUY_TO_OPEN cannot reference a position")
        return self


class PaperOrder(BaseModel):
    order_id: str
    client_order_id: str
    signal_id: Optional[int] = None
    proposal_key: Optional[str] = None
    position_id: Optional[str] = None
    contract_symbol: str
    side: OrderSide
    quantity: int
    limit_price: float
    status: OrderStatus
    fill_price: Optional[float] = None
    rejection_reason: Optional[str] = None
    last_quote_bid: Optional[float] = None
    last_quote_ask: Optional[float] = None
    last_checked_at: Optional[str] = None
    submitted_at: str
    updated_at: str
    filled_at: Optional[str] = None
    cancelled_at: Optional[str] = None


class PaperFill(BaseModel):
    fill_id: str
    order_id: str
    contract_symbol: str
    side: OrderSide
    quantity: int
    price: float
    quote_bid: Optional[float] = None
    quote_ask: Optional[float] = None
    filled_at: str


class PaperPosition(BaseModel):
    position_id: str
    entry_order_id: str
    signal_id: Optional[int] = None
    proposal_key: Optional[str] = None
    contract_symbol: str
    underlying: Optional[str] = None
    option_type: Optional[OptionType] = None
    contract_multiplier: float
    quantity: int                      # open quantity; never negative
    original_quantity: int
    entry_price: float
    current_mark: Optional[float] = None
    mark_as_of: Optional[str] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    status: PositionStatus
    opened_at: str
    closed_at: Optional[str] = None
    updated_at: str


class BrokerOrderResult(BaseModel):
    ok: bool
    order: Optional[PaperOrder] = None
    fill: Optional[PaperFill] = None
    position: Optional[PaperPosition] = None
    idempotent_replay: bool = False
    message: str = ""


#
# Policy engine (deterministic; no LLM)
#

PolicyPurpose = Literal["ENTRY", "EXIT"]


class RiskLimits(BaseModel):
    """Exact limits used for one decision (persisted as the policy snapshot)."""
    required_trading_mode: Literal["paper"] = "paper"
    allowed_underlyings: list[str]
    allowed_strategies: list[Strategy]
    max_contracts: int
    max_premium_per_trade: float
    max_daily_loss: float
    max_open_positions: int
    min_dte: float
    max_dte: float
    min_abs_delta: float
    max_abs_delta: float
    max_spread_pct: float
    min_open_interest: float
    min_volume: float
    max_quote_age_seconds: float
    fee_per_contract: float
    max_underlying_move_pct: float
    max_target_progress_pct: float
    signal_expiry_hours: dict[str, float]     # by timeframe bucket


class KillSwitchState(BaseModel):
    env_enabled: bool
    file_enabled: bool
    file_path: str

    @property
    def active(self) -> bool:
        return self.env_enabled or self.file_enabled


class OpenPositionSummary(BaseModel):
    position_id: str
    signal_id: Optional[int] = None
    proposal_key: Optional[str] = None
    contract_symbol: str
    quantity: int
    entry_price: float
    contract_multiplier: float
    premium_at_risk: float


class PendingEntrySummary(BaseModel):
    order_id: str
    signal_id: Optional[int] = None
    proposal_key: Optional[str] = None
    contract_symbol: str
    quantity: int
    limit_price: float
    premium_at_risk: float


class AccountSnapshot(BaseModel):
    open_position_count: int
    pending_entry_count: int
    open_premium_at_risk: float        # OPEN positions + PENDING entry orders
    realized_pnl_today: float          # UTC day, from SELL_TO_CLOSE fills
    unrealized_pnl: float
    available_paper_buying_power: Optional[float] = None


class ThesisContext(BaseModel):
    """Current state of the originating signal, from the DB + latest Agent 1 run."""
    signal: Optional[SignalAnalysisInput]   # levels, timing, price at signal; None if missing
    signal_status: str                 # evaluator status: DEVELOPING/WORKING/TARGET_HIT/...
    signal_active: bool
    agent1_state: Optional[SignalState] = None


class PolicyEvaluationInput(BaseModel):
    purpose: PolicyPurpose = "ENTRY"
    proposal: OptionProposal
    thesis: ThesisContext
    underlying_price: Optional[float]
    quote: Optional[OptionContract]    # FRESH quote; None if the fetch failed
    account: AccountSnapshot
    open_positions: list[OpenPositionSummary]
    pending_entries: list[PendingEntrySummary]
    filled_entry_proposal_keys: list[str]
    now_ms: int
    limits: RiskLimits
    trading_mode: str
    kill_switch: KillSwitchState


class PolicySnapshot(BaseModel):
    limits: RiskLimits
    trading_mode: str
    kill_switch: KillSwitchState


class PolicyDecision(BaseModel):
    decision_id: str
    evaluation_key: str
    purpose: PolicyPurpose
    approved: bool
    decision: Literal["APPROVED", "REJECTED"]
    reasons: list[str]
    checked_at: str
    proposal_key: str
    signal_id: Optional[int] = None
    policy_snapshot: PolicySnapshot
    calculated_max_loss: Optional[float] = None
    calculated_total_open_risk: Optional[float] = None
    quote_age_seconds: Optional[float] = None


#
# Execution service
#

ExecutionStatus = Literal[
    "CLAIMED", "POLICY_REJECTED", "SUBMITTING", "ORDER_PENDING", "ORDER_FILLED",
    "ORDER_REJECTED", "ORDER_CANCELLED", "FAILED", "INTERRUPTED",
    "REFUSED",          # not persisted: proposal missing / not ready (retryable later)
]


class ExecutionResult(BaseModel):
    ok: bool
    status: ExecutionStatus
    proposal_key: str
    execution_id: Optional[str] = None
    execution_key: Optional[str] = None
    replay: bool = False
    persisted: bool = True
    policy_decision_id: Optional[str] = None
    policy_reasons: list[str] = Field(default_factory=list)
    client_order_id: Optional[str] = None
    broker_order_id: Optional[str] = None
    broker_order_status: Optional[OrderStatus] = None
    error_code: Optional[str] = None
    message: str = ""


#
# Agent 3 — Risk Analysis
#

RiskRecommendation = Literal["HOLD", "REDUCE", "EXIT", "DATA_UNAVAILABLE"]

RiskState = Literal["HARD_EXIT", "DATA_UNAVAILABLE", "CONTEXTUAL", "THROTTLED", "FALLBACK"]

LLMStatus = Literal["CALLED", "THROTTLED", "SKIPPED_HARD_EXIT", "SKIPPED_DATA", "FAILED", "NOT_CONFIGURED"]

RISK_REASON_CODES = (
    "THESIS_INTACT", "THESIS_DETERIORATING", "DELTA_DECAY", "IV_CRUSH", "IV_EXPANSION",
    "THETA_DECAY", "NEAR_INVALIDATION", "NEAR_TARGET", "ADVERSE_MOMENTUM",
    "VOLATILITY_EXPANSION", "VOLATILITY_CONTRACTION", "RISK_REWARD_DETERIORATED",
    "HORIZON_ELAPSING", "PROFIT_PROTECTION", "OTHER",
)


class RiskRuleLimits(BaseModel):
    hard_premium_stop_pct: float
    max_position_loss: float          # 0 = disabled
    max_hold_days: float
    max_quote_age_seconds: float
    llm_interval_minutes: float
    material_move_pct: float
    material_pnl_change_pct: float


class RiskThesis(BaseModel):
    direction: Direction
    entry_level: Optional[float]
    target: Optional[float]
    invalidation: Optional[float]
    timeframe: Optional[str]
    signal_time_ms: int
    signal_status: str                         # evaluator status
    evaluator_target_time_ms: Optional[int] = None
    evaluator_invalidation_time_ms: Optional[int] = None
    recent_high: Optional[float] = None        # since the signal (evaluator state)
    recent_low: Optional[float] = None


class RiskPosition(BaseModel):
    status: str
    quantity: int
    contract_multiplier: float
    entry_option_price: float
    opened_at_ms: int


class Candle(BaseModel):
    timestamp: int
    high: float
    low: float
    close: float


class LastLLMAssessment(BaseModel):
    assessed_at_ms: int
    underlying_price: Optional[float]
    unrealized_pnl_pct: Optional[float]
    recommendation: RiskRecommendation
    risk_state: RiskState
    confidence: float


class RiskAnalysisInput(BaseModel):
    position_id: str
    signal_id: Optional[int]
    proposal_key: Optional[str]
    contract_symbol: str
    thesis: RiskThesis
    position: RiskPosition
    quote: Optional[OptionContract]            # FRESH; None if unavailable
    quote_error: Optional[str] = None
    option_expiry_ts_ms: Optional[int] = None
    underlying_price: Optional[float]
    underlying_price_at_entry: Optional[float] = None
    last_candle: Optional[Candle] = None       # latest coarse candle (intrabar ordering)
    market_context: Optional[MarketContext] = None
    now_ms: int
    limits: RiskRuleLimits


class RiskMetrics(BaseModel):
    """Deterministic numbers. PnL uses the BID (realistic exit), mark only for display."""
    entry_cost: float
    liquidation_value: Optional[float]
    unrealized_pnl: Optional[float]
    unrealized_pnl_pct: Optional[float]
    mark_value: Optional[float]
    mark_pnl: Optional[float]
    option_bid: Optional[float]
    option_mark: Optional[float]
    quote_age_seconds: Optional[float]
    target_progress_pct: Optional[float]
    distance_to_target_pct: Optional[float]
    distance_to_invalidation_pct: Optional[float]
    underlying_move_since_entry_pct: Optional[float]
    time_in_trade_hours: float
    hold_remaining_hours: float
    dte: Optional[float]


class RiskLLMOutput(BaseModel):
    recommendation: Literal["HOLD", "REDUCE", "EXIT"]
    confidence: float
    reason_codes: list[str]
    rationale: str


class RiskAssessment(BaseModel):
    assessment_id: str
    assessment_key: str
    position_id: str
    signal_id: Optional[int]
    proposal_key: Optional[str]
    recommendation: RiskRecommendation
    risk_state: RiskState
    hard_rule_triggered: bool
    hard_rule_codes: list[str]
    data_issue_codes: list[str]
    llm_used: bool
    llm_status: LLMStatus
    llm_trigger: Optional[str] = None
    last_contextual_recommendation: Optional[RiskRecommendation] = None
    last_contextual_at_ms: Optional[int] = None
    confidence: float
    reasons: list[str]
    metrics: RiskMetrics
    underlying_price: Optional[float]
    option_bid: Optional[float]
    option_mark: Optional[float]
    unrealized_pnl: Optional[float]
    unrealized_pnl_pct: Optional[float]
    target_progress_pct: Optional[float]
    distance_to_invalidation_pct: Optional[float]
    time_in_trade_hours: float
    assessed_at: str
    input_snapshot: RiskAnalysisInput
    validation_notes: list[str] = Field(default_factory=list)


#
# Agent 4 — Trade lifecycle (deterministic)
#

TradeState = Literal[
    "PROPOSED", "POLICY_APPROVED", "ORDER_SUBMITTED", "ORDER_PENDING",
    "OPEN", "WORKING", "AT_RISK", "REDUCE_REQUESTED", "EXIT_REQUESTED",
    "CLOSE_ORDER_PENDING", "CLOSED", "REJECTED", "CANCELLED", "FAILED",
]

TradeAction = Literal["NONE", "REQUEST_REDUCE", "REQUEST_EXIT", "WAIT_FOR_DATA", "MARK_CLOSED"]

ExitRequestStatus = Literal["ACTIVE", "COMPLETED", "SUPERSEDED"]


class LifecycleLimits(BaseModel):
    at_risk_invalidation_distance_pct: float
    at_risk_pnl_pct: float


class BrokerPositionState(BaseModel):
    position_id: str
    status: str                        # OPEN / CLOSED
    quantity: int
    contract_symbol: str


class BrokerOrderState(BaseModel):
    order_id: str
    side: str
    status: str
    quantity: int


class RiskAssessmentSummary(BaseModel):
    assessment_id: str
    position_id: str
    recommendation: RiskRecommendation
    risk_state: RiskState
    hard_rule_triggered: bool
    hard_rule_codes: list[str]
    data_issue_codes: list[str]
    reasons: list[str]
    unrealized_pnl_pct: Optional[float] = None
    distance_to_invalidation_pct: Optional[float] = None
    assessed_at_ms: int


class ExitRequestSummary(BaseModel):
    exit_request_id: str
    request_key: str
    requested_action: Literal["REDUCE", "EXIT"]
    status: ExitRequestStatus
    hard_exit: bool


class TradeManagementInput(BaseModel):
    trade_id: str
    signal_id: Optional[int]
    proposal_key: Optional[str]
    execution_id: Optional[str]
    position_id: str
    current_state: str                 # validated against TradeState (fails closed)
    position: Optional[BrokerPositionState]
    close_order: Optional[BrokerOrderState] = None     # latest SELL_TO_CLOSE for the position
    latest_risk_assessment: Optional[RiskAssessmentSummary] = None
    assessment_already_processed: bool = False
    active_exit_request: Optional[ExitRequestSummary] = None
    active_reduce_request: Optional[ExitRequestSummary] = None
    opened_at: Optional[str] = None
    now_ms: int
    limits: LifecycleLimits


class ExitRequestIntent(BaseModel):
    """What Agent 4 wants persisted. Quantity is deterministic, never from an LLM."""
    request_key: str
    requested_action: Literal["REDUCE", "EXIT"]
    requested_quantity: Optional[int]  # EXIT: full open quantity; REDUCE: None (advisory, V1)
    hard_exit: bool
    reason_codes: list[str]
    data_issue_codes: list[str]


class TradeManagementDecision(BaseModel):
    trade_id: str
    previous_state: str
    next_state: str
    action: TradeAction
    reason_codes: list[str]
    reasons: list[str]
    risk_assessment_id: Optional[str]
    transition_allowed: bool
    decided_at: str
    exit_request: Optional[ExitRequestIntent] = None
    evidence_for_exit_request_id: Optional[str] = None
    resolve_requests: dict[str, ExitRequestStatus] = Field(default_factory=dict)


class ExitRequest(BaseModel):
    exit_request_id: str
    request_key: str
    trade_id: str
    position_id: str
    signal_id: Optional[int]
    proposal_key: Optional[str]
    risk_assessment_id: Optional[str]
    requested_action: Literal["REDUCE", "EXIT"]
    requested_quantity: Optional[int]
    hard_exit: bool
    reason_codes: list[str]
    data_issue_codes: list[str]
    status: ExitRequestStatus
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None
