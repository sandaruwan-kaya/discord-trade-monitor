from pydantic import BaseModel, Field
from typing import Optional, List


class TradeSignal(BaseModel):
    is_trade_signal: bool = False

    symbol: Optional[str] = None
    bias: Optional[str] = None
    timeframe: Optional[str] = None

    support_levels: List[float] = Field(default_factory=list)
    resistance_levels: List[float] = Field(default_factory=list)
    targets: List[float] = Field(default_factory=list)

    confirmation_above: Optional[float] = None
    confirmation_below: Optional[float] = None

    invalidation_above: Optional[float] = None
    invalidation_below: Optional[float] = None

    summary: Optional[str] = None
