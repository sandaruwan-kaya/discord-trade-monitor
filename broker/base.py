"""
Venue-neutral broker interface.

Implementations (V1): PaperBroker only. A future Deribit or
Robinhood adapter implements this same interface; callers never
depend on a vendor.

Order sides are limited to BUY_TO_OPEN and SELL_TO_CLOSE, so
naked/short option selling cannot be expressed through this API.
"""
from abc import ABC, abstractmethod

from schemas import (
    BrokerOrderResult,
    OptionChain,
    OptionContract,
    OrderRequest,
    PaperOrder,
    PaperPosition,
)


class BrokerError(RuntimeError):
    pass


class BrokerConfigError(BrokerError):
    """Broker refused to start (e.g. TRADING_MODE is not 'paper')."""


class BrokerBase(ABC):
    name: str

    #
    # Market data (delegated to an OptionChainProvider)
    #

    @abstractmethod
    def get_option_chain(
        self,
        underlying: str,
        *,
        option_type: str | None = None,
        min_dte: float | None = None,
        max_dte: float | None = None
    ) -> OptionChain: ...

    @abstractmethod
    def get_option_contract(self, contract_symbol: str) -> OptionContract: ...

    #
    # Orders
    #

    @abstractmethod
    def submit_order(self, request: OrderRequest) -> BrokerOrderResult:
        """Idempotent on request.client_order_id."""

    @abstractmethod
    def get_order(self, order_id: str) -> PaperOrder | None: ...

    @abstractmethod
    def get_order_by_client_id(self, client_order_id: str) -> PaperOrder | None:
        """Stable-identity lookup used for crash reconciliation (never submits)."""

    def process_pending_orders(self) -> list[BrokerOrderResult]:
        """Simulation hook: re-check resting orders. Real venues fill on their own."""
        return []

    @abstractmethod
    def cancel_order(self, order_id: str) -> BrokerOrderResult:
        """Only PENDING orders can be cancelled."""

    #
    # Positions
    #

    @abstractmethod
    def get_position(self, position_id: str, refresh_mark: bool = True) -> PaperPosition | None: ...

    @abstractmethod
    def close_position(
        self,
        position_id: str,
        limit_price: float,
        client_order_id: str,
        quantity: int | None = None
    ) -> BrokerOrderResult:
        """Submits a SELL_TO_CLOSE for (part of) an open position."""
