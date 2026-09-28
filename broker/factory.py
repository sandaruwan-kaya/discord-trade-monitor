"""
Broker factory. The ONLY broker it can build is PaperBroker.

There is intentionally no branch for any other mode: a non-paper
TRADING_MODE is refused, never routed somewhere else.
"""
import config
from broker.base import BrokerConfigError
from broker.paper_broker import PaperBroker


def create_broker(chain_provider, **kwargs) -> PaperBroker:
    mode = config.trading_mode()

    if mode != "paper":
        raise BrokerConfigError(
            f"TRADING_MODE={mode!r} refused: only paper trading exists in this codebase"
        )

    return PaperBroker(chain_provider, **kwargs)
