"""
Paper broker demo: live Binance.US BTC price -> mock Deribit-style chain
-> PaperBroker on a SCRATCH database (never the real one). No Telegram,
no agents, no execution service. Hits the network for the spot price only.

    TRADE_MONITOR_DB=/tmp/paper_demo.db python scripts/paper_broker_demo.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import database
from broker.factory import create_broker
from schemas import OrderRequest
from tools.market_tools import BinanceUSProvider
from tools.options_tools import MockDeribitChainProvider


class ShiftableSpot:
    """Real Binance.US spot, with an optional simulated % shift for the demo."""

    def __init__(self):
        self.real = BinanceUSProvider()
        self.shift_pct = 0.0

    def get_current_price(self, symbol):
        return self.real.get_current_price(symbol) * (1 + self.shift_pct / 100)


def show(label, result):
    o = result.order
    print(f"\n### {label}")
    print(f"ok={result.ok} replay={result.idempotent_replay} message={result.message!r}")
    if o:
        print(o.model_dump_json(indent=2, include={
            "order_id", "client_order_id", "signal_id", "proposal_key", "contract_symbol", "side",
            "quantity", "limit_price", "status", "fill_price", "rejection_reason",
            "last_quote_bid", "last_quote_ask", "submitted_at", "filled_at", "position_id"}))
    if result.position:
        print("position:", result.position.model_dump_json(indent=2, include={
            "position_id", "contract_symbol", "quantity", "entry_price", "current_mark",
            "unrealized_pnl", "realized_pnl", "status", "opened_at", "closed_at"}))


def main():
    db = config.db_path()
    if db == str(config.PROJECT_DIR / "trade_monitor.db"):
        sys.exit("refusing to run against the real database; set TRADE_MONITOR_DB to a scratch path")
    if os.path.exists(db):
        os.remove(db)
    database.initialize_database()

    spot = ShiftableSpot()
    chain_provider = MockDeribitChainProvider(spot)
    broker = create_broker(chain_provider)

    chain = broker.get_option_chain("BTCUSDT", option_type="call")
    calls = [c for c in chain.contracts if 20 < c.dte < 30]
    atm = min(calls, key=lambda c: abs(c.strike - chain.underlying_price))
    otm = min((c for c in calls if c.strike > atm.strike), key=lambda c: c.strike)
    print(f"BTC spot (Binance.US): {chain.underlying_price}")
    print(f"ATM quote: {atm.contract_symbol} bid {atm.bid} / ask {atm.ask} mark {atm.mark:.2f} x{atm.contract_multiplier}")

    entry = OrderRequest(client_order_id="signal:900:entry:open", signal_id=900, proposal_key="signal:900:entry",
                         contract_symbol=atm.contract_symbol, side="BUY_TO_OPEN", quantity=2, limit_price=atm.ask)
    show("1. BUY_TO_OPEN at the ask -> FILLED", broker.submit_order(entry))

    resting = OrderRequest(client_order_id="signal:901:entry:open", signal_id=901, proposal_key="signal:901:entry",
                           contract_symbol=otm.contract_symbol, side="BUY_TO_OPEN", quantity=1,
                           limit_price=otm.bid)
    show("2. BUY_TO_OPEN limit at the bid (below ask) -> PENDING", broker.submit_order(resting))

    show("3. Same client_order_id resubmitted -> idempotent replay", broker.submit_order(entry))
    counts = {t: database.get_connection().execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("paper_orders", "paper_fills", "paper_positions")}
    print("row counts after replay:", counts)

    print("\n### 4. Restart: new broker instance reads SQLite state")
    del broker
    broker = create_broker(chain_provider)
    print("PENDING orders:", [(o.order_id, o.contract_symbol, o.limit_price) for o in broker.list_orders("PENDING")])
    print("OPEN positions:", [(p.position_id, p.contract_symbol, p.quantity, p.entry_price) for p in broker.list_positions("OPEN")])

    print("\n### 5. Mark-to-market after a SIMULATED +1.5% spot move")
    spot.shift_pct = 1.5
    [position] = broker.mark_to_market()
    print(position.model_dump_json(indent=2, include={"position_id", "entry_price", "current_mark", "unrealized_pnl"}))

    quote = broker.get_option_contract(position.contract_symbol)
    show("6. SELL_TO_CLOSE at the bid -> position CLOSED",
         broker.close_position(position.position_id, limit_price=quote.bid,
                               client_order_id="signal:900:entry:close"))

    show("7. Cancel the resting order", broker.cancel_order(broker.list_orders("PENDING")[0].order_id))


if __name__ == "__main__":
    main()
