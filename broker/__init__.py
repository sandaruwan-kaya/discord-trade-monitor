"""
Broker layer. V1 contains ONLY the paper broker.

There is no live broker implementation in this repository, and
agents never receive a broker object. From Step 7 on, only the
execution service will call submit_order / close_position / cancel_order.
"""
