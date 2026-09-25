from models import TradeSignal


TERMINAL_STATUSES = {
    "TARGET_HIT",
    "INVALIDATED"
}


def evaluate_signal(
    signal: TradeSignal,
    candles: list
):
    if not candles:
        raise ValueError("No market candles supplied")

    bias = (signal.bias or "").lower()

    result = {
        "current_price": candles[-1]["close"],
        "highest_price": max(c["high"] for c in candles),
        "lowest_price": min(c["low"] for c in candles),

        "confirmation_above_hit": False,
        "confirmation_below_hit": False,

        "confirmation_hit": False,
        "target_hit": False,
        "invalidated": False,

        "confirmation_time": None,
        "target_time": None,
        "invalidation_time": None,

        "status": "DEVELOPING"
    }

    #
    # Walk through candles in chronological order.
    #
    # This is important because:
    #
    # target hit -> invalidation later
    #
    # should remain TARGET_HIT.
    #

    for candle in candles:

        high = candle["high"]
        low = candle["low"]
        timestamp = candle["timestamp"]

        #
        # 1. Check invalidation.
        #
        # If the setup is invalidated before target,
        # we close the signal immediately.
        #

        invalidated_now = False

        if signal.invalidation_below is not None:
            if low <= signal.invalidation_below:
                invalidated_now = True

        if signal.invalidation_above is not None:
            if high >= signal.invalidation_above:
                invalidated_now = True

        if invalidated_now:
            result["invalidated"] = True
            result["invalidation_time"] = timestamp
            result["status"] = "INVALIDATED"

            return result

        #
        # 2. Check confirmations independently.
        #

        if (
            signal.confirmation_above is not None
            and not result["confirmation_above_hit"]
            and high >= signal.confirmation_above
        ):
            result["confirmation_above_hit"] = True

            if result["confirmation_time"] is None:
                result["confirmation_time"] = timestamp

        if (
            signal.confirmation_below is not None
            and not result["confirmation_below_hit"]
            and low <= signal.confirmation_below
        ):
            result["confirmation_below_hit"] = True

            if result["confirmation_time"] is None:
                result["confirmation_time"] = timestamp

        #
        # For most signals we treat either configured
        # confirmation as sufficient.
        #

        result["confirmation_hit"] = (
            result["confirmation_above_hit"]
            or result["confirmation_below_hit"]
        )

        #
        # 3. Check targets.
        #
        # We only evaluate targets according to
        # the inferred directional bias.
        #

        if signal.targets:

            if "bull" in bias:
                first_target = min(signal.targets)

                if high >= first_target:
                    result["target_hit"] = True
                    result["target_time"] = timestamp
                    result["status"] = "TARGET_HIT"

                    return result

            elif "bear" in bias:
                first_target = max(signal.targets)

                if low <= first_target:
                    result["target_hit"] = True
                    result["target_time"] = timestamp
                    result["status"] = "TARGET_HIT"

                    return result

        #
        # 4. If confirmed but not closed,
        # signal is working.
        #

        if result["confirmation_hit"]:
            result["status"] = "WORKING"

    return result