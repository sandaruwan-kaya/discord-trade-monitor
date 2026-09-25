from models import TradeSignal


def evaluate_signal(
    signal: TradeSignal,
    candles: list
):

    if not candles:
        raise ValueError(
            "No market candles supplied"
        )

    current_price = candles[-1]["close"]

    highest_price = max(
        candle["high"]
        for candle in candles
    )

    lowest_price = min(
        candle["low"]
        for candle in candles
    )

    result = {
        "current_price": current_price,
        "highest_price": highest_price,
        "lowest_price": lowest_price,

        "confirmation_hit": False,
        "target_hit": False,
        "invalidated": False,

        "status": "DEVELOPING"
    }

    #
    # Confirmation
    #

    if signal.confirmation_above is not None:

        result["confirmation_hit"] = (
            highest_price
            >= signal.confirmation_above
        )

    if signal.confirmation_below is not None:

        result["confirmation_hit"] = (
            lowest_price
            <= signal.confirmation_below
        )

    #
    # Target
    #

    bias = (
        signal.bias or ""
    ).lower()

    if signal.targets:

        if "bull" in bias:

            first_target = min(
                signal.targets
            )

            result["target_hit"] = (
                highest_price
                >= first_target
            )

        elif "bear" in bias:

            first_target = max(
                signal.targets
            )

            result["target_hit"] = (
                lowest_price
                <= first_target
            )

    #
    # Invalidation
    #

    if signal.invalidation_below is not None:

        if lowest_price <= signal.invalidation_below:
            result["invalidated"] = True

    if signal.invalidation_above is not None:

        if highest_price >= signal.invalidation_above:
            result["invalidated"] = True

    #
    # Final status
    #

    if result["invalidated"]:

        result["status"] = "INVALIDATED"

    elif result["target_hit"]:

        result["status"] = "TARGET_HIT"

    elif result["confirmation_hit"]:

        result["status"] = "WORKING"

    else:

        result["status"] = "DEVELOPING"

    return result
