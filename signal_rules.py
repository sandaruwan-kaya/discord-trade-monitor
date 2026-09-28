"""
Deterministic signal rules shared by the monitor and the agents:

- direction from the parsed bias
- timeframe bucket + expiry
- target sanity validation
"""
import re

import config


TERMINAL_STATUSES = {
    "TARGET_HIT",
    "INVALIDATED",
    "EXPIRED"
}


def direction_from_bias(bias) -> str:
    """bullish / bearish / neutral. Never guesses when bias is missing."""
    text = (bias or "").lower()

    if "bull" in text:
        return "bullish"

    if "bear" in text:
        return "bearish"

    return "neutral"


#
# Timeframe buckets
#

_MINUTES = re.compile(r"\b(\d+)\s*(m|min|mins|minute|minutes)\b")
_HOURS = re.compile(r"\b(\d+)\s*(h|hr|hrs|hour|hours)\b")


def classify_timeframe(timeframe) -> str:
    """Map free-text timeframe from the LLM into a bucket."""
    text = (timeframe or "").strip().lower()

    if not text:
        return "unknown"

    if any(w in text for w in ("week", "month", "1w", "1mo", "position", "long term", "long-term")):
        return "position"

    if any(w in text for w in ("swing", "daily", "1d", "multi-day", "days")):
        return "swing"

    if "scalp" in text:
        return "scalp"

    minutes = _MINUTES.search(text)
    if minutes:
        return "scalp" if int(minutes.group(1)) <= 15 else "intraday"

    hours = _HOURS.search(text)
    if hours:
        return "intraday" if int(hours.group(1)) < 4 else "swing"

    if any(w in text for w in ("intraday", "today", "session", "short")):
        return "intraday"

    return "unknown"


def signal_expiry_ms(timeframe) -> int:
    hours = config.signal_expiry_hours(classify_timeframe(timeframe))
    return int(hours * 3600 * 1000)


def is_expired(signal_time_ms, timeframe, now_ms) -> bool:
    return now_ms - signal_time_ms > signal_expiry_ms(timeframe)


#
# Target sanity
#

def check_targets(
    bias,
    targets,
    price_at_signal=None,
    confirmation_above=None,
    confirmation_below=None,
    invalidation_above=None,
    invalidation_below=None
):
    """
    Split targets into (valid, rejected).

    rejected is a list of {"target": x, "reason": "..."}.

    Bullish targets must sit above the price at signal, the
    confirmation-above level and the invalidation-below level.
    Bearish targets mirror that. Neutral bias: nothing rejected
    (the evaluator does not check targets for neutral signals).
    """
    direction = direction_from_bias(bias)
    valid = []
    rejected = []

    for target in targets or []:
        reason = None

        if direction == "bullish":
            if invalidation_below is not None and target <= invalidation_below:
                reason = f"bullish target {target} is at/below invalidation {invalidation_below}"
            elif confirmation_above is not None and target <= confirmation_above:
                reason = f"bullish target {target} is at/below confirmation {confirmation_above}"
            elif price_at_signal is not None and target <= price_at_signal:
                reason = f"bullish target {target} is at/below price at signal {price_at_signal}"

        elif direction == "bearish":
            if invalidation_above is not None and target >= invalidation_above:
                reason = f"bearish target {target} is at/above invalidation {invalidation_above}"
            elif confirmation_below is not None and target >= confirmation_below:
                reason = f"bearish target {target} is at/above confirmation {confirmation_below}"
            elif price_at_signal is not None and target >= price_at_signal:
                reason = f"bearish target {target} is at/above price at signal {price_at_signal}"

        if reason:
            rejected.append({"target": target, "reason": reason})
        else:
            valid.append(target)

    return valid, rejected


def check_signal_targets(signal, price_at_signal):
    return check_targets(
        signal.bias,
        signal.targets,
        price_at_signal=price_at_signal,
        confirmation_above=signal.confirmation_above,
        confirmation_below=signal.confirmation_below,
        invalidation_above=signal.invalidation_above,
        invalidation_below=signal.invalidation_below
    )
