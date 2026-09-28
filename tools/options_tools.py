"""
Option-chain abstraction, mock Deribit-style chain, and the
deterministic pre-filter that runs BEFORE any LLM sees a contract.

    OptionChainProvider (protocol)
      └── MockDeribitChainProvider   V1 paper only, priced off the real underlying
      (later: DeribitOptionsProvider, RobinhoodOptionsProvider)

Agent 2 depends on OptionChainProvider + prefilter(), never on a venue.
"""
import hashlib
import math
import re
import time
from datetime import date, datetime, timedelta, timezone
from typing import Protocol

import config
from schemas import (
    OptionCandidate,
    OptionChain,
    OptionContract,
    OptionsPlaybook,
    ThesisSnapshot,
)


DAY_MS = 86_400_000


class OptionChainError(RuntimeError):
    pass


class OptionChainProvider(Protocol):
    name: str

    def supports(self, underlying: str) -> bool: ...

    def get_expiries(self, underlying: str) -> list[date]: ...

    def get_option_chain(
        self,
        underlying: str,
        *,
        option_type: str | None = None,
        min_dte: float | None = None,
        max_dte: float | None = None
    ) -> OptionChain: ...

    def get_option_contract(self, contract_symbol: str) -> OptionContract: ...


#
# Black-Scholes (European, r = 0), stdlib only
#

def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def black_scholes(option_type, spot, strike, years, iv):
    """Returns dict(price, delta, gamma, theta_per_day, vega_per_vol_pt)."""
    if years <= 0 or iv <= 0:
        intrinsic = max(spot - strike, 0) if option_type == "call" else max(strike - spot, 0)
        delta = (1.0 if spot > strike else 0.0) if option_type == "call" else (-1.0 if spot < strike else 0.0)
        return {"price": intrinsic, "delta": delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    sqrt_t = math.sqrt(years)
    d1 = (math.log(spot / strike) + 0.5 * iv * iv * years) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t

    if option_type == "call":
        price = spot * _norm_cdf(d1) - strike * _norm_cdf(d2)
        delta = _norm_cdf(d1)
    else:
        price = strike * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1

    return {
        "price": max(price, 0.0),
        "delta": delta,
        "gamma": _norm_pdf(d1) / (spot * iv * sqrt_t),
        "theta": -(spot * _norm_pdf(d1) * iv) / (2 * sqrt_t) / 365,
        "vega": spot * _norm_pdf(d1) * sqrt_t / 100,
    }


def round_up_to_tick(price, tick):
    return round(math.ceil(price / tick - 1e-9) * tick, 10)


def round_down_to_tick(price, tick):
    return round(math.floor(price / tick + 1e-9) * tick, 10)


#
# Mock Deribit-style chain
#

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

_SYMBOL = re.compile(r"^(?P<index>[A-Z]+)-(?P<day>\d{1,2})(?P<mon>[A-Z]{3})(?P<yy>\d{2})-(?P<strike>\d+(?:\.\d+)?)-(?P<cp>[CP])$")


def deribit_symbol(index, expiry: date, strike, option_type):
    strike_txt = f"{strike:g}" if strike != int(strike) else str(int(strike))
    return f"{index}-{expiry.day}{MONTHS[expiry.month - 1]}{expiry.year % 100:02d}-{strike_txt}-{'C' if option_type == 'call' else 'P'}"


def _jitter(key, lo=0.8, hi=1.2):
    digest = hashlib.sha256(key.encode()).digest()
    return lo + (hi - lo) * digest[0] / 255


class MockDeribitChainProvider:
    """
    Deterministic Deribit-style chain priced off the REAL underlying.

    - expiries 08:00 UTC: next 2 dailies, Fridays within 90 days,
      last Friday of the next 3 months
    - strikes: grid +-25% around spot (per-index step)
    - IV: base * smile * term structure (+ small put premium)
    - Black-Scholes marks + greeks, spread widening away from ATM
      and near expiry, liquidity decaying away from ATM
    - multiplier / tick / base IV / strike step configurable per index
    """
    name = "mock_deribit"

    INDEX_BY_SYMBOL = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}

    SMILE = 2.0
    STRIKE_RANGE = 0.25
    HORIZON_DAYS = 90

    def __init__(self, market, now_ms=None):
        self.market = market
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))

    #
    # Interface
    #

    def supports(self, underlying):
        return underlying in self.INDEX_BY_SYMBOL

    def get_expiries(self, underlying):
        self._index(underlying)
        return [e.date() for e in self._expiry_datetimes(self._now_ms())]

    def get_option_chain(self, underlying, *, option_type=None, min_dte=None, max_dte=None):
        index = self._index(underlying)
        now_ms = self._now_ms()
        spot = float(self.market.get_current_price(underlying))

        expiries = self._expiry_datetimes(now_ms)
        contracts = []

        for expiry in expiries:
            for strike in self._strikes(index, spot):
                for cp in ("call", "put"):
                    if option_type and cp != option_type:
                        continue
                    contract = self._price(underlying, index, spot, expiry, strike, cp, now_ms)
                    if min_dte is not None and contract.dte < min_dte:
                        continue
                    if max_dte is not None and contract.dte > max_dte:
                        continue
                    contracts.append(contract)

        return OptionChain(
            underlying=underlying,
            underlying_price=spot,
            as_of_ms=now_ms,
            source=self.name,
            expiries=[e.date() for e in expiries],
            contracts=contracts
        )

    def get_option_contract(self, contract_symbol):
        match = _SYMBOL.match(contract_symbol)
        if not match:
            raise OptionChainError(f"unrecognized contract symbol {contract_symbol!r}")

        index = match["index"]
        underlying = next((s for s, i in self.INDEX_BY_SYMBOL.items() if i == index), None)
        if underlying is None:
            raise OptionChainError(f"unsupported index {index}")

        expiry = datetime(
            2000 + int(match["yy"]), MONTHS.index(match["mon"]) + 1, int(match["day"]),
            8, tzinfo=timezone.utc
        )
        now_ms = self._now_ms()

        if expiry.timestamp() * 1000 <= now_ms:
            raise OptionChainError(f"{contract_symbol} has expired")

        spot = float(self.market.get_current_price(underlying))
        cp = "call" if match["cp"] == "C" else "put"

        return self._price(underlying, index, spot, expiry, float(match["strike"]), cp, now_ms)

    #
    # Generation
    #

    def _index(self, underlying):
        if underlying not in self.INDEX_BY_SYMBOL:
            raise OptionChainError(f"no mock option chain for {underlying}")
        return self.INDEX_BY_SYMBOL[underlying]

    def _expiry_datetimes(self, now_ms):
        now = datetime.fromtimestamp(now_ms / 1000, timezone.utc)
        first = now.replace(hour=8, minute=0, second=0, microsecond=0)
        if first <= now:
            first += timedelta(days=1)

        expiries = {first, first + timedelta(days=1)}
        horizon = now + timedelta(days=self.HORIZON_DAYS)

        day = first
        while day <= horizon:
            if day.weekday() == 4:
                expiries.add(day)
            day += timedelta(days=1)

        for months_ahead in range(1, 4):
            year = now.year + (now.month - 1 + months_ahead) // 12
            month = (now.month - 1 + months_ahead) % 12 + 1
            last = datetime(year + (month // 12), month % 12 + 1, 1, 8, tzinfo=timezone.utc) - timedelta(days=1)
            while last.weekday() != 4:
                last -= timedelta(days=1)
            if last > now:
                expiries.add(last)

        return sorted(expiries)

    def _strikes(self, index, spot):
        step = config.mock_option_setting(index, "STRIKE_STEP")
        low = math.ceil(spot * (1 - self.STRIKE_RANGE) / step) * step
        high = math.floor(spot * (1 + self.STRIKE_RANGE) / step) * step
        count = int(round((high - low) / step)) + 1
        return [round(low + i * step, 10) for i in range(count)]

    def _is_monthly(self, expiry):
        return expiry.weekday() == 4 and (expiry + timedelta(days=7)).month != expiry.month

    def _price(self, underlying, index, spot, expiry, strike, cp, now_ms):
        expiry_ms = int(expiry.timestamp() * 1000)
        dte = (expiry_ms - now_ms) / DAY_MS
        years = max(dte, 0) / 365

        moneyness = math.log(strike / spot)
        term = 1 + 0.10 * math.exp(-dte / 7)
        iv = config.mock_option_setting(index, "BASE_IV") * (1 + self.SMILE * moneyness ** 2) * term
        if cp == "put":
            iv *= 1.02

        bs = black_scholes(cp, spot, strike, years, iv)
        mark = bs["price"]
        tick = config.mock_option_setting(index, "TICK")

        spread_frac = 0.015 + 0.6 * abs(moneyness) + 0.03 * max(0.0, (7 - dte) / 7)
        half = max(tick / 2, mark * spread_frac / 2)

        bid = round_down_to_tick(mark - half, tick)
        ask = round_up_to_tick(mark + half, tick)
        if bid < tick:
            bid = 0.0

        symbol = deribit_symbol(index, expiry.date(), strike, cp)

        expiry_weight = 1.5 if self._is_monthly(expiry) else (1.0 if expiry.weekday() == 4 else 0.3)
        oi = (
            config.mock_option_setting(index, "PEAK_OI")
            * math.exp(-(moneyness / 0.08) ** 2)
            * expiry_weight
            * _jitter(symbol)
        )
        volume = oi * 0.1 * _jitter(symbol + ":vol", 0.5, 1.5)

        return OptionContract(
            contract_symbol=symbol,
            underlying=underlying,
            underlying_index=index,
            option_type=cp,
            style="european",
            strike=strike,
            expiry=expiry.date(),
            expiry_ts_ms=expiry_ms,
            dte=dte,
            bid=bid,
            ask=ask,
            mark=mark,
            iv=iv,
            delta=bs["delta"],
            gamma=bs["gamma"],
            theta=bs["theta"],
            vega=bs["vega"],
            open_interest=round(oi),
            volume_24h=round(volume),
            contract_multiplier=config.mock_option_setting(index, "MULTIPLIER"),
            min_quantity=1,
            quantity_step=1,
            tick_size=tick,
            underlying_price=spot,
            as_of_ms=now_ms,
            source=self.name
        )


#
# Deterministic pre-filter
#

STRATEGY_BY_DIRECTION = {"bullish": "LONG_CALL", "bearish": "LONG_PUT"}
OPTION_TYPE_BY_STRATEGY = {"LONG_CALL": "call", "LONG_PUT": "put"}

# Days after entry at which the thesis is assumed to play out,
# used only to estimate option value at target / invalidation.
HORIZON_DAYS = {
    "scalp": 0.25,
    "intraday": 1.0,
    "short_term": 3.0,
    "swing": 5.0,
    "position": 20.0,
    "unknown": 2.0,
}


STALE_ENTRY_REASON = "stale_entry_move_extended"


def entry_move_pct(direction, entry_level, current_price):
    """
    How far the underlying has already moved beyond the entry level
    in the intended trade direction, in percent. Negative = not yet
    reached / pulled back.

        bullish: (current - entry) / entry * 100
        bearish: (entry - current) / entry * 100
    """
    if direction == "bullish":
        return (current_price - entry_level) / entry_level * 100
    if direction == "bearish":
        return (entry_level - current_price) / entry_level * 100
    raise ValueError(f"no entry move for direction {direction!r}")


def check_entry_extension(direction, entry_level, current_price, max_move_pct):
    """
    Chase protection. Returns (allowed, move_pct, reason).

    Boundary: a move EXACTLY equal to max_move_pct is allowed; only a
    move strictly greater is rejected. Rounded to 9 decimals so float
    noise cannot flip the boundary.
    """
    if entry_level is None or entry_level <= 0:
        return False, None, f"{STALE_ENTRY_REASON}: no entry reference level to measure the move from"

    move = entry_move_pct(direction, entry_level, current_price)

    if round(move, 9) > max_move_pct:
        return False, move, (
            f"{STALE_ENTRY_REASON}: {direction} move {move:+.3f}% beyond entry "
            f"{entry_level} (price {current_price}) exceeds max {max_move_pct:g}%"
        )

    return True, move, None


def build_playbook() -> OptionsPlaybook:
    return OptionsPlaybook(
        allowed_strategies=["LONG_CALL", "LONG_PUT"],
        allowed_underlyings=config.allowed_underlyings(),
        min_dte=config.risk("MIN_DTE"),
        max_dte=config.risk("MAX_DTE"),
        min_abs_delta=config.risk("MIN_ABS_DELTA"),
        max_abs_delta=config.risk("MAX_ABS_DELTA"),
        max_spread_pct=config.risk("MAX_BID_ASK_SPREAD_PCT"),
        min_open_interest=config.risk("MIN_OPEN_INTEREST"),
        min_volume=config.risk("MIN_OPTION_VOLUME"),
        max_premium_per_trade=config.risk("MAX_PREMIUM_PER_TRADE"),
        max_contracts=config.risk("MAX_CONTRACTS"),
        max_quote_age_seconds=config.risk("MAX_QUOTE_AGE_SECONDS"),
        fee_per_contract=config.risk("OPTION_FEE_PER_CONTRACT"),
        max_llm_candidates=config.agent2_max_llm_candidates(),
    )


def _value_at(contract: OptionContract, price, days_elapsed):
    """Estimated option value if the underlying is at `price` after `days_elapsed`."""
    if price is None:
        return None
    remaining_years = max(contract.dte - days_elapsed, 0) / 365
    if contract.iv is None:
        return black_scholes(contract.option_type, price, contract.strike, 0, 0)["price"]
    return black_scholes(contract.option_type, price, contract.strike, remaining_years, contract.iv)["price"]


def reject_reason(contract: OptionContract, option_type, thesis: ThesisSnapshot, playbook: OptionsPlaybook, now_ms):
    """First failing rule for one contract, or None. Order matches the documented table."""
    if contract.option_type != option_type:
        return "wrong_option_type"

    if not (playbook.min_dte <= contract.dte <= playbook.max_dte):
        return "dte_out_of_range"

    if contract.delta is None or not (playbook.min_abs_delta <= abs(contract.delta) <= playbook.max_abs_delta):
        return "delta_out_of_range"

    if not contract.bid or not contract.ask or contract.bid <= 0 or contract.ask < contract.bid:
        return "no_valid_quote"

    if now_ms - contract.as_of_ms > playbook.max_quote_age_seconds * 1000:
        return "stale_quote"

    mid = (contract.bid + contract.ask) / 2
    if (contract.ask - contract.bid) / mid * 100 > playbook.max_spread_pct:
        return "spread_too_wide"

    if (contract.open_interest or 0) < playbook.min_open_interest or (contract.volume_24h or 0) < playbook.min_volume:
        return "insufficient_liquidity"

    if option_type == "call" and contract.strike >= thesis.target:
        return "strike_beyond_target"
    if option_type == "put" and contract.strike <= thesis.target:
        return "strike_beyond_target"

    return None


def size_order(contract: OptionContract, playbook: OptionsPlaybook):
    """Deterministic (limit_price, quantity, max_loss) or None if unaffordable."""
    mid = (contract.bid + contract.ask) / 2
    limit = min(contract.ask, round_up_to_tick(mid, contract.tick_size))
    cost_per_contract = limit * contract.contract_multiplier + playbook.fee_per_contract

    if cost_per_contract <= 0:
        return None

    affordable = math.floor(playbook.max_premium_per_trade / cost_per_contract + 1e-9)
    quantity = min(playbook.max_contracts, affordable)
    quantity -= quantity % contract.quantity_step

    if quantity < contract.min_quantity:
        return None

    return limit, quantity, round(cost_per_contract * quantity, 8)


def prefilter(chain: OptionChain, thesis: ThesisSnapshot, playbook: OptionsPlaybook, now_ms: int):
    """
    Returns (candidates, filter_summary). Candidates are ranked and
    capped at playbook.max_llm_candidates. Every rejection is counted.
    """
    summary: dict[str, int] = {}

    def reject(reason, n=1):
        summary[reason] = summary.get(reason, 0) + n

    if chain.underlying not in playbook.allowed_underlyings:
        reject("underlying_not_allowed", len(chain.contracts))
        return [], summary

    strategy = STRATEGY_BY_DIRECTION.get(thesis.direction)
    if strategy is None or strategy not in playbook.allowed_strategies:
        reject("no_allowed_strategy_for_direction", len(chain.contracts))
        return [], summary

    option_type = OPTION_TYPE_BY_STRATEGY[strategy]
    horizon = HORIZON_DAYS[thesis.time_horizon]
    scored = []

    for contract in chain.contracts:
        reason = reject_reason(contract, option_type, thesis, playbook, now_ms)
        if reason:
            reject(reason)
            continue

        sized = size_order(contract, playbook)
        if sized is None:
            reject("unaffordable")
            continue

        limit, quantity, max_loss = sized
        mid = (contract.bid + contract.ask) / 2
        intrinsic = (
            max(thesis.target - contract.strike, 0) if option_type == "call"
            else max(contract.strike - thesis.target, 0)
        )
        value_at_target = _value_at(contract, thesis.target, horizon)
        value_at_invalidation = _value_at(contract, thesis.invalidation, horizon)

        scored.append({
            "strategy": strategy,
            "contract": contract,
            "mid": mid,
            "spread_pct": (contract.ask - contract.bid) / mid * 100,
            "limit_price": limit,
            "quantity": quantity,
            "estimated_max_loss": max_loss,
            "breakeven": contract.strike + limit if option_type == "call" else contract.strike - limit,
            "intrinsic_at_target": intrinsic,
            "estimated_value_at_target": value_at_target,
            "estimated_value_at_invalidation": value_at_invalidation,
            "reward_to_risk_at_target": (value_at_target - limit) / limit,
        })

    scored.sort(key=lambda c: (
        -c["reward_to_risk_at_target"],
        abs(abs(c["contract"].delta) - 0.5),
        c["spread_pct"],
        -(c["contract"].open_interest or 0),
    ))

    kept = scored[:playbook.max_llm_candidates]
    if len(scored) > len(kept):
        reject("ranked_below_cutoff", len(scored) - len(kept))

    candidates = [
        OptionCandidate(candidate_id=f"C{i}", deterministic_rank=i, **c)
        for i, c in enumerate(kept, start=1)
    ]

    return candidates, summary
