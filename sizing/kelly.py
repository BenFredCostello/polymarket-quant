"""
kelly.py
--------
Correlation-aware fractional Kelly bet sizing for binary prediction markets.

Standard Kelly for binary bet:
  f* = (p*b - (1-p)) / b
where p = model probability, b = net odds (payout per $1 risked)

Extensions:
  1. Fractional Kelly: scale by fraction (default 0.25) to account for model uncertainty
  2. Correlation penalty: reduce sizing when cross-asset correlations are elevated
  3. Edge threshold: only bet when model_prob - market_prob > min_edge
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
from typing import Optional

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from config import KELLY_FRACTION, MIN_EDGE, MAX_EDGE, MAX_POSITION, CORR_THRESHOLD

logger = logging.getLogger(__name__)

DEFAULT_KELLY_FRACTION = KELLY_FRACTION
DEFAULT_MIN_EDGE       = MIN_EDGE
DEFAULT_MAX_EDGE       = MAX_EDGE
DEFAULT_MAX_POSITION   = MAX_POSITION
DEFAULT_CORR_THRESHOLD = CORR_THRESHOLD


@dataclass
class BetSizing:
    asset: str
    market_prob: float     # Polymarket implied probability
    model_prob: float      # our model's probability
    edge: float            # model_prob - market_prob
    kelly_f: float         # raw (full) Kelly fraction
    fractional_kelly: float  # after Kelly fraction scaling
    corr_penalty: float    # correlation multiplier [0,1]
    final_stake: float     # final recommended stake as fraction of bankroll
    should_bet: bool
    reasoning: str


def polymarket_odds(market_prob: float) -> float:
    """
    Convert Polymarket YES price to net odds b.
    If you buy YES at price p, you risk p to win (1-p).
    Net odds b = (1-p)/p  (i.e. payout per $1 risked).
    """
    assert 0 < market_prob < 1, f"market_prob must be in (0,1), got {market_prob}"
    return (1.0 - market_prob) / market_prob


def kelly_criterion(
    p: float,
    b: float,
) -> float:
    """
    Standard Kelly fraction for a binary bet.

    f* = (p*b - (1-p)) / b = p - (1-p)/b

    Parameters
    ----------
    p : model probability of winning (YES resolves)
    b : net odds — payout per $1 staked

    Returns
    -------
    float — optimal fraction of bankroll to wager (can be negative = don't bet)
    """
    return (p * b - (1 - p)) / b


def size_bet(
    asset: str,
    model_prob: float,
    market_prob: float,
    corr_penalty: float = 1.0,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    min_edge: float = DEFAULT_MIN_EDGE,
    max_edge: float = DEFAULT_MAX_EDGE,
    max_position: float = DEFAULT_MAX_POSITION,
    title: str = "",
    side: str = "YES",
    pricing_date: str = "",
    expiry: str = "",
    current_price: float = None,
    sigma: float = None,
) -> BetSizing:
    """
    Compute the recommended stake for a single binary bet.

    Parameters
    ----------
    asset         : asset name (for logging)
    model_prob    : model's estimated probability of YES
    market_prob   : Polymarket's implied probability of YES
    corr_penalty  : multiplier from correlation analysis [0,1]
    kelly_fraction : fractional Kelly scaling factor
    min_edge      : minimum edge to trigger a bet
    max_position  : maximum stake as fraction of bankroll

    Returns
    -------
    BetSizing dataclass
    """
    assert 0 < model_prob < 1, "model_prob must be in (0,1)"
    assert 0 < market_prob < 1, "market_prob must be in (0,1)"
    assert 0 < corr_penalty <= 1, "corr_penalty must be in (0,1]"

    edge = model_prob - market_prob
    b = polymarket_odds(market_prob)
    f_raw = kelly_criterion(model_prob, b)

    # Apply fractional scaling
    f_fractional = f_raw * kelly_fraction

    # Apply correlation penalty
    f_adj = f_fractional * corr_penalty

    # Cap at max_position, floor at 0 (never short)
    final_stake = float(np.clip(f_adj, 0.0, max_position))

    should_bet = (edge >= min_edge) and (edge <= max_edge) and (f_raw > 0)
    reasoning = _build_reasoning(edge, f_raw, f_fractional, corr_penalty, final_stake, min_edge, max_edge)

    result = BetSizing(
        asset=asset,
        market_prob=market_prob,
        model_prob=model_prob,
        edge=edge,
        kelly_f=f_raw,
        fractional_kelly=f_fractional,
        corr_penalty=corr_penalty,
        final_stake=final_stake,
        should_bet=should_bet,
        reasoning=reasoning,
    )

    title_part = f" (title={title[:60]})" if title else ""
    date_part = f"  [{pricing_date}]" if pricing_date else ""
    expiry_part = f"  expiry={expiry}" if expiry else ""
    price_part = f"  S=${current_price:,.0f}" if current_price is not None else ""
    sigma_part = f"  σ={sigma:.0%}" if sigma is not None else ""
    msg = (f"  {asset}:{title_part}{date_part}{expiry_part}{price_part}{sigma_part}  edge={edge:+.2%}  Kelly={f_raw:.2%}  "
           f"fractional={f_fractional:.2%}  corr_mult={corr_penalty:.2f}  "
           f"stake={final_stake:.2%}")
    if should_bet:
        logger.info(msg + f"  ✅ {side}")
    else:
        logger.debug(msg + "  ❌ SKIP")
    return result


def size_portfolio(
    opportunities: list[dict],
    corr_matrix: Optional[pd.DataFrame] = None,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    min_edge: float = DEFAULT_MIN_EDGE,
    max_position: float = DEFAULT_MAX_POSITION,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
) -> list[BetSizing]:
    """
    Size a portfolio of bets with correlation-aware Kelly.

    Parameters
    ----------
    opportunities : list of dicts with keys: asset, model_prob, market_prob
    corr_matrix   : current rolling correlation matrix (from correlation.py)
    ...

    Returns
    -------
    list of BetSizing objects, sorted by final_stake descending
    """
    # Compute correlation penalties
    if corr_matrix is not None:
        from analysis.correlation import correlation_penalty
        penalties = correlation_penalty(corr_matrix, threshold=corr_threshold)
    else:
        penalties = {}

    results = []
    for opp in opportunities:
        asset = opp["asset"]
        penalty = penalties.get(asset, 1.0)
        sizing = size_bet(
            asset=asset,
            model_prob=opp["model_prob"],
            market_prob=opp["market_prob"],
            corr_penalty=penalty,
            kelly_fraction=kelly_fraction,
            min_edge=min_edge,
            max_position=max_position,
        )
        results.append(sizing)

    # Portfolio-level check: total exposure
    total_stake = sum(r.final_stake for r in results if r.should_bet)
    if total_stake > 0.50:
        logger.warning(
            f"⚠️  Total portfolio stake {total_stake:.1%} exceeds 50%. "
            "Consider reducing individual positions."
        )

    results.sort(key=lambda x: -x.final_stake)
    return results


def _build_reasoning(edge, f_raw, f_frac, corr_penalty, final_stake, min_edge, max_edge=float("inf")) -> str:
    parts = []
    if edge > max_edge:
        parts.append(f"edge {edge:+.1%} exceeds max {max_edge:.1%} — skipped as artifact")
    elif edge < min_edge:
        parts.append(f"edge {edge:+.1%} below min {min_edge:.1%}")
    elif f_raw <= 0:
        parts.append("negative Kelly — model does not favour YES")
    else:
        parts.append(f"edge {edge:+.1%}")
        parts.append(f"Kelly {f_raw:.1%} → fractional {f_frac:.1%}")
        if corr_penalty < 1.0:
            parts.append(f"corr penalty ×{corr_penalty:.2f}")
        parts.append(f"stake {final_stake:.1%} of bankroll")
    return " | ".join(parts)


def backtest_sizing(
    historical_bets: list[dict],
    bankroll: float = 1000.0,
    **kelly_kwargs,
) -> pd.DataFrame:
    """
    Simulate historical P&L using the Kelly sizer.

    Parameters
    ----------
    historical_bets : list of dicts with keys:
        asset, model_prob, market_prob, outcome (1=YES, 0=NO)
    bankroll        : starting bankroll

    Returns
    -------
    DataFrame with columns: asset, stake_$, outcome, pnl, bankroll
    """
    records = []
    current_bankroll = bankroll

    for bet in historical_bets:
        sizing = size_bet(
            asset=bet["asset"],
            model_prob=bet["model_prob"],
            market_prob=bet["market_prob"],
            corr_penalty=bet.get("corr_penalty", 1.0),
            title=bet.get("title", ""),
            side=bet.get("side", "YES"),
            pricing_date=str(bet.get("pricing_date", "")),
            expiry=str(bet.get("expiry", "")),
            current_price=bet.get("current_price"),
            sigma=bet.get("sigma"),
            **{k: v for k, v in kelly_kwargs.items()
               if k in ("kelly_fraction", "min_edge", "max_edge", "max_position")},
        )

        if not sizing.should_bet:
            records.append({
                "asset": bet["asset"],
                "stake_$": 0,
                "outcome": bet["outcome"],
                "pnl": 0,
                "bankroll": current_bankroll,
                "stake_frac": 0,
            })
            continue

        stake = sizing.final_stake * current_bankroll
        b = polymarket_odds(sizing.market_prob)
        pnl = stake * b if bet["outcome"] == 1 else -stake
        current_bankroll += pnl

        records.append({
            "asset": bet["asset"],
            "stake_$": stake,
            "outcome": bet["outcome"],
            "pnl": pnl,
            "bankroll": current_bankroll,
            "stake_frac": sizing.final_stake,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df["cumulative_pnl"] = df["pnl"].cumsum()
        df["bankroll_pct_chg"] = (df["bankroll"] / bankroll - 1) * 100
    return df


def validate_kelly() -> None:
    """Validation tests for the Kelly sizer."""
    # Test 1: Zero edge → don't bet
    s = size_bet("BTC", model_prob=0.50, market_prob=0.50)
    assert not s.should_bet, "No edge should not trigger a bet."

    # Test 2: Clear positive edge → bet
    s = size_bet("BTC", model_prob=0.65, market_prob=0.50)
    assert s.should_bet, "Clear edge should trigger a bet."
    assert s.final_stake > 0

    # Test 3: Negative Kelly (model_prob < market_prob) → no bet
    s = size_bet("BTC", model_prob=0.30, market_prob=0.60)
    assert not s.should_bet, "Negative Kelly should not bet."

    # Test 4: Max position cap
    s = size_bet("BTC", model_prob=0.99, market_prob=0.01, max_position=0.10)
    assert s.final_stake <= 0.10, f"Should be capped at 10%, got {s.final_stake:.2%}"

    # Test 5: Correlation penalty reduces stake
    s_no_penalty = size_bet("ETH", model_prob=0.65, market_prob=0.50, corr_penalty=1.0)
    s_with_penalty = size_bet("ETH", model_prob=0.65, market_prob=0.50, corr_penalty=0.5)
    assert s_with_penalty.final_stake < s_no_penalty.final_stake, \
        "Correlation penalty should reduce stake."

    # Test 6: Kelly formula correctness
    # With p=0.6, market_prob=0.5, b=(1-0.5)/0.5=1.0 → f* = (0.6*1 - 0.4)/1 = 0.2
    s = size_bet("BTC", model_prob=0.60, market_prob=0.50, kelly_fraction=1.0)
    assert abs(s.kelly_f - 0.20) < 0.001, f"Kelly formula error: {s.kelly_f:.4f}"

    # Test 7: Portfolio sizing
    opps = [
        {"asset": "BTC", "model_prob": 0.65, "market_prob": 0.50},
        {"asset": "ETH", "model_prob": 0.60, "market_prob": 0.50},
        {"asset": "SOL", "model_prob": 0.40, "market_prob": 0.50},
    ]
    results = size_portfolio(opps)
    assert len(results) == 3
    sol_result = next(r for r in results if r.asset == "SOL")
    assert not sol_result.should_bet, "SOL with no edge should not bet."

    logger.info("✅ All Kelly sizing validation tests passed.")
    print("✅ All Kelly sizing validation tests passed.")


if __name__ == "__main__":
    validate_kelly()

    opps = [
        {"asset": "BTC", "model_prob": 0.62, "market_prob": 0.50},
        {"asset": "ETH", "model_prob": 0.58, "market_prob": 0.50},
        {"asset": "SOL", "model_prob": 0.71, "market_prob": 0.52},
    ]

    print("\n── Portfolio Sizing ──")
    results = size_portfolio(opps, min_edge=0.05)
    print(f"\n{'Asset':<6} {'Edge':>7} {'Kelly':>7} {'Stake':>7} {'Bet?'}")
    print("-" * 40)
    for r in results:
        bet_str = "✅" if r.should_bet else "❌"
        print(f"{r.asset:<6} {r.edge:>+7.1%} {r.kelly_f:>7.1%} {r.final_stake:>7.1%}  {bet_str}")
