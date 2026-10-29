"""
rebalance.py
------------
Decide which open positions to sell (fully or partially) so the book respects
the live portfolio limits in config.py. Pure logic — prices come in from
LiveTrader.price_open_positions(), sells are recorded by tracker.record_sell().

Rules, applied in order to a per-trade "keep fraction" (starts at 1.0):
  1. Edge gone      — model prob minus best bid below EXIT_EDGE → sell all
  2. Little upside  — token mid above MAX_ENTRY_PRICE → sell all
  3. Kelly trim     — position value above today's Kelly target → trim to target
  4. Group cap      — one asset + expiry above MAX_GROUP_EXPOSURE → scale down
  5. Total cap      — all open positions above MAX_TOTAL_EXPOSURE → scale down
Caps are measured on cost basis (stake_usd), the same basis as "capital at risk".
"""

import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MAX_TOTAL_EXPOSURE, MAX_GROUP_EXPOSURE, MAX_ENTRY_PRICE, EXIT_EDGE, MIN_SELL_USD,
)
from tracker import entry_price


def group_key(trade: dict) -> tuple[str, str]:
    """Bets on the same asset and expiry date move together — cap them as one."""
    return trade["asset"], (trade.get("expiry") or "")[:10]


@dataclass
class SellOrder:
    trade: dict
    fraction: float          # of the current position
    price: float             # best bid
    cost: float              # cost basis being sold
    proceeds: float
    reasons: list[str] = field(default_factory=list)

    @property
    def pnl(self) -> float:
        return self.proceeds - self.cost


def plan_sells(priced: list[dict], open_trades: list[dict], nav: float) -> list[SellOrder]:
    """
    priced      : output of LiveTrader.price_open_positions()
    open_trades : every OPEN trade (unpriced ones still count toward caps)
    nav         : portfolio value the caps are a fraction of
    """
    keep    = {p["trade"]["id"]: 1.0 for p in priced if p["bid"] > 0}
    info    = {p["trade"]["id"]: p for p in priced if p["bid"] > 0}
    reasons = {tid: [] for tid in keep}

    # 1–3. Per-position rules
    for tid, p in info.items():
        t = p["trade"]
        value = t["stake_usd"] / entry_price(t) * p["mid"]
        if p["model_prob"] - p["bid"] < EXIT_EDGE:
            keep[tid] = 0.0
            reasons[tid].append(f"edge gone (model {p['model_prob']:.1%} vs bid {p['bid']:.1%})")
        elif p["mid"] > MAX_ENTRY_PRICE:
            keep[tid] = 0.0
            reasons[tid].append(f"little upside left (price {p['mid']:.3f})")
        elif value > p["kelly_target_usd"]:
            keep[tid] = p["kelly_target_usd"] / value
            reasons[tid].append(f"above Kelly target (${value:.0f} -> ${p['kelly_target_usd']:.0f})")

    def _scale_to_cap(trades: list[dict], cap_usd: float, label: str) -> None:
        fixed   = sum(t["stake_usd"] for t in trades if t["id"] not in keep)
        movable = sum(t["stake_usd"] * keep[t["id"]] for t in trades if t["id"] in keep)
        if fixed + movable <= cap_usd or movable <= 0:
            return
        s = max(cap_usd - fixed, 0.0) / movable
        for t in trades:
            if t["id"] in keep and keep[t["id"]] > 0:
                keep[t["id"]] *= s
                reasons[t["id"]].append(label)

    # 4. Group caps
    groups: dict[tuple, list[dict]] = {}
    for t in open_trades:
        groups.setdefault(group_key(t), []).append(t)
    for (asset, expiry), trades in groups.items():
        _scale_to_cap(trades, MAX_GROUP_EXPOSURE * nav,
                      f"{asset} {expiry} group over {MAX_GROUP_EXPOSURE:.0%} cap")

    # 5. Total cap
    _scale_to_cap(open_trades, MAX_TOTAL_EXPOSURE * nav,
                  f"total exposure over {MAX_TOTAL_EXPOSURE:.0%} cap")

    orders = []
    for tid, k in keep.items():
        frac = 1.0 - k
        if frac <= 1e-6:
            continue
        p = info[tid]
        t = p["trade"]
        shares   = t["stake_usd"] / entry_price(t)
        proceeds = shares * frac * p["bid"]
        if proceeds < MIN_SELL_USD and k > 0:
            continue   # skip tiny trims; full exits always go through
        orders.append(SellOrder(t, frac, p["bid"], t["stake_usd"] * frac, proceeds, reasons[tid]))

    orders.sort(key=lambda o: -o.proceeds)
    return orders


def cap_new_stake(stake: float, trade_like: dict, open_trades: list[dict],
                  nav: float, cash: float) -> float:
    """
    Shrink a new bet's stake to fit the total cap, its group cap and available
    cash. trade_like needs asset and expiry. Returns 0 if nothing fits.
    """
    total = sum(t["stake_usd"] for t in open_trades)
    grp   = sum(t["stake_usd"] for t in open_trades if group_key(t) == group_key(trade_like))
    room  = min(
        MAX_TOTAL_EXPOSURE * nav - total,
        MAX_GROUP_EXPOSURE * nav - grp,
        cash,
    )
    return max(min(stake, room), 0.0)
