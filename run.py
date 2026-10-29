"""
run.py
------
Single-run entry point. Run this whenever you want to:
  1. Check if any open bets have resolved
  2. Re-price open positions and sell down any that break the portfolio limits
  3. Scan for new opportunities and record confirmed bets
  4. Refresh the Excel file

Usage:
  python run.py               # full run: check resolutions + rebalance + scan + prompt
  python run.py --check       # only check resolutions, skip scan
  python run.py --rebalance   # check resolutions + sell down positions, skip scan
  python run.py --export      # only regenerate Excel from existing data
"""

import os
import sys
import logging
import subprocess
from datetime import datetime

# Project root on path before any local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress noisy INFO logs from data-fetching modules
logging.basicConfig(level=logging.WARNING)

from tracker import (
    _load, _save, record_bet, check_resolutions, export_excel,
    get_portfolio_stats, record_sell, EXCEL_FILE, STARTING_BANKROLL,
)
from config import MIN_STAKE_USD

DIVIDER = "-" * 62


def _banner():
    print()
    print("=" * 62)
    print("  POLYMARKET QUANT  -  Live Trading System")
    print("=" * 62)


def _section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def run_check():
    """Check open positions for resolution and update the store."""
    _section("Checking open positions for resolution")
    resolved = check_resolutions()

    if not resolved:
        print("  No open positions have resolved yet.")
    else:
        for t in resolved:
            icon = "[WON]" if t["status"] == "WON" else "[LOST]"
            print(f"  {icon}  {t['asset']} | {t['question'][:52]}")
            print(f"          P&L: ${t['pnl']:+.2f}   resolved ->{t['outcome']}")

    return resolved


def _push_excel(path) -> None:
    """Commit just the Excel file and push it. Never fails the run."""
    repo = os.path.dirname(os.path.abspath(__file__))
    name = os.path.basename(path)

    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)

    try:
        git("add", "--", name)
        if git("diff", "--cached", "--quiet", "--", name).returncode == 0:
            return   # unchanged
        msg = f"Update trades Excel {datetime.now():%Y-%m-%d %H:%M}"
        # Commit only this path so anything else staged is left alone
        r = git("commit", "-m", msg, "--", name)
        if r.returncode != 0:
            print(f"  (Excel not committed: {(r.stderr or r.stdout).strip()[:200]})")
            return
        r = git("push")
        if r.returncode != 0:
            print(f"  (Excel committed but push failed: {r.stderr.strip()[:200]})")
        else:
            print("  Excel pushed to GitHub.")
    except FileNotFoundError:
        print("  (git not found — Excel not pushed)")


def _export() -> str:
    """Regenerate the Excel and push it to GitHub."""
    try:
        path = export_excel()
    except PermissionError:
        print("  (Excel is open — close it and run `python run.py --export` to refresh)")
        return None
    if path:
        _push_excel(path)
    return path


def _make_trader(bankroll: float):
    from execution.trader import LiveTrader

    # Suppress any remaining INFO noise after the import sets up logging
    logging.getLogger().setLevel(logging.WARNING)
    return LiveTrader(bankroll=bankroll, paper_trade=True)


def _parse_choice(choice: str, n: int) -> list[int]:
    """'all' / 'none' / '1 3' -> zero-based indices."""
    if choice in ("none", ""):
        return []
    if choice == "all":
        return list(range(n))
    indices = []
    for token in choice.split():
        try:
            idx = int(token) - 1
            if 0 <= idx < n:
                indices.append(idx)
            else:
                print(f"  Ignored out-of-range: {token}")
        except ValueError:
            print(f"  Ignored unrecognised input: {token}")
    return indices


def run_rebalance(trader, nav: float) -> None:
    """Re-price open positions and offer sells that bring the book within limits."""
    from execution.rebalance import plan_sells

    _section("Re-pricing open positions")
    data        = _load()
    open_trades = [t for t in data["trades"] if t["status"] == "OPEN"]
    if not open_trades:
        print("  No open positions.")
        return

    print("  (Fetching prices & running models…)")
    priced = trader.price_open_positions(open_trades)
    orders = plan_sells(priced, open_trades, nav)
    if not orders:
        print("  All open positions within limits — nothing to sell.")
        return

    for i, o in enumerate(orders, 1):
        t = o.trade
        print(f"\n  [{i}]  {t['asset']}  |  {t['side']}  |  Sell {o.fraction:.0%} @ {o.price:.3f}  ->  "
              f"${o.proceeds:.2f}  (P&L ${o.pnl:+.2f})")
        print(f"       {t['question'][:65]}")
        print(f"       Expires: {(t.get('expiry') or '')[:10]}   Why: {'; '.join(o.reasons)}")

    print(f"\n  Selling all would free ${sum(o.proceeds for o in orders):.2f} "
          f"(realised P&L ${sum(o.pnl for o in orders):+.2f}).")
    print(f"  Enter numbers to sell (e.g. 1  or  1 3  or  all  or  none):")
    chosen = [orders[i] for i in _parse_choice(input("  > ").strip().lower(), len(orders))]

    if not chosen:
        print("  No sells recorded this run.")
        return
    print()
    for o in chosen:
        sold = record_sell(o.trade["id"], o.fraction, o.price)
        print(f"  Sold  [{sold['id']}]  {o.trade['asset']} {o.trade['side']}  "
              f"{o.fraction:.0%} @ {o.price:.3f}  ->  ${o.proceeds:.2f}  (P&L ${sold['pnl']:+.2f})")


def run_scan(trader) -> list:
    """Run one scan cycle and return BET-flagged signals."""
    _section("Scanning Polymarket for opportunities")
    print(f"  Bankroll: ${trader.bankroll:,.2f}")
    print("  (This takes ~1–2 min while fetching prices & running models…)\n")

    signals = trader.scan()
    bets    = [s for s in signals if s.action == "BET"]

    print(f"\n  Priced {len(signals)} markets ->{len(bets)} bet opportunities found.")
    return bets


def prompt_and_record(bets: list, bankroll: float, nav: float) -> float:
    """Display opportunities, prompt for selection, record confirmed bets."""
    from execution.rebalance import cap_new_stake

    if not bets:
        return bankroll

    # Drop any market already in an open position
    data = _load()
    open_trades = [t for t in data["trades"] if t["status"] == "OPEN"]
    open_cids = {t["condition_id"] for t in open_trades}
    already_open = [b for b in bets if b.condition_id in open_cids]
    bets = [b for b in bets if b.condition_id not in open_cids]
    if already_open:
        print(f"  Skipped {len(already_open)} opportunit{'y' if len(already_open) == 1 else 'ies'} already in open positions.")

    # Shrink stakes to fit the exposure caps and cash, best edge first.
    # Any subset of the bets shown below also fits.
    bets.sort(key=lambda s: -abs(s.edge_bs))
    sim_open, sim_cash, fitted = list(open_trades), bankroll, []
    for sig in bets:
        like  = {"asset": sig.asset, "expiry": sig.end_date or ""}
        stake = cap_new_stake(sig.recommended_stake_usd, like, sim_open, nav, sim_cash)
        if stake < MIN_STAKE_USD:
            continue
        sig.recommended_stake_usd = round(stake, 2)
        sim_open.append({**like, "stake_usd": stake})
        sim_cash -= stake
        fitted.append(sig)
    n_capped = len(bets) - len(fitted)
    if n_capped:
        print(f"  Skipped {n_capped} opportunit{'y' if n_capped == 1 else 'ies'} — no room under exposure caps / cash.")
    bets = fitted
    if not bets:
        return bankroll

    _section("Bet opportunities")
    for i, sig in enumerate(bets, 1):
        model_prob = (sig.model_prob_bs + sig.model_prob_mc) / 2
        print(f"\n  [{i}]  {sig.asset}  |  {sig.side}  |  "
              f"Edge: {sig.edge_bs:+.1%}  |  Stake: ${sig.recommended_stake_usd:.2f}")
        print(f"       {sig.question[:65]}")
        print(f"       Market: {sig.market_prob:.1%}   Model: {model_prob:.1%}   "
              f"Expires: {(sig.end_date or '')[:10]}   "
              f"Vol24h: ${sig.volume_24h:,.0f}")

    print(f"\n  Enter numbers to record (e.g. 1  or  1 3  or  all  or  none):")
    to_bet = [bets[i] for i in _parse_choice(input("  > ").strip().lower(), len(bets))]

    if not to_bet:
        print("  No bets recorded this run.")
        return bankroll

    print()
    skipped = 0
    for sig in to_bet:
        if sig.recommended_stake_usd > bankroll:
            skipped += 1
            continue
        trade = record_bet(sig)
        bankroll -= sig.recommended_stake_usd
        print(f"  Recorded  [{trade['id']}]  {sig.asset} {sig.side}  "
              f"${sig.recommended_stake_usd:.2f}  (bankroll ->${bankroll:.2f})")

    if skipped:
        print(f"  Skipped {skipped} bet(s) — insufficient bankroll.")
    return bankroll


def main():
    _banner()

    mode_check     = "--check"     in sys.argv
    mode_export    = "--export"    in sys.argv
    mode_rebalance = "--rebalance" in sys.argv

    # ── Load state ─────────────────────────────────────────────────────────────
    data             = _load()
    starting_broll   = data.get("starting_bankroll", STARTING_BANKROLL)
    trades           = data.get("trades", [])
    stats            = get_portfolio_stats(trades, starting_broll)

    print(f"\n  Portfolio: ${stats['net_value']:.2f}  |  "
          f"Open positions: {stats['open_bets']}  |  "
          f"Resolved: {stats['resolved_bets']}  |  "
          f"Win rate: {stats['win_rate']:.0%}" if stats["resolved_bets"] else
          f"\n  Portfolio: ${stats['net_value']:.2f}  |  No resolved bets yet.")

    if mode_export:
        path = _export()
        if path: print(f"\n  Excel refreshed -> {path}\n")
        return

    # ── Step 1: Check resolutions ──────────────────────────────────────────────
    resolved = run_check()

    # Reload stats after resolution update
    data   = _load()
    trades = data.get("trades", [])
    stats  = get_portfolio_stats(trades, starting_broll)

    if mode_check:
        path = _export()
        if path: print(f"\n  Excel updated ->{path}")
        print(f"  Portfolio value: ${stats['net_value']:.2f}\n")
        return

    # ── Step 2: Rebalance open positions ───────────────────────────────────────
    trader = _make_trader(bankroll=stats["net_value"])
    run_rebalance(trader, nav=stats["net_value"])

    data  = _load()
    stats = get_portfolio_stats(data["trades"], starting_broll)

    if mode_rebalance:
        path = _export()
        if path: print(f"\n  Excel updated ->{path}")
        print(f"  Portfolio value: ${stats['net_value']:.2f}   "
              f"Capital at risk: ${stats['at_risk']:.2f}\n")
        return

    # ── Step 3: Scan ───────────────────────────────────────────────────────────
    # Kelly sizes on total portfolio value (the correct wealth basis).
    # Available capital (liquid cash) is tracked separately for affordability checks.
    trader.bankroll = stats["net_value"]
    available = max(stats["net_value"] - stats["at_risk"], 0)
    bets      = run_scan(trader)

    # ── Step 4: Record bets ────────────────────────────────────────────────────
    prompt_and_record(bets, bankroll=available, nav=stats["net_value"])

    # ── Final: Export Excel ────────────────────────────────────────────────────
    _section("Exporting Excel")
    path = _export()
    if path: print(f"  Saved ->{path}")

    data   = _load()
    stats  = get_portfolio_stats(data["trades"], starting_broll)
    print(f"\n  Portfolio value:  ${stats['net_value']:.2f}")
    print(f"  Open positions:   {stats['open_bets']}")
    print(f"  Capital at risk:  ${stats['at_risk']:.2f}")
    print()


if __name__ == "__main__":
    main()
