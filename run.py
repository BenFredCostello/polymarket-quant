"""
run.py
------
Single-run entry point. Run this whenever you want to:
  1. Check if any open bets have resolved
  2. Scan for new opportunities and record confirmed bets
  3. Refresh the Excel file

Usage:
  python run.py               # full run: check resolutions + scan + prompt
  python run.py --check       # only check resolutions, skip scan
  python run.py --export      # only regenerate Excel from existing data
"""

import os
import sys
import logging

# Project root on path before any local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress noisy INFO logs from data-fetching modules
logging.basicConfig(level=logging.WARNING)

from tracker import (
    _load, _save, record_bet, check_resolutions, export_excel,
    get_portfolio_stats, EXCEL_FILE, STARTING_BANKROLL,
)

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


def run_scan(bankroll: float) -> list:
    """Run one scan cycle and return BET-flagged signals."""
    _section("Scanning Polymarket for opportunities")
    print(f"  Bankroll available: ${bankroll:,.2f}")
    print("  (This takes ~1–2 min while fetching prices & running models…)\n")

    from execution.trader import LiveTrader

    # Suppress any remaining INFO noise after the import sets up logging
    logging.getLogger().setLevel(logging.WARNING)

    trader  = LiveTrader(bankroll=bankroll, paper_trade=True)
    signals = trader.scan()
    bets    = [s for s in signals if s.action == "BET"]

    print(f"\n  Priced {len(signals)} markets ->{len(bets)} bet opportunities found.")
    return bets


def prompt_and_record(bets: list, bankroll: float) -> float:
    """Display opportunities, prompt for selection, record confirmed bets."""
    if not bets:
        return bankroll

    # Drop any market already in an open position
    data = _load()
    open_cids = {t["condition_id"] for t in data["trades"] if t["status"] == "OPEN"}
    already_open = [b for b in bets if b.condition_id in open_cids]
    bets = [b for b in bets if b.condition_id not in open_cids]
    if already_open:
        print(f"  Skipped {len(already_open)} opportunit{'y' if len(already_open) == 1 else 'ies'} already in open positions.")
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
    choice = input("  > ").strip().lower()

    if choice in ("none", ""):
        to_bet = []
    elif choice == "all":
        to_bet = bets
    else:
        indices = []
        for token in choice.split():
            try:
                idx = int(token) - 1
                if 0 <= idx < len(bets):
                    indices.append(idx)
                else:
                    print(f"  Ignored out-of-range: {token}")
            except ValueError:
                print(f"  Ignored unrecognised input: {token}")
        to_bet = [bets[i] for i in indices]

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

    mode_check  = "--check"  in sys.argv
    mode_export = "--export" in sys.argv

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
        path = export_excel()
        print(f"\n  Excel refreshed -> {path}\n")
        return

    # ── Step 1: Check resolutions ──────────────────────────────────────────────
    resolved = run_check()

    # Reload stats after resolution update
    data   = _load()
    trades = data.get("trades", [])
    stats  = get_portfolio_stats(trades, starting_broll)

    if mode_check:
        path = export_excel()
        print(f"\n  Excel updated ->{path}")
        print(f"  Portfolio value: ${stats['net_value']:.2f}\n")
        return

    # ── Step 2: Scan ───────────────────────────────────────────────────────────
    # Kelly sizes on total portfolio value (the correct wealth basis).
    # Available capital (liquid cash) is tracked separately for affordability checks.
    available = max(stats["net_value"] - stats["at_risk"], 0)
    bets      = run_scan(bankroll=stats["net_value"])

    # ── Step 3: Record bets ────────────────────────────────────────────────────
    prompt_and_record(bets, bankroll=available)

    # ── Final: Export Excel ────────────────────────────────────────────────────
    _section("Exporting Excel")
    path = export_excel()
    print(f"  Saved ->{path}")

    data   = _load()
    stats  = get_portfolio_stats(data["trades"], starting_broll)
    print(f"\n  Portfolio value:  ${stats['net_value']:.2f}")
    print(f"  Open positions:   {stats['open_bets']}")
    print(f"  Capital at risk:  ${stats['at_risk']:.2f}")
    print()


if __name__ == "__main__":
    main()
