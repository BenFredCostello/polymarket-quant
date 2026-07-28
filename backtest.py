"""
backtest.py
-----------
Backtest the BS/MC pricer against resolved Polymarket crypto contracts.

Pipeline:
  1. Fetch resolved markets from Polymarket Gamma API
  2. For each contract, find the pricing date (expiry - 7 days)
  3. Fetch historical market prices from Gamma API at that date
  4. Fetch historical S and sigma from yfinance at that date
  5. Price with BS + MC using calibrated jump params
  6. Feed into Kelly backtest_sizing() and plot equity curve
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BACKTEST_LIMIT, LOOKBACK_DAYS, PRICING_OFFSET_MEAN_DAYS, PRICING_OFFSET_SIGMA_LN,
    MIN_T_DAYS, MAX_T_DAYS, BANKROLL, SAVE_PLOTS, MC_PATHS,
    KELLY_FRACTION, MIN_EDGE, MAX_EDGE, MAX_POSITION, MAX_PAYOUT,
)
from data.fetch_crypto import build_crypto_dataset, build_dvol_cache, get_implied_vol
from data.fetch_polymarket import _resolve_expiry, fetch_resolved_markets, fetch_all_market_probs
from pricing.black_scholes import BlackScholesBinaryPricer
from pricing.monte_carlo import MonteCarloBinaryPricer, calibrate_jump_params, DEFAULT_JUMP_PARAMS
from sizing.kelly import backtest_sizing
from analysis.correlation import get_current_corr_matrix, correlation_penalty

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Precompute lognormal mu so np.random.lognormal produces the right mean in day-space.
# E[X] = exp(mu + sigma²/2) = PRICING_OFFSET_MEAN_DAYS
_OFFSET_MU_LN = np.log(PRICING_OFFSET_MEAN_DAYS) - PRICING_OFFSET_SIGMA_LN ** 2 / 2

# ── Crypto data helpers ────────────────────────────────────────────────────────

def build_historical_lookup(crypto_data: dict) -> dict:
    """Build lookup dict for quick historical data access."""
    return {
        asset: df[["Close", "realised_vol_30d", "log_return"]].copy()
        for asset, df in crypto_data.items()
    }


def get_historical_inputs(lookup: dict, asset: str, pricing_date: pd.Timestamp):
    """Get S and sigma at pricing date from crypto data."""
    df = lookup.get(asset)
    if df is None:
        return None
    
    available = df[df.index <= pricing_date]
    if available.empty:
        return None
    
    row = available.iloc[-1]
    
    # Check if data is stale (>5 days old)
    if (pricing_date - available.index[-1]).days > 5:
        return None
    
    S = float(row["Close"])
    sigma = float(row["realised_vol_30d"]) if not pd.isna(row["realised_vol_30d"]) else None
    
    if sigma is None or sigma <= 0 or S <= 0:
        return None
    
    return S, sigma


# ── Main backtest ──────────────────────────────────────────────────────────────

def run_backtest(
    limit: int = BACKTEST_LIMIT,
    save_plots: bool = SAVE_PLOTS,
) -> pd.DataFrame:

    # ── 1. Fetch resolved contracts ────────────────────────────────────────────
    logger.info("Fetching resolved Polymarket contracts...")
    resolved = fetch_resolved_markets(limit=limit)
    logger.info(f"  {len(resolved)} resolved contracts fetched.")

    # Filter for valid contracts
    valid = [
        m for m in resolved
        if m.outcome in ("YES", "NO") and m.strike is not None and m.strike > 0
    ]
    logger.info(f"  {len(valid)} contracts with known outcome and strike.")
    
    if not valid:
        logger.error("No valid resolved contracts found. Exiting.")
        return pd.DataFrame()

    # ── 2. Fetch crypto history and calibrate jump params ──────────────────────
    logger.info(f"Fetching {LOOKBACK_DAYS} days of crypto history...")
    crypto_data = build_crypto_dataset(lookback_days=LOOKBACK_DAYS)
    lookup = build_historical_lookup(crypto_data)

    jump_params = {}
    for asset, df_asset in crypto_data.items():
        log_ret = df_asset["log_return"].dropna().values
        try:
            jump_params[asset] = calibrate_jump_params(log_ret)
            logger.info(
                f"Calibrated jump params: λ={jump_params[asset].lam:.2f}/yr  "
                f"μ_j={jump_params[asset].mu_j:.4f}  σ_j={jump_params[asset].sigma_j:.4f}"
            )
        except Exception as e:
            logger.warning(f"Jump calibration failed for {asset}: {e}. Using default.")
            jump_params[asset] = DEFAULT_JUMP_PARAMS

    # ── 3. Pre-filter: resolve expiry + spot/vol for each contract ─────────────
    bs_pricer = BlackScholesBinaryPricer(r=0.0)
    mc_pricer = MonteCarloBinaryPricer(n_paths=MC_PATHS, use_antithetic=True)

    priceable = []      # (market, expiry, pricing_ts, S, sigma)
    pricing_dates = []
    skipped = 0

    for market in valid:
        expiry = _resolve_expiry(market)
        if expiry is None:
            skipped += 1
            continue

        expiry_ts = pd.Timestamp(expiry)

        # Sample pricing offset from lognormal: mean≈PRICING_OFFSET_MEAN_DAYS, right-skewed tail.
        # createdAt from the /events endpoint is unreliable (all sub-markets share the
        # event's timestamp), so we always sample independently per contract.
        offset_days = int(np.clip(
            round(np.random.lognormal(_OFFSET_MU_LN, PRICING_OFFSET_SIGMA_LN)),
            MIN_T_DAYS, MAX_T_DAYS,
        ))
        pricing_ts = expiry_ts - timedelta(days=offset_days)

        # Check time to expiry bounds
        T_days = (expiry_ts - pricing_ts).days
        if T_days < MIN_T_DAYS or T_days > MAX_T_DAYS:
            skipped += 1
            continue
        
        # Get historical crypto data
        inputs = get_historical_inputs(lookup, market.asset, pricing_ts)
        if inputs is None:
            skipped += 1
            continue
        
        priceable.append((market, expiry, pricing_ts) + inputs)
        pricing_dates.append(pricing_ts)

    logger.info(f"  {len(priceable)} contracts priceable, {skipped} skipped.")

    if not priceable:
        logger.error("No priceable contracts found.")
        return pd.DataFrame()

    # ── 3b. Pre-fetch Deribit DVOL for BTC/ETH ────────────────────────────────
    logger.info("Pre-fetching Deribit implied volatility (DVOL)...")
    dvol_cache = build_dvol_cache(pricing_dates)

    # ── 4. Fetch historical market probabilities from Gamma API ─────────────────
    logger.info(f"Fetching historical market prices for {len(priceable)} contracts...")
    markets_list  = [item[0] for item in priceable]
    expiry_dates  = [pd.Timestamp(item[1]).date() for item in priceable]
    historical_probs = fetch_all_market_probs(markets_list, pricing_dates, expiry_dates)
    clob_hit = sum(1 for p in historical_probs if p is not None)

    # ── 5. Price each contract and build dataset ───────────────────────────────
    records = []

    for i, (market, expiry, pricing_ts, S, hist_sigma) in enumerate(priceable):
        K = market.strike
        is_put = market.option_type == "put"
        jp = jump_params.get(market.asset, DEFAULT_JUMP_PARAMS)
        T_at_pricing = (pd.Timestamp(expiry) - pricing_ts).days / 365.0

        sigma, sigma_source = get_implied_vol(market.asset, pricing_ts, dvol_cache, hist_sigma)

        try:
            # Price using both models
            bs_call = bs_pricer.price(S, K, T_at_pricing, sigma)
            mc_out = mc_pricer.price(S, K, T_at_pricing, sigma, jp)
            mc_call = mc_out["price"]
            
            # Adjust for put/call
            bs_p = (1.0 - bs_call) if is_put else bs_call
            mc_p = (1.0 - mc_call) if is_put else mc_call
            
            # Average the two models
            model_prob = float(np.clip((bs_p + mc_p) / 2, 1e-4, 1 - 1e-4))
            
        except Exception as e:
            logger.debug(f"Pricing error for {market.condition_id}: {e}")
            continue

        # Get market probability (historical if available, otherwise use model)
        hist_prob = historical_probs[i]
        if hist_prob is not None:
            market_prob = hist_prob
            source = "clob"
        else:
            market_prob = model_prob  # This yields zero edge - no bet
            source = "model_fallback"

        outcome = 1 if market.outcome == "YES" else 0
        edge = model_prob - market_prob

        # NO-side betting: when market overestimates YES, buy NO instead
        if source == "clob" and edge < -MIN_EDGE:
            side = "NO"
            adj_model_prob = 1.0 - model_prob
            adj_market_prob = 1.0 - market_prob
            adj_outcome = 1 - outcome
        else:
            side = "YES"
            adj_model_prob = model_prob
            adj_market_prob = market_prob
            adj_outcome = outcome

        records.append({
            "condition_id": market.condition_id,
            "asset": market.asset,
            "question": market.question[:70],
            "option_type": market.option_type,
            "strike": K,
            "expiry": expiry,
            "pricing_date": pricing_ts.date(),
            "S_at_pricing": S,
            "sigma": sigma,
            "sigma_source": sigma_source,
            "T_years": T_at_pricing,
            "bs_prob": bs_p,
            "mc_prob": mc_p,
            "model_prob": model_prob,
            "market_prob": market_prob,
            "market_prob_source": source,
            "edge": edge,
            "outcome": outcome,
            "side": side,
            "adj_model_prob": adj_model_prob,
            "adj_market_prob": adj_market_prob,
            "adj_outcome": adj_outcome,
        })

    logger.info(f"Priced {len(records)} contracts successfully.")
    
    if not records:
        logger.error("No contracts could be priced.")
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("pricing_date").reset_index(drop=True)

    # ── 6. Kelly backtest ──────────────────────────────────────────────────────
    # Compute correlation penalties once from the full history
    log_returns = {
        asset: frame["log_return"].dropna()
        for asset, frame in crypto_data.items()
        if "log_return" in frame.columns
    }
    corr_matrix  = get_current_corr_matrix(log_returns, window=30)
    corr_penalties = correlation_penalty(corr_matrix)
    logger.info(f"Correlation penalties: {corr_penalties}")

    historical_bets = (
        df[["asset", "adj_model_prob", "adj_market_prob", "adj_outcome", "question", "side",
            "pricing_date", "sigma", "expiry", "S_at_pricing"]]
        .rename(columns={"adj_model_prob": "model_prob", "adj_market_prob": "market_prob",
                         "adj_outcome": "outcome", "question": "title",
                         "S_at_pricing": "current_price"})
        .to_dict("records")
    )
    for bet in historical_bets:
        bet["corr_penalty"] = corr_penalties.get(bet["asset"], 1.0)
        bet["expiry"] = str(pd.Timestamp(bet["expiry"]).date()) if pd.notna(bet["expiry"]) else ""

    pnl_df = backtest_sizing(
        historical_bets,
        bankroll=BANKROLL,
        kelly_fraction=KELLY_FRACTION,
        min_edge=MIN_EDGE,
        max_edge=MAX_EDGE,
        max_position=MAX_POSITION,
        max_payout=MAX_PAYOUT,
    ).reset_index(drop=True)

    # Merge results
    new_cols = [c for c in pnl_df.columns if c not in df.columns]
    df = pd.concat([df, pnl_df[new_cols]], axis=1)

    # ── 7. Summary statistics ──────────────────────────────────────────────────
    bets = df[df["stake_frac"] > 0]
    total_bets = len(bets)
    win_rate = float(bets["adj_outcome"].mean()) if total_bets else 0.0
    total_pnl = float(pnl_df["pnl"].sum())
    final_bank = BANKROLL + total_pnl
    roi = total_pnl / BANKROLL * 100
    avg_adj_edge = float((bets["adj_model_prob"] - bets["adj_market_prob"]).mean()) if total_bets else 0.0

    # Brier scores
    model_brier = float(np.mean((df["model_prob"].values - df["outcome"].values) ** 2))
    clob_mask = df["market_prob_source"] == "clob"
    market_brier = (
        float(np.mean((df.loc[clob_mask, "market_prob"].values - df.loc[clob_mask, "outcome"].values) ** 2))
        if clob_mask.any() else float("nan")
    )

    # ── 8. Output results ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"BACKTEST RESULTS ({len(df)} contracts, {total_bets} bets placed)")
    logger.info(f"  CLOB history    : {clob_hit}/{len(df)}")
    logger.info(f"  Win rate        : {win_rate:.1%}")
    logger.info(f"  Total P&L       : ${total_pnl:+,.2f}")
    logger.info(f"  Final bankroll  : ${final_bank:,.2f}  (ROI: {roi:+.1f}%)")
    logger.info(f"  Avg adj edge    : {avg_adj_edge:+.1%}  (placed bets, bet-side)")
    logger.info(f"  Model Brier     : {model_brier:.4f}")
    logger.info(f"  Market Brier    : {market_brier:.4f}  (CLOB only)")
    logger.info("=" * 60)

    print(f"\n{'='*60}")
    print(f"BACKTEST SUMMARY — {len(df)} contracts, {total_bets} bets placed")
    print(f"  CLOB history    : {clob_hit}/{len(df)}")
    print(f"  Win rate        : {win_rate:.1%}")
    print(f"  Total P&L       : ${total_pnl:+,.2f}")
    print(f"  Final bankroll  : ${final_bank:,.2f}  (ROI: {roi:+.1f}%)")
    print(f"  Avg adj edge    : {avg_adj_edge:+.1%}  (placed bets, bet-side)")
    print(f"  Model Brier     : {model_brier:.4f}")
    print(f"  Market Brier    : {market_brier:.4f}  (CLOB only)")
    print(f"{'='*60}\n")

    # Accuracy breakdown — placed bets only, with effective direction
    # call+YES = betting price goes UP; call+NO = betting price does NOT go up (effective put)
    # put+YES = betting price goes DOWN; put+NO = betting price does NOT go down (effective call)
    print("\nAccuracy by effective direction  (placed bets only, adj_outcome=1 means bet won):")
    if total_bets > 0:
        bets_acc = bets.copy()
        bets_acc["effective_direction"] = bets_acc.apply(
            lambda r: "up  (call YES / put NO)" if (
                (r["option_type"] == "call" and r["side"] == "YES") or
                (r["option_type"] == "put"  and r["side"] == "NO")
            ) else "down (call NO / put YES)",
            axis=1,
        )
        acc = (
            bets_acc.groupby(["effective_direction", "option_type", "side"])
            .agg(win_rate=("adj_outcome", "mean"), n=("adj_outcome", "size"))
            .reset_index()
        )
        acc["win_rate"] = acc["win_rate"].map("{:.1%}".format)
        print(acc.to_string(index=False))
    else:
        print("  (no bets placed)")

    # Show top bets
    if total_bets > 0:
        print("\nTop 10 Bets by Edge:")
        top_bets = bets.nlargest(10, "edge")[
            ["asset", "option_type", "strike", "side", "model_prob", "market_prob", "edge", "adj_outcome", "pnl"]
        ]
        print(top_bets.to_string(index=False))
    else:
        print("\nNo bets were placed. Try reducing MIN_EDGE or checking market data quality.")

    # Plot results
    if save_plots:
        _plot_results(df, pnl_df, clob_hit, save_plots=save_plots)
    
    return df


# ── Plotting ───────────────────────────────────────────────────────────────────

def _plot_results(df: pd.DataFrame, pnl_df: pd.DataFrame, clob_hit: int, save_plots: bool = SAVE_PLOTS) -> None:
    """Generate visualization plots."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D

    plt.style.use('dark_background')

    fig = plt.figure(figsize=(16, 15))
    fig.patch.set_facecolor("#0f0f1a")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.3, height_ratios=[2, 2, 1])

    colors = {"BTC": "#f7931a", "ETH": "#627eea", "SOL": "#9945ff", "good": "#00e5ff", "bad": "#ff6b6b"}

    # ── 1. Equity Curve ────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#0f0f1a")
    bankroll = BANKROLL + pnl_df["pnl"].cumsum().values
    dates = pd.to_datetime(df["pricing_date"])
    ax1.plot(dates, bankroll, color=colors["good"], lw=2, label="Bankroll")
    ax1.axhline(BANKROLL, color="#888", ls="--", lw=1, alpha=0.5, label="Starting")
    ax1.fill_between(dates, BANKROLL, bankroll,
                     where=bankroll >= BANKROLL, alpha=0.15, color=colors["good"])
    ax1.fill_between(dates, BANKROLL, bankroll,
                     where=bankroll < BANKROLL, alpha=0.15, color=colors["bad"])
    bet_mask = pnl_df["stake_frac"].values > 0
    if bet_mask.any():
        ax1.scatter(dates[bet_mask], bankroll[bet_mask],
                    color="white", s=50, zorder=5, label="Bets", alpha=0.85)
    final_bank = float(bankroll[-1]) if len(bankroll) else BANKROLL
    roi = (final_bank - BANKROLL) / BANKROLL * 100
    _weekly = (
        pd.DataFrame({"pricing_date": pd.to_datetime(df["pricing_date"]), "pnl": pnl_df["pnl"].values})
        .groupby(pd.Grouper(key="pricing_date", freq="W"))["pnl"]
        .sum()
    )
    _weekly_ret = _weekly / BANKROLL
    sharpe = (
        float((_weekly_ret.mean() / _weekly_ret.std()) * np.sqrt(52))
        if len(_weekly_ret) > 1 and _weekly_ret.std() > 0
        else 0.0
    )
    start_str = dates.min().strftime("%Y-%m-%d") if not dates.empty else ""
    end_str = dates.max().strftime("%Y-%m-%d") if not dates.empty else ""
    ax1.set_title(
        f"Equity Curve — ${final_bank:,.2f} final  |  ROI: {roi:+.1f}%  |  [{start_str} to {end_str}]",
        color="white", fontsize=12,
    )
    # Sharpe annotated on the line near the last point
    if len(bankroll):
        ax1.annotate(
            f"Sharpe: {sharpe:.2f}",
            xy=(dates.iloc[-1], bankroll[-1]),
            xytext=(-80, 12), textcoords="offset points",
            color=colors["good"], fontsize=11, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=colors["good"], alpha=0.4, lw=1),
        )
    ax1.set_xlabel("Date", color="white")
    ax1.set_ylabel("Bankroll ($)", color="white")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30, ha="right")
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#333")
    ax1.legend(facecolor="#0f0f1a", edgecolor="#333", labelcolor="white")

    # ── 2. Edge vs Stake scatter ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#0f0f1a")

    clob_df = df[df["market_prob_source"] == "clob"].copy()
    clob_df["adj_edge"] = clob_df["adj_model_prob"] - clob_df["adj_market_prob"]

    # Below-threshold: hypothetical Kelly stake with won/lost coloring, cyan outline
    below_thresh = clob_df[(clob_df["stake_frac"] == 0) & (clob_df["adj_edge"] <= MAX_EDGE)].copy()
    if not below_thresh.empty:
        def _hypo_stake_below(row):
            mp = row["adj_market_prob"]
            if mp <= 0 or mp >= 1:
                return 0.0
            b = (1.0 - mp) / mp
            f_raw = (row["adj_model_prob"] * b - (1.0 - row["adj_model_prob"])) / b
            return float(np.clip(f_raw * KELLY_FRACTION, 0.0, MAX_POSITION))
        below_thresh["hypo_stake"] = below_thresh.apply(_hypo_stake_below, axis=1)
        win_mask = below_thresh["adj_outcome"] == 1
        ax2.scatter(below_thresh.loc[win_mask, "adj_edge"], below_thresh.loc[win_mask, "hypo_stake"],
                    color="#44ff88", s=30, alpha=0.6, marker="^",
                    edgecolors="#00e5ff", linewidths=1.0, label="Won (<min edge)", zorder=2)
        ax2.scatter(below_thresh.loc[~win_mask, "adj_edge"], below_thresh.loc[~win_mask, "hypo_stake"],
                    color="#ff6b6b", s=30, alpha=0.6, marker="^",
                    edgecolors="#00e5ff", linewidths=1.0, label="Lost (<min edge)", zorder=2)

    # Above-max-threshold: compute hypothetical Kelly stake and show won/lost
    above_max = clob_df[(clob_df["stake_frac"] == 0) & (clob_df["adj_edge"] > MAX_EDGE)].copy()
    if not above_max.empty:
        def _hypo_stake(row):
            mp = row["adj_market_prob"]
            if mp <= 0 or mp >= 1:
                return 0.0
            b = (1.0 - mp) / mp
            f_raw = (row["adj_model_prob"] * b - (1.0 - row["adj_model_prob"])) / b
            return float(np.clip(f_raw * KELLY_FRACTION, 0.0, MAX_POSITION))
        above_max["hypo_stake"] = above_max.apply(_hypo_stake, axis=1)
        win_mask = above_max["adj_outcome"] == 1
        ax2.scatter(above_max.loc[win_mask, "adj_edge"], above_max.loc[win_mask, "hypo_stake"],
                    color="#44ff88", s=60, alpha=0.8, marker="D",
                    edgecolors="#ffaa00", linewidths=1.5, label="Won (>max edge)", zorder=4)
        ax2.scatter(above_max.loc[~win_mask, "adj_edge"], above_max.loc[~win_mask, "hypo_stake"],
                    color="#ff6b6b", s=60, alpha=0.8, marker="D",
                    edgecolors="#ffaa00", linewidths=1.5, label="Lost (>max edge)", zorder=4)

    bets = clob_df[clob_df["stake_frac"] > 0].copy()
    if not bets.empty:
        adj_edge = bets["adj_edge"]
        win_mask = bets["adj_outcome"] == 1
        ax2.scatter(adj_edge[win_mask], bets.loc[win_mask, "stake_frac"],
                    color="#44ff88", s=45, alpha=0.85, label="Won", zorder=3)
        ax2.scatter(adj_edge[~win_mask], bets.loc[~win_mask, "stake_frac"],
                    color="#ff6b6b", s=45, alpha=0.85, label="Lost", zorder=3)
        ax2.axvline(adj_edge.mean(), color="white", ls="--", lw=1.5, label=f"Mean: {adj_edge.mean():.1%}")
    else:
        ax2.text(0.5, 0.5, "No bets placed", color="white", ha="center", va="center", transform=ax2.transAxes)
    ax2.axvline(0, color="white", ls="-", lw=1, alpha=0.3)
    ax2.axvline(MIN_EDGE, color="#00e5ff", ls=":", lw=1.5, alpha=0.8, label=f"Min edge: {MIN_EDGE:.0%}")
    ax2.axvline(MAX_EDGE, color="#ffaa00", ls=":", lw=1.5, alpha=0.8, label=f"Max edge: {MAX_EDGE:.0%}")
    ax2.set_title("Edge vs Stake (all found bets)", color="white", fontsize=11)
    ax2.set_xlabel("Adjusted edge (model − market)", color="white")
    ax2.set_ylabel("Stake fraction", color="white")
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#333")
    ax2.legend(facecolor="#0f0f1a", edgecolor="#333", labelcolor="white", fontsize=9)

    # ── 3. Calibration scatter + binned curve (CLOB rows only) ────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#0f0f1a")
    ax3.plot([0, 1], [0, 1], "--", color="#555", lw=1.5, alpha=0.7)

    # Exclude model_fallback rows — their market_prob == model_prob so they add
    # no calibration signal and inflate apparent precision
    calib_df = df[df["market_prob_source"] == "clob"].copy()

    # Dynamic shape per option type
    all_markers = ["o", "^", "s", "D", "v", "P", "X", "*"]
    option_types = sorted(calib_df["option_type"].dropna().unique().tolist())
    marker_map = {ot: all_markers[i % len(all_markers)] for i, ot in enumerate(option_types)}

    # Color = stake size (plasma gradient)
    max_stake = max(calib_df["stake_frac"].max() if not calib_df.empty else 0, 1e-6)
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=0, vmax=max_stake)
    sc = None
    legend_handles = []
    for ot, mkr in marker_map.items():
        sub = calib_df[calib_df["option_type"] == ot]
        jitter = np.random.uniform(-0.025, 0.025, len(sub))
        sc = ax3.scatter(
            sub["adj_model_prob"].values,
            sub["adj_outcome"].values + jitter,
            c=sub["stake_frac"].values, cmap=cmap, norm=norm,
            marker=mkr, s=35, alpha=0.65, zorder=3,
        )
        legend_handles.append(
            Line2D([0], [0], marker=mkr, color="w", markerfacecolor="white",
                   markersize=7, label=ot, linestyle="None")
        )

    # Binned calibration curve
    bin_edges = np.linspace(0, 1, 11)
    bin_cx, bin_fy = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (calib_df["adj_model_prob"] >= lo) & (calib_df["adj_model_prob"] < hi)
        if mask.sum() >= 2:
            bin_cx.append((lo + hi) / 2)
            bin_fy.append(float(calib_df.loc[mask, "adj_outcome"].mean()))
    if bin_cx:
        ax3.plot(bin_cx, bin_fy, "w-o", lw=2, ms=6, zorder=10, label="Calibration")
        legend_handles.append(Line2D([0], [0], color="w", lw=2, label="Calibration curve"))

    if sc is not None:
        cb = plt.colorbar(sc, ax=ax3)
        cb.set_label("Stake size", color="white")
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    ax3.set_xlim(0, 1)
    ax3.set_ylim(-0.12, 1.12)
    n_calib = len(calib_df)
    ax3.set_title(f"Model Calibration — CLOB only  (n={n_calib}, shape = type, colour = stake)", color="white", fontsize=11)
    ax3.set_xlabel("Adjusted Model Probability (bet-side)", color="white")
    ax3.set_ylabel("Adjusted Outcome\n(1 = bet won, 0 = bet lost)", color="white")
    ax3.tick_params(colors="white")
    ax3.spines[:].set_color("#333")
    ax3.legend(handles=legend_handles, facecolor="#0f0f1a", edgecolor="#333", labelcolor="white", fontsize=8)

    # ── 4. Outcome × Bet-category table ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_facecolor("#0f0f1a")
    ax4.axis("off")

    def _categorize(row):
        if row["market_prob_source"] == "model_fallback":
            return "No market data"
        if row["stake_frac"] > 0:
            return f"Bet {row['side']}"
        raw_edge = row["edge"]
        if raw_edge > MAX_EDGE:
            return "Above threshold  (+)"
        if raw_edge < -MAX_EDGE:
            return "Above threshold  (−)"
        if raw_edge >= 0:
            return "Below threshold  (+)"
        return "Below threshold  (−)"

    df_t = df.copy()
    df_t["category"] = df_t.apply(_categorize, axis=1)
    df_t["outcome_label"] = df_t["outcome"].map({1: "YES resolved", 0: "NO resolved"})

    col_order = ["Bet YES", "Bet NO",
                 "Above threshold  (+edge)", "Above threshold  (−edge)",
                 "Below threshold  (+edge)", "Below threshold  (−edge)",
                 "No market data"]
    col_labels = [c for c in col_order if c in df_t["category"].values]
    row_labels = ["YES resolved", "NO resolved"]

    table_data = [
        [str(int(((df_t["outcome_label"] == rl) & (df_t["category"] == cl)).sum()))
         for cl in col_labels]
        for rl in row_labels
    ]

    tbl = ax4.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#444")
        if r == 0 or c == -1:
            cell.set_facecolor("#1e1e3a")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#0f0f1a" if r % 2 else "#161626")
            cell.set_text_props(color="white")
    ax4.set_title("Outcome × Bet Category", color="white", fontsize=11, pad=8)

    plt.suptitle(
        f"Polymarket Backtest — {len(df)} contracts  |  Kelly {KELLY_FRACTION:.0%}  |  "
        f"Min edge {MIN_EDGE:.0%}  |  Historical data: {clob_hit}/{len(df)}",
        color="white", fontsize=14, y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_plots:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results.png")
        plt.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
        logger.info(f"Plot saved to {path}")

    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    df = run_backtest(limit=limit)