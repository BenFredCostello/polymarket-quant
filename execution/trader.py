"""
trader.py
---------
Live market scanner: connects Polymarket API, runs pricer on each contract,
flags opportunities above edge threshold, outputs Kelly-sized stakes.
Paper trading mode by default.
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone, date
from dataclasses import dataclass, field

# Ensure the project root (parent of this file's directory) is on sys.path
# so that `from data.x import ...`, `from pricing.x import ...` etc. all resolve
# regardless of where Python is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sizing.kelly import DEFAULT_MIN_EDGE, DEFAULT_KELLY_FRACTION, DEFAULT_MAX_POSITION, DEFAULT_MAX_PAYOUT
try:
    from config import MIN_STAKE_USD
except ImportError:
    MIN_STAKE_USD = 1.0

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_BANKROLL = 1_000.0
PAPER_TRADE             = True        # set False only when real API key is configured
SCAN_INTERVAL_SECS      = 300         # 5 minutes between scans

# Volume filters — applied to PolymarketContract fields
MIN_VOLUME_24H          = 500         # minimum 24-hour volume (USDC) to consider
MIN_VOLUME_TOTAL        = 2_000       # minimum total lifetime volume (USDC) to consider
# Contracts with total volume >= this are treated as "established" and skip the 24h floor
ESTABLISHED_VOLUME      = 50_000

MAX_T_YEARS             = 180 / 365  # reject markets expiring more than 6 months out


@dataclass
class TradeSignal:
    condition_id: str
    question: str
    asset: str
    strike: float | None
    end_date: str | None
    market_prob: float
    model_prob_bs: float
    model_prob_mc: float
    edge_bs: float
    edge_mc: float
    recommended_stake_frac: float
    recommended_stake_usd: float
    corr_penalty: float
    volume_total: float
    volume_24h: float
    option_type: str = "call"   # "call" or "put" — from fetch_polymarket
    side: str = "YES"           # "YES" or "NO" — which token to buy
    yes_token: str = ""         # CLOB YES token ID for resolution checking
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    action: str = "SKIP"    # BET or SKIP


class LiveTrader:
    """
    Orchestrates the full pipeline:
      1. Fetch open markets (via new async fetch_polymarket)
      2. Filter by volume, validity, and expiry
      3. Fetch crypto prices + vol
      4. Price each contract (BS + MC)
      5. Apply Kelly sizing with correlation adjustment
      6. Log / paper trade
    """

    def __init__(
        self,
        bankroll: float = DEFAULT_BANKROLL,
        min_edge: float = DEFAULT_MIN_EDGE,
        kelly_fraction: float = DEFAULT_KELLY_FRACTION,
        max_position: float = DEFAULT_MAX_POSITION,
        max_payout: float = DEFAULT_MAX_PAYOUT,
        paper_trade: bool = PAPER_TRADE,
        min_volume_24h: float = MIN_VOLUME_24H,
        min_volume_total: float = MIN_VOLUME_TOTAL,
    ):
        self.bankroll = bankroll
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
        self.max_position = max_position
        self.max_payout = max_payout
        self.paper_trade = paper_trade
        self.min_volume_24h = min_volume_24h
        self.min_volume_total = min_volume_total
        self.trade_log: list[TradeSignal] = []

        self._init_components()

    def _init_components(self) -> None:
        from data.fetch_crypto import build_crypto_dataset, get_current_dvol, DVOL_ASSETS
        from data.fetch_polymarket import fetch_open_markets, _resolve_expiry
        from pricing.black_scholes import BlackScholesBinaryPricer
        from pricing.monte_carlo import MonteCarloBinaryPricer, calibrate_jump_params
        from analysis.correlation import get_current_corr_matrix, correlation_penalty
        from sizing.kelly import size_bet, size_portfolio

        from data.fetch_polymarket import _parse_yes_token
        self._fetch_crypto = build_crypto_dataset
        self._fetch_markets = fetch_open_markets
        self._resolve_expiry = _resolve_expiry
        self._parse_yes_token = _parse_yes_token
        self._bs = BlackScholesBinaryPricer(r=0.0)
        self._mc = MonteCarloBinaryPricer(n_paths=50_000, use_antithetic=True)
        self._calibrate_jump_params = calibrate_jump_params
        self._jump_params: dict = {}
        self._get_corr_matrix = get_current_corr_matrix
        self._corr_penalty = correlation_penalty
        self._size_bet = size_bet
        self._size_portfolio = size_portfolio
        self._get_current_dvol = get_current_dvol
        self._dvol_assets = DVOL_ASSETS

        logger.info("✅ All components initialised.")

    def _volume_filter(self, market) -> bool:
        """
        Accept a contract if it meets volume thresholds.

        Logic:
          - Established markets (total vol >= ESTABLISHED_VOLUME): skip 24h floor —
            they may have been live for weeks and still be genuinely liquid.
          - All others: must have both sufficient 24h AND total volume.
        """
        vol_total = market.volume
        vol_24h   = market.volume24hr

        if vol_total >= ESTABLISHED_VOLUME:
            # Already liquid enough historically; 24h floor not required
            return True

        # New / smaller markets: require active recent trading
        return vol_24h >= self.min_volume_24h and vol_total >= self.min_volume_total

    def _time_to_expiry(self, market) -> float | None:
        """
        Use fetch_polymarket's _resolve_expiry to get the expiry date, then convert
        to years until expiry. Returns None if expired, unresolvable, or beyond MAX_T.
        """
        expiry: date | None = self._resolve_expiry(market)
        if expiry is None:
            return None

        today = datetime.now(timezone.utc).date()
        days_remaining = (expiry - today).days

        if days_remaining < 1:
            return None   # already expired or resolves today

        T = days_remaining / 365.0
        if T > MAX_T_YEARS:
            return None   # too far out to price reliably

        return T

    def scan(self) -> list[TradeSignal]:
        """Run one full scan cycle. Returns list of trade signals."""
        logger.info("=" * 60)
        logger.info(f"Starting scan at {datetime.now().isoformat()}")

        # ── 1. Fetch crypto price data ─────────────────────────────────
        try:
            crypto_data = self._fetch_crypto(lookback_days=120)
        except Exception as e:
            logger.error(f"Crypto data fetch failed: {e}")
            return []

        # ── 2. Fetch + filter Polymarket contracts ─────────────────────
        try:
            raw_markets = self._fetch_markets(limit=500)
        except Exception as e:
            logger.error(f"Polymarket fetch failed: {e}")
            return []

        # Apply is_valid (built into PolymarketContract) then volume filter
        valid_markets = [m for m in raw_markets if m.is_valid]
        liquid_markets = [m for m in valid_markets if self._volume_filter(m)]

        logger.info(
            f"Markets: {len(raw_markets)} raw → {len(valid_markets)} valid → "
            f"{len(liquid_markets)} liquid (vol24h≥${self.min_volume_24h:,.0f} "
            f"or total≥${ESTABLISHED_VOLUME:,.0f})"
        )

        # ── 3. Correlation matrix ──────────────────────────────────────
        log_returns = {
            asset: df["log_return"].dropna()
            for asset, df in crypto_data.items()
            if "log_return" in df.columns
        }

        # ── 3a. Calibrate jump params per asset from recent log returns ───
        for asset, ret_series in log_returns.items():
            try:
                self._jump_params[asset] = self._calibrate_jump_params(ret_series.values)
            except Exception as e:
                logger.warning(f"Jump param calibration failed for {asset}: {e}. Using default.")
                from pricing.monte_carlo import DEFAULT_JUMP_PARAMS
                self._jump_params[asset] = DEFAULT_JUMP_PARAMS

        try:
            corr_matrix = self._get_corr_matrix(log_returns, window=30)
            penalties = self._corr_penalty(corr_matrix)
        except Exception as e:
            logger.warning(f"Correlation calc failed: {e}. Using no penalty.")
            penalties = {}

        # ── 3b. Fetch Deribit DVOL for BTC/ETH ────────────────────────
        dvol_sigmas: dict[str, float] = {}
        for asset in self._dvol_assets:
            dvol = self._get_current_dvol(asset)
            if dvol is not None:
                dvol_sigmas[asset] = dvol
                logger.info(f"  Deribit DVOL {asset}: {dvol:.1%}")

        # ── 4. Price each market ───────────────────────────────────────
        signals: list[TradeSignal] = []

        for market in liquid_markets:
            asset_df = crypto_data.get(market.asset)
            if asset_df is None or asset_df.empty:
                logger.debug(f"No crypto data for asset '{market.asset}' — skipping.")
                continue

            S      = float(asset_df["Close"].iloc[-1])
            K      = market.strike
            T      = self._time_to_expiry(market)
            hist_sigma = float(asset_df["realised_vol_30d"].dropna().iloc[-1])
            sigma  = dvol_sigmas.get(market.asset, hist_sigma)
            market_prob = market.implied_prob   # yes_price from PolymarketContract

            if K is None or T is None or K <= 0 or S <= 0 or sigma <= 0:
                logger.debug(
                    f"Skipping {market.condition_id}: incomplete pricing inputs "
                    f"(K={K}, T={T}, S={S:.0f}, σ={sigma:.3f})."
                )
                continue

            # Direction comes from fetch_polymarket: "put" = dip/below, "call" = above/reach
            is_put = market.option_type == "put"

            try:
                bs_call = self._bs.price(S, K, T, sigma)
                jp      = self._jump_params.get(market.asset)
                mc_out  = self._mc.price(S, K, T, sigma, jp) if jp else self._mc.price(S, K, T, sigma)
                mc_call = mc_out["price"]
            except Exception as e:
                logger.warning(f"Pricing error for {market.condition_id}: {e}")
                continue

            # Flip to put probability if the question is about downside
            bs_p = (1.0 - bs_call) if is_put else bs_call
            mc_p = (1.0 - mc_call) if is_put else mc_call

            model_prob  = float(np.clip((bs_p + mc_p) / 2, 1e-6, 1 - 1e-6))
            market_prob = float(np.clip(market_prob, 1e-6, 1 - 1e-6))
            edge        = model_prob - market_prob
            corr_pen    = penalties.get(market.asset, 1.0)

            # If edge is strongly negative, the market is overpricing YES —
            # flip and bet NO instead. From Kelly's perspective, buying NO at
            # price (1 - market_prob) with model probability (1 - model_prob) is
            # equivalent to a positive-edge YES bet on the complement.
            if edge < -self.min_edge:
                bet_side       = "NO"
                adj_model_prob = 1.0 - model_prob   # model prob that NO resolves
                adj_market_prob = 1.0 - market_prob  # cost of buying NO token
            else:
                bet_side       = "YES"
                adj_model_prob = model_prob
                adj_market_prob = market_prob

            sizing = self._size_bet(
                asset=market.asset,
                model_prob=adj_model_prob,
                market_prob=adj_market_prob,
                corr_penalty=corr_pen,
                kelly_fraction=self.kelly_fraction,
                min_edge=self.min_edge,
                max_position=self.max_position,
                max_payout=self.max_payout,
            )

            stake_usd = sizing.final_stake * self.bankroll
            if sizing.should_bet:
                stake_usd = max(stake_usd, MIN_STAKE_USD)

            signal = TradeSignal(
                condition_id=market.condition_id,
                question=market.question[:80],
                asset=market.asset,
                strike=K,
                end_date=market.end_date,
                market_prob=market_prob,
                model_prob_bs=bs_p,
                model_prob_mc=mc_p,
                edge_bs=bs_p - market_prob,
                edge_mc=mc_p - market_prob,
                recommended_stake_frac=sizing.final_stake,
                recommended_stake_usd=stake_usd,
                corr_penalty=corr_pen,
                volume_total=market.volume,
                volume_24h=market.volume24hr,
                option_type=market.option_type,
                side=bet_side,
                yes_token=self._parse_yes_token(market.token_ids) or "",
                action="BET" if sizing.should_bet else "SKIP",
            )
            signals.append(signal)

        # ── 5. Log and place bets ──────────────────────────────────────
        bets = [s for s in signals if s.action == "BET"]
        logger.info(
            f"Found {len(bets)} bet opportunities from {len(signals)} priced markets."
        )

        for sig in bets:
            self._execute(sig)

        self.trade_log.extend(signals)
        return signals

    def _execute(self, signal: TradeSignal) -> None:
        """Place bet or log as paper trade."""
        if self.paper_trade:
            logger.info(
                f"[PAPER] BET: {signal.asset}  |  "
                f"Q: {signal.question[:50]}  |  "
                f"Side: {signal.side}  |  "
                f"Edge: {signal.edge_bs:+.1%} (BS) / {signal.edge_mc:+.1%} (MC)  |  "
                f"Stake: ${signal.recommended_stake_usd:.2f}  |  "
                f"Vol24h: ${signal.volume_24h:,.0f}  Total: ${signal.volume_total:,.0f}"
            )
            return

        # Real execution via py_clob_client
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import OrderArgs, OrderType

            api_key = os.environ.get("POLYMARKET_API_KEY")
            if not api_key:
                raise EnvironmentError("POLYMARKET_API_KEY not set.")

            client = ClobClient(
                host="https://clob.polymarket.com",
                key=api_key,
                chain_id=137,
            )

            order = client.create_and_post_order(OrderArgs(
                token_id=signal.condition_id,
                price=signal.market_prob,
                size=signal.recommended_stake_usd,
                side="BUY",
                order_type=OrderType.GTC,
            ))
            logger.info(f"✅ Order placed: {order}")

        except ImportError:
            logger.error("py_clob_client not installed. Run: pip install py-clob-client")
        except Exception as e:
            logger.error(f"Order placement failed for {signal.condition_id}: {e}")

    def run_loop(self, n_scans: int = None) -> None:
        """Run continuous scanning loop."""
        logger.info(
            f"Starting live trader | paper={self.paper_trade} | bankroll=${self.bankroll:,.0f}"
        )
        count = 0
        while True:
            signals = self.scan()
            self._print_summary(signals)
            count += 1
            if n_scans is not None and count >= n_scans:
                break
            logger.info(f"Next scan in {SCAN_INTERVAL_SECS}s...")
            time.sleep(SCAN_INTERVAL_SECS)

    def _print_summary(self, signals: list[TradeSignal]) -> None:
        if not signals:
            print("No signals this scan.")
            return

        df = pd.DataFrame([{
            "Asset":        s.asset,
            "Action":       s.action,
            "Market P":     f"{s.market_prob:.1%}",
            "BS P":         f"{s.model_prob_bs:.1%}",
            "MC P":         f"{s.model_prob_mc:.1%}",
            "Edge":         f"{s.edge_bs:+.1%}",
            "Stake USD":    f"${s.recommended_stake_usd:.2f}",
            "Corr Mult":    f"{s.corr_penalty:.2f}",
            "Vol 24h":      f"${s.volume_24h:,.0f}",
            "Vol Total":    f"${s.volume_total:,.0f}",
            "Type":         s.option_type.upper(),
            "Side":         s.side,
            "Q":            s.question[:40] + "…",
        } for s in signals])

        print("\n" + "=" * 120)
        print(f"SCAN RESULTS — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 120)

        bets_df = df[df["Action"] == "BET"]
        if bets_df.empty:
            print("  No bets this scan.")
        else:
            print(bets_df.to_string(index=False))

        skipped = (df["Action"] == "SKIP").sum()
        if skipped:
            print(f"\n  ({skipped} markets priced but skipped — insufficient edge)")
        print("=" * 120 + "\n")

    def get_trade_log(self) -> pd.DataFrame:
        """Return all signals logged so far as a DataFrame."""
        return pd.DataFrame([s.__dict__ for s in self.trade_log])

    @staticmethod
    def validate_trader_components() -> None:
        """Smoke test that all imports resolve correctly."""
        from data.fetch_crypto import build_crypto_dataset
        from data.fetch_polymarket import fetch_open_markets, _resolve_expiry
        from pricing.black_scholes import BlackScholesBinaryPricer, validate_pricer
        from pricing.monte_carlo import MonteCarloBinaryPricer, validate_mc_pricer
        from analysis.calibration import validate_calibration
        from analysis.correlation import validate_correlation
        from sizing.kelly import validate_kelly

        validate_pricer()
        validate_mc_pricer()
        validate_calibration()
        validate_correlation()
        validate_kelly()

        logger.info("✅ All component validations passed.")
        print("\n✅ ALL COMPONENT VALIDATIONS PASSED — system is ready.")


if __name__ == "__main__":
    import sys

    if "--validate" in sys.argv:
        LiveTrader.validate_trader_components()
    elif "--paper" in sys.argv:
        trader = LiveTrader(bankroll=1000.0, paper_trade=True)
        trader.run_loop(n_scans=1)
    else:
        print("Usage:")
        print("  python trader.py --validate   # run all component checks")
        print("  python trader.py --paper      # one paper-trade scan")