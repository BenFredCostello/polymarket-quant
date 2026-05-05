"""
black_scholes.py
----------------
Closed-form Black-Scholes binary (digital) call option pricer.

A binary call pays $1 if S(T) > K at expiry, else $0.
BS formula: price = e^(-rT) * N(d2)
where d2 = [ln(S/K) + (r - 0.5*σ²)*T] / (σ*√T)

Output is a risk-neutral probability directly comparable to
Polymarket's implied probability.
"""

import numpy as np
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 365.0


class BlackScholesBinaryPricer:
    """
    Prices a binary call option under Black-Scholes assumptions.

    Parameters
    ----------
    S : float   Current spot price
    K : float   Strike price (threshold)
    T : float   Time to expiry in years
    sigma : float  Annualised volatility (e.g. 0.80 = 80%)
    r : float   Risk-free rate (default 0.0 for crypto)
    """

    def __init__(self, r: float = 0.0):
        self.r = r

    def _d2(self, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0:
            raise ValueError("Time to expiry T must be > 0.")
        if S <= 0 or K <= 0:
            raise ValueError("S and K must be positive.")
        if sigma <= 0:
            raise ValueError("Volatility sigma must be positive.")
        return (np.log(S / K) + (self.r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    def price(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Return the model probability / price of the binary call.

        Returns
        -------
        float in [0, 1] — risk-neutral probability that S(T) > K
        """
        d2 = self._d2(S, K, T, sigma)
        discount = np.exp(-self.r * T)
        p = discount * norm.cdf(d2)
        return float(np.clip(p, 0.0, 1.0))

    def delta(self, S: float, K: float, T: float, sigma: float) -> float:
        """Sensitivity of binary price to spot (dP/dS)."""
        d2 = self._d2(S, K, T, sigma)
        return float(np.exp(-self.r * T) * norm.pdf(d2) / (S * sigma * np.sqrt(T)))

    def vega(self, S: float, K: float, T: float, sigma: float) -> float:
        """Sensitivity of binary price to vol (dP/dσ)."""
        d2 = self._d2(S, K, T, sigma)
        d1 = d2 + sigma * np.sqrt(T)
        return float(-np.exp(-self.r * T) * norm.pdf(d2) * d1 / sigma)

    def implied_edge(
        self, S: float, K: float, T: float, sigma: float, market_prob: float
    ) -> float:
        """
        Edge = model probability − market (Polymarket) implied probability.
        Positive = model thinks YES is underpriced on Polymarket.
        """
        model_p = self.price(S, K, T, sigma)
        return model_p - market_prob


def days_to_years(days: float) -> float:
    return days / TRADING_DAYS_PER_YEAR


def validate_pricer() -> None:
    """
    Known-value tests for the BS binary pricer.
    ATM binary (S=K, r=0) → price ≈ 0.5 for all T, σ.
    Deep ITM (S >> K)     → price ≈ 1.0
    Deep OTM (S << K)     → price ≈ 0.0
    """
    pricer = BlackScholesBinaryPricer(r=0.0)
    tol = 1e-3

    # Test 1: ATM binary at very low vol should approach 0.5
    # Note: at high vol, ATM binary < 0.5 because d2 has negative drift correction (-0.5σ²T)
    p_atm_lowvol = pricer.price(S=100, K=100, T=1.0, sigma=0.01)
    assert abs(p_atm_lowvol - 0.5) < 0.02, f"ATM low-vol test failed: {p_atm_lowvol:.4f}"
    # At high vol ATM, d2 = -0.5*σ*sqrt(T) < 0, so price < 0.5 — this is correct
    p_atm_highvol = pricer.price(S=100, K=100, T=1.0, sigma=0.80)
    assert 0.25 < p_atm_highvol < 0.50, f"ATM high-vol test failed: {p_atm_highvol:.4f}"

    # Test 2: Deep ITM — S=200, K=100, short T
    p_itm = pricer.price(S=200, K=100, T=0.01, sigma=0.80)
    assert p_itm > 0.95, f"Deep ITM test failed: {p_itm:.4f}"

    # Test 3: Deep OTM — S=50, K=100, short T
    p_otm = pricer.price(S=50, K=100, T=0.01, sigma=0.80)
    assert p_otm < 0.05, f"Deep OTM test failed: {p_otm:.4f}"

    # Test 4: Price in [0,1]
    for s in [50, 100, 150, 200]:
        p = pricer.price(S=s, K=100, T=0.25, sigma=1.0)
        assert 0 <= p <= 1, f"Price out of bounds: {p}"

    # Test 5: Monotone in S
    prices = [pricer.price(S=s, K=100, T=0.5, sigma=0.8) for s in range(50, 200, 10)]
    assert all(prices[i] < prices[i+1] for i in range(len(prices)-1)), \
        "Price should be monotone increasing in S"

    # Test 6: Known value — BS binary with r=0, ATM, σ=0.5, T=1y
    # d2 = (ln(1) + (0 - 0.5*0.25)*1) / (0.5*1) = -0.125/0.5 = -0.25 → N(-0.25) ≈ 0.4013
    p_known = pricer.price(S=100, K=100, T=1.0, sigma=0.5)
    assert abs(p_known - norm.cdf(-0.25)) < tol, f"Known value test failed: {p_known:.4f}"

    logger.info("✅ All Black-Scholes validation tests passed.")
    print("✅ All BS pricer validation tests passed.")


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    validate_pricer()

    pricer = BlackScholesBinaryPricer(r=0.0)
    print("\n── BTC example ──")
    S = 65_000     # current BTC price
    K = 70_000     # market asks: will BTC be above $70k?
    T = days_to_years(30)
    sigma = 0.75   # 75% annualised vol

    model_prob = pricer.price(S, K, T, sigma)
    market_prob = 0.38  # hypothetical Polymarket price

    print(f"  S={S:,}  K={K:,}  T={T:.3f}y  σ={sigma:.0%}")
    print(f"  Model probability : {model_prob:.2%}")
    print(f"  Market probability: {market_prob:.2%}")
    print(f"  Edge              : {pricer.implied_edge(S, K, T, sigma, market_prob):+.2%}")
    print(f"  Delta             : {pricer.delta(S, K, T, sigma):.6f}")
    print(f"  Vega              : {pricer.vega(S, K, T, sigma):.6f}")
