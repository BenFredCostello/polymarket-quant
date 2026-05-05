"""
monte_carlo.py
--------------
Merton Jump Diffusion Monte Carlo pricer for binary call options.

Model: dS = (μ - λ*κ)*S*dt + σ*S*dW + J*S*dN
where:
  - GBM component: σ*dW
  - Poisson jump component: dN ~ Poisson(λ), J ~ LogNormal(μ_j, σ_j)
  - κ = E[J-1] = exp(μ_j + 0.5*σ_j²) - 1

Binary call price = e^(-rT) * E[1_{S(T) > K}]
"""

import sys
import os
import numpy as np
import logging
from dataclasses import dataclass

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from config import JUMP_LAM, JUMP_MU_J, JUMP_SIGMA_J

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 365.0


@dataclass
class JumpParams:
    """Merton Jump Diffusion parameters."""
    lam: float    # Jump intensity (jumps per year, e.g. 5-10 for crypto)
    mu_j: float   # Mean log jump size (e.g. -0.05 = -5% avg jump)
    sigma_j: float  # Std of log jump size (e.g. 0.15)

    def validate(self) -> None:
        assert self.lam >= 0, f"Jump intensity λ must be ≥ 0, got {self.lam}"
        assert self.sigma_j > 0, f"Jump vol σ_j must be > 0, got {self.sigma_j}"

    @property
    def kappa(self) -> float:
        """E[J-1] = compensator to keep process martingale under risk-neutral measure."""
        return np.exp(self.mu_j + 0.5 * self.sigma_j ** 2) - 1


# Default calibrated parameters for crypto (conservative estimate)
DEFAULT_JUMP_PARAMS = JumpParams(lam=JUMP_LAM, mu_j=JUMP_MU_J, sigma_j=JUMP_SIGMA_J)


class MonteCarloBinaryPricer:
    """
    Prices binary call options under Merton Jump Diffusion via Monte Carlo.

    Parameters
    ----------
    n_paths : int    Number of simulation paths (default 100_000)
    n_steps : int    Time steps per path (default 252 — daily)
    seed    : int    RNG seed for reproducibility
    r       : float  Risk-free rate
    use_antithetic : bool  Use antithetic variates for variance reduction
    """

    def __init__(
        self,
        n_paths: int = 100_000,
        n_steps: int = 252,
        seed: int = 42,
        r: float = 0.0,
        use_antithetic: bool = True,
    ):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed
        self.r = r
        self.use_antithetic = use_antithetic

    def _simulate_paths(
        self,
        S0: float,
        T: float,
        sigma: float,
        jp: JumpParams,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Simulate terminal asset prices S(T) under jump diffusion.

        Returns 1-D array of shape (n_paths,) with terminal prices.
        """
        jp.validate()
        dt = T / self.n_steps
        n = self.n_paths // 2 if self.use_antithetic else self.n_paths

        # ── GBM component ──────────────────────────────────────────────────
        # Drift adjustment: (r - λκ - 0.5σ²)*dt
        drift = (self.r - jp.lam * jp.kappa - 0.5 * sigma ** 2) * dt
        diffusion_scale = sigma * np.sqrt(dt)

        Z = rng.standard_normal((n, self.n_steps))
        if self.use_antithetic:
            Z = np.concatenate([Z, -Z], axis=0)  # antithetic pairs

        log_returns_gbm = drift + diffusion_scale * Z  # (n_paths, n_steps)

        # ── Jump component ──────────────────────────────────────────────────
        # Number of jumps in each dt interval ~ Poisson(λ*dt)
        N_jumps = rng.poisson(jp.lam * dt, size=(self.n_paths, self.n_steps))

        # Each jump magnitude: log(J) ~ Normal(μ_j, σ_j)
        # Sum of N_jumps iid normal rvs = N_jumps * μ_j + sqrt(N_jumps) * σ_j * Z_j
        Z_j = rng.standard_normal((self.n_paths, self.n_steps))
        log_jump_contrib = N_jumps * jp.mu_j + np.sqrt(N_jumps) * jp.sigma_j * Z_j

        # ── Terminal price ──────────────────────────────────────────────────
        total_log_ret = (log_returns_gbm + log_jump_contrib).sum(axis=1)
        S_T = S0 * np.exp(total_log_ret)
        return S_T

    def price(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        jp: JumpParams = DEFAULT_JUMP_PARAMS,
    ) -> dict:
        """
        Price a binary call option.

        Returns
        -------
        dict with:
            price      : float  — MC estimated probability / price
            std_error  : float  — standard error of estimate
            conf_interval : tuple  — 95% CI
        """
        if T <= 0:
            raise ValueError("T must be > 0")
        if S <= 0 or K <= 0:
            raise ValueError("S and K must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        rng = np.random.default_rng(self.seed)
        S_T = self._simulate_paths(S, T, sigma, jp, rng)

        payoffs = (S_T > K).astype(float)
        discount = np.exp(-self.r * T)

        price = discount * payoffs.mean()
        std_err = discount * payoffs.std() / np.sqrt(len(payoffs))
        ci = (price - 1.96 * std_err, price + 1.96 * std_err)

        return {
            "price": float(np.clip(price, 0, 1)),
            "std_error": float(std_err),
            "conf_interval": (float(np.clip(ci[0], 0, 1)), float(np.clip(ci[1], 0, 1))),
            "n_paths": self.n_paths,
        }

    def compare_to_bs(
        self, S: float, K: float, T: float, sigma: float, bs_price: float,
        jp: JumpParams = DEFAULT_JUMP_PARAMS,
    ) -> dict:
        """
        Compare MC jump diffusion price to BS price.
        The difference is the 'fat tail premium'.
        """
        mc_result = self.price(S, K, T, sigma, jp)
        fat_tail_premium = mc_result["price"] - bs_price
        return {
            **mc_result,
            "bs_price": bs_price,
            "fat_tail_premium": fat_tail_premium,
        }


def calibrate_jump_params(log_returns: np.ndarray) -> JumpParams:
    """
    Naive calibration of jump parameters from historical log return series.

    Strategy:
      - Identify 'jumps' as returns beyond 2.5 sigma from mean
      - Estimate λ from jump frequency
      - Estimate μ_j, σ_j from the distribution of identified jump returns
    """
    assert len(log_returns) > 30, "Need at least 30 observations to calibrate."

    mu = log_returns.mean()
    sigma = log_returns.std()
    threshold = 2.5 * sigma

    jump_mask = np.abs(log_returns - mu) > threshold
    n_jumps = jump_mask.sum()
    n_obs = len(log_returns)

    lam = n_jumps / (n_obs / TRADING_DAYS_PER_YEAR)  # jumps per year
    lam = max(lam, 0.1)  # floor

    if n_jumps >= 5:
        jump_returns = log_returns[jump_mask]
        mu_j = float(jump_returns.mean())
        sigma_j = float(jump_returns.std())
    else:
        logger.warning("Too few jumps detected; using conservative defaults.")
        mu_j = -0.04
        sigma_j = 0.12

    sigma_j = max(sigma_j, 0.05)  # floor
    params = JumpParams(lam=float(lam), mu_j=mu_j, sigma_j=sigma_j)
    logger.info(f"Calibrated jump params: λ={lam:.2f}/yr  μ_j={mu_j:.3f}  σ_j={sigma_j:.3f}")
    return params


def validate_mc_pricer() -> None:
    """Validation tests for the MC pricer."""
    pricer = MonteCarloBinaryPricer(n_paths=200_000, seed=0)

    # Test 1: Deep ITM should approach 1
    result = pricer.price(S=200, K=100, T=0.01, sigma=0.8)
    assert result["price"] > 0.95, f"Deep ITM failed: {result['price']:.4f}"

    # Test 2: Deep OTM should approach 0
    result = pricer.price(S=50, K=100, T=0.01, sigma=0.8)
    assert result["price"] < 0.05, f"Deep OTM failed: {result['price']:.4f}"

    # Test 3: ATM at low vol ~ 0.5; at high vol < 0.5 (correct — same as BS)
    result_low = pricer.price(S=100, K=100, T=1.0, sigma=0.01)
    assert 0.40 < result_low["price"] < 0.60, f"ATM low-vol failed: {result_low['price']:.4f}"
    result_high = pricer.price(S=100, K=100, T=1.0, sigma=0.8)
    assert 0.20 < result_high["price"] < 0.55, f"ATM high-vol range failed: {result_high['price']:.4f}"

    # Test 4: Price in [0,1]
    for K in [80, 100, 120]:
        r = pricer.price(S=100, K=K, T=0.5, sigma=0.9)
        assert 0 <= r["price"] <= 1, f"Out of bounds: {r['price']}"

    # Test 5: MC converges toward BS for no-jump case (λ=0)
    from pricing.black_scholes import BlackScholesBinaryPricer
    bs = BlackScholesBinaryPricer(r=0.0)
    bs_p = bs.price(S=100, K=110, T=0.5, sigma=0.6)
    no_jump = JumpParams(lam=0.0, mu_j=0.0, sigma_j=0.01)
    mc_result = pricer.price(S=100, K=110, T=0.5, sigma=0.6, jp=no_jump)
    diff = abs(mc_result["price"] - bs_p)
    assert diff < 0.03, f"MC should converge to BS (no jumps): |{mc_result['price']:.4f} - {bs_p:.4f}| = {diff:.4f}"

    logger.info("✅ All MC pricer validation tests passed.")
    print("✅ All MC pricer validation tests passed.")


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    validate_mc_pricer()

    from pricing.black_scholes import BlackScholesBinaryPricer, days_to_years
    pricer = MonteCarloBinaryPricer(n_paths=100_000, use_antithetic=True)
    bs = BlackScholesBinaryPricer()

    S, K, T, sigma = 65_000, 70_000, days_to_years(30), 0.75
    jp = DEFAULT_JUMP_PARAMS

    bs_p = bs.price(S, K, T, sigma)
    mc_out = pricer.compare_to_bs(S, K, T, sigma, bs_price=bs_p, jp=jp)

    print(f"\nBTC binary call: S={S:,}  K={K:,}  T={T:.3f}y  σ={sigma:.0%}")
    print(f"  BS price          : {bs_p:.4f}")
    print(f"  MC (jump diff)    : {mc_out['price']:.4f}  ±{mc_out['std_error']:.4f}")
    print(f"  95% CI            : [{mc_out['conf_interval'][0]:.4f}, {mc_out['conf_interval'][1]:.4f}]")
    print(f"  Fat tail premium  : {mc_out['fat_tail_premium']:+.4f}")
    print(f"  Jump params       : λ={jp.lam}  μ_j={jp.mu_j}  σ_j={jp.sigma_j}")
