"""
correlation.py
--------------
Rolling pairwise correlation analysis between crypto assets.
Detects regime changes, altcoin season decoupling, and crash-correlation spikes.
Outputs correlation matrix for Kelly sizing input.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
from itertools import combinations

logger = logging.getLogger(__name__)

CORR_WINDOWS = [30, 90]
HIGH_CORR_THRESHOLD = 0.80    # flag when correlation exceeds this
CRASH_CORR_THRESHOLD = 0.95   # flag potential crash regime


def compute_pairwise_rolling_corr(
    log_returns: dict[str, pd.Series],
    window: int = 30,
) -> pd.DataFrame:
    """
    Compute rolling pairwise correlations.

    Parameters
    ----------
    log_returns : dict mapping asset name → daily log return Series
    window      : rolling window in days

    Returns
    -------
    DataFrame with columns like 'BTC_ETH_corr_30d', indexed by date
    """
    assets = list(log_returns.keys())
    assert len(assets) >= 2, "Need at least 2 assets for correlation."

    # Align all series on common dates
    df = pd.DataFrame(log_returns).dropna()

    corr_df = pd.DataFrame(index=df.index)
    for a1, a2 in combinations(assets, 2):
        col = f"{a1}_{a2}_corr_{window}d"
        corr_df[col] = df[a1].rolling(window).corr(df[a2])

    logger.info(f"Computed {len(corr_df.columns)} rolling correlation pairs (window={window}d).")
    return corr_df


def compute_all_correlations(
    log_returns: dict[str, pd.Series],
    windows: list[int] = CORR_WINDOWS,
) -> pd.DataFrame:
    """Compute rolling correlations for all specified windows."""
    frames = [compute_pairwise_rolling_corr(log_returns, w) for w in windows]
    return pd.concat(frames, axis=1)


def get_current_corr_matrix(
    log_returns: dict[str, pd.Series],
    window: int = 30,
) -> pd.DataFrame:
    """
    Return the most recent rolling correlation matrix.
    This is the input to Kelly sizing.
    """
    assets = list(log_returns.keys())
    df = pd.DataFrame(log_returns).dropna()
    recent = df.tail(window)

    corr_matrix = recent.corr()
    _validate_corr_matrix(corr_matrix)
    return corr_matrix


def detect_regime_changes(
    corr_df: pd.DataFrame,
    high_threshold: float = HIGH_CORR_THRESHOLD,
    crash_threshold: float = CRASH_CORR_THRESHOLD,
) -> pd.DataFrame:
    """
    Flag dates where correlations enter high or crash regimes.

    Returns DataFrame with columns: date, pair, correlation, regime
    """
    records = []
    for col in corr_df.columns:
        series = corr_df[col].dropna()
        for date, val in series.items():
            if val >= crash_threshold:
                regime = "CRASH"
            elif val >= high_threshold:
                regime = "HIGH"
            else:
                continue
            records.append({"date": date, "pair": col, "correlation": val, "regime": regime})

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("date")
    logger.info(f"Detected {len(df)} high-correlation regime events.")
    return df


def detect_altcoin_decoupling(
    log_returns: dict[str, pd.Series],
    btc_key: str = "BTC",
    window: int = 30,
    decouple_threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Detect altcoin season: when BTC correlation with alts drops below threshold.
    """
    if btc_key not in log_returns:
        logger.warning(f"{btc_key} not in log_returns; skipping decoupling detection.")
        return pd.DataFrame()

    df = pd.DataFrame(log_returns).dropna()
    records = []
    for asset in log_returns:
        if asset == btc_key:
            continue
        rolling_corr = df[btc_key].rolling(window).corr(df[asset])
        decouple = rolling_corr[rolling_corr < decouple_threshold]
        for date, val in decouple.items():
            records.append({"date": date, "altcoin": asset, f"btc_corr_{window}d": val})

    return pd.DataFrame(records)


def correlation_penalty(
    corr_matrix: pd.DataFrame,
    threshold: float = HIGH_CORR_THRESHOLD,
) -> dict[str, float]:
    """
    Compute per-asset correlation penalty factors in [0, 1].
    Assets with high average cross-correlation receive smaller multipliers.

    Returns
    -------
    dict asset → multiplier (1.0 = no penalty, 0.0 = fully penalised)
    """
    assets = corr_matrix.index.tolist()
    penalties = {}
    for asset in assets:
        others = [a for a in assets if a != asset]
        if not others:
            penalties[asset] = 1.0
            continue
        avg_corr = corr_matrix.loc[asset, others].abs().mean()
        # Linear penalty: full penalty when avg_corr == 1
        excess = max(0.0, avg_corr - threshold) / (1.0 - threshold)
        penalties[asset] = float(np.clip(1.0 - excess, 0.25, 1.0))
    logger.info(f"Correlation penalties: {penalties}")
    return penalties


def _validate_corr_matrix(corr: pd.DataFrame) -> None:
    """Validate correlation matrix properties."""
    # Symmetric
    assert np.allclose(corr.values, corr.values.T, atol=1e-8), \
        "Correlation matrix is not symmetric."
    # Diagonal = 1
    assert np.allclose(np.diag(corr.values), 1.0, atol=1e-8), \
        "Diagonal elements are not 1."
    # Values in [-1, 1]
    assert (corr.values >= -1 - 1e-8).all() and (corr.values <= 1 + 1e-8).all(), \
        "Correlation values out of [-1, 1] range."
    # Positive semi-definite (eigenvalues >= 0)
    eigs = np.linalg.eigvalsh(corr.values)
    assert (eigs >= -1e-8).all(), f"Correlation matrix not PSD. Min eigenvalue: {eigs.min():.4f}"
    logger.info("✅ Correlation matrix validation passed.")


def plot_rolling_correlations(
    corr_df: pd.DataFrame,
    window: int = 30,
    save_path: str = None,
) -> None:
    """Plot rolling correlation time series."""
    cols = [c for c in corr_df.columns if f"_{window}d" in c]
    if not cols:
        logger.warning("No columns found for the specified window.")
        return

    fig, axes = plt.subplots(len(cols), 1, figsize=(14, 3.5 * len(cols)), sharex=True)
    fig.patch.set_facecolor("#0f0f1a")
    if len(cols) == 1:
        axes = [axes]

    colors = ["#00e5ff", "#ff6b6b", "#ffd166", "#06d6a0"]
    for ax, col, color in zip(axes, cols, colors):
        ax.set_facecolor("#0f0f1a")
        pair_name = col.replace(f"_corr_{window}d", "").replace("_", "/")
        series = corr_df[col].dropna()
        ax.plot(series.index, series.values, color=color, lw=1.5, alpha=0.9, label=pair_name)
        ax.axhline(HIGH_CORR_THRESHOLD, color="#ff6b6b", ls="--", lw=1, alpha=0.6, label="High threshold")
        ax.axhline(0.5, color="#888", ls=":", lw=1, alpha=0.4)
        ax.fill_between(series.index, series.values, alpha=0.1, color=color)
        ax.set_ylim(-1, 1)
        ax.set_ylabel("Correlation", color="white")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")
        ax.legend(facecolor="#0f0f1a", edgecolor="#333", labelcolor="white", loc="upper left")
        ax.set_title(f"{pair_name} — {window}d Rolling Correlation", color="white")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.xlabel("Date", color="white")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.tight_layout()
    plt.show()


def validate_correlation() -> None:
    """Validation tests for the correlation module."""
    rng = np.random.default_rng(0)
    n = 200

    # Strongly correlated BTC/ETH, uncorrelated BTC/SOL
    btc = pd.Series(rng.normal(0, 0.05, n), name="BTC")
    eth = btc + rng.normal(0, 0.01, n)
    eth.name = "ETH"
    sol = pd.Series(rng.normal(0, 0.05, n), name="SOL")

    returns = {"BTC": btc, "ETH": eth, "SOL": sol}

    corr_df = compute_pairwise_rolling_corr(returns, window=30)
    assert "BTC_ETH_corr_30d" in corr_df.columns
    assert "BTC_SOL_corr_30d" in corr_df.columns

    corr_matrix = get_current_corr_matrix(returns, window=30)
    _validate_corr_matrix(corr_matrix)

    # BTC/ETH should be highly correlated
    assert corr_matrix.loc["BTC", "ETH"] > 0.8, \
        f"BTC/ETH should be high-corr, got {corr_matrix.loc['BTC', 'ETH']:.3f}"

    penalties = correlation_penalty(corr_matrix)
    assert all(0.0 <= v <= 1.0 for v in penalties.values()), "Penalties out of range."
    # BTC and ETH should have higher penalty than SOL
    assert penalties["BTC"] <= penalties["SOL"] or penalties["ETH"] <= penalties["SOL"]

    logger.info("✅ Correlation validation tests passed.")
    print("✅ Correlation validation tests passed.")


if __name__ == "__main__":
    validate_correlation()

    # Demo with generated data
    rng = np.random.default_rng(42)
    n = 365
    dates = pd.date_range("2023-01-01", periods=n)
    btc = pd.Series(rng.normal(0, 0.04, n), index=dates)
    eth = btc * 0.85 + rng.normal(0, 0.02, n)
    sol = btc * 0.5 + rng.normal(0, 0.05, n)

    returns = {"BTC": btc, "ETH": eth, "SOL": sol}
    corr_matrix = get_current_corr_matrix(returns, window=30)
    print("\nCurrent 30d Correlation Matrix:")
    print(corr_matrix.round(3).to_string())

    penalties = correlation_penalty(corr_matrix)
    print("\nKelly correlation penalties:")
    for asset, p in penalties.items():
        print(f"  {asset}: {p:.3f}")
