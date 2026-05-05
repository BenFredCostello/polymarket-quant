"""
fetch_crypto.py
---------------
Pull BTC, ETH, SOL daily OHLCV via yfinance and compute rolling realised volatility.
"""

import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ASSETS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
}

VOL_WINDOWS = [30, 90]  # days
TRADING_DAYS_PER_YEAR = 365  # crypto trades 24/7


def fetch_ohlcv(ticker: str, start: str, end: str = None) -> pd.DataFrame:
    """Download OHLCV data for a single ticker."""
    end = end or datetime.today().strftime("%Y-%m-%d")
    logger.info(f"Fetching {ticker} from {start} to {end}")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker or date range.")

    # yfinance ≥0.2 returns MultiIndex columns (field, ticker) — flatten to field names only
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.dropna(inplace=True)

    logger.info(f"  {ticker}: {len(df)} rows, {df.index[0].date()} → {df.index[-1].date()}")
    return df


def compute_log_returns(close: pd.Series) -> pd.Series:
    """Compute daily log returns."""
    return np.log(close / close.shift(1)).dropna()


def compute_realised_vol(close: pd.Series, window: int) -> pd.Series:
    """
    Annualised rolling realised volatility from log returns.
    Vol = std(log returns over window) * sqrt(TRADING_DAYS_PER_YEAR)
    """
    log_ret = compute_log_returns(close)
    rv = log_ret.rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    rv.name = f"realised_vol_{window}d"
    return rv


def build_crypto_dataset(
    assets: dict = ASSETS,
    lookback_days: int = 365,
    end_date: str = None,
) -> dict[str, pd.DataFrame]:
    """
    Pull OHLCV + realised vol for all assets.

    Returns
    -------
    dict mapping asset name → DataFrame with columns:
        Open, High, Low, Close, Volume, log_return,
        realised_vol_30d, realised_vol_90d
    """
    end_date = end_date or datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=lookback_days + 90)).strftime("%Y-%m-%d")

    results = {}
    for name, ticker in assets.items():
        try:
            df = fetch_ohlcv(ticker, start=start_date, end=end_date)
            df["log_return"] = compute_log_returns(df["Close"])
            for w in VOL_WINDOWS:
                df[f"realised_vol_{w}d"] = compute_realised_vol(df["Close"], w)
            # Trim to requested lookback
            cutoff = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            df = df[df.index >= cutoff]
            results[name] = df
        except Exception as e:
            logger.error(f"Failed to fetch {name} ({ticker}): {e}")

    _validate_dataset(results)
    return results


def _validate_dataset(data: dict) -> None:
    """Sanity checks on the assembled dataset."""
    assert len(data) > 0, "No assets loaded — check network and tickers."
    for name, df in data.items():
        assert not df["Close"].isnull().all(), f"{name}: Close prices are all NaN."
        assert f"realised_vol_30d" in df.columns, f"{name}: Missing 30d vol column."
        assert f"realised_vol_90d" in df.columns, f"{name}: Missing 90d vol column."
        assert len(df) >= 30, f"{name}: Too few rows ({len(df)}) — increase lookback_days."
        latest_vol = float(df["realised_vol_30d"].dropna().iloc[-1])
        assert 0.0 < latest_vol < 10.0, f"{name}: Suspicious vol value: {latest_vol:.4f}"
    logger.info("✅ Dataset validation passed.")


def get_latest_vol(data: dict, asset: str, window: int = 30) -> float:
    """Return the most recent realised vol for an asset."""
    col = f"realised_vol_{window}d"
    df = data.get(asset)
    if df is None:
        raise KeyError(f"Asset '{asset}' not in dataset.")
    series = df[col].dropna()
    if series.empty:
        raise ValueError(f"No vol data for {asset} with window {window}d.")
    return float(series.iloc[-1])


# ── Deribit Implied Volatility (DVOL) ─────────────────────────────────────────

DVOL_ASSETS   = {"BTC", "ETH"}
_DERIBIT_BASE = "https://www.deribit.com/api/v2"


def fetch_dvol_series(currency: str, start_date: date, end_date: date) -> dict:
    """
    Fetch Deribit DVOL index for BTC or ETH at daily resolution.
    Returns {date: annualised_iv} where iv is a decimal (0.60 = 60% vol).
    """
    start_ms = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
    end_ms   = int(datetime.combine(end_date,   datetime.max.time()).timestamp() * 1000)
    try:
        resp = requests.get(
            f"{_DERIBIT_BASE}/public/get_volatility_index_data",
            params={"currency": currency, "start_timestamp": start_ms,
                    "end_timestamp": end_ms, "resolution": "1D"},
            timeout=30,
        )
        if resp.status_code != 200:
            logger.warning(f"Deribit DVOL {currency}: HTTP {resp.status_code}")
            return {}
        rows = resp.json().get("result", {}).get("data", [])
        # Each row: [timestamp_ms, open, high, low, close]
        return {
            datetime.fromtimestamp(r[0] / 1000).date(): r[4] / 100.0
            for r in rows
        }
    except Exception as e:
        logger.warning(f"Deribit DVOL fetch failed for {currency}: {e}")
        return {}


def build_dvol_cache(pricing_dates: list) -> dict:
    """Pre-fetch DVOL for all DVOL_ASSETS covering the full pricing date range."""
    if not pricing_dates:
        return {}
    start = min(d.date() if hasattr(d, "date") else d for d in pricing_dates) - timedelta(days=10)
    end   = max(d.date() if hasattr(d, "date") else d for d in pricing_dates) + timedelta(days=2)
    cache = {}
    for asset in DVOL_ASSETS:
        series = fetch_dvol_series(asset, start, end)
        cache[asset] = series
        logger.info(f"  Deribit DVOL {asset}: {len(series)} daily obs ({start} to {end})")
    return cache


def get_current_dvol(asset: str) -> float | None:
    """Fetch the most recent Deribit DVOL for live trading. Returns annualised IV or None."""
    if asset not in DVOL_ASSETS:
        return None
    today = datetime.today().date()
    series = fetch_dvol_series(asset, today - timedelta(days=5), today)
    if not series:
        return None
    return series[max(series.keys())]


def get_implied_vol(asset: str, pricing_date, dvol_cache: dict, hist_sigma: float) -> tuple:
    """
    Return (sigma, source).
    BTC/ETH: Deribit DVOL with ±3-day search window, else fall back to hist.
    Others: always hist.
    """
    if asset in DVOL_ASSETS and asset in dvol_cache:
        series = dvol_cache[asset]
        if hasattr(pricing_date, "date"):
            pricing_date = pricing_date.date()
        for delta in range(4):
            for d in (pricing_date - timedelta(days=delta),
                      pricing_date + timedelta(days=delta)):
                if d in series:
                    return series[d], "deribit_dvol"
    return hist_sigma, "realized_vol"


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = build_crypto_dataset(lookback_days=180)
    for asset, df in data.items():
        vol30 = get_latest_vol(data, asset, 30)
        vol90 = get_latest_vol(data, asset, 90)
        print(f"{asset}: latest close = ${df['Close'].iloc[-1]:,.2f}  |  "
              f"30d vol = {vol30:.1%}  |  90d vol = {vol90:.1%}")