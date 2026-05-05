# Polymarket Crypto Binary Options — Quant Trading System

# My Notes
Seems to poll like every 5 min getting like the averages of past data, inefficient need to fix.

A systematic trading strategy for Polymarket crypto binary options, combining options pricing theory with portfolio-aware bet sizing.

## Structure

```
polymarket_quant/
├── data/
│   ├── fetch_crypto.py        # yfinance pipeline + realised vol
│   └── fetch_polymarket.py    # Polymarket CLOB/Gamma API client
├── pricing/
│   ├── black_scholes.py       # BS binary option pricer
│   └── monte_carlo.py         # Merton Jump Diffusion MC pricer
├── analysis/
│   ├── calibration.py         # Historical calibration vs Polymarket
│   └── correlation.py         # Rolling correlation analysis
├── sizing/
│   └── kelly.py               # Correlation-adjusted fractional Kelly
├── execution/
│   └── trader.py              # Live market scanner + bet placement
└── notebooks/
    ├── 01_exploration.ipynb
    ├── 02_pricing_validation.ipynb
    └── 03_backtest_results.ipynb
```

## Setup

```bash
pip install yfinance numpy scipy pandas matplotlib plotly requests
pip install py-clob-client   # for live execution only
```

## Run Component Checks

```bash
cd polymarket_quant
python -m execution.trader --validate
```

This runs all validation tests across every module.

## Individual Module Tests

```bash
python -m pricing.black_scholes      # validate + demo BS pricer
python -m pricing.monte_carlo        # validate + demo MC pricer
python -m analysis.calibration       # validate + synthetic calibration demo
python -m analysis.correlation       # validate + correlation matrix demo
python -m sizing.kelly               # validate + portfolio sizing demo
python -m data.fetch_crypto          # pull live BTC/ETH/SOL data
python -m data.fetch_polymarket      # pull live Polymarket markets
```

## Paper Trading

```bash
python -m execution.trader --paper
```

## Architecture

| Stage | Module | What it does |
|---|---|---|
| 1 | `data/fetch_crypto.py` | Pull OHLCV + rolling realised vol |
| 2 | `pricing/black_scholes.py` | BS binary pricer, risk-neutral prob |
| 3 | `pricing/monte_carlo.py` | Jump diffusion MC, fat tail premium |
| 4 | `analysis/calibration.py` | Brier scores, systematic bias detection |
| 5 | `analysis/correlation.py` | Rolling pairwise corr, regime detection |
| 6 | `sizing/kelly.py` | Fractional + correlation-penalised Kelly |
| 7 | `execution/trader.py` | Scanner, pricer, sizer, executor |

## Key Design Decisions

**Why fractional Kelly (25%)?**  
Model uncertainty is real. Our vol input is realised vol, not implied — the market may be pricing information we don't have. 1/4 Kelly dramatically reduces drawdown in adverse scenarios while capturing most of the EV.

**Why Jump Diffusion over BS?**  
Crypto exhibits fat tails and gap behaviour (exchange hacks, regulatory events, liquidation cascades). BS systematically underprices OTM options. The fat tail premium from MC is informative — large BS/MC divergence signals high model risk.

**Why correlation-adjust Kelly?**  
BTC/ETH correlation is typically 0.7–0.9. Betting independently-sized Kelly on both is equivalent to betting a correlated portfolio — naive sizing implicitly overleverages the macro crypto factor.

## Environment Variables

```
POLYMARKET_API_KEY=<your_key>    # required for live execution only
```

## Risks

- Model uses realised vol as σ input — stale in fast-moving markets
- Jump parameters calibrated from history — regime changes invalidate them
- Polymarket CLOB liquidity can be thin — slippage not modelled
- No position-level stop losses implemented
