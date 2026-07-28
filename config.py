# config.py
# Central configuration for all scripts.
# Change values here; every script reads from this file.

# ── fetch_polymarket ──────────────────────────────────────────────────────────
FETCH_TIMEOUT              = 60       # seconds; generous for slow Gamma API
FETCH_CLOB_WINDOW_DAYS     = 5        # ±days tolerance when matching CLOB price history

OPEN_MAX_PAGES             = 10       # pages of events to scan for open markets
OPEN_PER_PAGE              = 100      # events per page
OPEN_MIN_EVENT_VOLUME      = 1_000.0  # min event volume24hr to stop scanning

RESOLVED_MAX_PAGES         = 1000      # pages of events to scan for resolved markets
RESOLVED_PER_PAGE          = 100      # events per page
RESOLVED_MIN_EVENT_VOLUME  = 1_000.0  # min event volume to stop scanning
RESOLVED_MIN_MARKET_VOL    = 100      # min individual market volume to include
RESOLVED_CONCURRENT        = 30       # max simultaneous requests to Gamma API

# ── backtest ──────────────────────────────────────────────────────────────────
BACKTEST_LIMIT             = 500      # max resolved contracts to fetch
LOOKBACK_DAYS              = 365*3      # days of crypto history to download
PRICING_OFFSET_MEAN_DAYS   = 21       # lognormal mean days before expiry (≈3 weeks)
PRICING_OFFSET_SIGMA_LN    = 0.9      # lognormal sigma — controls right tail; ~0.5% reach 6 months
MIN_T_DAYS                 = 7        # skip contracts with fewer days to expiry (was 3; near-expiry deep-OTM produces artifact edges)
MAX_T_DAYS                 = 180      # skip contracts with more days to expiry
BANKROLL                   = 1_000.0  # starting bankroll ($)
SAVE_PLOTS                 = True     # save equity curve + charts to PNG
MC_PATHS                   = 1_000   # Monte Carlo paths per pricing call

# ── kelly / sizing ────────────────────────────────────────────────────────────
KELLY_FRACTION             = 0.3     # fractional Kelly multiplier (1.0 = full Kelly)
MIN_EDGE                   = 0.005     # min probability edge required to place a bet
MAX_POSITION               = 0.15     # max fraction of bankroll on any single bet
MAX_EDGE                   = 0.2     # edges above this are skipped as near-expiry/vol artifacts
MAX_PAYOUT                 = 0.5     # max potential win as a fraction of bankroll (caps long-shot bets)
CORR_THRESHOLD             = 0.75     # correlation above which penalty is applied
MIN_STAKE_USD              = 1.0      # minimum bet size in dollars

# ── monte_carlo / jump diffusion ──────────────────────────────────────────────
JUMP_LAM                   = 8.0      # jump intensity (jumps per year)
JUMP_MU_J                  = -0.04   # mean log-jump size
JUMP_SIGMA_J               = 0.12    # std dev of log-jump size

# ── performance / reporting ───────────────────────────────────────────────────
RISK_FREE_RATE             = 0.045   # annualised risk-free rate (US 3-month T-bill approx)
