"""
fetch_polymarket.py
-------------------
Pull live and historical crypto threshold markets from the Polymarket CLOB API.
Matches markets to underlying assets (BTC, ETH, SOL).

All keyword searches run in parallel via asyncio + aiohttp.
Install: pip install aiohttp
"""

import asyncio
import re
import time
import logging
import json
import sys
import os
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
import requests
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from config import (
    FETCH_TIMEOUT, FETCH_CLOB_WINDOW_DAYS,
    OPEN_MAX_PAGES, OPEN_PER_PAGE, OPEN_MIN_EVENT_VOLUME,
    RESOLVED_MAX_PAGES, RESOLVED_PER_PAGE, RESOLVED_MIN_EVENT_VOLUME,
    RESOLVED_MIN_MARKET_VOL, RESOLVED_CONCURRENT,
)

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"
DEFAULT_TIMEOUT = FETCH_TIMEOUT

ASSET_KEYWORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana", "sol"],
}

PRICE_THRESHOLD_PATTERNS = [
    r"above \$[\d,]+",
    r"over \$[\d,]+",
    r"exceed[s]? \$[\d,]+",
    r"hit[s]? \$[\d,]+",
    r"reach[es]* \$[\d,]+",
    r"higher than \$[\d,]+",
    r"end.*above",
    r"price.*above",
    r"close above",
    r"trade above",
    r"surpass \$[\d,]+",
    r"\$[\d,]+ by \w+ 20\d\d",
    # Looser: "reach $85,000 in May" / "hit $100k by end of year"
    r"\$[\d,]+[kK]? (in|by) (january|february|march|april|may|june|july|august|september|october|november|december|end|q[1-4])",
    r"be above \$[\d,]+",
    r"stay above \$[\d,]+",
]

NOVELTY_REJECT_PATTERNS = [
    r"before gta",
    r"all.time high",
    r"ever reach",
    r"this decade",
    r"this year.*million",
    r"before.*inaug",
    # 5-min micro-markets: "Bitcoin Up or Down - January 20, 7:15AM-7:20AM ET"
    r"up or down",
    r"\d+:\d+[ap]m.*-.*\d+:\d+[ap]m",
]

# Non-crypto assets to reject even if they hit price threshold patterns
NON_CRYPTO_REJECT_PATTERNS = [
    r"s&p",
    r"spx",
    r"\bgold\b",
    r"\bsilver\b",
    r"\brwa\b",
    r"nasdaq",
    r"\bdow\b",
    r"\bfdv\b",
]

MAX_T_YEARS = 180 / 365


# ── Expiry resolution ──────────────────────────────────────────────────────────

import calendar
from datetime import date, datetime

MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

def _resolve_expiry(contract) -> Optional[date]:
    """
    Best-effort expiry date from end_date field or question text.
    - Uses end_date if present and parseable
    - Falls back to parsing question: "in May", "by June 30", "before July", etc.
    - "in <month>" -> last day of that month in current or next year
    - "before <month>" / "by <month>" -> 1st of that month (conservative)
    - "on May 4" / "April 27-May 3" -> specific date
    """
    today = date.today()

    # Try end_date from API first
    if contract.end_date:
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(contract.end_date[:19], fmt[:len(fmt)]).date()
            except ValueError:
                continue

    q = contract.question.lower()

    # "on May 4" / "on May 3" — specific day
    m = re.search(r"\bon (january|february|march|april|may|june|july|august|september|october|november|december) (\d{1,2})", q)
    if m:
        month = MONTH_NAMES[m.group(1)]
        day = int(m.group(2))
        year = today.year if date(today.year, month, day) >= today else today.year + 1
        return date(year, month, day)

    # "by June 30, 2026" / "by December 31, 2026"
    m = re.search(r"by (january|february|march|april|may|june|july|august|september|october|november|december) (\d{1,2}),? (20\d\d)", q)
    if m:
        return date(int(m.group(3)), MONTH_NAMES[m.group(1)], int(m.group(2)))

    # "April 27-May 3" ranges — use end of range
    m = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december) (\d{1,2})-(\d{1,2})", q)
    if m:
        month = MONTH_NAMES[m.group(1)]
        day = int(m.group(3))
        year = today.year if date(today.year, month, day) >= today else today.year + 1
        return date(year, month, day)

    # "in May" / "in Q2" -> last day of that month/quarter
    m = re.search(r"\bin (january|february|march|april|may|june|july|august|september|october|november|december)", q)
    if m:
        month = MONTH_NAMES[m.group(1)]
        year = today.year if month >= today.month else today.year + 1
        last_day = calendar.monthrange(year, month)[1]
        return date(year, month, last_day)

    # "before May" / "by May" -> first of that month
    m = re.search(r"\b(before|by) (january|february|march|april|may|june|july|august|september|october|november|december)", q)
    if m:
        month = MONTH_NAMES[m.group(2)]
        year = today.year if month >= today.month else today.year + 1
        return date(year, month, 1)

    # "by December 31, 2026" already caught above; also catch "December 31, 2026" without "by"
    m = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december) (\d{1,2}),? (20\d\d)", q)
    if m:
        return date(int(m.group(3)), MONTH_NAMES[m.group(1)], int(m.group(2)))

    return None


@dataclass
class PolymarketContract:
    condition_id: str
    question: str
    asset: str
    strike: Optional[float]
    end_date: Optional[str]
    yes_price: float
    no_price: float
    volume: float
    volume24hr: float
    resolved: bool
    outcome: Optional[str] = None
    market_slug: str = ""
    price_uncertain: bool = False   # True if price could not be read from API
    option_type: str = "call"       # "call" (above/reach/hit) or "put" (dip/below/drop)
    extra: dict = field(default_factory=dict)
    token_ids: dict = field(default_factory=dict)  # Add this line

    @property
    def implied_prob(self) -> float:
        return self.yes_price

    @property
    def is_valid(self) -> bool:
        """
        True if the contract looks like a clean, tradeable price-threshold market.
        Checks:
          - price is not pinned at 0 or 1 (not resolved/junk)
          - has a parseable expiry date
          - strike price was detected
          - question contains an action word (reach/hit/dip/above/surpass etc.)
          - not price_uncertain
        """
        if self.price_uncertain:
            return False
        if self.yes_price >= 0.99 or self.yes_price <= 0.05:
            return False
        if self.strike is None:
            return False
        if _resolve_expiry(self) is None:
            return False
        action_words = ["reach", "hit", "dip", "above", "surpass", "exceed", "over", "higher"]
        if not any(w in self.question.lower() for w in action_words):
            return False
        # Reject negated/ambiguous directional questions — can't price "not reach" or "stay above"
        # correctly without knowing the contract's settlement logic
        if re.search(r"\bnot\s+(reach|hit|surpass|exceed|go|trade|close)\b", self.question, re.IGNORECASE):
            return False
        if re.search(r"\bstay\s+above\b|\bremain\s+above\b|\bhold\s+above\b", self.question, re.IGNORECASE):
            return False
        # Must have exactly one $ price (not "hit $60k or $80k first" style)
        if len(re.findall(r"\$[\d,]+[kK]?", self.question)) != 1:
            return False
        # Must mention a month or year (date-bounded)
        months = ["january","february","march","april","may","june",
                  "july","august","september","october","november","december"]
        if not any(m in self.question.lower() for m in months) and not re.search(r"20\d\d", self.question):
            return False
        return True


# ── Sync helpers (CLOB only) ───────────────────────────────────────────────────

def _get(url: str, params: dict = None) -> dict | list:
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            logger.warning(f"HTTP {resp.status_code} on {url}: {e}")
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
            else:
                raise
        except requests.RequestException as e:
            logger.warning(f"Request error (attempt {attempt+1}): {e}")
            time.sleep(1)
    raise ConnectionError(f"Failed to fetch {url} after 3 attempts.")


# ── Parsing helpers ────────────────────────────────────────────────────────────

def _parse_strike(question: str) -> Optional[float]:
    """
    Extract strike price from question. Handles:
      $150k / $150K  -> 150_000
      $1,000,000     -> 1_000_000
      $95,000        -> 95_000
    Returns None if no price found.
    """
    # FIX: capture k/K as separate group 2 so match.end() is correct
    match = re.search(r"\$([0-9,]+(?:\.[0-9]+)?)([kK])?", question)
    if not match:
        return None
    raw = match.group(1).replace(",", "")
    val = float(raw)
    if match.group(2):  # explicit k/K suffix in group 2
        val *= 1_000
    return val


CALL_PATTERNS = re.compile(
    r"\b(above|over|exceed|exceeds|exceeded|exceeding|reach|reaches|hit|hits|surpass|surpasses|higher)\b",
    re.IGNORECASE,
)

PUT_PATTERNS = re.compile(
    r"\b(dip|dips|drop|drops|fall|falls|below|under|crash|crashes)\b",
    re.IGNORECASE,
)

def _detect_option_type(question: str) -> str:
    """
    Return "call" for upside-threshold questions (above/reach/hit/surpass),
    "put" for downside-threshold questions (dip/drop/fall/below).
    Call takes precedence when both indicators appear (e.g. double-barrier markets
    whose resolution text mentions both directions).
    """
    if CALL_PATTERNS.search(question):
        return "call"
    if PUT_PATTERNS.search(question):
        return "put"
    return "call"


def _classify_asset(question: str) -> Optional[str]:
    q = question.lower()
    for asset, keywords in ASSET_KEYWORDS.items():
        if any(re.search(rf"\b{kw}\b", q) for kw in keywords):
            return asset
    return None


def _is_price_threshold_market(question: str) -> bool:
    q = question.lower()
    if not any(re.search(pat, q) for pat in PRICE_THRESHOLD_PATTERNS):
        return False
    if any(re.search(pat, q) for pat in NOVELTY_REJECT_PATTERNS):
        return False
    if any(re.search(pat, q) for pat in NON_CRYPTO_REJECT_PATTERNS):
        return False
    return True


# ── Price extraction ───────────────────────────────────────────────────────────

def _extract_prices(m: dict) -> tuple[float, float]:
    """
    Extract YES/NO prices from a Gamma market dict.
    From --raw output, the API returns:
      outcomePrices: "[\"0.585\", \"0.415\"]"  (JSON string, not a list)
      outcomes:      "[\"Yes\", \"No\"]"        (JSON string, not a list)
      bestBid: 0.58, bestAsk: 0.59, lastTradePrice: 0.58

    Tries in order:
      1. tokens[].outcome + tokens[].price
      2. outcomePrices + outcomes (JSON-parsed strings)
      3. bestBid/bestAsk midpoint
      4. lastTradePrice
    Returns (yes_price, no_price), defaulting to (0.5, 0.5) if nothing works.
    """
    def _parse_json_field(v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return []
        return v if isinstance(v, list) else []

    # Method 1: tokens array
    tokens = m.get("tokens", [])
    yes_price = no_price = None
    for t in tokens:
        outcome = t.get("outcome", "").upper()
        raw = t.get("price")
        if raw is None:
            continue
        price = float(raw)
        if outcome == "YES":
            yes_price = price
        elif outcome == "NO":
            no_price = price
    if yes_price is not None:
        if no_price is None:
            no_price = 1.0 - yes_price
        return yes_price, no_price

    # Method 2: outcomePrices + outcomes — both are JSON strings in Gamma API
    outcome_prices = _parse_json_field(m.get("outcomePrices", []))
    outcomes = _parse_json_field(m.get("outcomes", []))
    if outcome_prices and outcomes:
        for label, price_str in zip(outcomes, outcome_prices):
            try:
                price = float(price_str)
            except (ValueError, TypeError):
                continue
            if str(label).upper() == "YES":
                yes_price = price
            elif str(label).upper() == "NO":
                no_price = price
        if yes_price is not None:
            if no_price is None:
                no_price = 1.0 - yes_price
            return yes_price, no_price

    # Method 3: bestBid/bestAsk midpoint
    best_bid = m.get("bestBid")
    best_ask = m.get("bestAsk")
    if best_bid is not None and best_ask is not None:
        yes_price = (float(best_bid) + float(best_ask)) / 2
        return yes_price, 1.0 - yes_price

    # Method 4: lastTradePrice
    last = m.get("lastTradePrice")
    if last is not None:
        return float(last), 1.0 - float(last)

    return 0.5, 0.5


# ── CLOB price helpers ─────────────────────────────────────────────────────────

def _parse_yes_token(token_ids_raw) -> str | None:
    """Parse clobTokenIds (JSON string or list) and return the YES token (index 0)."""
    if not token_ids_raw:
        return None
    try:
        ids = json.loads(token_ids_raw) if isinstance(token_ids_raw, str) else token_ids_raw
        return ids[0] if ids else None
    except Exception:
        return None


def fetch_clob_price_at_date(yes_token: str, target_date, expiry_date=None) -> float | None:
    """
    Fetch the YES price nearest to target_date from the CLOB prices-history endpoint.
    Falls back through fidelities (daily → hourly → per-minute) to find data.
    Excludes post-expiry settlement prices when expiry_date is provided.
    Accepts up to ±5 day window to handle sparse markets.
    """
    if not yes_token:
        return None

    if isinstance(target_date, datetime):
        target_ts = int(target_date.timestamp())
    else:
        target_ts = int(datetime.combine(target_date, datetime.min.time()).timestamp())

    expiry_cutoff = None
    if expiry_date is not None:
        _ed = expiry_date if isinstance(expiry_date, datetime) else datetime.combine(expiry_date, datetime.min.time())
        expiry_cutoff = int(_ed.timestamp())

    max_delta = FETCH_CLOB_WINDOW_DAYS * 86400

    for fidelity in (1440, 60, 1):
        try:
            resp = requests.get(
                f"{CLOB_BASE}/prices-history",
                params={"market": yes_token, "interval": "max", "fidelity": fidelity},
                timeout=30,
            )
            if resp.status_code != 200:
                logger.debug(f"CLOB prices-history HTTP {resp.status_code} for {yes_token[:20]}")
                return None
            history = resp.json().get("history", [])
            if not history:
                continue
            # Exclude post-expiry settlement prices
            if expiry_cutoff is not None:
                history = [h for h in history if int(h["t"]) < expiry_cutoff]
            if not history:
                continue
            best = min(history, key=lambda h: abs(int(h["t"]) - target_ts))
            if abs(int(best["t"]) - target_ts) > max_delta:
                continue
            return float(np.clip(float(best["p"]), 0.01, 0.99))
        except Exception as e:
            logger.debug(f"Error fetching CLOB price for {yes_token[:20]} fidelity={fidelity}: {e}")
    return None


def fetch_all_market_probs(markets_list: list, pricing_dates: list, expiry_dates: list = None) -> list:
    """Fetch historical YES-token prices for a list of contracts via CLOB."""
    from tqdm import tqdm
    results = []
    for i, (market, pricing_date) in enumerate(tqdm(
        zip(markets_list, pricing_dates),
        total=len(markets_list),
        desc="Fetching historical prices",
    )):
        yes_token = _parse_yes_token(market.token_ids)
        d = pricing_date.date() if hasattr(pricing_date, "date") else pricing_date
        expiry = expiry_dates[i] if expiry_dates else None
        results.append(fetch_clob_price_at_date(yes_token, d, expiry_date=expiry))
    logger.info(f"  Fetched {sum(1 for p in results if p is not None)}/{len(markets_list)} historical prices")
    return results


# ── Async fetch layer ──────────────────────────────────────────────────────────

async def _fetch_events_page(
    session: aiohttp.ClientSession,
    offset: int,
    active: bool,
    closed: bool,
    per_page: int = 100,
    order: str = "volume24hr",
) -> list[dict]:
    """
    Fetch one page of EVENTS sorted by the given field desc.
    Events contain nested markets[] — this is how Polymarket groups related sub-markets
    (e.g. "What price will Bitcoin hit in May?" contains reach-80k, dip-75k, etc.)
    """
    timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT, connect=10, sock_read=30)
    params = {
        "limit": per_page,
        "offset": offset,
        "active": str(active).lower(),
        "closed": str(closed).lower(),
        "order": order,
        "ascending": "false",
    }
    try:
        async with session.get(f"{GAMMA_BASE}/events", params=params, timeout=timeout) as resp:
            resp.raise_for_status()
            data = await resp.json()
            page = data if isinstance(data, list) else data.get("events", data.get("results", []))
            return page or []
    except Exception as e:
        logger.warning(f"Events page fetch failed at offset {offset}: {e}")
        return []


async def _fetch_all_parallel(active: bool, closed: bool) -> dict[str, dict]:
    """
    Fetch events in parallel, explode nested markets, filter client-side for crypto.
    Uses /events endpoint sorted by volume24hr so high-traffic crypto events surface first.
    Stops when a page has no events above MIN_EVENT_VOLUME24HR.
    """
    MIN_EVENT_VOLUME = OPEN_MIN_EVENT_VOLUME
    MAX_PAGES = OPEN_MAX_PAGES
    PER_PAGE = OPEN_PER_PAGE

    offsets = [i * PER_PAGE for i in range(MAX_PAGES)]
    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        pages = await asyncio.gather(
            *[_fetch_events_page(session, off, active, closed, PER_PAGE) for off in offsets],
            return_exceptions=True,
        )

    seen: dict[str, dict] = {}
    total_events = 0
    for page in pages:
        if isinstance(page, Exception) or not page:
            continue
        # Stop once event volume drops off
        page_vols = [float(e.get("volume24hr", e.get("volume", 0)) or 0) for e in page]
        if not page_vols or max(page_vols) < MIN_EVENT_VOLUME:
            break
        total_events += len(page)
        for event in page:
            # Each event has a nested markets list
            markets = event.get("markets", [])
            for m in markets:
                q = m.get("question", "").lower()
                if not any(re.search(rf"\b{kw}\b", q) for kws in ASSET_KEYWORDS.values() for kw in kws):
                    continue
                # Store event volume separately — don't conflate with market's own trading vol
                m["_event_volume24hr"] = float(event.get("volume24hr") or 0)
                m["_event_volume"] = float(event.get("volume") or 0)
                # Use market's own volume if present, else 0 (not inherited from event)
                if "volume24hr" not in m:
                    m["volume24hr"] = 0
                if "volume" not in m:
                    m["volume"] = 0
                cid = m.get("conditionId", m.get("id", ""))
                if cid and cid not in seen:
                    seen[cid] = m

    logger.info(f"  Scanned {total_events} events -> {len(seen)} crypto sub-markets")
    return seen


# ── Contract builder ───────────────────────────────────────────────────────────

def _build_contracts(seen: dict[str, dict], resolved: bool, limit: int) -> list[PolymarketContract]:
    contracts = []
    for m in seen.values():
        question = m.get("question", "")
        asset = _classify_asset(question)
        if not asset or not _is_price_threshold_market(question):
            continue



        # Skip markets that look like zombies: end date in the past, or yes/no price
        # is stuck at exactly 0/1 with no recent trading activity (volume < 1000)
        end_raw = m.get("endDate") or m.get("end_date", "")
        if end_raw:
            try:
                from datetime import datetime, timezone
                end_dt = datetime.fromisoformat(end_raw.replace("Z", "+00:00"))
                if end_dt < datetime.now(timezone.utc):
                    continue  # already expired
            except ValueError:
                pass

        # Skip resolved/settled markets — price pinned at 0 or 1 means outcome is known
        yes_price_raw, no_price_raw = _extract_prices(m)
        if yes_price_raw >= 0.99 or yes_price_raw <= 0.01:
            continue

        # Also skip if the market is explicitly marked resolved/archived
        if m.get("resolved") or m.get("archived") or m.get("closed"):
            continue

        # Skip if market has no own 24hr volume — means it's delisted or broken on frontend
        own_vol24 = float(m.get("volume24hr") or 0)
        if own_vol24 == 0:
            continue

        yes_price, no_price = _extract_prices(m)
        winner = None
        if resolved:
            if yes_price > 0.99:
                winner = "YES"
            elif no_price > 0.99:
                winner = "NO"

        contracts.append(PolymarketContract(
            condition_id=m.get("conditionId", m.get("id", "")),
            question=question,
            asset=asset,
            strike=_parse_strike(question),
            end_date=m.get("endDate") or m.get("end_date"),
            yes_price=yes_price,
            no_price=no_price,
            volume=float(m.get("volume", 0)),
            volume24hr=float(m.get("volume24hr", 0)),
            resolved=resolved,
            outcome=winner if resolved else None,
            market_slug=m.get("slug", ""),
            price_uncertain=(yes_price == 0.5 and no_price == 0.5),
            option_type=_detect_option_type(question),
            extra=m,
            token_ids=m.get("clobTokenIds", {}),
        ))

        if len(contracts) >= limit:
            break

    return contracts


# ── Resolved market keyword search ────────────────────────────────────────────
# The events endpoint used for open markets surfaces very few closed contracts.
# Instead we search the /markets endpoint with targeted phrases so we find the
# long-form dollar-price markets that have CLOB price history.

_RESOLVED_MIN_VOL = RESOLVED_MIN_MARKET_VOL

_RESOLVED_QUERIES = [
    "will the price of bitcoin", "will the price of ethereum", "will the price of solana",
    "will bitcoin", "will ethereum", "will solana",
    "bitcoin above $", "bitcoin reach $", "bitcoin below $",
    "ethereum above $", "ethereum reach $", "ethereum below $",
    "solana above $", "solana reach $", "solana below $",
    "btc above", "eth above", "sol above",
]
_RESOLVED_PER_PAGE   = RESOLVED_PER_PAGE
_RESOLVED_PAGES      = 2000    # only used by the commented-out keyword-search method
_RESOLVED_CONCURRENT = 20      # only used by the commented-out keyword-search method


async def _fetch_resolved_page(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    query: str,
    offset: int,
) -> list[dict]:
    async with sem:
        timeout = aiohttp.ClientTimeout(total=60, connect=10, sock_read=30)
        params = {"q": query, "closed": "true", "limit": _RESOLVED_PER_PAGE,
                  "offset": offset, "order": "volume", "ascending": "false",
                  "enableOrderBook": "true"}
        try:
            async with session.get(f"{GAMMA_BASE}/markets", params=params, timeout=timeout) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data if isinstance(data, list) else data.get("markets", [])
        except Exception as e:
            logger.debug(f"Resolved fetch failed '{query}' offset={offset}: {e}")
            return []


# async def _fetch_all_resolved_parallel() -> dict[str, dict]:
#     tasks = [(q, off) for q in _RESOLVED_QUERIES
#              for off in range(0, _RESOLVED_PAGES * _RESOLVED_PER_PAGE, _RESOLVED_PER_PAGE)]
#     sem = asyncio.Semaphore(_RESOLVED_CONCURRENT)
#     connector = aiohttp.TCPConnector(ssl=False)
#     async with aiohttp.ClientSession(connector=connector) as session:
#         pages = await asyncio.gather(
#             *[_fetch_resolved_page(session, sem, q, off) for q, off in tasks],
#             return_exceptions=True,
#         )
#     seen: dict[str, dict] = {}
#     for page in pages:
#         if isinstance(page, Exception) or not page:
#             continue
#         for m in page:
#             cid = m.get("conditionId", m.get("id", ""))
#             if cid and cid not in seen:
#                 seen[cid] = m
#     logger.info(f"  Resolved fetch: {len(seen)} unique candidates "
#                 f"({len(_RESOLVED_QUERIES)} queries × {_RESOLVED_PAGES} pages)")
#     return seen
async def _fetch_all_resolved_parallel() -> dict[str, dict]:
    """
    Fetch closed/resolved crypto price-threshold markets via the /events endpoint.
    Sorted by volume descending so high-volume markets surface first.
    ~100 pages × 100 events each = 10,000 events scanned.
    Far more efficient than 36,000 keyword searches against /markets.
    """
    MAX_PAGES = RESOLVED_MAX_PAGES
    PER_PAGE = RESOLVED_PER_PAGE
    MIN_VOL = RESOLVED_MIN_EVENT_VOLUME

    offsets = [i * PER_PAGE for i in range(MAX_PAGES)]
    sem = asyncio.Semaphore(RESOLVED_CONCURRENT)

    async def _fetch_page_throttled(session, off):
        async with sem:
            return await _fetch_events_page(
                session, off, active=False, closed=True, per_page=PER_PAGE, order="volume"
            )

    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        pages = await asyncio.gather(
            *[_fetch_page_throttled(session, off) for off in offsets],
            return_exceptions=True,
        )

    seen: dict[str, dict] = {}
    total_events = 0
    for page in pages:
        if isinstance(page, Exception) or not page:
            continue
        page_vols = [float(e.get("volume", 0) or 0) for e in page]
        if not page_vols or max(page_vols) < MIN_VOL:
            break  # pages are volume-sorted desc; once this page is low, all later ones are too
        total_events += len(page)
        for event in page:
            for m in event.get("markets", []):
                q = m.get("question", "").lower()
                if not any(re.search(rf"\b{kw}\b", q)
                           for kws in ASSET_KEYWORDS.values() for kw in kws):
                    continue
                cid = m.get("conditionId", m.get("id", ""))
                if cid and cid not in seen:
                    seen[cid] = m

    logger.info(f"  Resolved fetch: {len(seen)} unique crypto candidates "
                f"from {total_events} events ({MAX_PAGES} pages)")
    return seen


def _build_resolved_contracts(seen: dict[str, dict], limit: int) -> list[PolymarketContract]:
    contracts = []
    for m in seen.values():
        question = m.get("question", "")
        asset = _classify_asset(question)
        if not asset or not _is_price_threshold_market(question):
            continue
        if float(m.get("volume", 0)) < _RESOLVED_MIN_VOL:
            continue
        if not _parse_yes_token(m.get("clobTokenIds")):
            continue
        # Skip AMM-only markets — no pre-expiry CLOB price history available
        if not m.get("enableOrderBook", False):
            continue
        yes_price, no_price = _extract_prices(m)
        winner = None
        if yes_price >= 0.99:
            winner = "YES"
        elif no_price >= 0.99:
            winner = "NO"
        if not winner:
            res = m.get("resolution")
            if isinstance(res, dict):
                outcome = res.get("outcome")
                if outcome:
                    winner = "YES" if outcome == "Yes" else "NO"
        if not winner:
            continue
        contracts.append(PolymarketContract(
            condition_id=m.get("conditionId", m.get("id", "")),
            question=question,
            asset=asset,
            strike=_parse_strike(question),
            end_date=m.get("endDate") or m.get("end_date"),
            yes_price=yes_price,
            no_price=no_price,
            volume=float(m.get("volume", 0)),
            volume24hr=float(m.get("volume24hr", 0)),
            resolved=True,
            outcome=winner,
            market_slug=m.get("slug", ""),
            price_uncertain=False,
            option_type=_detect_option_type(question),
            extra=m,
            token_ids=m.get("clobTokenIds", {}),
        ))
        if len(contracts) >= limit:
            break
    return contracts


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_open_markets(limit: int = 500) -> list[PolymarketContract]:
    """Pull open crypto price-threshold markets. All keyword searches run in parallel."""
    logger.info("Fetching open markets (parallel)...")
    seen = asyncio.run(_fetch_all_parallel(active=True, closed=False))
    logger.info(f"  {len(seen)} unique markets before filtering")
    contracts = _build_contracts(seen, resolved=False, limit=limit)
    logger.info(f"Found {len(contracts)} crypto price-threshold markets.")
    _validate_markets(contracts)
    return contracts


def fetch_resolved_markets(limit: int = 500) -> list[PolymarketContract]:
    """
    Fetch resolved crypto price-threshold markets for backtesting.
    Uses parallel keyword searches against Gamma /markets (closed=true).
    """
    logger.info(f"Fetching resolved markets ({len(_RESOLVED_QUERIES)} queries × "
                f"{_RESOLVED_PAGES} pages, concurrency={_RESOLVED_CONCURRENT})...")
    seen = asyncio.run(_fetch_all_resolved_parallel())
    contracts = _build_resolved_contracts(seen, limit)
    logger.info(f"Found {len(contracts)} resolved crypto price-threshold markets.")
    return contracts


def get_orderbook_midpoint(condition_id: str) -> Optional[float]:
    """CLOB best bid/ask midpoint for a YES token."""
    try:
        data = _get(f"{CLOB_BASE}/book", params={"token_id": condition_id})
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        if bids and asks:
            return (float(bids[0]["price"]) + float(asks[0]["price"])) / 2
    except Exception as e:
        logger.debug(f"Orderbook unavailable for {condition_id}: {e}")
    return None


def _validate_markets(contracts: list[PolymarketContract]) -> None:
    if not contracts:
        logger.warning("No contracts returned — market may be inactive or API changed.")
        return
    for c in contracts:
        assert 0.0 <= c.yes_price <= 1.0, f"Invalid YES price {c.yes_price} for {c.condition_id}"
        assert c.asset in ASSET_KEYWORDS, f"Unknown asset: {c.asset}"
    logger.info("Market validation passed.")


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if "--raw" in sys.argv:
        # Dump raw price fields from first bitcoin market to verify _extract_prices
        resp = requests.get(
            f"{GAMMA_BASE}/markets",
            params={"limit": 5, "active": "true", "closed": "false", "q": "bitcoin"},
            timeout=60,
        )
        markets = resp.json()
        if isinstance(markets, dict):
            markets = markets.get("markets", markets.get("results", []))
        m = markets[0] if markets else {}
        price_keys = {k: v for k, v in m.items() if any(x in k.lower() for x in
            ["price", "token", "outcome", "bid", "ask", "prob", "yes", "no"])}
        print(json.dumps(price_keys, indent=2))
        print(f"\n_extract_prices -> {_extract_prices(m)}")
        sys.exit(0)

    if "--debug" in sys.argv:
        seen = asyncio.run(_fetch_all_parallel(active=True, closed=False))
        raw_list = list(seen.values())
        print(f"\nRaw API returned {len(raw_list)} unique crypto markets\n")
        print(f"{'Vol':>10}  {'End':>10}  {'Asset?':<6}  {'Threshold?':<10}  {'Rejected?':<10}  Question")
        print("-" * 100)
        for m in sorted(raw_list, key=lambda x: -float(x.get("volume", 0))):
            q = m.get("question", "")
            vol = float(m.get("volume", 0))
            end = str(m.get("endDate") or m.get("end_date", ""))[:10]
            asset = _classify_asset(q) or "-"
            is_thresh = any(re.search(p, q.lower()) for p in PRICE_THRESHOLD_PATTERNS)
            is_novelty = any(re.search(p, q.lower()) for p in NOVELTY_REJECT_PATTERNS)
            print(f"{vol:>10,.0f}  {end:>10}  {asset:<6}  {str(is_thresh):<10}  {str(is_novelty):<10}  {q[:60]}")
    else:
        markets = fetch_open_markets(limit=100)
        print(f"\n{'Asset':<6} {'Implied Prob':>12} {'Strike':>12} {'Expiry':>12} {'Vol 24h':>13} {'Vol Total':>13} {'Type':>5} {'Valid':>5}  Question")
        print("-" * 140)
        for m in sorted((m for m in markets if m.volume24hr > 0), key=lambda x: -x.volume24hr)[:30]:
            strike_str = f"${m.strike:,.0f}" if m.strike else "  N/A"
            expiry = _resolve_expiry(m)
            expiry_str = expiry.strftime("%Y-%m-%d") if expiry else "  unknown"
            flag = " [price?]" if m.price_uncertain else ""
            vol24_str = f"${m.volume24hr:,.0f}" if m.volume24hr else "  $0"
            vol_str   = f"${m.volume:,.0f}"     if m.volume     else "  $0"
            valid_str = "YES" if m.is_valid else "NO"
            type_str  = m.option_type.upper()
            print(f"{m.asset:<6} {m.implied_prob:>12.1%} {strike_str:>12} {expiry_str:>12} {vol24_str:>13} {vol_str:>13} {type_str:>5} {valid_str:>5}  {m.question[:45]}{flag}")