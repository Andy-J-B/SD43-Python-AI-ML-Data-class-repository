"""
STUDENT TEMPLATE - data_client.py
Provides functions that fetch financial data from yfinance, Finnhub,
Financial Modeling Prep, and NewsAPI.  Each function is already set up
with the correct import, signature, and error-handling skeleton.

YOUR JOB: Fill in the logic between the comments.
"""

import os
import asyncio
import yfinance as yf
import httpx
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
FMP_KEY = os.getenv("FMP_API_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_API_KEY", "")

_http = httpx.Client(timeout=10)
_async_http = None


def _get_async_client():
    """Return (or create) the shared async HTTP client."""
    global _async_http
    # If there is no async client yet, or it has been closed,
    # create a new one and store it.  Then return it.
    pass


async def _close_async_client():
    """Cleanly close the async client if it exists and is open."""
    global _async_http
    pass


def get_usd_to_cad() -> float:
    """
    Fetch the current USD → CAD exchange rate.
    Try the primary URL first; fall back to the jsDelivr mirror on failure.
    Return the cad value from the JSON response.
    If everything fails, return a default of 1.35.
    """
    primary_url = "https://latest.currency-api.pages.dev/v1/currencies/usd.json"
    fallback_url = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json"
    # --- YOUR CODE HERE ---
    pass


def get_ticker_info(ticker: str) -> dict:
    """
    Use yfinance to fetch the full info dictionary for the given ticker.
    Wrap the call in a try/except so that any error returns an empty dict.
    """
    # --- YOUR CODE HERE ---
    pass


def get_current_price(ticker: str) -> tuple[float, float]:
    """
    Return a tuple of (current_price, previous_close) for the ticker.
    Use yfinance's fast_info to get 'lastPrice' and 'previousClose'.
    Round both values to 2 decimal places.
    If previousClose raises KeyError, fall back to current price.
    If anything fails, return (0.0, 0.0).
    """
    # --- YOUR CODE HERE ---
    pass


async def get_current_prices_batch(
    tickers: list[str],
) -> dict[str, tuple[float, float]]:
    """
    Fetch prices for multiple tickers concurrently.
    Use asyncio.get_event_loop().run_in_executor to wrap the synchronous
    get_current_price call for each ticker.  Gather all results and return
    a dictionary mapping ticker → (current, previous_close).
    """
    # --- YOUR CODE HERE ---
    pass


def get_advanced_metrics(ticker: str) -> dict:
    """
    Fetch comprehensive fundamental data.
    Try Financial Modeling Prep (FMP) first if the API key is set.
    Fall back to yfinance info, computing metrics like P/E, margins,
    growth, and margin of safety (compared to analyst target).
    """
    base_ticker = ticker.split(".")[0]
    if base_ticker == "VISA":
        base_ticker = "V"

    # --- YOUR CODE HERE ---
    pass


def get_historical_data(ticker: str, period: str = "2y") -> dict:
    """
    Download daily historical OHLCV data for the ticker using yfinance.
    Return a dictionary with keys: 'dates', 'open', 'high', 'low',
    'close', 'volume'.  Each value is a list.
    If the data is empty, return {"error": "..."}.
    """
    # --- YOUR CODE HERE ---
    pass


def get_macro_news() -> list:
    """
    Fetch general market news.
    Try Finnhub first if the API key exists, then NewsAPI,
    and finally return a single fallback headline.
    Each article should be a dict with 'title', 'publisher', and 'link'.
    """
    # --- YOUR CODE HERE ---
    pass


def get_ticker_news(ticker: str, limit: int = 5) -> list:
    """
    Fetch the latest news articles for a specific ticker using yfinance.
    Return a list of dicts with 'title', 'publisher', and 'link'.
    Limit the result to `limit` articles.
    """
    # --- YOUR CODE HERE ---
    pass
