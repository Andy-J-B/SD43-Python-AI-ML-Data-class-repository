"""
STUDENT TEMPLATE - analysis/correlation.py
Compute pairwise Pearson / Spearman correlations, rolling correlations,
and correlation matrices between tickers.
"""

import numpy as np
from typing import Optional
from .. import data_client as dc


def fetch_aligned_prices(tickers: list[str], period: str = "1y") -> dict[str, Optional[list[float]]]:
    """
    Fetch historical close prices for each ticker and trim them all to
    the same length (the minimum length among all tickers).
    Return a dict mapping ticker → list of close prices.
    """
    # --- YOUR CODE HERE ---
    pass


def compute_correlation_matrix(tickers: list[str], period: str = "1y") -> dict:
    """
    Compute the Pearson and Spearman correlation matrices for the given tickers.
    Steps:
      1. Fetch aligned prices.
      2. Compute daily returns from prices.
      3. Use np.corrcoef for Pearson correlation.
      4. For Spearman, rank the returns and compute correlation of the ranks.
    Return a dict containing:
      - 'tickers': list of valid tickers
      - 'pearson_matrix': 2D list of Pearson values
      - 'spearman_matrix': 2D list of Spearman values
      - 'pairs': list of dicts with ticker1, ticker2, pearson, spearman,
                 strength, direction (sorted by absolute Pearson descending)
      - 'highest_correlation' / 'lowest_correlation': extreme pairs
    """
    # --- YOUR CODE HERE ---
    pass


def compute_rolling_correlation(
    ticker1: str,
    ticker2: str,
    period: str = "2y",
    window: int = 30,
) -> dict:
    """
    Compute the rolling Pearson correlation between two tickers over
    a sliding window of `window` days.
    Steps:
      1. Fetch historical close prices for both tickers.
      2. Trim to the same length.
      3. Compute daily returns.
      4. For each position from `window` onward, compute the correlation
         of the returns in that window.
    Return a dict with average, min, max, latest, and the full list of
    rolling correlations with dates.
    """
    # --- YOUR CODE HERE ---
    pass


def _correlation_strength(r: float) -> str:
    """
    Classify a Pearson correlation coefficient as a descriptive string.
    >= 0.8 → "very strong"
    >= 0.6 → "strong"
    >= 0.4 → "moderate"
    >= 0.2 → "weak"
    < 0.2  → "very weak"
    """
    # --- YOUR CODE HERE ---
    pass
