"""
STUDENT TEMPLATE - analysis/technical.py
Technical analysis indicators: moving averages, RSI, MACD,
Bollinger Bands, support/resistance levels, and trend detection.
"""

import numpy as np
from typing import Optional


def compute_sma(data: list[float], window: int) -> list[Optional[float]]:
    """
    Compute the Simple Moving Average of `data` over the given `window`.
    For the first (window-1) positions there isn't enough data,
    so fill those with None.  For each position i from (window-1) onward,
    average the values from i-window+1 to i (inclusive) and round to 2 places.
    """
    # --- YOUR CODE HERE ---
    pass


def compute_ema(data: list[float], window: int) -> list[Optional[float]]:
    """
    Compute the Exponential Moving Average.
    Seed the first EMA value as the SMA of the first `window` points.
    Then for each subsequent value, apply the formula:
        ema = (data[i] - previous_ema) * multiplier + previous_ema
    where multiplier = 2 / (window + 1).
    Round each EMA to 2 decimal places.
    """
    # --- YOUR CODE HERE ---
    pass


def compute_rsi(data: list[float], window: int = 14) -> list[Optional[float]]:
    """
    Compute the Relative Strength Index over `window` periods.
    Start by calculating the average gain and average loss over the
    first `window` price changes.  From there, use the smoothed
    Wilder's method for subsequent values.  The RSI formula is:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    The first `window` entries should be None.
    """
    # --- YOUR CODE HERE ---
    pass


def compute_macd(
    data: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[list[Optional[float]], list[Optional[float]], list[Optional[float]]]:
    """
    Compute the MACD line, signal line, and histogram.
    MACD line = EMA(fast) - EMA(slow).
    Signal line = EMA of the MACD line (over `signal` periods).
    Histogram = MACD line - signal line.
    Return a tuple of three lists: (macd, signal, histogram).
    Positions where data is insufficient should be None.
    """
    # --- YOUR CODE HERE ---
    pass


def compute_bollinger_bands(
    data: list[float],
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[list[Optional[float]], list[Optional[float]], list[Optional[float]]]:
    """
    Compute Bollinger Bands.
    Middle band = SMA of `data` over `window`.
    Upper band = middle band + (num_std * standard deviation).
    Lower band = middle band - (num_std * standard deviation).
    Use np.std to compute the standard deviation of the window slice.
    Return (upper_band, middle_band, lower_band).
    """
    # --- YOUR CODE HERE ---
    pass


def detect_support_resistance(
    data: list[float],
    window: int = 20,
    threshold: float = 0.02,
) -> tuple[list[float], list[float]]:
    """
    Find local minima (support) and local maxima (resistance) in the price data.
    A point is a local minimum if it is lower than all points `window`
    positions to the left and right.  A local maximum if it is higher.
    Return (sorted_supports, sorted_resistances) with unique rounded values.
    """
    # --- YOUR CODE HERE ---
    pass


def compute_volume_trend(volume: list[int], window: int = 20) -> list[Optional[float]]:
    """
    Compute a simple moving average of the volume data.
    Reuse the compute_sma function on the volume (converted to float).
    """
    # --- YOUR CODE HERE ---
    pass


def analyze_trend(data_close: list[float]) -> str:
    """
    Determine the trend direction by comparing the current price
    to the 50-period and 200-period SMAs.
    If above both → "STRONG BULLISH"
    If below both → "STRONG BEARISH"
    If above 50 but below 200 (or no 200 available) → determine accordingly.
    If insufficient data (< 50 points) → "Insufficient data".
    """
    # --- YOUR CODE HERE ---
    pass


def compute_volume_price_divergence(
    close: list[float],
    volume: list[int],
    window: int = 14,
) -> list[Optional[str]]:
    """
    Detect divergence between price movement and volume trend.
    For each position from (2*window) onward, compare the price change
    over the last `window` periods to the average volume over the same window.
    Return a label: STRONG_BULLISH, WEAK_BULLISH, STRONG_BEARISH, or WEAK_BEARISH.
    """
    # --- YOUR CODE HERE ---
    pass
