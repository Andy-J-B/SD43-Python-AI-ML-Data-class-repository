"""
STUDENT TEMPLATE - analysis/portfolio.py
Portfolio optimization using Modern Portfolio Theory (Markowitz).
Monte Carlo simulation to find the efficient frontier.
"""

import numpy as np
import math
from typing import Optional
from .. import data_client as dc


def fetch_returns(tickers: list[str], period: str = "1y") -> dict[str, Optional[list[float]]]:
    """
    Download historical data for each ticker and compute daily returns.
    Use dc.get_historical_data for each ticker.
    If the data contains an error key, store None for that ticker.
    Otherwise compute percentage returns from the close prices.
    """
    # --- YOUR CODE HERE ---
    pass


def build_return_matrix(
    tickers: list[str],
    returns_dict: dict[str, Optional[list[float]]],
) -> tuple[np.ndarray, list[str]]:
    """
    Build a 2D numpy array from the returns dictionary, keeping only
    tickers that have valid data.  Trim all arrays to the same minimum
    length.  Return (matrix, valid_tickers).
    """
    # --- YOUR CODE HERE ---
    pass


def annualized_return(returns: np.ndarray, trading_days: int = 252) -> float:
    """
    Compute the annualized return from an array of daily returns.
    Multiply the mean return by trading_days.
    """
    # --- YOUR CODE HERE ---
    pass


def annualized_covariance(returns: np.ndarray, trading_days: int = 252) -> np.ndarray:
    """
    Compute the annualized covariance matrix.
    Use np.cov with ddof=0 and multiply by trading_days.
    """
    # --- YOUR CODE HERE ---
    pass


def monte_carlo_portfolios(
    tickers: list[str],
    returns_dict: dict[str, Optional[list[float]]],
    num_portfolios: int = 10000,
) -> dict:
    """
    Run a Monte Carlo simulation to find optimal portfolios.
    For each iteration:
      1. Generate random weights that sum to 1.
      2. Compute portfolio return = dot(weights, mean_returns).
      3. Compute portfolio std = sqrt(weights^T * covariance * weights).
      4. Compute Sharpe ratio = (return - 0.05) / std.
    Track all results and find:
      - The portfolio with the maximum Sharpe ratio.
      - The portfolio with the minimum volatility.
    Return a dict with the efficient frontier data and optimal portfolios.
    """
    # --- YOUR CODE HERE ---
    pass


def equal_weight_portfolio(tickers: list[str], returns_dict: dict) -> dict:
    """
    Calculate metrics for an equally-weighted portfolio.
    Each asset gets weight = 1 / number_of_assets.
    Compute return, volatility, and Sharpe ratio.
    """
    # --- YOUR CODE HERE ---
    pass
