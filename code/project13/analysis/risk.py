"""
STUDENT TEMPLATE - analysis/risk.py
Risk metrics: volatility, Value at Risk, Conditional VaR,
maximum drawdown, Sharpe / Sortino ratios, beta, alpha.
"""

import numpy as np
import math
from typing import Optional


def compute_returns(prices: list[float]) -> list[float]:
    """
    Convert a list of prices into a list of daily returns.
    Each return = (price[i] - price[i-1]) / price[i-1].
    The result has one fewer element than the input.
    """
    # --- YOUR CODE HERE ---
    pass


def annualized_volatility(returns: list[float], trading_days: int = 252) -> float:
    """
    Compute the annualized volatility (standard deviation of returns
    multiplied by the square root of trading_days).  Return as a percentage.
    Use np.std with ddof=1 for sample standard deviation.
    """
    # --- YOUR CODE HERE ---
    pass


def value_at_risk(returns: list[float], confidence: float = 0.95) -> float:
    """
    Calculate the Value at Risk at the given confidence level.
    Use np.percentile to find the return at the (1-confidence) percentile.
    Return the result as a percentage (multiply by 100).
    """
    # --- YOUR CODE HERE ---
    pass


def conditional_var(returns: list[float], confidence: float = 0.95) -> float:
    """
    Calculate Conditional VaR (Expected Shortfall).
    Find the VaR threshold, then average all returns that are <= that threshold.
    Return the result as a percentage.
    """
    # --- YOUR CODE HERE ---
    pass


def maximum_drawdown(prices: list[float]) -> float:
    """
    Compute the maximum percentage drawdown from a peak.
    Track the highest price seen so far.  At each point, calculate
    (peak - current) / peak.  Keep the maximum value and return as a percentage.
    """
    # --- YOUR CODE HERE ---
    pass


def sharpe_ratio(
    returns: list[float],
    risk_free_rate: float = 0.05,
    trading_days: int = 252,
) -> float:
    """
    Compute the Sharpe Ratio:
        (annualized_return - risk_free_rate) / annualized_volatility
    Use np.mean(returns) * trading_days for annualized return.
    """
    # --- YOUR CODE HERE ---
    pass


def sortino_ratio(
    returns: list[float],
    risk_free_rate: float = 0.05,
    trading_days: int = 252,
) -> float:
    """
    Compute the Sortino Ratio, which is like the Sharpe Ratio but
    only penalises negative (downside) volatility.
    Filter returns to only negative values and compute their std.
    """
    # --- YOUR CODE HERE ---
    pass


def beta(returns: list[float], market_returns: list[float]) -> float:
    """
    Calculate the beta of the stock relative to the market.
    Beta = covariance(stock, market) / variance(market).
    Use np.cov to get the covariance matrix, then extract [0][1].
    """
    # --- YOUR CODE HERE ---
    pass


def alpha(
    returns: list[float],
    market_returns: list[float],
    risk_free_rate: float = 0.05,
    trading_days: int = 252,
) -> float:
    """
    Calculate Jensen's Alpha:
        alpha = portfolio_return - [risk_free_rate + beta * (market_return - risk_free_rate)]
    Return as a percentage (multiply by 100).
    """
    # --- YOUR CODE HERE ---
    pass


def calculate_calmar_ratio(returns: list[float], prices: list[float]) -> float:
    """
    Calmar Ratio = annualized return / maximum drawdown (as a decimal, not %).
    """
    # --- YOUR CODE HERE ---
    pass


def comprehensive_risk_analysis(prices: list[float], market_prices: Optional[list[float]] = None) -> dict:
    """
    Run all risk calculations and return a dictionary of results.
    Include: Volatility, VaR (95% and 99%), CVaR, Max Drawdown,
    Sharpe, Sortino, Calmar, Beta, and Alpha.
    If market_prices is None, set Beta and Alpha to "N/A".
    """
    # --- YOUR CODE HERE ---
    pass
