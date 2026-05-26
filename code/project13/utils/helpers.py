"""
STUDENT TEMPLATE - utils/helpers.py
Small utility functions used across the project.
"""

import re
from datetime import datetime, timedelta
from typing import Optional


def parse_date(date_str: str) -> Optional[datetime]:
    """
    Try to parse a date string using several common formats.
    Return a datetime object on success, or None if all formats fail.
    Check formats like YYYY-MM-DD, YYYY/MM/DD, MM/DD/YYYY, DD-MM-YYYY.
    """
    formats = ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d-%m-%Y"]
    # --- YOUR CODE HERE ---
    pass


def format_currency(value: float, symbol: str = "$") -> str:
    """
    Format a large number into a human-readable currency string.
    If the value is >= 1 billion, show it as X.XXB.
    If >= 1 million, show as X.XXM.
    If >= 1 thousand, show as X.XXK.
    Otherwise show as $X.XX.
    """
    # --- YOUR CODE HERE ---
    pass


def sanitize_filename(name: str) -> str:
    """
    Replace any character that is not a letter, digit, hyphen,
    underscore, or dot with an underscore.
    """
    # --- YOUR CODE HERE ---
    pass


def timeframe_to_days(timeframe: str) -> int:
    """
    Convert a yfinance timeframe string ('1d', '5d', '1mo', '1y', etc.)
    into an approximate number of days.  Use a simple lookup dictionary.
    Default to 730 (2 years) if the key is not found.
    """
    mapping = {
        "1d": 1, "5d": 5, "1mo": 30, "3mo": 90,
        "6mo": 180, "1y": 365, "2y": 730, "5y": 1825, "10y": 3650,
    }
    # --- YOUR CODE HERE ---
    pass


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Divide numerator by denominator.
    If denominator is zero, return the default value instead.
    """
    # --- YOUR CODE HERE ---
    pass
