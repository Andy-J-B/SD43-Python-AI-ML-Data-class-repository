"""
STUDENT TEMPLATE - reports/terminal.py
Terminal output formatting using the Rich library (with a plain-text fallback).
"""

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from typing import Optional


def _console():
    """Return a Rich Console if available, otherwise None."""
    if RICH_AVAILABLE:
        return Console()
    return None


def print_header(title: str):
    """
    Print a section header.
    With Rich: draw a horizontal rule with the title in cyan bold.
    Without Rich: print a line of '=' characters around the title.
    """
    # --- YOUR CODE HERE ---
    pass


def print_price_summary(ticker: str, current: float, prev_close: float, info: dict):
    """
    Print a price summary panel showing company name, ticker,
    current price, daily change (with colour: green for positive, red for negative),
    and previous close.
    """
    # --- YOUR CODE HERE ---
    pass


def print_table(title: str, data: dict):
    """
    Print a two-column key-value table (Metric | Value).
    With Rich: use a bordered table with the title.
    Without Rich: print each key-value pair on its own line.
    """
    # --- YOUR CODE HERE ---
    pass


def print_list(title: str, items: list[dict], keys: list[str], headers: Optional[list[str]] = None):
    """
    Print a multi-column table from a list of dictionaries.
    `keys` specifies which dictionary keys to display (in order).
    `headers` specifies the column headers (defaults to keys).
    """
    # --- YOUR CODE HERE ---
    pass


def print_positive(text: str):
    """Print text in green (or with a [+] prefix without Rich)."""
    # --- YOUR CODE HERE ---
    pass


def print_negative(text: str):
    """Print text in red (or with a [-] prefix without Rich)."""
    # --- YOUR CODE HERE ---
    pass


def print_neutral(text: str):
    """Print text in yellow (or with a [~] prefix without Rich)."""
    # --- YOUR CODE HERE ---
    pass


def print_info(text: str):
    """Print dim/info text (or plain text without Rich)."""
    # --- YOUR CODE HERE ---
    pass


def print_prediction_table(prediction: dict):
    """
    Print the LSTM prediction results in a formatted table.
    Show last price, predicted range, trend (coloured), confidence, RMSE values,
    and a list of future dates with predicted prices.
    """
    # --- YOUR CODE HERE ---
    pass
