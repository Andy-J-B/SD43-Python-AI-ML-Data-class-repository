"""
STUDENT TEMPLATE - analysis/fundamental.py
Retrieve and score fundamental financial data for a ticker.
"""

from .. import data_client as dc


def analyze_fundamentals(ticker: str) -> dict:
    """
    Build a comprehensive dictionary of fundamental metrics for the ticker.
    Use dc.get_ticker_info and dc.get_advanced_metrics to get the raw data.
    Extract and structure these fields:
    - Company Name, Sector, Industry, Market Cap, Enterprise Value
    - Current Price, Previous Close
    - Trailing P/E, Forward P/E, Price to Book
    - Debt-to-Equity, Revenue Growth, Profit Margins
    - Free Cash Flow, 52-Week High/Low
    - Analyst Target, Recommendation, Margin of Safety
    Also compute the PEG Ratio if P/E and revenue growth are available.
    Format Dividend Yield as a percentage string.
    If advanced metrics contain an error, return {"error": "..."}.
    """
    # --- YOUR CODE HERE ---
    pass


def score_company(metrics: dict) -> dict:
    """
    Score a company on a 12-point scale based on fundamental data.
    Award points for:
    - Low P/E (undervalued) → up to 3 points
    - High profit margins → up to 3 points
    - Strong revenue growth → up to 3 points
    - High margin of safety → up to 3 points
    Return a dict with:
      'score': total points,
      'max_score': 12,
      'overall': "STRONG BUY" / "BUY" / "HOLD" / "SELL / AVOID",
      'details': dict explaining each sub-score.
    """
    # --- YOUR CODE HERE ---
    pass
