"""
STUDENT TEMPLATE - main.py
CLI entry point for the Stock AI Analyzer.

This file is mostly provided for you — its job is to parse command-line
arguments and dispatch to the analysis functions you implement in the
other modules.  You do NOT need to modify this file unless you want to
add new commands or change the argument structure.
"""

import argparse
import sys
from typing import Optional

from . import data_client as dc
from .analysis import technical as tech
from .analysis import fundamental as fund
from .analysis import risk as risk_mod
from .analysis import portfolio as port
from .analysis import correlation as corr
from .models import sentiment_analyzer as sentiment
from .models import lstm_predictor as lstm
from .reports import terminal as term
from .reports import pdf_report as pdf


def cmd_analyze(args):
    """
    Run a full analysis (technical + fundamental + risk) for a ticker.
    Steps:
      1. Fetch the current price and ticker info.
      2. Print the price summary.
      3. Download 2 years of historical data.
      4. Compute technical indicators (RSI, MACD, trend, support/resistance).
      5. Print the technical analysis table.
      6. Fetch and print fundamental analysis + scoring.
      7. Compute and print risk metrics.
      8. If --pdf flag is set, generate and save a PDF report.
    """
    # --- YOUR CODE HERE ---


def cmd_predict(args):
    """
    Run the LSTM price prediction model for a ticker.
    Steps:
      1. Call lstm.predict_price with the ticker, days, and retrain flag.
      2. Print the prediction results table.
      3. If matplotlib is available, generate a price + prediction chart.
      4. If --pdf is set, generate a PDF report.
    """
    # --- YOUR CODE HERE ---


def cmd_sentiment(args):
    """
    Analyze news sentiment for a ticker.
    Steps:
      1. Call sentiment.analyze_ticker_sentiment with the ticker and limit.
      2. Print the overall sentiment (coloured by bullish/bearish/neutral).
      3. Print the article-by-article breakdown table.
      4. If --pdf is set and matplotlib is available, save a sentiment chart.
    """
    # --- YOUR CODE HERE ---


def cmd_optimize(args):
    """
    Run portfolio optimization on a set of tickers.
    Steps:
      1. Fetch returns for all tickers.
      2. Run the Monte Carlo portfolio simulation.
      3. Also compute the equal-weight portfolio.
      4. Print the equal-weight, max Sharpe, and min volatility portfolios.
      5. If --pdf is set, generate the efficient frontier chart and PDF report.
    """
    # --- YOUR CODE HERE ---


def cmd_correlate(args):
    """
    Compute pairwise correlations between tickers.
    Steps:
      1. Call corr.compute_correlation_matrix.
      2. Print the correlation matrix as a grid.
      3. Print the sorted pairwise results table.
      4. If --rolling is set, compute and print rolling correlation
         for the first two tickers.
      5. If --pdf is set, generate the heatmap and PDF report.
    """
    # --- YOUR CODE HERE ---


def cmd_report(args):
    """
    Generate a comprehensive PDF report with optional prediction and sentiment.
    Steps:
      1. Fetch price data and compute all metrics (technical, fundamental, risk).
      2. Optionally run the LSTM prediction.
      3. Optionally fetch news sentiment.
      4. Build price chart(s).
      5. Generate the PDF report.
    """
    # --- YOUR CODE HERE ---


def build_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser with all subcommands.
    Returns the configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Stock AI Analyzer - Comprehensive stock analysis with AI/ML capabilities",
    )
    parser.add_argument("--pdf", action="store_true", help="Generate PDF reports where applicable")
    sub = parser.add_subparsers(dest="command", required=True)
    analyze_p = sub.add_parser("analyze", help="Full technical + fundamental + risk analysis")
    analyze_p.add_argument("ticker", type=str, help="Stock ticker symbol")
    analyze_p.add_argument("--pdf", action="store_true", help="Generate PDF report")
    predict_p = sub.add_parser("predict", help="LSTM price prediction")
    predict_p.add_argument("ticker", type=str, help="Stock ticker symbol")
    predict_p.add_argument("--days", type=int, default=30, help="Days to predict (default: 30)")
    predict_p.add_argument("--retrain", action="store_true", help="Force retrain the model")
    predict_p.add_argument("--pdf", action="store_true", help="Generate PDF report")
    sentiment_p = sub.add_parser("sentiment", help="News sentiment analysis")
    sentiment_p.add_argument("ticker", type=str, help="Stock ticker symbol")
    sentiment_p.add_argument("--limit", type=int, default=10, help="Number of news articles (default: 10)")
    sentiment_p.add_argument("--pdf", action="store_true", help="Generate PDF report")
    optimize_p = sub.add_parser("optimize", help="Portfolio optimization (Markowitz)")
    optimize_p.add_argument("tickers", type=str, nargs="+", help="Stock ticker symbols")
    optimize_p.add_argument("--simulations", type=int, default=10000, help="Monte Carlo simulations (default: 10000)")
    optimize_p.add_argument("--pdf", action="store_true", help="Generate PDF report")
    correlate_p = sub.add_parser("correlate", help="Correlation analysis between tickers")
    correlate_p.add_argument("tickers", type=str, nargs="+", help="Stock ticker symbols")
    correlate_p.add_argument("--period", type=str, default="1y", help="Historical period (default: 1y)")
    correlate_p.add_argument("--rolling", action="store_true", help="Show rolling correlation for first two tickers")
    correlate_p.add_argument("--rolling-window", type=int, default=30, help="Rolling window in days (default: 30)")
    correlate_p.add_argument("--pdf", action="store_true", help="Generate PDF report")
    report_p = sub.add_parser("report", help="Generate comprehensive PDF report")
    report_p.add_argument("ticker", type=str, help="Stock ticker symbol")
    report_p.add_argument("--include-prediction", action="store_true", help="Include LSTM price prediction")
    report_p.add_argument("--predict-days", type=int, default=30, help="Days to predict (default: 30)")
    report_p.add_argument("--include-sentiment", action="store_true", help="Include news sentiment analysis")
    report_p.add_argument("--sentiment-limit", type=int, default=10, help="Number of news articles (default: 10)")
    return parser


def main():
    """
    Parse arguments and dispatch to the appropriate command handler.
    """
    parser = build_parser()
    args = parser.parse_args()
    cmd_map = {
        "analyze": cmd_analyze,
        "predict": cmd_predict,
        "sentiment": cmd_sentiment,
        "optimize": cmd_optimize,
        "correlate": cmd_correlate,
        "report": cmd_report,
    }
    cmd_fn = cmd_map.get(args.command)
    if cmd_fn:
        if hasattr(args, "pdf") and args.pdf:
            if not pdf.FPDF_AVAILABLE:
                term.print_neutral("Warning: fpdf2 not installed. PDF generation disabled.")
        cmd_fn(args)


if __name__ == "__main__":
    main()
