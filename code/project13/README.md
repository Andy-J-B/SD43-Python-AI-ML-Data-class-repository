# Stock AI Analyzer — Student Project

A comprehensive stock analysis tool with AI/ML capabilities: LSTM price prediction, news sentiment analysis, technical indicators, fundamental scoring, portfolio optimization, and correlation analysis.

## Getting Started

```bash
pip install -r requirements.txt
```

## Project Structure

```
project13/
├── main.py                  # CLI entry point (argparse) — mostly provided
├── data_client.py           # Financial data API client — implement the fetchers
├── analysis/
│   ├── technical.py         # SMA, RSI, MACD, Bollinger Bands, trend
│   ├── fundamental.py       # P/E, margins, growth scoring
│   ├── risk.py              # VaR, Sharpe, drawdown, beta/alpha
│   ├── portfolio.py         # Monte Carlo efficient frontier
│   └── correlation.py       # Pearson/Spearman matrices, rolling correlation
├── models/
│   ├── sentiment_analyzer.py # VADER lexicon news sentiment
│   └── lstm_predictor.py     # PyTorch LSTM price forecasting
├── reports/
│   ├── terminal.py           # Rich-formatted terminal output
│   └── pdf_report.py         # FPDF + Matplotlib PDF generation
└── utils/
    └── helpers.py            # Date parsing, currency formatting
```

## Commands (run from repo root)

| Command | Example | Description |
|---------|---------|-------------|
| `analyze` | `analyze AAPL` | Technical + fundamental + risk analysis |
| `predict` | `predict AAPL --days 30` | LSTM neural network price forecast |
| `sentiment` | `sentiment AAPL --limit 10` | News headline sentiment scoring |
| `optimize` | `optimize AAPL MSFT GOOGL` | Portfolio optimization |
| `correlate` | `correlate AAPL MSFT` | Correlation analysis |
| `report` | `report AAPL --include-prediction` | Full PDF report |

Run with:
```
python -m code.project13.main analyze AAPL
```

Add `--pdf` to any command to generate a PDF report.

## Your Task

Each file contains function stubs with prose comments describing what to implement.
Replace every `pass` with your own code.  Do not change function names or signatures.
