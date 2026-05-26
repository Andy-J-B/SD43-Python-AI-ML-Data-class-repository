"""
STUDENT TEMPLATE - models/lstm_predictor.py
LSTM neural network for stock price prediction using PyTorch.
"""

import numpy as np
import os
from pathlib import Path
from typing import Optional
from .. import data_client as dc

MODEL_DIR = Path(__file__).parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

try:
    import torch
    import torch.nn as nn

    class PriceLSTM(nn.Module):
        """
        A simple LSTM model for time-series forecasting.
        Architecture:
          - LSTM layer(s) with configurable hidden_size and num_layers
          - Dropout between LSTM layers if num_layers > 1
          - A final linear (fully-connected) layer that outputs a single value
        The forward method should:
          1. Pass the input through the LSTM
          2. Take the last time-step's output
          3. Pass it through the linear layer
          4. Return the predicted value
        """

        def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2):
            super().__init__()
            # --- YOUR CODE HERE ---
            pass

        def forward(self, x):
            # --- YOUR CODE HERE ---
            pass

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _normalize(series: list[float]) -> tuple[np.ndarray, float, float]:
    """
    Normalize a list of prices to the range [-1, 1] using min-max scaling.
    Return (normalized_array, min_value, max_value).
    If all values are the same, avoid division by zero by making max = min + 1.
    """
    # --- YOUR CODE HERE ---
    pass


def _denormalize(normalized: np.ndarray, mn: float, mx: float) -> np.ndarray:
    """
    Reverse the normalization to get back original-scale prices.
    """
    # --- YOUR CODE HERE ---
    pass


def _create_sequences(
    data: np.ndarray,
    seq_length: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create input/output sequences for the LSTM.
    For each position i from seq_length to len(data):
      - X = data[i - seq_length : i]   (past seq_length values)
      - y = data[i]                     (the next value)
    Return (X, y) as numpy arrays of dtype float32.
    """
    # --- YOUR CODE HERE ---
    pass


def predict_price(
    ticker: str,
    days_to_predict: int = 30,
    seq_length: int = 60,
    epochs: int = 50,
    force_retrain: bool = False,
) -> dict:
    """
    Train (or load) an LSTM model for the given ticker and predict future prices.
    Steps:
      1. Fetch 2 years of historical data via dc.get_historical_data.
      2. Normalize the close prices.
      3. Create training sequences (80% train / 20% test split).
      4. Convert data to PyTorch tensors.
      5. If a saved model exists and force_retrain is False, load it.
         Otherwise, train a new PriceLSTM model.
      6. Use the trained model to predict `days_to_predict` future values
         by iteratively feeding the last known sequence.
      7. Denormalize all predictions back to original prices.
      8. Compute train/test RMSE and a prediction confidence score.
    Return a dict with ticker, last_price, future_dates, future_prices,
    train/test RMSE, confidence, trend, and the train/test actual vs predicted.
    If PyTorch is not available or data is insufficient, return an error dict.
    """
    # --- YOUR CODE HERE ---
    pass
