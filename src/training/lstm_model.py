"""
src/training/lstm_model.py
============================
NDVI time-series forecasting model built on PyTorch LSTM.

Architecture
------------
  Input:  (batch, seq_len=8, input_size=1)  ← weekly NDVI scalars
  LSTM:   2 layers, hidden=64, dropout=0.2
  FC head: Linear(64→32) → ReLU → Linear(32→2)
  Output: (batch, 2)  ← next 2-week NDVI forecast

Usage
-----
    from src.training.lstm_model import NDVIForecaster
    model = NDVIForecaster()
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NDVIForecaster(nn.Module):
    """
    Two-layer LSTM that forecasts 2 future NDVI values from an 8-step history.

    Parameters
    ----------
    input_size : Number of features per time step (default 1 — scalar NDVI).
    hidden     : Number of LSTM hidden units per layer.
    layers     : Number of stacked LSTM layers.
    dropout    : Dropout probability applied between LSTM layers (ignored when
                 layers=1).
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden:     int = 64,
        layers:     int = 2,
        dropout:    float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.layers = layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor of shape ``(batch, seq_len, input_size)``.

        Returns
        -------
        Tensor of shape ``(batch, 2)`` — 2-step NDVI forecast.
        """
        out, (h, c) = self.lstm(x)       # out: (batch, seq_len, hidden)
        return self.fc(out[:, -1, :])    # use last time step's hidden state


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = NDVIForecaster()
    dummy = torch.randn(16, 8, 1)        # batch=16, seq=8, features=1
    pred  = model(dummy)
    print(f"Input  : {tuple(dummy.shape)}")
    print(f"Output : {tuple(pred.shape)}")
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params : {total:,} total  |  {trainable:,} trainable")
    print("✓ lstm_model.py OK")
