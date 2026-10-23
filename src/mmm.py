"""M3: Marketing Mix Model. Use by M5 (ROI) and M6 (optimiser).

M5/M6 must only call these methods, so they can start against a stub.
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class MMMModel(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "MMMModel": ...

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predicted revenue (LKR) per row of df (channel spends + controls)."""

    @abstractmethod
    def response_curve(self, channel: str, grid: np.ndarray) -> np.ndarray:
        """Incremental revenue (LKR) of `channel` at each monthly spend in grid,
        other channels held at historical mean."""

    @abstractmethod
    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-row decomposition: baseline, each channel, each control (LKR)."""
