"""M6: budget optimiser. Maximise predicted revenue s.t. sum(spend) == budget and channel bounds.
Fixed budget => maximising revenue == maximising ROI."""
import numpy as np


def optimise(model, budget: float, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    raise NotImplementedError("M6: SLSQP multi-start + differential evolution cross-check")
