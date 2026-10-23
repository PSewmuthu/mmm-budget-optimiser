"""M2: adstock and saturation transforms (reference implementations; M2 extends + tests)."""
import numpy as np


def adstock(x, decay: float) -> np.ndarray:
    """Geometric adstock: a_t = x_t + decay * a_{t-1}. 0 <= decay < 1."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    carry = 0.0
    for i, v in enumerate(x):
        carry = v + decay * carry
        out[i] = carry
    return out


def hill(x, half_sat: float, slope: float = 1.0) -> np.ndarray:
    """Hill saturation in [0, 1): x^s / (x^s + k^s). Diminishing returns for slope >= ~1."""
    x = np.asarray(x, dtype=float)
    return x**slope / (x**slope + half_sat**slope)
