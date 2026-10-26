"""M2: Adstock and Hill saturation transformations."""

import numpy as np


def adstock(x, decay: float) -> np.ndarray:
    """Apply geometric adstock: a[t] = x[t] + decay * a[t-1].

    Args:
        x: One-dimensional, finite, non-negative spend values in time order.
        decay: Carry-over fraction in [0, 1).

    Returns:
        NumPy array containing adstocked spend.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional array.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values.")
    if np.any(x < 0):
        raise ValueError("x must contain non-negative spend values.")
    if not np.isscalar(decay) or not np.isfinite(decay) or not 0 <= decay < 1:
        raise ValueError("decay must be finite and satisfy 0 <= decay < 1.")

    out = np.empty_like(x)
    carry = 0.0

    for i, value in enumerate(x):
        carry = value + decay * carry
        out[i] = carry

    return out


def hill(x, half_sat: float, slope: float = 1.0) -> np.ndarray:
    """Apply Hill saturation, returning values between 0 and 1.

    half_sat is the spend level at which saturation equals 0.5.
    slope controls the steepness of the response curve.
    """
    x = np.asarray(x, dtype=float)

    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values.")
    if np.any(x < 0):
        raise ValueError("x must contain non-negative spend values.")
    if not np.isscalar(half_sat) or not np.isfinite(half_sat) or half_sat <= 0:
        raise ValueError("half_sat must be finite and greater than zero.")
    if not np.isscalar(slope) or not np.isfinite(slope) or slope <= 0:
        raise ValueError("slope must be finite and greater than zero.")

    # Log-space calculation avoids unnecessary overflow for large inputs.
    with np.errstate(divide="ignore"):
        z = slope * (np.log(x) - np.log(half_sat))

    return np.exp(-np.logaddexp(0.0, -z))
