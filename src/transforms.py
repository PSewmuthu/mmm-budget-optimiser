
"""M2: Adstock and saturation transformations."""

import numpy as np


def _validate_spend(x, allow_scalar=False):
    """Validate finite, non-negative spend values."""
    x = np.asarray(x, dtype=float)

    if allow_scalar and x.ndim == 0:
        pass
    elif x.ndim != 1:
        raise ValueError("x must be a one-dimensional array.")

    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values.")
    if np.any(x < 0):
        raise ValueError("x must contain non-negative values.")

    return x


def adstock(x, decay: float) -> np.ndarray:
    """Apply geometric adstock: a[t] = x[t] + decay * a[t-1].

    Args:
        x: One-dimensional spend values in time order.
        decay: Carry-over fraction in [0, 1).

    Returns:
        NumPy array of adstocked spend.
    """
    x = _validate_spend(x)

    if (
        not np.isscalar(decay)
        or not np.isfinite(decay)
        or not 0 <= decay < 1
    ):
        raise ValueError("decay must be finite and satisfy 0 <= decay < 1.")

    out = np.empty_like(x)
    carry = 0.0

    for i, value in enumerate(x):
        carry = value + decay * carry
        out[i] = carry

    return out


def delayed_peak_adstock(
    x,
    decay: float,
    peak_delay: int = 1,
) -> np.ndarray:
    """Apply an exploratory delayed-peak carry-over kernel.

    The kernel weight is largest at peak_delay and declines
    geometrically with distance from that lag.
    """
    x = _validate_spend(x)

    if (
        not np.isscalar(decay)
        or not np.isfinite(decay)
        or not 0 < decay < 1
    ):
        raise ValueError("decay must be finite and satisfy 0 < decay < 1.")

    if (
        not isinstance(peak_delay, (int, np.integer))
        or isinstance(peak_delay, (bool, np.bool_))
        or peak_delay < 0
    ):
        raise ValueError("peak_delay must be a non-negative integer.")

    if len(x) == 0:
        return np.array([], dtype=float)

    lags = np.arange(len(x))
    weights = decay ** np.abs(lags - peak_delay)
    weights = weights / weights.max()

    return np.convolve(x, weights, mode="full")[:len(x)]


def hill(
    x,
    half_sat: float,
    slope: float = 1.0,
) -> np.ndarray:
    """Apply Hill saturation with a relative response in [0, 1].

    half_sat is the spend level where the response equals 0.5.
    slope controls the steepness of the curve.

    Scalar and one-dimensional inputs are supported.
    """
    x = _validate_spend(x, allow_scalar=True)

    if (
        not np.isscalar(half_sat)
        or not np.isfinite(half_sat)
        or half_sat <= 0
    ):
        raise ValueError(
            "half_sat must be finite and greater than zero."
        )

    if (
        not np.isscalar(slope)
        or not np.isfinite(slope)
        or slope <= 0
    ):
        raise ValueError(
            "slope must be finite and greater than zero."
        )

    # Log-space calculation improves numerical stability.
    with np.errstate(divide="ignore"):
        z = slope * (np.log(x) - np.log(half_sat))

    return np.exp(-np.logaddexp(0.0, -z))


def negative_exponential(
    x,
    scale: float,
) -> np.ndarray:
    """Apply negative-exponential saturation.

    response = 1 - exp(-x / scale)

    Scale must be positive. Scalar and one-dimensional inputs
    are supported.
    """
    x = _validate_spend(x, allow_scalar=True)

    if (
        not np.isscalar(scale)
        or not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError(
            "scale must be finite and greater than zero."
        )

    return -np.expm1(-x / scale)
