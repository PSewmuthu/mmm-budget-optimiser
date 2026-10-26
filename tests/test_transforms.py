import numpy as np
import pytest

from src.transforms import adstock, hill


def test_adstock_zero_decay_is_identity():
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(adstock(x, 0.0), x)


def test_adstock_carries_over():
    assert np.allclose(adstock([10, 0, 0], 0.5), [10, 5, 2.5])


def test_adstock_empty_input():
    assert np.array_equal(adstock([], 0.5), np.array([]))


@pytest.mark.parametrize("decay", [-0.1, 1.0, 1.5, np.nan, np.inf])
def test_adstock_rejects_invalid_decay(decay):
    with pytest.raises(ValueError):
        adstock([1, 2, 3], decay)


def test_adstock_rejects_negative_spend():
    with pytest.raises(ValueError):
        adstock([1, -2, 3], 0.5)


def test_adstock_rejects_multidimensional_input():
    with pytest.raises(ValueError):
        adstock([[1, 2], [3, 4]], 0.5)


def test_hill_half_saturation_and_monotone():
    assert np.isclose(hill(100.0, 100.0, 1.0), 0.5)
    y = hill(np.linspace(1, 1000, 50), 100.0, 1.5)
    assert np.all(np.diff(y) > 0)
    assert y.max() < 1


def test_hill_zero_spend_returns_zero():
    assert np.isclose(hill(0.0, 100.0), 0.0)


@pytest.mark.parametrize(
    "half_sat,slope",
    [(0, 1), (-1, 1), (100, 0), (100, -1), (np.nan, 1), (100, np.inf)],
)
def test_hill_rejects_invalid_parameters(half_sat, slope):
    with pytest.raises(ValueError):
        hill([1, 2, 3], half_sat, slope)


def test_hill_rejects_negative_spend():
    with pytest.raises(ValueError):
        hill([1, -2, 3], 100.0)