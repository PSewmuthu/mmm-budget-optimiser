import numpy as np

from src.transforms import adstock, hill


def test_adstock_zero_decay_is_identity():
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(adstock(x, 0.0), x)


def test_adstock_carries_over():
    assert np.allclose(adstock([10, 0, 0], 0.5), [10, 5, 2.5])


def test_hill_half_saturation_and_monotone():
    assert np.isclose(hill(100.0, 100.0, 1.0), 0.5)
    y = hill(np.linspace(1, 1000, 50), 100.0, 1.5)
    assert np.all(np.diff(y) > 0) and y.max() < 1
