
import numpy as np
import pytest

from src.transforms import (
    adstock,
    delayed_peak_adstock,
    hill,
    negative_exponential,
)


# ---------------------------------------------------------
# Geometric adstock tests
# ---------------------------------------------------------

def test_adstock_zero_decay_is_identity():
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(adstock(x, 0.0), x)


def test_adstock_carries_over():
    assert np.allclose(
        adstock([10, 0, 0], 0.5),
        [10, 5, 2.5],
    )


def test_adstock_empty_input():
    assert np.array_equal(
        adstock([], 0.5),
        np.array([]),
    )


@pytest.mark.parametrize(
    "decay",
    [-0.1, 1.0, 1.5, np.nan, np.inf],
)
def test_adstock_rejects_invalid_decay(decay):
    with pytest.raises(ValueError):
        adstock([1, 2, 3], decay)


def test_adstock_rejects_negative_spend():
    with pytest.raises(ValueError):
        adstock([1, -2, 3], 0.5)


def test_adstock_rejects_multidimensional_input():
    with pytest.raises(ValueError):
        adstock([[1, 2], [3, 4]], 0.5)


# ---------------------------------------------------------
# Delayed-peak adstock tests
# ---------------------------------------------------------

def test_delayed_peak_adstock_has_delayed_peak():
    result = delayed_peak_adstock(
        [100, 0, 0],
        decay=0.5,
        peak_delay=1,
    )

    assert np.argmax(result) == 1
    assert np.allclose(result, [50, 100, 50])


def test_delayed_peak_adstock_empty_input():
    result = delayed_peak_adstock(
        [],
        decay=0.5,
        peak_delay=1,
    )

    assert np.array_equal(result, np.array([]))


@pytest.mark.parametrize(
    "decay",
    [0, 1, -0.1, np.nan, np.inf],
)
def test_delayed_peak_adstock_rejects_invalid_decay(decay):
    with pytest.raises(ValueError):
        delayed_peak_adstock([1, 2, 3], decay)


def test_delayed_peak_adstock_rejects_negative_peak_delay():
    with pytest.raises(ValueError):
        delayed_peak_adstock(
            [1, 2, 3],
            decay=0.5,
            peak_delay=-1,
        )


def test_delayed_peak_adstock_rejects_non_integer_peak_delay():
    with pytest.raises(ValueError):
        delayed_peak_adstock(
            [1, 2, 3],
            decay=0.5,
            peak_delay=1.5,
        )


# ---------------------------------------------------------
# Hill saturation tests
# ---------------------------------------------------------

def test_hill_half_saturation_and_monotone():
    assert np.isclose(
        hill(100.0, 100.0, 1.0),
        0.5,
    )

    y = hill(
        np.linspace(1, 1000, 50),
        half_sat=100.0,
        slope=1.5,
    )

    assert np.all(np.diff(y) > 0)
    assert y.max() < 1


def test_hill_zero_spend_returns_zero():
    assert np.isclose(
        hill(0.0, 100.0),
        0.0,
    )


def test_hill_supports_scalar_and_array_inputs():
    scalar_result = hill(100.0, 100.0, 1.0)
    array_result = hill(
        np.array([0.0, 100.0, 200.0]),
        100.0,
        1.0,
    )

    assert np.isscalar(scalar_result)
    assert array_result.shape == (3,)
    assert np.isclose(array_result[1], 0.5)


@pytest.mark.parametrize(
    "half_sat,slope",
    [
        (0, 1),
        (-1, 1),
        (100, 0),
        (100, -1),
        (np.nan, 1),
        (100, np.inf),
    ],
)
def test_hill_rejects_invalid_parameters(half_sat, slope):
    with pytest.raises(ValueError):
        hill([1, 2, 3], half_sat, slope)


def test_hill_rejects_negative_spend():
    with pytest.raises(ValueError):
        hill([1, -2, 3], 100.0)


# ---------------------------------------------------------
# Negative-exponential saturation tests
# ---------------------------------------------------------

def test_negative_exponential_reference_value():
    result = negative_exponential(100.0, 100.0)

    assert np.isclose(
        result,
        1.0 - np.exp(-1.0),
    )


def test_negative_exponential_zero_spend():
    assert np.isclose(
        negative_exponential(0.0, 100.0),
        0.0,
    )


def test_negative_exponential_is_monotone_and_bounded():
    x = np.array([0, 10, 50, 100, 500])

    result = negative_exponential(x, scale=100.0)

    assert np.all(np.diff(result) > 0)
    assert np.all(result >= 0)
    assert np.all(result < 1)


def test_negative_exponential_supports_scalar_and_array_inputs():
    scalar_result = negative_exponential(100.0, 100.0)

    array_result = negative_exponential(
        np.array([0.0, 100.0, 200.0]),
        scale=100.0,
    )

    assert np.isscalar(scalar_result)
    assert array_result.shape == (3,)
    assert np.isclose(array_result[1], 1.0 - np.exp(-1.0))


@pytest.mark.parametrize(
    "scale",
    [0, -1, np.nan, np.inf],
)
def test_negative_exponential_rejects_invalid_scale(scale):
    with pytest.raises(ValueError):
        negative_exponential([1, 2, 3], scale)


def test_negative_exponential_rejects_negative_spend():
    with pytest.raises(ValueError):
        negative_exponential([1, -2, 3], scale=100.0)


# ---------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "function",
    [
        lambda: adstock([1, np.nan, 3], 0.5),
        lambda: adstock([1, np.inf, 3], 0.5),
        lambda: hill([1, np.nan, 3], 100.0),
        lambda: negative_exponential([1, np.inf, 3], 100.0),
    ],
)
def test_transform_functions_reject_non_finite_spend(function):
    with pytest.raises(ValueError):
        function()
