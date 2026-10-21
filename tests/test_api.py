import numpy as np
import pytest

from enet import reference


def test_fit_path_shape_and_descending_alphas():
    rng = np.random.default_rng(20)
    X = rng.normal(size=(200, 30))
    y = X @ rng.normal(size=30) + 0.1 * rng.normal(size=200)

    alphas, W = reference.fit_path(X, y, n_alphas=7, l1_ratio=0.5, tol=1e-8)

    assert alphas.shape == (7,)
    assert W.shape == (30, 7)
    assert np.all(np.diff(alphas) < 0)


def test_fit_path_last_alpha_matches_direct_solve():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(200, 30))
    y = X @ rng.normal(size=30) + 0.1 * rng.normal(size=200)

    alphas, W = reference.fit_path(X, y, n_alphas=5, l1_ratio=0.5, tol=1e-9)
    w, _ = reference.coordinate_descent(X, y, float(alphas[-1]), 0.5, tol=1e-9)

    np.testing.assert_allclose(W[:, -1], w, atol=1e-7)


def test_fit_path_requires_positive_l1_ratio():
    X = np.zeros((10, 3))
    y = np.zeros(10)
    with pytest.raises(ValueError):
        reference.fit_path(X, y, l1_ratio=0.0)


@pytest.mark.parametrize("l1_ratio", [0.3, 0.5, 1.0])
def test_duality_gap_small_at_solution(l1_ratio):
    rng = np.random.default_rng(22)
    X = rng.normal(size=(200, 30))
    y = X @ rng.normal(size=30) + 0.1 * rng.normal(size=200)
    alpha = 0.05

    w, b = reference.coordinate_descent(X, y, alpha, l1_ratio, tol=1e-10,
                                        max_iter=100_000)
    gap = reference.duality_gap(X, y, w, b, alpha, l1_ratio)

    assert gap >= -1e-9
    assert gap < 1e-8


def test_alpha_max_is_infinite_for_ridge():
    X = np.ones((10, 3))
    y = np.ones(10)
    assert reference.alpha_max(X, y, 0.0) == np.inf


def test_invalid_l1_ratio_raises():
    X = np.zeros((10, 3))
    y = np.zeros(10)
    with pytest.raises(ValueError):
        reference.coordinate_descent(X, y, 0.1, 1.5)
