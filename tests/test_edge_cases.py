import numpy as np
import pytest
from scipy import sparse

from enet import reference


def test_n_less_than_d_runs_and_is_finite():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(50, 300))
    y = X @ rng.normal(size=300) + 0.1 * rng.normal(size=50)

    w, b = reference.coordinate_descent(X, y, 0.01, 0.5, tol=1e-8, max_iter=10_000)

    assert np.all(np.isfinite(w))
    assert np.isfinite(b)


def test_zero_column_gets_zero_coefficient():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(200, 20))
    X[:, 5] = 0.0
    y = rng.normal(size=200)

    w, _ = reference.coordinate_descent(X, y, 0.05, 0.5, tol=1e-8, max_iter=10_000)

    assert w[5] == pytest.approx(0.0, abs=1e-10)


def test_constant_y_gives_zero_coefficients():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(200, 20))
    y = np.full(200, 3.0)

    w, _ = reference.coordinate_descent(X, y, 0.05, 0.5, tol=1e-8, max_iter=10_000)

    np.testing.assert_allclose(w, 0.0, atol=1e-6)


def test_duplicate_columns_stay_finite():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(200, 20))
    X[:, 1] = X[:, 0]
    y = rng.normal(size=200)

    w, _ = reference.coordinate_descent(X, y, 0.05, 1.0, tol=1e-8, max_iter=10_000)

    assert np.all(np.isfinite(w))


def test_small_max_iter_returns():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(200, 20))
    y = rng.normal(size=200)

    w, b = reference.coordinate_descent(X, y, 0.5, 0.5, max_iter=1)

    assert w.shape == (20,)
    assert np.isfinite(b)


def test_sparse_csc_input_accepted():
    rng = np.random.default_rng(15)
    X = sparse.random(100, 200, density=0.05, format="csc",
                      random_state=rng, data_rvs=lambda k: rng.normal(size=k))
    y = rng.normal(size=100)

    w, _ = reference.coordinate_descent(X, y, 0.02, 0.5, tol=1e-8, max_iter=10_000)

    assert w.shape == (200,)
