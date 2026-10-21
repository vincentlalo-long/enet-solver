import numpy as np
import pytest
from scipy import sparse

from sklearn.linear_model import ElasticNet, Lasso

from enet import reference


def _solve(X, y, alpha, l1_ratio, tol=1e-10, max_iter=100_000):
    return reference.coordinate_descent(
        X, y, alpha, l1_ratio, tol=tol, max_iter=max_iter
    )


@pytest.mark.parametrize("l1_ratio", [0.1, 0.5, 1.0])
def test_matches_sklearn_dense(l1_ratio):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 30))
    w_true = rng.normal(size=30)
    y = X @ w_true + 0.1 * rng.normal(size=200)
    alpha = 0.05

    w, b = _solve(X, y, alpha, l1_ratio)

    sk = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True,
                    tol=1e-10, max_iter=100_000)
    sk.fit(X, y)

    np.testing.assert_allclose(w, sk.coef_, atol=1e-6)
    np.testing.assert_allclose(b, sk.intercept_, atol=1e-6)


def test_matches_lasso():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 40))
    y = X @ rng.normal(size=40) + 0.2 * rng.normal(size=150)
    alpha = 0.08

    w, b = _solve(X, y, alpha, 1.0)

    sk = Lasso(alpha=alpha, fit_intercept=True, tol=1e-10, max_iter=100_000)
    sk.fit(X, y)

    np.testing.assert_allclose(w, sk.coef_, atol=1e-6)
    np.testing.assert_allclose(b, sk.intercept_, atol=1e-6)


def test_matches_sklearn_sparse():
    rng = np.random.default_rng(2)
    X_sp = sparse.random(300, 500, density=0.02, format="csc",
                         random_state=rng, data_rvs=lambda k: rng.normal(size=k))
    w_true = rng.normal(size=500)
    y = np.asarray(X_sp @ w_true).ravel() + 0.1 * rng.normal(size=300)
    alpha = 0.02

    w, b = _solve(X_sp, y, alpha, 0.5)

    sk = ElasticNet(alpha=alpha, l1_ratio=0.5, fit_intercept=True,
                    tol=1e-10, max_iter=100_000)
    sk.fit(X_sp, y)

    np.testing.assert_allclose(w, sk.coef_, atol=1e-4)
    np.testing.assert_allclose(b, sk.intercept_, atol=1e-4)


def test_kkt_small_after_solve():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 50))
    y = X @ rng.normal(size=50) + 0.1 * rng.normal(size=200)
    alpha, l1_ratio = 0.1, 0.5

    w, b = _solve(X, y, alpha, l1_ratio)
    viol = reference.kkt_violation(X, y, w, b, alpha, l1_ratio)

    assert viol < 1e-6


def test_alpha_max_gives_zero_solution():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 30))
    y = rng.normal(size=200)
    l1_ratio = 0.7

    a_max = reference.alpha_max(X, y, l1_ratio)
    w, _ = reference.coordinate_descent(X, y, a_max, l1_ratio, tol=1e-10)

    np.testing.assert_allclose(w, 0.0, atol=1e-8)
