import numpy as np
import pytest
from scipy import sparse

import enet
from enet import reference

pytestmark = pytest.mark.skipif(
    enet.backend is None, reason="C++ core not built (run `pip install -e .`)"
)


def _data(kind):
    rng = np.random.default_rng(7)
    if kind == "dense":
        X = rng.normal(size=(200, 40))
    else:
        X = sparse.random(200, 400, density=0.03, format="csc",
                          random_state=rng, data_rvs=lambda k: rng.normal(size=k))
    y = rng.normal(size=200)
    return X, y


@pytest.mark.parametrize("kind", ["dense", "sparse"])
@pytest.mark.parametrize("l1_ratio", [0.5, 1.0])
def test_cpp_matches_reference(kind, l1_ratio):
    X, y = _data(kind)
    alpha, tol = 0.03, 1e-10

    w_ref, b_ref = reference.coordinate_descent(
        X, y, alpha, l1_ratio, tol=tol, max_iter=100_000)
    w_cpp, b_cpp = enet.backend.coordinate_descent_cpp(
        X, y, alpha, l1_ratio, tol=tol, max_iter=100_000)

    np.testing.assert_allclose(w_cpp, w_ref, atol=1e-9)
    np.testing.assert_allclose(b_cpp, b_ref, atol=1e-9)


def test_cpp_matches_sklearn():
    from sklearn.linear_model import ElasticNet

    rng = np.random.default_rng(8)
    X = rng.normal(size=(200, 30))
    y = X @ rng.normal(size=30) + 0.1 * rng.normal(size=200)
    alpha, l1_ratio = 0.05, 0.5

    w, b = enet.backend.coordinate_descent_cpp(
        X, y, alpha, l1_ratio, tol=1e-10, max_iter=100_000)
    sk = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True,
                    tol=1e-10, max_iter=100_000).fit(X, y)

    np.testing.assert_allclose(w, sk.coef_, atol=1e-6)
    np.testing.assert_allclose(b, sk.intercept_, atol=1e-6)


def test_cpp_rejects_bad_l1_ratio():
    X, y = _data("dense")
    with pytest.raises(ValueError):
        enet.backend.coordinate_descent_cpp(X, y, 0.1, 1.5)
