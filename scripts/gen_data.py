"""Synthetic data generators for correctness tests and benchmarks.

Synthetic data is used for Gate 1 (controlled N, D, density, ground truth);
real sparse data is added later for the benchmark table.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def make_dense(n_samples=500, n_features=50, n_informative=10, noise=0.1,
               random_state=0):
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    w = np.zeros(n_features)
    idx = rng.choice(n_features, size=min(n_informative, n_features), replace=False)
    w[idx] = rng.normal(size=idx.size)
    y = X @ w + noise * rng.normal(size=n_samples)
    return X, y, w


def make_sparse(n_samples=500, n_features=20_000, density=0.01, n_informative=50,
                noise=0.1, random_state=0):
    rng = np.random.default_rng(random_state)
    X = sparse.random(n_samples, n_features, density=density, format="csc",
                      random_state=rng, data_rvs=lambda k: rng.normal(size=k))
    w = np.zeros(n_features)
    idx = rng.choice(n_features, size=min(n_informative, n_features), replace=False)
    w[idx] = rng.normal(size=idx.size)
    y = np.asarray(X @ w).ravel() + noise * rng.normal(size=n_samples)
    return X, y, w


def make_ill_conditioned(n_samples=200, n_features=100, n_informative=20,
                         noise=0.1, cond=1e6, random_state=0):
    rng = np.random.default_rng(random_state)
    Q, _ = np.linalg.qr(rng.normal(size=(n_features, n_features)))
    sv = np.geomspace(1.0, 1.0 / cond, n_features)
    X = rng.normal(size=(n_samples, n_features)) @ (Q * sv) @ Q.T
    w = np.zeros(n_features)
    idx = rng.choice(n_features, size=min(n_informative, n_features), replace=False)
    w[idx] = rng.normal(size=idx.size)
    y = X @ w + noise * rng.normal(size=n_samples)
    return X, y, w


if __name__ == "__main__":
    Xd, yd, wd = make_dense()
    print(f"dense:  X={Xd.shape}  nnz_ratio=1.0  w_true_nnz={np.count_nonzero(wd)}")
    Xs, ys, ws = make_sparse()
    print(f"sparse: X={Xs.shape}  density={Xs.nnz / (Xs.shape[0] * Xs.shape[1]):.4f}  "
          f"w_true_nnz={np.count_nonzero(ws)}")
    Xc, yc, wc = make_ill_conditioned()
    print(f"ill-cond: X={Xc.shape}  w_true_nnz={np.count_nonzero(wc)}")
