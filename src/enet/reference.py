"""NumPy reference implementation of the sparse ElasticNet solver.

Maps 1-1 onto docs/theory.html.

Objective (scikit-learn convention):
    f(w, b) = 1/(2N) * ||y - Xw - b||^2
              + alpha * l1_ratio * ||w||_1
              + 0.5 * alpha * (1 - l1_ratio) * ||w||^2

`X` is either a dense ndarray (N, D) or a scipy.sparse matrix (converted to CSC).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse as sp


def _is_sparse(X):
    return sp.issparse(X)


def _column_stats(X, fit_intercept):
    """Column sums, column means and centered squared norms.

    z_c[j] = ||x_j - mean(x_j)||^2 = ||x_j||^2 - N * mean(x_j)^2.
    When fit_intercept is False, no centering is applied.
    """
    if _is_sparse(X):
        Xc = X.tocsc()
        colsum = np.asarray(Xc.sum(axis=0)).ravel()
        colsq = np.asarray(Xc.multiply(Xc).sum(axis=0)).ravel()
    else:
        Xd = np.asarray(X, dtype=np.float64)
        colsum = Xd.sum(axis=0)
        colsq = np.einsum("ij,ij->j", Xd, Xd)

    n_samples = X.shape[0]
    if fit_intercept:
        xbar = colsum / n_samples
        z_c = colsq - n_samples * xbar ** 2
    else:
        xbar = np.zeros_like(colsum)
        z_c = colsq.copy()
    return colsum, xbar, np.maximum(z_c, 0.0)


def _kkt_from_gradient(v, w, lambda1):
    """Per-coordinate KKT violation given the smooth-part gradient v."""
    pos = w > 0
    neg = w < 0
    zero = ~(pos | neg)
    viol = np.empty_like(w, dtype=np.float64)
    viol[pos] = np.abs(v[pos] + lambda1)
    viol[neg] = np.abs(v[neg] - lambda1)
    viol[zero] = np.maximum(np.abs(v[zero]) - lambda1, 0.0)
    return viol


def soft_threshold(z, gamma):
    """S(z, gamma) = sign(z) * max(|z| - gamma, 0).  [theory Section 5]"""
    z = np.asarray(z, dtype=np.float64)
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)


def objective(X, y, w, b, alpha, l1_ratio):
    """ElasticNet objective f(w).  [theory Section 4]"""
    lambda1 = alpha * l1_ratio
    lambda2 = alpha * (1.0 - l1_ratio)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    r = y - (X @ w) - b
    n_samples = X.shape[0]
    return float(
        (r @ r) / (2.0 * n_samples)
        + lambda1 * np.abs(w).sum()
        + 0.5 * lambda2 * (w @ w)
    )


def alpha_max(X, y, l1_ratio, fit_intercept=True):
    """Smallest alpha whose solution is exactly zero.  [theory Section 8]

    lambda1_max = max_j |x_j^T y_c| / N, with y_c centered when fit_intercept.
    alpha_max = lambda1_max / l1_ratio.  For l1_ratio == 0 (pure Ridge) the
    solution is never exactly zero, so infinity is returned.
    """
    if l1_ratio <= 0.0:
        return np.inf
    y = np.asarray(y, dtype=np.float64)
    yc = y - y.mean() if fit_intercept else y
    Xt_yc = np.asarray(X.T @ yc).ravel()
    n_samples = X.shape[0]
    lambda1_max = np.max(np.abs(Xt_yc)) / n_samples
    return float(lambda1_max / l1_ratio)


def kkt_violation(X, y, w, b, alpha, l1_ratio, fit_intercept=True):
    """max_j violation of the per-coordinate KKT conditions.  [theory Section 7]"""
    lambda1 = alpha * l1_ratio
    lambda2 = alpha * (1.0 - l1_ratio)
    n_samples = X.shape[0]
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    r = y - (X @ w) - b
    if fit_intercept:
        r = r - r.mean()
    Xt_r = np.asarray(X.T @ r).ravel()
    v = -(Xt_r / n_samples) + lambda2 * w
    return float(np.max(_kkt_from_gradient(v, w, lambda1))) if w.size else 0.0


def duality_gap(X, y, w, b, alpha, l1_ratio, fit_intercept=True):
    """Primal-dual gap, for fair comparison with scikit-learn.  [theory Section 7]

    Uses theta = -r/N. For plain Lasso (l1_ratio == 1) theta is rescaled into
    the dual feasible set so the returned value stays a valid upper bound.
    """
    lambda1 = alpha * l1_ratio
    lambda2 = alpha * (1.0 - l1_ratio)
    n_samples = X.shape[0]
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    r = y - (X @ w) - b
    if fit_intercept:
        r = r - r.mean()

    primal = objective(X, y, w, b, alpha, l1_ratio)
    theta = -r / n_samples

    def _dual(theta_vec):
        return float(-theta_vec @ y - 0.5 * n_samples * (theta_vec @ theta_vec))

    if lambda2 > 0.0:
        z = (np.asarray(X.T @ r).ravel()) / n_samples
        h_star = np.sum(np.maximum(np.abs(z) - lambda1, 0.0) ** 2) / (2.0 * lambda2)
        dual = _dual(theta) - float(h_star)
    else:
        Xt_theta = np.asarray(X.T @ theta).ravel()
        z_max = np.max(np.abs(Xt_theta)) if Xt_theta.size else 0.0
        if z_max > lambda1 > 0.0:
            theta = theta * (lambda1 / z_max)
        dual = _dual(theta)

    return float(primal - dual)


def _cd_dense(X, y, lambda1, lambda2, colsum, xbar, z_c, w, b,
              tol, max_iter):
    n_samples, n_features = X.shape
    r_raw = y - (X @ w)

    for _ in range(max_iter):
        for j in range(n_features):
            xj = X[:, j]
            rho = float(xj @ r_raw) - b * n_samples * xbar[j] + w[j] * z_c[j]
            denom = z_c[j] + n_samples * lambda2
            w_new = float(soft_threshold(rho, n_samples * lambda1)) / denom if denom > 0.0 else 0.0
            delta = w_new - w[j]
            if delta != 0.0:
                r_raw -= delta * xj
                b -= xbar[j] * delta
                w[j] = w_new

        Xt_r = X.T @ r_raw - b * colsum
        v = -(Xt_r / n_samples) + lambda2 * w
        if np.max(_kkt_from_gradient(v, w, lambda1)) <= tol:
            break

    return w, float(b)


def _cd_sparse(X, y, lambda1, lambda2, colsum, xbar, z_c, w, b,
               tol, max_iter):
    Xc = X.tocsc()
    n_samples, n_features = Xc.shape
    data, indices, indptr = Xc.data, Xc.indices, Xc.indptr
    r_raw = y - (Xc @ w)

    for _ in range(max_iter):
        for j in range(n_features):
            lo, hi = indptr[j], indptr[j + 1]
            idx = indices[lo:hi]
            d = data[lo:hi]

            xj_r = float(d @ r_raw[idx]) if d.size else 0.0
            rho = xj_r - b * n_samples * xbar[j] + w[j] * z_c[j]
            denom = z_c[j] + n_samples * lambda2
            w_new = float(soft_threshold(rho, n_samples * lambda1)) / denom if denom > 0.0 else 0.0
            delta = w_new - w[j]
            if delta != 0.0:
                r_raw[idx] -= delta * d
                b -= xbar[j] * delta
                w[j] = w_new

        Xt_r = np.asarray(Xc.T @ r_raw).ravel() - b * colsum
        v = -(Xt_r / n_samples) + lambda2 * w
        if np.max(_kkt_from_gradient(v, w, lambda1)) <= tol:
            break

    return w, float(b)


def coordinate_descent(X, y, alpha, l1_ratio, *, tol=1e-4, max_iter=1000,
                       fit_intercept=True, w0=None, b0=None):
    """Solve ElasticNet by coordinate descent.  [theory Section 6-7]

    lambda1 = alpha * l1_ratio, lambda2 = alpha * (1 - l1_ratio).
    Per coordinate j:
        z_j   = ||x_j||^2
        rho_j = x_j^T r + w_j * z_j
        w_j  <- S(rho_j, N * lambda1) / (z_j + N * lambda2)
        r    <- r - (w_j_new - w_j) * x_j
    Sweep j = 0..D-1 until max_j kkt_violation_j <= tol.

    Returns (w, b). `w0`, `b0` enable warm start. The intercept is never
    penalized and is kept consistent as b = mean(y) - mean(X) @ w.
    """
    if not 0.0 <= l1_ratio <= 1.0:
        raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio}")
    if alpha < 0.0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")

    y = np.asarray(y, dtype=np.float64).ravel()
    n_samples, n_features = X.shape
    lambda1 = alpha * l1_ratio
    lambda2 = alpha * (1.0 - l1_ratio)

    colsum, xbar, z_c = _column_stats(X, fit_intercept)
    ybar = float(y.mean()) if fit_intercept else 0.0

    w = np.zeros(n_features, dtype=np.float64) if w0 is None else np.array(w0, dtype=np.float64)
    if b0 is not None:
        b = float(b0)
    elif fit_intercept:
        b = ybar - float(xbar @ w)
    else:
        b = 0.0

    solver = _cd_sparse if _is_sparse(X) else _cd_dense
    return solver(X, y, lambda1, lambda2, colsum, xbar, z_c, w, b,
                  tol, max_iter)


def fit_path(X, y, *, n_alphas=20, l1_ratio=0.5, eps=1e-3, fit_intercept=True,
             **kwargs):
    """Warm-started regularization path from alpha_max downward.  [theory Section 8]

    Returns (alphas, W) with alphas descending and W of shape (D, n_alphas).
    """
    if l1_ratio <= 0.0:
        raise ValueError("fit_path needs l1_ratio > 0 (alpha_max is infinite for Ridge)")

    a_max = alpha_max(X, y, l1_ratio, fit_intercept=fit_intercept)
    alphas = np.geomspace(a_max, a_max * eps, n_alphas)
    n_features = X.shape[1]
    W = np.zeros((n_features, n_alphas), dtype=np.float64)

    w = None
    b = None
    for i, alpha in enumerate(alphas):
        w, b = coordinate_descent(
            X, y, float(alpha), l1_ratio, fit_intercept=fit_intercept,
            w0=w, b0=b, **kwargs
        )
        W[:, i] = w

    return alphas, W
