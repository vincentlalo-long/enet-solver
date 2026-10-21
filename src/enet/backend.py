"""Thin Python wrapper around the C++ coordinate-descent core."""
from __future__ import annotations

import numpy as np
from scipy import sparse as sp

from . import _core


def coordinate_descent_cpp(X, y, alpha, l1_ratio, *, tol=1e-4, max_iter=1000,
                           fit_intercept=True):
    """Same contract as reference.coordinate_descent, backed by C++."""
    if sp.issparse(X):
        Xc = X.tocsc()
    else:
        Xc = sp.csc_matrix(np.asarray(X, dtype=np.float64))

    y = np.asarray(y, dtype=np.float64).ravel()
    w, b = _core.coordinate_descent_csc(
        np.ascontiguousarray(Xc.data, dtype=np.float64),
        np.ascontiguousarray(Xc.indices, dtype=np.int64),
        np.ascontiguousarray(Xc.indptr, dtype=np.int64),
        y, float(alpha), float(l1_ratio), bool(fit_intercept),
        float(tol), int(max_iter),
    )
    return np.asarray(w, dtype=np.float64), float(b)
