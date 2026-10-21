// pybind11 C++ core: the coordinate-descent loop over a CSC matrix.
//
// Mirrors reference.coordinate_descent. Statistics (column sums, centered norms)
// and the intercept bookkeeping are recomputed here so the module is self-contained.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace py = pybind11;

namespace {

inline double soft_threshold(double z, double gamma) {
    if (z > gamma) return z - gamma;
    if (z < -gamma) return z + gamma;
    return 0.0;
}

inline double kkt_violation(double v, double w, double lambda1) {
    if (w > 0.0) return std::abs(v + lambda1);
    if (w < 0.0) return std::abs(v - lambda1);
    return std::max(std::abs(v) - lambda1, 0.0);
}

}  // namespace

py::tuple coordinate_descent_csc(
    py::array_t<double, py::array::c_style | py::array::forcecast> data_,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> indices_,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> indptr_,
    py::array_t<double, py::array::c_style | py::array::forcecast> y_,
    double alpha,
    double l1_ratio,
    bool fit_intercept,
    double tol,
    int max_iter) {

    auto data = data_.unchecked<1>();
    auto indices = indices_.unchecked<1>();
    auto indptr = indptr_.unchecked<1>();
    auto y = y_.unchecked<1>();

    const std::int64_t n_features = indptr.shape(0) - 1;
    const std::int64_t n_samples = y.shape(0);

    if (l1_ratio < 0.0 || l1_ratio > 1.0) {
        throw std::invalid_argument("l1_ratio must be in [0, 1]");
    }
    if (alpha < 0.0) {
        throw std::invalid_argument("alpha must be >= 0");
    }

    std::vector<double> colsum(n_features, 0.0);
    std::vector<double> colsq(n_features, 0.0);
    for (std::int64_t j = 0; j < n_features; ++j) {
        double s = 0.0;
        double sq = 0.0;
        for (std::int64_t k = indptr(j); k < indptr(j + 1); ++k) {
            const double v = data(k);
            s += v;
            sq += v * v;
        }
        colsum[j] = s;
        colsq[j] = sq;
    }

    std::vector<double> xbar(n_features, 0.0);
    std::vector<double> zc(n_features, 0.0);
    double ybar = 0.0;
    if (fit_intercept) {
        for (std::int64_t j = 0; j < n_features; ++j) {
            xbar[j] = colsum[j] / static_cast<double>(n_samples);
            zc[j] = colsq[j] - static_cast<double>(n_samples) * xbar[j] * xbar[j];
            if (zc[j] < 0.0) zc[j] = 0.0;
        }
        double sy = 0.0;
        for (std::int64_t i = 0; i < n_samples; ++i) sy += y(i);
        ybar = sy / static_cast<double>(n_samples);
    } else {
        for (std::int64_t j = 0; j < n_features; ++j) zc[j] = colsq[j];
    }

    const double lambda1 = alpha * l1_ratio;
    const double lambda2 = alpha * (1.0 - l1_ratio);
    const double n = static_cast<double>(n_samples);

    std::vector<double> w(n_features, 0.0);
    std::vector<double> r(n_samples);
    for (std::int64_t i = 0; i < n_samples; ++i) r[i] = y(i);
    double b = fit_intercept ? ybar : 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        for (std::int64_t j = 0; j < n_features; ++j) {
            const std::int64_t lo = indptr(j);
            const std::int64_t hi = indptr(j + 1);

            double xj_r = 0.0;
            for (std::int64_t k = lo; k < hi; ++k) {
                xj_r += data(k) * r[indices(k)];
            }

            const double rho = xj_r - b * n * xbar[j] + w[j] * zc[j];
            const double denom = zc[j] + n * lambda2;
            const double w_new = denom > 0.0
                ? soft_threshold(rho, n * lambda1) / denom
                : 0.0;
            const double delta = w_new - w[j];
            if (delta != 0.0) {
                for (std::int64_t k = lo; k < hi; ++k) {
                    r[indices(k)] -= delta * data(k);
                }
                b -= xbar[j] * delta;
                w[j] = w_new;
            }
        }

        double max_viol = 0.0;
        for (std::int64_t j = 0; j < n_features; ++j) {
            double xtr = 0.0;
            for (std::int64_t k = indptr(j); k < indptr(j + 1); ++k) {
                xtr += data(k) * r[indices(k)];
            }
            xtr -= b * colsum[j];
            const double v = -xtr / n + lambda2 * w[j];
            const double viol = kkt_violation(v, w[j], lambda1);
            if (viol > max_viol) max_viol = viol;
        }
        if (max_viol <= tol) break;
    }

    py::array_t<double> w_out(n_features);
    auto w_out_view = w_out.mutable_unchecked<1>();
    for (std::int64_t j = 0; j < n_features; ++j) w_out_view(j) = w[j];

    return py::make_tuple(w_out, b);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ coordinate-descent core for sparse ElasticNet";
    m.def("coordinate_descent_csc", &coordinate_descent_csc,
          py::arg("data"), py::arg("indices"), py::arg("indptr"),
          py::arg("y"), py::arg("alpha"), py::arg("l1_ratio"),
          py::arg("fit_intercept") = true,
          py::arg("tol") = 1e-4, py::arg("max_iter") = 1000);
}
