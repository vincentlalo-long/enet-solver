# Benchmark results

- Machine: AMD Ryzen 7 7735HS with Radeon Graphics
- Python 3.14.7, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1
- Backend under test: C++
- Single thread (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`), one warmup, median of repeated runs

## Matched-KKT timing

The time column is the median wall time to reach the target accuracy. For scikit-learn, the fastest `tol` from a fixed grid with achieved `KKT <= target` is used, so the comparison is at equal accuracy rather than equal `tol`. `sk/ours > 1` means our solver is faster.

| dataset | shape | nnz | target KKT | ours (s) | sklearn (s) | sk/ours | ours KKT | sk KKT | sk tol |
|---|---|---|---|---|---|---|---|---|---|
| dense-500x50 | 500×50 | 25000 | 1.00e-04 | 0.0008 | 0.0006 | 0.78× | 4.67e-06 | 4.94e-10 | 1.00e-07 |
| dense-500x50 | 500×50 | 25000 | 1.00e-06 | 0.0008 | 0.0006 | 0.80× | 2.77e-07 | 4.94e-10 | 1.00e-07 |
| dense-500x50 | 500×50 | 25000 | 1.00e-08 | 0.0009 | 0.0006 | 0.72× | 4.94e-10 | 4.94e-10 | 1.00e-07 |
| dense-500x50 | 500×50 | 25000 | 1.00e-10 | 0.0009 | 0.0006 | 0.69× | 5.06e-11 | 5.06e-11 | 1.00e-08 |
| sparse-500x20000 | 500×20000 | 100000 | 1.00e-04 | 0.0037 | 0.0172 | 4.68× | 5.21e-05 | 2.99e-06 | 1.00e-03 |
| sparse-500x20000 | 500×20000 | 100000 | 1.00e-06 | 0.0067 | 0.0181 | 2.72× | 6.80e-07 | 2.38e-07 | 1.00e-04 |
| sparse-500x20000 | 500×20000 | 100000 | 1.00e-08 | 0.0099 | 0.0205 | 2.08× | 7.92e-09 | 2.65e-09 | 1.00e-06 |
| sparse-500x20000 | 500×20000 | 100000 | 1.00e-10 | 0.0126 | 0.0229 | 1.82× | 8.51e-11 | 3.00e-11 | 1.00e-08 |
| sparse-N<<D-200x50000 | 200×50000 | 100000 | 1.00e-04 | 0.1136 | 0.1491 | 1.31× | 9.75e-05 | 2.27e-05 | 1.00e-03 |
| sparse-N<<D-200x50000 | 200×50000 | 100000 | 1.00e-06 | 0.2864 | 0.2948 | 1.03× | 9.84e-07 | 2.55e-07 | 1.00e-05 |
| sparse-N<<D-200x50000 | 200×50000 | 100000 | 1.00e-08 | 0.4334 | 0.4457 | 1.03× | 9.14e-09 | 2.51e-09 | 1.00e-07 |
| sparse-N<<D-200x50000 | 200×50000 | 100000 | 1.00e-10 | 0.6110 | 0.5851 | 0.96× | 9.39e-11 | 5.08e-11 | 1.00e-09 |
| ill-cond-200x100 | 200×100 | 20000 | 1.00e-04 | 0.0013 | 0.0008 | 0.67× | 9.06e-05 | 7.01e-05 | 1.00e-03 |
| ill-cond-200x100 | 200×100 | 20000 | 1.00e-06 | 0.0020 | 0.0009 | 0.44× | 8.66e-07 | 7.24e-08 | 1.00e-06 |
| ill-cond-200x100 | 200×100 | 20000 | 1.00e-08 | 0.0030 | 0.0010 | 0.34× | 9.41e-09 | 5.86e-09 | 1.00e-07 |
| ill-cond-200x100 | 200×100 | 20000 | 1.00e-10 | 0.0036 | 0.0011 | 0.30× | 9.73e-11 | 5.29e-11 | 1.00e-09 |

## Scaling at fixed target KKT = 1e-6

`vsD`: N = 500, density = 0.01, D varies. `vsN`: D = 5000, density = 0.01, N varies. `alpha = 0.05 * alpha_max` in every case, so the relative problem difficulty stays comparable as the size changes.

| regime | N | D | nnz | alpha | ours (s) | sklearn (s) | sk/ours |
|---|---|---|---|---|---|---|---|
| vsD | 500 | 1000 | 5000 | 0.0028 | 0.0002 | 0.0021 | 8.34× |
| vsD | 500 | 5000 | 25000 | 0.0026 | 0.0051 | 0.0103 | 2.02× |
| vsD | 500 | 20000 | 100000 | 0.0027 | 0.0450 | 0.0808 | 1.80× |
| vsD | 500 | 50000 | 250000 | 0.0030 | 0.1896 | 0.2897 | 1.53× |
| vsN | 200 | 5000 | 10000 | 0.0066 | 0.0105 | 0.0122 | 1.16× |
| vsN | 500 | 5000 | 25000 | 0.0026 | 0.0050 | 0.0102 | 2.03× |
| vsN | 2000 | 5000 | 100000 | 0.0044 | 0.0011 | 0.0133 | 11.57× |
| vsN | 5000 | 5000 | 250000 | 0.0021 | 0.0027 | 0.0317 | 11.67× |

## Reading the numbers honestly

- This is a learning project, not a claim that it beats scikit-learn. scikit-learn is a mature library with active-set screening, a carefully tuned convergence criterion, and years of optimization.
- The C++ core is competitive in some sparse regimes because it does one thing: coordinate descent with a residual update over CSC. It has no SIMD, no OpenMP, no active set and no strong rules.
- Dense problems are slower than scikit-learn. The dense path is not the target of this project.
- Timings include the Python wrapper and, for dense input, the one-time conversion to CSC. They are end-to-end API timings, not pure kernel time.
- Accuracy is matched through the final KKT violation computed by the same function for both solutions, so differences in `tol` semantics do not drive the conclusion.
- Both solvers can overshoot the target (ours checks once per sweep; scikit-learn is picked from a coarse `tol` grid), so the timings mean 'time to reach at least the target'. scikit-learn's overshoot is typically larger, which inflates its time and makes the comparison conservative *against* scikit-learn.
- Dense timings are around 1 ms; treat sub-0.2 ms differences as noise.
